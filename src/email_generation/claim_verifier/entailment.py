from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    DecomposedClaim,
    EntailmentCheckResult,
    EntailmentVerdict,
    VerificationFinding,
    VerifierFindingSeverity,
)
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import CanonicalFact, EvidenceRecord, ResearchPackage

logger = logging.getLogger(__name__)


class RawEntailmentOutput(BaseModel):
    """Structured LLM output for fact entailment verification."""

    verdict: str = Field(description="Must be ENTAILED, CONTRADICTED, or NEUTRAL")
    supporting_fact_ids: List[str] = Field(default_factory=list, description="IDs of matching CanonicalFact")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of supporting EvidenceRecord")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    rationale: str = Field(description="Entailment or contradiction rationale")


class FactEntailmentGate:
    """Gate 2: Two-tier semantic fact entailment verifier.

    Tier 1: Deterministic candidate filtering (token overlap, negation checks, candidate fact retrieval).
    Tier 2: Structured LLM entailment verification with strict ID validation against ResearchPackage.
    """

    NEGATION_WORDS = {"delayed", "cancelled", "canceled", "failed", "postponed", "halted", "stopped", "rejected", "declined", "never", "not"}

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.provider_name = provider_name
        self.parser = PydanticParser(RawEntailmentOutput)

    def verify(
        self,
        claims: List[DecomposedClaim],
        research_package: Optional[ResearchPackage],
    ) -> Tuple[List[EntailmentCheckResult], List[VerificationFinding]]:
        """Verify prospect-facing claims against canonical facts and evidence records."""
        results: List[EntailmentCheckResult] = []
        findings: List[VerificationFinding] = []

        if not research_package or not research_package.facts:
            # No research package available -> all prospect observations are unsupported
            for claim in claims:
                if not claim.is_product_claim and claim.category in (ClaimCategory.PROSPECT_OBSERVATION, ClaimCategory.METRIC_IMPACT) and claim.is_material:
                    res = EntailmentCheckResult(
                        claim_id=claim.claim_id,
                        verdict=EntailmentVerdict.NEUTRAL,
                        matched_fact_ids=[],
                        matched_evidence_ids=[],
                        confidence=0.0,
                        rationale="No research facts available to verify prospect assertion.",
                    )
                    results.append(res)
                    findings.append(
                        VerificationFinding(
                            gate="FactEntailment",
                            severity=VerifierFindingSeverity.ERROR,
                            code="MISSING_RESEARCH_EVIDENCE",
                            message=f"No research facts exist to verify claim: '{claim.text}'",
                            claim_id=claim.claim_id,
                            section=claim.section,
                            text_span=(claim.span_start, claim.span_end),
                        )
                    )
            return results, findings

        # Build lookup tables for strict validation
        fact_by_id: Dict[str, CanonicalFact] = {f.fact_id: f for f in research_package.facts}
        evidence_by_id: Dict[str, EvidenceRecord] = {}
        fact_to_ev_ids: Dict[str, Set[str]] = {}

        for f in research_package.facts:
            ev_set: Set[str] = set()
            for ev in f.evidence_records:
                evidence_by_id[ev.evidence_id] = ev
                ev_set.add(ev.evidence_id)
            fact_to_ev_ids[f.fact_id] = ev_set

        # Filter claims that require fact entailment verification
        prospect_claims = [
            c for c in claims
            if not c.is_product_claim
            and c.category in (ClaimCategory.PROSPECT_OBSERVATION, ClaimCategory.METRIC_IMPACT)
            and c.is_material
        ]

        for claim in prospect_claims:
            # Tier 1: Candidate Filtering
            candidate_facts, polarity_contradiction = self._filter_candidate_facts(claim, research_package.facts)

            if polarity_contradiction:
                # Deterministic negation / polarity contradiction detected
                res = EntailmentCheckResult(
                    claim_id=claim.claim_id,
                    verdict=EntailmentVerdict.CONTRADICTED,
                    matched_fact_ids=[f.fact_id for f in candidate_facts],
                    matched_evidence_ids=[ev.evidence_id for f in candidate_facts for ev in f.evidence_records],
                    confidence=1.0,
                    rationale=f"Deterministic polarity conflict with evidence: {candidate_facts[0].claim_text}",
                )
                results.append(res)
                findings.append(
                    VerificationFinding(
                        gate="FactEntailment",
                        severity=VerifierFindingSeverity.ERROR,
                        code="CONTRADICTED_FACT",
                        message=f"Claim '{claim.text}' is contradicted by fact: '{candidate_facts[0].claim_text}'",
                        claim_id=claim.claim_id,
                        section=claim.section,
                        text_span=(claim.span_start, claim.span_end),
                        details={"fact_id": candidate_facts[0].fact_id},
                    )
                )
                continue

            if not candidate_facts:
                # No relevant candidate facts found -> NEUTRAL
                res = EntailmentCheckResult(
                    claim_id=claim.claim_id,
                    verdict=EntailmentVerdict.NEUTRAL,
                    matched_fact_ids=[],
                    matched_evidence_ids=[],
                    confidence=0.0,
                    rationale="No relevant facts found in research package matching this claim.",
                )
                results.append(res)
                findings.append(
                    VerificationFinding(
                        gate="FactEntailment",
                        severity=VerifierFindingSeverity.ERROR,
                        code="UNSUPPORTED_PROSPECT_ASSERTION",
                        message=f"Material prospect assertion is unsupported by research: '{claim.text}'",
                        claim_id=claim.claim_id,
                        section=claim.section,
                        text_span=(claim.span_start, claim.span_end),
                    )
                )
                continue

            # Tier 2: Semantic Entailment (LLM or deterministic fallback)
            check_res, check_findings = self._evaluate_entailment_tier2(
                claim=claim,
                candidate_facts=candidate_facts,
                fact_by_id=fact_by_id,
                evidence_by_id=evidence_by_id,
                fact_to_ev_ids=fact_to_ev_ids,
            )
            results.append(check_res)
            findings.extend(check_findings)

        return results, findings

    def _filter_candidate_facts(
        self,
        claim: DecomposedClaim,
        all_facts: Sequence[CanonicalFact],
    ) -> Tuple[List[CanonicalFact], bool]:
        """Tier 1: Filter facts by entity/keyword overlap and check deterministic polarity."""
        claim_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", claim.text.lower()))
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "saw", "team", "your"}
        meaningful_tokens = claim_tokens - stop_words

        matched_facts: List[CanonicalFact] = []
        polarity_contradiction = False

        for f in all_facts:
            fact_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", f.claim_text.lower()))
            overlap = meaningful_tokens.intersection(fact_tokens)

            # Check evidence excerpts as well
            ev_overlap = any(
                bool(meaningful_tokens.intersection(set(re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", ev.verbatim_excerpt.lower()))))
                for ev in f.evidence_records
            )

            if len(overlap) >= 1 or ev_overlap:
                matched_facts.append(f)
                # Check for negation conflict
                fact_negations = fact_tokens.intersection(self.NEGATION_WORDS)
                claim_negations = claim_tokens.intersection(self.NEGATION_WORDS)
                if (fact_negations and not claim_negations) or (claim_negations and not fact_negations):
                    # Check if they share critical nouns (e.g. migration, deployment, acquisition)
                    nouns = overlap - self.NEGATION_WORDS
                    if len(nouns) >= 1:
                        polarity_contradiction = True

        return matched_facts, polarity_contradiction

    def _evaluate_entailment_tier2(
        self,
        claim: DecomposedClaim,
        candidate_facts: List[CanonicalFact],
        fact_by_id: Dict[str, CanonicalFact],
        evidence_by_id: Dict[str, EvidenceRecord],
        fact_to_ev_ids: Dict[str, Set[str]],
    ) -> Tuple[EntailmentCheckResult, List[VerificationFinding]]:
        """Tier 2: Evaluate entailment via GeneratorEngine or deterministic rule fallback."""
        findings: List[VerificationFinding] = []

        # If no generator engine provided, use deterministic exact evidence matching
        if self.engine is None:
            return self._deterministic_fallback_entailment(
                claim, candidate_facts, fact_to_ev_ids
            )

        # Build prompt for GeneratorEngine
        facts_payload = []
        for f in candidate_facts:
            evs = [
                {"evidence_id": ev.evidence_id, "excerpt": ev.verbatim_excerpt}
                for ev in f.evidence_records
            ]
            facts_payload.append({
                "fact_id": f.fact_id,
                "fact_text": f.claim_text,
                "evidence_records": evs,
            })

        system_msg = (
            "You are a strict, formal semantic verifier for commercial outreach. "
            "Evaluate whether the provided Premise (canonical facts and verbatim excerpts) "
            "strictly ENTAILS, CONTRADICTS, or is NEUTRAL to the Hypothesis (atomic claim). "
            "An ENTAILED verdict REQUIRES explicit fact_id and evidence_id references from the supplied premise. "
            "Do NOT fabricate IDs. If evidence is missing or insufficient, return NEUTRAL."
        )

        user_msg = (
            f"Hypothesis (Claim to verify):\n\"{claim.text}\"\n\n"
            f"Premise (Known Research Facts & Excerpts):\n{facts_payload}\n\n"
            "Respond in JSON matching the schema: "
            "{\"verdict\": \"ENTAILED\"|\"CONTRADICTED\"|\"NEUTRAL\", "
            "\"supporting_fact_ids\": [...], \"supporting_evidence_ids\": [...], "
            "\"confidence\": 0.0-1.0, \"rationale\": \"...\"}"
        )

        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=system_msg))
        prompt.add_message(Message(role=Role.USER, content=user_msg))

        request = GenerationRequest(
            prompt=prompt,
            decoding=DecodeConfig(temperature=0.0, max_tokens=300),
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "semantic_fact_entailment", "claim_id": claim.claim_id},
        )


        try:
            resp = self.engine.generate(request)
            raw: RawEntailmentOutput = resp.parsed_output
            verdict_str = (raw.verdict or "").strip().upper()
        except Exception as exc:
            logger.error("Entailment LLM call failed for claim %s: %s", claim.claim_id, exc)
            findings.append(
                VerificationFinding(
                    gate="FactEntailment",
                    severity=VerifierFindingSeverity.ERROR,
                    code="VERIFIER_SERVICE_UNAVAILABLE",
                    message=f"Fact entailment verifier service failed: {exc}",
                    claim_id=claim.claim_id,
                    section=claim.section,
                    text_span=(claim.span_start, claim.span_end),
                )
            )
            return EntailmentCheckResult(
                claim_id=claim.claim_id,
                verdict=EntailmentVerdict.NEUTRAL,
                matched_fact_ids=[],
                matched_evidence_ids=[],
                confidence=0.0,
                rationale=f"Verifier service failure: {exc}",
            ), findings

        # Strict validation of returned IDs
        valid_fact_ids: List[str] = [fid for fid in raw.supporting_fact_ids if fid in fact_by_id]
        valid_ev_ids: List[str] = [eid for eid in raw.supporting_evidence_ids if eid in evidence_by_id]

        # Verify evidence belongs to cited facts
        strictly_valid_ev_ids: List[str] = []
        for fid in valid_fact_ids:
            allowed_evs = fact_to_ev_ids.get(fid, set())
            for eid in valid_ev_ids:
                if eid in allowed_evs and eid not in strictly_valid_ev_ids:
                    strictly_valid_ev_ids.append(eid)

        # Check if model fabricated invalid IDs
        fabricated_facts = set(raw.supporting_fact_ids) - set(valid_fact_ids)
        fabricated_evs = set(raw.supporting_evidence_ids) - set(valid_ev_ids)
        if fabricated_facts or fabricated_evs:
            findings.append(
                VerificationFinding(
                    gate="FactEntailment",
                    severity=VerifierFindingSeverity.ERROR,
                    code="FABRICATED_EVIDENCE_REFERENCE",
                    message=f"Model returned invalid fact/evidence IDs: facts={list(fabricated_facts)}, evs={list(fabricated_evs)}",
                    claim_id=claim.claim_id,
                    section=claim.section,
                    text_span=(claim.span_start, claim.span_end),
                )
            )

        # Normalize verdict
        if verdict_str == "ENTAILED":
            # Must have at least one valid fact ID and one valid evidence ID
            if not valid_fact_ids or not strictly_valid_ev_ids:
                verdict = EntailmentVerdict.NEUTRAL
                findings.append(
                    VerificationFinding(
                        gate="FactEntailment",
                        severity=VerifierFindingSeverity.ERROR,
                        code="MISSING_EVIDENCE_REFERENCE",
                        message=f"Claim was marked ENTAILED but lacked valid supporting fact/evidence IDs: '{claim.text}'",
                        claim_id=claim.claim_id,
                        section=claim.section,
                        text_span=(claim.span_start, claim.span_end),
                    )
                )
            else:
                verdict = EntailmentVerdict.ENTAILED
        elif verdict_str == "CONTRADICTED":
            verdict = EntailmentVerdict.CONTRADICTED
            findings.append(
                VerificationFinding(
                    gate="FactEntailment",
                    severity=VerifierFindingSeverity.ERROR,
                    code="CONTRADICTED_FACT",
                    message=f"Claim is contradicted by research evidence: '{claim.text}'. Rationale: {raw.rationale}",
                    claim_id=claim.claim_id,
                    section=claim.section,
                    text_span=(claim.span_start, claim.span_end),
                    details={"facts": valid_fact_ids, "evidence": strictly_valid_ev_ids},
                )
            )
        else:
            verdict = EntailmentVerdict.NEUTRAL
            findings.append(
                VerificationFinding(
                    gate="FactEntailment",
                    severity=VerifierFindingSeverity.ERROR,
                    code="UNSUPPORTED_PROSPECT_ASSERTION",
                    message=f"Material prospect assertion is unsupported by research: '{claim.text}'",
                    claim_id=claim.claim_id,
                    section=claim.section,
                    text_span=(claim.span_start, claim.span_end),
                )
            )

        return EntailmentCheckResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            matched_fact_ids=valid_fact_ids,
            matched_evidence_ids=strictly_valid_ev_ids,
            confidence=raw.confidence,
            rationale=raw.rationale,
        ), findings

    def _deterministic_fallback_entailment(
        self,
        claim: DecomposedClaim,
        candidate_facts: List[CanonicalFact],
        fact_to_ev_ids: Dict[str, Set[str]],
    ) -> Tuple[EntailmentCheckResult, List[VerificationFinding]]:
        """Deterministic rule-based entailment check for offline testing."""
        findings: List[VerificationFinding] = []

        # Check if candidate facts corroborate the claim keywords
        claim_words = set(re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", claim.text.lower()))
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "saw", "team", "your"}
        claim_keywords = claim_words - stop_words

        matched_fids: List[str] = []
        matched_eids: List[str] = []

        for f in candidate_facts:
            f_words = set(re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", f.claim_text.lower()))
            # If 50% or more of keywords match fact or evidence
            overlap = claim_keywords.intersection(f_words)
            if len(overlap) >= max(1, len(claim_keywords) // 2):
                matched_fids.append(f.fact_id)
                for ev in f.evidence_records:
                    if ev.evidence_id not in matched_eids:
                        matched_eids.append(ev.evidence_id)

        if matched_fids and matched_eids:
            return EntailmentCheckResult(
                claim_id=claim.claim_id,
                verdict=EntailmentVerdict.ENTAILED,
                matched_fact_ids=matched_fids,
                matched_evidence_ids=matched_eids,
                confidence=0.9,
                rationale="Corroborated by research facts and verbatim source evidence.",
            ), findings

        findings.append(
            VerificationFinding(
                gate="FactEntailment",
                severity=VerifierFindingSeverity.ERROR,
                code="UNSUPPORTED_PROSPECT_ASSERTION",
                message=f"Material prospect assertion is unsupported by research: '{claim.text}'",
                claim_id=claim.claim_id,
                section=claim.section,
                text_span=(claim.span_start, claim.span_end),
            )
        )
        return EntailmentCheckResult(
            claim_id=claim.claim_id,
            verdict=EntailmentVerdict.NEUTRAL,
            matched_fact_ids=[],
            matched_evidence_ids=[],
            confidence=0.0,
            rationale="Insufficient evidence corroborating prospect claim.",
        ), findings
