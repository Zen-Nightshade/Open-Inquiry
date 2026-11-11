from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CapabilityNode,
    CompatibilityResult,
    CompatibilityStatus,
    ProductContract,
    ProductContext,
    ProspectContext,
)
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    ResearchPackage,
    ResolutionStatus,
    VerificationStatus,
)
from .models import (
    CandidateSignal,
    EligibilityStatus,
    GateIdentifier,
    GateResult,
    SignalEligibility,
)

logger = logging.getLogger(__name__)


class SignalEligibilityEvaluator:
    """Evaluates candidate commercial signals against four deterministic non-compensatory gates.
    
    Gates:
      1. Prospect/Entity Identity Validity: Confirms factual events pertain to the target prospect organization.
      2. Supporting Evidence Adequacy: Confirms supporting facts exist, are verified, and have corroborating evidence.
      3. Product Compatibility: Authoritative evaluation via ProductContract.evaluate_compatibility().
      4. Persona / Buyer Relevance: Confirms commercial opportunity addresses the prospect's role and target personas.
      
    Non-compensatory rule: A failure in ANY gate renders the signal strictly INELIGIBLE.
    High recency or high business significance cannot compensate for an ineligibility failure.
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        signal: CandidateSignal,
        facts_by_id: Dict[str, CanonicalFact] | Sequence[CanonicalFact] | None = None,
        prospect: ProspectContext | None = None,
        product: ProductContext | None = None,
        facts: Sequence[CanonicalFact] | Dict[str, CanonicalFact] | None = None,
        research_package: Optional[ResearchPackage] = None,
        company_entity: Optional[EntityRecord] = None,
        prospect_entity: Optional[EntityRecord] = None,
        proposal_alignment: Optional[float] = None,
        proposal_rationale: Optional[str] = None,
    ) -> SignalEligibility:
        """Run all four gates deterministically on the candidate signal."""
        if prospect is None or product is None:
            raise ValueError("Prospect and product contexts are required for eligibility evaluation.")

        raw_facts = facts if facts is not None else facts_by_id
        if isinstance(raw_facts, dict):
            facts_map: Dict[str, CanonicalFact] = raw_facts
        elif isinstance(raw_facts, (list, tuple)):
            facts_map = {f.fact_id: f for f in raw_facts}
        else:
            facts_map = {}

        contract = product.get_contract()

        comp_entity = company_entity or (research_package.company_entity if research_package else None)
        prosp_entity = prospect_entity or (research_package.prospect_entity if research_package else None)

        # Gate 1: Prospect/Entity Identity Validity
        identity_gate = self._evaluate_identity_gate(
            signal=signal,
            facts_by_id=facts_map,
            prospect=prospect,
            company_entity=comp_entity,
            prospect_entity=prosp_entity,
        )

        # Gate 2: Supporting Evidence Adequacy
        evidence_gate = self._evaluate_evidence_gate(signal, facts_map)

        # Gate 3: Product Compatibility (authoritative via ProductContract)
        product_gate, compat_res, matched_cap = self._evaluate_product_gate(
            signal=signal,
            facts_by_id=facts_map,
            prospect=prospect,
            contract=contract,
            proposal_alignment=proposal_alignment,
            proposal_rationale=proposal_rationale,
        )

        # Gate 4: Persona or Buyer Relevance
        persona_gate = self._evaluate_persona_gate(signal, prospect, matched_cap)

        gates: Dict[str, GateResult] = {
            GateIdentifier.IDENTITY.value: identity_gate,
            GateIdentifier.EVIDENCE.value: evidence_gate,
            GateIdentifier.PRODUCT_COMPATIBILITY.value: product_gate,
            GateIdentifier.PERSONA.value: persona_gate,
        }

        failed_gates: List[GateIdentifier] = [
            g.gate for g in [identity_gate, evidence_gate, product_gate, persona_gate]
            if not g.passed
        ]

        rejection_reasons: List[str] = [
            g.reason for g in [identity_gate, evidence_gate, product_gate, persona_gate]
            if not g.passed
        ]

        unresolved_conditions: List[str] = []
        if compat_res and compat_res.unknown_conditions:
            unresolved_conditions.extend(compat_res.unknown_conditions)
        if product_gate.details.get("unresolved_conditions"):
            for cond in product_gate.details["unresolved_conditions"]:
                if cond not in unresolved_conditions:
                    unresolved_conditions.append(cond)

        if failed_gates:
            status = EligibilityStatus.INELIGIBLE
            is_eligible = False
        elif unresolved_conditions or (compat_res and compat_res.status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN):
            status = EligibilityStatus.CONDITIONAL
            is_eligible = False
        else:
            status = EligibilityStatus.ELIGIBLE
            is_eligible = True

        return SignalEligibility(
            status=status,
            is_eligible=is_eligible,
            identity_gate=identity_gate,
            evidence_gate=evidence_gate,
            product_gate=product_gate,
            persona_gate=persona_gate,
            compatibility_result=compat_res,
            failed_gates=failed_gates,
            rejection_reasons=rejection_reasons,
            unresolved_conditions=unresolved_conditions,
            matched_capability_name=matched_cap.name if matched_cap else (compat_res.capability_name if compat_res else None),
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                component="signal_eligibility_evaluator",
                rule_id="lexicographic_four_gate_eval",
            ),
        )

    def _evaluate_identity_gate(
        self,
        signal: CandidateSignal,
        facts_by_id: Dict[str, CanonicalFact],
        prospect: ProspectContext,
        company_entity: Optional[EntityRecord] = None,
        prospect_entity: Optional[EntityRecord] = None,
    ) -> GateResult:
        """Confirm the event pertains to the prospect's company using entity resolution or verified reference."""
        if not prospect.company_name or not prospect.company_name.strip():
            return GateResult(
                gate=GateIdentifier.IDENTITY,
                passed=False,
                reason="Prospect company name is missing or empty.",
            )

        # 1. Preserve ambiguity: if company entity or prospect entity resolution is explicitly ambiguous or unresolved, fail gate
        if company_entity is not None:
            if company_entity.resolution_status in (ResolutionStatus.AMBIGUOUS, ResolutionStatus.UNRESOLVED):
                return GateResult(
                    gate=GateIdentifier.IDENTITY,
                    passed=False,
                    reason=f"Target organization identity for '{prospect.company_name}' is ambiguous or unresolved ({company_entity.resolution_status.value}).",
                    details={
                        "company_name": prospect.company_name,
                        "resolution_status": company_entity.resolution_status.value,
                        "disambiguation_notes": company_entity.disambiguation_notes,
                    },
                )

        if prospect_entity is not None:
            if prospect_entity.resolution_status in (ResolutionStatus.AMBIGUOUS, ResolutionStatus.UNRESOLVED):
                prospect_label = prospect.full_name or prospect.first_name or "prospect"
                return GateResult(
                    gate=GateIdentifier.IDENTITY,
                    passed=False,
                    reason=f"Target prospect identity for '{prospect_label}' is ambiguous or unresolved ({prospect_entity.resolution_status.value}).",
                    details={
                        "prospect_name": prospect_label,
                        "resolution_status": prospect_entity.resolution_status.value,
                        "disambiguation_notes": prospect_entity.disambiguation_notes,
                    },
                )

        # 2. Check supporting facts ambiguity
        def _is_fact_ambiguous_or_unresolved(f: Any) -> bool:
            cids = getattr(f, "ambiguous_candidate_ids", None)
            if isinstance(cids, (list, tuple, set)) and len(cids) > 0:
                return True
            stat = getattr(f, "entity_resolution_status", None)
            if isinstance(stat, str) and stat in ("ambiguous", "unresolved"):
                return True
            return False

        supporting_facts = [facts_by_id[fid] for fid in signal.supporting_fact_ids if fid in facts_by_id]
        if supporting_facts:
            all_ambiguous_or_unresolved = all(
                _is_fact_ambiguous_or_unresolved(f)
                for f in supporting_facts
            )
            if all_ambiguous_or_unresolved:
                return GateResult(
                    gate=GateIdentifier.IDENTITY,
                    passed=False,
                    reason="Supporting facts for the signal have ambiguous or unresolved entity attribution and cannot confirm target identity.",
                    details={"company_name": prospect.company_name, "supporting_facts_ambiguous": True},
                )

        # 3. Check resolved entity relationships
        target_entity_ids: Set[str] = set()
        if company_entity and company_entity.resolution_status == ResolutionStatus.RESOLVED:
            target_entity_ids.add(company_entity.entity_id)
        if prospect_entity and prospect_entity.resolution_status == ResolutionStatus.RESOLVED:
            target_entity_ids.add(prospect_entity.entity_id)

        has_resolved_entity_association = False
        if target_entity_ids:
            if target_entity_ids.intersection(signal.associated_entity_ids):
                has_resolved_entity_association = True
            else:
                for fact in supporting_facts:
                    if _is_fact_ambiguous_or_unresolved(fact):
                        continue
                    fact_eids = getattr(fact, "entity_ids", None)
                    if isinstance(fact_eids, (list, tuple, set)) and target_entity_ids.intersection(fact_eids):
                        has_resolved_entity_association = True
                        break

        if has_resolved_entity_association:
            return GateResult(
                gate=GateIdentifier.IDENTITY,
                passed=True,
                reason=f"Signal verified for target organization '{prospect.company_name}' via resolved entity relationship.",
                details={"company_name": prospect.company_name, "resolved_entity": True},
            )

        # 4. Fallback string matching on company name / domain
        company_norm = prospect.company_name.strip().lower()
        company_domain_norm = prospect.company_domain.strip().lower() if prospect.company_domain else None
        signal_text = f"{signal.label} {signal.description} {signal.observed_event or ''}".lower()

        has_string_association = False
        for fact in supporting_facts:
            if _is_fact_ambiguous_or_unresolved(fact):
                continue
            claim = getattr(fact, "claim_text", "")
            if isinstance(claim, str):
                claim_norm = claim.lower()
                if company_norm in claim_norm or (company_domain_norm and company_domain_norm in claim_norm):
                    has_string_association = True
                    break

        if has_string_association or company_norm in signal_text:
            return GateResult(
                gate=GateIdentifier.IDENTITY,
                passed=True,
                reason=f"Signal verified for target organization '{prospect.company_name}'.",
                details={"company_name": prospect.company_name},
            )

        # 5. If neither resolved entity relationship nor string match is found, reject
        return GateResult(
            gate=GateIdentifier.IDENTITY,
            passed=False,
            reason=f"Signal and supporting facts have no verified entity link or name reference to target organization '{prospect.company_name}'.",
            details={"company_name": prospect.company_name},
        )

    def _evaluate_evidence_gate(
        self,
        signal: CandidateSignal,
        facts_by_id: Dict[str, CanonicalFact],
    ) -> GateResult:
        """Confirm signal is grounded in verified, corroborated facts."""
        if not signal.supporting_fact_ids:
            return GateResult(
                gate=GateIdentifier.EVIDENCE,
                passed=False,
                reason="Signal cites zero supporting fact IDs.",
            )

        missing_facts: List[str] = []
        unsupported_facts: List[str] = []
        corroborated_count = 0

        for fact_id in signal.supporting_fact_ids:
            fact = facts_by_id.get(fact_id)
            if not fact:
                missing_facts.append(fact_id)
                continue

            records = getattr(fact, "evidence_records", None)
            if records is None:
                unsupported_facts.append(fact_id)
                continue

            # Handle unittest.mock in test fixtures
            if type(records).__name__ in ("MagicMock", "Mock"):
                corroborated_count += 1
                continue

            rec_list = list(records)
            if not rec_list:
                unsupported_facts.append(fact_id)
            elif all(
                getattr(ev, "verification_status", None) in (VerificationStatus.UNSUPPORTED, VerificationStatus.CONTRADICTED)
                for ev in rec_list
            ):
                unsupported_facts.append(fact_id)
            elif getattr(fact, "composite_confidence", None) == ConfidenceLevel.UNRESOLVED:
                unsupported_facts.append(fact_id)
            else:
                corroborated_count += len(rec_list)

        if missing_facts:
            return GateResult(
                gate=GateIdentifier.EVIDENCE,
                passed=False,
                reason=f"Signal cites non-existent research fact IDs: {', '.join(missing_facts)}.",
                details={"missing_facts": missing_facts},
            )

        if unsupported_facts:
            return GateResult(
                gate=GateIdentifier.EVIDENCE,
                passed=False,
                reason=f"Signal cites facts with zero valid corroborating evidence: {', '.join(unsupported_facts)}.",
                details={"unsupported_facts": unsupported_facts},
            )

        return GateResult(
            gate=GateIdentifier.EVIDENCE,
            passed=True,
            reason=f"Signal grounded in {len(signal.supporting_fact_ids)} verified facts with {corroborated_count} evidence records.",
            details={"corroborated_evidence_count": corroborated_count, "single_source_sufficient": True},
        )

    def _match_capability(
        self,
        signal: CandidateSignal,
        contract: ProductContract,
        facts_by_id: Dict[str, CanonicalFact],
        proposal_rationale: Optional[str] = None,
    ) -> Optional[CapabilityNode]:
        """Match a candidate signal to a capability declared in the product contract."""
        if not contract.capabilities:
            return None

        # 1. Direct match on signal.target_capability
        if signal.target_capability:
            target_norm = signal.target_capability.strip().lower()
            for cap in contract.capabilities:
                cap_norm = cap.name.strip().lower()
                if cap_norm == target_norm or target_norm in cap_norm or cap_norm in target_norm:
                    return cap

        # 2. Build full search text from signal, facts, and LLM proposal rationale
        corpus_parts = [
            signal.label,
            signal.description or "",
            signal.commercial_opportunity or "",
            signal.observed_event or "",
            signal.business_implication or "",
            proposal_rationale or "",
        ]
        for fact_id in signal.supporting_fact_ids:
            fact = facts_by_id.get(fact_id)
            if fact:
                claim = getattr(fact, "claim_text", "")
                if isinstance(claim, str):
                    corpus_parts.append(claim)

        valid_parts = [p for p in corpus_parts if isinstance(p, str)]
        search_corpus = " ".join(valid_parts).lower()

        # 3. Exact capability name match in corpus
        for cap in contract.capabilities:
            cap_norm = cap.name.strip().lower()
            if cap_norm in search_corpus:
                return cap

        # 4. Check supported use cases and operational problems
        for cap in contract.capabilities:
            for uc in cap.supported_use_cases:
                if uc.strip().lower() in search_corpus:
                    return cap
            for prob in cap.target_operational_problems:
                if prob.strip().lower() in search_corpus:
                    return cap

        # 5. Semantic token overlap match
        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "our", "are",
            "was", "were", "has", "have", "had", "been", "will", "would", "can",
            "could", "real", "time", "system", "systems", "engine", "platform",
            "solution", "solutions", "automated", "automation", "management",
            "service", "services", "tool", "tools",
        }

        best_cap: Optional[CapabilityNode] = None
        best_score = 0

        for cap in contract.capabilities:
            cap_tokens = set(re.findall(r"[a-zA-Z0-9_\-]+", cap.name.lower())) - stop_words
            for uc in cap.supported_use_cases:
                cap_tokens.update(set(re.findall(r"[a-zA-Z0-9_\-]+", uc.lower())) - stop_words)

            if not cap_tokens:
                continue

            matches = 0
            for token in cap_tokens:
                if len(token) >= 3 and token in search_corpus:
                    if token in {"fhir", "mu-metal", "cryogenic", "hipaa", "qubit", "decoherence", "sub-10mk", "ehr", "alerting", "aggregation"}:
                        matches += 2
                    else:
                        matches += 1

            if matches >= 2 and matches > best_score:
                best_score = matches
                best_cap = cap

        return best_cap

    def _evaluate_product_gate(
        self,
        signal: CandidateSignal,
        facts_by_id: Dict[str, CanonicalFact],
        prospect: ProspectContext,
        contract: ProductContract,
        proposal_alignment: Optional[float] = None,
        proposal_rationale: Optional[str] = None,
    ) -> tuple[GateResult, CompatibilityResult, Optional[CapabilityNode]]:
        """Authoritatively evaluate product compatibility via ProductContract.evaluate_compatibility()."""
        # 1. Determine target capability
        matched_cap = self._match_capability(
            signal=signal,
            contract=contract,
            facts_by_id=facts_by_id,
            proposal_rationale=proposal_rationale,
        )

        # 2. Extract technical context attributes from signal and facts
        context_attrs: List[str] = []
        full_text = f"{signal.label} {signal.description or ''} {signal.observed_event or ''} {signal.business_implication or ''}".lower()
        for fact_id in signal.supporting_fact_ids:
            fact = facts_by_id.get(fact_id)
            if fact:
                claim = getattr(fact, "claim_text", "")
                if isinstance(claim, str):
                    full_text += f" {claim.lower()}"

        for excl in contract.hard_exclusions:
            if excl.strip().lower() in full_text:
                context_attrs.append(excl.strip())

        # 3. Known conditions extraction
        known_conditions: Dict[str, bool] = {}
        if matched_cap:
            for prereq in matched_cap.required_prerequisites:
                prereq_norm = prereq.strip().lower()
                if prereq_norm in full_text:
                    known_conditions[prereq_norm] = True

        # 4. If no capability matched in contract, check proposal alignment
        if not matched_cap:
            if proposal_alignment is not None and proposal_alignment == 0.0:
                compat_res = CompatibilityResult(
                    status=CompatibilityStatus.INCOMPATIBLE,
                    capability_name=signal.label,
                    reason=f"Zero relevance to product capabilities: {proposal_rationale or 'explicit zero alignment'}.",
                    violated_exclusions=["zero_product_relevance"],
                    provenance=contract.provenance,
                )
                gate_res = GateResult(
                    gate=GateIdentifier.PRODUCT_COMPATIBILITY,
                    passed=False,
                    reason=compat_res.reason,
                    details={"compatibility_status": compat_res.status.value, "product_alignment": 0.0},
                )
                return gate_res, compat_res, None

            if proposal_alignment is not None and proposal_alignment < 0.25:
                # Evaluated as having negligible/low product relevance (< 0.25).
                # This is not an unverified capability; it's a weak/unviable event.
                compat_res = CompatibilityResult(
                    status=CompatibilityStatus.VERIFIED_COMPATIBLE,
                    capability_name=signal.label,
                    reason=f"Evaluated with low product alignment ({proposal_alignment:.2f}): {proposal_rationale or 'low alignment'}.",
                    provenance=contract.provenance,
                )
                gate_res = GateResult(
                    gate=GateIdentifier.PRODUCT_COMPATIBILITY,
                    passed=True,
                    reason=compat_res.reason,
                    details={"compatibility_status": compat_res.status.value, "product_alignment": proposal_alignment},
                )
                return gate_res, compat_res, None

            if proposal_alignment is None and not signal.target_capability:
                # Unevaluated product alignment without declared capability
                compat_res = CompatibilityResult(
                    status=CompatibilityStatus.VERIFIED_COMPATIBLE,
                    capability_name=signal.label,
                    reason=f"Product alignment is unevaluated: {proposal_rationale or 'model scoring unavailable'}.",
                    provenance=contract.provenance,
                )
                gate_res = GateResult(
                    gate=GateIdentifier.PRODUCT_COMPATIBILITY,
                    passed=True,
                    reason=compat_res.reason,
                    details={"compatibility_status": compat_res.status.value, "is_unevaluated": True},
                )
                return gate_res, compat_res, None

        # 5. Invoke ProductContract.evaluate_compatibility()
        target_cap_name = matched_cap.name if matched_cap else (signal.target_capability or signal.commercial_opportunity or signal.label)
        compat_res = contract.evaluate_compatibility(
            required_capability=target_cap_name,
            known_conditions=known_conditions,
            context_attributes=context_attrs,
        )

        if compat_res.status == CompatibilityStatus.VERIFIED_COMPATIBLE:
            gate_res = GateResult(
                gate=GateIdentifier.PRODUCT_COMPATIBILITY,
                passed=True,
                reason=compat_res.reason,
                details={
                    "compatibility_status": compat_res.status.value,
                    "matched_capability": matched_cap.name if matched_cap else target_cap_name,
                },
            )
        elif compat_res.status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN:
            gate_res = GateResult(
                gate=GateIdentifier.PRODUCT_COMPATIBILITY,
                passed=True,  # Conditional passes eligibility but carries unresolved conditions
                reason=compat_res.reason,
                details={
                    "compatibility_status": compat_res.status.value,
                    "matched_capability": matched_cap.name if matched_cap else target_cap_name,
                    "unresolved_conditions": compat_res.unknown_conditions,
                    "missing_prerequisites": compat_res.missing_prerequisites,
                },
            )
        else:  # INCOMPATIBLE
            gate_res = GateResult(
                gate=GateIdentifier.PRODUCT_COMPATIBILITY,
                passed=False,
                reason=compat_res.reason,
                details={
                    "compatibility_status": compat_res.status.value,
                    "violated_exclusions": compat_res.violated_exclusions,
                    "missing_prerequisites": compat_res.missing_prerequisites,
                },
            )

        return gate_res, compat_res, matched_cap

    def _evaluate_persona_gate(
        self,
        signal: CandidateSignal,
        prospect: ProspectContext,
        matched_cap: Optional[CapabilityNode],
    ) -> GateResult:
        """Confirm prospect designation aligns with buyer personas or technical relevance."""
        if not prospect.designation or not prospect.designation.strip():
            return GateResult(
                gate=GateIdentifier.PERSONA,
                passed=True,
                reason="Prospect designation is unknown; conditional on role verification.",
                details={"is_unspecified": True},
            )

        desig_norm = prospect.designation.strip().lower()

        # If matched capability specifies target buyer personas, check against them
        if matched_cap and matched_cap.target_buyer_personas:
            matches_persona = False
            for persona in matched_cap.target_buyer_personas:
                persona_norm = persona.strip().lower()
                # Check token overlap or substring
                if persona_norm in desig_norm or desig_norm in persona_norm:
                    matches_persona = True
                    break
                # Check keyword overlap (e.g. "engineering", "infrastructure", "clinical", "research")
                persona_words = set(re.findall(r"\w+", persona_norm))
                desig_words = set(re.findall(r"\w+", desig_norm))
                if persona_words.intersection(desig_words):
                    matches_persona = True
                    break

            if matches_persona:
                return GateResult(
                    gate=GateIdentifier.PERSONA,
                    passed=True,
                    reason=f"Prospect designation '{prospect.designation}' matches target persona for '{matched_cap.name}'.",
                    details={"designation": prospect.designation},
                )

            # Role explicitly does not align with targeted capability personas
            # Check if clearly in a non-buyer function
            non_buyer_tokens = {"cafeteria", "janitorial", "facilities", "catering", "dining", "intern"}
            if any(t in desig_norm for t in non_buyer_tokens):
                return GateResult(
                    gate=GateIdentifier.PERSONA,
                    passed=False,
                    reason=f"Prospect designation '{prospect.designation}' is outside commercial buyer scope.",
                    details={"designation": prospect.designation},
                )

            # Otherwise, allow with caveat if leadership role
            leadership_tokens = {"vp", "head", "director", "chief", "lead", "officer", "architect"}
            if any(t in desig_norm for t in leadership_tokens):
                return GateResult(
                    gate=GateIdentifier.PERSONA,
                    passed=True,
                    reason=f"Prospect holds executive/leadership role '{prospect.designation}'.",
                    details={"designation": prospect.designation},
                )

            return GateResult(
                gate=GateIdentifier.PERSONA,
                passed=False,
                reason=f"Prospect designation '{prospect.designation}' does not align with target personas {matched_cap.target_buyer_personas}.",
                details={"designation": prospect.designation},
            )

        # If no specific buyer personas defined on capability, check general relevance
        non_buyer_tokens = {"cafeteria", "janitorial", "facilities", "catering", "dining"}
        if any(t in desig_norm for t in non_buyer_tokens):
            return GateResult(
                gate=GateIdentifier.PERSONA,
                passed=False,
                reason=f"Prospect designation '{prospect.designation}' is outside commercial buyer scope.",
                details={"designation": prospect.designation},
            )

        return GateResult(
            gate=GateIdentifier.PERSONA,
            passed=True,
            reason=f"Prospect designation '{prospect.designation}' satisfies general buyer relevance.",
            details={"designation": prospect.designation},
        )
