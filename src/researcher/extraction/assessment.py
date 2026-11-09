from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EvidenceRecord,
    ResolutionStatus,
    SourceRecord,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


def extract_numerical_tokens(text: str) -> Set[str]:
    """Extract dollar amounts, percentages, and numbers from text."""
    matches = re.findall(r"\$?\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:million|billion|m|b|k|%))?", text, re.IGNORECASE)
    return {m.strip().lower() for m in matches if m.strip()}


class EvidenceAssessment(BaseModel):
    """Detailed evaluation of how an evidence excerpt supports or contradicts an assertion."""

    model_config = ConfigDict(frozen=True)

    assessment_status: VerificationStatus
    confidence: ConfidenceLevel
    is_independent: bool = True
    support_rationale: str
    temporal_notes: Optional[str] = None
    entity_notes: Optional[str] = None
    conflict_detected: bool = False
    conflict_description: Optional[str] = None
    target_prospect_attributed: bool = True


class EvidenceAssessor:
    """Evaluates factual support, temporal consistency, entity alignment, and syndication independence."""

    def __init__(self) -> None:
        pass

    def assess_evidence_support(
        self,
        claim_text: str,
        evidence: EvidenceRecord,
        source: Optional[SourceRecord] = None,
        entity: Optional[EntityRecord] = None,
        target_entity_id: Optional[str] = None,
    ) -> EvidenceAssessment:
        """Assess the degree of factual support provided by an evidence excerpt for a claim."""
        excerpt_lower = evidence.verbatim_excerpt.lower()
        claim_lower = claim_text.lower()

        # 1. Deterministic Negation & Polarity Mismatch Check
        contradiction_keywords = {
            "denied", "refuted", "false", "did not raise", "did not acquire",
            "stepped down", "resigned", "canceled", "cancelled", "laid off",
            "unfounded", "untrue", "disputed", "failed to",
        }
        excerpt_has_negation = any(k in excerpt_lower for k in contradiction_keywords)
        claim_has_negation = any(k in claim_lower for k in contradiction_keywords)

        is_contradicted = excerpt_has_negation and not claim_has_negation

        # 2. Unsupported Numerical Qualifier Check
        nums_claim = extract_numerical_tokens(claim_lower)
        nums_excerpt = extract_numerical_tokens(excerpt_lower)
        has_unsupported_nums = bool(nums_claim - nums_excerpt)

        # 3. Support / Entailment Analysis
        if is_contradicted:
            status = VerificationStatus.CONTRADICTED
            confidence = ConfidenceLevel.MEDIUM
            rationale = "Excerpt explicitly refutes or negates the asserted claim."
            conflict = True
            conflict_desc = f"Evidence directly contradicts claim '{claim_text}': {evidence.verbatim_excerpt}"
        elif has_unsupported_nums:
            # The claim introduces numerical metrics (e.g. $50M) absent from the excerpt
            status = VerificationStatus.UNSUPPORTED
            confidence = ConfidenceLevel.LOW
            rationale = (
                f"Claim introduces numerical metrics {nums_claim - nums_excerpt} "
                f"not established or supported by the excerpt."
            )
            conflict = False
            conflict_desc = None
        else:
            claim_words = {w for w in re.findall(r"\w+", claim_lower) if len(w) > 3}
            matched_words = {w for w in claim_words if w in excerpt_lower}
            overlap_ratio = len(matched_words) / len(claim_words) if claim_words else 0.0

            if overlap_ratio >= 0.7:
                status = VerificationStatus.SUPPORTED
                confidence = evidence.confidence
                rationale = "Excerpt directly entails and substantiates key components of the claim."
            elif overlap_ratio >= 0.35:
                status = VerificationStatus.INFERRED
                # Inferred support cannot yield HIGH confidence, even from high-authority sources
                confidence = (
                    ConfidenceLevel.MEDIUM
                    if evidence.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
                    else ConfidenceLevel.LOW
                )
                rationale = "Excerpt partially supports the claim, requiring reasonable domain inference."
            else:
                status = VerificationStatus.UNSUPPORTED
                confidence = ConfidenceLevel.LOW
                rationale = "Excerpt lacks sufficient factual overlap or substantiation for the claim."

            conflict = False
            conflict_desc = None

        # 4. Temporal Semantics Check
        # Distinguish what the source explicitly states vs when the event occurs
        temporal_notes = None
        if evidence.event_date:
            today = datetime.date.today()
            if evidence.event_date > today:
                # Event is scheduled or expected in the future
                announcement_terms = {
                    "announced", "plans", "planning", "targeting", "targets",
                    "scheduled", "aims", "set to", "expected to", "unveiled plans",
                }
                completed_terms = {
                    "launched", "completed", "acquired", "closed", "shipped", "achieved", "became",
                }

                claim_announces = any(w in claim_lower for w in announcement_terms)
                claim_claims_completed = any(w in claim_lower for w in completed_terms) and not claim_announces

                if claim_claims_completed:
                    # An announced future outcome cannot be marked as already completed
                    status = VerificationStatus.UNSUPPORTED
                    temporal_notes = (
                        f"Event date {evidence.event_date} is in the future; claim asserts completed outcome."
                    )
                elif claim_announces:
                    # An announcement or plan whose event date is in the future IS directly supported
                    temporal_notes = f"Future event ({evidence.event_date}) referenced in announcement/plan."
                    # If excerpt supported the announcement, retain SUPPORTED
                else:
                    temporal_notes = f"Anticipated future event date {evidence.event_date}."
                    if status == VerificationStatus.SUPPORTED:
                        status = VerificationStatus.INFERRED
            else:
                # Event is in the past (historical)
                pub_date = source.publication_date if source else None
                if pub_date and evidence.event_date < pub_date:
                    temporal_notes = f"Historical event ({evidence.event_date}) referenced relative to publication date {pub_date}."
                elif evidence.event_date < today:
                    temporal_notes = f"Historical event ({evidence.event_date}) referenced."

        # 5. Entity Resolution & Target Attribution Guardrails
        entity_notes = None
        target_attributed = True
        if entity:
            if target_entity_id and entity.entity_id != target_entity_id:
                # The assertion refers to a different entity (wrong person or affiliated entity)
                target_attributed = False
                entity_notes = (
                    f"Assertion refers to candidate entity '{entity.raw_name}' ({entity.entity_id}), "
                    f"NOT target prospect ({target_entity_id}). Target prospect attribution is excluded."
                )
            elif entity.resolution_status in (ResolutionStatus.UNRESOLVED, ResolutionStatus.AMBIGUOUS):
                target_attributed = False
                if confidence == ConfidenceLevel.HIGH:
                    confidence = ConfidenceLevel.MEDIUM
                entity_notes = (
                    f"Subject entity '{entity.raw_name}' has resolution status '{entity.resolution_status.value}'. "
                    f"Candidate assertion preserved, but not attributed to target prospect."
                )

        return EvidenceAssessment(
            assessment_status=status,
            confidence=confidence,
            is_independent=True,
            support_rationale=rationale,
            temporal_notes=temporal_notes,
            entity_notes=entity_notes,
            conflict_detected=conflict,
            conflict_description=conflict_desc,
            target_prospect_attributed=target_attributed,
        )

    def check_independence(
        self,
        source_id_a: str,
        source_id_b: str,
        syndication_groups: Optional[Dict[str, Set[str]]] = None,
        source_domains: Optional[Dict[str, str]] = None,
        excerpt_a: Optional[str] = None,
        excerpt_b: Optional[str] = None,
    ) -> bool:
        """Determine if two sources are genuinely independent or syndicated copies of the same release.
        
        Rules:
          - Identical source IDs -> NOT independent (repeated mention in same document).
          - Confirmed common origin: same syndication group -> NOT independent.
          - Conservative unknown provenance: missing domain data -> NOT independent.
          - Same domain -> NOT independent.
          - Suspected syndication: identical or near-identical excerpt text across different domains -> NOT independent.
          - Different domains, no origin group match, and distinct reporting -> INDEPENDENT.
        """
        if source_id_a == source_id_b:
            return False

        # 1. Confirmed common origin via syndication group membership
        if syndication_groups:
            for group_id, members in syndication_groups.items():
                if source_id_a in members and source_id_b in members:
                    logger.info("Sources %s and %s share origin group %s (confirmed syndication)", source_id_a, source_id_b, group_id)
                    return False

        # 2. Conservative handling of unknown provenance
        if not source_domains:
            logger.info("Missing domain mapping for sources %s / %s; cannot assume independence.", source_id_a, source_id_b)
            return False

        dom_a = source_domains.get(source_id_a)
        dom_b = source_domains.get(source_id_b)
        if not dom_a or not dom_b:
            logger.info("Unknown domain for source %s or %s; handling conservatively.", source_id_a, source_id_b)
            return False

        if dom_a.lower() == dom_b.lower():
            return False

        # 3. Suspected syndication across different domains
        if excerpt_a and excerpt_b:
            clean_a = re.sub(r"\s+", " ", excerpt_a.strip().lower())
            clean_b = re.sub(r"\s+", " ", excerpt_b.strip().lower())
            if clean_a == clean_b:
                logger.info("Sources %s (%s) and %s (%s) share identical text (suspected syndication)", source_id_a, dom_a, source_id_b, dom_b)
                return False

            words_a = {w for w in re.findall(r"\w+", clean_a) if len(w) > 3}
            words_b = {w for w in re.findall(r"\w+", clean_b) if len(w) > 3}
            if words_a and words_b:
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                if overlap >= 0.85:
                    logger.info("Sources %s and %s share >= 85%% excerpt overlap (suspected wire syndication)", source_id_a, source_id_b)
                    return False

        return True

