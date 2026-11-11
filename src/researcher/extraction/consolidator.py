from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.contradiction_detector import ContradictionDetector, ContradictionStatus
from src.researcher.extraction.assessment import EvidenceAssessor
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
)
from src.researcher.temporal import is_temporally_sequential

logger = logging.getLogger(__name__)


from enum import Enum


class ConflictSeverity(str, Enum):
    CONFIRMED = "confirmed"
    POTENTIAL = "potential"


class DetectedConflict(BaseModel):
    """Contradiction detected between two factual assertions regarding the same entity."""

    model_config = ConfigDict(frozen=True)

    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_id_a: str
    fact_id_b: str
    description: str
    entity_id: Optional[str] = None
    severity: ConflictSeverity = Field(default=ConflictSeverity.CONFIRMED)


class FactConsolidationResult(BaseModel):
    """Outcome of consolidating new facts into accumulator, tracking detected conflicts."""

    model_config = ConfigDict(frozen=True)

    consolidated_facts: List[CanonicalFact]
    conflicts: List[DetectedConflict] = Field(default_factory=list)


def extract_numerical_tokens(text: str) -> Set[str]:
    """Extract dollar amounts, percentages, and numbers from claim text."""
    matches = re.findall(r"\$?\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:million|billion|m|b|k|%))?", text, re.IGNORECASE)
    return {m.strip().lower() for m in matches if m.strip()}


class FactConsolidator:
    """Consolidates new facts into the research accumulator, detecting syndication duplicates and conflicts."""

    def __init__(
        self,
        assessor: Optional[EvidenceAssessor] = None,
        detector: Optional[ContradictionDetector] = None,
    ) -> None:
        self.assessor = assessor or EvidenceAssessor()
        self.detector = detector or ContradictionDetector()

    def consolidate(
        self,
        new_facts: List[CanonicalFact],
        accumulator: ResearchAccumulator,
        syndication_groups: Optional[Dict[str, Set[str]]] = None,
        source_domains: Optional[Dict[str, str]] = None,
    ) -> FactConsolidationResult:
        """Consolidate facts into accumulator, preserving all evidence and identifying conflicts."""
        detected_conflicts: List[DetectedConflict] = []

        for new_fact in new_facts:
            matched_fact_id: Optional[str] = None
            is_independent = True
            conflict_found = False

            # Compare against existing facts in accumulator
            for existing in accumulator.facts:
                # 1. Check for Contradiction / Conflict FIRST
                conflict_res = self._check_for_conflict(new_fact, existing)
                if conflict_res:
                    conflict_desc, severity = conflict_res
                    conflict = DetectedConflict(
                        fact_id_a=existing.fact_id,
                        fact_id_b=new_fact.fact_id,
                        description=conflict_desc,
                        entity_id=new_fact.entity_ids[0] if new_fact.entity_ids else None,
                        severity=severity,
                    )
                    detected_conflicts.append(conflict)
                    conflict_found = True
                    logger.warning("Detected conflict [%s]: %s", severity.value, conflict_desc)
                    # When in conflict, do not merge; both facts must be preserved separately!
                    break

                # 2. Check for Match / Deduplication
                if self._are_facts_matching(new_fact, existing):
                    matched_fact_id = existing.fact_id
                    # Determine whether evidence is independent
                    is_independent = self._check_evidence_independence(
                        new_fact.evidence_records,
                        existing.evidence_records,
                        syndication_groups,
                        source_domains,
                    )
                    break

            if matched_fact_id and not conflict_found:
                accumulator.merge_or_add_fact(
                    new_fact=new_fact,
                    match_fact_id=matched_fact_id,
                    is_independent=is_independent,
                    syndication_groups=syndication_groups,
                    source_domains=source_domains,
                )
            else:
                accumulator.merge_or_add_fact(
                    new_fact=new_fact,
                    syndication_groups=syndication_groups,
                    source_domains=source_domains,
                )

        return FactConsolidationResult(
            consolidated_facts=list(accumulator.facts),
            conflicts=detected_conflicts,
        )

    def _are_facts_matching(self, fact_a: CanonicalFact, fact_b: CanonicalFact) -> bool:
        """Determine if two facts represent the same underlying assertion."""
        # Must share category
        if fact_a.fact_category != fact_b.fact_category:
            return False

        # If there is a direct conflict, they do not match
        if self._check_for_conflict(fact_a, fact_b):
            return False

        # Must share entity if entities are identified
        if fact_a.entity_ids and fact_b.entity_ids:
            if not (set(fact_a.entity_ids) & set(fact_b.entity_ids)):
                return False

        # Temporal compatibility: if both have dates, they must be within ~30 days or same year
        if fact_a.primary_event_date and fact_b.primary_event_date:
            diff_days = abs((fact_a.primary_event_date - fact_b.primary_event_date).days)
            if diff_days > 45:
                return False

        # Bitemporal validity check: sequential distinct dates are distinct historical assertions
        if fact_a.valid_from and fact_b.valid_from and is_temporally_sequential(fact_a.valid_from, fact_b.valid_from):
            return False

        # Text comparison
        text_a = fact_a.claim_text.lower().strip()
        text_b = fact_b.claim_text.lower().strip()

        if text_a == text_b:
            return True

        # Check numerical tokens: if numerical metrics differ (e.g. $50M vs $30M), they do NOT match
        nums_a = extract_numerical_tokens(text_a)
        nums_b = extract_numerical_tokens(text_b)
        if nums_a and nums_b and nums_a != nums_b:
            return False

        # If both facts share numerical tokens (e.g. $50 million) and round/series, they match
        series_matches_a = re.findall(r"\b(?:series\s+[a-z]|seed|ipo|q[1-4])\b", text_a)
        series_matches_b = re.findall(r"\b(?:series\s+[a-z]|seed|ipo|q[1-4])\b", text_b)
        if nums_a and nums_b and nums_a == nums_b and series_matches_a and series_matches_a == series_matches_b:
            return True

        # Significant word overlap
        words_a = {w for w in re.findall(r"\w+", text_a) if len(w) > 3}
        words_b = {w for w in re.findall(r"\w+", text_b) if len(w) > 3}

        if words_a and words_b:
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            return overlap >= 0.60

        return False

    def _check_for_conflict(
        self, fact_a: CanonicalFact, fact_b: CanonicalFact
    ) -> Optional[Tuple[str, ConflictSeverity]]:
        """Check if two facts directly contradict each other regarding subject, predicate, object, qualifiers, and temporal scope."""
        # Must share category and entity
        if fact_a.fact_category != fact_b.fact_category:
            return None

        if not fact_a.entity_ids or not fact_b.entity_ids:
            return None

        if not (set(fact_a.entity_ids) & set(fact_b.entity_ids)):
            return None

        # 0. Check structured molecular proposition contradiction if available
        if fact_a.molecular_facts and fact_b.molecular_facts:
            for mol_a in fact_a.molecular_facts:
                for mol_b in fact_b.molecular_facts:
                    eval_res = self.detector.evaluate(mol_a, mol_b)
                    if eval_res.status == ContradictionStatus.CONTRADICTION:
                        return (eval_res.reason, ConflictSeverity.CONFIRMED)
                    elif eval_res.status == ContradictionStatus.POTENTIAL_CONTRADICTION:
                        return (eval_res.reason, ConflictSeverity.POTENTIAL)

        text_a = fact_a.claim_text.lower().strip()
        text_b = fact_b.claim_text.lower().strip()

        # 1. Funding metrics conflict: compare only when metric scope is compatible
        nums_a = extract_numerical_tokens(text_a)
        nums_b = extract_numerical_tokens(text_b)
        if nums_a and nums_b and nums_a != nums_b:
            total_qualifiers = {"total", "raised to date", "cumulatively", "in total", "all rounds"}
            a_is_total = any(q in text_a for q in total_qualifiers)
            b_is_total = any(q in text_b for q in total_qualifiers)

            # If one is total funding and the other is a round, they are compatible scopes
            if not (a_is_total ^ b_is_total):
                # Check if both refer to the same specific round (e.g. Series B, seed)
                series_matches_a = re.findall(r"\b(?:series\s+[a-z]|seed|ipo|q[1-4])\b", text_a)
                series_matches_b = re.findall(r"\b(?:series\s+[a-z]|seed|ipo|q[1-4])\b", text_b)
                if series_matches_a and series_matches_b and set(series_matches_a) == set(series_matches_b):
                    return (
                        f"[CONFIRMED] Conflicting metrics for {series_matches_a[0].upper()}: "
                        f"'{fact_a.claim_text}' vs '{fact_b.claim_text}'",
                        ConflictSeverity.CONFIRMED,
                    )

        # 2. Employment status contradiction: distinguish sequential career milestones from concurrent incompatible claims
        if fact_a.fact_category in (FactCategory.HIRING, FactCategory.ORGANIZATION_CHANGE):
            joined_words = {
                "joined", "appointed", "named", "promoted", "hired", "started",
                "serves", "serving", "heads", "leads", "works", "employed",
            }
            left_words = {"resigned", "stepped down", "left", "departed", "ousted", "fired"}
            current_words = {"currently", "current", "serving as", "heads", "leads"}

            a_joins = any(w in text_a for w in joined_words)
            a_leaves = any(w in text_a for w in left_words)
            b_joins = any(w in text_b for w in joined_words)
            b_leaves = any(w in text_b for w in left_words)

            a_is_current = any(w in text_a for w in current_words)
            b_is_current = any(w in text_b for w in current_words)

            is_join_vs_leave = (
                ((a_joins or a_is_current) and b_leaves)
                or ((b_joins or b_is_current) and a_leaves)
            )
            if is_join_vs_leave:
                date_a = fact_a.primary_event_date
                date_b = fact_b.primary_event_date
                # If distinct sequential dates exist, they represent sequential milestones
                if date_a and date_b and date_a != date_b:
                    logger.info(
                        "Sequential employment milestones: %s (%s) vs %s (%s) are not contradictory",
                        fact_a.claim_text, date_a, fact_b.claim_text, date_b
                    )
                    return None

                a_is_current = any(w in text_a for w in current_words)
                b_is_current = any(w in text_b for w in current_words)

                # Confirmed conflict: identical event dates, or explicit concurrent current status
                if (date_a and date_b and date_a == date_b) or a_is_current or b_is_current:
                    return (
                        f"[CONFIRMED] Contradictory employment status: '{fact_a.claim_text}' vs '{fact_b.claim_text}'",
                        ConflictSeverity.CONFIRMED,
                    )

                # Potential conflict: missing dates prevent determining whether milestones were sequential
                return (
                    f"[POTENTIAL] Potential conflict (missing dates): '{fact_a.claim_text}' vs '{fact_b.claim_text}'",
                    ConflictSeverity.POTENTIAL,
                )

        return None

    def _check_evidence_independence(
        self,
        new_ev_list: List[EvidenceRecord],
        existing_ev_list: List[EvidenceRecord],
        syndication_groups: Optional[Dict[str, Set[str]]],
        source_domains: Optional[Dict[str, str]],
    ) -> bool:
        """Check if any new evidence comes from a genuinely independent source."""
        for new_ev in new_ev_list:
            for ex_ev in existing_ev_list:
                if not self.assessor.check_independence(
                    new_ev.source_id,
                    ex_ev.source_id,
                    syndication_groups,
                    source_domains,
                    excerpt_a=new_ev.verbatim_excerpt,
                    excerpt_b=ex_ev.verbatim_excerpt,
                ):
                    return False
        return True
