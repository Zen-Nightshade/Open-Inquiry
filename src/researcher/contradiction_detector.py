from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.molecular_models import (
    Modality,
    MolecularFact,
    NormalizedPredicate,
    Polarity,
    ValueType,
)
from src.researcher.temporal import (
    intervals_overlap,
    is_temporally_sequential,
    normalize_to_comparable_bounds,
)

logger = logging.getLogger(__name__)


class ContradictionStatus(str, Enum):
    CONTRADICTION = "contradiction"              # Confirmed mutually exclusive assertions
    POTENTIAL_CONTRADICTION = "potential_contradiction"  # Plausible conflict requiring reconciliation/unconfirmed
    NO_CONTRADICTION = "no_contradiction"        # Assertions are compatible or independent


class ContradictionEvaluation(BaseModel):
    """Detailed evaluation result of comparing two molecular factual propositions."""

    model_config = ConfigDict(frozen=True)

    status: ContradictionStatus
    severity: str = Field(description="'confirmed', 'potential', or 'none'")
    reason: str = Field(description="Explanation of the contradiction or compatibility")
    fact_id_a: str = Field(description="Parent CanonicalFact ID or Molecular ID A")
    fact_id_b: str = Field(description="Parent CanonicalFact ID or Molecular ID B")
    molecular_id_a: Optional[str] = None
    molecular_id_b: Optional[str] = None
    conflicting_attributes: List[str] = Field(default_factory=list)


class ContradictionDetector:
    """Conservative contradiction detector for structured molecular facts with bitemporal awareness."""

    # Note: FACILITY_OPENED / FACILITY_CLOSED and LEADERSHIP_APPOINTED / LEADERSHIP_DEPARTED
    # are distinct point-in-time lifecycle events and must NOT be treated as automatic contradictions.
    _OPPOSED_PREDICATES: Dict[NormalizedPredicate, NormalizedPredicate] = {}

    def evaluate(self, fact_a: MolecularFact, fact_b: MolecularFact) -> ContradictionEvaluation:
        """Evaluate whether two molecular propositions contradict each other."""
        # 1. Subject Identity Check
        # If both have entity IDs and they differ -> NO_CONTRADICTION
        if fact_a.subject_entity_id and fact_b.subject_entity_id:
            if fact_a.subject_entity_id != fact_b.subject_entity_id:
                return self._no_conflict(fact_a, fact_b, "Subjects refer to distinct resolved entity IDs.")
        else:
            # Normalize names and compare
            name_a = fact_a.subject_name.strip().lower()
            name_b = fact_b.subject_name.strip().lower()
            if name_a and name_b and name_a != name_b and not (name_a in name_b or name_b in name_a):
                return self._no_conflict(fact_a, fact_b, f"Subjects '{fact_a.subject_name}' and '{fact_b.subject_name}' appear to be distinct.")

        # 2. Predicate Relationship Check
        norm_a = fact_a.normalized_predicate
        norm_b = fact_b.normalized_predicate

        # If both are unnormalized, fallback to raw comparison
        if not norm_a or not norm_b:
            if fact_a.raw_predicate.strip().lower() != fact_b.raw_predicate.strip().lower():
                return self._no_conflict(fact_a, fact_b, "Unnormalized predicates describe distinct actions.")
            # If same raw predicate, proceed to value check
        else:
            # Both normalized
            is_same_pred = norm_a == norm_b
            is_opposed_pred = self._OPPOSED_PREDICATES.get(norm_a) == norm_b

            is_employment_pair = (
                norm_a in (NormalizedPredicate.LEADERSHIP_APPOINTED, NormalizedPredicate.LEADERSHIP_DEPARTED) and
                norm_b in (NormalizedPredicate.LEADERSHIP_APPOINTED, NormalizedPredicate.LEADERSHIP_DEPARTED)
            )
            is_facility_pair = (
                norm_a in (NormalizedPredicate.FACILITY_OPENED, NormalizedPredicate.FACILITY_CLOSED) and
                norm_b in (NormalizedPredicate.FACILITY_OPENED, NormalizedPredicate.FACILITY_CLOSED)
            )

            if not is_same_pred and not is_opposed_pred and not is_employment_pair and not is_facility_pair:
                # Different, non-opposed predicates (e.g. FUNDING_RAISED vs HIRING_EXECUTED)
                return self._no_conflict(fact_a, fact_b, f"Predicates '{norm_a.value}' and '{norm_b.value}' are mutually compatible.")

        # 3. Polarity Contradiction
        # Same predicate + opposite polarity for overlapping scope -> CONTRADICTION
        if (norm_a == norm_b or fact_a.raw_predicate.strip().lower() == fact_b.raw_predicate.strip().lower()):
            if fact_a.polarity != fact_b.polarity:
                # Check location/scope match
                loc_a = fact_a.qualifiers.get("location")
                loc_b = fact_b.qualifiers.get("location")
                if not loc_a or not loc_b or loc_a.lower() == loc_b.lower():
                    date_a = fact_a.valid_from or (fact_a.event_date.isoformat() if fact_a.event_date else None)
                    date_b = fact_b.valid_from or (fact_b.event_date.isoformat() if fact_b.event_date else None)
                    if is_temporally_sequential(date_a, date_b):
                        return self._no_conflict(fact_a, fact_b, f"Polarity change across sequential periods: {date_a} vs {date_b}.")

                    return ContradictionEvaluation(
                        status=ContradictionStatus.CONTRADICTION,
                        severity="confirmed",
                        reason=f"Opposite polarity on predicate: '{fact_a.atom_text}' asserts positive while '{fact_b.atom_text}' asserts negative.",
                        fact_id_a=fact_a.parent_canonical_id,
                        fact_id_b=fact_b.parent_canonical_id,
                        molecular_id_a=fact_a.molecular_id,
                        molecular_id_b=fact_b.molecular_id,
                        conflicting_attributes=["polarity"],
                    )

        # 4. Modality Check: Planned vs Executed
        # Invariant: A planned future event and an executed completed event are NOT a contradiction
        # e.g. "plans to hire 500" vs "hired 50" can both be true
        if fact_a.modality == Modality.PLANNED and fact_b.modality == Modality.DEFINITIVE:
            return self._no_conflict(fact_a, fact_b, "Planned forward-looking target is compatible with executed historical state.")
        if fact_a.modality == Modality.DEFINITIVE and fact_b.modality == Modality.PLANNED:
            return self._no_conflict(fact_a, fact_b, "Executed historical state is compatible with planned forward-looking target.")

        # 5. Domain-Specific Contradiction Rules

        # A. Funding Raised: Check Round / Metric Conflict
        if norm_a == NormalizedPredicate.FUNDING_RAISED and norm_b == NormalizedPredicate.FUNDING_RAISED:
            return self._check_funding_contradiction(fact_a, fact_b)

        # B. Employment / Leadership Status: Joined vs Departed
        if (norm_a in (NormalizedPredicate.LEADERSHIP_APPOINTED, NormalizedPredicate.LEADERSHIP_DEPARTED) and
            norm_b in (NormalizedPredicate.LEADERSHIP_APPOINTED, NormalizedPredicate.LEADERSHIP_DEPARTED)):
            return self._check_employment_contradiction(fact_a, fact_b)

        # C. Facility Opened vs Closed
        if (norm_a in (NormalizedPredicate.FACILITY_OPENED, NormalizedPredicate.FACILITY_CLOSED) and
            norm_b in (NormalizedPredicate.FACILITY_OPENED, NormalizedPredicate.FACILITY_CLOSED)):
            return self._check_facility_contradiction(fact_a, fact_b)

        # D. Employee Count Metrics
        if norm_a == NormalizedPredicate.EMPLOYEE_COUNT and norm_b == NormalizedPredicate.EMPLOYEE_COUNT:
            return self._check_headcount_contradiction(fact_a, fact_b)

        return self._no_conflict(fact_a, fact_b, "Assertions are mutually compatible across available qualifiers.")

    def _check_funding_contradiction(
        self, fact_a: MolecularFact, fact_b: MolecularFact
    ) -> ContradictionEvaluation:
        """Evaluate potential funding contradictions with scope awareness (round vs total)."""
        round_a = fact_a.qualifiers.get("round")
        round_b = fact_b.qualifiers.get("round")

        # Total qualifiers check
        text_a = (fact_a.atom_text + " " + fact_a.object_value.raw_value).lower()
        text_b = (fact_b.atom_text + " " + fact_b.object_value.raw_value).lower()
        total_markers = ("total", "raised to date", "cumulatively", "in total", "all rounds")
        a_is_total = any(m in text_a for m in total_markers)
        b_is_total = any(m in text_b for m in total_markers)

        # If one is total funding and the other is a round funding -> NOT a contradiction
        if a_is_total ^ b_is_total:
            return self._no_conflict(fact_a, fact_b, "Compatible scopes: one fact reports round funding while the other reports cumulative total funding.")

        # If both specify the same round (e.g. Series B)
        if round_a and round_b and round_a.lower() == round_b.lower():
            val_a = fact_a.object_value.normalized_value
            val_b = fact_b.object_value.normalized_value
            unit_a = fact_a.object_value.unit
            unit_b = fact_b.object_value.unit

            if (val_a is not None and val_b is not None and
                unit_a and unit_b and unit_a == unit_b and val_a != val_b):
                return ContradictionEvaluation(
                    status=ContradictionStatus.CONTRADICTION,
                    severity="confirmed",
                    reason=f"[CONFIRMED] Conflicting metrics for {round_a.upper()}: '{fact_a.atom_text}' ({val_a} {unit_a}) vs '{fact_b.atom_text}' ({val_b} {unit_b})",
                    fact_id_a=fact_a.parent_canonical_id,
                    fact_id_b=fact_b.parent_canonical_id,
                    molecular_id_a=fact_a.molecular_id,
                    molecular_id_b=fact_b.molecular_id,
                    conflicting_attributes=["object_value", "normalized_value"],
                )

        return self._no_conflict(fact_a, fact_b, "Funding metrics describe different rounds, dates, or non-conflicting scopes.")

    def _check_employment_contradiction(
        self, fact_a: MolecularFact, fact_b: MolecularFact
    ) -> ContradictionEvaluation:
        """Evaluate leadership/employment assertions (appointment vs departure)."""
        is_opposed = (
            (fact_a.normalized_predicate == NormalizedPredicate.LEADERSHIP_APPOINTED and
             fact_b.normalized_predicate == NormalizedPredicate.LEADERSHIP_DEPARTED) or
            (fact_a.normalized_predicate == NormalizedPredicate.LEADERSHIP_DEPARTED and
             fact_b.normalized_predicate == NormalizedPredicate.LEADERSHIP_APPOINTED)
        )

        if is_opposed:
            # Appointment and departure are distinct lifecycle events, NOT automatic contradictions!
            date_a = fact_a.valid_from or (fact_a.event_date.isoformat() if fact_a.event_date else None)
            date_b = fact_b.valid_from or (fact_b.event_date.isoformat() if fact_b.event_date else None)
            if is_temporally_sequential(date_a, date_b):
                return self._no_conflict(fact_a, fact_b, f"Sequential career milestones with distinct dates: {date_a} vs {date_b}.")

            # If both explicitly claim mutually exclusive concurrent current status on the exact same date
            text_a = fact_a.atom_text.lower()
            text_b = fact_b.atom_text.lower()
            current_markers = ("currently", "current", "serving as", "serves as", "heads", "leads")
            a_is_current = any(m in text_a for m in current_markers)
            b_is_current = any(m in text_b for m in current_markers)

            if (date_a and date_b and date_a == date_b) or a_is_current or b_is_current:
                return ContradictionEvaluation(
                    status=ContradictionStatus.CONTRADICTION,
                    severity="confirmed",
                    reason=f"[CONFIRMED] Contradictory concurrent employment status: '{fact_a.atom_text}' vs '{fact_b.atom_text}'",
                    fact_id_a=fact_a.parent_canonical_id,
                    fact_id_b=fact_b.parent_canonical_id,
                    molecular_id_a=fact_a.molecular_id,
                    molecular_id_b=fact_b.molecular_id,
                    conflicting_attributes=["normalized_predicate"],
                )

            # Potential conflict: missing dates prevent determining whether milestones were sequential
            return ContradictionEvaluation(
                status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                severity="potential",
                reason=f"[POTENTIAL] Potential conflict (missing dates): '{fact_a.atom_text}' vs '{fact_b.atom_text}'",
                fact_id_a=fact_a.parent_canonical_id,
                fact_id_b=fact_b.parent_canonical_id,
                molecular_id_a=fact_a.molecular_id,
                molecular_id_b=fact_b.molecular_id,
                conflicting_attributes=["normalized_predicate"],
            )

        return self._no_conflict(fact_a, fact_b, "Employment milestones are compatible.")

    def _check_facility_contradiction(
        self, fact_a: MolecularFact, fact_b: MolecularFact
    ) -> ContradictionEvaluation:
        """Evaluate facility opened vs closed contradictions."""
        loc_a = fact_a.qualifiers.get("location") or fact_a.object_value.raw_value
        loc_b = fact_b.qualifiers.get("location") or fact_b.object_value.raw_value

        # If locations clearly differ -> separate facilities
        if loc_a and loc_b and loc_a.lower() != loc_b.lower():
            return self._no_conflict(fact_a, fact_b, "Facility events refer to distinct geographic locations.")

        is_opposed = fact_a.normalized_predicate != fact_b.normalized_predicate
        if is_opposed:
            date_a = fact_a.valid_from or (fact_a.event_date.isoformat() if fact_a.event_date else None)
            date_b = fact_b.valid_from or (fact_b.event_date.isoformat() if fact_b.event_date else None)
            if is_temporally_sequential(date_a, date_b):
                return self._no_conflict(fact_a, fact_b, f"Sequential facility lifecycle milestones: {date_a} vs {date_b}.")

            # Distinct lifecycle events (opening and closing) are NOT automatic contradictions
            return self._no_conflict(
                fact_a, fact_b, "Facility opened and facility closed are distinct lifecycle events, not a contradiction."
            )

        return self._no_conflict(fact_a, fact_b, "Facility events are compatible.")

    def _check_headcount_contradiction(
        self, fact_a: MolecularFact, fact_b: MolecularFact
    ) -> ContradictionEvaluation:
        """Evaluate headcount metrics, respecting scope and bitemporal dates."""
        loc_a = fact_a.qualifiers.get("location")
        loc_b = fact_b.qualifiers.get("location")

        # Regional office headcount vs global headcount are compatible
        if loc_a and loc_b and loc_a.lower() != loc_b.lower():
            return self._no_conflict(fact_a, fact_b, "Headcount metrics describe different office locations.")

        val_a = fact_a.object_value.normalized_value
        val_b = fact_b.object_value.normalized_value

        # If values match, no contradiction
        if val_a is not None and val_b is not None and val_a == val_b:
            return self._no_conflict(fact_a, fact_b, "Headcount values are identical.")

        # Check bitemporal valid dates
        date_a = fact_a.valid_from or (fact_a.event_date.isoformat() if fact_a.event_date else None)
        date_b = fact_b.valid_from or (fact_b.event_date.isoformat() if fact_b.event_date else None)

        if val_a is not None and val_b is not None and val_a != val_b:
            if is_temporally_sequential(date_a, date_b, fact_a.valid_to, fact_b.valid_to):
                # Sequential dates reflect organic workforce growth or change over time
                return self._no_conflict(
                    fact_a,
                    fact_b,
                    f"Headcount metrics describe sequential dates ({date_a} vs {date_b}) representing evolution over time.",
                )

            # Check for overlapping known validity periods
            overlaps, is_certain = intervals_overlap(date_a, fact_a.valid_to, date_b, fact_b.valid_to)
            if overlaps and is_certain:
                return ContradictionEvaluation(
                    status=ContradictionStatus.CONTRADICTION,
                    severity="confirmed",
                    reason=f"[CONFIRMED] Conflicting concurrent headcount metrics for overlapping period ({date_a} vs {date_b}): '{fact_a.atom_text}' ({val_a}) vs '{fact_b.atom_text}' ({val_b})",
                    fact_id_a=fact_a.parent_canonical_id,
                    fact_id_b=fact_b.parent_canonical_id,
                    molecular_id_a=fact_a.molecular_id,
                    molecular_id_b=fact_b.molecular_id,
                    conflicting_attributes=["object_value", "valid_from"],
                )

            # Missing or uncertain dates: cannot establish simultaneity or sequence
            return ContradictionEvaluation(
                status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                severity="potential",
                reason=f"[POTENTIAL] Conflicting headcount metrics with uncertain or missing dates ({date_a} vs {date_b}): '{fact_a.atom_text}' ({val_a}) vs '{fact_b.atom_text}' ({val_b})",
                fact_id_a=fact_a.parent_canonical_id,
                fact_id_b=fact_b.parent_canonical_id,
                molecular_id_a=fact_a.molecular_id,
                molecular_id_b=fact_b.molecular_id,
                conflicting_attributes=["object_value"],
            )

        return self._no_conflict(fact_a, fact_b, "Headcount metrics are compatible.")

    def _no_conflict(
        self, fact_a: MolecularFact, fact_b: MolecularFact, reason: str
    ) -> ContradictionEvaluation:
        return ContradictionEvaluation(
            status=ContradictionStatus.NO_CONTRADICTION,
            severity="none",
            reason=reason,
            fact_id_a=fact_a.parent_canonical_id,
            fact_id_b=fact_b.parent_canonical_id,
            molecular_id_a=fact_a.molecular_id,
            molecular_id_b=fact_b.molecular_id,
            conflicting_attributes=[],
        )
