from __future__ import annotations

from datetime import date
import uuid
import pytest

from src.researcher.contradiction_detector import ContradictionDetector, ContradictionStatus
from src.researcher.extraction.consolidator import ConflictSeverity, FactConsolidator
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    VerificationStatus,
)
from src.researcher.molecular_models import (
    Modality,
    MolecularFact,
    MolecularValue,
    NormalizedPredicate,
    Polarity,
    ValueType,
)
from src.researcher.temporal import TemporalKind


def test_facility_opened_and_facility_closed_are_not_contradictory():
    """Verify FACILITY_OPENED and FACILITY_CLOSED are distinct lifecycle events, not mutual contradictions."""
    detector = ContradictionDetector()
    parent_id = str(uuid.uuid4())

    fact_opened = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme opened a manufacturing plant in Detroit",
        subject_name="Acme",
        raw_predicate="opened",
        normalized_predicate=NormalizedPredicate.FACILITY_OPENED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.LOCATION, raw_value="plant in Detroit"),
        valid_from="2018",
        temporal_kind=TemporalKind.EVENT,
    )

    fact_closed = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme closed a manufacturing plant in Detroit",
        subject_name="Acme",
        raw_predicate="closed",
        normalized_predicate=NormalizedPredicate.FACILITY_CLOSED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.LOCATION, raw_value="plant in Detroit"),
        valid_from="2023",
        temporal_kind=TemporalKind.EVENT,
    )

    res = detector.evaluate(fact_opened, fact_closed)
    assert res.status == ContradictionStatus.NO_CONTRADICTION


def test_leadership_appointed_and_leadership_departed_are_not_contradictory():
    """Verify LEADERSHIP_APPOINTED and LEADERSHIP_DEPARTED are distinct career milestones, not contradictions."""
    detector = ContradictionDetector()
    parent_id = str(uuid.uuid4())

    fact_appointed = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Alice was appointed CTO of Acme",
        subject_name="Acme",
        raw_predicate="appointed",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_APPOINTED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="CTO"),
        qualifiers={"person": "Alice"},
        valid_from="2020",
        temporal_kind=TemporalKind.EVENT,
    )

    fact_departed = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Alice stepped down as CTO of Acme",
        subject_name="Acme",
        raw_predicate="stepped down as",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_DEPARTED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="CTO"),
        qualifiers={"person": "Alice"},
        valid_from="2024",
        temporal_kind=TemporalKind.EVENT,
    )

    res = detector.evaluate(fact_appointed, fact_departed)
    assert res.status == ContradictionStatus.NO_CONTRADICTION


def test_sequential_headcount_is_growth_over_time_not_contradiction():
    """Headcount reported as 500 in 2022 and 800 in 2024 represents growth over time, not a contradiction."""
    detector = ContradictionDetector()
    parent_id = str(uuid.uuid4())

    fact_2022 = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme has 500 employees",
        subject_name="Acme",
        raw_predicate="has headcount",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="500", normalized_value=500),
        valid_from="2022",
        temporal_kind=TemporalKind.STATE,
    )

    fact_2024 = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme has 800 employees",
        subject_name="Acme",
        raw_predicate="has headcount",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="800", normalized_value=800),
        valid_from="2024",
        temporal_kind=TemporalKind.STATE,
    )

    res = detector.evaluate(fact_2022, fact_2024)
    assert res.status == ContradictionStatus.NO_CONTRADICTION


def test_overlapping_headcount_with_conflicting_counts_is_contradiction():
    """Conflicting headcount assertions within overlapping temporal bounds constitute a confirmed contradiction."""
    detector = ContradictionDetector()
    parent_id = str(uuid.uuid4())

    fact_a = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme has 500 employees in March 2024",
        subject_name="Acme",
        raw_predicate="has headcount",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="500", normalized_value=500),
        valid_from="2024-03-01",
        valid_to="2024-03-31",
        temporal_kind=TemporalKind.STATE,
    )

    fact_b = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme has 1200 employees in March 2024",
        subject_name="Acme",
        raw_predicate="has headcount",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="1200", normalized_value=1200),
        valid_from="2024-03-15",
        valid_to="2024-03-20",
        temporal_kind=TemporalKind.STATE,
    )

    res = detector.evaluate(fact_a, fact_b)
    assert res.status == ContradictionStatus.CONTRADICTION
    assert "(500)" in res.reason and "(1200)" in res.reason


def test_missing_dates_do_not_assume_simultaneity():
    """When dates are unknown or missing, conflicting metrics flag POTENTIAL rather than CONFIRMED simultaneity."""
    detector = ContradictionDetector()
    parent_id = str(uuid.uuid4())

    fact_no_date = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme has 500 employees",
        subject_name="Acme",
        raw_predicate="has headcount",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="500", normalized_value=500),
        valid_from=None,
        temporal_kind=TemporalKind.STATE,
    )

    fact_dated = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme has 900 employees in 2024",
        subject_name="Acme",
        raw_predicate="has headcount",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="900", normalized_value=900),
        valid_from="2024",
        temporal_kind=TemporalKind.STATE,
    )

    res = detector.evaluate(fact_no_date, fact_dated)
    # Because date is missing, system does NOT assume simultaneity
    assert res.status == ContradictionStatus.POTENTIAL_CONTRADICTION


def test_consolidator_preserves_sequential_facts_without_merging():
    """Verify FactConsolidator retains both 2022 and 2024 assertions as distinct records in the accumulator."""
    consolidator = FactConsolidator()
    accumulator = ResearchAccumulator()

    ev1 = EvidenceRecord(
        source_id="src_1",
        source_url="https://source1.example.com",
        verbatim_excerpt="Acme had 500 employees in 2022",
    )
    ev2 = EvidenceRecord(
        source_id="src_2",
        source_url="https://source2.example.com",
        verbatim_excerpt="Acme expanded to 800 employees in 2024",
    )

    fact_2022 = CanonicalFact(
        fact_id="fact_2022",
        claim_text="Acme has 500 employees in 2022",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        entity_ids=("ent_acme",),
        valid_from="2022",
        evidence_records=[ev1],
    )

    fact_2024 = CanonicalFact(
        fact_id="fact_2024",
        claim_text="Acme has 800 employees in 2024",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        entity_ids=("ent_acme",),
        valid_from="2024",
        evidence_records=[ev2],
    )

    result = consolidator.consolidate([fact_2022, fact_2024], accumulator)

    # Both facts must be preserved separately in accumulator
    assert len(result.consolidated_facts) == 2
    fact_ids = {f.fact_id for f in result.consolidated_facts}
    assert "fact_2022" in fact_ids
    assert "fact_2024" in fact_ids
    # No false conflict reported
    assert len(result.conflicts) == 0
