from datetime import date
import pytest

from src.researcher.contradiction_detector import ContradictionDetector, ContradictionStatus
from src.researcher.molecular_models import (
    Modality,
    MolecularFact,
    MolecularValue,
    NormalizedPredicate,
    Polarity,
    ValueType,
)


@pytest.fixture
def detector():
    return ContradictionDetector()


def test_different_subjects_no_contradiction(detector):
    """Assertions about different entities are not contradictory."""
    fact_a = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Acme Corp raised $50 million in Series B",
        subject_name="Acme Corp",
        subject_entity_id="ent_acme",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$50M", normalized_value=50000000.0, unit="USD"),
        qualifiers={"round": "Series B"},
    )

    fact_b = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Beta Inc raised $30 million in Series B",
        subject_name="Beta Inc",
        subject_entity_id="ent_beta",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$30M", normalized_value=30000000.0, unit="USD"),
        qualifiers={"round": "Series B"},
    )

    eval_res = detector.evaluate(fact_a, fact_b)
    assert eval_res.status == ContradictionStatus.NO_CONTRADICTION
    assert eval_res.severity == "none"


def test_funding_same_round_different_amounts_is_confirmed_contradiction(detector):
    """When two facts state conflicting amounts for the exact same round, it is a confirmed contradiction."""
    fact_a = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Vanguard AI raised $50 million in Series B",
        subject_name="Vanguard AI",
        subject_entity_id="ent_vanguard",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$50 million", normalized_value=50000000.0, unit="USD"),
        qualifiers={"round": "Series B"},
    )

    fact_b = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Vanguard AI secured $30 million in Series B",
        subject_name="Vanguard AI",
        subject_entity_id="ent_vanguard",
        raw_predicate="secured",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$30 million", normalized_value=30000000.0, unit="USD"),
        qualifiers={"round": "Series B"},
    )

    eval_res = detector.evaluate(fact_a, fact_b)
    assert eval_res.status == ContradictionStatus.CONTRADICTION
    assert eval_res.severity == "confirmed"
    assert "conflicting metrics for series b" in eval_res.reason.lower()


def test_funding_round_vs_total_is_not_contradiction(detector):
    """Total funding ($50M) vs round funding ($30M) have compatible scopes."""
    fact_round = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Vanguard AI raised $30 million in Series B",
        subject_name="Vanguard AI",
        subject_entity_id="ent_vanguard",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$30 million", normalized_value=30000000.0, unit="USD"),
        qualifiers={"round": "Series B"},
    )

    fact_total = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Vanguard AI raised $50 million total funding to date",
        subject_name="Vanguard AI",
        subject_entity_id="ent_vanguard",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$50 million total funding to date", normalized_value=50000000.0, unit="USD"),
        qualifiers={},
    )

    eval_res = detector.evaluate(fact_round, fact_total)
    assert eval_res.status == ContradictionStatus.NO_CONTRADICTION
    assert "compatible scopes" in eval_res.reason.lower()


def test_planned_vs_executed_is_not_contradiction(detector):
    """A forward-looking planned target and an executed state are compatible."""
    fact_planned = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Acme Corp plans to hire 500 engineers",
        subject_name="Acme Corp",
        subject_entity_id="ent_acme",
        raw_predicate="plans to hire",
        normalized_predicate=NormalizedPredicate.HIRING_PLANNED,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="500 engineers", normalized_value=500, unit="engineers"),
        modality=Modality.PLANNED,
    )

    fact_executed = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Acme Corp hired 50 engineers",
        subject_name="Acme Corp",
        subject_entity_id="ent_acme",
        raw_predicate="hired",
        normalized_predicate=NormalizedPredicate.HIRING_EXECUTED,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="50 engineers", normalized_value=50, unit="engineers"),
        modality=Modality.DEFINITIVE,
    )

    eval_res = detector.evaluate(fact_planned, fact_executed)
    assert eval_res.status == ContradictionStatus.NO_CONTRADICTION


def test_sequential_employment_milestones_not_contradiction(detector):
    """Joining in 2021 and resigning in 2024 represent sequential career milestones."""
    fact_join = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Sarah Connor joined Vanguard AI in 2021",
        subject_name="Sarah Connor",
        subject_entity_id="ent_sarah",
        raw_predicate="joined",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_APPOINTED,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="Vanguard AI"),
        event_date=date(2021, 5, 1),
    )

    fact_leave = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Sarah Connor resigned from Vanguard AI in 2024",
        subject_name="Sarah Connor",
        subject_entity_id="ent_sarah",
        raw_predicate="resigned",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_DEPARTED,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="from Vanguard AI"),
        event_date=date(2024, 6, 1),
    )

    eval_res = detector.evaluate(fact_join, fact_leave)
    assert eval_res.status == ContradictionStatus.NO_CONTRADICTION
    assert "sequential career milestones" in eval_res.reason.lower()


def test_concurrent_employment_opposed_is_confirmed_contradiction(detector):
    """Currently serving vs resigned (or same event date) is a confirmed contradiction."""
    fact_serve = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Sarah Connor currently serves as VP of Engineering",
        subject_name="Sarah Connor",
        subject_entity_id="ent_sarah",
        raw_predicate="serves as",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_APPOINTED,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="VP of Engineering"),
    )

    fact_resign = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Sarah Connor resigned as VP of Engineering",
        subject_name="Sarah Connor",
        subject_entity_id="ent_sarah",
        raw_predicate="resigned",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_DEPARTED,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="VP of Engineering"),
    )

    eval_res = detector.evaluate(fact_serve, fact_resign)
    assert eval_res.status == ContradictionStatus.CONTRADICTION
    assert eval_res.severity == "confirmed"


def test_missing_dates_employment_opposed_is_potential_contradiction(detector):
    """Undated joined vs resigned without concurrent assertion is a potential contradiction."""
    fact_join = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Sarah Connor joined Vanguard AI as VP of Engineering",
        subject_name="Sarah Connor",
        subject_entity_id="ent_sarah",
        raw_predicate="joined",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_APPOINTED,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="Vanguard AI"),
    )

    fact_resign = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Sarah Connor stepped down as VP of Engineering",
        subject_name="Sarah Connor",
        subject_entity_id="ent_sarah",
        raw_predicate="stepped down",
        normalized_predicate=NormalizedPredicate.LEADERSHIP_DEPARTED,
        object_value=MolecularValue(value_type=ValueType.STRING, raw_value="VP of Engineering"),
    )

    eval_res = detector.evaluate(fact_join, fact_resign)
    assert eval_res.status == ContradictionStatus.POTENTIAL_CONTRADICTION
    assert eval_res.severity == "potential"
    assert "potential conflict (missing dates)" in eval_res.reason.lower()


def test_opposite_polarity_is_confirmed_contradiction(detector):
    """Affirming and explicitly negating the same predicate for the same location is a contradiction."""
    fact_pos = MolecularFact(
        parent_canonical_id="p1",
        atom_text="Acme Corp opened an office in Munich",
        subject_name="Acme Corp",
        subject_entity_id="ent_acme",
        raw_predicate="opened",
        normalized_predicate=NormalizedPredicate.FACILITY_OPENED,
        object_value=MolecularValue(value_type=ValueType.LOCATION, raw_value="office in Munich"),
        polarity=Polarity.POSITIVE,
        qualifiers={"location": "Munich"},
    )

    fact_neg = MolecularFact(
        parent_canonical_id="p2",
        atom_text="Acme Corp did not open an office in Munich",
        subject_name="Acme Corp",
        subject_entity_id="ent_acme",
        raw_predicate="did not open",
        normalized_predicate=NormalizedPredicate.FACILITY_OPENED,
        object_value=MolecularValue(value_type=ValueType.LOCATION, raw_value="office in Munich"),
        polarity=Polarity.NEGATIVE,
        qualifiers={"location": "Munich"},
    )

    eval_res = detector.evaluate(fact_pos, fact_neg)
    assert eval_res.status == ContradictionStatus.CONTRADICTION
    assert eval_res.severity == "confirmed"
    assert "opposite polarity" in eval_res.reason.lower()
