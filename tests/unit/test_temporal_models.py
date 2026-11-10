from __future__ import annotations

from datetime import date, datetime, timezone
import uuid
import pytest

from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    FactCategory,
    VerificationStatus,
)
from src.researcher.molecular_models import (
    MOLECULAR_SCHEMA_VERSION,
    Modality,
    MolecularFact,
    MolecularValue,
    NormalizedPredicate,
    Polarity,
    ValueType,
    deserialize_molecular_facts,
    serialize_molecular_facts,
)
from src.researcher.temporal import (
    DatePrecision,
    TemporalKind,
    determine_temporal_kind,
    intervals_overlap,
    is_temporally_sequential,
    normalize_to_comparable_bounds,
    parse_partial_date,
)


def test_parse_partial_date_year_month_day_and_timestamps():
    """Verify partial date parser preserves explicit granularity without synthetic day fabrication."""
    # Year-only
    val, prec = parse_partial_date("2024")
    assert val == "2024"
    assert prec == DatePrecision.YEAR

    # Year-month
    val, prec = parse_partial_date("2024-03")
    assert val == "2024-03"
    assert prec == DatePrecision.MONTH

    # Year-month-day
    val, prec = parse_partial_date("2024-03-15")
    assert val == "2024-03-15"
    assert prec == DatePrecision.DAY

    # ISO timestamp
    val, prec = parse_partial_date("2024-03-15T14:30:00Z")
    assert val == "2024-03-15T14:30:00Z"
    assert prec == DatePrecision.TIMESTAMP

    # Python date and datetime objects
    d = date(2023, 11, 5)
    val, prec = parse_partial_date(d)
    assert val == "2023-11-05"
    assert prec == DatePrecision.DAY

    dt = datetime(2023, 11, 5, 8, 30, tzinfo=timezone.utc)
    val, prec = parse_partial_date(dt)
    assert "2023-11-05" in val
    assert prec == DatePrecision.TIMESTAMP


def test_parse_partial_date_textual_patterns():
    """Verify textual date expressions like 'March 2024' are parsed correctly."""
    val, prec = parse_partial_date("March 2024")
    assert val == "2024-03"
    assert prec == DatePrecision.MONTH

    val, prec = parse_partial_date("in 2022")
    assert val == "2022"
    assert prec == DatePrecision.YEAR


def test_parse_partial_date_unknown_never_fabricates():
    """Unknown or missing dates must remain unknown and never substitute current time."""
    for empty_val in [None, "", "   ", "unknown", "None", "n/a", "N/A", "null"]:
        val, prec = parse_partial_date(empty_val)
        assert val is None
        assert prec == DatePrecision.UNKNOWN


def test_determine_temporal_kind():
    """Distinguish point-in-time lifecycle events from durative states."""
    # Events
    assert determine_temporal_kind("facility_opened") == TemporalKind.EVENT
    assert determine_temporal_kind("facility_closed") == TemporalKind.EVENT
    assert determine_temporal_kind("funding_raised") == TemporalKind.EVENT
    assert determine_temporal_kind("leadership_appointed") == TemporalKind.EVENT
    assert determine_temporal_kind("leadership_departed") == TemporalKind.EVENT
    assert determine_temporal_kind("product_launched") == TemporalKind.EVENT

    # States
    assert determine_temporal_kind("employee_count") == TemporalKind.STATE
    assert determine_temporal_kind("headquartered_in") == TemporalKind.STATE
    assert determine_temporal_kind("operating_in") == TemporalKind.STATE
    assert determine_temporal_kind("hiring_planned") == TemporalKind.STATE


def test_normalize_to_comparable_bounds():
    """Ensure comparable bounds expand safely for mathematical comparison without altering raw strings."""
    assert normalize_to_comparable_bounds("2024") == ("2024-01-01", "2024-12-31")
    assert normalize_to_comparable_bounds("2024-02") == ("2024-02-01", "2024-02-29")  # 2024 is leap year
    assert normalize_to_comparable_bounds("2023-02") == ("2023-02-01", "2023-02-28")
    assert normalize_to_comparable_bounds("2024-03-15") == ("2024-03-15", "2024-03-15")
    assert normalize_to_comparable_bounds(None) == (None, None)


def test_intervals_overlap_and_sequentiality():
    """Verify interval overlap and temporal sequencing checks."""
    # Overlapping: 2024 year encompasses 2024-03
    overlap, certain = intervals_overlap("2024", None, "2024-03", None)
    assert overlap is True
    assert certain is True

    # Overlapping: same day
    overlap, certain = intervals_overlap("2024-05-01", None, "2024-05-01", None)
    assert overlap is True
    assert certain is True

    # Non-overlapping: 2022 vs 2024
    overlap, certain = intervals_overlap("2022", None, "2024", None)
    assert overlap is False
    assert certain is True

    # Sequential: 2022 is strictly before 2024
    assert is_temporally_sequential("2022", "2024") is True
    assert is_temporally_sequential("2024-01", "2024-06") is True
    assert is_temporally_sequential("2024", "2024-03") is False

    # Missing dates: cannot establish overlap with certainty
    overlap, certain = intervals_overlap(None, None, "2024", None)
    assert overlap is False
    assert certain is False
    assert is_temporally_sequential(None, "2024") is False


def test_molecular_fact_bitemporal_fields_and_evidence_provenance():
    """Verify MolecularFact captures bitemporal fields and distinguishes inherited vs independent evidence."""
    parent_id = str(uuid.uuid4())
    val = MolecularValue(value_type=ValueType.NUMBER, raw_value="500", normalized_value=500)

    fact = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme Corp employs 500 staff",
        subject_name="Acme Corp",
        raw_predicate="employs",
        normalized_predicate=NormalizedPredicate.EMPLOYEE_COUNT,
        is_normalized=True,
        object_value=val,
        temporal_kind=TemporalKind.STATE,
        valid_from="2022",
        valid_to="2023-12-31",
        observed_at="2024-01-10T12:00:00Z",
        recorded_at="2024-01-10T12:05:00Z",
        evidence_ids=("ev_src_1", "ev_src_2"),
        inherited_evidence=False,
    )

    assert fact.temporal_kind == TemporalKind.STATE
    assert fact.valid_from == "2022"
    assert fact.valid_to == "2023-12-31"
    assert fact.observed_at == "2024-01-10T12:00:00Z"
    assert fact.recorded_at == "2024-01-10T12:05:00Z"
    assert fact.evidence_ids == ("ev_src_1", "ev_src_2")
    assert fact.inherited_evidence is False

    # Roundtrip to/from dict
    d = fact.to_dict()
    assert d["valid_from"] == "2022"
    assert d["valid_to"] == "2023-12-31"
    assert d["observed_at"] == "2024-01-10T12:00:00Z"
    assert d["recorded_at"] == "2024-01-10T12:05:00Z"
    assert d["temporal_kind"] == "state"
    assert d["evidence_ids"] == ["ev_src_1", "ev_src_2"]
    assert d["inherited_evidence"] is False

    reconstructed = MolecularFact.from_dict(d)
    assert reconstructed.valid_from == "2022"
    assert reconstructed.temporal_kind == TemporalKind.STATE
    assert reconstructed.evidence_ids == ("ev_src_1", "ev_src_2")
    assert reconstructed.inherited_evidence is False


def test_molecular_fact_v1_backward_compatibility():
    """Verify v1 payload without bitemporal fields deserializes cleanly with defaults."""
    v1_dict = {
        "atom_text": "Legacy atom",
        "subject_name": "Legacy Subject",
        "raw_predicate": "opened",
        "normalized_predicate": "facility_opened",
        "is_normalized": True,
        "object_value": {"value_type": "text", "raw_value": "Austin"},
        "modality": "definitive",
        "polarity": "positive",
    }
    fact = MolecularFact.from_dict(v1_dict)
    assert fact.atom_text == "Legacy atom"
    assert fact.valid_from is None
    assert fact.valid_to is None
    assert fact.temporal_kind == TemporalKind.EVENT
    assert fact.inherited_evidence is True
    assert fact.evidence_ids == ()


def test_canonical_fact_typed_molecular_facts_collection():
    """Verify CanonicalFact.molecular_facts is typed as Tuple[MolecularFact, ...] and accepts bitemporal fields."""
    parent_id = str(uuid.uuid4())
    mol = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme raised $10M",
        subject_name="Acme",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.CURRENCY, raw_value="$10M", normalized_value=10000000.0),
        valid_from="2024-05",
        temporal_kind=TemporalKind.EVENT,
    )

    ev = EvidenceRecord(
        source_id="src_1",
        source_url="https://example.com/raise",
        verbatim_excerpt="Acme raised $10M",
    )
    cf = CanonicalFact(
        fact_id=parent_id,
        claim_text="Acme raised $10M in May 2024",
        fact_category=FactCategory.FUNDING,
        primary_event_date=date(2024, 5, 1),
        valid_from="2024-05",
        valid_to=None,
        observed_at="2024-05-02T10:00:00Z",
        recorded_at="2024-05-02T10:01:00Z",
        evidence_records=[ev],
        molecular_facts=(mol,),
    )

    assert isinstance(cf.molecular_facts, tuple)
    assert len(cf.molecular_facts) == 1
    assert isinstance(cf.molecular_facts[0], MolecularFact)
    assert cf.valid_from == "2024-05"
    assert cf.observed_at == "2024-05-02T10:00:00Z"
    assert cf.recorded_at == "2024-05-02T10:01:00Z"
