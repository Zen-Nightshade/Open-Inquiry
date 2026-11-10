import json
import uuid
from datetime import date
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


def test_molecular_value_typing_and_dict_roundtrip():
    """Verify MolecularValue represents typed values and roundtrips cleanly."""
    val_currency = MolecularValue(
        value_type=ValueType.CURRENCY,
        raw_value="$50 million",
        normalized_value=50000000.0,
        unit="USD",
    )
    val_dict = val_currency.to_dict()
    assert val_dict["value_type"] == "currency"
    assert val_dict["normalized_value"] == 50000000.0
    assert val_dict["unit"] == "USD"

    rebuilt = MolecularValue.from_dict(val_dict)
    assert rebuilt.value_type == ValueType.CURRENCY
    assert rebuilt.normalized_value == 50000000.0
    assert rebuilt.unit == "USD"
    assert rebuilt.raw_value == "$50 million"


def test_molecular_value_unknown_fallback():
    """Unknown or unexpected types fall back safely to ValueType.UNKNOWN."""
    bad_data = {"value_type": "non_existent_type", "raw_value": "custom"}
    val = MolecularValue.from_dict(bad_data)
    assert val.value_type == ValueType.UNKNOWN
    assert val.raw_value == "custom"


def test_molecular_fact_instantiation_and_immutability():
    """Verify MolecularFact is frozen and references parent canonical fact."""
    parent_id = str(uuid.uuid4())
    val = MolecularValue(value_type=ValueType.LOCATION, raw_value="Austin, TX", normalized_value="Austin, TX")
    
    fact = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Vanguard AI opened an office in Austin, TX",
        subject_name="Vanguard AI",
        subject_entity_id="ent_vanguard",
        raw_predicate="opened an office in",
        normalized_predicate=NormalizedPredicate.FACILITY_OPENED,
        is_normalized=True,
        object_value=val,
        modality=Modality.DEFINITIVE,
        polarity=Polarity.POSITIVE,
        qualifiers={"location": "Austin, TX"},
        event_date=date(2025, 4, 1),
        is_decomposed=True,
        confidence=ConfidenceLevel.HIGH,
        verification_status=VerificationStatus.SUPPORTED,
    )

    assert fact.parent_canonical_id == parent_id
    assert fact.is_decomposed is True
    assert fact.normalized_predicate == NormalizedPredicate.FACILITY_OPENED
    assert fact.confidence == ConfidenceLevel.HIGH

    # Immutability check
    with pytest.raises(Exception):
        fact.atom_text = "Mutated text"  # type: ignore[misc]


def test_multiple_molecular_propositions_under_one_canonical_fact():
    """Verify one CanonicalFact can host multiple distinct MolecularFact propositions."""
    parent_id = str(uuid.uuid4())
    
    atom1 = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme Corp opened a facility in Hyderabad",
        subject_name="Acme Corp",
        raw_predicate="opened",
        normalized_predicate=NormalizedPredicate.FACILITY_OPENED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.LOCATION, raw_value="facility in Hyderabad"),
        modality=Modality.DEFINITIVE,
        is_decomposed=True,
    )

    atom2 = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Acme Corp plans to hire 500 engineers",
        subject_name="Acme Corp",
        raw_predicate="plans to hire",
        normalized_predicate=NormalizedPredicate.HIRING_PLANNED,
        is_normalized=True,
        object_value=MolecularValue(value_type=ValueType.NUMBER, raw_value="500 engineers", normalized_value=500, unit="engineers"),
        modality=Modality.PLANNED,
        is_decomposed=True,
    )

    canonical = CanonicalFact(
        fact_id=parent_id,
        claim_text="Acme Corp opened a facility in Hyderabad and plans to hire 500 engineers.",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[
            EvidenceRecord(
                source_id="src_news",
                verbatim_excerpt="Acme Corp opened a facility in Hyderabad and plans to hire 500 engineers.",
            )
        ],
        molecular_facts=(atom1, atom2),
    )

    assert len(canonical.molecular_facts) == 2
    assert canonical.molecular_facts[0].normalized_predicate == NormalizedPredicate.FACILITY_OPENED
    assert canonical.molecular_facts[0].modality == Modality.DEFINITIVE
    assert canonical.molecular_facts[1].normalized_predicate == NormalizedPredicate.HIRING_PLANNED
    assert canonical.molecular_facts[1].modality == Modality.PLANNED


def test_versioned_serialization_and_deserialization_roundtrip():
    """Verify versioned JSON serialization and defensive deserialization."""
    parent_id = "parent-123"
    atom = MolecularFact(
        parent_canonical_id=parent_id,
        atom_text="Vanguard AI raised $50 million in Series B",
        subject_name="Vanguard AI",
        raw_predicate="raised",
        normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
        is_normalized=True,
        object_value=MolecularValue(
            value_type=ValueType.CURRENCY,
            raw_value="$50 million",
            normalized_value=50000000.0,
            unit="USD",
        ),
        qualifiers={"round": "Series B"},
        event_date=date(2025, 6, 15),
        is_decomposed=False,
    )

    serialized = serialize_molecular_facts([atom])
    parsed_json = json.loads(serialized)
    assert parsed_json["schema_version"] == MOLECULAR_SCHEMA_VERSION
    assert len(parsed_json["facts"]) == 1

    deserialized = deserialize_molecular_facts(serialized)
    assert len(deserialized) == 1
    reconstructed = deserialized[0]
    assert reconstructed.parent_canonical_id == parent_id
    assert reconstructed.normalized_predicate == NormalizedPredicate.FUNDING_RAISED
    assert reconstructed.object_value.normalized_value == 50000000.0
    assert reconstructed.object_value.unit == "USD"
    assert reconstructed.event_date == date(2025, 6, 15)


def test_deserialization_handles_legacy_nulls_and_malformed_input():
    """Verify deserialize_molecular_facts safely handles None, empty, and bad data."""
    assert deserialize_molecular_facts(None) == ()
    assert deserialize_molecular_facts("") == ()
    assert deserialize_molecular_facts("   ") == ()
    assert deserialize_molecular_facts("not valid json") == ()
    assert deserialize_molecular_facts("[]") == ()  # Not an envelope dict
    assert deserialize_molecular_facts('{"schema_version": 1, "facts": "not a list"}') == ()
    
    # Partial malformed item in facts list is safely skipped
    mixed_payload = json.dumps({
        "schema_version": 1,
        "facts": [
            "bad string",
            {
                "parent_canonical_id": "p1",
                "atom_text": "Valid atom",
                "subject_name": "Acme",
                "raw_predicate": "opened",
                "object_value": {"raw_value": "office"},
            }
        ]
    })
    results = deserialize_molecular_facts(mixed_payload)
    assert len(results) == 1
    assert results[0].atom_text == "Valid atom"
