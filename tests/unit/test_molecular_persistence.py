from __future__ import annotations

import sqlite3
import tempfile
import uuid

import pytest

from src.db.connection import get_connection
from src.db.repositories.assertion_repo import AssertionRepository
from src.db.schema import EXPECTED_DOMAIN_TABLES, init_schema, verify_schema
from src.researcher.molecular_models import (
    Modality,
    MolecularFact,
    MolecularValue,
    NormalizedPredicate,
    Polarity,
    ValueType,
    deserialize_molecular_facts,
    serialize_molecular_facts,
)


def _setup_session(conn: sqlite3.Connection, session_id: str) -> None:
    now = "2026-11-10T00:00:00Z"
    prospect_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES (?, ?, ?, ?);",
        (prospect_id, "Jane Doe", "Acme Corp", now),
    )
    conn.execute(
        "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at, status) VALUES (?, ?, ?, ?, ?);",
        (session_id, prospect_id, "ctx_test", now, "in_progress"),
    )
    conn.commit()


def test_schema_migration_idempotent_and_table_count_invariant():
    """Verify that molecular data migration preserves exactly 36 domain tables and is idempotent."""
    assert len(EXPECTED_DOMAIN_TABLES) == 36
    assert "canonical_facts" in EXPECTED_DOMAIN_TABLES

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")

    # Run init_schema three times to verify idempotence
    init_schema(conn)
    init_schema(conn)
    init_schema(conn)

    verification = verify_schema(conn)
    assert verification["is_valid"] is True
    assert verification["missing_tables"] == []

    # Check table_info for canonical_facts
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(canonical_facts);")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    assert "molecular_data_json" in columns
    assert columns["molecular_data_json"] == "TEXT"

    conn.close()


def test_molecular_facts_roundtrip_persistence():
    """Verify roundtrip serialization, database persistence, and deserialization of molecular facts."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)

        session_id = str(uuid.uuid4())
        fact_id = str(uuid.uuid4())
        _setup_session(conn, session_id)

        repo = AssertionRepository(conn)

        # Create molecular facts
        mol_value_1 = MolecularValue(
            raw_value="$25M",
            value_type=ValueType.CURRENCY,
            normalized_value=25000000.0,
            unit="USD",
        )
        mol_fact_1 = MolecularFact(
            parent_canonical_id=fact_id,
            atom_text="Acme Corp raised $25M in Series B in 2024.",
            subject_name="Acme Corp",
            raw_predicate="raised",
            normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
            object_value=mol_value_1,
            modality=Modality.DEFINITIVE,
            polarity=Polarity.POSITIVE,
            is_decomposed=False,
            qualifiers={"round": "Series B"},
        )

        serialized = serialize_molecular_facts([mol_fact_1])
        assert serialized is not None

        # Persist fact with molecular data
        created = repo.create_canonical_fact(
            fact_id=fact_id,
            session_id=session_id,
            claim_text="Acme Corp raised $25M in Series B in 2024.",
            fact_category="funding",
            primary_event_date="2024",
            composite_confidence="high",
            molecular_data_json=serialized,
        )

        assert created["fact_id"] == fact_id
        assert created["molecular_data_json"] == serialized

        # Retrieve and deserialize
        fetched = repo.get_canonical_fact(fact_id)
        assert fetched is not None
        assert fetched["molecular_data_json"] is not None

        deserialized = deserialize_molecular_facts(fetched["molecular_data_json"])
        assert len(deserialized) == 1
        retrieved_mol = deserialized[0]

        assert retrieved_mol.parent_canonical_id == fact_id
        assert retrieved_mol.subject_name == "Acme Corp"
        assert retrieved_mol.raw_predicate == "raised"
        assert retrieved_mol.normalized_predicate == NormalizedPredicate.FUNDING_RAISED
        assert retrieved_mol.object_value.value_type == ValueType.CURRENCY
        assert retrieved_mol.object_value.normalized_value == 25000000.0
        assert retrieved_mol.object_value.unit == "USD"
        assert retrieved_mol.modality == Modality.DEFINITIVE
        assert retrieved_mol.polarity == Polarity.POSITIVE
        assert retrieved_mol.qualifiers == {"round": "Series B"}

        conn.close()


def test_legacy_canonical_fact_compatibility_and_upsert_coalesce():
    """Verify backward compatibility with null molecular data and non-destructive upserts."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)

        session_id = str(uuid.uuid4())
        fact_id = str(uuid.uuid4())
        _setup_session(conn, session_id)

        repo = AssertionRepository(conn)

        # Insert legacy fact without molecular data
        legacy_fact = repo.create_canonical_fact(
            fact_id=fact_id,
            session_id=session_id,
            claim_text="Acme Corp announced product Alpha.",
            fact_category="company_activity",
            composite_confidence="medium",
            molecular_data_json=None,
        )
        assert legacy_fact["molecular_data_json"] is None

        # Deserializing None returns empty tuple
        assert deserialize_molecular_facts(legacy_fact["molecular_data_json"]) == ()

        # Upsert with molecular data
        mol_fact = MolecularFact(
            parent_canonical_id=fact_id,
            atom_text="Acme Corp launched product Alpha.",
            subject_name="Acme Corp",
            raw_predicate="launched",
            normalized_predicate=NormalizedPredicate.PRODUCT_LAUNCHED,
            object_value=MolecularValue(raw_value="Alpha", value_type=ValueType.STRING),
        )
        serialized = serialize_molecular_facts([mol_fact])

        updated = repo.create_canonical_fact(
            fact_id=fact_id,
            session_id=session_id,
            claim_text="Acme Corp announced product Alpha.",
            fact_category="company_activity",
            composite_confidence="high",
            molecular_data_json=serialized,
        )
        assert updated["composite_confidence"] == "high"
        assert updated["molecular_data_json"] == serialized

        # Upsert with None does NOT erase existing molecular_data_json due to coalesce
        re_updated = repo.create_canonical_fact(
            fact_id=fact_id,
            session_id=session_id,
            claim_text="Acme Corp announced product Alpha.",
            fact_category="company_activity",
            composite_confidence="high",
            molecular_data_json=None,
        )
        assert re_updated["molecular_data_json"] == serialized

        conn.close()


def test_conflict_persistence_with_molecular_contradiction():
    """Verify that conflicts identified between molecular facts can be recorded and queried."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)

        session_id = str(uuid.uuid4())
        fact1_id = str(uuid.uuid4())
        fact2_id = str(uuid.uuid4())
        conflict_id = str(uuid.uuid4())
        _setup_session(conn, session_id)

        repo = AssertionRepository(conn)

        # Fact 1: Raised $10M Series A
        mol1 = MolecularFact(
            parent_canonical_id=fact1_id,
            atom_text="Acme Corp raised $10M Series A.",
            subject_name="Acme Corp",
            raw_predicate="raised",
            normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
            object_value=MolecularValue(
                raw_value="$10M",
                value_type=ValueType.CURRENCY,
                normalized_value=10000000.0,
            ),
            qualifiers={"round": "Series A"},
        )
        repo.create_canonical_fact(
            fact_id=fact1_id,
            session_id=session_id,
            claim_text="Acme Corp raised $10M Series A.",
            molecular_data_json=serialize_molecular_facts([mol1]),
        )

        # Fact 2: Raised $20M Series A
        mol2 = MolecularFact(
            parent_canonical_id=fact2_id,
            atom_text="Acme Corp raised $20M Series A.",
            subject_name="Acme Corp",
            raw_predicate="raised",
            normalized_predicate=NormalizedPredicate.FUNDING_RAISED,
            object_value=MolecularValue(
                raw_value="$20M",
                value_type=ValueType.CURRENCY,
                normalized_value=20000000.0,
            ),
            qualifiers={"round": "Series A"},
        )
        repo.create_canonical_fact(
            fact_id=fact2_id,
            session_id=session_id,
            claim_text="Acme Corp raised $20M Series A.",
            molecular_data_json=serialize_molecular_facts([mol2]),
        )

        # Record conflict
        conflict_desc = "Molecular contradiction on FUNDING_RAISED for Acme Corp: $10000000.0 vs $20000000.0"
        conflict = repo.create_conflict(
            conflict_id=conflict_id,
            session_id=session_id,
            description=conflict_desc,
            severity="confirmed",
            status="open",
        )
        assert conflict["conflict_id"] == conflict_id

        # Link both facts to the conflict
        repo.link_conflict_fact(conflict_id, fact1_id)
        repo.link_conflict_fact(conflict_id, fact2_id)

        # Query conflict facts
        linked_facts = repo.get_conflict_facts(conflict_id)
        assert len(linked_facts) == 2
        fact_ids = {f["fact_id"] for f in linked_facts}
        assert fact1_id in fact_ids
        assert fact2_id in fact_ids

        # Query session conflicts
        conflicts = repo.get_conflicts_for_session(session_id)
        assert len(conflicts) == 1
        assert conflicts[0]["description"] == conflict_desc

        conn.close()
