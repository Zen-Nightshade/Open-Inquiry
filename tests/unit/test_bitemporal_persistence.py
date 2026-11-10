from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import date

import pytest

from src.db.connection import get_connection
from src.db.repositories.assertion_repo import AssertionRepository
from src.db.schema import EXPECTED_DOMAIN_TABLES, init_schema, verify_schema
from src.researcher.models import CanonicalFact, FactCategory
from src.researcher.molecular_models import (
    Modality,
    MolecularFact,
    MolecularValue,
    NormalizedPredicate,
    Polarity,
    ValueType,
)
from src.researcher.temporal import TemporalKind


def _setup_session(conn: sqlite3.Connection, session_id: str, prospect_id: str = "pros_test") -> None:
    now = "2026-11-10T00:00:00Z"
    conn.execute(
        "INSERT OR IGNORE INTO prospects (prospect_id, full_name, company_name, created_at) VALUES (?, ?, ?, ?);",
        (prospect_id, "Jane Doe", "Acme Corp", now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO research_sessions (session_id, prospect_id, context_id, started_at, status) VALUES (?, ?, ?, ?, ?);",
        (session_id, prospect_id, "ctx_test", now, "in_progress"),
    )
    conn.commit()


def test_schema_invariant_and_bitemporal_columns():
    """Verify that bitemporal support retains exactly 36 physical domain tables and adds indexed columns."""
    assert len(EXPECTED_DOMAIN_TABLES) == 36
    assert "canonical_facts" in EXPECTED_DOMAIN_TABLES

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")

    # Run init_schema multiple times to verify idempotence
    init_schema(conn)
    init_schema(conn)

    verification = verify_schema(conn)
    assert verification["is_valid"] is True
    assert verification["missing_tables"] == []

    # Check columns in canonical_facts
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(canonical_facts);")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    assert "valid_from" in columns
    assert "valid_to" in columns
    assert "observed_at" in columns
    assert "recorded_at" in columns

    # Check temporal index exists
    cursor.execute("PRAGMA index_list(canonical_facts);")
    indexes = {row[1] for row in cursor.fetchall()}
    assert "idx_canonical_facts_temporal" in indexes

    conn.close()


def test_bitemporal_backfill_for_legacy_rows():
    """Verify legacy rows without bitemporal columns are backfilled from primary_event_date and created_at."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    init_schema(conn)

    session_id = str(uuid.uuid4())
    _setup_session(conn, session_id)

    fact_id = str(uuid.uuid4())
    created = "2024-01-15T10:00:00Z"
    event_d = "2023-11-01"

    conn.execute(
        """
        INSERT INTO canonical_facts (
            fact_id, session_id, claim_text, fact_category, primary_event_date,
            composite_confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (fact_id, session_id, "Legacy assertion", "financial", event_d, "high", created),
    )
    conn.commit()

    # Re-run init_schema to trigger the idempotent migration & backfill
    init_schema(conn)

    cursor = conn.cursor()
    cursor.execute(
        "SELECT valid_from, valid_to, observed_at, recorded_at FROM canonical_facts WHERE fact_id = ?;",
        (fact_id,),
    )
    row = cursor.fetchone()
    assert row[0] == event_d  # valid_from backfilled from primary_event_date
    assert row[1] is None     # valid_to remains None (no invented expiration)
    assert row[2] == created  # observed_at backfilled from created_at
    assert row[3] == created  # recorded_at backfilled from created_at

    conn.close()


def test_bitemporal_persistence_and_historical_preservation():
    """Preserve sequential assertions without overwriting: Source A (500 in 2022) and Source B (800 in 2024)."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)

        session_id = str(uuid.uuid4())
        prospect_id = "pros_acme"
        _setup_session(conn, session_id, prospect_id)

        # Create entity in entities table
        now = "2026-11-10T00:00:00Z"
        conn.execute(
            "INSERT INTO entities (entity_id, entity_type, raw_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?);",
            (prospect_id, "organization", "Acme Corp", now, now),
        )
        conn.commit()

        repo = AssertionRepository(conn)

        # 1. Persist Fact 1 (Source A in 2022)
        fact_id_1 = str(uuid.uuid4())
        repo.create_canonical_fact(
            fact_id=fact_id_1,
            session_id=session_id,
            claim_text="Acme employs 500 people",
            fact_category="organization_change",
            primary_event_date="2022-01-01",
            composite_confidence="high",
            valid_from="2022",
            valid_to="2023-12-31",
            observed_at="2022-01-05T09:00:00Z",
            recorded_at="2022-01-05T09:15:00Z",
        )

        # 2. Persist Fact 2 (Source B in 2024)
        fact_id_2 = str(uuid.uuid4())
        repo.create_canonical_fact(
            fact_id=fact_id_2,
            session_id=session_id,
            claim_text="Acme employs 800 people",
            fact_category="organization_change",
            primary_event_date="2024-01-01",
            composite_confidence="high",
            valid_from="2024",
            valid_to=None,
            observed_at="2024-02-10T14:00:00Z",
            recorded_at="2024-02-10T14:05:00Z",
        )

        # Link both facts to prospect entity via fact_entity_links
        conn.execute("INSERT INTO fact_entity_links (fact_id, entity_id) VALUES (?, ?);", (fact_id_1, prospect_id))
        conn.execute("INSERT INTO fact_entity_links (fact_id, entity_id) VALUES (?, ?);", (fact_id_2, prospect_id))
        conn.commit()

        # Both records must be preserved
        facts = repo.get_facts_for_session(session_id)
        assert len(facts) == 2
        f1 = next(f for f in facts if f["fact_id"] == fact_id_1)
        assert f1["valid_from"] == "2022"
        assert f1["valid_to"] == "2023-12-31"
        assert f1["observed_at"] == "2022-01-05T09:00:00Z"
        assert f1["recorded_at"] == "2022-01-05T09:15:00Z"

        f2 = next(f for f in facts if f["fact_id"] == fact_id_2)
        assert f2["valid_from"] == "2024"
        assert f2["valid_to"] is None
        assert f2["observed_at"] == "2024-02-10T14:00:00Z"
        assert f2["recorded_at"] == "2024-02-10T14:05:00Z"

        # 3. Query current assertions (as of 2024)
        current = repo.get_current_assertions(prospect_id, as_of_valid_time="2024-06-01")
        assert len(current) == 1
        assert current[0]["fact_id"] == fact_id_2

        # 4. Point-in-time historical reconstruction: as of 2023
        as_of_2023 = repo.get_assertions_as_of(prospect_id, valid_time="2023-06-01")
        assert len(as_of_2023) == 1
        assert as_of_2023[0]["fact_id"] == fact_id_1

        # 5. Point-in-time historical reconstruction: recorded cutoff
        # Before 2024 fact was recorded
        as_of_early_recorded = repo.get_assertions_as_of(
            prospect_id,
            recorded_cutoff="2023-01-01T00:00:00Z"
        )
        assert len(as_of_early_recorded) == 1
        assert as_of_early_recorded[0]["fact_id"] == fact_id_1

        # 6. Assertion history
        history = repo.get_assertion_history(prospect_id)
        assert len(history) == 2
        # Ordered by valid_from ASC
        assert history[0]["fact_id"] == fact_id_1
        assert history[1]["fact_id"] == fact_id_2

        conn.close()
