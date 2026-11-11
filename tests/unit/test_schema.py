from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from repo_paths import DATA_DIR, REPO_ROOT, get_db_path
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.connection import get_connection
from src.db.schema import (
    EXPECTED_DOMAIN_TABLES,
    FTS_VIRTUAL_TABLES,
    check_fts_integrity,
    init_schema,
    is_fts_table,
    load_schema_sql,
    rebuild_fts_indexes,
    verify_schema,
)
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator


def test_schema_sql_loads_cleanly():
    """Verify that schema.sql is found and contains non-empty SQL text."""
    sql = load_schema_sql()
    assert len(sql) > 0
    assert "CREATE TABLE IF NOT EXISTS prospects" in sql
    assert "migration_ledger" not in sql


def test_fresh_schema_initialization():
    """Verify that init_schema initializes all 36 domain tables and passes verification."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    
    init_schema(conn)
    result = verify_schema(conn)

    assert result["is_valid"] is True
    assert result["missing_tables"] == []
    assert len(EXPECTED_DOMAIN_TABLES) == 36

    # Verify no migration ledger exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migration_ledger';")
    assert cursor.fetchone() is None

    # Verify each expected table exists
    for table_name in EXPECTED_DOMAIN_TABLES:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        assert cursor.fetchone() is not None, f"Expected table {table_name} missing from sqlite_master"
    conn.close()


def test_schema_initialization_idempotent():
    """Verify that running init_schema multiple times succeeds without error."""
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    init_schema(conn)
    init_schema(conn)

    result = verify_schema(conn)
    assert result["is_valid"] is True
    conn.close()


def test_verify_schema_detects_missing_tables():
    """Verify that verify_schema correctly detects incomplete or empty databases."""
    conn = sqlite3.connect(":memory:")
    # Empty DB
    result = verify_schema(conn)
    assert result["is_valid"] is False
    assert len(result["missing_tables"]) == 36

    # Partially initialized DB (only 1 table)
    conn.execute("CREATE TABLE prospects (prospect_id TEXT PRIMARY KEY);")
    result = verify_schema(conn)
    assert result["is_valid"] is False
    assert "prospects" not in result["missing_tables"]
    assert len(result["missing_tables"]) == 35
    conn.close()


def test_schema_foreign_keys_and_compound_constraints():
    """Verify foreign key enforcement and compound FK on sufficiency_assessments."""
    conn = get_connection(":memory:")
    init_schema(conn)

    # 1. Foreign key enforcement on research_sessions -> prospects
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) "
            "VALUES ('s1', 'nonexistent_prospect', 'c1', '2026-11-08');"
        )

    # Insert valid prospect and session
    conn.execute(
        "INSERT INTO prospects (prospect_id, full_name, company_name, created_at) "
        "VALUES ('p1', 'Elena Rostova', 'Vanguard AI', '2026-11-08');"
    )
    conn.execute(
        "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) "
        "VALUES ('s1', 'p1', 'c1', '2026-11-08');"
    )

    # 2. Compound foreign key on sufficiency_assessments -> research_iterations(session_id, iteration_id)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO sufficiency_assessments ("
            "  assessment_id, session_id, iteration_id, decision, coverage_status, identity_status, "
            "  evidence_status, marginal_value_status, explanation, objective_details_json, remaining_gaps_json, "
            "  recommended_actions_json, dimension_details_json, assessed_at"
            ") VALUES ('a1', 's1', 'nonexistent_it', 'sufficient', 'good', 'good', 'good', 'good', 'exp', '[]', '[]', '[]', '{}', '2026-11-08');"
        )

    # Insert corresponding research_iterations record
    conn.execute(
        "INSERT INTO research_iterations (iteration_id, session_id, iteration_number, queries_executed, facts_extracted, started_at) "
        "VALUES ('it1', 's1', 1, 2, 5, '2026-11-08');"
    )
    # Now inserting sufficiency_assessments must succeed
    conn.execute(
        "INSERT INTO sufficiency_assessments ("
        "  assessment_id, session_id, iteration_id, decision, coverage_status, identity_status, "
        "  evidence_status, marginal_value_status, explanation, objective_details_json, remaining_gaps_json, "
        "  recommended_actions_json, dimension_details_json, assessed_at"
        ") VALUES ('a1', 's1', 'it1', 'sufficient', 'good', 'good', 'good', 'good', 'exp', '[]', '[]', '[]', '{}', '2026-11-08');"
    )

    # 3. Check constraints on conflicts severity
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO conflicts (conflict_id, session_id, description, status, severity, created_at) "
            "VALUES ('c1', 's1', 'Test conflict', 'open', 'invalid_severity', '2026-11-08');"
        )

    # Valid severity works
    conn.execute(
        "INSERT INTO conflicts (conflict_id, session_id, description, status, severity, created_at) "
        "VALUES ('c1', 's1', 'Test conflict', 'open', 'confirmed', '2026-11-08');"
    )

    conn.close()


def test_database_status_does_not_create_file(tmp_path: Path):
    """Verify get_database_status does not create an empty file if the file does not exist."""
    db_file = tmp_path / "nonexistent" / "test.db"
    assert not db_file.exists()

    store = StateStore(db_path=db_file)
    status = store.get_database_status()

    assert status["file_exists"] is False
    assert status["is_initialized"] is False
    assert status["is_schema_valid"] is False
    assert status["expected_tables_found"] == 0
    assert status["total_expected_tables"] == 36
    assert status["table_count"] == 0
    assert not db_file.exists()  # Crucial: file must NOT have been created


def test_db_path_resolution_and_cwd_independence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify get_db_path resolves default to DATA_DIR/open_inquiry.db and relative paths against REPO_ROOT."""
    # 1. Default path without env vars
    monkeypatch.delenv("OPEN_INQUIRY_DB_PATH", raising=False)
    monkeypatch.delenv("ZAMP_DB_PATH", raising=False)
    default_p = get_db_path()
    assert default_p == DATA_DIR / "open_inquiry.db"

    # 2. Relative OPEN_INQUIRY_DB_PATH should resolve relative to REPO_ROOT, regardless of CWD
    monkeypatch.setenv("OPEN_INQUIRY_DB_PATH", "data/custom.db")
    custom_p = get_db_path()
    assert custom_p == (REPO_ROOT / "data" / "custom.db").resolve()

    # Change working directory and verify it still resolves to REPO_ROOT
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert get_db_path() == (REPO_ROOT / "data" / "custom.db").resolve()
    finally:
        os.chdir(original_cwd)

    # 3. Absolute OPEN_INQUIRY_DB_PATH
    abs_path = (tmp_path / "absolute.db").resolve()
    monkeypatch.setenv("OPEN_INQUIRY_DB_PATH", str(abs_path))
    assert get_db_path() == abs_path


def test_orchestrator_auto_initialization(tmp_path: Path):
    """Verify ApplicationOrchestrator auto-initializes a fresh database file."""
    db_file = tmp_path / "auto_init" / "fresh.db"
    assert not db_file.exists()

    store = StateStore(db_path=db_file)
    orchestrator = ApplicationOrchestrator(state_store=store)

    prospect = ProspectContext(full_name="Alex Rivera", company_name="Novatech")
    product = ProductContext(
        product_name="Open Inquiry",
        description="Autonomous research engine",
        capabilities=["Research", "Signal Strategy"],
        value_proposition="AI research",
    )
    outreach = OutreachContext(
        objective="AI Partnership",
        sender_identity="Antigravity",
    )

    result = orchestrator.run(
        prospect=prospect,
        product=product,
        outreach=outreach,
        max_research_iterations=1,
    )

    assert result is not None
    assert db_file.exists()
    status = store.get_database_status()
    assert status["file_exists"] is True
    assert status["is_schema_valid"] is True
    assert status["expected_tables_found"] == 36


def test_orchestrator_incompatible_schema_preserves_database(tmp_path: Path):
    """Verify ApplicationOrchestrator stops on incompatible schema without deleting or overwriting data."""
    db_file = tmp_path / "incompatible.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE custom_user_data (id TEXT PRIMARY KEY, val TEXT);")
    conn.execute("INSERT INTO custom_user_data VALUES ('1', 'preserve_me');")
    conn.commit()
    conn.close()

    store = StateStore(db_path=db_file)
    orchestrator = ApplicationOrchestrator(state_store=store)

    prospect = ProspectContext(full_name="Alex Rivera", company_name="Novatech")
    product = ProductContext(
        product_name="Open Inquiry",
        description="Autonomous research engine",
        capabilities=["Research", "Signal Strategy"],
        value_proposition="AI research",
    )
    outreach = OutreachContext(
        objective="AI Partnership",
        sender_identity="Antigravity",
    )

    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.run(
            prospect=prospect,
            product=product,
            outreach=outreach,
            max_research_iterations=1,
        )

    assert "incompatible schema" in str(exc_info.value).lower()
    assert "manual reset policy" in str(exc_info.value).lower()

    # Verify existing table and data were NOT dropped or deleted
    conn2 = sqlite3.connect(str(db_file))
    cur = conn2.execute("SELECT val FROM custom_user_data WHERE id='1';")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "preserve_me"
    conn2.close()


def test_fts5_schema_elements_and_integrity():
    """Verify FTS5 virtual tables, sync triggers, is_fts_table predicate, and integrity checks."""
    conn = sqlite3.connect(":memory:")
    init_schema(conn)

    # Verify is_fts_table classification
    for v_table in FTS_VIRTUAL_TABLES:
        assert is_fts_table(v_table) is True
        assert is_fts_table(f"{v_table}_data") is True
        assert is_fts_table(f"{v_table}_idx") is True
        assert is_fts_table(f"{v_table}_config") is True
        assert is_fts_table(f"{v_table}_docsize") is True

    for d_table in EXPECTED_DOMAIN_TABLES:
        assert is_fts_table(d_table) is False

    # Verify virtual tables exist in sqlite_master
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('source_snapshots_fts', 'evidence_passages_fts');")
    found_virtual = {r[0] for r in cur.fetchall()}
    assert found_virtual == {"source_snapshots_fts", "evidence_passages_fts"}

    # Verify synchronization triggers exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%_fts%' OR name LIKE '%source_snapshots_%' OR name LIKE '%evidence_passages_%';")
    triggers = {r[0] for r in cur.fetchall()}
    expected_triggers = {
        "source_snapshots_ai",
        "source_snapshots_ad",
        "source_snapshots_au",
        "evidence_passages_ai",
        "evidence_passages_ad",
        "evidence_passages_au",
    }
    assert expected_triggers.issubset(triggers)

    # Verify rebuild and integrity checks pass
    rebuild_fts_indexes(conn)
    integrity = check_fts_integrity(conn)
    assert integrity == {
        "source_snapshots_fts": True,
        "evidence_passages_fts": True,
    }
    conn.close()
