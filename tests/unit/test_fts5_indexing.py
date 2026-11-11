from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.db.connection import get_connection
from src.db.schema import (
    EXPECTED_DOMAIN_TABLES,
    FTS_VIRTUAL_TABLES,
    check_fts_integrity,
    init_schema,
    is_fts_table,
    rebuild_fts_indexes,
    verify_schema,
)
from src.db.state_store import StateStore
from src.researcher.retrieval import HistoricalRetrievalService


def test_source_snapshot_indexing_triggers():
    """Verify AFTER INSERT, UPDATE, and DELETE triggers keep source_snapshots_fts in sync."""
    conn = get_connection(":memory:")
    init_schema(conn)

    # 1. Setup prerequisite records
    conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Alice', 'Acme', '2026-11-10');")
    conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s1', 'p1', 'c1', '2026-11-10');")
    conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://acme.com', 'acme.com', '2026-11-10');")

    # 2. INSERT into source_snapshots
    conn.execute(
        "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
        "VALUES ('snap1', 'src1', 's1', 'Acme Robotics Announces Autonomous Forklift', 'Acme Robotics has launched an autonomous forklift with lidar guidance.', 'success', '2026-11-10');"
    )
    conn.commit()

    # Query FTS
    cur = conn.execute("SELECT rowid, title FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'forklift';")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert "Forklift" in rows[0][1]

    # 3. UPDATE in source_snapshots
    conn.execute(
        "UPDATE source_snapshots SET title = 'Acme Robotics Launches Autonomous Drone', raw_text = 'Acme Robotics has unveiled an aerial drone.' WHERE snapshot_id = 'snap1';"
    )
    conn.commit()

    # Old keyword must not match
    cur = conn.execute("SELECT rowid FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'forklift';")
    assert len(cur.fetchall()) == 0

    # New keyword must match
    cur = conn.execute("SELECT rowid, title FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'drone';")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert "Drone" in rows[0][1]

    # 4. DELETE from source_snapshots
    conn.execute("DELETE FROM source_snapshots WHERE snapshot_id = 'snap1';")
    conn.commit()

    cur = conn.execute("SELECT rowid FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'drone';")
    assert len(cur.fetchall()) == 0
    conn.close()


def test_evidence_passage_indexing_triggers():
    """Verify AFTER INSERT, UPDATE, and DELETE triggers keep evidence_passages_fts in sync."""
    conn = get_connection(":memory:")
    init_schema(conn)

    conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Bob', 'Beta', '2026-11-10');")
    conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s1', 'p1', 'c1', '2026-11-10');")
    conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://beta.com', 'beta.com', '2026-11-10');")
    conn.execute(
        "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
        "VALUES ('snap1', 'src1', 's1', 'Beta Cloud', 'Beta Cloud announcement', 'success', '2026-11-10');"
    )

    # 1. INSERT into evidence_passages
    conn.execute(
        "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
        "VALUES ('ev1', 'snap1', 'src1', 'Our multi-region database delivers sub-millisecond latency.', '2026-11-10');"
    )
    conn.commit()

    cur = conn.execute("SELECT rowid FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'latency';")
    rows = cur.fetchall()
    assert len(rows) == 1

    # 2. UPDATE verbatim_excerpt
    conn.execute("UPDATE evidence_passages SET verbatim_excerpt = 'High throughput pipeline processes millions of events.' WHERE evidence_id = 'ev1';")
    conn.commit()

    # Old term gone
    cur = conn.execute("SELECT rowid FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'latency';")
    assert len(cur.fetchall()) == 0

    # New term present
    cur = conn.execute("SELECT rowid FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'throughput';")
    assert len(cur.fetchall()) == 1

    # 3. DELETE evidence passage
    conn.execute("DELETE FROM evidence_passages WHERE evidence_id = 'ev1';")
    conn.commit()

    cur = conn.execute("SELECT rowid FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'throughput';")
    assert len(cur.fetchall()) == 0
    conn.close()


def test_transaction_rollback_reverts_fts_index():
    """Verify that database transaction rollback cleanly reverts FTS5 virtual table updates."""
    conn = get_connection(":memory:")
    init_schema(conn)

    conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Carol', 'Gamma', '2026-11-10');")
    conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s1', 'p1', 'c1', '2026-11-10');")
    conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://gamma.com', 'gamma.com', '2026-11-10');")
    conn.commit()

    # Begin transaction, insert snapshot, and roll back
    try:
        conn.execute("BEGIN TRANSACTION;")
        conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_abort', 'src1', 's1', 'Aborted Snapshot Title', 'Aborted secret text', 'success', '2026-11-10');"
        )
        # Verify it exists inside the transaction
        cur = conn.execute("SELECT rowid FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'secret';")
        assert len(cur.fetchall()) == 1
        # Explicit rollback
        conn.rollback()
    except Exception:
        conn.rollback()

    # Outside the rolled back transaction, FTS must be clean
    cur = conn.execute("SELECT rowid FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'secret';")
    assert len(cur.fetchall()) == 0
    conn.close()


def test_rebuild_fts_indexes_backfills_preexisting_data():
    """Verify rebuild_fts_indexes backfills data into FTS index."""
    conn = get_connection(":memory:")
    init_schema(conn)

    conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Dave', 'Delta', '2026-11-10');")
    conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s1', 'p1', 'c1', '2026-11-10');")
    conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://delta.com', 'delta.com', '2026-11-10');")
    conn.execute(
        "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
        "VALUES ('snap1', 'src1', 's1', 'Delta Machine Learning', 'Machine learning distributed cluster training', 'success', '2026-11-10');"
    )
    conn.execute(
        "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
        "VALUES ('ev1', 'snap1', 'src1', 'Distributed cluster training scales linearly across 128 GPUs.', '2026-11-10');"
    )
    conn.commit()

    # Rebuild indexes
    rebuild_fts_indexes(conn)

    cur1 = conn.execute("SELECT rowid FROM source_snapshots_fts WHERE source_snapshots_fts MATCH 'cluster';")
    assert len(cur1.fetchall()) == 1

    cur2 = conn.execute("SELECT rowid FROM evidence_passages_fts WHERE evidence_passages_fts MATCH 'GPUs';")
    assert len(cur2.fetchall()) == 1

    # Verify integrity
    integrity = check_fts_integrity(conn)
    assert integrity["source_snapshots_fts"] is True
    assert integrity["evidence_passages_fts"] is True
    conn.close()


def test_service_administrative_rebuild_and_integrity(tmp_path: Path):
    """Verify HistoricalRetrievalService administrative rebuild and integrity check methods."""
    db_file = tmp_path / "test_admin.db"
    store = StateStore(db_file)
    store.initialize()

    service = HistoricalRetrievalService(state_store=store)
    # Rebuild must run idempotently
    service.rebuild_indexes()

    # Integrity must return clean
    status = service.verify_integrity()
    assert status == {
        "source_snapshots_fts": True,
        "evidence_passages_fts": True,
    }
