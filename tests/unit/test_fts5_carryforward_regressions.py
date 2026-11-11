from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest

from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.researcher.retrieval import (
    HistoricalRetrievalService,
    RetrievalScope,
    RetrievedRecordType,
)


@pytest.fixture
def clean_store(tmp_path: Path) -> StateStore:
    db_path = str(tmp_path / "test_regressions.db")
    store = StateStore(db_path)
    with store.uow() as uow:
        init_schema(uow.conn)
    return store


def test_no_fabricated_run_attribution_for_multi_run_session(clean_store: StateStore):
    """Verify that multiple runs sharing a research session do not fabricate run_id attribution."""
    with clean_store.uow() as uow:
        # Create prospect and session
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_multi', 'Multi Exec', 'Multi Corp', '2026-01-01');")
        uow.conn.execute(
            "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at, status) "
            "VALUES ('sess_shared', 'p_multi', 'ctx_multi', '2026-01-01', 'completed');"
        )
        # Create two runs referencing the same session
        uow.conn.execute(
            "INSERT INTO application_runs (run_id, prospect_id, context_id, research_session_id, status, prospect_snapshot_json, product_snapshot_json, outreach_snapshot_json, stage_statuses_json, started_at) "
            "VALUES ('run_earlier', 'p_multi', NULL, 'sess_shared', 'completed', '{}', '{}', '{}', '{}', '2026-01-01T10:00:00');"
        )
        uow.conn.execute(
            "INSERT INTO application_runs (run_id, prospect_id, context_id, research_session_id, status, prospect_snapshot_json, product_snapshot_json, outreach_snapshot_json, stage_statuses_json, started_at) "
            "VALUES ('run_later', 'p_multi', NULL, 'sess_shared', 'completed', '{}', '{}', '{}', '{}', '2026-01-02T10:00:00');"
        )
        # Create source and snapshot
        uow.conn.execute(
            "INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_1', 'https://multi.corp/news', 'multi.corp', '2026-01-01');"
        )
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_1', 'src_1', 'sess_shared', 'Multi Corp Expansion', 'Announcing massive logistics robotics expansion.', 'success', '2026-01-01T12:00:00');"
        )
        uow.conn.commit()

    service = HistoricalRetrievalService(clean_store)

    # 1. Unscoped search across the session: run_id must be None (no fabricated 'run_later')
    resp_unscoped = service.search("robotics expansion")
    assert resp_unscoped.total_matches == 1
    assert resp_unscoped.items[0].run_id is None
    assert resp_unscoped.items[0].session_id == "sess_shared"
    assert resp_unscoped.items[0].project_id == "p_multi"

    # 2. Scoped to run_earlier: run_id must be attributed to run_earlier
    resp_earlier = service.search("robotics expansion", scope=RetrievalScope(run_id="run_earlier"))
    assert resp_earlier.total_matches == 1
    assert resp_earlier.items[0].run_id == "run_earlier"

    # 3. Scoped to run_later: run_id must be attributed to run_later
    resp_later = service.search("robotics expansion", scope=RetrievalScope(run_id="run_later"))
    assert resp_later.total_matches == 1
    assert resp_later.items[0].run_id == "run_later"

    # 4. Scoped to non-existent run: returns 0 matches
    resp_none = service.search("robotics expansion", scope=RetrievalScope(run_id="run_nonexistent"))
    assert resp_none.total_matches == 0


def test_project_isolation_for_snapshots_and_passages(clean_store: StateStore):
    """Verify cross-project isolation: prospect_A search never retrieves prospect_B snapshots or passages."""
    with clean_store.uow() as uow:
        # Prospect A
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('proj_A', 'Alpha Lead', 'Alpha Tech', '2026-02-01');")
        uow.conn.execute(
            "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at, status) "
            "VALUES ('sess_A', 'proj_A', 'ctx_A', '2026-02-01', 'completed');"
        )
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_A', 'https://alpha.tech', 'alpha.tech', '2026-02-01');")
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_A', 'src_A', 'sess_A', 'Alpha Architecture', 'Next-gen distributed edge microservices platform.', 'success', '2026-02-01T10:00:00');"
        )
        uow.conn.execute(
            "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
            "VALUES ('ev_A', 'snap_A', 'src_A', 'Alpha edge microservices architecture latency benchmark.', '2026-02-01T10:05:00');"
        )

        # Prospect B
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('proj_B', 'Beta Scientist', 'Beta Bio', '2026-02-01');")
        uow.conn.execute(
            "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at, status) "
            "VALUES ('sess_B', 'proj_B', 'ctx_B', '2026-02-01', 'completed');"
        )
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_B', 'https://beta.bio', 'beta.bio', '2026-02-01');")
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_B', 'src_B', 'sess_B', 'Beta Architecture', 'Next-gen biological microservices computing.', 'success', '2026-02-01T11:00:00');"
        )
        uow.conn.execute(
            "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
            "VALUES ('ev_B', 'snap_B', 'src_B', 'Beta biological microservices computational pipeline.', '2026-02-01T11:05:00');"
        )
        uow.conn.commit()

    service = HistoricalRetrievalService(clean_store)

    # Scoped to proj_A: must only retrieve A
    res_A = service.search("microservices", scope=RetrievalScope(project_id="proj_A"))
    assert res_A.total_matches == 2
    for item in res_A.items:
        assert item.project_id == "proj_A"
        assert "Alpha" in item.title or "Alpha" in item.raw_text

    # Scoped to proj_B: must only retrieve B
    res_B = service.search("microservices", scope=RetrievalScope(project_id="proj_B"))
    assert res_B.total_matches == 2
    for item in res_B.items:
        assert item.project_id == "proj_B"
        assert "Beta" in item.title or "Beta" in item.raw_text

    # Passage-only search scoped to proj_A
    res_pass_A = service.search(
        "microservices",
        scope=RetrievalScope(project_id="proj_A", record_type=RetrievedRecordType.EVIDENCE_RECORD),
    )
    assert res_pass_A.total_matches == 1
    assert res_pass_A.items[0].record_id == "ev_A"
    assert res_pass_A.items[0].project_id == "proj_A"


def test_auto_backfill_pre_m33_database_on_access(tmp_path: Path):
    """Verify that reopening a pre-M3.3 database without FTS tables auto-creates and backfills FTS indexes."""
    db_file = str(tmp_path / "pre_m33_legacy.db")

    # Manually initialize legacy schema WITHOUT calling _ensure_fts_indexes
    conn = sqlite3.connect(db_file)
    with open("src/db/schema.sql", "r") as f:
        conn.executescript(f.read())

    # Insert legacy source snapshot and evidence passage
    conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_legacy', 'Legacy Tech', 'Legacy Systems', '2025-11-01');")
    conn.execute(
        "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at, status) "
        "VALUES ('sess_legacy', 'p_legacy', 'ctx_leg', '2025-11-01', 'completed');"
    )
    conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_leg', 'https://legacy.io', 'legacy.io', '2025-11-01');")
    conn.execute(
        "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
        "VALUES ('snap_leg', 'src_leg', 'sess_legacy', 'Legacy Telemetry Spec', 'Proprietary telemetry streaming over gRPC.', 'success', '2025-11-01');"
    )
    conn.execute(
        "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
        "VALUES ('ev_leg', 'snap_leg', 'src_leg', 'Latency benchmarks show sub-millisecond gRPC streaming telemetry.', '2025-11-01');"
    )
    conn.commit()

    # Verify FTS tables do NOT exist yet
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='source_snapshots_fts';")
    assert cur.fetchone() is None
    conn.close()

    # Now open with StateStore and HistoricalRetrievalService
    store = StateStore(db_file)
    service = HistoricalRetrievalService(store)

    # Calling search must auto-create FTS tables and backfill pre-existing records!
    resp = service.search("telemetry gRPC")
    assert resp.total_matches == 2

    snap_matches = [i for i in resp.items if i.record_type == RetrievedRecordType.SOURCE_SNAPSHOT]
    ev_matches = [i for i in resp.items if i.record_type == RetrievedRecordType.EVIDENCE_RECORD]
    assert len(snap_matches) == 1
    assert snap_matches[0].record_id == "snap_leg"
    assert len(ev_matches) == 1
    assert ev_matches[0].record_id == "ev_leg"


def test_relevance_score_semantics_raw_bm25(clean_store: StateStore):
    """Verify that rank and score preserve raw BM25 without abs() pseudo-probability inversion."""
    with clean_store.uow() as uow:
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_score', 'Score Chief', 'Score Corp', '2026-03-01');")
        uow.conn.execute(
            "INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at, status) "
            "VALUES ('sess_score', 'p_score', 'ctx_score', '2026-03-01', 'completed');"
        )
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_score', 'https://score.corp', 'score.corp', '2026-03-01');")
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_s', 'src_score', 'sess_score', 'BM25 Semantics Verification', 'Detailed analysis of BM25 ranking mechanics.', 'success', '2026-03-01');"
        )
        uow.conn.commit()

    service = HistoricalRetrievalService(clean_store)
    resp = service.search("BM25 ranking mechanics")
    assert resp.total_matches == 1

    item = resp.items[0]
    # In SQLite FTS5, BM25 returns a negative number where lower/more negative is better match
    assert item.rank <= 0.0
    assert item.score == item.rank
    # Verify score is NOT inverted by abs()
    assert item.score <= 0.0
