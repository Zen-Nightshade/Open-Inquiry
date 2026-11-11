from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.db.state_store import StateStore
from src.researcher.retrieval import (
    FTSQueryParser,
    HistoricalRetrievalService,
    RetrievalScope,
    RetrievedRecordType,
)


@pytest.fixture
def populated_store(tmp_path: Path) -> StateStore:
    db_file = tmp_path / "retrieval_test.db"
    store = StateStore(db_file)
    store.initialize()

    with store.uow() as uow:
        # Tenants / Prospects
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('proj_alpha', 'Alice', 'Alpha Corp', '2026-11-10');")
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('proj_beta', 'Bob', 'Beta Industries', '2026-11-10');")

        # Sessions
        uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_alpha_1', 'proj_alpha', 'ctx_a1', '2026-11-10');")
        uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_beta_1', 'proj_beta', 'ctx_b1', '2026-11-10');")

        # Runs
        uow.conn.execute(
            "INSERT INTO application_runs (run_id, prospect_id, context_id, research_session_id, status, prospect_snapshot_json, product_snapshot_json, outreach_snapshot_json, stage_statuses_json, started_at) "
            "VALUES ('run_alpha_1', 'proj_alpha', NULL, 'sess_alpha_1', 'completed', '{}', '{}', '{}', '{}', '2026-11-10');"
        )
        uow.conn.execute(
            "INSERT INTO application_runs (run_id, prospect_id, context_id, research_session_id, status, prospect_snapshot_json, product_snapshot_json, outreach_snapshot_json, stage_statuses_json, started_at) "
            "VALUES ('run_beta_1', 'proj_beta', NULL, 'sess_beta_1', 'completed', '{}', '{}', '{}', '{}', '2026-11-10');"
        )

        # Sources
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_a', 'https://alphacorp.com/news/ai', 'alphacorp.com', '2026-11-10');")
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_b', 'https://betaind.com/press', 'betaind.com', '2026-11-10');")

        # Snapshots
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_a1', 'src_a', 'sess_alpha_1', 'Alpha Corp Deploys Robotics', 'Alpha Corp has deployed autonomous warehouse robotics with computer vision.', 'success', '2026-11-01');"
        )
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_b1', 'src_b', 'sess_beta_1', 'Beta Industries Robotics Venture', 'Beta Industries is investing heavily in robotics and automated assembly lines.', 'success', '2026-11-05');"
        )

        # Evidence Passages
        uow.conn.execute(
            "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
            "VALUES ('ev_a1', 'snap_a1', 'src_a', 'autonomous warehouse robotics deployed across 15 distribution centers', '2026-11-01');"
        )
        uow.conn.execute(
            "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
            "VALUES ('ev_b1', 'snap_b1', 'src_b', 'automated assembly lines increase throughput by 40 percent', '2026-11-05');"
        )
    return store


def test_keyword_and_phrase_search(populated_store: StateStore):
    """Verify single-word, multi-word, and exact quoted phrase retrieval."""
    service = HistoricalRetrievalService(populated_store)

    # 1. Single keyword matches across snapshot and evidence
    res = service.search("robotics")
    assert res.total_matches >= 2
    record_ids = {item.record_id for item in res.items}
    assert "snap_a1" in record_ids
    assert "snap_b1" in record_ids

    # 2. Quoted phrase matches exact sequence
    res_phrase = service.search('"warehouse robotics"')
    assert res_phrase.total_matches >= 1
    for item in res_phrase.items:
        assert "warehouse" in item.raw_text.lower() and "robotics" in item.raw_text.lower()
        assert "beta" not in item.raw_text.lower()

    # 3. Disjoint phrase should yield 0 matches
    res_none = service.search('"robotics warehouse"')
    assert res_none.total_matches == 0
    assert len(res_none.items) == 0


def test_safe_malformed_and_edge_case_queries(populated_store: StateStore):
    """Verify query parser sanitizes reserved operators, punctuation, and unclosed quotes without errors."""
    service = HistoricalRetrievalService(populated_store)

    # Empty and whitespace queries
    res_empty = service.search("")
    assert res_empty.total_matches == 0
    assert res_empty.sanitized_query == ""

    res_ws = service.search("     ")
    assert res_ws.total_matches == 0
    assert res_ws.sanitized_query == ""

    # Reserved operators in isolation
    for op in ["NOT", "AND", "OR", "not", "and", "or"]:
        res_op = service.search(op)
        assert isinstance(res_op.total_matches, int)
        assert res_op.sanitized_query == f'"{op.upper()}"'

    # Unclosed quotes
    res_unclosed = service.search('"autonomous robotics')
    assert isinstance(res_unclosed.total_matches, int)
    assert res_unclosed.total_matches >= 1

    # Colons and special characters
    res_punct = service.search("robotics: (vision) [autonomous]!")
    assert isinstance(res_punct.total_matches, int)
    assert res_punct.total_matches >= 1

    # Operators between words
    res_bool = service.search("robotics AND vision NOT assembly")
    assert isinstance(res_bool.total_matches, int)
    assert res_bool.total_matches >= 1
    # Must only match Alpha, not Beta
    for item in res_bool.items:
        assert "assembly" not in item.raw_text.lower()


def test_bm25_ranking_and_snippets(populated_store: StateStore):
    """Verify BM25 ranking orders best matches first and snippets highlight [[MATCH]] markers."""
    service = HistoricalRetrievalService(populated_store)

    res = service.search("autonomous warehouse robotics")
    assert res.total_matches >= 1
    top_item = res.items[0]

    # Verify raw BM25 score (FTS5 BM25 convention: negative, unbounded lexical score)
    assert top_item.rank <= 0.0
    assert top_item.score == top_item.rank

    # Verify snippet contains markers
    has_match_tag = any("[[MATCH]]" in item.snippet for item in res.items)
    assert has_match_tag is True
    has_closing_tag = any("[[/MATCH]]" in item.snippet for item in res.items)
    assert has_closing_tag is True


def test_record_type_filtering(populated_store: StateStore):
    """Verify filtering by record_type returns only snapshots or evidence records."""
    service = HistoricalRetrievalService(populated_store)

    # Only snapshots
    scope_snap = RetrievalScope(record_type=RetrievedRecordType.SOURCE_SNAPSHOT)
    res_snap = service.search("robotics", scope=scope_snap)
    assert res_snap.total_matches >= 1
    for item in res_snap.items:
        assert item.record_type == RetrievedRecordType.SOURCE_SNAPSHOT

    # Only evidence passages
    scope_ev = RetrievalScope(record_type=RetrievedRecordType.EVIDENCE_RECORD)
    res_ev = service.search("robotics", scope=scope_ev)
    assert res_ev.total_matches >= 1
    for item in res_ev.items:
        assert item.record_type == RetrievedRecordType.EVIDENCE_RECORD


def test_tenant_and_project_scoping_prevents_leakage(populated_store: StateStore):
    """Verify scoping strictly enforces project, run, session, and domain isolation."""
    service = HistoricalRetrievalService(populated_store)

    # 1. Project / Tenant isolation: Alpha scope should never see Beta records
    scope_alpha = RetrievalScope(project_id="proj_alpha")
    res_alpha = service.search("robotics", scope=scope_alpha)
    assert res_alpha.total_matches >= 1
    for item in res_alpha.items:
        assert item.project_id == "proj_alpha"
        assert item.session_id == "sess_alpha_1"
        assert "beta" not in item.raw_text.lower()

    # Beta scope should never see Alpha records
    scope_beta = RetrievalScope(project_id="proj_beta")
    res_beta = service.search("robotics", scope=scope_beta)
    assert res_beta.total_matches >= 1
    for item in res_beta.items:
        assert item.project_id == "proj_beta"
        assert item.session_id == "sess_beta_1"
        assert "alpha" not in item.raw_text.lower()

    # 2. Run scoping
    scope_run_a = RetrievalScope(run_id="run_alpha_1")
    res_run_a = service.search("robotics", scope=scope_run_a)
    assert res_run_a.total_matches >= 1
    for item in res_run_a.items:
        assert item.run_id == "run_alpha_1"

    # Non-existent run
    scope_fake_run = RetrievalScope(run_id="nonexistent_run")
    res_fake = service.search("robotics", scope=scope_fake_run)
    assert res_fake.total_matches == 0
    assert len(res_fake.items) == 0

    # 3. Domain scoping
    scope_domain = RetrievalScope(domain="alphacorp.com")
    res_dom = service.search("robotics", scope=scope_domain)
    assert res_dom.total_matches >= 1
    for item in res_dom.items:
        assert item.domain == "alphacorp.com"


def test_pagination_limits_and_offset(populated_store: StateStore):
    """Verify limit and offset pagination parameters."""
    service = HistoricalRetrievalService(populated_store)

    # Limit 1
    res1 = service.search("robotics", limit=1, offset=0)
    assert len(res1.items) == 1
    assert res1.total_matches >= 2
    first_id = res1.items[0].record_id

    # Offset 1
    res2 = service.search("robotics", limit=1, offset=1)
    assert len(res2.items) == 1
    second_id = res2.items[0].record_id

    assert first_id != second_id
