from __future__ import annotations

import sqlite3
import pytest

from src.db.repositories.assertion_repo import AssertionRepository
from src.db.repositories.source_repo import SourceRepository
from src.db.schema import EXPECTED_DOMAIN_TABLES, init_schema, verify_schema
from src.researcher.syndication.models import (
    DependenceType,
    DetectionMethod,
    SyndicationCluster,
)


def _setup_session(conn: sqlite3.Connection, session_id: str = "sess_1", prospect_id: str = "pros_1") -> None:
    now = "2026-03-15T00:00:00Z"
    conn.execute(
        "INSERT OR IGNORE INTO prospects (prospect_id, full_name, company_name, created_at) VALUES (?, ?, ?, ?);",
        (prospect_id, "Jane Doe", "Acme Corp", now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO research_sessions (session_id, prospect_id, context_id, started_at, status) VALUES (?, ?, ?, ?, ?);",
        (session_id, prospect_id, "ctx_test", now, "in_progress"),
    )
    conn.commit()


def test_schema_36_tables_invariant_with_syndication():
    """Verify that adding syndication columns and indexes preserves exactly 36 domain tables."""
    conn = sqlite3.connect(":memory:")
    init_schema(conn)

    assert len(EXPECTED_DOMAIN_TABLES) == 36
    res = verify_schema(conn)
    assert res["is_valid"] is True
    assert res["missing_tables"] == []

    # Verify syndication columns exist on source_snapshots
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(source_snapshots);")
    snap_cols = {row[1] for row in cursor.fetchall()}
    assert "raw_content_hash" in snap_cols
    assert "normalized_content_hash" in snap_cols
    assert "minhash_signature" in snap_cols

    # Verify syndication columns exist on evidence_origin_groups
    cursor.execute("PRAGMA table_info(evidence_origin_groups);")
    orig_cols = {row[1] for row in cursor.fetchall()}
    assert "dependence_type" in orig_cols
    assert "primary_detection_method" in orig_cols
    assert "average_similarity" in orig_cols

    # Verify independent_source_count on canonical_facts
    cursor.execute("PRAGMA table_info(canonical_facts);")
    fact_cols = {row[1] for row in cursor.fetchall()}
    assert "independent_source_count" in fact_cols

    # Verify migration idempotence
    init_schema(conn)
    init_schema(conn)
    conn.close()


def test_snapshot_fingerprint_persistence_and_hash_lookup():
    """Verify storing and looking up source snapshots by deterministic raw and normalized hashes."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)

    source_repo = SourceRepository(conn)

    # Insert a source
    source_repo.upsert_source(
        source_id="src_tech",
        canonical_url="https://tech.example.com/acme-raises",
        domain="tech.example.com",
        source_type="news",
    )

    # Insert snapshot with initial hashes
    snap = source_repo.create_snapshot(
        snapshot_id="snap_101",
        source_id="src_tech",
        fetch_status="success",
        title="Acme Raises $50M",
        raw_text="Acme Corp raised $50M today in Series B.",
        raw_content_hash="raw_hash_abc123",
        normalized_content_hash="norm_hash_xyz789",
        minhash_signature="[12, 34, 56]",
    )
    assert snap["snapshot_id"] == "snap_101"
    assert snap["raw_content_hash"] == "raw_hash_abc123"
    assert snap["normalized_content_hash"] == "norm_hash_xyz789"
    assert snap["minhash_signature"] == "[12, 34, 56]"

    # Lookup by raw hash
    found_raw = source_repo.get_snapshot_by_raw_hash("raw_hash_abc123")
    assert found_raw is not None
    assert found_raw["snapshot_id"] == "snap_101"

    # Lookup by normalized hash
    found_norm = source_repo.get_snapshot_by_normalized_hash("norm_hash_xyz789")
    assert found_norm is not None
    assert found_norm["snapshot_id"] == "snap_101"

    # Update fingerprint after processing
    updated = source_repo.record_snapshot_fingerprint(
        snapshot_id="snap_101",
        raw_content_hash="raw_hash_new456",
        normalized_content_hash="norm_hash_new999",
        minhash_signature="[99, 88, 77]",
    )
    assert updated is True

    found_updated = source_repo.get_snapshot_by_raw_hash("raw_hash_new456")
    assert found_updated is not None
    assert found_updated["normalized_content_hash"] == "norm_hash_new999"

    conn.close()


def test_syndication_cluster_roundtrip_persistence():
    """Verify persisting a SyndicationCluster to evidence_origin_groups and evidence_origin_members."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)

    source_repo = SourceRepository(conn)

    # Seed sources
    source_repo.upsert_source(
        source_id="src_wire",
        canonical_url="https://wire.com/rel",
        domain="wire.com",
        source_type="press_release",
    )
    source_repo.upsert_source(
        source_id="src_yahoo",
        canonical_url="https://yahoo.com/rel",
        domain="yahoo.com",
        source_type="news",
    )
    source_repo.upsert_source(
        source_id="src_bloom",
        canonical_url="https://bloom.com/rel",
        domain="bloom.com",
        source_type="news",
    )

    cluster = SyndicationCluster(
        cluster_id="syn_cluster_001",
        canonical_source_id="src_wire",
        member_source_ids={"src_wire", "src_yahoo", "src_bloom"},
        dependence_type=DependenceType.NEAR_DUPLICATE,
        primary_detection_method=DetectionMethod.MINHASH_LSH,
        average_similarity=0.88,
        description="Wire syndication cluster for Series B announcement",
    )

    saved = source_repo.save_syndication_cluster(cluster)
    assert saved["group_id"] == "syn_cluster_001"
    assert saved["canonical_source_id"] == "src_wire"
    assert saved["dependence_type"] == "near_duplicate"
    assert saved["primary_detection_method"] == "minhash_lsh"
    assert saved["average_similarity"] == pytest.approx(0.88)

    members = source_repo.get_origin_members("syn_cluster_001")
    member_ids = {m["source_id"] for m in members}
    assert member_ids == {"src_wire", "src_yahoo", "src_bloom"}

    # Query groups map
    groups_map = source_repo.get_syndication_groups()
    assert "syn_cluster_001" in groups_map
    assert groups_map["syn_cluster_001"] == {"src_wire", "src_yahoo", "src_bloom"}

    conn.close()


def test_canonical_fact_independent_source_count_persistence():
    """Verify persistence of independent_source_count on canonical facts."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)

    _setup_session(conn, session_id="sess_1", prospect_id="prospect_1")

    assertion_repo = AssertionRepository(conn)
    fact = assertion_repo.create_canonical_fact(
        fact_id="fact_100",
        session_id="sess_1",
        claim_text="Acme Corp closed Series B funding",
        fact_category="funding",
        composite_confidence="medium",
        independent_source_count=1,
    )

    assert fact["fact_id"] == "fact_100"
    assert fact["independent_source_count"] == 1

    retrieved = assertion_repo.get_canonical_fact("fact_100")
    assert retrieved is not None
    assert retrieved["independent_source_count"] == 1

    conn.close()
