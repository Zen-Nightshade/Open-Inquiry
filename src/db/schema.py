from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"

EXPECTED_DOMAIN_TABLES: Set[str] = {
    "jobs",
    "job_attempts",
    "model_executions",
    "prospects",
    "product_versions",
    "outreach_contexts",
    "outreach_drafts",
    "human_review_events",
    "research_sessions",
    "research_objectives",
    "research_queries",
    "workflow_events",
    "sources",
    "search_executions",
    "search_result_occurrences",
    "source_snapshots",
    "evidence_passages",
    "evidence_origin_groups",
    "evidence_origin_members",
    "entities",
    "entity_identifiers",
    "entity_aliases",
    "entity_affiliations",
    "entity_relationships",
    "entity_candidates",
    "identity_assessments",
    "canonical_facts",
    "fact_entity_links",
    "fact_evidence_links",
    "conflicts",
    "conflict_facts",
    "assessment_history",
    "resolution_decisions",
    "research_iterations",
    "sufficiency_assessments",
    "application_runs",
}

FTS_VIRTUAL_TABLES: Set[str] = {
    "source_snapshots_fts",
    "evidence_passages_fts",
}

FTS_SHADOW_SUFFIXES: Tuple[str, ...] = (
    "_data",
    "_idx",
    "_docsize",
    "_config",
    "_content",
)


def is_fts_table(table_name: str) -> bool:
    """Determine whether a table is an FTS5 virtual table or internal shadow table."""
    return table_name in FTS_VIRTUAL_TABLES or any(
        table_name.startswith(f"{v}_") for v in FTS_VIRTUAL_TABLES
    )


def load_schema_sql() -> str:
    """Read the complete authoritative database schema SQL."""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Authoritative schema file missing at {SCHEMA_FILE}")
    return SCHEMA_FILE.read_text(encoding="utf-8")


def _ensure_model_executions_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added observability columns exist on model_executions table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(model_executions);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("run_id", "TEXT"),
        ("stage_name", "TEXT"),
        ("status", "TEXT NOT NULL DEFAULT 'completed'"),
        ("error_type", "TEXT"),
        ("error_message", "TEXT"),
        ("attempt_num", "INTEGER NOT NULL DEFAULT 1"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE model_executions ADD COLUMN {col_name} {col_def};")
    conn.commit()


def _ensure_application_runs_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added columns exist on application_runs table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(application_runs);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("draft_id", "TEXT REFERENCES outreach_drafts(draft_id) ON DELETE SET NULL"),
        ("feedback_round_json", "TEXT"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE application_runs ADD COLUMN {col_name} {col_def};")
    conn.commit()


def _ensure_outreach_drafts_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added columns exist on outreach_drafts table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(outreach_drafts);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("current_revision", "INTEGER NOT NULL DEFAULT 1"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE outreach_drafts ADD COLUMN {col_name} {col_def};")
    conn.commit()


def _ensure_human_review_events_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added columns exist on human_review_events table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(human_review_events);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("action", "TEXT NOT NULL DEFAULT 'APPROVED'"),
        ("decision", "TEXT"),
        ("reviewed_revision", "INTEGER NOT NULL DEFAULT 1"),
        ("resulting_revision", "INTEGER NOT NULL DEFAULT 1"),
        ("reason_code", "TEXT"),
        ("original_subject", "TEXT"),
        ("original_body", "TEXT"),
        ("resulting_subject", "TEXT"),
        ("resulting_body", "TEXT"),
        ("char_edit_distance", "INTEGER"),
        ("word_edit_distance", "INTEGER"),
        ("normalized_char_ratio", "REAL"),
        ("normalized_word_ratio", "REAL"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE human_review_events ADD COLUMN {col_name} {col_def};")
    conn.commit()


def _ensure_canonical_facts_molecular_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added molecular columns exist on canonical_facts table idempotently."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(canonical_facts);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("molecular_data_json", "TEXT"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE canonical_facts ADD COLUMN {col_name} {col_def};")
    conn.commit()


def _ensure_canonical_facts_bitemporal_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added bitemporal columns exist on canonical_facts and backfill legacy rows idempotently."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(canonical_facts);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("valid_from", "TEXT"),
        ("valid_to", "TEXT"),
        ("observed_at", "TEXT"),
        ("recorded_at", "TEXT"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE canonical_facts ADD COLUMN {col_name} {col_def};")

    # Idempotent backfill: populate recorded_at/observed_at from created_at and valid_from from primary_event_date
    conn.execute("UPDATE canonical_facts SET recorded_at = created_at WHERE recorded_at IS NULL;")
    conn.execute("UPDATE canonical_facts SET observed_at = created_at WHERE observed_at IS NULL;")
    conn.execute("UPDATE canonical_facts SET valid_from = primary_event_date WHERE valid_from IS NULL AND primary_event_date IS NOT NULL;")

    # Ensure index on bitemporal dimensions exists
    conn.execute("CREATE INDEX IF NOT EXISTS idx_canonical_facts_temporal ON canonical_facts(valid_from, valid_to, recorded_at);")
    conn.commit()


def _ensure_source_snapshots_syndication_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added syndication fingerprint columns exist on source_snapshots idempotently."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(source_snapshots);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("raw_content_hash", "TEXT"),
        ("normalized_content_hash", "TEXT"),
        ("minhash_signature", "TEXT"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE source_snapshots ADD COLUMN {col_name} {col_def};")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_source_snapshots_raw_hash ON source_snapshots(raw_content_hash);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source_snapshots_norm_hash ON source_snapshots(normalized_content_hash);")
    conn.commit()


def _ensure_evidence_origin_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added syndication metadata columns exist on evidence_origin_groups idempotently."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(evidence_origin_groups);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    columns_to_add = [
        ("dependence_type", "TEXT"),
        ("primary_detection_method", "TEXT"),
        ("average_similarity", "REAL"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE evidence_origin_groups ADD COLUMN {col_name} {col_def};")
    conn.commit()


def _ensure_canonical_facts_syndication_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added independent_source_count exists on canonical_facts idempotently."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(canonical_facts);")
    cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if not cols:
        return

    if "independent_source_count" not in cols:
        conn.execute("ALTER TABLE canonical_facts ADD COLUMN independent_source_count INTEGER;")
    conn.commit()


def _ensure_entity_resolution_columns(conn: sqlite3.Connection) -> None:
    """Ensure newly added ambiguous candidate attribution and decision metadata columns exist idempotently."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(canonical_facts);")
    c_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if c_cols:
        if "ambiguous_candidates_json" not in c_cols:
            conn.execute("ALTER TABLE canonical_facts ADD COLUMN ambiguous_candidates_json TEXT;")
        if "entity_resolution_status" not in c_cols:
            conn.execute("ALTER TABLE canonical_facts ADD COLUMN entity_resolution_status TEXT;")

    cursor.execute("PRAGMA table_info(identity_assessments);")
    a_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
    if a_cols:
        if "decision_metadata_json" not in a_cols:
            conn.execute("ALTER TABLE identity_assessments ADD COLUMN decision_metadata_json TEXT;")
    conn.commit()


def _ensure_fts_indexes(conn: sqlite3.Connection) -> None:
    """Ensure SQLite FTS5 virtual tables and synchronization triggers exist."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _test_fts5_probe USING fts5(x);")
        conn.execute("DROP TABLE IF EXISTS _test_fts5_probe;")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f"SQLite FTS5 extension is not available in current environment: {exc}") from exc

    # 1. source_snapshots_fts external content virtual table
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS source_snapshots_fts USING fts5(
        title,
        raw_text,
        content='source_snapshots',
        content_rowid='rowid'
    );
    """)

    # 2. evidence_passages_fts external content virtual table
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_passages_fts USING fts5(
        verbatim_excerpt,
        content='evidence_passages',
        content_rowid='rowid'
    );
    """)

    # 3. Triggers for source_snapshots
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS source_snapshots_ai AFTER INSERT ON source_snapshots BEGIN
        INSERT INTO source_snapshots_fts(rowid, title, raw_text) VALUES (new.rowid, new.title, new.raw_text);
    END;
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS source_snapshots_ad AFTER DELETE ON source_snapshots BEGIN
        INSERT INTO source_snapshots_fts(source_snapshots_fts, rowid, title, raw_text) VALUES ('delete', old.rowid, old.title, old.raw_text);
    END;
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS source_snapshots_au AFTER UPDATE ON source_snapshots BEGIN
        INSERT INTO source_snapshots_fts(source_snapshots_fts, rowid, title, raw_text) VALUES ('delete', old.rowid, old.title, old.raw_text);
        INSERT INTO source_snapshots_fts(rowid, title, raw_text) VALUES (new.rowid, new.title, new.raw_text);
    END;
    """)

    # 4. Triggers for evidence_passages
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS evidence_passages_ai AFTER INSERT ON evidence_passages BEGIN
        INSERT INTO evidence_passages_fts(rowid, verbatim_excerpt) VALUES (new.rowid, new.verbatim_excerpt);
    END;
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS evidence_passages_ad AFTER DELETE ON evidence_passages BEGIN
        INSERT INTO evidence_passages_fts(evidence_passages_fts, rowid, verbatim_excerpt) VALUES ('delete', old.rowid, old.verbatim_excerpt);
    END;
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS evidence_passages_au AFTER UPDATE ON evidence_passages BEGIN
        INSERT INTO evidence_passages_fts(evidence_passages_fts, rowid, verbatim_excerpt) VALUES ('delete', old.rowid, old.verbatim_excerpt);
        INSERT INTO evidence_passages_fts(rowid, verbatim_excerpt) VALUES (new.rowid, new.verbatim_excerpt);
    END;
    """)

    # 5. Idempotent rebuild to ensure any pre-existing rows are indexed
    conn.execute("INSERT INTO source_snapshots_fts(source_snapshots_fts) VALUES('rebuild');")
    conn.execute("INSERT INTO evidence_passages_fts(evidence_passages_fts) VALUES('rebuild');")
    conn.commit()


def rebuild_fts_indexes(conn: sqlite3.Connection) -> None:
    """Force an administrative rebuild of all FTS5 full-text indexes."""
    conn.execute("INSERT INTO source_snapshots_fts(source_snapshots_fts) VALUES('rebuild');")
    conn.execute("INSERT INTO evidence_passages_fts(evidence_passages_fts) VALUES('rebuild');")
    conn.commit()


def check_fts_integrity(conn: sqlite3.Connection) -> Dict[str, bool]:
    """Execute SQLite FTS5 integrity-check commands across virtual tables."""
    conn.execute("INSERT INTO source_snapshots_fts(source_snapshots_fts) VALUES('integrity-check');")
    conn.execute("INSERT INTO evidence_passages_fts(evidence_passages_fts) VALUES('integrity-check');")
    return {
        "source_snapshots_fts": True,
        "evidence_passages_fts": True,
    }


def init_schema(conn: sqlite3.Connection) -> None:
    """Execute the authoritative schema script against the provided connection idempotently.
    
    Guarantees foreign key enforcement and verifies the existence of all 36 domain tables.
    """
    conn.execute("PRAGMA foreign_keys = ON;")
    schema_sql = load_schema_sql()

    try:
        conn.executescript(schema_sql)
        _ensure_model_executions_columns(conn)
        _ensure_application_runs_columns(conn)
        _ensure_outreach_drafts_columns(conn)
        _ensure_human_review_events_columns(conn)
        _ensure_canonical_facts_molecular_columns(conn)
        _ensure_canonical_facts_bitemporal_columns(conn)
        _ensure_source_snapshots_syndication_columns(conn)
        _ensure_evidence_origin_columns(conn)
        _ensure_canonical_facts_syndication_columns(conn)
        _ensure_entity_resolution_columns(conn)
        _ensure_fts_indexes(conn)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize database schema: {exc}") from exc

    verification = verify_schema(conn)
    if not verification["is_valid"]:
        raise RuntimeError(f"Database schema initialization incomplete: {'; '.join(verification['errors'])}")


def verify_schema(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Verify that all 36 domain tables exist and conform to expected structural constraints."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    all_tables = {row["name"] if isinstance(row, sqlite3.Row) else row[0] for row in cursor.fetchall()}
    domain_tables = {name for name in all_tables if not is_fts_table(name)}

    missing_tables = sorted(list(EXPECTED_DOMAIN_TABLES - domain_tables))
    errors: List[str] = []
    if missing_tables:
        errors.append(f"Missing {len(missing_tables)} tables: {missing_tables}")

    if not missing_tables:
        # Verify critical columns on key tables
        cursor.execute("PRAGMA table_info(research_objectives);")
        obj_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
        if "is_critical" not in obj_cols:
            errors.append("research_objectives missing is_critical column")

        cursor.execute("PRAGMA table_info(conflicts);")
        conflict_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
        if "severity" not in conflict_cols:
            errors.append("conflicts missing severity column")

        cursor.execute("PRAGMA table_info(application_runs);")
        run_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
        if "stage_statuses_json" not in run_cols or "strategy_result_json" not in run_cols:
            errors.append("application_runs missing required JSON snapshot columns")

    is_valid = len(errors) == 0 and len(missing_tables) == 0
    return {
        "is_valid": is_valid,
        "missing_tables": missing_tables,
        "errors": errors,
    }

