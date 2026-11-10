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
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize database schema: {exc}") from exc

    verification = verify_schema(conn)
    if not verification["is_valid"]:
        raise RuntimeError(f"Database schema initialization incomplete: {'; '.join(verification['errors'])}")


def verify_schema(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Verify that all 36 domain tables exist and conform to expected structural constraints."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    tables = {row["name"] if isinstance(row, sqlite3.Row) else row[0] for row in cursor.fetchall()}

    missing_tables = sorted(list(EXPECTED_DOMAIN_TABLES - tables))
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

