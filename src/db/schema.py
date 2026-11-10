from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import List, Set, Tuple

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

