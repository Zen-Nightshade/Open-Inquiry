from __future__ import annotations

import hashlib
import logging
import sqlite3
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from repo_paths import get_db_path
from src.db.connection import get_connection

logger = logging.getLogger(__name__)

# Context variables propagated across the asynchronous or synchronous execution flow
active_run_id: ContextVar[Optional[str]] = ContextVar("active_run_id", default=None)
active_job_id: ContextVar[Optional[str]] = ContextVar("active_job_id", default=None)
active_stage_name: ContextVar[Optional[str]] = ContextVar("active_stage_name", default=None)
active_attempt_num: ContextVar[int] = ContextVar("active_attempt_num", default=1)


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute deterministic SHA-256 hash of prompt content without storing raw text."""
    return hashlib.sha256((prompt_text or "").encode("utf-8")).hexdigest()


def record_model_execution(
    provider: str,
    model_name: str,
    task_type: str,
    prompt_text: str,
    duration_ms: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    status: str = "completed",
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    stage_name: Optional[str] = None,
    attempt_num: Optional[int] = None,
    execution_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Persist an execution record to the model_executions table.

    Guarantees:
    - Never raises an exception; logging warnings on storage failure to protect stage execution.
    - Captures sha256 prompt hash rather than full prompt text by default.
    - Never fabricates token counts or metrics.
    - Connects executions to run_id and job_id.
    """
    resolved_exec_id = execution_id or f"exec_{uuid.uuid4().hex}"
    resolved_run_id = run_id or active_run_id.get()
    resolved_job_id = job_id or active_job_id.get()
    resolved_stage = stage_name or active_stage_name.get()
    resolved_attempt = attempt_num or active_attempt_num.get()
    prompt_hash = compute_prompt_hash(prompt_text)
    now = datetime.now(timezone.utc).isoformat()

    try:
        target_path = get_db_path(db_path)
        # Avoid creating empty database files if path doesn't exist yet and is not memory
        if str(target_path) != ":memory:" and not Path(target_path).exists():
            return resolved_exec_id

        conn = get_connection(target_path)
        try:
            # Check available columns on model_executions table to handle legacy tables
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(model_executions);")
            cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}

            if not cols:
                # Table not created yet
                return resolved_exec_id

            if resolved_job_id:
                job_exists = cursor.execute("SELECT 1 FROM jobs WHERE job_id = ? LIMIT 1;", (resolved_job_id,)).fetchone()
                if not job_exists:
                    resolved_job_id = None

            col_names = [
                "execution_id",
                "job_id",
                "provider",
                "model_name",
                "task_type",
                "prompt_hash",
                "input_tokens",
                "output_tokens",
                "duration_ms",
                "executed_at",
            ]
            values = [
                resolved_exec_id,
                resolved_job_id,
                provider,
                model_name,
                task_type,
                prompt_hash,
                input_tokens,
                output_tokens,
                duration_ms,
                now,
            ]

            # Dynamically include extended columns if present
            if "run_id" in cols:
                col_names.append("run_id")
                values.append(resolved_run_id)
            if "stage_name" in cols:
                col_names.append("stage_name")
                values.append(resolved_stage)
            if "status" in cols:
                col_names.append("status")
                values.append(status)
            if "error_type" in cols:
                col_names.append("error_type")
                values.append(error_type)
            if "error_message" in cols:
                col_names.append("error_message")
                values.append(error_message)
            if "attempt_num" in cols:
                col_names.append("attempt_num")
                values.append(resolved_attempt)

            placeholders = ", ".join(["?"] * len(col_names))
            sql = f"INSERT INTO model_executions ({', '.join(col_names)}) VALUES ({placeholders});"
            conn.execute(sql, values)
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Failed to record model execution %s: %s", resolved_exec_id, exc)

    return resolved_exec_id


def get_model_executions(
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve model execution records by job_id, run_id, or all."""
    target_path = get_db_path(db_path)
    if str(target_path) != ":memory:" and not Path(target_path).exists():
        return []

    conn = get_connection(target_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(model_executions);")
        cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
        if not cols:
            return []

        if run_id and "run_id" in cols:
            cursor.execute(
                "SELECT * FROM model_executions WHERE run_id = ? ORDER BY executed_at ASC;",
                (run_id,),
            )
        elif job_id:
            cursor.execute(
                "SELECT * FROM model_executions WHERE job_id = ? ORDER BY executed_at ASC;",
                (job_id,),
            )
        else:
            cursor.execute("SELECT * FROM model_executions ORDER BY executed_at ASC;")

        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
