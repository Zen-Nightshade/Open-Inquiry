from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class JobRepository:
    """Repository managing operational background jobs, attempts, LLM execution metrics, and workflow events."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    def create_job(
        self,
        job_id: str,
        job_type: str,
        status: str = "pending",
        idempotency_key: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO jobs (job_id, job_type, status, idempotency_key, created_at)
        VALUES (?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, (job_id, job_type, status, idempotency_key, now))
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_job_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM jobs WHERE idempotency_key = ?;",
            (idempotency_key,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_job_status(
        self,
        job_id: str,
        status: str,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        sql = """
        UPDATE jobs
        SET status = ?,
            started_at = COALESCE(?, started_at),
            completed_at = COALESCE(?, completed_at),
            error_message = ?
        WHERE job_id = ?;
        """
        cursor = self.conn.execute(
            sql,
            (status, started_at, completed_at, error_message, job_id),
        )
        return cursor.rowcount > 0

    def record_attempt(
        self,
        attempt_id: str,
        job_id: str,
        attempt_num: int,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = started_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO job_attempts (
            attempt_id, job_id, attempt_num, started_at, completed_at, outcome, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (attempt_id, job_id, attempt_num, now, completed_at, outcome, error),
        )
        cursor = self.conn.execute(
            "SELECT * FROM job_attempts WHERE attempt_id = ?;",
            (attempt_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_attempts(self, job_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY attempt_num ASC;",
            (job_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def record_model_execution(
        self,
        execution_id: str,
        job_id: Optional[str],
        provider: str,
        model_name: str,
        task_type: str,
        prompt_hash: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
        executed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = executed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO model_executions (
            execution_id, job_id, provider, model_name, task_type,
            prompt_hash, input_tokens, output_tokens, duration_ms, executed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                execution_id,
                job_id,
                provider,
                model_name,
                task_type,
                prompt_hash,
                input_tokens,
                output_tokens,
                duration_ms,
                now,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM model_executions WHERE execution_id = ?;",
            (execution_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_model_executions(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if job_id:
            cursor = self.conn.execute(
                "SELECT * FROM model_executions WHERE job_id = ? ORDER BY executed_at ASC;",
                (job_id,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM model_executions ORDER BY executed_at ASC;"
            )
        return [dict(r) for r in cursor.fetchall()]

    def record_workflow_event(
        self,
        event_id: str,
        event_type: str,
        session_id: Optional[str] = None,
        job_id: Optional[str] = None,
        payload: Optional[str] = None,
        occurred_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = occurred_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO workflow_events (
            event_id, session_id, job_id, event_type, payload, occurred_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (event_id, session_id, job_id, event_type, payload, now),
        )
        cursor = self.conn.execute(
            "SELECT * FROM workflow_events WHERE event_id = ?;",
            (event_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_workflow_events(
        self,
        session_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if job_id:
            conditions.append("job_id = ?")
            params.append(job_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM workflow_events {where} ORDER BY occurred_at ASC;"
        cursor = self.conn.execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]
