from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RunRepository:
    """Repository managing lifecycle and persistence for top-level application runs."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    def create_run(
        self,
        run_id: str,
        prospect_id: str,
        prospect_snapshot_json: str,
        product_snapshot_json: str,
        outreach_snapshot_json: str,
        stage_statuses_json: str,
        status: str = "pending",
        job_id: Optional[str] = None,
        context_id: Optional[str] = None,
        research_session_id: Optional[str] = None,
        strategy_result_json: Optional[str] = None,
        warnings_json: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a new application run record."""
        now = started_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO application_runs (
            run_id, job_id, prospect_id, context_id, research_session_id,
            status, prospect_snapshot_json, product_snapshot_json,
            outreach_snapshot_json, strategy_result_json, stage_statuses_json,
            warnings_json, error_message, started_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                run_id,
                job_id,
                prospect_id,
                context_id,
                research_session_id,
                status,
                prospect_snapshot_json,
                product_snapshot_json,
                outreach_snapshot_json,
                strategy_result_json,
                stage_statuses_json,
                warnings_json,
                error_message,
                now,
            ),
        )
        return self.get_run(run_id) or {}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an application run by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM application_runs WHERE run_id = ?;",
            (run_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        research_session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        strategy_result_json: Optional[str] = None,
        stage_statuses_json: Optional[str] = None,
        warnings_json: Optional[str] = None,
        error_message: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> bool:
        """Update fields on an existing application run record."""
        assignments: List[str] = []
        params: List[Any] = []

        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if research_session_id is not None:
            assignments.append("research_session_id = ?")
            params.append(research_session_id)
        if context_id is not None:
            assignments.append("context_id = ?")
            params.append(context_id)
        if strategy_result_json is not None:
            assignments.append("strategy_result_json = ?")
            params.append(strategy_result_json)
        if stage_statuses_json is not None:
            assignments.append("stage_statuses_json = ?")
            params.append(stage_statuses_json)
        if warnings_json is not None:
            assignments.append("warnings_json = ?")
            params.append(warnings_json)
        if error_message is not None:
            assignments.append("error_message = ?")
            params.append(error_message)
        if completed_at is not None:
            assignments.append("completed_at = ?")
            params.append(completed_at)

        if not assignments:
            return True

        params.append(run_id)
        sql = f"UPDATE application_runs SET {', '.join(assignments)} WHERE run_id = ?;"
        cursor = self.conn.execute(sql, tuple(params))
        return cursor.rowcount > 0

    def list_runs(
        self,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List application runs ordered by start time descending."""
        if status is not None:
            sql = "SELECT * FROM application_runs WHERE status = ? ORDER BY started_at DESC LIMIT ?;"
            cursor = self.conn.execute(sql, (status, limit))
        else:
            sql = "SELECT * FROM application_runs ORDER BY started_at DESC LIMIT ?;"
            cursor = self.conn.execute(sql, (limit,))
        return [dict(r) for r in cursor.fetchall()]

    def delete_run(self, run_id: str) -> bool:
        """Delete an application run by ID."""
        cursor = self.conn.execute("DELETE FROM application_runs WHERE run_id = ?;", (run_id,))
        return cursor.rowcount > 0
