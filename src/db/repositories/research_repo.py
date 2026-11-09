from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ResearchRepository:
    """Repository managing research sessions, objectives, query intents, and search provider executions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    # Sessions
    def create_session(
        self,
        session_id: str,
        prospect_id: str,
        context_id: str,
        status: str = "in_progress",
        started_at: Optional[str] = None,
        is_sufficient: bool = False,
        is_complete: bool = False,
        sufficiency_notes: Optional[str] = None,
        missing_information: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = started_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO research_sessions (
            session_id, prospect_id, context_id, status, started_at,
            is_sufficient, is_complete, sufficiency_notes, missing_information
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                session_id,
                prospect_id,
                context_id,
                status,
                now,
                1 if is_sufficient else 0,
                1 if is_complete else 0,
                sufficiency_notes,
                missing_information,
            ),
        )
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM research_sessions WHERE session_id = ?;",
            (session_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_session(
        self,
        session_id: str,
        status: Optional[str] = None,
        is_sufficient: Optional[bool] = None,
        is_complete: Optional[bool] = None,
        completed_at: Optional[str] = None,
        sufficiency_notes: Optional[str] = None,
        missing_information: Optional[str] = None,
    ) -> bool:
        clauses = []
        params = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if is_sufficient is not None:
            clauses.append("is_sufficient = ?")
            params.append(1 if is_sufficient else 0)
        if is_complete is not None:
            clauses.append("is_complete = ?")
            params.append(1 if is_complete else 0)
        if completed_at is not None:
            clauses.append("completed_at = ?")
            params.append(completed_at)
        if sufficiency_notes is not None:
            clauses.append("sufficiency_notes = ?")
            params.append(sufficiency_notes)
        if missing_information is not None:
            clauses.append("missing_information = ?")
            params.append(missing_information)

        if not clauses:
            return False

        params.append(session_id)
        sql = f"UPDATE research_sessions SET {', '.join(clauses)} WHERE session_id = ?;"
        cursor = self.conn.execute(sql, tuple(params))
        return cursor.rowcount > 0

    def list_sessions(self, prospect_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if prospect_id:
            cursor = self.conn.execute(
                "SELECT * FROM research_sessions WHERE prospect_id = ? ORDER BY started_at DESC;",
                (prospect_id,),
            )
        else:
            cursor = self.conn.execute("SELECT * FROM research_sessions ORDER BY started_at DESC;")
        return [dict(r) for r in cursor.fetchall()]

    # Objectives
    def create_objective(
        self,
        objective_id: str,
        session_id: str,
        type: str,
        target_entity: str,
        description: str,
        temporal_scope: Optional[str] = None,
        priority: int = 5,
        is_critical: bool = True,
    ) -> Dict[str, Any]:
        sql = """
        INSERT INTO research_objectives (
            objective_id, session_id, type, target_entity, description,
            temporal_scope, priority, is_critical
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                objective_id,
                session_id,
                type,
                target_entity,
                description,
                temporal_scope,
                priority,
                1 if is_critical else 0,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM research_objectives WHERE objective_id = ?;",
            (objective_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_objectives(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM research_objectives WHERE session_id = ? ORDER BY priority ASC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Queries
    def create_query(
        self,
        query_id: str,
        session_id: str,
        query_text: str,
        objective_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO research_queries (query_id, session_id, objective_id, query_text, created_at)
        VALUES (?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, (query_id, session_id, objective_id, query_text, now))
        cursor = self.conn.execute(
            "SELECT * FROM research_queries WHERE query_id = ?;",
            (query_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_queries(
        self,
        session_id: Optional[str] = None,
        objective_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if objective_id:
            conditions.append("objective_id = ?")
            params.append(objective_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM research_queries {where} ORDER BY created_at ASC;"
        cursor = self.conn.execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]

    # Search executions
    def record_search_execution(
        self,
        execution_id: str,
        query_id: str,
        provider: str,
        status: str,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        executed_at: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = executed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO search_executions (
            execution_id, query_id, provider, status, error_message,
            duration_ms, executed_at, idempotency_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                execution_id,
                query_id,
                provider,
                status,
                error_message,
                duration_ms,
                now,
                idempotency_key,
            ),
        )
        return self.get_search_execution(execution_id) or {}

    def get_search_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM search_executions WHERE execution_id = ?;",
            (execution_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_search_execution_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM search_executions WHERE idempotency_key = ?;",
            (idempotency_key,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # Search result occurrences
    def record_search_result_occurrence(
        self,
        occurrence_id: str,
        execution_id: str,
        source_id: str,
        position: int,
        original_url: str,
        title: Optional[str] = None,
        snippet: Optional[str] = None,
    ) -> Dict[str, Any]:
        sql = """
        INSERT INTO search_result_occurrences (
            occurrence_id, execution_id, source_id, position, title, snippet, original_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (occurrence_id, execution_id, source_id, position, title, snippet, original_url),
        )
        cursor = self.conn.execute(
            "SELECT * FROM search_result_occurrences WHERE occurrence_id = ?;",
            (occurrence_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_occurrences_for_execution(self, execution_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM search_result_occurrences WHERE execution_id = ? ORDER BY position ASC;",
            (execution_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_occurrences_for_source(self, source_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM search_result_occurrences WHERE source_id = ?;",
            (source_id,),
        )
        return [dict(r) for r in cursor.fetchall()]
