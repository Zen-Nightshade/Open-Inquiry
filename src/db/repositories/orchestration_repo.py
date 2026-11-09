from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


class OrchestrationRepository:
    """Repository managing research iterations and sufficiency assessments."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    # Iterations
    def create_iteration(
        self,
        iteration_id: str,
        session_id: str,
        iteration_number: int,
        status: str = "in_progress",
        queries_executed: int = 0,
        snapshots_fetched: int = 0,
        facts_extracted: int = 0,
        conflicts_detected: int = 0,
        started_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = started_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO research_iterations (
            iteration_id, session_id, iteration_number, status,
            queries_executed, snapshots_fetched, facts_extracted,
            conflicts_detected, started_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                iteration_id,
                session_id,
                iteration_number,
                status,
                queries_executed,
                snapshots_fetched,
                facts_extracted,
                conflicts_detected,
                now,
                error_message,
            ),
        )
        return self.get_iteration(iteration_id) or {}

    def get_iteration(self, iteration_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM research_iterations WHERE iteration_id = ?;",
            (iteration_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_iteration_by_number(self, session_id: str, iteration_number: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM research_iterations WHERE session_id = ? AND iteration_number = ?;",
            (session_id, iteration_number),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_latest_iteration(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM research_iterations WHERE session_id = ? ORDER BY iteration_number DESC LIMIT 1;",
            (session_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_iterations(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM research_iterations WHERE session_id = ? ORDER BY iteration_number ASC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def update_iteration(
        self,
        iteration_id: str,
        status: Optional[str] = None,
        queries_executed: Optional[int] = None,
        snapshots_fetched: Optional[int] = None,
        facts_extracted: Optional[int] = None,
        conflicts_detected: Optional[int] = None,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        clauses = []
        params = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if queries_executed is not None:
            clauses.append("queries_executed = ?")
            params.append(queries_executed)
        if snapshots_fetched is not None:
            clauses.append("snapshots_fetched = ?")
            params.append(snapshots_fetched)
        if facts_extracted is not None:
            clauses.append("facts_extracted = ?")
            params.append(facts_extracted)
        if conflicts_detected is not None:
            clauses.append("conflicts_detected = ?")
            params.append(conflicts_detected)
        if completed_at is not None:
            clauses.append("completed_at = ?")
            params.append(completed_at)
        if error_message is not None:
            clauses.append("error_message = ?")
            params.append(error_message)

        if not clauses:
            return False

        params.append(iteration_id)
        sql = f"UPDATE research_iterations SET {', '.join(clauses)} WHERE iteration_id = ?;"
        cursor = self.conn.execute(sql, tuple(params))
        return cursor.rowcount > 0

    # Assessments
    def create_assessment(
        self,
        assessment_id: str,
        session_id: str,
        iteration_id: str,
        decision: str,
        coverage_status: str,
        identity_status: str,
        evidence_status: str,
        marginal_value_status: str,
        explanation: str,
        objective_details_json: Union[str, Any],
        remaining_gaps_json: Union[str, Any],
        recommended_actions_json: Union[str, Any],
        dimension_details_json: Union[str, Any],
        assessed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = assessed_at or datetime.now(timezone.utc).isoformat()
        obj_json = objective_details_json if isinstance(objective_details_json, str) else json.dumps(objective_details_json)
        gaps_json = remaining_gaps_json if isinstance(remaining_gaps_json, str) else json.dumps(remaining_gaps_json)
        actions_json = recommended_actions_json if isinstance(recommended_actions_json, str) else json.dumps(recommended_actions_json)
        dim_json = dimension_details_json if isinstance(dimension_details_json, str) else json.dumps(dimension_details_json)

        sql = """
        INSERT INTO sufficiency_assessments (
            assessment_id, session_id, iteration_id, decision,
            coverage_status, identity_status, evidence_status, marginal_value_status,
            explanation, objective_details_json, remaining_gaps_json,
            recommended_actions_json, dimension_details_json, assessed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                assessment_id,
                session_id,
                iteration_id,
                decision,
                coverage_status,
                identity_status,
                evidence_status,
                marginal_value_status,
                explanation,
                obj_json,
                gaps_json,
                actions_json,
                dim_json,
                now,
            ),
        )
        return self.get_assessment(assessment_id) or {}

    def get_assessment(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM sufficiency_assessments WHERE assessment_id = ?;",
            (assessment_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_assessment_for_iteration(self, iteration_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM sufficiency_assessments WHERE iteration_id = ?;",
            (iteration_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_latest_assessment(self, session_id: str) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT sa.* FROM sufficiency_assessments sa
        JOIN research_iterations ri ON sa.session_id = ri.session_id AND sa.iteration_id = ri.iteration_id
        WHERE sa.session_id = ?
        ORDER BY ri.iteration_number DESC
        LIMIT 1;
        """
        cursor = self.conn.execute(sql, (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_assessments(self, session_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT sa.*, ri.iteration_number FROM sufficiency_assessments sa
        JOIN research_iterations ri ON sa.session_id = ri.session_id AND sa.iteration_id = ri.iteration_id
        WHERE sa.session_id = ?
        ORDER BY ri.iteration_number ASC;
        """
        cursor = self.conn.execute(sql, (session_id,))
        return [dict(r) for r in cursor.fetchall()]
