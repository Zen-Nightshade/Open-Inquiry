from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AssertionRepository:
    """Repository managing canonical facts, entity links, evidence links, conflicts, and assessment history."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    # Canonical facts
    def create_canonical_fact(
        self,
        fact_id: str,
        session_id: str,
        claim_text: str,
        fact_category: str = "other",
        primary_event_date: Optional[str] = None,
        composite_confidence: str = "medium",
        molecular_data_json: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO canonical_facts (
            fact_id, session_id, claim_text, fact_category,
            primary_event_date, composite_confidence, molecular_data_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fact_id) DO UPDATE SET
            composite_confidence = excluded.composite_confidence,
            primary_event_date = coalesce(canonical_facts.primary_event_date, excluded.primary_event_date),
            molecular_data_json = coalesce(excluded.molecular_data_json, canonical_facts.molecular_data_json);
        """
        self.conn.execute(
            sql,
            (
                fact_id,
                session_id,
                claim_text,
                fact_category,
                primary_event_date,
                composite_confidence,
                molecular_data_json,
                now,
            ),
        )
        return self.get_canonical_fact(fact_id) or {}

    def get_canonical_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM canonical_facts WHERE fact_id = ?;",
            (fact_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_facts_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM canonical_facts WHERE session_id = ? ORDER BY created_at ASC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Fact-Entity links
    def link_fact_entity(self, fact_id: str, entity_id: str) -> bool:
        sql = """
        INSERT INTO fact_entity_links (fact_id, entity_id)
        VALUES (?, ?)
        ON CONFLICT(fact_id, entity_id) DO NOTHING;
        """
        cursor = self.conn.execute(sql, (fact_id, entity_id))
        return cursor.rowcount > 0

    def get_entities_for_fact(self, fact_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT e.* FROM entities e
        JOIN fact_entity_links l ON e.entity_id = l.entity_id
        WHERE l.fact_id = ?;
        """
        cursor = self.conn.execute(sql, (fact_id,))
        return [dict(r) for r in cursor.fetchall()]

    def get_facts_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT f.* FROM canonical_facts f
        JOIN fact_entity_links l ON f.fact_id = l.fact_id
        WHERE l.entity_id = ?;
        """
        cursor = self.conn.execute(sql, (entity_id,))
        return [dict(r) for r in cursor.fetchall()]

    def get_prospect_facts(self, session_id: str, prospect_entity_id: str) -> List[Dict[str, Any]]:
        """Retrieve facts attributed specifically to the resolved target prospect entity."""
        sql = """
        SELECT DISTINCT f.* FROM canonical_facts f
        JOIN fact_entity_links fel ON f.fact_id = fel.fact_id
        JOIN entities e ON fel.entity_id = e.entity_id
        WHERE f.session_id = ?
          AND e.entity_id = ?
          AND e.resolution_status = 'resolved';
        """
        cursor = self.conn.execute(sql, (session_id, prospect_entity_id))
        return [dict(r) for r in cursor.fetchall()]

    def get_candidate_facts(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve candidate-level facts linked to unresolved, ambiguous, or partially resolved entities."""
        sql = """
        SELECT DISTINCT f.* FROM canonical_facts f
        JOIN fact_entity_links fel ON f.fact_id = fel.fact_id
        JOIN entities e ON fel.entity_id = e.entity_id
        WHERE f.session_id = ?
          AND e.resolution_status IN ('unresolved', 'ambiguous', 'partially_resolved');
        """
        cursor = self.conn.execute(sql, (session_id,))
        return [dict(r) for r in cursor.fetchall()]

    # Fact-Evidence links
    def link_fact_evidence(
        self,
        fact_id: str,
        evidence_id: str,
        support_type: str = "supporting",
    ) -> bool:
        sql = """
        INSERT INTO fact_evidence_links (fact_id, evidence_id, support_type)
        VALUES (?, ?, ?)
        ON CONFLICT(fact_id, evidence_id) DO UPDATE SET support_type = excluded.support_type;
        """
        cursor = self.conn.execute(sql, (fact_id, evidence_id, support_type))
        return cursor.rowcount > 0

    def get_evidence_for_fact(self, fact_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT ep.*, fel.support_type FROM evidence_passages ep
        JOIN fact_evidence_links fel ON ep.evidence_id = fel.evidence_id
        WHERE fel.fact_id = ?;
        """
        cursor = self.conn.execute(sql, (fact_id,))
        return [dict(r) for r in cursor.fetchall()]

    # Conflicts
    def create_conflict(
        self,
        conflict_id: str,
        session_id: str,
        description: str,
        status: str = "open",
        severity: str = "unknown",
        resolved_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO conflicts (conflict_id, session_id, description, status, severity, resolved_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, (conflict_id, session_id, description, status, severity, resolved_at, now))
        return self.get_conflict(conflict_id) or {}

    def get_conflict(self, conflict_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM conflicts WHERE conflict_id = ?;",
            (conflict_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_conflicts_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM conflicts WHERE session_id = ? ORDER BY created_at ASC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def link_conflict_fact(self, conflict_id: str, fact_id: str) -> bool:
        sql = """
        INSERT INTO conflict_facts (conflict_id, fact_id)
        VALUES (?, ?)
        ON CONFLICT(conflict_id, fact_id) DO NOTHING;
        """
        cursor = self.conn.execute(sql, (conflict_id, fact_id))
        return cursor.rowcount > 0

    def get_conflict_facts(self, conflict_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT f.* FROM canonical_facts f
        JOIN conflict_facts cf ON f.fact_id = cf.fact_id
        WHERE cf.conflict_id = ?;
        """
        cursor = self.conn.execute(sql, (conflict_id,))
        return [dict(r) for r in cursor.fetchall()]

    # Assessment history
    def record_assessment_history(
        self,
        assessment_id: str,
        session_id: str,
        iteration: int,
        assessor: str,
        finding_type: str,
        details: str,
        assessed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = assessed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO assessment_history (
            assessment_id, session_id, iteration, assessor,
            finding_type, details, assessed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (assessment_id, session_id, iteration, assessor, finding_type, details, now),
        )
        cursor = self.conn.execute(
            "SELECT * FROM assessment_history WHERE assessment_id = ?;",
            (assessment_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_assessment_history(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM assessment_history WHERE session_id = ? ORDER BY iteration ASC, assessed_at ASC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Resolution decisions
    def record_resolution_decision(
        self,
        decision_id: str,
        conflict_id: str,
        rationale: str,
        decided_by: str,
        decided_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = decided_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO resolution_decisions (decision_id, conflict_id, rationale, decided_by, decided_at)
        VALUES (?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, (decision_id, conflict_id, rationale, decided_by, now))
        # Also update conflict status to resolved
        self.conn.execute(
            "UPDATE conflicts SET status = 'resolved', resolved_at = ? WHERE conflict_id = ?;",
            (now, conflict_id),
        )
        cursor = self.conn.execute(
            "SELECT * FROM resolution_decisions WHERE decision_id = ?;",
            (decision_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_resolution_decision(self, conflict_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM resolution_decisions WHERE conflict_id = ?;",
            (conflict_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
