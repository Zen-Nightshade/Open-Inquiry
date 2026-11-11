from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class EntityRepository:
    """Repository managing entities, identifiers, aliases, affiliations, relationships, candidates, and assessments."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    # Entities
    def create_entity(
        self,
        entity_id: str,
        entity_type: str,
        raw_name: str,
        canonical_name: Optional[str] = None,
        resolution_status: str = "unresolved",
        confidence: str = "unresolved",
        disambiguation_notes: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        up_now = updated_at or now
        sql = """
        INSERT INTO entities (
            entity_id, entity_type, raw_name, canonical_name,
            resolution_status, confidence, disambiguation_notes,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                entity_id,
                entity_type,
                raw_name,
                canonical_name,
                resolution_status,
                confidence,
                disambiguation_notes,
                now,
                up_now,
            ),
        )
        return self.get_entity(entity_id) or {}

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?;",
            (entity_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_entity(
        self,
        entity_id: str,
        canonical_name: Optional[str] = None,
        resolution_status: Optional[str] = None,
        confidence: Optional[str] = None,
        disambiguation_notes: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> bool:
        clauses = []
        params = []
        if canonical_name is not None:
            clauses.append("canonical_name = ?")
            params.append(canonical_name)
        if resolution_status is not None:
            clauses.append("resolution_status = ?")
            params.append(resolution_status)
        if confidence is not None:
            clauses.append("confidence = ?")
            params.append(confidence)
        if disambiguation_notes is not None:
            clauses.append("disambiguation_notes = ?")
            params.append(disambiguation_notes)

        now = updated_at or datetime.now(timezone.utc).isoformat()
        clauses.append("updated_at = ?")
        params.append(now)

        params.append(entity_id)
        sql = f"UPDATE entities SET {', '.join(clauses)} WHERE entity_id = ?;"
        cursor = self.conn.execute(sql, tuple(params))
        return cursor.rowcount > 0

    # Identifiers
    def add_identifier(
        self,
        identifier_id: str,
        entity_id: str,
        id_type: str,
        id_value: str,
        source_id: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = observed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO entity_identifiers (
            identifier_id, entity_id, id_type, id_value, source_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_id, id_type, id_value) DO NOTHING;
        """
        self.conn.execute(sql, (identifier_id, entity_id, id_type, id_value, source_id, now))
        cursor = self.conn.execute(
            "SELECT * FROM entity_identifiers WHERE entity_id = ? AND id_type = ? AND id_value = ?;",
            (entity_id, id_type, id_value),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_identifiers(self, entity_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM entity_identifiers WHERE entity_id = ?;",
            (entity_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def find_entity_by_identifier(self, id_type: str, id_value: str) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT e.* FROM entities e
        JOIN entity_identifiers i ON e.entity_id = i.entity_id
        WHERE i.id_type = ? AND i.id_value = ?
        LIMIT 1;
        """
        cursor = self.conn.execute(sql, (id_type, id_value))
        row = cursor.fetchone()
        return dict(row) if row else None

    # Aliases
    def add_alias(
        self,
        alias_id: str,
        entity_id: str,
        alias: str,
        source_id: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = observed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO entity_aliases (alias_id, entity_id, alias, source_id, observed_at)
        VALUES (?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, (alias_id, entity_id, alias, source_id, now))
        cursor = self.conn.execute(
            "SELECT * FROM entity_aliases WHERE alias_id = ?;",
            (alias_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_aliases(self, entity_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM entity_aliases WHERE entity_id = ?;",
            (entity_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Affiliations
    def add_affiliation(
        self,
        affiliation_id: str,
        entity_id: str,
        organization_id: Optional[str] = None,
        role: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_id: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = observed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO entity_affiliations (
            affiliation_id, entity_id, organization_id, role,
            start_date, end_date, source_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                affiliation_id,
                entity_id,
                organization_id,
                role,
                start_date,
                end_date,
                source_id,
                now,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM entity_affiliations WHERE affiliation_id = ?;",
            (affiliation_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_affiliations(
        self,
        entity_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        if organization_id:
            conditions.append("organization_id = ?")
            params.append(organization_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM entity_affiliations {where} ORDER BY start_date DESC;"
        cursor = self.conn.execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]

    # Relationships
    def add_relationship(
        self,
        relationship_id: str,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        valid_to: Optional[str] = None,
        source_id: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = observed_at or datetime.now(timezone.utc).isoformat()
        effective_until = valid_to or valid_until
        sql = """
        INSERT INTO entity_relationships (
            relationship_id, from_entity_id, to_entity_id, relationship_type,
            valid_from, valid_until, source_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                relationship_id,
                from_entity_id,
                to_entity_id,
                relationship_type,
                valid_from,
                effective_until,
                source_id,
                now,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM entity_relationships WHERE relationship_id = ?;",
            (relationship_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}
        d = dict(row)
        d["valid_to"] = d.get("valid_until")
        return d

    def get_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT * FROM entity_relationships
        WHERE from_entity_id = ? OR to_entity_id = ?
        ORDER BY observed_at DESC;
        """
        cursor = self.conn.execute(sql, (entity_id, entity_id))
        results = []
        for r in cursor.fetchall():
            d = dict(r)
            d["valid_to"] = d.get("valid_until")
            results.append(d)
        return results

    # Candidates
    def record_candidate(
        self,
        candidate_id: str,
        session_id: str,
        entity_type: str,
        raw_name: str,
        attributes: str,
        source_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO entity_candidates (
            candidate_id, session_id, entity_type, raw_name, attributes, source_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (candidate_id, session_id, entity_type, raw_name, attributes, source_id, now),
        )
        cursor = self.conn.execute(
            "SELECT * FROM entity_candidates WHERE candidate_id = ?;",
            (candidate_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_candidates(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM entity_candidates WHERE session_id = ? ORDER BY created_at ASC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Assessments
    def record_identity_assessment(
        self,
        assessment_id: str,
        session_id: str,
        entity_id: str,
        candidate_ids: str,
        resolution_status: str,
        confidence: str,
        rationale: str,
        assessed_by: str,
        decision_metadata_json: Optional[str] = None,
        assessed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = assessed_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO identity_assessments (
            assessment_id, session_id, entity_id, candidate_ids,
            resolution_status, confidence, rationale, assessed_by,
            decision_metadata_json, assessed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                assessment_id,
                session_id,
                entity_id,
                candidate_ids,
                resolution_status,
                confidence,
                rationale,
                assessed_by,
                decision_metadata_json,
                now,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM identity_assessments WHERE assessment_id = ?;",
            (assessment_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_assessments(
        self,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM identity_assessments {where} ORDER BY assessed_at ASC;"
        cursor = self.conn.execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]
