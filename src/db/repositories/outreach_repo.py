from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class OutreachRepository:
    """Repository managing prospects, outreach contexts, drafts, and human review events."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    def create_prospect(
        self,
        prospect_id: str,
        full_name: str,
        company_name: str,
        company_domain: Optional[str] = None,
        designation: Optional[str] = None,
        linkedin_url: Optional[str] = None,
        location: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO prospects (
            prospect_id, full_name, company_name, company_domain,
            designation, linkedin_url, location, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                prospect_id,
                full_name,
                company_name,
                company_domain,
                designation,
                linkedin_url,
                location,
                now,
            ),
        )
        return self.get_prospect(prospect_id) or {}

    def get_prospect(self, prospect_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM prospects WHERE prospect_id = ?;",
            (prospect_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_prospects(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        if include_deleted:
            sql = "SELECT * FROM prospects ORDER BY created_at DESC;"
            cursor = self.conn.execute(sql)
        else:
            sql = "SELECT * FROM prospects WHERE deleted_at IS NULL ORDER BY created_at DESC;"
            cursor = self.conn.execute(sql)
        return [dict(r) for r in cursor.fetchall()]

    def soft_delete_prospect(
        self,
        prospect_id: str,
        deleted_at: Optional[str] = None,
    ) -> bool:
        now = deleted_at or datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "UPDATE prospects SET deleted_at = ? WHERE prospect_id = ? AND deleted_at IS NULL;",
            (now, prospect_id),
        )
        return cursor.rowcount > 0

    def create_outreach_context(
        self,
        context_id: str,
        prospect_id: str,
        version_id: str,
        context_json: str,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO outreach_contexts (
            context_id, prospect_id, version_id, context_json, created_at
        ) VALUES (?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, (context_id, prospect_id, version_id, context_json, now))
        return self.get_outreach_context(context_id) or {}

    def get_outreach_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM outreach_contexts WHERE context_id = ?;",
            (context_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_outreach_draft(
        self,
        draft_id: str,
        context_id: str,
        research_session_id: str,
        subject: str,
        body: str,
        word_count: int,
        product_snapshot_json: str,
        claim_verifications: str,
        quality_evaluation: str,
        generation_attempt: int = 1,
        human_review_status: str = "pending_review",
        reviewer_notes: Optional[str] = None,
        current_revision: int = 1,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO outreach_drafts (
            draft_id, context_id, research_session_id, subject, body,
            word_count, generation_attempt, product_snapshot_json,
            claim_verifications, quality_evaluation, human_review_status,
            reviewer_notes, current_revision, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                draft_id,
                context_id,
                research_session_id,
                subject,
                body,
                word_count,
                generation_attempt,
                product_snapshot_json,
                claim_verifications,
                quality_evaluation,
                human_review_status,
                reviewer_notes,
                current_revision,
                now,
            ),
        )
        return self.get_outreach_draft(draft_id) or {}

    def get_outreach_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM outreach_drafts WHERE draft_id = ?;",
            (draft_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_outreach_drafts(
        self,
        context_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if context_id is not None:
            conditions.append("context_id = ?")
            params.append(context_id)
        if not include_deleted:
            conditions.append("deleted_at IS NULL")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM outreach_drafts {where} ORDER BY created_at DESC;"
        cursor = self.conn.execute(sql, tuple(params))
        return [dict(r) for r in cursor.fetchall()]

    def list_outreach_drafts_paginated(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        context_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query drafts with deterministic pagination (created_at ASC, draft_id ASC) and optional status filtering."""
        conditions = []
        params: List[Any] = []
        if status is not None:
            conditions.append("human_review_status = ?")
            params.append(status)
        if context_id is not None:
            conditions.append("context_id = ?")
            params.append(context_id)
        if not include_deleted:
            conditions.append("deleted_at IS NULL")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM outreach_drafts {where};"
        cursor = self.conn.execute(count_sql, tuple(params))
        total_count = cursor.fetchone()[0]

        sql = f"""
        SELECT * FROM outreach_drafts
        {where}
        ORDER BY created_at ASC, draft_id ASC
        LIMIT ? OFFSET ?;
        """
        cursor = self.conn.execute(sql, tuple(params + [limit, offset]))
        rows = [dict(r) for r in cursor.fetchall()]
        return rows, total_count

    def update_outreach_draft_review(
        self,
        draft_id: str,
        human_review_status: str,
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        sql = """
        UPDATE outreach_drafts
        SET human_review_status = ?, reviewer_notes = COALESCE(?, reviewer_notes)
        WHERE draft_id = ?;
        """
        cursor = self.conn.execute(sql, (human_review_status, reviewer_notes, draft_id))
        return cursor.rowcount > 0

    def cas_update_draft_edit(
        self,
        draft_id: str,
        expected_revision: int,
        new_subject: str,
        new_body: str,
        new_word_count: int,
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        """Optimistic concurrency compare-and-swap update for draft edits.
        
        Increments current_revision by 1 while keeping status as pending_review.
        Returns True if exactly one row was updated; False otherwise.
        """
        sql = """
        UPDATE outreach_drafts
        SET subject = ?,
            body = ?,
            word_count = ?,
            current_revision = current_revision + 1,
            reviewer_notes = COALESCE(?, reviewer_notes)
        WHERE draft_id = ?
          AND current_revision = ?
          AND human_review_status = 'pending_review';
        """
        cursor = self.conn.execute(
            sql,
            (new_subject, new_body, new_word_count, reviewer_notes, draft_id, expected_revision),
        )
        return cursor.rowcount == 1

    def cas_update_draft_status(
        self,
        draft_id: str,
        expected_revision: int,
        new_status: str,
        reviewer_notes: Optional[str] = None,
    ) -> bool:
        """Optimistic concurrency compare-and-swap update for final decision status transitions.
        
        Transitions status to approved or rejected on the expected revision.
        Returns True if exactly one row was updated; False otherwise.
        """
        sql = """
        UPDATE outreach_drafts
        SET human_review_status = ?,
            reviewer_notes = COALESCE(?, reviewer_notes)
        WHERE draft_id = ?
          AND current_revision = ?
          AND human_review_status = 'pending_review';
        """
        cursor = self.conn.execute(
            sql,
            (new_status, reviewer_notes, draft_id, expected_revision),
        )
        return cursor.rowcount == 1

    def record_human_review_event(
        self,
        event_id: str,
        draft_id: str,
        action: str = "APPROVED",
        decision: Optional[str] = None,
        reviewed_revision: int = 1,
        resulting_revision: int = 1,
        reviewer_id: Optional[str] = None,
        notes: Optional[str] = None,
        reason_code: Optional[str] = None,
        original_subject: Optional[str] = None,
        original_body: Optional[str] = None,
        resulting_subject: Optional[str] = None,
        resulting_body: Optional[str] = None,
        char_edit_distance: Optional[int] = None,
        word_edit_distance: Optional[int] = None,
        normalized_char_ratio: Optional[float] = None,
        normalized_word_ratio: Optional[float] = None,
        decided_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an immutable, append-only human review event."""
        now = decided_at or datetime.now(timezone.utc).isoformat()
        if decision and action == "APPROVED" and decision.lower() in ("approved", "rejected", "needs_revision"):
            if decision.lower() == "rejected":
                action = "REJECTED"
            elif decision.lower() == "needs_revision":
                action = "EDITED"
            else:
                action = "APPROVED"
        decision_val = decision or action.lower()

        sql = """
        INSERT INTO human_review_events (
            event_id, draft_id, action, decision, reviewed_revision, resulting_revision,
            reviewer_id, notes, reason_code, original_subject, original_body,
            resulting_subject, resulting_body, char_edit_distance, word_edit_distance,
            normalized_char_ratio, normalized_word_ratio, decided_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                event_id,
                draft_id,
                action,
                decision_val,
                reviewed_revision,
                resulting_revision,
                reviewer_id,
                notes,
                reason_code,
                original_subject,
                original_body,
                resulting_subject,
                resulting_body,
                char_edit_distance,
                word_edit_distance,
                normalized_char_ratio,
                normalized_word_ratio,
                now,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM human_review_events WHERE event_id = ?;",
            (event_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_human_review_events(self, draft_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve review events ordered chronologically."""
        if draft_id is not None:
            cursor = self.conn.execute(
                "SELECT * FROM human_review_events WHERE draft_id = ? ORDER BY decided_at ASC, event_id ASC;",
                (draft_id,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM human_review_events ORDER BY decided_at ASC, event_id ASC;",
            )
        return [dict(r) for r in cursor.fetchall()]

