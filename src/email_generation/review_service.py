from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.db.state_store import StateStore
from src.email_generation.edit_distance import compute_draft_edit_distance
from src.email_generation.review_models import (
    ConcurrencyConflictError,
    DraftNotFoundError,
    DraftReviewHistory,
    DraftRevision,
    EditDistanceStats,
    InvalidReviewActionError,
    InvalidReviewTransitionError,
    RejectionReason,
    ReviewAction,
    ReviewEvent,
    ReviewerActivity,
    ReviewTelemetrySummary,
)


def compute_percentile(sorted_values: List[float], p: float) -> float:
    """Compute percentile using deterministic linear interpolation (0.0 <= p <= 100.0)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])

    index = (p / 100.0) * (n - 1)
    k = int(index)
    f = index - k
    if k >= n - 1:
        return float(sorted_values[-1])
    return float(sorted_values[k] + f * (sorted_values[k + 1] - sorted_values[k]))


class ReviewService:
    """Domain service managing the human review lifecycle, revision increments, and telemetry."""

    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store

    def list_drafts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve paginated outreach drafts with optional status filtering."""
        with self.state_store.uow() as uow:
            return uow.outreach.list_outreach_drafts_paginated(limit=limit, offset=offset, status=status)

    def get_draft(self, draft_id: str) -> Dict[str, Any]:
        """Fetch raw outreach draft metadata by ID."""
        with self.state_store.uow() as uow:
            draft = uow.outreach.get_outreach_draft(draft_id)
            if not draft:
                raise DraftNotFoundError(f"Draft '{draft_id}' not found")
            return draft

    def get_draft_history(self, draft_id: str) -> DraftReviewHistory:
        """Construct full chronological revision history and logged review events for a draft."""
        with self.state_store.uow() as uow:
            draft = uow.outreach.get_outreach_draft(draft_id)
            if not draft:
                raise DraftNotFoundError(f"Draft '{draft_id}' not found")

            events_raw = uow.outreach.get_human_review_events(draft_id)

        events: List[ReviewEvent] = []
        for e in events_raw:
            events.append(
                ReviewEvent(
                    event_id=e["event_id"],
                    draft_id=e["draft_id"],
                    action=ReviewAction(e["action"]),
                    reviewed_revision=e["reviewed_revision"],
                    resulting_revision=e["resulting_revision"],
                    reviewer_id=e.get("reviewer_id"),
                    notes=e.get("notes"),
                    reason_code=RejectionReason(e["reason_code"]) if e.get("reason_code") else None,
                    original_subject=e.get("original_subject"),
                    original_body=e.get("original_body"),
                    resulting_subject=e.get("resulting_subject"),
                    resulting_body=e.get("resulting_body"),
                    char_edit_distance=e.get("char_edit_distance"),
                    word_edit_distance=e.get("word_edit_distance"),
                    normalized_char_ratio=e.get("normalized_char_ratio"),
                    normalized_word_ratio=e.get("normalized_word_ratio"),
                    timestamp=e["decided_at"],
                )
            )

        first_edit = next(
            (ev for ev in events if ev.action == ReviewAction.EDITED and ev.reviewed_revision == 1),
            None,
        )
        initial_sub = first_edit.original_subject if first_edit and first_edit.original_subject else draft["subject"]
        initial_body = first_edit.original_body if first_edit and first_edit.original_body else draft["body"]

        revisions: List[DraftRevision] = [
            DraftRevision(
                revision_number=1,
                subject=initial_sub,
                body=initial_body,
                word_count=len(initial_body.split()),
                created_at=draft["created_at"],
                action_leading_to_revision=None,
            )
        ]

        for ev in events:
            if ev.action == ReviewAction.EDITED:
                revisions.append(
                    DraftRevision(
                        revision_number=ev.resulting_revision,
                        subject=ev.resulting_subject or "",
                        body=ev.resulting_body or "",
                        word_count=len((ev.resulting_body or "").split()),
                        created_at=ev.timestamp,
                        action_leading_to_revision=ev.action,
                        reviewer_id=ev.reviewer_id,
                        notes=ev.notes,
                    )
                )

        return DraftReviewHistory(
            draft_id=draft["draft_id"],
            context_id=draft["context_id"],
            current_revision=draft.get("current_revision", 1),
            status=draft["human_review_status"],
            initial_subject=initial_sub,
            initial_body=initial_body,
            current_subject=draft["subject"],
            current_body=draft["body"],
            created_at=draft["created_at"],
            reviewer_notes=draft.get("reviewer_notes"),
            revisions=revisions,
            events=events,
        )

    def edit_draft(
        self,
        draft_id: str,
        new_subject: str,
        new_body: str,
        expected_revision: int,
        reviewer_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ReviewEvent:
        """Record an edit action, increment revision, update draft text, and log edit distance metrics."""
        if not new_subject or not new_subject.strip():
            raise InvalidReviewActionError("new_subject must not be empty")
        if not new_body or not new_body.strip():
            raise InvalidReviewActionError("new_body must not be empty")

        new_word_count = len(new_body.split())
        event_id = str(uuid.uuid4())

        with self.state_store.uow() as uow:
            draft = uow.outreach.get_outreach_draft(draft_id)
            if not draft:
                raise DraftNotFoundError(f"Draft '{draft_id}' not found")

            if draft["human_review_status"] != "pending_review":
                raise InvalidReviewTransitionError(
                    f"Cannot edit draft '{draft_id}' with finalized status '{draft['human_review_status']}'"
                )

            orig_subject = draft["subject"]
            orig_body = draft["body"]
            metrics = compute_draft_edit_distance(
                original_subject=orig_subject,
                original_body=orig_body,
                resulting_subject=new_subject,
                resulting_body=new_body,
            )

            success = uow.outreach.cas_update_draft_edit(
                draft_id=draft_id,
                expected_revision=expected_revision,
                new_subject=new_subject,
                new_body=new_body,
                new_word_count=new_word_count,
                reviewer_notes=notes,
            )
            if not success:
                current_rev = draft.get("current_revision", 1)
                raise ConcurrencyConflictError(
                    f"Draft revision conflict on '{draft_id}': expected revision {expected_revision}, "
                    f"but actual revision is {current_rev}"
                )

            event_dict = uow.outreach.record_human_review_event(
                event_id=event_id,
                draft_id=draft_id,
                action="EDITED",
                reviewed_revision=expected_revision,
                resulting_revision=expected_revision + 1,
                reviewer_id=reviewer_id,
                notes=notes,
                original_subject=orig_subject,
                original_body=orig_body,
                resulting_subject=new_subject,
                resulting_body=new_body,
                char_edit_distance=metrics.char_distance,
                word_edit_distance=metrics.word_distance,
                normalized_char_ratio=metrics.normalized_char_ratio,
                normalized_word_ratio=metrics.normalized_word_ratio,
            )

            return ReviewEvent(
                event_id=event_dict["event_id"],
                draft_id=event_dict["draft_id"],
                action=ReviewAction.EDITED,
                reviewed_revision=event_dict["reviewed_revision"],
                resulting_revision=event_dict["resulting_revision"],
                reviewer_id=event_dict.get("reviewer_id"),
                notes=event_dict.get("notes"),
                original_subject=event_dict.get("original_subject"),
                original_body=event_dict.get("original_body"),
                resulting_subject=event_dict.get("resulting_subject"),
                resulting_body=event_dict.get("resulting_body"),
                char_edit_distance=event_dict.get("char_edit_distance"),
                word_edit_distance=event_dict.get("word_edit_distance"),
                normalized_char_ratio=event_dict.get("normalized_char_ratio"),
                normalized_word_ratio=event_dict.get("normalized_word_ratio"),
                timestamp=event_dict["decided_at"],
            )

    def approve_draft(
        self,
        draft_id: str,
        expected_revision: int,
        reviewer_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ReviewEvent:
        """Record an approval decision, transition status to 'approved', preserving current revision."""
        event_id = str(uuid.uuid4())

        with self.state_store.uow() as uow:
            draft = uow.outreach.get_outreach_draft(draft_id)
            if not draft:
                raise DraftNotFoundError(f"Draft '{draft_id}' not found")

            if draft["human_review_status"] != "pending_review":
                raise InvalidReviewTransitionError(
                    f"Cannot approve draft '{draft_id}' with finalized status '{draft['human_review_status']}'"
                )

            success = uow.outreach.cas_update_draft_status(
                draft_id=draft_id,
                expected_revision=expected_revision,
                new_status="approved",
                reviewer_notes=notes,
            )
            if not success:
                current_rev = draft.get("current_revision", 1)
                raise ConcurrencyConflictError(
                    f"Draft revision conflict on '{draft_id}': expected revision {expected_revision}, "
                    f"but actual revision is {current_rev}"
                )

            event_dict = uow.outreach.record_human_review_event(
                event_id=event_id,
                draft_id=draft_id,
                action="APPROVED",
                reviewed_revision=expected_revision,
                resulting_revision=expected_revision,
                reviewer_id=reviewer_id,
                notes=notes,
            )

            return ReviewEvent(
                event_id=event_dict["event_id"],
                draft_id=event_dict["draft_id"],
                action=ReviewAction.APPROVED,
                reviewed_revision=event_dict["reviewed_revision"],
                resulting_revision=event_dict["resulting_revision"],
                reviewer_id=event_dict.get("reviewer_id"),
                notes=event_dict.get("notes"),
                timestamp=event_dict["decided_at"],
            )

    def reject_draft(
        self,
        draft_id: str,
        expected_revision: int,
        reason_code: RejectionReason | str,
        reviewer_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ReviewEvent:
        """Record a rejection decision with mandatory reason code, transitioning status to 'rejected'."""
        if not reason_code:
            raise InvalidReviewActionError("reason_code is required when rejecting a draft")

        if isinstance(reason_code, str):
            try:
                reason_enum = RejectionReason(reason_code)
            except ValueError:
                raise InvalidReviewActionError(f"Invalid rejection reason: {reason_code}")
        else:
            reason_enum = reason_code

        event_id = str(uuid.uuid4())

        with self.state_store.uow() as uow:
            draft = uow.outreach.get_outreach_draft(draft_id)
            if not draft:
                raise DraftNotFoundError(f"Draft '{draft_id}' not found")

            if draft["human_review_status"] != "pending_review":
                raise InvalidReviewTransitionError(
                    f"Cannot reject draft '{draft_id}' with finalized status '{draft['human_review_status']}'"
                )

            success = uow.outreach.cas_update_draft_status(
                draft_id=draft_id,
                expected_revision=expected_revision,
                new_status="rejected",
                reviewer_notes=notes,
            )
            if not success:
                current_rev = draft.get("current_revision", 1)
                raise ConcurrencyConflictError(
                    f"Draft revision conflict on '{draft_id}': expected revision {expected_revision}, "
                    f"but actual revision is {current_rev}"
                )

            event_dict = uow.outreach.record_human_review_event(
                event_id=event_id,
                draft_id=draft_id,
                action="REJECTED",
                reviewed_revision=expected_revision,
                resulting_revision=expected_revision,
                reviewer_id=reviewer_id,
                notes=notes,
                reason_code=reason_enum.value,
            )

            return ReviewEvent(
                event_id=event_dict["event_id"],
                draft_id=event_dict["draft_id"],
                action=ReviewAction.REJECTED,
                reviewed_revision=event_dict["reviewed_revision"],
                resulting_revision=event_dict["resulting_revision"],
                reviewer_id=event_dict.get("reviewer_id"),
                notes=event_dict.get("notes"),
                reason_code=reason_enum,
                timestamp=event_dict["decided_at"],
            )

    def get_telemetry(self) -> ReviewTelemetrySummary:
        """Aggregate metrics, rejection breakdown, edit distance statistics, and reviewer activity."""
        with self.state_store.uow() as uow:
            events = uow.outreach.get_human_review_events()

        total_review_events = len(events)
        distinct_draft_ids = set(e["draft_id"] for e in events)
        total_drafts_reviewed = len(distinct_draft_ids)

        drafts_with_edits = set(e["draft_id"] for e in events if e["action"] == "EDITED")
        drafts_with_edits_count = len(drafts_with_edits)
        edited_count = sum(1 for e in events if e["action"] == "EDITED")

        approved_drafts = set(e["draft_id"] for e in events if e["action"] == "APPROVED")
        rejected_drafts = set(e["draft_id"] for e in events if e["action"] == "REJECTED")
        final_decision_drafts = approved_drafts | rejected_drafts
        final_decision_count = len(final_decision_drafts)
        approved_count = len(approved_drafts)
        rejected_count = len(rejected_drafts)

        approval_rate = (approved_count / final_decision_count) if final_decision_count > 0 else 0.0
        rejection_rate = (rejected_count / final_decision_count) if final_decision_count > 0 else 0.0
        edit_frequency = (drafts_with_edits_count / total_drafts_reviewed) if total_drafts_reviewed > 0 else 0.0

        # Rejection reasons breakdown
        reasons_breakdown: Dict[str, int] = {}
        for e in events:
            if e["action"] == "REJECTED":
                code = e.get("reason_code") or "OTHER"
                reasons_breakdown[code] = reasons_breakdown.get(code, 0) + 1

        # Edit distance statistics strictly over EDITED events
        edit_events = [e for e in events if e["action"] == "EDITED"]
        edit_count = len(edit_events)

        char_dists = sorted([float(e["char_edit_distance"]) for e in edit_events if e.get("char_edit_distance") is not None])
        word_dists = sorted([float(e["word_edit_distance"]) for e in edit_events if e.get("word_edit_distance") is not None])
        norm_char_ratios = sorted([float(e["normalized_char_ratio"]) for e in edit_events if e.get("normalized_char_ratio") is not None])
        norm_word_ratios = sorted([float(e["normalized_word_ratio"]) for e in edit_events if e.get("normalized_word_ratio") is not None])

        def _stats(arr: List[float]) -> Tuple[float, float, float]:
            if not arr:
                return 0.0, 0.0, 0.0
            mean = sum(arr) / len(arr)
            median = compute_percentile(arr, 50.0)
            p90 = compute_percentile(arr, 90.0)
            return round(mean, 4), round(median, 4), round(p90, 4)

        mean_c, med_c, p90_c = _stats(char_dists)
        mean_w, med_w, p90_w = _stats(word_dists)
        mean_nc, med_nc, p90_nc = _stats(norm_char_ratios)
        mean_nw, med_nw, p90_nw = _stats(norm_word_ratios)

        edit_stats = EditDistanceStats(
            count=edit_count,
            mean_char_distance=mean_c,
            median_char_distance=med_c,
            p90_char_distance=p90_c,
            mean_word_distance=mean_w,
            median_word_distance=med_w,
            p90_word_distance=p90_w,
            mean_normalized_char_ratio=mean_nc,
            median_normalized_char_ratio=med_nc,
            p90_normalized_char_ratio=p90_nc,
            mean_normalized_word_ratio=mean_nw,
            median_normalized_word_ratio=med_nw,
            p90_normalized_word_ratio=p90_nw,
        )

        # Reviewer activity grouped by reviewer identity
        reviewer_map: Dict[str, Dict[str, int]] = {}
        for e in events:
            rev_id = e.get("reviewer_id") or "anonymous"
            if rev_id not in reviewer_map:
                reviewer_map[rev_id] = {"total": 0, "approved": 0, "rejected": 0, "edited": 0}
            reviewer_map[rev_id]["total"] += 1
            act = e["action"]
            if act == "APPROVED":
                reviewer_map[rev_id]["approved"] += 1
            elif act == "REJECTED":
                reviewer_map[rev_id]["rejected"] += 1
            elif act == "EDITED":
                reviewer_map[rev_id]["edited"] += 1

        reviewer_activity = {
            rev_id: ReviewerActivity(
                reviewer_id=rev_id,
                total_actions=data["total"],
                approvals=data["approved"],
                rejections=data["rejected"],
                edits=data["edited"],
            )
            for rev_id, data in reviewer_map.items()
        }

        return ReviewTelemetrySummary(
            total_drafts_reviewed=total_drafts_reviewed,
            total_review_events=total_review_events,
            final_decision_drafts_count=final_decision_count,
            approved_count=approved_count,
            rejected_count=rejected_count,
            edited_count=edited_count,
            drafts_with_edits_count=drafts_with_edits_count,
            approval_rate=round(approval_rate, 4),
            rejection_rate=round(rejection_rate, 4),
            edit_frequency=round(edit_frequency, 4),
            percentile_method="linear_interpolation",
            rejection_reasons_breakdown=reasons_breakdown,
            edit_distance_stats=edit_stats,
            reviewer_activity=reviewer_activity,
        )
