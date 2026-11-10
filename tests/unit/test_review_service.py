from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.db.state_store import StateStore
from src.email_generation.review_models import (
    ConcurrencyConflictError,
    DraftNotFoundError,
    InvalidReviewActionError,
    InvalidReviewTransitionError,
    RejectionReason,
    ReviewAction,
)
from src.email_generation.review_service import ReviewService, compute_percentile


@pytest.fixture
def test_store():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "review_test.db"
        store = StateStore(db_path=db_path)
        store.initialize()
        yield store


def _create_sample_draft(
    store: StateStore,
    draft_id: str = "draft-1",
    subject: str = "Initial Subject",
    body: str = "Initial draft body content for testing.",
    context_id: Optional[str] = None,
    status: str = "pending_review",
) -> str:
    ctx_id = context_id or f"ctx-{draft_id}"
    with store.uow() as uow:
        # Create prerequisite prospect and context
        uow.outreach.create_prospect(
            prospect_id=f"p-{draft_id}",
            full_name="Jane Doe",
            company_name="Acme Corp",
        )
        uow.products.create_product_version(
            version_id=f"pv-{draft_id}",
            product_name="PulseRadar",
            snapshot_json="{}",
        )
        uow.outreach.create_outreach_context(
            context_id=ctx_id,
            prospect_id=f"p-{draft_id}",
            version_id=f"pv-{draft_id}",
            context_json="{}",
        )
        uow.outreach.create_outreach_draft(
            draft_id=draft_id,
            context_id=ctx_id,
            research_session_id=f"sess-{draft_id}",
            subject=subject,
            body=body,
            word_count=len(body.split()),
            product_snapshot_json="{}",
            claim_verifications="[]",
            quality_evaluation="{}",
            human_review_status=status,
            current_revision=1,
        )
    return draft_id


# 1. Approve Draft
def test_approve_draft(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-approve")

    event = service.approve_draft(
        draft_id=draft_id,
        expected_revision=1,
        reviewer_id="reviewer-alice",
        notes="Grounded and verified.",
    )

    assert event.action == ReviewAction.APPROVED
    assert event.reviewed_revision == 1
    assert event.resulting_revision == 1
    assert event.reviewer_id == "reviewer-alice"
    assert event.reason_code is None

    draft = service.get_draft(draft_id)
    assert draft["human_review_status"] == "approved"
    assert draft["current_revision"] == 1


# 2. Reject Draft with all valid reason codes
def test_reject_with_all_valid_reasons(test_store):
    service = ReviewService(test_store)

    for i, reason in enumerate(RejectionReason):
        draft_id = _create_sample_draft(test_store, f"d-reject-{i}")
        event = service.reject_draft(
            draft_id=draft_id,
            expected_revision=1,
            reason_code=reason,
            reviewer_id="reviewer-bob",
            notes=f"Rejection due to {reason.value}",
        )

        assert event.action == ReviewAction.REJECTED
        assert event.reason_code == reason
        assert event.reviewed_revision == 1
        assert event.resulting_revision == 1

        draft = service.get_draft(draft_id)
        assert draft["human_review_status"] == "rejected"


# 3. Reject Draft without reason fails
def test_reject_without_reason_fails(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-reject-no-reason")

    with pytest.raises(InvalidReviewActionError):
        service.reject_draft(
            draft_id=draft_id,
            expected_revision=1,
            reason_code="",
        )

    with pytest.raises(InvalidReviewActionError):
        service.reject_draft(
            draft_id=draft_id,
            expected_revision=1,
            reason_code="NON_EXISTENT_REASON",
        )


# 4. Edit draft increments revision
def test_edit_draft_increments_revision(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-edit-inc")

    event = service.edit_draft(
        draft_id=draft_id,
        new_subject="Revised Subject Line",
        new_body="Completely rewritten body text with better value alignment.",
        expected_revision=1,
        reviewer_id="editor-carol",
    )

    assert event.action == ReviewAction.EDITED
    assert event.reviewed_revision == 1
    assert event.resulting_revision == 2

    draft = service.get_draft(draft_id)
    assert draft["current_revision"] == 2
    assert draft["subject"] == "Revised Subject Line"
    assert "Completely rewritten body" in draft["body"]


# 5. Edit draft remains pending_review
def test_edit_draft_remains_pending_review(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-edit-pending")

    service.edit_draft(
        draft_id=draft_id,
        new_subject="Updated Subject",
        new_body="Updated body content.",
        expected_revision=1,
    )

    draft = service.get_draft(draft_id)
    assert draft["human_review_status"] == "pending_review"


# 6. Edit computes exact edit distances
def test_edit_computes_exact_edit_distances(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(
        test_store,
        "d-edit-dist",
        subject="Hello Alice",
        body="Short body message.",
    )

    event = service.edit_draft(
        draft_id=draft_id,
        new_subject="Hello Alice",
        new_body="Short body message with extra detailed benchmarks.",
        expected_revision=1,
    )

    assert event.char_edit_distance is not None
    assert event.char_edit_distance > 0
    assert event.word_edit_distance is not None
    assert event.word_edit_distance > 0
    assert 0.0 < event.normalized_char_ratio <= 1.0
    assert 0.0 < event.normalized_word_ratio <= 1.0


# 7. Approve latest revision after multiple edits
def test_approve_latest_revision(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-multi-rev")

    # Rev 1 -> Rev 2
    service.edit_draft(
        draft_id=draft_id,
        new_subject="Subject v2",
        new_body="Body v2",
        expected_revision=1,
    )

    # Rev 2 -> Rev 3
    service.edit_draft(
        draft_id=draft_id,
        new_subject="Subject v3",
        new_body="Body v3",
        expected_revision=2,
    )

    # Approve Rev 3
    event = service.approve_draft(
        draft_id=draft_id,
        expected_revision=3,
        reviewer_id="reviewer-dave",
    )

    assert event.action == ReviewAction.APPROVED
    assert event.reviewed_revision == 3
    assert event.resulting_revision == 3

    draft = service.get_draft(draft_id)
    assert draft["current_revision"] == 3
    assert draft["human_review_status"] == "approved"


# 8. Earlier revisions preserved in history
def test_earlier_revisions_preserved_in_history(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(
        test_store,
        "d-hist",
        subject="Initial Subject v1",
        body="Initial Body v1",
    )

    service.edit_draft(
        draft_id=draft_id,
        new_subject="Subject v2",
        new_body="Body v2",
        expected_revision=1,
        reviewer_id="rev-1",
        notes="Revision 2 edit",
    )
    service.edit_draft(
        draft_id=draft_id,
        new_subject="Subject v3",
        new_body="Body v3",
        expected_revision=2,
        reviewer_id="rev-2",
        notes="Revision 3 edit",
    )
    service.approve_draft(
        draft_id=draft_id,
        expected_revision=3,
        reviewer_id="rev-3",
    )

    history = service.get_draft_history(draft_id)
    assert history.current_revision == 3
    assert history.status == "approved"
    assert len(history.revisions) == 3

    # Rev 1
    assert history.revisions[0].revision_number == 1
    assert history.revisions[0].subject == "Initial Subject v1"
    assert history.revisions[0].body == "Initial Body v1"
    assert history.revisions[0].action_leading_to_revision is None

    # Rev 2
    assert history.revisions[1].revision_number == 2
    assert history.revisions[1].subject == "Subject v2"
    assert history.revisions[1].body == "Body v2"
    assert history.revisions[1].action_leading_to_revision == ReviewAction.EDITED

    # Rev 3
    assert history.revisions[2].revision_number == 3
    assert history.revisions[2].subject == "Subject v3"
    assert history.revisions[2].body == "Body v3"

    # Events count
    assert len(history.events) == 3


# 9. Concurrent review conflict detected
def test_concurrent_review_conflict_detected(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-concurrency")

    # Reviewer A edits draft (rev 1 -> rev 2)
    service.edit_draft(
        draft_id=draft_id,
        new_subject="Subject v2",
        new_body="Body v2",
        expected_revision=1,
    )

    # Reviewer B attempts to approve with stale revision 1
    with pytest.raises(ConcurrencyConflictError) as exc_info:
        service.approve_draft(
            draft_id=draft_id,
            expected_revision=1,
            reviewer_id="reviewer-stale",
        )
    assert "revision conflict" in str(exc_info.value).lower()

    # Reviewer B attempts to edit with stale revision 1
    with pytest.raises(ConcurrencyConflictError):
        service.edit_draft(
            draft_id=draft_id,
            new_subject="Subject v2 conflict",
            new_body="Body v2 conflict",
            expected_revision=1,
        )


# 10. Action on finalized draft rejected
def test_action_on_final_draft_rejected(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-finalized")

    service.approve_draft(draft_id=draft_id, expected_revision=1)

    # Cannot edit approved draft
    with pytest.raises(InvalidReviewTransitionError):
        service.edit_draft(
            draft_id=draft_id,
            new_subject="Post-approval edit",
            new_body="Post-approval body",
            expected_revision=1,
        )

    # Cannot reject approved draft
    with pytest.raises(InvalidReviewTransitionError):
        service.reject_draft(
            draft_id=draft_id,
            expected_revision=1,
            reason_code=RejectionReason.OTHER,
        )


# 11. Atomic rollback on failure
def test_atomic_rollback_on_failure(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-rollback")

    # Stale revision conflict triggers exception inside UnitOfWork
    with pytest.raises(ConcurrencyConflictError):
        service.edit_draft(
            draft_id=draft_id,
            new_subject="New subject",
            new_body="New body",
            expected_revision=999,  # Wrong revision
        )

    # Verify no review events created
    with test_store.uow() as uow:
        events = uow.outreach.get_human_review_events(draft_id)
        assert len(events) == 0

        draft = uow.outreach.get_outreach_draft(draft_id)
        assert draft["current_revision"] == 1
        assert draft["human_review_status"] == "pending_review"


# 12. Immutable review history triggers prevent UPDATE and DELETE
def test_immutable_review_history_triggers(test_store):
    service = ReviewService(test_store)
    draft_id = _create_sample_draft(test_store, "d-immutable")

    event = service.approve_draft(
        draft_id=draft_id,
        expected_revision=1,
        reviewer_id="reviewer-1",
    )

    with test_store.uow() as uow:
        # Direct SQL UPDATE must be aborted by trigger
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)) as exc_update:
            uow.conn.execute(
                "UPDATE human_review_events SET notes = 'tampered' WHERE event_id = ?;",
                (event.event_id,),
            )
        assert "append-only" in str(exc_update.value)

        # Direct SQL DELETE must be aborted by trigger
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)) as exc_delete:
            uow.conn.execute(
                "DELETE FROM human_review_events WHERE event_id = ?;",
                (event.event_id,),
            )
        assert "append-only" in str(exc_delete.value)


# 13. Telemetry approval and rejection rates
def test_telemetry_approval_rejection_rates(test_store):
    service = ReviewService(test_store)

    # 3 approved drafts, 1 rejected draft, 1 left pending
    for i in range(3):
        d = _create_sample_draft(test_store, f"d-tele-app-{i}")
        service.approve_draft(d, expected_revision=1)

    d_rej = _create_sample_draft(test_store, "d-tele-rej")
    service.reject_draft(d_rej, expected_revision=1, reason_code=RejectionReason.HALLUCINATION)

    _create_sample_draft(test_store, "d-tele-pending")  # No review action

    telemetry = service.get_telemetry()
    assert telemetry.total_drafts_reviewed == 4
    assert telemetry.final_decision_drafts_count == 4
    assert telemetry.approved_count == 3
    assert telemetry.rejected_count == 1
    assert telemetry.approval_rate == 0.75
    assert telemetry.rejection_rate == 0.25
    assert telemetry.rejection_reasons_breakdown.get("HALLUCINATION") == 1


# 14. Telemetry edit frequency
def test_telemetry_edit_frequency(test_store):
    service = ReviewService(test_store)

    # 4 reviewed drafts: 2 have edits, 2 approved directly
    d1 = _create_sample_draft(test_store, "d-ef-1")
    service.edit_draft(d1, "S1", "B1", expected_revision=1)
    service.approve_draft(d1, expected_revision=2)

    d2 = _create_sample_draft(test_store, "d-ef-2")
    service.edit_draft(d2, "S2", "B2", expected_revision=1)
    service.reject_draft(d2, expected_revision=2, reason_code=RejectionReason.TONE_MISMATCH)

    d3 = _create_sample_draft(test_store, "d-ef-3")
    service.approve_draft(d3, expected_revision=1)

    d4 = _create_sample_draft(test_store, "d-ef-4")
    service.approve_draft(d4, expected_revision=1)

    telemetry = service.get_telemetry()
    assert telemetry.total_drafts_reviewed == 4
    assert telemetry.drafts_with_edits_count == 2
    assert telemetry.edit_frequency == 0.5


# 15. Telemetry edit distance percentiles (linear interpolation)
def test_telemetry_edit_distance_percentiles(test_store):
    service = ReviewService(test_store)

    # Perform 3 edits with varying distance
    d1 = _create_sample_draft(test_store, "d-perc-1", subject="Sub", body="WordA")
    service.edit_draft(d1, "Sub", "WordA WordB", expected_revision=1)

    d2 = _create_sample_draft(test_store, "d-perc-2", subject="Sub", body="WordA")
    service.edit_draft(d2, "Sub", "WordA WordB WordC WordD", expected_revision=1)

    d3 = _create_sample_draft(test_store, "d-perc-3", subject="Sub", body="WordA")
    service.edit_draft(d3, "Sub", "WordA WordB WordC WordD WordE WordF", expected_revision=1)

    telemetry = service.get_telemetry()
    stats = telemetry.edit_distance_stats
    assert stats.count == 3
    assert stats.mean_char_distance > 0
    assert stats.median_char_distance > 0
    assert stats.p90_char_distance >= stats.median_char_distance
    assert stats.p90_word_distance >= stats.median_word_distance
    assert telemetry.percentile_method == "linear_interpolation"


# 16. Reviewer activity grouped by identity
def test_telemetry_reviewer_activity(test_store):
    service = ReviewService(test_store)

    d1 = _create_sample_draft(test_store, "d-act-1")
    service.edit_draft(d1, "S", "B", expected_revision=1, reviewer_id="user-alice")
    service.approve_draft(d1, expected_revision=2, reviewer_id="user-alice")

    d2 = _create_sample_draft(test_store, "d-act-2")
    service.reject_draft(
        d2,
        expected_revision=1,
        reason_code=RejectionReason.TIMING_IRRELEVANCE,
        reviewer_id="user-bob",
    )

    telemetry = service.get_telemetry()
    activity = telemetry.reviewer_activity

    assert "user-alice" in activity
    assert activity["user-alice"].total_actions == 2
    assert activity["user-alice"].edits == 1
    assert activity["user-alice"].approvals == 1

    assert "user-bob" in activity
    assert activity["user-bob"].total_actions == 1
    assert activity["user-bob"].rejections == 1


# 17. List drafts paginated
def test_list_drafts_paginated(test_store):
    service = ReviewService(test_store)

    for i in range(5):
        _create_sample_draft(test_store, f"d-page-{i}", status="pending_review")

    drafts_p1, total = service.list_drafts(limit=2, offset=0, status="pending_review")
    assert total == 5
    assert len(drafts_p1) == 2

    drafts_p2, total = service.list_drafts(limit=2, offset=2, status="pending_review")
    assert len(drafts_p2) == 2
    assert drafts_p1[0]["draft_id"] != drafts_p2[0]["draft_id"]


# 18. Linear interpolation percentile helper unit test
def test_compute_percentile_helper():
    assert compute_percentile([], 50.0) == 0.0
    assert compute_percentile([42.0], 50.0) == 42.0
    assert compute_percentile([10.0, 20.0, 30.0], 50.0) == 20.0
    assert compute_percentile([10.0, 20.0, 30.0], 90.0) == 28.0
    assert compute_percentile([10.0, 20.0, 30.0], 0.0) == 10.0
    assert compute_percentile([10.0, 20.0, 30.0], 100.0) == 30.0
