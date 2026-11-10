from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewAction(str, Enum):
    """Categorical action taken by a human reviewer on an outreach draft."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EDITED = "EDITED"


class RejectionReason(str, Enum):
    """Structured reason required when a human reviewer rejects a draft."""

    HALLUCINATION = "HALLUCINATION"
    TONE_MISMATCH = "TONE_MISMATCH"
    WRONG_CAPABILITY = "WRONG_CAPABILITY"
    TIMING_IRRELEVANCE = "TIMING_IRRELEVANCE"
    INSUFFICIENT_PERSONALIZATION = "INSUFFICIENT_PERSONALIZATION"
    OTHER = "OTHER"


class ReviewEvent(BaseModel):
    """An immutable, append-only record of a human reviewer action on a draft revision."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(description="Unique collision-resistant event identifier")
    draft_id: str = Field(description="Foreign key to the reviewed outreach draft")
    action: ReviewAction = Field(description="Action executed: APPROVED, REJECTED, or EDITED")
    reviewed_revision: int = Field(ge=1, description="Revision number on which the reviewer acted")
    resulting_revision: int = Field(ge=1, description="Revision number resulting from the action")
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of reviewer who performed the action")
    notes: Optional[str] = Field(default=None, description="Optional textual reviewer commentary")
    reason_code: Optional[RejectionReason] = Field(
        default=None,
        description="Mandatory structured reason code if action is REJECTED",
    )
    original_subject: Optional[str] = Field(default=None, description="Subject prior to action")
    original_body: Optional[str] = Field(default=None, description="Body prior to action")
    resulting_subject: Optional[str] = Field(default=None, description="Subject resulting from action")
    resulting_body: Optional[str] = Field(default=None, description="Body resulting from action")
    char_edit_distance: Optional[int] = Field(default=None, ge=0, description="Levenshtein char distance for EDITED")
    word_edit_distance: Optional[int] = Field(default=None, ge=0, description="Levenshtein word distance for EDITED")
    normalized_char_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized char ratio")
    normalized_word_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized word ratio")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp when action was recorded",
    )

    @model_validator(mode="after")
    def validate_action_semantics(self) -> ReviewEvent:
        if self.action == ReviewAction.REJECTED:
            if not self.reason_code:
                raise ValueError("reason_code is required when action is REJECTED")
            if self.resulting_revision != self.reviewed_revision:
                raise ValueError("REJECTED action must have resulting_revision == reviewed_revision")
        elif self.action == ReviewAction.APPROVED:
            if self.reason_code is not None:
                raise ValueError("reason_code must not be provided when action is APPROVED")
            if self.resulting_revision != self.reviewed_revision:
                raise ValueError("APPROVED action must have resulting_revision == reviewed_revision")
        elif self.action == ReviewAction.EDITED:
            if self.reason_code is not None:
                raise ValueError("reason_code must not be provided when action is EDITED")
            if self.resulting_revision != self.reviewed_revision + 1:
                raise ValueError("EDITED action must increment resulting_revision by exactly 1")
            if not self.resulting_subject or not self.resulting_body:
                raise ValueError("resulting_subject and resulting_body are required for EDITED action")
            if self.char_edit_distance is None or self.word_edit_distance is None:
                raise ValueError("char_edit_distance and word_edit_distance are required for EDITED action")
        return self


class DraftRevision(BaseModel):
    """Snapshot of a specific revision of an outreach draft."""

    model_config = ConfigDict(frozen=True)

    revision_number: int = Field(ge=1, description="1-based revision sequence number")
    subject: str = Field(description="Subject line for this revision")
    body: str = Field(description="Body content for this revision")
    word_count: int = Field(ge=0, description="Word count of the body")
    created_at: str = Field(description="ISO timestamp when revision was created or committed")
    action_leading_to_revision: Optional[ReviewAction] = Field(
        default=None,
        description="Action that produced this revision (None for initial v1)",
    )
    reviewer_id: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)


class DraftReviewHistory(BaseModel):
    """Comprehensive chronological review history and all revisions for an outreach draft."""

    model_config = ConfigDict(frozen=True)

    draft_id: str
    context_id: str
    current_revision: int
    status: str
    initial_subject: str
    initial_body: str
    current_subject: str
    current_body: str
    created_at: str
    reviewer_notes: Optional[str] = None
    revisions: List[DraftRevision]
    events: List[ReviewEvent]


class EditDistanceStats(BaseModel):
    """Aggregate statistics for edit distances computed strictly over EDITED events."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    mean_char_distance: float = Field(default=0.0)
    median_char_distance: float = Field(default=0.0)
    p90_char_distance: float = Field(default=0.0)
    mean_word_distance: float = Field(default=0.0)
    median_word_distance: float = Field(default=0.0)
    p90_word_distance: float = Field(default=0.0)
    mean_normalized_char_ratio: float = Field(default=0.0)
    median_normalized_char_ratio: float = Field(default=0.0)
    p90_normalized_char_ratio: float = Field(default=0.0)
    mean_normalized_word_ratio: float = Field(default=0.0)
    median_normalized_word_ratio: float = Field(default=0.0)
    p90_normalized_word_ratio: float = Field(default=0.0)


class ReviewerActivity(BaseModel):
    """Summary of actions executed by a specific reviewer."""

    model_config = ConfigDict(frozen=True)

    reviewer_id: str
    total_actions: int = Field(ge=0)
    approvals: int = Field(ge=0)
    rejections: int = Field(ge=0)
    edits: int = Field(ge=0)


class ReviewTelemetrySummary(BaseModel):
    """Authoritative aggregate metrics across all human review events."""

    model_config = ConfigDict(frozen=True)

    total_drafts_reviewed: int = Field(ge=0, description="Count of distinct drafts with >= 1 review event")
    total_review_events: int = Field(ge=0, description="Total number of review events logged")
    final_decision_drafts_count: int = Field(ge=0, description="Distinct drafts with final approved/rejected status")
    approved_count: int = Field(ge=0, description="Drafts approved")
    rejected_count: int = Field(ge=0, description="Drafts rejected")
    edited_count: int = Field(ge=0, description="Total EDITED events")
    drafts_with_edits_count: int = Field(ge=0, description="Distinct drafts with >= 1 edit")
    approval_rate: float = Field(ge=0.0, le=1.0, description="approved / final_decision_drafts_count")
    rejection_rate: float = Field(ge=0.0, le=1.0, description="rejected / final_decision_drafts_count")
    edit_frequency: float = Field(ge=0.0, le=1.0, description="drafts_with_edits / total_drafts_reviewed")
    percentile_method: str = Field(default="linear_interpolation")
    rejection_reasons_breakdown: Dict[str, int] = Field(default_factory=dict)
    edit_distance_stats: EditDistanceStats
    reviewer_activity: Dict[str, ReviewerActivity] = Field(default_factory=dict)


class ReviewError(Exception):
    """Base exception for review workflow domain errors."""
    pass


class DraftNotFoundError(ReviewError):
    """Raised when an operation targets a non-existent draft ID."""
    pass


class InvalidReviewTransitionError(ReviewError):
    """Raised when an operation attempts to mutate or act upon an already finalized draft."""
    pass


class ConcurrencyConflictError(ReviewError):
    """Raised when an optimistic concurrency check detects a conflicting revision update."""
    pass


class InvalidReviewActionError(ReviewError):
    """Raised when an invalid review action or missing reason code is submitted."""
    pass
