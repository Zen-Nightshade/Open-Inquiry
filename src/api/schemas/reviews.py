from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.email_generation.review_models import RejectionReason


class ApproveDraftRequest(BaseModel):
    """Payload to approve an outreach draft on a specific revision."""

    expected_revision: int = Field(ge=1, description="Expected current revision number (CAS check)")
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of the human reviewer")
    notes: Optional[str] = Field(default=None, description="Optional reviewer commentary")


class RejectDraftRequest(BaseModel):
    """Payload to reject an outreach draft with mandatory reason code."""

    expected_revision: int = Field(ge=1, description="Expected current revision number (CAS check)")
    reason_code: RejectionReason = Field(description="Structured reason code for draft rejection")
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of the human reviewer")
    notes: Optional[str] = Field(default=None, description="Optional reviewer commentary")


class EditDraftRequest(BaseModel):
    """Payload to edit subject and/or body of an outreach draft, creating a new revision."""

    expected_revision: int = Field(ge=1, description="Expected current revision number (CAS check)")
    new_subject: str = Field(min_length=1, description="Updated draft subject line")
    new_body: str = Field(min_length=1, description="Updated draft body copy")
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of the editor")
    notes: Optional[str] = Field(default=None, description="Optional edit rationale")


class PaginatedDraftsResponse(BaseModel):
    """Deterministic paginated collection of outreach drafts."""

    drafts: List[Dict[str, Any]]
    total_count: int
    limit: int
    offset: int
