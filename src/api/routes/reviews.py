from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_review_service
from src.api.schemas.reviews import (
    ApproveDraftRequest,
    EditDraftRequest,
    PaginatedDraftsResponse,
    RejectDraftRequest,
)
from src.email_generation.review_models import (
    ConcurrencyConflictError,
    DraftNotFoundError,
    DraftReviewHistory,
    InvalidReviewActionError,
    InvalidReviewTransitionError,
    ReviewEvent,
    ReviewTelemetrySummary,
)
from src.email_generation.review_service import ReviewService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reviews & Drafts"])


@router.get("/api/v1/drafts", response_model=PaginatedDraftsResponse)
@router.get("/drafts", response_model=PaginatedDraftsResponse, include_in_schema=False)
def list_drafts(
    limit: int = Query(50, ge=1, le=200, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    status: Optional[str] = Query(None, description="Filter by human review status"),
    service: ReviewService = Depends(get_review_service),
) -> PaginatedDraftsResponse:
    drafts, total = service.list_drafts(limit=limit, offset=offset, status=status)
    return PaginatedDraftsResponse(
        drafts=drafts,
        total_count=total,
        limit=limit,
        offset=offset,
    )


@router.get("/api/v1/drafts/{draft_id}")
@router.get("/drafts/{draft_id}", include_in_schema=False)
def get_draft(
    draft_id: str,
    service: ReviewService = Depends(get_review_service),
) -> Dict[str, Any]:
    try:
        return service.get_draft(draft_id)
    except DraftNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/api/v1/drafts/{draft_id}/history", response_model=DraftReviewHistory)
@router.get("/drafts/{draft_id}/history", response_model=DraftReviewHistory, include_in_schema=False)
def get_draft_history(
    draft_id: str,
    service: ReviewService = Depends(get_review_service),
) -> DraftReviewHistory:
    try:
        return service.get_draft_history(draft_id)
    except DraftNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/api/v1/drafts/{draft_id}/approve", response_model=ReviewEvent)
@router.post("/drafts/{draft_id}/approve", response_model=ReviewEvent, include_in_schema=False)
def approve_draft(
    draft_id: str,
    payload: ApproveDraftRequest,
    service: ReviewService = Depends(get_review_service),
) -> ReviewEvent:
    try:
        return service.approve_draft(
            draft_id=draft_id,
            expected_revision=payload.expected_revision,
            reviewer_id=payload.reviewer_id,
            notes=payload.notes,
        )
    except DraftNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (ConcurrencyConflictError, InvalidReviewTransitionError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except InvalidReviewActionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post("/api/v1/drafts/{draft_id}/reject", response_model=ReviewEvent)
@router.post("/drafts/{draft_id}/reject", response_model=ReviewEvent, include_in_schema=False)
def reject_draft(
    draft_id: str,
    payload: RejectDraftRequest,
    service: ReviewService = Depends(get_review_service),
) -> ReviewEvent:
    try:
        return service.reject_draft(
            draft_id=draft_id,
            expected_revision=payload.expected_revision,
            reason_code=payload.reason_code,
            reviewer_id=payload.reviewer_id,
            notes=payload.notes,
        )
    except DraftNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (ConcurrencyConflictError, InvalidReviewTransitionError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except InvalidReviewActionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post("/api/v1/drafts/{draft_id}/edit", response_model=ReviewEvent)
@router.post("/drafts/{draft_id}/edit", response_model=ReviewEvent, include_in_schema=False)
def edit_draft(
    draft_id: str,
    payload: EditDraftRequest,
    service: ReviewService = Depends(get_review_service),
) -> ReviewEvent:
    try:
        return service.edit_draft(
            draft_id=draft_id,
            new_subject=payload.new_subject,
            new_body=payload.new_body,
            expected_revision=payload.expected_revision,
            reviewer_id=payload.reviewer_id,
            notes=payload.notes,
        )
    except DraftNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (ConcurrencyConflictError, InvalidReviewTransitionError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except InvalidReviewActionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.get("/api/v1/reviews/telemetry", response_model=ReviewTelemetrySummary)
@router.get("/reviews/telemetry", response_model=ReviewTelemetrySummary, include_in_schema=False)
def get_telemetry(
    service: ReviewService = Depends(get_review_service),
) -> ReviewTelemetrySummary:
    return service.get_telemetry()
