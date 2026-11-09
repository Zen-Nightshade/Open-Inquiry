from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status

from src.api.dependencies import get_orchestrator, get_state_store
from src.api.schemas.common import LinkItem
from src.api.schemas.research import ResearchInspectionResponse
from src.api.schemas.runs import (
    CreateRunRequest,
    CreateRunResponse,
    RunDetailResponse,
    RunListResponse,
    RunSummaryResponse,
)
from src.api.schemas.stages import RunStagesResponse, StageRecordResponse
from src.api.schemas.strategy import StrategyInspectionResponse
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator
from src.orchestration.models import RunStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/runs", tags=["Runs"])


def _build_run_links(run_id: str) -> List[LinkItem]:
    return [
        LinkItem(href=f"/api/v1/runs/{run_id}", rel="self", method="GET"),
        LinkItem(href=f"/api/v1/runs/{run_id}/stages", rel="stages", method="GET"),
        LinkItem(href=f"/api/v1/runs/{run_id}/research", rel="research", method="GET"),
        LinkItem(href=f"/api/v1/runs/{run_id}/strategy", rel="strategy", method="GET"),
    ]


def _background_execute_run(
    db_path: Optional[str],
    prospect: ProspectContext,
    product: ProductContext,
    outreach: OutreachContext,
    run_id: str,
) -> None:
    """Worker task executed in FastAPI background threadpool."""
    try:
        store = StateStore(db_path=db_path)
        orchestrator = ApplicationOrchestrator(state_store=store)
        orchestrator.run(
            prospect=prospect,
            product=product,
            outreach=outreach,
            run_id=run_id,
        )
    except Exception as exc:
        logger.error("Background execution for run %s failed: %s", run_id, exc, exc_info=True)


@router.post(
    "",
    response_model=CreateRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and start an application run",
    description="Validates prospect, product, and outreach context, initializes the run, and executes the research and strategy pipeline asynchronously in the background.",
)
def create_run(
    req: CreateRunRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    orchestrator: ApplicationOrchestrator = Depends(get_orchestrator),
) -> CreateRunResponse:
    try:
        domain_prospect = req.prospect.to_domain()
        domain_product = req.product.to_domain()
        domain_outreach = req.outreach.to_domain()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Context validation failed: {exc}",
        )

    try:
        snapshot = orchestrator.create_pending_run(
            prospect=domain_prospect,
            product=domain_product,
            outreach=domain_outreach,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))

    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(
        _background_execute_run,
        db_path=db_path,
        prospect=domain_prospect,
        product=domain_product,
        outreach=domain_outreach,
        run_id=snapshot.run_id,
    )

    return CreateRunResponse(
        run_id=snapshot.run_id,
        status=snapshot.status.value,
        message="Run accepted and initialized. Executing research and strategy pipeline asynchronously.",
        links=_build_run_links(snapshot.run_id),
    )


@router.get(
    "",
    response_model=RunListResponse,
    summary="List application runs",
    description="Query historical application runs with optional status and company filtering.",
)
def list_runs(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of runs to return"),
    status: Optional[str] = Query(None, description="Filter runs by status (e.g. completed, failed)"),
    company: Optional[str] = Query(None, description="Filter runs by prospect company name substring"),
    orchestrator: ApplicationOrchestrator = Depends(get_orchestrator),
) -> RunListResponse:
    run_status = None
    if status:
        try:
            run_status = RunStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter '{status}'. Valid options: {[s.value for s in RunStatus]}",
            )

    rows = orchestrator.list_runs(limit=limit, status=run_status)
    summaries: List[RunSummaryResponse] = []

    for r in rows:
        prospect_name = "N/A"
        company_name = "N/A"
        if r.get("prospect_snapshot_json"):
            try:
                p_data = json.loads(r["prospect_snapshot_json"])
                prospect_name = p_data.get("full_name", "N/A")
                company_name = p_data.get("company_name", "N/A")
            except Exception:
                pass

        if company and company.lower() not in company_name.lower():
            continue

        warnings_count = 0
        if r.get("warnings_json"):
            try:
                warnings_count = len(json.loads(r["warnings_json"]))
            except Exception:
                pass

        summaries.append(
            RunSummaryResponse(
                run_id=r["run_id"],
                status=r["status"],
                prospect_name=prospect_name,
                company_name=company_name,
                started_at=r["started_at"],
                completed_at=r.get("completed_at"),
                warnings_count=warnings_count,
                links=_build_run_links(r["run_id"]),
            )
        )

    return RunListResponse(total=len(summaries), runs=summaries)


@router.get(
    "/{run_id}",
    response_model=RunDetailResponse,
    summary="Get application run details",
    description="Retrieve the current status, snapshots, and stage timeline for an application run.",
)
def get_run(
    run_id: str,
    orchestrator: ApplicationOrchestrator = Depends(get_orchestrator),
) -> RunDetailResponse:
    snapshot = orchestrator.get_run(run_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    return RunDetailResponse(
        run_id=snapshot.run_id,
        job_id=snapshot.job_id,
        research_session_id=snapshot.research_session_id,
        status=snapshot.status.value,
        prospect=snapshot.prospect.model_dump(),
        product=snapshot.product.model_dump(),
        outreach=snapshot.outreach.model_dump(),
        stage_statuses=snapshot.stage_statuses,
        warnings=snapshot.warnings,
        error_message=snapshot.error_message,
        started_at=snapshot.started_at,
        completed_at=snapshot.completed_at,
        links=_build_run_links(snapshot.run_id),
    )


@router.get(
    "/{run_id}/stages",
    response_model=RunStagesResponse,
    summary="Inspect run execution stages",
    description="Retrieve the chronological record of execution pipeline stages, durations, and diagnostic details.",
)
def get_run_stages(
    run_id: str,
    orchestrator: ApplicationOrchestrator = Depends(get_orchestrator),
) -> RunStagesResponse:
    snapshot = orchestrator.get_run(run_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    stages = orchestrator.get_run_stages(run_id)
    stage_responses = [
        StageRecordResponse(
            stage=s.stage.value,
            status=s.status.value,
            started_at=s.started_at,
            completed_at=s.completed_at,
            duration_ms=s.duration_ms,
            details=s.details,
            error_message=s.error_message,
        )
        for s in stages
    ]

    return RunStagesResponse(
        run_id=run_id,
        total_stages=len(stage_responses),
        stages=stage_responses,
    )


@router.get(
    "/{run_id}/research",
    response_model=ResearchInspectionResponse,
    summary="Inspect research evidence and provenance",
    description="Retrieve canonical facts, full evidence passage quotes, source URLs, credibility scores, entity assessments, and contradictions.",
)
def get_run_research(
    run_id: str,
    orchestrator: ApplicationOrchestrator = Depends(get_orchestrator),
) -> ResearchInspectionResponse:
    snapshot = orchestrator.get_run(run_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    research_data = orchestrator.get_research_inspection(run_id)
    if not research_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No research session found for run '{run_id}' (run status: {snapshot.status.value}).",
        )

    return ResearchInspectionResponse(**research_data)


@router.get(
    "/{run_id}/strategy",
    response_model=StrategyInspectionResponse,
    summary="Inspect outreach strategy and explainability",
    description="Retrieve the synthesized commercial strategy blueprint, ranked signals, prospect-product alignment, and underlying evidence chain.",
)
def get_run_strategy(
    run_id: str,
    orchestrator: ApplicationOrchestrator = Depends(get_orchestrator),
) -> StrategyInspectionResponse:
    snapshot = orchestrator.get_run(run_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found.",
        )

    strategy_data = orchestrator.get_strategy_explainability(run_id)
    if "error" in strategy_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No viable strategy available for run '{run_id}' (run status: {snapshot.status.value}): {strategy_data['error']}",
        )

    return StrategyInspectionResponse(**strategy_data)
