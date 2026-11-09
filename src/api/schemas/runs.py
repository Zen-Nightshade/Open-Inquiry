from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.api.schemas.common import LinkItem
from src.api.schemas.context import OutreachInput, ProductInput, ProspectInput


class CreateRunRequest(BaseModel):
    prospect: ProspectInput = Field(..., description="Target prospect details")
    product: ProductInput = Field(..., description="Product value proposition and capabilities")
    outreach: OutreachInput = Field(..., description="Outreach objective and sender constraints")


class CreateRunResponse(BaseModel):
    run_id: str = Field(..., description="Unique identifier for the created execution run")
    status: str = Field(..., description="Current status of the run, e.g. 'pending'")
    message: str = Field(..., description="Informational message about the asynchronous execution")
    links: List[LinkItem] = Field(default_factory=list, description="Hypermedia resource links")


class RunSummaryResponse(BaseModel):
    run_id: str = Field(..., description="Unique identifier for the run")
    status: str = Field(..., description="Execution status")
    prospect_name: str = Field(..., description="Target prospect name")
    company_name: str = Field(..., description="Prospect company name")
    started_at: str = Field(..., description="Run start timestamp")
    completed_at: Optional[str] = Field(None, description="Run completion timestamp")
    warnings_count: int = Field(0, description="Number of execution warnings recorded")
    links: List[LinkItem] = Field(default_factory=list, description="Hypermedia links")


class RunListResponse(BaseModel):
    total: int = Field(..., description="Total runs matching query")
    runs: List[RunSummaryResponse] = Field(default_factory=list, description="List of run summaries")


class RunDetailResponse(BaseModel):
    run_id: str = Field(..., description="Unique run identifier")
    job_id: Optional[str] = Field(None, description="Associated job ID")
    research_session_id: Optional[str] = Field(None, description="Associated research session ID")
    status: str = Field(..., description="Execution status")
    prospect: Dict[str, Any] = Field(..., description="Prospect context")
    product: Dict[str, Any] = Field(..., description="Product context")
    outreach: Dict[str, Any] = Field(..., description="Outreach context")
    stage_statuses: List[Dict[str, Any]] = Field(default_factory=list, description="High-level stages summary")
    warnings: List[str] = Field(default_factory=list, description="Warnings encountered during run")
    error_message: Optional[str] = Field(None, description="Error message if run failed")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    links: List[LinkItem] = Field(default_factory=list, description="Hypermedia links")
