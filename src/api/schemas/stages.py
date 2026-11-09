from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StageRecordResponse(BaseModel):
    stage: str = Field(..., description="Stage name (e.g. validation, research, signal_strategy)")
    status: str = Field(..., description="Stage execution status (e.g. completed, failed)")
    started_at: str = Field(..., description="Stage start timestamp")
    completed_at: Optional[str] = Field(None, description="Stage completion timestamp")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Stage diagnostic metadata")
    error_message: Optional[str] = Field(None, description="Error message if stage failed")


class RunStagesResponse(BaseModel):
    run_id: str = Field(..., description="Run identifier")
    total_stages: int = Field(..., description="Count of executed stage records")
    stages: List[StageRecordResponse] = Field(default_factory=list, description="Ordered stage records")
