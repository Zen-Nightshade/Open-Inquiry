from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.email_generation.models import VerifiedOutreachDraft
from src.researcher.evidence_request.models import FeedbackRoundResult
from src.researcher.models import ResearchPackage
from src.signal_strategy.models import StrategyResult


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    INSUFFICIENT_RESEARCH = "insufficient_research"
    NO_VIABLE_SIGNAL = "no_viable_signal"
    FAILED = "failed"


class StageIdentifier(str, Enum):
    VALIDATION = "validation"
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    SUFFICIENCY_GATE = "sufficiency_gate"
    SIGNAL_STRATEGY = "signal_strategy"
    RESEARCH_FEEDBACK = "research_feedback"
    EMAIL_GENERATION = "email_generation"
    FINALIZATION = "finalization"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageExecutionRecord(BaseModel):
    """Execution audit record for an individual application pipeline stage."""

    model_config = ConfigDict(frozen=True)

    stage: StageIdentifier
    status: StageStatus
    started_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class ApplicationRunSnapshot(BaseModel):
    """Complete, deserialized representation of a persisted application run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    job_id: Optional[str] = None
    prospect_id: str
    context_id: Optional[str] = None
    research_session_id: Optional[str] = None
    draft_id: Optional[str] = None
    status: RunStatus
    prospect: ProspectContext
    product: ProductContext
    outreach: OutreachContext
    strategy_result: Optional[StrategyResult] = None
    feedback_round: Optional[FeedbackRoundResult] = None
    stage_statuses: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None


class RunResult(BaseModel):
    """Outcome produced by the ApplicationOrchestrator."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    prospect: ProspectContext
    product: ProductContext
    outreach: OutreachContext
    research_session_id: Optional[str] = None
    research_package: Optional[ResearchPackage] = None
    strategy_result: Optional[StrategyResult] = None
    feedback_round: Optional[FeedbackRoundResult] = None
    draft: Optional[VerifiedOutreachDraft] = None
    draft_id: Optional[str] = None
    stage_records: List[StageExecutionRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
