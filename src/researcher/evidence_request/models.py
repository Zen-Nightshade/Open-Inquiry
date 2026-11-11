from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .validator import BiasedEvidenceRequestError, validate_hypothesis_blindness


class EvidenceGapReason(str, Enum):
    """Categorized factual uncertainty motivating an EvidenceRequest."""

    MISSING_PREREQUISITE = "missing_prerequisite"
    UNRESOLVED_EMPLOYMENT = "unresolved_employment"
    UNCERTAIN_TEMPORAL_STATUS = "uncertain_temporal_status"
    MATERIAL_CONTRADICTION = "material_contradiction"
    UNRESOLVED_CAPABILITY_PREMISE = "unresolved_capability_premise"


class EvidenceRequestStatus(str, Enum):
    """Lifecycle status of an EvidenceRequest."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    UNRESOLVED = "unresolved"
    FAILED = "failed"
    DEFERRED = "deferred"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REJECTED_BIASED = "rejected_biased"


class EvidenceRequest(BaseModel):
    """Typed, validated contract for investigating an unresolved factual premise without commercial bias."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(default_factory=lambda: f"evr_{uuid.uuid4().hex[:12]}")
    run_id: str = Field(description="Associated application run ID")
    target_entity_id: str = Field(description="Target entity ID to investigate (must be sufficiently resolved)")
    target_entity_name: Optional[str] = Field(default=None, description="Human-readable name of target entity")
    requested_predicate: str = Field(description="Controlled predicate or factual relationship under investigation")
    neutral_question: str = Field(description="Neutral factual question without commercial persuasion")
    neutral_search_terms: List[str] = Field(description="Neutral search terms seeking evidence for the premise")
    disconfirming_search_terms: List[str] = Field(
        description="Counter-evidence search terms actively seeking disconfirming facts"
    )
    valid_from: Optional[str] = Field(default=None, description="Temporal scope lower bound")
    valid_to: Optional[str] = Field(default=None, description="Temporal scope upper bound")
    gap_reason: EvidenceGapReason = Field(description="Underlying reason for the factual uncertainty")
    provenance_stage: str = Field(description="Strategy stage finding generating this request")
    status: EvidenceRequestStatus = Field(default=EvidenceRequestStatus.PENDING)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(default=None)
    motivated_by_fact_ids: List[str] = Field(
        default_factory=list,
        description="IDs of existing facts motivating the gap investigation",
    )
    motivated_by_signal_id: Optional[str] = Field(
        default=None,
        description="ID of candidate signal that identified this gap",
    )
    budget_cost_units: int = Field(default=1, ge=1, description="Budget units required to execute this request")
    rejection_reason: Optional[str] = Field(default=None, description="Reason if rejected as biased or invalid")
    resulting_fact_ids: List[str] = Field(
        default_factory=list,
        description="Fact IDs produced by researcher investigation",
    )
    resulting_source_ids: List[str] = Field(
        default_factory=list,
        description="Source IDs fetched during investigation",
    )
    searched_scope: List[str] = Field(
        default_factory=list,
        description="List of executed queries or domains searched",
    )

    @model_validator(mode="after")
    def validate_contract_and_neutrality(self) -> EvidenceRequest:
        """Enforce structural validity and hypothesis blindness."""
        if not self.target_entity_id or not self.target_entity_id.strip():
            raise ValueError("EvidenceRequest must specify a non-empty target_entity_id.")
        if not self.requested_predicate or not self.requested_predicate.strip():
            raise ValueError("EvidenceRequest must specify a non-empty requested_predicate.")
        if not self.neutral_question or not self.neutral_question.strip():
            raise ValueError("EvidenceRequest must specify a non-empty neutral_question.")
        if not self.neutral_search_terms:
            raise ValueError("EvidenceRequest must provide at least one neutral_search_term.")
        if not self.disconfirming_search_terms:
            raise ValueError(
                "EvidenceRequest must provide at least one disconfirming_search_term to ensure balanced investigation."
            )

        if self.valid_from and self.valid_to:
            if self.valid_from > self.valid_to:
                raise ValueError(
                    f"valid_from '{self.valid_from}' cannot be later than valid_to '{self.valid_to}'."
                )

        # Enforce deterministic hypothesis blindness
        validate_hypothesis_blindness(
            neutral_question=self.neutral_question,
            neutral_search_terms=self.neutral_search_terms,
            disconfirming_search_terms=self.disconfirming_search_terms,
        )
        return self


class EvidenceFeedbackBudget(BaseModel):
    """Deterministic budget tracker enforcing the 15% ceiling and max 1 round limits."""

    model_config = ConfigDict(frozen=True)

    baseline_budget_units: int = Field(default=20, ge=1, description="Original baseline research budget units")
    max_feedback_budget_ratio: float = Field(default=0.15, description="Maximum ratio of baseline budget for feedback (15%)")
    max_feedback_rounds: int = Field(default=1, description="Strict maximum feedback rounds per application run")
    max_requests_per_round: int = Field(default=3, description="Strict maximum requests allowed in one feedback round")
    feedback_budget_cap: int = Field(default=3, ge=1, description="Calculated integer cap on feedback units")
    feedback_units_consumed: int = Field(default=0, ge=0, description="Total units expended in feedback research")
    rounds_executed: int = Field(default=0, ge=0, description="Number of feedback rounds executed")
    requests_executed_count: int = Field(default=0, ge=0, description="Number of requests executed")

    @classmethod
    def create(cls, baseline_budget_units: int = 20) -> EvidenceFeedbackBudget:
        """Create budget with deterministically calculated 15% cap."""
        # 15% of baseline budget, minimum 1, floored to integer
        cap = max(1, int(baseline_budget_units * 0.15))
        return cls(
            baseline_budget_units=baseline_budget_units,
            feedback_budget_cap=cap,
            feedback_units_consumed=0,
            rounds_executed=0,
            requests_executed_count=0,
        )

    def can_execute_round(self) -> bool:
        """Check if an additional feedback round is permitted."""
        return self.rounds_executed < self.max_feedback_rounds

    def can_consume(self, cost_units: int = 1) -> bool:
        """Check if consuming cost_units would stay within the feedback budget cap."""
        return (self.feedback_units_consumed + cost_units) <= self.feedback_budget_cap

    def with_consumption(self, cost_units: int = 1, increment_request: bool = True) -> EvidenceFeedbackBudget:
        """Return a new budget model reflecting consumed units and request count."""
        if not self.can_consume(cost_units):
            raise ValueError(
                f"Budget exhaustion: consuming {cost_units} units would exceed feedback cap "
                f"({self.feedback_units_consumed + cost_units} > {self.feedback_budget_cap})."
            )
        return self.model_copy(
            update={
                "feedback_units_consumed": self.feedback_units_consumed + cost_units,
                "requests_executed_count": self.requests_executed_count + (1 if increment_request else 0),
            }
        )

    def with_round_completed(self) -> EvidenceFeedbackBudget:
        """Return budget reflecting round completion."""
        return self.model_copy(
            update={
                "rounds_executed": self.rounds_executed + 1,
            }
        )


class FeedbackRoundResult(BaseModel):
    """Auditable summary of an executed feedback round."""

    model_config = ConfigDict(frozen=True)

    round_number: int = Field(default=1)
    run_id: str
    requests: List[EvidenceRequest] = Field(default_factory=list)
    budget: EvidenceFeedbackBudget
    strategy_revised: bool = Field(default=False, description="True if new evidence caused the strategy to be revised")
    strategy_invalidated: bool = Field(default=False, description="True if new evidence invalidated the original strategy")
    status: str = Field(default="completed", description="completed, partial, exhausted, or deferred")
    summary: str = Field(default="")
