from __future__ import annotations

import math
from pydantic import BaseModel, ConfigDict, Field


class GlobalResearchBudget(BaseModel):
    """Unified per-run research budget shared across baseline research and feedback research.
    
    Semantics:
      - 1 budget unit = 1 actual search-provider query execution (supporting or disconfirming).
      - Query generation without search execution consumes 0 units.
      - Page fetches and content extractions do not consume query units.
      - A retry that issues a live search consumes 1 unit.
      - The feedback allowance is a strict sub-limit: floor(0.15 * total_budget_units).
      - Feedback research can only consume units if BOTH remaining global budget > 0 
        AND remaining feedback allowance > 0.
      - The budget NEVER resets across stages, DAG construction, or objective adaptation.
    """

    model_config = ConfigDict(frozen=True)

    total_budget_units: int = Field(default=20, ge=1, description="Configured total query executions allowed for this run")
    baseline_units_consumed: int = Field(default=0, ge=0, description="Query units consumed in baseline research")
    feedback_units_consumed: int = Field(default=0, ge=0, description="Query units consumed in feedback research")
    max_feedback_budget_ratio: float = Field(default=0.15, description="Maximum proportion of total budget allowed for feedback (15%)")
    max_feedback_rounds: int = Field(default=1, description="Strict maximum feedback rounds per application run")
    max_requests_per_round: int = Field(default=3, description="Strict maximum requests allowed in one feedback round")
    feedback_rounds_executed: int = Field(default=0, ge=0, description="Number of feedback rounds completed")
    feedback_requests_executed: int = Field(default=0, ge=0, description="Number of individual feedback requests executed")

    @property
    def feedback_budget_cap(self) -> int:
        """Deterministic integer ceiling on feedback query units."""
        return max(1, math.floor(self.total_budget_units * self.max_feedback_budget_ratio))

    @property
    def total_units_consumed(self) -> int:
        """Total query units consumed across all research phases."""
        return self.baseline_units_consumed + self.feedback_units_consumed

    @property
    def remaining_total_units(self) -> int:
        """Remaining queries available across the entire application run."""
        return max(0, self.total_budget_units - self.total_units_consumed)

    @property
    def remaining_feedback_units(self) -> int:
        """Queries available for feedback research, bounded by both the feedback cap and global remainder."""
        cap_remaining = max(0, self.feedback_budget_cap - self.feedback_units_consumed)
        return min(self.remaining_total_units, cap_remaining)

    @property
    def is_exhausted(self) -> bool:
        """True if the global research budget has been fully expended."""
        return self.remaining_total_units <= 0

    @property
    def is_feedback_exhausted(self) -> bool:
        """True if feedback research cannot consume any more units."""
        return self.remaining_feedback_units <= 0

    def can_consume_baseline(self, cost_units: int = 1) -> bool:
        """Check if baseline research can consume the requested units without exceeding the global budget."""
        if cost_units < 0:
            return False
        return self.total_units_consumed + cost_units <= self.total_budget_units

    def consume_baseline(self, cost_units: int = 1) -> GlobalResearchBudget:
        """Return a new budget model reflecting baseline research query consumption."""
        if not self.can_consume_baseline(cost_units):
            raise ValueError(
                f"Global budget exhaustion: consuming {cost_units} baseline units would exceed total budget "
                f"({self.total_units_consumed + cost_units} > {self.total_budget_units})."
            )
        return self.model_copy(
            update={
                "baseline_units_consumed": self.baseline_units_consumed + cost_units,
            }
        )

    def can_consume_feedback(self, cost_units: int = 1) -> bool:
        """Check if feedback research can consume the requested units respecting both global and feedback caps."""
        if cost_units < 0:
            return False
        if self.feedback_rounds_executed >= self.max_feedback_rounds:
            return False
        if (self.total_units_consumed + cost_units) > self.total_budget_units:
            return False
        if (self.feedback_units_consumed + cost_units) > self.feedback_budget_cap:
            return False
        return True

    def consume_feedback(self, cost_units: int = 1, increment_request: bool = True) -> GlobalResearchBudget:
        """Return a new budget model reflecting feedback research consumption."""
        if not self.can_consume_feedback(cost_units):
            raise ValueError(
                f"Budget violation: cannot consume {cost_units} feedback units (remaining total: "
                f"{self.remaining_total_units}, remaining feedback: {self.remaining_feedback_units})."
            )
        return self.model_copy(
            update={
                "feedback_units_consumed": self.feedback_units_consumed + cost_units,
                "feedback_requests_executed": self.feedback_requests_executed + (1 if increment_request else 0),
            }
        )

    def complete_feedback_round(self) -> GlobalResearchBudget:
        """Return a budget model with the feedback round incremented."""
        return self.model_copy(
            update={
                "feedback_rounds_executed": self.feedback_rounds_executed + 1,
            }
        )
