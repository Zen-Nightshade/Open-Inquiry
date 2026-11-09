from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, List, Optional
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ResearchObjectiveType(str, Enum):
    """Categorization of research objective."""

    NEUTRAL_DISCOVERY = "neutral_discovery"
    PRODUCT_ORIENTED = "product_oriented"
    IDENTITY_RESOLUTION = "identity_resolution"


class TargetEntity(str, Enum):
    """Subject of the research objective."""

    PROSPECT = "prospect"
    COMPANY = "company"
    BOTH = "both"


class ResearchObjective(BaseModel):
    """Structured research objective targeting an individual prospect or organization."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    objective_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ResearchObjectiveType = Field(
        default=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        validation_alias=AliasChoices("type", "objective_type", "category"),
    )
    target_entity: TargetEntity = Field(
        default=TargetEntity.COMPANY,
        validation_alias=AliasChoices("target_entity", "entity", "target", "subject"),
    )
    description: str = Field(
        min_length=5,
        description="Clear, actionable statement of what information to uncover",
    )
    temporal_scope: Optional[str] = Field(
        default=None,
        description="Temporal restriction, e.g. 'last_12_months', 'recent', 'career_history'",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority ranking from 1 (highest) to 10 (lowest)",
    )
    is_critical: bool = Field(
        default=True,
        description="Whether this research objective is critical for outreach sufficiency",
    )
    max_queries: int = Field(
        default=3,
        ge=1,
        le=8,
        validation_alias=AliasChoices("max_queries", "query_budget", "queries", "budget"),
        description="Maximum search queries to allocate to this objective",
    )

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean = v.strip().lower()
            if "identity" in clean:
                return ResearchObjectiveType.IDENTITY_RESOLUTION
            elif "product" in clean:
                return ResearchObjectiveType.PRODUCT_ORIENTED
            elif "neutral" in clean or "discovery" in clean:
                return ResearchObjectiveType.NEUTRAL_DISCOVERY
        return v

    @field_validator("target_entity", mode="before")
    @classmethod
    def _validate_target_entity(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("prospect", "company", "both"):
                return TargetEntity(clean)
            if any(term in clean for term in ("person", "individual", "founder", "executive", "lead", "prospect", "name")):
                return TargetEntity.PROSPECT
            if any(term in clean for term in ("company", "org", "firm", "business", "enterprise", "team")):
                return TargetEntity.COMPANY
            # Fallback based on typical multi-objective structure
            return TargetEntity.COMPANY
        return v


class ResearchPlan(BaseModel):
    """Structured plan orchestrating the research phase."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objectives: List[ResearchObjective] = Field(
        min_length=1,
        description="Ordered list of research objectives to pursue",
    )
    total_query_budget: int = Field(
        default=12,
        ge=1,
        description="Maximum total queries allowed across all objectives",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum turns of iterative gap-filling searches",
    )

    @model_validator(mode="after")
    def validate_plan_composition(self) -> ResearchPlan:
        """Ensure the plan contains necessary baseline objective types."""
        types = {obj.type for obj in self.objectives}
        # Must have at least one neutral discovery or identity resolution objective
        if (
            ResearchObjectiveType.NEUTRAL_DISCOVERY not in types
            and ResearchObjectiveType.IDENTITY_RESOLUTION not in types
        ):
            raise ValueError(
                "ResearchPlan must contain at least one NEUTRAL_DISCOVERY or IDENTITY_RESOLUTION objective"
            )
        return self
