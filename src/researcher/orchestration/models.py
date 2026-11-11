from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance


class SufficiencyDimension(str, Enum):
    """The four epistemic dimensions of research sufficiency."""

    OBJECTIVE_COVERAGE = "objective_coverage"
    IDENTITY_CERTAINTY = "identity_certainty"
    EVIDENCE_ADEQUACY = "evidence_adequacy"
    MARGINAL_RESEARCH_VALUE = "marginal_research_value"


class CoverageStatus(str, Enum):
    """Epistemic coverage status of a research objective."""

    ADEQUATE = "adequate"
    PARTIAL = "partial"
    UNADDRESSED = "unaddressed"


class SearchViability(str, Enum):
    """Likelihood of obtaining further evidence on an objective via search."""

    PROMISING = "promising"
    UNCERTAIN = "uncertain"
    EXHAUSTED = "exhausted"


class PropositionType(str, Enum):
    """Semantic nature of the claim proposition."""

    SPEECH_ACT_ANNOUNCEMENT = "speech_act_announcement"
    COMMERCIAL_REALIZATION = "commercial_realization"
    EVALUATIVE_CLAIM = "evaluative_claim"


class ConflictSeverity(str, Enum):
    """Structured severity of an evidence contradiction."""

    CONFIRMED = "confirmed"
    POTENTIAL = "potential"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Actionable recommendation for researcher progression."""

    TARGETED_SEARCH = "targeted_search"
    IDENTITY_VERIFICATION = "identity_verification"
    CONFLICT_INVESTIGATION = "conflict_investigation"
    TERMINATE_SUFFICIENCY = "terminate_sufficiency"
    TERMINATE_BUDGET = "terminate_budget"


class SufficiencyDecision(str, Enum):
    """Final stopping or continuing verdict of an iteration."""

    SUFFICIENT = "sufficient"
    CONTINUE = "continue"
    INSUFFICIENT = "insufficient"


class ObjectiveAssessment(BaseModel):
    """Evaluation of an individual research objective."""

    model_config = ConfigDict(frozen=True)

    objective_id: str
    description: str
    coverage_status: CoverageStatus
    viability: SearchViability
    proposition_type: PropositionType = PropositionType.SPEECH_ACT_ANNOUNCEMENT
    is_critical: bool = True
    supporting_fact_ids: Tuple[str, ...] = ()
    missing_aspects: Tuple[str, ...] = ()


class DimensionAssessment(BaseModel):
    """Summary assessment for a single sufficiency dimension."""

    model_config = ConfigDict(frozen=True)

    dimension_name: SufficiencyDimension
    status: str
    summary: str
    details: Tuple[str, ...] = ()


class DimensionAssessments(BaseModel):
    """Structured container of assessments across all four dimensions."""

    model_config = ConfigDict(frozen=True)

    objective_coverage: DimensionAssessment
    identity_certainty: DimensionAssessment
    evidence_adequacy: DimensionAssessment
    marginal_research_value: DimensionAssessment


class RecommendedAction(BaseModel):
    """Actionable next step directed at an identified gap."""

    model_config = ConfigDict(frozen=True)

    action_type: ActionType
    target_objective_id: Optional[str] = None
    gap_description: str
    suggested_focus: str
    rationale: str


class MarginalValueAssessment(BaseModel):
    """Assessment of expected marginal returns from further searches."""

    model_config = ConfigDict(frozen=True)

    identity_resolution_viable: bool = False
    conflict_resolution_viable: bool = False
    unaddressed_critical_viable: bool = False
    diminishing_returns: bool = False


class ResearchSufficiencyAssessment(BaseModel):
    """Comprehensive, attributable sufficiency evaluation for an iteration."""

    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    iteration_id: str
    decision: SufficiencyDecision
    dimensions: DimensionAssessments
    objective_assessments: Tuple[ObjectiveAssessment, ...] = ()
    identity_uncertainties: Tuple[str, ...] = ()
    evidence_deficiencies: Tuple[str, ...] = ()
    unresolved_conflicts: Tuple[str, ...] = ()
    remaining_gaps: Tuple[str, ...] = ()
    recommended_actions: Tuple[RecommendedAction, ...] = ()
    explanation: str
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")

    @property
    def is_sufficient(self) -> bool:
        return self.decision == SufficiencyDecision.SUFFICIENT
