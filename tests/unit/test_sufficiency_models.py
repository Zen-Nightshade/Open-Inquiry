from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.researcher.orchestration.models import (
    ActionType,
    ConflictSeverity,
    CoverageStatus,
    DimensionAssessment,
    DimensionAssessments,
    ObjectiveAssessment,
    PropositionType,
    RecommendedAction,
    ResearchSufficiencyAssessment,
    SearchViability,
    SufficiencyDecision,
    SufficiencyDimension,
)


def test_objective_assessment_immutability():
    obj = ObjectiveAssessment(
        objective_id="obj-1",
        description="Discover core product milestones",
        coverage_status=CoverageStatus.ADEQUATE,
        viability=SearchViability.EXHAUSTED,
        proposition_type=PropositionType.COMMERCIAL_REALIZATION,
        is_critical=True,
        supporting_fact_ids=("fact-1", "fact-2"),
        missing_aspects=(),
    )
    assert obj.is_critical is True
    assert len(obj.supporting_fact_ids) == 2

    with pytest.raises(ValidationError):
        obj.coverage_status = CoverageStatus.PARTIAL  # type: ignore


def test_research_sufficiency_assessment_properties():
    dim_cov = DimensionAssessment(
        dimension_name=SufficiencyDimension.OBJECTIVE_COVERAGE,
        status="adequate",
        summary="All core objectives covered",
    )
    dim_id = DimensionAssessment(
        dimension_name=SufficiencyDimension.IDENTITY_CERTAINTY,
        status="confirmed",
        summary="Target prospect and company verified",
    )
    dim_ev = DimensionAssessment(
        dimension_name=SufficiencyDimension.EVIDENCE_ADEQUACY,
        status="adequate",
        summary="Independent sources corroborate facts",
    )
    dim_mv = DimensionAssessment(
        dimension_name=SufficiencyDimension.MARGINAL_RESEARCH_VALUE,
        status="diminished",
        summary="Diminishing returns on further queries",
    )
    dims = DimensionAssessments(
        objective_coverage=dim_cov,
        identity_certainty=dim_id,
        evidence_adequacy=dim_ev,
        marginal_research_value=dim_mv,
    )

    assessment = ResearchSufficiencyAssessment(
        session_id="sess-1",
        iteration_id="iter-1",
        decision=SufficiencyDecision.SUFFICIENT,
        dimensions=dims,
        explanation="Threshold satisfied",
    )

    assert assessment.is_sufficient is True
    assert assessment.decision == SufficiencyDecision.SUFFICIENT
    assert isinstance(assessment.remaining_gaps, tuple)

    with pytest.raises(ValidationError):
        assessment.decision = SufficiencyDecision.CONTINUE  # type: ignore
