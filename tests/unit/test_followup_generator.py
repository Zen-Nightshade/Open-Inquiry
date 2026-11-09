from __future__ import annotations

import pytest

from src.context.models import ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.orchestration.models import (
    ActionType,
    DimensionAssessment,
    DimensionAssessments,
    RecommendedAction,
    ResearchSufficiencyAssessment,
    SufficiencyDecision,
    SufficiencyDimension,
)
from src.researcher.search.followup_generator import FollowUpQueryGenerator
from src.researcher.search.query_generator import QueryListOutput


class MockLLMEngine(GeneratorEngine):
    def __init__(self, queries: list[str]):
        self.queries = queries

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        output = QueryListOutput(queries=self.queries) if self.queries else None
        return GenerationResponse(
            text="{}",
            provider_name="mock_groq",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=output,
        )


@pytest.fixture
def sample_prospect():
    return ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
    )


@pytest.fixture
def sample_assessment():
    dim = DimensionAssessment(
        dimension_name=SufficiencyDimension.OBJECTIVE_COVERAGE,
        status="partial",
        summary="Need deployment verification",
    )
    dims = DimensionAssessments(
        objective_coverage=dim,
        identity_certainty=dim,
        evidence_adequacy=dim,
        marginal_research_value=dim,
    )
    action = RecommendedAction(
        action_type=ActionType.TARGETED_SEARCH,
        target_objective_id="obj-commercial",
        gap_description="Independent verification required for production deployment",
        suggested_focus="Third-party customer case studies and news",
        rationale="Company announcements alone are insufficient",
    )
    return ResearchSufficiencyAssessment(
        session_id="s1",
        iteration_id="it-1",
        decision=SufficiencyDecision.CONTINUE,
        dimensions=dims,
        remaining_gaps=("Verified enterprise customer names",),
        recommended_actions=(action,),
        explanation="Continuing for deployment evidence",
    )


def test_followup_generator_success_and_deduplication(sample_prospect, sample_assessment):
    existing = [
        "vanguard ai customer case studies",
        "elena rostova vanguard ai",
    ]
    # LLM outputs 3 queries, one of which is already in existing
    mock_queries = [
        "vanguard ai customer case studies",  # Duplicate! Should be skipped
        "vanguard ai enterprise clients",
        "vanguard ai production deployment reviews",
    ]

    generator = FollowUpQueryGenerator(engine=MockLLMEngine(mock_queries))
    followup = generator.generate_followup_queries(
        prospect=sample_prospect,
        assessment=sample_assessment,
        existing_queries=existing,
        max_queries=2,
    )

    assert len(followup) == 2
    # Verify duplicate was skipped
    query_texts = [q.query_text.lower() for q in followup]
    assert "vanguard ai customer case studies" not in query_texts
    assert "vanguard ai enterprise clients" in query_texts
    assert "vanguard ai production deployment reviews" in query_texts
    # Verify objective link
    assert followup[0].objective_id == "obj-commercial"


def test_followup_generator_deterministic_fallback(sample_prospect, sample_assessment):
    # LLM returns empty list or fails
    generator = FollowUpQueryGenerator(engine=MockLLMEngine([]))
    followup = generator.generate_followup_queries(
        prospect=sample_prospect,
        assessment=sample_assessment,
        existing_queries=[],
        max_queries=3,
    )

    assert len(followup) > 0
    assert any("Vanguard AI" in q.query_text for q in followup)
