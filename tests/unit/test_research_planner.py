from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    TargetEntity,
)
from src.researcher.planning.planner import ResearchPlanner


@pytest.fixture
def sample_prospect():
    return ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
        linkedin_url="https://linkedin.com/in/erostova",
        location="Zurich, Switzerland",
    )


@pytest.fixture
def sample_outreach():
    return OutreachContext(
        objective="Invite as keynote speaker on robust alignment",
        sender_identity="Antigravity Team",
        relationship="cold",
        tone="academic and professional",
    )


def test_research_objective_validation():
    obj = ResearchObjective(
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Elena's current role and academic publications at Vanguard AI",
        priority=1,
        max_queries=3,
    )
    assert obj.priority == 1
    assert obj.max_queries == 3
    assert obj.is_critical is True

    non_crit = ResearchObjective(
        type=ResearchObjectiveType.PRODUCT_ORIENTED,
        target_entity=TargetEntity.COMPANY,
        description="Explore secondary peripheral tools used by the team",
        priority=6,
        is_critical=False,
    )
    assert non_crit.is_critical is False

    # Priority must be between 1 and 10
    with pytest.raises(ValidationError):
        ResearchObjective(
            type=ResearchObjectiveType.IDENTITY_RESOLUTION,
            target_entity=TargetEntity.PROSPECT,
            description="Short",
            priority=15,
        )


def test_research_plan_validation():
    obj1 = ResearchObjective(
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify role and career history",
    )
    obj2 = ResearchObjective(
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Investigate company funding and research lab setup",
    )
    plan = ResearchPlan(objectives=[obj1, obj2], total_query_budget=10)
    assert len(plan.objectives) == 2

    # Must contain at least one NEUTRAL_DISCOVERY or IDENTITY_RESOLUTION
    product_only_obj = ResearchObjective(
        type=ResearchObjectiveType.PRODUCT_ORIENTED,
        target_entity=TargetEntity.COMPANY,
        description="Check if company needs compiler tooling",
    )
    with pytest.raises(ValueError) as exc:
        ResearchPlan(objectives=[product_only_obj])
    assert "must contain at least one NEUTRAL_DISCOVERY or IDENTITY_RESOLUTION" in str(exc.value)


def test_planner_structured_output_parsing(sample_prospect, sample_outreach):
    valid_json = """{
        "session_id": "test-session-123",
        "objectives": [
            {
                "objective_id": "obj-1",
                "type": "identity_resolution",
                "target_entity": "prospect",
                "description": "Confirm current role at Vanguard AI",
                "priority": 1,
                "max_queries": 2
            },
            {
                "objective_id": "obj-2",
                "type": "neutral_discovery",
                "target_entity": "company",
                "description": "Recent funding rounds and team expansions",
                "priority": 2,
                "max_queries": 3
            }
        ],
        "total_query_budget": 8,
        "max_iterations": 3
    }"""

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            parsed = request.output_schema.parse(valid_json)
            return GenerationResponse(
                text=valid_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=valid_json),
                parsed_output=parsed,
            )

    planner = ResearchPlanner(engine=MockEngine())
    plan = planner.plan(
        prospect=sample_prospect,
        outreach=sample_outreach,
        session_id="custom-session-id",
    )

    assert isinstance(plan, ResearchPlan)
    assert plan.session_id == "custom-session-id"
    assert len(plan.objectives) == 2
    assert plan.objectives[0].type == ResearchObjectiveType.IDENTITY_RESOLUTION
    assert plan.objectives[0].target_entity == TargetEntity.PROSPECT
    assert plan.objectives[1].type == ResearchObjectiveType.NEUTRAL_DISCOVERY
    assert plan.objectives[1].target_entity == TargetEntity.COMPANY


def test_planner_handles_invalid_output(sample_prospect, sample_outreach):
    invalid_json = '{"objectives": "Not a list"}'

    class FailingEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            try:
                parsed = request.output_schema.parse(invalid_json)
            except Exception:
                parsed = None
            return GenerationResponse(
                text=invalid_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=invalid_json),
                parsed_output=parsed,
            )

    planner = ResearchPlanner(engine=FailingEngine())
    with pytest.raises(ValueError) as exc:
        planner.plan(prospect=sample_prospect, outreach=sample_outreach)
    assert "Failed to parse valid ResearchPlan" in str(exc.value)
