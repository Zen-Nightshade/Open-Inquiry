from __future__ import annotations

import pytest

from src.context.models import ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    TargetEntity,
)
from src.researcher.search.query_generator import (
    GeneratedQuery,
    QueryGenerator,
    QueryListOutput,
)


@pytest.fixture
def sample_prospect():
    return ProspectContext(
        full_name="Marcus Vance",
        company_name="Apex Quantum",
        company_domain="apexquantum.tech",
        designation="Chief Technology Officer",
    )


def test_query_generator_traceability_and_format(sample_prospect):
    objective = ResearchObjective(
        objective_id="obj-target-42",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Marcus Vance's role and history at Apex Quantum",
        max_queries=2,
    )

    mock_json = """{
        "queries": [
            "Marcus Vance Apex Quantum CTO",
            "Marcus Vance career history LinkedIn"
        ]
    }"""

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            parsed = request.output_schema.parse(mock_json)
            return GenerationResponse(
                text=mock_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=mock_json),
                parsed_output=parsed,
            )

    generator = QueryGenerator(engine=MockEngine())
    queries = generator.generate_queries(objective=objective, prospect=sample_prospect)

    assert len(queries) == 2
    for q in queries:
        assert isinstance(q, GeneratedQuery)
        assert q.objective_id == "obj-target-42"
        assert q.target_entity == TargetEntity.PROSPECT
        assert len(q.query_text) > 3


def test_query_generator_deduplication_and_existing_queries(sample_prospect):
    objective = ResearchObjective(
        objective_id="obj-company-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Discover Apex Quantum products and funding",
        max_queries=4,
    )

    # LLM returned duplicates and a query that caller already ran
    mock_json = """{
        "queries": [
            "Apex Quantum funding announcements",
            "Apex Quantum funding announcements",
            "ALREADY EXECUTED QUERY",
            "Apex Quantum new product release"
        ]
    }"""

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            parsed = request.output_schema.parse(mock_json)
            return GenerationResponse(
                text=mock_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=mock_json),
                parsed_output=parsed,
            )

    generator = QueryGenerator(engine=MockEngine())
    queries = generator.generate_queries(
        objective=objective,
        prospect=sample_prospect,
        existing_queries=["already executed query"],
    )

    query_texts = [q.query_text for q in queries]
    assert len(query_texts) == 2
    assert "Apex Quantum funding announcements" in query_texts
    assert "Apex Quantum new product release" in query_texts
    assert "ALREADY EXECUTED QUERY" not in query_texts


def test_query_cleaning_removes_unwanted_formatting(sample_prospect):
    objective = ResearchObjective(
        objective_id="obj-clean-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Test cleaning",
        max_queries=3,
    )

    mock_json = """{
        "queries": [
            "- \\"Apex Quantum Series A\\"",
            "* 'Apex Quantum leadership'",
            "`Apex Quantum customers`"
        ]
    }"""

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            parsed = request.output_schema.parse(mock_json)
            return GenerationResponse(
                text=mock_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=mock_json),
                parsed_output=parsed,
            )

    generator = QueryGenerator(engine=MockEngine())
    queries = generator.generate_queries(objective=objective, prospect=sample_prospect)

    assert queries[0].query_text == "Apex Quantum Series A"
    assert queries[1].query_text == "Apex Quantum leadership"
    assert queries[2].query_text == "Apex Quantum customers"
