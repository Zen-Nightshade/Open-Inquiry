from __future__ import annotations

import datetime
import tempfile
import uuid
from typing import Any, List, Optional
import pytest

from src.context.models import OutreachContext, ProspectContext
from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.entity.resolver import IdentityResolutionResult
from src.researcher.entity.service import EntityResolutionService
from src.researcher.extraction.fetcher import PageFetcher
from src.researcher.extraction.models import FetchOutcome, FetchStatus
from src.researcher.extraction.service import FactExtractionService
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchPackage,
    ResolutionStatus,
    SourceRecord,
    SourceType,
)
from src.researcher.orchestration.evaluator import (
    RawObjectiveAssessment,
    RawRecommendedAction,
    RawSufficiencyEvaluation,
    ResearchSufficiencyEvaluator,
)
from src.researcher.orchestration.models import (
    ActionType,
    CoverageStatus,
    PropositionType,
    SearchViability,
    SufficiencyDecision,
)
from src.researcher.orchestration.orchestrator import Researcher
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    TargetEntity,
)
from src.researcher.planning.planner import ResearchPlanner
from src.researcher.search.followup_generator import FollowUpQueryGenerator
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.query_generator import QueryGenerator, QueryListOutput
from src.researcher.search.search_client import SearchClient


class MockSearchClient(SearchClient):
    def __init__(self, query_mapping: dict[str, list[SearchResult]]):
        self.query_mapping = query_mapping
        self.executed_queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self.executed_queries.append(query)
        results = self.query_mapping.get(query, [])
        return SearchResponse(query=query, status=SearchStatus.SUCCESS, results=results)


class MockPageFetcher(PageFetcher):
    def __init__(self, page_contents: dict[str, tuple[str, str, str]]):
        """Mapping from url -> (title, domain, raw_text)."""
        self.page_contents = page_contents
        self.fetched_urls: list[str] = []

    def fetch(self, url: str) -> FetchOutcome:
        self.fetched_urls.append(url)
        title, domain, text = self.page_contents.get(
            url, ("Default Title", "example.com", "Default body text")
        )
        source_rec = SourceRecord(
            url=url,
            domain=domain,
            title=title,
            raw_text=text,
            fetch_date=datetime.date(2026, 11, 1),
            source_type=SourceType.COMPANY_WEBSITE if "acme.com" in domain else SourceType.NEWS,
        )
        return FetchOutcome(
            url=url,
            status=FetchStatus.SUCCESS,
            source_record=source_rec,
            raw_html=f"<html><body>{text}</body></html>",
        )


class MockLLMEngine(GeneratorEngine):
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        output = self.responses.pop(0) if self.responses else None
        return GenerationResponse(
            text="{}",
            provider_name="mock_llm",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=output,
        )


@pytest.fixture
def workflow_state_store():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


def test_integrated_multi_turn_research_workflow(workflow_state_store):
    """
    End-to-end integration test of the multi-turn researcher lifecycle:
    - Iteration 1: neutral discovery finds company blog announcement (partial commercial realization).
    - Evaluator identifies gap and outputs a targeted_search RecommendedAction.
    - FollowUpQueryGenerator translates the recommended action into an enterprise deployment search query.
    - Iteration 2: searches and fetches third-party evidence, satisfying the critical objective.
    - Evaluator reaches SUFFICIENT.
    - Resulting ResearchPackage is complete, sufficient, deeply immutable, with all artifacts persisted.
    """
    session_id = str(uuid.uuid4())
    prospect = ProspectContext(
        full_name="Sarah Connor",
        company_name="Acme Robotics",
        company_domain="acme.com",
        designation="VP of Engineering",
    )
    outreach = OutreachContext(
        objective="Discuss enterprise automation platform partnership",
        sender_identity="Antigravity Alliances",
        relationship="cold",
        tone="direct",
    )

    # 1. Planner output
    obj1 = ResearchObjective(
        objective_id="obj-commercial",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify commercial deployment of Acme autonomous robotics in customer warehouses",
        is_critical=True,
    )
    plan = ResearchPlan(session_id=session_id, objectives=[obj1])
    planner = ResearchPlanner(engine=MockLLMEngine([plan]))

    # 2. Query generator for initial plan
    initial_queries = QueryListOutput(queries=["acme robotics warehouse deployment"])
    query_generator = QueryGenerator(engine=MockLLMEngine([initial_queries]))

    # 3. FollowUpQueryGenerator for Turn 2
    followup_queries = QueryListOutput(queries=["acme robotics customer case study warehouse"])
    followup_generator = FollowUpQueryGenerator(engine=MockLLMEngine([followup_queries]))

    # 4. Search and Fetch fixtures
    search_mapping = {
        "acme robotics warehouse deployment": [
            SearchResult(
                url="https://acme.com/blog/robotics-announcement",
                original_url="https://acme.com/blog/robotics-announcement",
                title="Acme Announces Autonomous Warehouse Fleet",
                snippet="Acme Robotics today announced its new autonomous warehouse fleet.",
                domain="acme.com",
                position=1,
                query="acme robotics warehouse deployment",
            )
        ],
        "acme robotics customer case study warehouse": [
            SearchResult(
                url="https://logisticsreport.com/acme-fedex-deployment",
                original_url="https://logisticsreport.com/acme-fedex-deployment",
                title="Logistics Report: FedEx Deploys Acme Fleet in 5 Hubs",
                snippet="Independent case study: FedEx deployed 200 Acme autonomous robots across five regional hubs.",
                domain="logisticsreport.com",
                position=1,
                query="acme robotics customer case study warehouse",
            )
        ],
    }
    search_client = MockSearchClient(query_mapping=search_mapping)

    page_contents = {
        "https://acme.com/blog/robotics-announcement": (
            "Acme Announces Autonomous Warehouse Fleet",
            "acme.com",
            "Acme Robotics announces its autonomous warehouse fleet is available for commercial deployment.",
        ),
        "https://logisticsreport.com/acme-fedex-deployment": (
            "Logistics Report: FedEx Deploys Acme Fleet in 5 Hubs",
            "logisticsreport.com",
            "Independent case study: FedEx deployed 200 Acme autonomous robots across five regional hubs with 40% throughput increase.",
        ),
    }
    page_fetcher = MockPageFetcher(page_contents=page_contents)

    # 5. Entity Resolution mock (symmetrically resolved)
    class TestEntityService(EntityResolutionService):
        def resolve_entities_for_session(self, session_id, prospect, sources, search_results=None):
            return IdentityResolutionResult(
                target_prospect=prospect,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Sarah Connor confirmed at Acme Robotics via corroboration.",
                prospect_entity=EntityRecord(
                    entity_id="ent-sarah",
                    entity_type=EntityType.PERSON,
                    raw_name=prospect.full_name,
                    canonical_name="Sarah Connor",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                company_entity=EntityRecord(
                    entity_id="ent-acme",
                    entity_type=EntityType.ORGANIZATION,
                    raw_name=prospect.company_name,
                    canonical_name="Acme Robotics Inc.",
                    domain="acme.com",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
            )

    # 6. Fact Extraction mock: returns fact specific to fetched snapshot
    class TestFactService(FactExtractionService):
        def extract_and_persist_facts_for_session(self, session_id, snapshot_id, known_entities=None, accumulator=None):
            # Check what snapshot was fetched
            with self.state_store.uow() as uow:
                snap = uow.sources.get_snapshot(snapshot_id)
                src = uow.sources.get_source(snap["source_id"]) if snap else None

            if not snap or not src:
                return []

            if "acme.com" in src["domain"]:
                fact_id = "fact-acme-blog"
                claim = "Acme Robotics announced autonomous warehouse fleet availability"
                category = FactCategory.PRODUCT_LAUNCH
            else:
                fact_id = "fact-fedex-case-study"
                claim = "FedEx deployed 200 Acme autonomous robots across five regional hubs"
                category = FactCategory.ACHIEVEMENT

            ev = EvidenceRecord(
                source_id=src["source_id"],
                verbatim_excerpt=claim,
                confidence=ConfidenceLevel.HIGH,
            )
            fact = CanonicalFact(
                fact_id=fact_id,
                claim_text=claim,
                fact_category=category,
                evidence_records=[ev],
                composite_confidence=ConfidenceLevel.HIGH,
            )

            with self.state_store.uow() as uow:
                uow.assertions.create_canonical_fact(
                    fact_id=fact.fact_id,
                    session_id=session_id,
                    claim_text=fact.claim_text,
                    fact_category=fact.fact_category.value,
                    composite_confidence="high",
                )
                uow.sources.create_evidence_passage(
                    evidence_id=ev.evidence_id,
                    snapshot_id=snapshot_id,
                    source_id=src["source_id"],
                    verbatim_excerpt=ev.verbatim_excerpt,
                )
                uow.assertions.link_fact_evidence(
                    fact_id=fact.fact_id,
                    evidence_id=ev.evidence_id,
                )

            if accumulator:
                src_rec = SourceRecord(
                    source_id=src["source_id"],
                    url=src["canonical_url"],
                    domain=src["domain"],
                    raw_text=snap.get("cleaned_text") or snap.get("raw_text") or "",
                    fetch_date=datetime.date(2026, 11, 1),
                )
                accumulator.add_source(src_rec)
                accumulator.merge_or_add_fact(fact)

            return [fact]

    # 7. Evaluator raw evaluations: Turn 1 -> CONTINUE with RecommendedAction; Turn 2 -> SUFFICIENT
    eval_turn_1 = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-commercial",
                coverage_status="partial",
                viability="promising",
                proposition_type="commercial_realization",
                supporting_fact_ids=["fact-acme-blog"],
                missing_aspects=["Independent third-party verification of customer adoption in production"],
            )
        ],
        remaining_gaps=["Independent third-party verification of customer adoption in production"],
        recommended_actions=[
            RawRecommendedAction(
                action_type="targeted_search",
                target_objective_id="obj-commercial",
                gap_description="Need customer case study",
                suggested_focus="acme robotics customer case study warehouse",
                rationale="Commercial realization claims require third-party operational corroboration",
            )
        ],
        identity_resolution_viable=True,
    )

    eval_turn_2 = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-commercial",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="commercial_realization",
                supporting_fact_ids=["fact-fedex-case-study"],
                missing_aspects=[],
            )
        ],
        remaining_gaps=[],
        explanation="Commercial deployment successfully verified via third-party logistics case study.",
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockLLMEngine([eval_turn_1, eval_turn_2]))

    researcher = Researcher(
        state_store=workflow_state_store,
        planner=planner,
        query_generator=query_generator,
        followup_generator=followup_generator,
        search_client=search_client,
        page_fetcher=page_fetcher,
        entity_service=TestEntityService(state_store=workflow_state_store),
        fact_service=TestFactService(state_store=workflow_state_store),
        evaluator=evaluator,
    )

    # Execute workflow
    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        session_id=session_id,
        max_iterations=3,
    )

    # 8. Assertions on ResearchPackage
    assert package is not None
    assert package.is_complete is True
    assert package.is_sufficient is True
    assert len(package.missing_information) == 0

    # Verify both queries were used
    assert len(package.search_queries_used) == 2
    assert "acme robotics warehouse deployment" in package.search_queries_used
    assert "acme robotics customer case study warehouse" in package.search_queries_used

    # Verify both facts are present in package
    fact_ids = {f.fact_id for f in package.facts}
    assert "fact-acme-blog" in fact_ids
    assert "fact-fedex-case-study" in fact_ids

    # Verify entities
    assert package.prospect_entity is not None
    assert package.prospect_entity.canonical_name == "Sarah Connor"
    assert package.company_entity is not None
    assert package.company_entity.canonical_name == "Acme Robotics Inc."

    # 9. Assertions on Database State
    with workflow_state_store.uow() as uow:
        # Research session marked completed
        session_row = uow.research.get_session(session_id)
        assert session_row is not None
        assert session_row["status"] == "completed"

        # Iterations table has 2 completed iterations
        iterations = uow.orchestration.list_iterations(session_id)
        assert len(iterations) == 2
        assert iterations[0]["iteration_number"] == 1
        assert iterations[0]["status"] == "completed"
        assert iterations[1]["iteration_number"] == 2
        assert iterations[1]["status"] == "completed"

        # Sufficiency assessments table has 2 entries matching iteration sequence
        assessments = uow.orchestration.list_assessments(session_id)
        assert len(assessments) == 2
        assert assessments[0]["iteration_number"] == 1
        assert assessments[0]["decision"] == "continue"
        assert assessments[1]["iteration_number"] == 2
        assert assessments[1]["decision"] == "sufficient"

        # Source snapshots persisted
        snapshots = uow.sources.get_snapshots_for_session(session_id)
        assert len(snapshots) == 2
