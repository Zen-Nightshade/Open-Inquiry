from __future__ import annotations

import datetime
import tempfile
import uuid
from typing import List, Optional
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.entity.candidates import DiscoveredEntityCandidate
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
)
from src.researcher.orchestration.evaluator import (
    RawObjectiveAssessment,
    RawRecommendedAction,
    RawSufficiencyEvaluation,
    ResearchSufficiencyEvaluator,
)
from src.researcher.orchestration.models import (
    ActionType,
    DimensionAssessment,
    DimensionAssessments,
    RecommendedAction,
    ResearchSufficiencyAssessment,
    SearchViability,
    SufficiencyDecision,
    SufficiencyDimension,
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
from src.researcher.search.query_generator import GeneratedQuery, QueryGenerator, QueryListOutput
from src.researcher.search.search_client import SearchClient


class MockSearchClient(SearchClient):
    def __init__(self, query_to_results: Optional[dict[str, list[SearchResult]]] = None):
        self.query_to_results = query_to_results or {}

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        results = self.query_to_results.get(
            query,
            [
                SearchResult(
                    url="https://vanguardai.io/about",
                    original_url="https://vanguardai.io/about",
                    title="Vanguard AI About Us",
                    snippet="Vanguard AI building enterprise AI systems.",
                    domain="vanguardai.io",
                    position=1,
                    query=query,
                )
            ],
        )
        return SearchResponse(query=query, status=SearchStatus.SUCCESS, results=results)


class MockPageFetcher(PageFetcher):
    def __init__(self, url_to_html: Optional[dict[str, str]] = None):
        self.url_to_html = url_to_html or {}

    def fetch(self, url: str) -> FetchOutcome:
        text = self.url_to_html.get(url, "Default mock page body content.")
        source_rec = SourceRecord(
            url=url,
            domain="vanguardai.io" if "vanguardai.io" in url else "techcrunch.com",
            title="Mock Title",
            raw_text=text,
            fetch_date=datetime.date(2026, 11, 1),
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
            provider_name="mock_groq",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=output,
        )


@pytest.fixture
def test_state_store():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


@pytest.fixture
def prospect_and_outreach():
    prospect = ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
    )
    outreach = OutreachContext(
        objective="Invite as keynote speaker",
        sender_identity="Antigravity Team",
        relationship="cold",
        tone="professional",
    )
    return prospect, outreach


def test_researcher_multi_turn_continuation_and_sufficiency(test_state_store, prospect_and_outreach):
    """Test full multi-turn research lifecycle: Turn 1 CONTINUE -> Turn 2 SUFFICIENT."""
    prospect, outreach = prospect_and_outreach
    session_id = str(uuid.uuid4())

    # Turn 1: Partial discovery, recommends targeted search on customer deployment
    eval_turn_1 = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="partial",
                viability="promising",
                proposition_type="commercial_realization",
                supporting_fact_ids=[],
                missing_aspects=["Verified enterprise customer case studies"],
            )
        ],
        remaining_gaps=["Verified enterprise customer case studies"],
        recommended_actions=[
            RawRecommendedAction(
                action_type="targeted_search",
                target_objective_id="obj-1",
                gap_description="Need customer case studies",
                suggested_focus="Vanguard enterprise customer deployments",
                rationale="Confirm production adoption",
            )
        ],
        identity_resolution_viable=True,
    )

    # Turn 2: Adequate coverage, SUFFICIENT
    eval_turn_2 = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="commercial_realization",
                supporting_fact_ids=[],  # will be verified
            )
        ],
        remaining_gaps=[],
        explanation="All critical objectives satisfied with verified deployment evidence.",
    )

    # Setup researcher with mocked components
    # Planner generates 1 critical objective
    obj1 = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify Vanguard enterprise customer deployments",
        is_critical=True,
    )
    plan = ResearchPlan(session_id=session_id, objectives=[obj1])

    planner_engine = MockLLMEngine([plan])
    planner = ResearchPlanner(engine=planner_engine)

    query_gen_engine = MockLLMEngine([QueryListOutput(queries=["vanguard ai customers"])])
    query_generator = QueryGenerator(engine=query_gen_engine)

    followup_engine = MockLLMEngine([QueryListOutput(queries=["vanguard ai enterprise case studies"])])
    followup_generator = FollowUpQueryGenerator(engine=followup_engine)

    evaluator_engine = MockLLMEngine([eval_turn_1, eval_turn_2])
    evaluator = ResearchSufficiencyEvaluator(engine=evaluator_engine)

    # Setup mocked search & fetch
    search_client = MockSearchClient(
        query_to_results={
            "vanguard ai customers": [
                SearchResult(
                    url="https://techcrunch.com/vanguard-b2b",
                    original_url="https://techcrunch.com/vanguard-b2b",
                    title="Vanguard AI B2B",
                    snippet="Vanguard serves enterprise customers.",
                    domain="techcrunch.com",
                    position=1,
                    query="vanguard ai customers",
                )
            ],
            "vanguard ai enterprise case studies": [
                SearchResult(
                    url="https://forbes.com/vanguard-case-study",
                    original_url="https://forbes.com/vanguard-case-study",
                    title="Forbes Vanguard Case Study",
                    snippet="Fortune 500 company deploys Vanguard AI.",
                    domain="forbes.com",
                    position=1,
                    query="vanguard ai enterprise case studies",
                )
            ],
        }
    )

    page_fetcher = MockPageFetcher(
        url_to_html={
            "https://techcrunch.com/vanguard-b2b": "Elena Rostova is Head of Research at Vanguard AI.",
            "https://forbes.com/vanguard-case-study": "Vanguard AI deployed across 10 global enterprises.",
        }
    )

    # Mock entity service to resolve confirmed entities
    class MockEntityService(EntityResolutionService):
        def resolve_entities_for_session(self, session_id, prospect, sources, search_results=None):
            return IdentityResolutionResult(
                target_prospect=prospect,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Mock resolved for testing",
                prospect_entity=EntityRecord(
                    entity_id="p-elena",
                    entity_type=EntityType.PERSON,
                    raw_name=prospect.full_name,
                    canonical_name="Elena Rostova",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                company_entity=EntityRecord(
                    entity_id="c-vanguard",
                    entity_type=EntityType.ORGANIZATION,
                    raw_name=prospect.company_name,
                    canonical_name="Vanguard AI Inc.",
                    domain="vanguardai.io",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                candidates=[],
                assessments=[],
            )

    # Mock fact service to extract facts with third-party source
    class MockFactService(FactExtractionService):
        def extract_and_persist_facts_for_session(self, session_id, snapshot_id, known_entities=None, accumulator=None):
            ev = EvidenceRecord(
                source_id="src-forbes",
                verbatim_excerpt="Vanguard AI deployed across 10 global enterprises.",
            )
            fact = CanonicalFact(
                fact_id="f-deploy-1",
                claim_text="Vanguard AI deployed across 10 global enterprises",
                fact_category=FactCategory.PRODUCT_LAUNCH,
                evidence_records=[ev],
                composite_confidence=ConfidenceLevel.HIGH,
            )
            src = SourceRecord(
                source_id="src-forbes",
                url="https://forbes.com/vanguard-case-study",
                domain="forbes.com",
                raw_text="Forbes article",
                fetch_date=datetime.date(2026, 11, 1),
            )
            with self.state_store.uow() as uow:
                uow.sources.upsert_source(
                    source_id="src-forbes",
                    canonical_url="https://forbes.com/vanguard-case-study",
                    domain="forbes.com",
                )
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
                    source_id="src-forbes",
                    verbatim_excerpt=ev.verbatim_excerpt,
                )
                uow.assertions.link_fact_evidence(
                    fact_id=fact.fact_id,
                    evidence_id=ev.evidence_id,
                )
            if accumulator:
                accumulator.add_source(src)
                accumulator.merge_or_add_fact(fact)
            return [fact]

    # Pre-wire Turn 2 supporting fact id
    eval_turn_2.objective_assessments[0].supporting_fact_ids = ["f-deploy-1"]

    researcher = Researcher(
        state_store=test_state_store,
        planner=planner,
        query_generator=query_generator,
        followup_generator=followup_generator,
        search_client=search_client,
        page_fetcher=page_fetcher,
        entity_service=MockEntityService(state_store=test_state_store),
        fact_service=MockFactService(state_store=test_state_store),
        evaluator=evaluator,
    )

    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        session_id=session_id,
        max_iterations=3,
    )

    assert package is not None
    assert isinstance(package, ResearchPackage)
    assert package.is_complete is True
    assert isinstance(package.facts, tuple)
    assert isinstance(package.sources, tuple)
    assert isinstance(package.search_queries_used, tuple)
    assert len(package.search_queries_used) >= 2  # initial + follow-up

    # Verify iterations recorded in DB
    with test_state_store.uow() as uow:
        iterations = uow.orchestration.list_iterations(session_id)
        assert len(iterations) >= 1
        assessments = uow.orchestration.list_assessments(session_id)
        assert len(assessments) >= 1


def test_researcher_recovery_does_not_burn_iterations(test_state_store, prospect_and_outreach):
    """If iteration 1 fails midway, resuming re-executes iteration 1 without burning turn limit."""
    prospect, outreach = prospect_and_outreach
    session_id = str(uuid.uuid4())

    # Pre-seed an in-progress or failed iteration 1
    with test_state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id="p1",
            full_name=prospect.full_name,
            company_name=prospect.company_name,
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id="p1",
            context_id="c1",
        )
        uow.orchestration.create_iteration(
            iteration_id=f"{session_id}_it_1",
            session_id=session_id,
            iteration_number=1,
            status="failed",
            error_message="Network timeout during search",
        )

    # Setup researcher with successful single-turn evaluation
    eval_success = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="speech_act_announcement",
                supporting_fact_ids=[],
            )
        ],
        remaining_gaps=[],
        explanation="Complete",
    )

    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Elena role",
        is_critical=True,
    )
    plan = ResearchPlan(session_id=session_id, objectives=[obj])

    planner = ResearchPlanner(engine=MockLLMEngine([plan]))
    query_generator = QueryGenerator(engine=MockLLMEngine([QueryListOutput(queries=["elena vanguard"])]))
    evaluator = ResearchSufficiencyEvaluator(engine=MockLLMEngine([eval_success]))

    class MockEntityService(EntityResolutionService):
        def resolve_entities_for_session(self, session_id, prospect, sources, search_results=None):
            return IdentityResolutionResult(
                target_prospect=prospect,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Mock resolved for recovery testing",
                prospect_entity=EntityRecord(
                    entity_id="p-1",
                    entity_type=EntityType.PERSON,
                    raw_name=prospect.full_name,
                    canonical_name="Elena Rostova",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                company_entity=EntityRecord(
                    entity_id="c-1",
                    entity_type=EntityType.ORGANIZATION,
                    raw_name=prospect.company_name,
                    canonical_name="Vanguard AI",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                candidates=[],
                assessments=[],
            )

    class MockFactService(FactExtractionService):
        def extract_and_persist_facts_for_session(self, session_id, snapshot_id, known_entities=None, accumulator=None):
            fact = CanonicalFact(
                fact_id="f1",
                claim_text="Elena is Head of Research",
                evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Elena is Head of Research")],
            )
            if accumulator:
                accumulator.facts.append(fact)
            return [fact]

    eval_success.objective_assessments[0].supporting_fact_ids = ["f1"]

    researcher = Researcher(
        state_store=test_state_store,
        planner=planner,
        query_generator=query_generator,
        search_client=MockSearchClient(),
        page_fetcher=MockPageFetcher(),
        entity_service=MockEntityService(state_store=test_state_store),
        fact_service=MockFactService(state_store=test_state_store),
        evaluator=evaluator,
    )

    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        session_id=session_id,
        max_iterations=3,
    )

    assert package is not None
    # Verify iteration 1 was re-executed and completed, not burned into iteration 2
    with test_state_store.uow() as uow:
        it1 = uow.orchestration.get_iteration(f"{session_id}_it_1")
        assert it1["status"] == "completed"
        # Total iterations should remain 1
        all_iters = uow.orchestration.list_iterations(session_id)
        assert len(all_iters) == 1


def test_researcher_budget_exhaustion_yields_insufficient_package(test_state_store, prospect_and_outreach):
    """When max_iterations is reached without reaching full sufficiency, terminate gracefully with INSUFFICIENT package."""
    prospect, outreach = prospect_and_outreach
    session_id = str(uuid.uuid4())

    eval_turn_1 = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="partial",
                viability="promising",
                proposition_type="commercial_realization",
                supporting_fact_ids=[],
                missing_aspects=["Pricing tiers"],
            )
        ],
        remaining_gaps=["Pricing tiers"],
        recommended_actions=[
            RawRecommendedAction(
                action_type="targeted_search",
                target_objective_id="obj-1",
                gap_description="Pricing tiers missing",
                suggested_focus="Vanguard AI pricing",
                rationale="Investigate pricing model",
            )
        ],
        identity_resolution_viable=True,
    )

    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify pricing tiers",
        is_critical=True,
    )
    plan = ResearchPlan(session_id=session_id, objectives=[obj])

    planner = ResearchPlanner(engine=MockLLMEngine([plan]))
    query_generator = QueryGenerator(engine=MockLLMEngine([QueryListOutput(queries=["vanguard ai pricing"])]))
    followup_generator = FollowUpQueryGenerator(engine=MockLLMEngine([QueryListOutput(queries=["vanguard ai enterprise pricing plans"])]))
    evaluator = ResearchSufficiencyEvaluator(engine=MockLLMEngine([eval_turn_1]))

    class MockEntityService(EntityResolutionService):
        def resolve_entities_for_session(self, session_id, prospect, sources, search_results=None):
            return IdentityResolutionResult(
                target_prospect=prospect,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Resolved",
                prospect_entity=EntityRecord(
                    entity_id="p-1",
                    entity_type=EntityType.PERSON,
                    raw_name=prospect.full_name,
                    canonical_name="Elena Rostova",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                company_entity=EntityRecord(
                    entity_id="c-1",
                    entity_type=EntityType.ORGANIZATION,
                    raw_name=prospect.company_name,
                    canonical_name="Vanguard AI",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
            )

    class MockFactService(FactExtractionService):
        def extract_and_persist_facts_for_session(self, session_id, snapshot_id, known_entities=None, accumulator=None):
            return []

    researcher = Researcher(
        state_store=test_state_store,
        planner=planner,
        query_generator=query_generator,
        followup_generator=followup_generator,
        search_client=MockSearchClient(),
        page_fetcher=MockPageFetcher(),
        entity_service=MockEntityService(state_store=test_state_store),
        fact_service=MockFactService(state_store=test_state_store),
        evaluator=evaluator,
    )

    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        session_id=session_id,
        max_iterations=1,  # Budget = 1 turn
    )

    assert package is not None
    assert package.is_complete is True
    assert package.is_sufficient is False
    assert "Pricing tiers" in package.missing_information
    assert isinstance(package.facts, tuple)
    assert isinstance(package.sources, tuple)
    assert isinstance(package.search_queries_used, tuple)
    assert isinstance(package.missing_information, tuple)


def test_researcher_deep_immutability(test_state_store, prospect_and_outreach):
    """Verify that ResearchPackage collections cannot be mutated."""
    prospect, outreach = prospect_and_outreach
    session_id = str(uuid.uuid4())

    eval_complete = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="speech_act_announcement",
                supporting_fact_ids=["f-imm"],
            )
        ],
        remaining_gaps=[],
        explanation="Adequate",
    )

    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify role",
        is_critical=True,
    )
    plan = ResearchPlan(session_id=session_id, objectives=[obj])

    class MockEntityService(EntityResolutionService):
        def resolve_entities_for_session(self, session_id, prospect, sources, search_results=None):
            return IdentityResolutionResult(
                target_prospect=prospect,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                rationale="Resolved",
                prospect_entity=EntityRecord(
                    entity_id="p-1",
                    entity_type=EntityType.PERSON,
                    raw_name=prospect.full_name,
                    canonical_name="Elena Rostova",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
                company_entity=EntityRecord(
                    entity_id="c-1",
                    entity_type=EntityType.ORGANIZATION,
                    raw_name=prospect.company_name,
                    canonical_name="Vanguard AI",
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=ConfidenceLevel.HIGH,
                ),
            )

    class MockFactService(FactExtractionService):
        def extract_and_persist_facts_for_session(self, session_id, snapshot_id, known_entities=None, accumulator=None):
            f = CanonicalFact(
                fact_id="f-imm",
                claim_text="Verified Elena role",
                evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Verified")],
            )
            if accumulator:
                accumulator.facts.append(f)
            return [f]

    researcher = Researcher(
        state_store=test_state_store,
        planner=ResearchPlanner(engine=MockLLMEngine([plan])),
        query_generator=QueryGenerator(engine=MockLLMEngine([QueryListOutput(queries=["elena"])])) ,
        search_client=MockSearchClient(),
        page_fetcher=MockPageFetcher(),
        entity_service=MockEntityService(state_store=test_state_store),
        fact_service=MockFactService(state_store=test_state_store),
        evaluator=ResearchSufficiencyEvaluator(engine=MockLLMEngine([eval_complete])),
    )

    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        session_id=session_id,
        max_iterations=1,
    )

    with pytest.raises((AttributeError, TypeError)):
        package.facts.append(CanonicalFact(fact_id="hack", claim_text="hacked"))  # type: ignore

    with pytest.raises((AttributeError, TypeError)):
        package.sources.append(SourceRecord(source_id="hack", url="http://hack.com", domain="hack.com", raw_text="", fetch_date=datetime.date(2026, 1, 1)))  # type: ignore

    with pytest.raises(Exception):
        package.is_sufficient = False  # frozen pydantic model
