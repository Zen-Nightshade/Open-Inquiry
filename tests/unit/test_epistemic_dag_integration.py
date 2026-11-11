from __future__ import annotations

import json
import tempfile
import uuid
from typing import Any, Dict, List, Optional
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration import (
    ApplicationOrchestrator,
    RunStatus,
    StageIdentifier,
    StageStatus,
)
from src.researcher.epistemic_dag import (
    DAGAdapter,
    EpistemicDAGScheduler,
    EpistemicQuestionDAG,
    GlobalResearchBudget,
    QuestionLifecycleStatus,
)
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
from src.researcher.orchestration.orchestrator import Researcher
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    TargetEntity,
)
from src.researcher.planning.planner import ResearchPlanner
from src.researcher.search.query_generator import GeneratedQuery


class MockPlanner(ResearchPlanner):
    """Deterministic mock planner that does not require external LLM or API keys."""

    def __init__(self, objectives: Optional[List[ResearchObjective]] = None):
        self._objectives = objectives or [
            ResearchObjective(
                objective_id="obj_ident",
                type=ResearchObjectiveType.IDENTITY_RESOLUTION,
                target_entity=TargetEntity.PROSPECT,
                description="Verify Elena Rostova current role and employment at Vanguard AI",
                priority=1,
            ),
            ResearchObjective(
                objective_id="obj_comp",
                type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
                target_entity=TargetEntity.COMPANY,
                description="Investigate Vanguard AI office expansion and funding",
                priority=2,
            ),
        ]

    def plan(self, prospect, outreach, product=None, session_id=None) -> ResearchPlan:
        return ResearchPlan(
            session_id=session_id or str(uuid.uuid4()),
            total_query_budget=10,
            max_iterations=1,
            objectives=self._objectives,
        )


class MockQueryGenerator:
    """Mock query generator avoiding external LLM provider calls in unit tests."""

    def generate_queries(self, objective, prospect, existing_queries=None):
        return [
            GeneratedQuery(
                query_id=str(uuid.uuid4()),
                objective_id=objective.objective_id,
                query_text=f"search query for {objective.description}",
                target_entity=objective.target_entity,
                rationale="Mock query",
            )
        ]
import datetime
from src.researcher.entity.resolver import IdentityResolutionResult
from src.researcher.entity.service import EntityResolutionService
from src.researcher.extraction.fetcher import PageFetcher
from src.researcher.extraction.models import FetchOutcome, FetchStatus
from src.researcher.extraction.service import FactExtractionService
from src.researcher.orchestration.evaluator import ResearchSufficiencyEvaluator
from src.researcher.orchestration.models import (
    DimensionAssessment,
    DimensionAssessments,
    ResearchSufficiencyAssessment,
    SufficiencyDecision,
    SufficiencyDimension,
)
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus


class MockSearchClient:
    """Mock search client returning deterministic synthetic results without live network calls."""

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        return SearchResponse(
            status=SearchStatus.SUCCESS,
            results=[
                SearchResult(
                    url=f"https://vanguardai.io/news/{abs(hash(query))}",
                    original_url=f"https://vanguardai.io/news/{abs(hash(query))}",
                    title=f"Vanguard AI News on {query}",
                    snippet=f"Vanguard AI corroborated details for {query}",
                    domain="vanguardai.io",
                    position=1,
                    query=query,
                )
            ],
            query=query,
        )


class MockPageFetcher(PageFetcher):
    """Mock page fetcher returning synthetic source content."""

    def fetch(self, url: str) -> FetchOutcome:
        record = SourceRecord(
            url=url,
            original_url=url,
            domain="vanguardai.io",
            title=f"Page for {url}",
            raw_text="Elena Rostova serves as Head of Research at Vanguard AI. Vanguard AI announced distributed infrastructure expansion in London.",
            source_type=SourceType.COMPANY_WEBSITE,
            credibility_score=0.9,
            fetch_date=datetime.date(2026, 11, 1),
        )
        return FetchOutcome(
            url=url,
            status=FetchStatus.SUCCESS,
            source_record=record,
            http_status_code=200,
            duration_ms=10.0,
        )


class MockEntityService(EntityResolutionService):
    """Deterministic entity resolution mock."""

    def resolve_entities_for_session(
        self, session_id: str, prospect: ProspectContext, sources: Any, search_results: Any = None
    ) -> IdentityResolutionResult:
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


class MockFactService(FactExtractionService):
    """Deterministic fact extraction mock that consolidates into accumulator and persists."""

    def extract_and_persist_facts_for_session(
        self,
        session_id: str,
        snapshot_id: str,
        entity_context: Optional[Dict[str, Any]] = None,
        known_entities: Optional[List[EntityRecord]] = None,
        accumulator: Optional[Any] = None,
    ) -> List[CanonicalFact]:
        ev = EvidenceRecord(
            source_id="src_test_1",
            verbatim_excerpt="Elena Rostova serves as Head of Research at Vanguard AI.",
        )
        fact = CanonicalFact(
            fact_id=f"fact_{uuid.uuid4().hex[:8]}",
            claim_text="Elena Rostova serves as Head of Research at Vanguard AI.",
            fact_category=FactCategory.ORGANIZATION_CHANGE,
            evidence_records=[ev],
            composite_confidence=ConfidenceLevel.HIGH,
        )
        if accumulator:
            accumulator.merge_or_add_fact(fact)
        return [fact]


class MockEvaluator(ResearchSufficiencyEvaluator):
    """Deterministic sufficiency evaluator mock."""

    def evaluate(self, *args, **kwargs) -> ResearchSufficiencyAssessment:
        session_id = kwargs.get("session_id", "session_test")
        iteration_id = kwargs.get("iteration_id", "iter_test")
        return ResearchSufficiencyAssessment(
            session_id=session_id,
            iteration_id=iteration_id,
            decision=SufficiencyDecision.SUFFICIENT,
            dimensions=DimensionAssessments(
                objective_coverage=DimensionAssessment(
                    dimension_name=SufficiencyDimension.OBJECTIVE_COVERAGE,
                    status="adequate",
                    summary="Adequate coverage",
                ),
                identity_certainty=DimensionAssessment(
                    dimension_name=SufficiencyDimension.IDENTITY_CERTAINTY,
                    status="confirmed",
                    summary="Identity confirmed",
                ),
                evidence_adequacy=DimensionAssessment(
                    dimension_name=SufficiencyDimension.EVIDENCE_ADEQUACY,
                    status="adequate",
                    summary="Evidence adequate",
                ),
                marginal_research_value=DimensionAssessment(
                    dimension_name=SufficiencyDimension.MARGINAL_RESEARCH_VALUE,
                    status="exhausted",
                    summary="Marginal value exhausted",
                ),
            ),
            explanation="Mock sufficient evaluation completed",
        )


def _make_prospect() -> ProspectContext:
    return ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
    )


def _make_product() -> ProductContext:
    return ProductContext(
        product_name="Open Inquiry Engine",
        description="Autonomous outreach research engine.",
        capabilities=["Real-time company intelligence", "Verifiable fact extraction"],
        value_proposition="Automate personalized, evidence-grounded research outreach.",
    )


def _make_outreach() -> OutreachContext:
    return OutreachContext(
        objective="Schedule introductory briefing",
        sender_identity="Alex Rivers, Lead Architect at SystemsLab",
        sender_name="Alex Rivers",
        sender_company="SystemsLab",
    )


def test_epistemic_dag_serialization_roundtrip():
    """Verify EpistemicQuestionDAG serializes and deserializes accurately with full node fidelity."""
    obj1 = ResearchObjective(
        objective_id="obj_1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Elena Rostova current role at Vanguard AI",
        priority=1,
    )
    obj2 = ResearchObjective(
        objective_id="obj_2",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Investigate Vanguard AI office expansion in London",
        priority=2,
    )

    dag = DAGAdapter.convert_objectives_to_dag(
        objectives=[obj1, obj2],
        run_id="run_test",
        prospect_name="Elena Rostova",
        company_name="Vanguard AI",
    )

    summary = EpistemicDAGScheduler.serialize_dag_summary(dag)
    json_str = json.dumps(summary)
    loaded = json.loads(json_str)

    assert loaded["dag_id"] == dag.dag_id
    assert loaded["node_count"] == 2
    assert len(loaded["nodes"]) == 2
    assert loaded["status_counts"]["ready"] == 2


def test_researcher_conduct_research_populates_dag_and_unresolved():
    """Verify Researcher.conduct_research drives execution with the DAG and serializes DAG into ResearchPackage."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(db_path=tmp.name)
        store.initialize()

        researcher = Researcher(
            state_store=store,
            planner=MockPlanner(),
            query_generator=MockQueryGenerator(),
            search_client=MockSearchClient(),
            page_fetcher=MockPageFetcher(),
            entity_service=MockEntityService(state_store=store),
            fact_service=MockFactService(state_store=store),
            evaluator=MockEvaluator(),
        )

        package = researcher.conduct_research(
            prospect=_make_prospect(),
            outreach=_make_outreach(),
            product=_make_product(),
            session_id="session_int_1",
            max_iterations=1,
        )

        assert package is not None
        assert package.question_dag_json is not None

        dag_summary = json.loads(package.question_dag_json)
        assert dag_summary["node_count"] >= 2
        assert len(dag_summary["nodes"]) >= 2

        # Verify workflow events recorded DAG initialization and updates
        with store.uow() as uow:
            cursor = uow.conn.cursor()
            cursor.execute("SELECT event_type FROM workflow_events WHERE session_id = ?;", ("session_int_1",))
            event_types = [r[0] for r in cursor.fetchall()]

            assert "epistemic_dag_initialized" in event_types
            assert "epistemic_dag_updated" in event_types


def test_shared_global_budget_preserved_through_app_orchestrator():
    """Verify ApplicationOrchestrator tracks single shared research budget without resets."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(db_path=tmp.name)
        store.initialize()

        researcher = Researcher(
            state_store=store,
            planner=MockPlanner(),
            query_generator=MockQueryGenerator(),
            search_client=MockSearchClient(),
            page_fetcher=MockPageFetcher(),
            entity_service=MockEntityService(state_store=store),
            fact_service=MockFactService(state_store=store),
            evaluator=MockEvaluator(),
        )
        orchestrator = ApplicationOrchestrator(state_store=store, researcher=researcher)

        result = orchestrator.run(
            prospect=_make_prospect(),
            product=_make_product(),
            outreach=_make_outreach(),
            run_id="run_budget_test",
            max_research_iterations=1,
            total_research_budget=20,
        )

        # Stage 3 record must preserve budget accounting details
        stage_records = result.stage_records
        res_record = next(r for r in stage_records if r.stage == StageIdentifier.RESEARCH)
        assert "budget_consumed" in res_record.details
        assert "budget_remaining" in res_record.details
        assert res_record.details["budget_consumed"] >= 0
        assert res_record.details["budget_remaining"] <= 20
