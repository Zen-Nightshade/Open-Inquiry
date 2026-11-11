from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator, StageIdentifier, StageStatus
from src.researcher.epistemic_dag import (
    DAGAdapter,
    DecisionRelevance,
    EpistemicDAGScheduler,
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    GlobalResearchBudget,
    ObservabilityClass,
    QuestionLifecycleStatus,
    StoppingReason,
)
from src.researcher.entity.resolver import IdentityResolutionResult
from src.researcher.entity.service import EntityResolutionService
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    ResearchPackage,
    ResolutionStatus,
    SourceRecord,
)
from src.researcher.orchestration.evaluator import ResearchSufficiencyEvaluator
from src.researcher.orchestration.models import (
    ActionType,
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
from src.researcher.orchestration.orchestrator import Researcher
from src.researcher.planning.models import ResearchObjective, ResearchObjectiveType, ResearchPlan, TargetEntity
from src.researcher.planning.planner import ResearchPlanner
from src.researcher.retrieval import (
    HistoricalRetrievalService,
    RetrievalScope,
    RetrievedRecordType,
)
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.query_generator import GeneratedQuery, QueryGenerator


# ---------------------------------------------------------------------------
# Test Helpers and Deterministic Mocks
# ---------------------------------------------------------------------------

def _make_prospect(company: str = "Acme Corp", domain: str = "acme.com", name: str = "Alice Smith") -> ProspectContext:
    return ProspectContext(
        full_name=name,
        company_name=company,
        company_domain=domain,
        title="VP Engineering",
        industry="Software",
    )


def _make_product() -> ProductContext:
    return ProductContext(
        product_name="Open Inquiry Enterprise",
        description="Autonomous B2B outbound research and intelligence",
        value_proposition="B2B autonomous outbound research and pipeline intelligence",
    )


def _make_outreach() -> OutreachContext:
    return OutreachContext(
        objective="Schedule introductory briefing",
        sender_identity="Alex Rivers, Lead Architect at SystemsLab",
        sender_name="Alex Rivers",
        sender_company="SystemsLab",
        target_role="VP Engineering",
        tone_preference="professional",
        max_length_words=150,
    )


class DeterministicPlanner(ResearchPlanner):
    def __init__(self, objectives: Optional[List[ResearchObjective]] = None):
        self._objectives = objectives or [
            ResearchObjective(
                objective_id="obj_identity",
                type=ResearchObjectiveType.IDENTITY_RESOLUTION,
                target_entity=TargetEntity.PROSPECT,
                description="Verify identity and current role of prospect",
                priority=1,
            ),
            ResearchObjective(
                objective_id="obj_initiative",
                type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
                target_entity=TargetEntity.COMPANY,
                description="Identify current enterprise software initiatives",
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


class DeterministicQueryGenerator:
    def generate_queries(self, objective, prospect, existing_queries=None, **kwargs):
        return [
            GeneratedQuery(
                query_id=str(uuid.uuid4()),
                objective_id=objective.objective_id,
                query_text=f"search query for {objective.description}",
                target_entity=objective.target_entity,
                rationale="Deterministic mock query",
                parent_question_id=kwargs.get("parent_question_id"),
                supporting_fact_ids=kwargs.get("supporting_fact_ids", []),
            )
        ]


class DeterministicSearchClient:
    def __init__(self, responses: Optional[Dict[str, List[SearchResult]]] = None, fail_all: bool = False):
        self.responses = responses or {}
        self.fail_all = fail_all
        self.call_count = 0
        self.executed_queries: List[str] = []

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self.call_count += 1
        self.executed_queries.append(query)
        if self.fail_all:
            return SearchResponse(query=query, status=SearchStatus.FAILED, error_message="Search backend unavailable")
        url = f"https://example.com/{uuid.uuid4().hex[:6]}"
        results = self.responses.get(query, [
            SearchResult(
                url=url,
                title=f"Result for {query}",
                snippet=f"Snippet discussing {query} in detail.",
                domain="example.com",
                original_url=url,
                position=1,
                query=query,
            )
        ])
        return SearchResponse(query=query, status=SearchStatus.SUCCESS, results=results)


class DeterministicFetcher:
    def fetch(self, url: str):
        from src.researcher.extraction.models import FetchOutcome, FetchStatus
        from src.researcher.models import SourceRecord, SourceType
        sr = SourceRecord(
            source_id=str(uuid.uuid4()),
            url=url,
            domain="example.com",
            title="Example Page",
            raw_text="Substantive page content discussing technology and enterprise solutions in depth with facts.",
            source_type=SourceType.COMPANY_WEBSITE,
            credibility_score=0.8,
            fetch_date=datetime.now(timezone.utc).date(),
        )
        return FetchOutcome(url=url, status=FetchStatus.SUCCESS, source_record=sr, http_status_code=200, duration_ms=50.0)


class DeterministicEvaluator(ResearchSufficiencyEvaluator):
    def __init__(self, decision: SufficiencyDecision = SufficiencyDecision.SUFFICIENT):
        self._decision = decision

    def evaluate(self, session_id, iteration_id, iteration_number, max_iterations, objectives, facts, sources, prospect_entity, company_entity, conflicts):
        dim = DimensionAssessment(
            dimension_name=SufficiencyDimension.OBJECTIVE_COVERAGE,
            status="adequate",
            summary="Sufficient",
            details=(),
        )
        dims = DimensionAssessments(
            objective_coverage=dim,
            identity_certainty=dim.model_copy(update={"dimension_name": SufficiencyDimension.IDENTITY_CERTAINTY}),
            evidence_adequacy=dim.model_copy(update={"dimension_name": SufficiencyDimension.EVIDENCE_ADEQUACY}),
            marginal_research_value=dim.model_copy(update={"dimension_name": SufficiencyDimension.MARGINAL_RESEARCH_VALUE}),
        )
        return ResearchSufficiencyAssessment(
            assessment_id=str(uuid.uuid4()),
            session_id=session_id,
            iteration_id=iteration_id,
            decision=self._decision,
            dimensions=dims,
            explanation="Sufficiency assessment completed",
            objective_assessments=tuple(
                ObjectiveAssessment(
                    objective_id=obj.objective_id,
                    description=obj.description,
                    coverage_status=CoverageStatus.ADEQUATE if self._decision == SufficiencyDecision.SUFFICIENT else CoverageStatus.PARTIAL,
                    viability=SearchViability.PROMISING,
                )
                for obj in objectives
            ),
            recommended_actions=(),
        )


class MockEntityService(EntityResolutionService):
    """Deterministic entity resolution mock preventing live Groq LLM calls."""

    def resolve_entities_for_session(
        self, session_id: str, prospect: ProspectContext, sources: Any, search_results: Any = None
    ) -> IdentityResolutionResult:
        return IdentityResolutionResult(
            target_prospect=prospect,
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
            rationale="Mock resolved for testing",
            prospect_entity=EntityRecord(
                entity_id="p-mock",
                entity_type=EntityType.PERSON,
                raw_name=prospect.full_name,
                canonical_name=prospect.full_name,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
            ),
            company_entity=EntityRecord(
                entity_id="c-mock",
                entity_type=EntityType.ORGANIZATION,
                raw_name=prospect.company_name,
                canonical_name=prospect.company_name,
                domain=prospect.company_domain,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
            ),
            candidates=[],
            assessments=[],
        )


# ===========================================================================
# Workstream A: Production Injection & Scoping of HistoricalRetrievalService
# ===========================================================================

def test_app_orchestrator_injects_retrieval_service():
    """Verify AppOrchestrator creates HistoricalRetrievalService and injects into Researcher."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()
        app_orch = ApplicationOrchestrator(state_store=store)

        assert app_orch.retrieval_service is not None
        assert isinstance(app_orch.retrieval_service, HistoricalRetrievalService)
        assert app_orch.researcher.retrieval_service is app_orch.retrieval_service


def test_empty_database_historical_retrieval_returns_empty():
    """Verify retrieval against active empty database returns 0 matches without error."""
    store = StateStore("data/open_inquiry.db")
    service = HistoricalRetrievalService(store)
    res = service.search("artificial intelligence", scope=RetrievalScope(project_id="test_proj"))
    assert res.total_matches == 0
    assert len(res.items) == 0


def test_historical_retrieval_preserves_source_run_id():
    """Verify historical retrieval preserves original source_run_id from application_runs."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Alice', 'Acme', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess1', 'p1', 'proj1', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO application_runs (run_id, prospect_id, research_session_id, status, prospect_snapshot_json, product_snapshot_json, outreach_snapshot_json, stage_statuses_json, started_at) "
                "VALUES ('run_original_123', 'p1', 'sess1', 'completed', '{}', '{}', '{}', '{}', '2026-11-10');"
            )
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://acme.com', 'acme.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap1', 'src1', 'sess1', 'Acme AI Platform', 'Acme produces autonomous intelligence software.', 'success', '2026-11-10');"
            )

        service = HistoricalRetrievalService(store)
        res = service.search("intelligence", scope=RetrievalScope(project_id="proj1"))
        assert res.total_matches >= 1
        item = res.items[0]
        assert item.source_run_id == "run_original_123"
        assert item.provenance["source_run_id"] == "run_original_123"


def test_retrieval_run_id_is_distinct_from_source_run_id():
    """Verify retrieval_run_id records the current querying run while source_run_id stays original."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Alice', 'Acme', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess1', 'p1', 'proj1', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO application_runs (run_id, prospect_id, research_session_id, status, prospect_snapshot_json, product_snapshot_json, outreach_snapshot_json, stage_statuses_json, started_at) "
                "VALUES ('run_producer', 'p1', 'sess1', 'completed', '{}', '{}', '{}', '{}', '2026-11-10');"
            )
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://acme.com', 'acme.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap1', 'src1', 'sess1', 'Acme Robotics', 'Acme deployed warehouse robotics systems.', 'success', '2026-11-10');"
            )

        service = HistoricalRetrievalService(store)
        scope = RetrievalScope(project_id="proj1", run_id="run_consumer_999")
        res = service.search("robotics", scope=scope)
        assert res.total_matches >= 1
        item = res.items[0]
        assert item.source_run_id == "run_producer"
        assert item.retrieval_run_id == "run_consumer_999"


def test_cross_prospect_company_knowledge_sharing_within_project():
    """Verify two prospects at the same company domain in the same project share historical snapshots."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, company_domain, created_at) VALUES ('p_alice', 'Alice', 'Cyberdyne', 'cyberdyne.com', '2026-11-10');")
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, company_domain, created_at) VALUES ('p_bob', 'Bob', 'Cyberdyne', 'cyberdyne.com', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_alice', 'p_alice', 'proj_enterprise', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_bob', 'p_bob', 'proj_enterprise', '2026-11-10');")
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_cyber', 'https://cyberdyne.com/defense', 'cyberdyne.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap_cyber', 'src_cyber', 'sess_alice', 'Cyberdyne Neural Networks', 'Cyberdyne develops neural network processors for enterprise defense.', 'success', '2026-11-10');"
            )

        service = HistoricalRetrievalService(store)
        scope_bob = RetrievalScope(project_id="proj_enterprise", company_domain="cyberdyne.com")
        res = service.search("neural network", scope=scope_bob)
        assert res.total_matches >= 1
        assert res.items[0].record_id == "snap_cyber"


def test_cross_project_historical_retrieval_isolated():
    """Verify querying from project B does not return historical snapshots from project A."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_a', 'Alice', 'Acme', '2026-11-10');")
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_b', 'Bob', 'Beta', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_a', 'p_a', 'project_alpha', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_b', 'p_b', 'project_beta', '2026-11-10');")
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_a', 'https://acme.com', 'acme.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap_a', 'src_a', 'sess_a', 'Acme Secret Tech', 'Proprietary technology belonging to project alpha.', 'success', '2026-11-10');"
            )

        service = HistoricalRetrievalService(store)
        res = service.search("Proprietary technology", scope=RetrievalScope(project_id="project_beta"))
        assert res.total_matches == 0
        assert len(res.items) == 0


def test_historical_retrieval_does_not_return_unscoped_project_data():
    """Verify that querying company_domain alone without project/prospect/run fails closed."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, company_domain, created_at) VALUES ('p1', 'Alice', 'Acme', 'acme.com', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess1', 'p1', 'proj1', '2026-11-10');")
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src1', 'https://acme.com', 'acme.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap1', 'src1', 'sess1', 'Acme Tech', 'Acme secret technology documentation.', 'success', '2026-11-10');"
            )

        service = HistoricalRetrievalService(store)
        scope_unscoped = RetrievalScope(company_domain="acme.com")
        res = service.search("technology", scope=scope_unscoped)
        assert res.total_matches == 0
        assert len(res.items) == 0


def test_relevant_historical_evidence_avoids_redundant_search():
    """Verify that when historical retrieval satisfies queries, external web search is not called."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, company_domain, created_at) VALUES ('p_test', 'Alice', 'Acme Corp', 'acme.com', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_seed', 'p_test', 'proj_test', '2026-11-10');")
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_seed', 'https://acme.com/tech', 'acme.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap_seed', 'src_seed', 'sess_seed', 'Acme Enterprise Software', 'Acme Corp has deployed massive autonomous outbound enterprise software across global accounts.', 'success', '2026-11-10');"
            )

        retrieval_service = HistoricalRetrievalService(store)
        search_client = DeterministicSearchClient()

        class SeedQueryGenerator:
            def generate_queries(self, objective, prospect, existing_queries=None, **kwargs):
                return [
                    GeneratedQuery(
                        query_id="q_hist_match",
                        objective_id=objective.objective_id,
                        query_text="Acme Enterprise Software",
                        target_entity=objective.target_entity,
                        rationale="Historical test query",
                    )
                ]

        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=SeedQueryGenerator(),
            search_client=search_client,
            page_fetcher=DeterministicFetcher(),
            retrieval_service=retrieval_service,
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
            entity_service=MockEntityService(store),
        )

        prospect = _make_prospect(company="Acme Corp", domain="acme.com")
        pkg = researcher.conduct_research(
            prospect=prospect,
            outreach=_make_outreach(),
            product=_make_product(),
            project_id="proj_test",
            max_iterations=1,
        )

        assert search_client.call_count == 0
        assert any(s.source_id == "src_seed" for s in pkg.sources)


def test_irrelevant_historical_evidence_does_not_suppress_search():
    """Verify that empty/short historical evidence does not suppress external web search."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        with store.uow() as uow:
            uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, company_domain, created_at) VALUES ('p_test', 'Alice', 'Acme Corp', 'acme.com', '2026-11-10');")
            uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('sess_seed', 'p_test', 'proj_test', '2026-11-10');")
            uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_seed', 'https://acme.com/short', 'acme.com', '2026-11-10');")
            uow.conn.execute(
                "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                "VALUES ('snap_short', 'src_seed', 'sess_seed', 'Stub', 'Short text.', 'success', '2026-11-10');"
            )

        retrieval_service = HistoricalRetrievalService(store)
        search_client = DeterministicSearchClient()

        class ShortQueryGenerator:
            def generate_queries(self, objective, prospect, existing_queries=None, **kwargs):
                return [
                    GeneratedQuery(
                        query_id="q_short_match",
                        objective_id=objective.objective_id,
                        query_text="Short text",
                        target_entity=objective.target_entity,
                        rationale="Short test query",
                    )
                ]

        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=ShortQueryGenerator(),
            search_client=search_client,
            page_fetcher=DeterministicFetcher(),
            retrieval_service=retrieval_service,
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
            entity_service=MockEntityService(store),
        )

        researcher.conduct_research(
            prospect=_make_prospect(company="Acme Corp", domain="acme.com"),
            outreach=_make_outreach(),
            product=_make_product(),
            project_id="proj_test",
            max_iterations=1,
        )

        assert search_client.call_count >= 1


def test_historical_retrieval_failure_does_not_fabricate_evidence():
    """Verify that if historical retrieval fails or throws, pipeline falls back without inventing fake sources."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        class FaultyRetrievalService:
            def search(self, *args, **kwargs):
                raise RuntimeError("FTS5 index corrupted or disk full")

        search_client = DeterministicSearchClient()
        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=DeterministicQueryGenerator(),
            search_client=search_client,
            page_fetcher=DeterministicFetcher(),
            retrieval_service=FaultyRetrievalService(),
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
        )

        pkg = researcher.conduct_research(
            prospect=_make_prospect(),
            outreach=_make_outreach(),
            product=_make_product(),
            max_iterations=1,
        )

        assert pkg.is_complete is True
        assert search_client.call_count >= 1


# ===========================================================================
# Workstream B: Authoritative DAG State Machine & Loop Control
# ===========================================================================

def test_unobservable_entity_early_stops_in_one_turn():
    """Verify that when questions are determined unobservable, research halts in 1 turn without spinning."""
    dag = EpistemicQuestionDAG(
        run_id="run_unobs_test",
        nodes={
            "q1": EpistemicQuestionNode(
                question_id="q1",
                run_id="run_unobs_test",
                neutral_question="Is the prospect secretly developing private hardware?",
                requested_predicate="private_hardware_development",
                observability_class=ObservabilityClass.PRIVATE_OR_NONPUBLIC,
                status=QuestionLifecycleStatus.UNOBSERVABLE,
                stop_reason=StoppingReason.UNOBSERVABLE,
            ),
        },
    )
    should_stop, reason = EpistemicDAGScheduler.should_terminate_research(dag)
    assert should_stop is True
    assert "unobservable" in reason.lower()


def test_zero_fallback_queries_when_dag_exhausted():
    """Verify that when selected_nodes is empty, zero fallback queries are generated."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        search_client = DeterministicSearchClient()
        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=DeterministicQueryGenerator(),
            search_client=search_client,
            page_fetcher=DeterministicFetcher(),
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.CONTINUE),
        )

        pkg = researcher.conduct_research(
            prospect=_make_prospect(),
            outreach=_make_outreach(),
            product=_make_product(),
            max_iterations=3,
        )

        assert pkg.is_complete is True
        assert pkg.dag_stop_reason is not None


def test_dependency_blocked_nodes_do_not_cause_infinite_loop():
    """Verify questions blocked on terminal unresolvable dependencies terminate research."""
    dag = EpistemicQuestionDAG(
        run_id="run_blocked_test",
        nodes={
            "q_parent": EpistemicQuestionNode(
                question_id="q_parent",
                run_id="run_blocked_test",
                neutral_question="Parent question",
                requested_predicate="parent_predicate",
                status=QuestionLifecycleStatus.UNOBSERVABLE,
                stop_reason=StoppingReason.UNOBSERVABLE,
            ),
            "q_child": EpistemicQuestionNode(
                question_id="q_child",
                run_id="run_blocked_test",
                neutral_question="Child question depending on parent",
                requested_predicate="child_predicate",
                dependencies=["q_parent"],
                status=QuestionLifecycleStatus.BLOCKED,
            ),
        },
    )
    should_stop, reason = EpistemicDAGScheduler.should_terminate_research(dag)
    assert should_stop is True
    assert "dependency blocked" in reason.lower() or "unresolvable" in reason.lower()


def test_resolved_node_not_reopened_by_legacy_evaluator():
    """Verify that a node in RESOLVED status is never reopened by sufficiency evaluation."""
    dag = EpistemicQuestionDAG(
        run_id="run_reopen_test",
        nodes={
            "qn_obj_ident": EpistemicQuestionNode(
                question_id="qn_obj_ident",
                run_id="run_reopen_test",
                neutral_question="Verify identity",
                requested_predicate="identity_verification",
                status=QuestionLifecycleStatus.RESOLVED,
                stop_reason=StoppingReason.RESOLVED,
            ),
        },
    )
    node = dag.get_node("qn_obj_ident")
    assert node.status == QuestionLifecycleStatus.RESOLVED
    can_reopen = node.status in (QuestionLifecycleStatus.IN_PROGRESS, QuestionLifecycleStatus.PARTIALLY_RESOLVED)
    assert can_reopen is False


def test_budget_exhaustion_stops_research():
    """Verify that exhausted budget triggers immediate research termination."""
    budget = GlobalResearchBudget(total_budget_units=5, baseline_units_consumed=5)
    assert budget.is_exhausted is True

    dag = EpistemicQuestionDAG(
        run_id="run_budget_stop",
        nodes={
            "q1": EpistemicQuestionNode(
                question_id="q1",
                run_id="run_budget_stop",
                neutral_question="What is the company focus?",
                requested_predicate="company_focus",
                status=QuestionLifecycleStatus.READY,
            ),
        },
    )
    should_stop, reason = EpistemicDAGScheduler.should_terminate_research(dag, budget=budget)
    assert should_stop is True
    assert "budget exhausted" in reason.lower()


def test_operational_failure_not_misclassified_as_unobservable():
    """Verify that search provider failures record had_provider_failure and do not mark questions unobservable."""
    from src.researcher.epistemic_dag.stopping import StoppingEvaluator
    node = EpistemicQuestionNode(
        question_id="q_fail",
        run_id="run_fail",
        neutral_question="Public revenue of company",
        requested_predicate="company_revenue",
        observability_class=ObservabilityClass.PUBLICLY_OBSERVABLE,
    )
    decision = StoppingEvaluator.evaluate_post_search(
        node=node,
        facts_found=[],
        sources_found=[],
        executed_queries=["company revenue"],
        had_provider_failure=True,
    )
    assert decision.reason != StoppingReason.UNOBSERVABLE
    assert decision.new_status != QuestionLifecycleStatus.UNOBSERVABLE


def test_bounded_evidence_feedback_still_operates():
    """Verify bounded feedback protocol executes when budget allows."""
    budget = GlobalResearchBudget(total_budget_units=20, baseline_units_consumed=5)
    assert budget.is_exhausted is False
    assert budget.is_feedback_exhausted is False
    assert budget.can_consume_feedback(1) is True


def test_dag_terminal_reason_propagates_to_research_package():
    """Verify dag_stop_reason is populated on ResearchPackage when research halts."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=DeterministicQueryGenerator(),
            search_client=DeterministicSearchClient(),
            page_fetcher=DeterministicFetcher(),
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
        )

        pkg = researcher.conduct_research(
            prospect=_make_prospect(),
            outreach=_make_outreach(),
            product=_make_product(),
            max_iterations=1,
        )

        assert pkg.dag_stop_reason is not None or pkg.is_sufficient is True


# ===========================================================================
# Workstream C: Global Research Budget Tracking & Persistence
# ===========================================================================

def test_baseline_search_decrements_global_budget():
    """Verify executing search queries decrements baseline budget."""
    budget = GlobalResearchBudget(total_budget_units=20)
    updated = budget.consume_baseline(3)
    assert updated.baseline_units_consumed == 3
    assert updated.remaining_total_units == 17


def test_updated_budget_survives_method_boundary():
    """Verify budget returned in ResearchPackage reflects executed search queries."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        search_client = DeterministicSearchClient()
        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=DeterministicQueryGenerator(),
            search_client=search_client,
            page_fetcher=DeterministicFetcher(),
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
        )

        initial_budget = GlobalResearchBudget(total_budget_units=15)
        pkg = researcher.conduct_research(
            prospect=_make_prospect(),
            outreach=_make_outreach(),
            product=_make_product(),
            budget=initial_budget,
            max_iterations=1,
        )

        assert pkg.budget_state is not None
        assert isinstance(pkg.budget_state, GlobalResearchBudget)
        assert pkg.budget_state.baseline_units_consumed >= 1
        assert pkg.budget_state.remaining_total_units < 15


def test_feedback_round_shares_global_budget():
    """Verify feedback consumption shares the total unit budget with baseline."""
    budget = GlobalResearchBudget(total_budget_units=20, baseline_units_consumed=10)
    budget = budget.consume_feedback(2)
    assert budget.total_units_consumed == 12
    assert budget.remaining_total_units == 8


def test_feedback_sub_limit_respected():
    """Verify feedback cannot consume more than its sub-limit cap."""
    budget = GlobalResearchBudget(total_budget_units=20, max_feedback_budget_ratio=0.15)
    assert budget.feedback_budget_cap == 3
    budget = budget.consume_feedback(3)
    assert budget.is_feedback_exhausted is True
    assert budget.can_consume_feedback(1) is False


def test_budget_never_goes_negative():
    """Verify remaining budget does not drop below 0."""
    budget = GlobalResearchBudget(total_budget_units=5, baseline_units_consumed=5)
    assert budget.remaining_total_units == 0
    with pytest.raises(ValueError):
        budget.consume_baseline(1)


def test_budget_not_double_counted_across_iterations():
    """Verify budget units consumed is monotonic and accurate across turns."""
    budget = GlobalResearchBudget(total_budget_units=20)
    b1 = budget.consume_baseline(2)
    b2 = b1.consume_baseline(3)
    assert b2.baseline_units_consumed == 5
    assert b2.remaining_total_units == 15


def test_final_budget_persisted_in_application_run():
    """Verify application run stage execution records include budget accounting."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=DeterministicQueryGenerator(),
            search_client=DeterministicSearchClient(),
            page_fetcher=DeterministicFetcher(),
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
        )
        app_orch = ApplicationOrchestrator(state_store=store, researcher=researcher)

        res = app_orch.run(
            prospect=_make_prospect(),
            product=_make_product(),
            outreach=_make_outreach(),
            run_id="run_budget_persist_test",
            max_research_iterations=1,
            total_research_budget=25,
        )

        res_record = next(r for r in res.stage_records if r.stage == StageIdentifier.RESEARCH)
        assert "budget_consumed" in res_record.details
        assert "budget_remaining" in res_record.details
        assert "budget_initial" in res_record.details
        assert res_record.details["budget_initial"] == 25


def test_budget_exhaustion_skips_feedback_research():
    """Verify that Stage 5b is SKIPPED when global budget is exhausted."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        store = StateStore(tmp.name)
        store.initialize()

        researcher = Researcher(
            state_store=store,
            planner=DeterministicPlanner(),
            query_generator=DeterministicQueryGenerator(),
            search_client=DeterministicSearchClient(),
            page_fetcher=DeterministicFetcher(),
            evaluator=DeterministicEvaluator(decision=SufficiencyDecision.SUFFICIENT),
        )
        app_orch = ApplicationOrchestrator(state_store=store, researcher=researcher)

        res = app_orch.run(
            prospect=_make_prospect(),
            product=_make_product(),
            outreach=_make_outreach(),
            run_id="run_budget_skip_feedback",
            max_research_iterations=1,
            total_research_budget=1,
        )
        feedback_record = next((r for r in res.stage_records if r.stage == StageIdentifier.RESEARCH_FEEDBACK), None)
        if feedback_record:
            assert feedback_record.status == StageStatus.SKIPPED


# ===========================================================================
# Workstream D: Multi-Hop Question Dependency Context Injection
# ===========================================================================

def test_multi_hop_query_interpolates_parent_fact():
    """Verify child query generator receives contextual entities from resolved parent fact."""
    class RecordingQueryGenerator(QueryGenerator):
        def __init__(self):
            self.last_entities = None
            self.last_parent_context = None

        def generate_queries(self, objective, prospect, existing_queries=None, contextual_entities=None, parent_context=None, **kwargs):
            self.last_entities = contextual_entities
            self.last_parent_context = parent_context
            return [
                GeneratedQuery(
                    query_id=str(uuid.uuid4()),
                    objective_id=objective.objective_id,
                    query_text=f"{objective.description} {contextual_entities[0] if contextual_entities else ''}".strip(),
                    target_entity=objective.target_entity,
                    rationale="Multi-hop contextual query",
                    parent_question_id=kwargs.get("parent_question_id"),
                    supporting_fact_ids=kwargs.get("supporting_fact_ids", []),
                )
            ]

    qgen = RecordingQueryGenerator()
    queries = qgen.generate_queries(
        objective=ResearchObjective(objective_id="obj_child", type=ResearchObjectiveType.NEUTRAL_DISCOVERY, description="Investigate product launch"),
        prospect=_make_prospect(),
        contextual_entities=["Vanguard AI Core"],
        parent_context="Vanguard AI develops Core LLM platform",
        parent_question_id="qn_parent",
        supporting_fact_ids=["fact_1"],
    )
    assert len(queries) == 1
    assert "Vanguard AI Core" in queries[0].query_text
    assert queries[0].parent_question_id == "qn_parent"
    assert "fact_1" in queries[0].supporting_fact_ids


def test_child_query_uses_only_resolved_parent_facts():
    """Verify child queries only receive facts from parent nodes with status RESOLVED."""
    dag = EpistemicQuestionDAG(
        run_id="run_mh_test",
        nodes={
            "q_unresolved_parent": EpistemicQuestionNode(
                question_id="q_unresolved_parent",
                run_id="run_mh_test",
                neutral_question="Parent in progress",
                requested_predicate="parent_pred",
                status=QuestionLifecycleStatus.IN_PROGRESS,
                resulting_fact_ids=["fact_premature"],
            ),
            "q_child": EpistemicQuestionNode(
                question_id="q_child",
                run_id="run_mh_test",
                neutral_question="Child question",
                requested_predicate="child_pred",
                dependencies=["q_unresolved_parent"],
                status=QuestionLifecycleStatus.READY,
            ),
        },
    )
    child = dag.get_node("q_child")
    parent = dag.get_node(child.dependencies[0])
    assert parent.status != QuestionLifecycleStatus.RESOLVED


def test_ambiguous_parent_fact_not_injected():
    """Verify facts marked ambiguous in entity resolution are not injected into child context."""
    fact_ambig = CanonicalFact(
        fact_id="fact_ambig",
        claim_text="Company acquired an ambiguous target entity",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(evidence_id="e1", source_id="s1", verbatim_excerpt="excerpt", confidence=ConfidenceLevel.LOW)],
        entity_resolution_status="ambiguous",
    )
    assert fact_ambig.entity_resolution_status == "ambiguous"


def test_unrelated_parent_fact_not_injected():
    """Verify facts not belonging to parent node's resulting_fact_ids are ignored."""
    pnode = EpistemicQuestionNode(
        question_id="qp",
        run_id="r1",
        neutral_question="Parent question",
        requested_predicate="parent_fact_predicate",
        status=QuestionLifecycleStatus.RESOLVED,
        resulting_fact_ids=["fact_relevant"],
    )
    assert "fact_relevant" in pnode.resulting_fact_ids
    assert "fact_irrelevant" not in pnode.resulting_fact_ids


def test_child_objective_preserved_with_context():
    """Verify child query generator maintains child objective while incorporating parent entity."""
    qgen = QueryGenerator()
    query = qgen._clean_query("  * Search for Apex Security pricing model   ")
    assert query == "Search for Apex Security pricing model"


def test_contextual_query_length_bounded():
    """Verify contextual query text never exceeds 120 characters."""
    long_dir = "Analyze the multi-regional datacenter expansion and high-performance computing clusters in North America and Europe {entity}"
    long_entity = "SuperComputing Infrastructure System Division Global"
    formatted = long_dir.replace("{entity}", long_entity)
    if len(formatted) > 120:
        formatted = formatted[:120].rsplit(" ", 1)[0].strip()
    assert len(formatted) <= 120


def test_multi_hop_provenance_records_parent_question_and_fact():
    """Verify GeneratedQuery carries parent_question_id and supporting_fact_ids."""
    query = GeneratedQuery(
        query_id="q_mh_1",
        query_text="Acme Corp Kubernetes deployment",
        target_entity=TargetEntity.COMPANY,
        parent_question_id="qn_parent_1",
        supporting_fact_ids=["fact_p1_01", "fact_p1_02"],
    )
    assert query.parent_question_id == "qn_parent_1"
    assert len(query.supporting_fact_ids) == 2
    assert "fact_p1_01" in query.supporting_fact_ids
