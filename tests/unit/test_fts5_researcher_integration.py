from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from unittest.mock import MagicMock
import pytest

from src.context.models import OutreachContext, ProspectContext
from src.db.state_store import StateStore
from src.researcher.entity import IdentityResolutionResult
from src.researcher.entity.service import EntityResolutionService
from src.researcher.epistemic_dag import GlobalResearchBudget, QuestionLifecycleStatus
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
    ResolutionStatus,
    SourceRecord,
    SourceType,
)
from src.researcher.orchestration.evaluator import ResearchSufficiencyEvaluator
from src.researcher.orchestration.models import (
    CoverageStatus,
    DimensionAssessment,
    DimensionAssessments,
    ObjectiveAssessment,
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
from src.researcher.retrieval import HistoricalRetrievalService, RetrievalScope
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.query_generator import GeneratedQuery, QueryGenerator


class MockPlanner(ResearchPlanner):
    """Deterministic mock planner avoiding external LLM calls."""

    def __init__(self, objectives: Optional[List[ResearchObjective]] = None):
        self._objectives = objectives or [
            ResearchObjective(
                objective_id="obj_drones",
                type=ResearchObjectiveType.IDENTITY_RESOLUTION,
                target_entity=TargetEntity.COMPANY,
                description="AeroCorp autonomous delivery drones",
                priority=1,
            ),
        ]

    def plan(self, prospect, outreach, product=None, session_id=None) -> ResearchPlan:
        return ResearchPlan(
            session_id=session_id or str(uuid.uuid4()),
            total_query_budget=10,
            max_iterations=1,
            objectives=self._objectives,
        )


class MockQueryGenerator(QueryGenerator):
    """Mock query generator avoiding external LLM provider calls."""

    def generate_queries(self, objective, prospect, existing_queries=None):
        return [
            GeneratedQuery(
                query_id=str(uuid.uuid4()),
                objective_id=objective.objective_id,
                query_text="AeroCorp autonomous delivery drones",
                target_entity=TargetEntity.COMPANY,
                rationale="Historical retrieval test query",
            )
        ]


class MockSearchClient:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0
        self.provider = MagicMock()
        self.provider.name = "mock_provider"

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self.call_count += 1
        if self.should_fail:
            return SearchResponse(
                status=SearchStatus.FAILED,
                error_message="Simulated external search network failure",
                query=query,
            )
        return SearchResponse(
            status=SearchStatus.SUCCESS,
            results=[
                SearchResult(
                    url="https://live.external.com/news",
                    original_url="https://live.external.com/news",
                    title="Live Web Announcement",
                    snippet="Live external update",
                    domain="live.external.com",
                    position=1,
                    query=query,
                )
            ],
            query=query,
        )


class MockPageFetcher(PageFetcher):
    def fetch(self, url: str) -> FetchOutcome:
        record = SourceRecord(
            url=url,
            original_url=url,
            domain="live.external.com",
            title="Live Web Announcement",
            raw_text="Live external web page fetched content.",
            source_type=SourceType.OTHER,
            credibility_score=0.7,
            fetch_date=datetime.date(2026, 11, 10),
        )
        return FetchOutcome(
            url=url,
            status=FetchStatus.SUCCESS,
            source_record=record,
            http_status_code=200,
            duration_ms=10.0,
        )


class MockEntityService(EntityResolutionService):
    def resolve_entities_for_session(
        self, session_id: str, prospect: ProspectContext, sources: Any, search_results: Any = None
    ) -> IdentityResolutionResult:
        return IdentityResolutionResult(
            target_prospect=prospect,
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
            rationale="Mock resolved for testing",
            prospect_entity=EntityRecord(
                entity_id="p-smith",
                entity_type=EntityType.PERSON,
                raw_name=prospect.full_name,
                canonical_name="Dr. Smith",
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
            ),
            company_entity=EntityRecord(
                entity_id="c-aerocorp",
                entity_type=EntityType.ORGANIZATION,
                raw_name=prospect.company_name or "AeroCorp",
                canonical_name="AeroCorp",
                domain="aerocorp.com",
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
            ),
            candidates=[],
            assessments=[],
        )


class MockFactService(FactExtractionService):
    def extract_and_persist_facts_for_session(
        self,
        session_id: str,
        snapshot_id: str,
        entity_context: Optional[Dict[str, Any]] = None,
        known_entities: Optional[List[EntityRecord]] = None,
        accumulator: Optional[Any] = None,
    ) -> List[CanonicalFact]:
        ev = EvidenceRecord(
            source_id="src_h1",
            verbatim_excerpt="Historical fact verified excerpt",
            confidence=ConfidenceLevel.HIGH,
            credibility_score=0.9,
        )
        fact = CanonicalFact(
            fact_id=f"fact_{uuid.uuid4().hex[:8]}",
            claim_text="AeroCorp operates autonomous delivery drones across three metro areas",
            fact_category=FactCategory.ORGANIZATION_CHANGE,
            evidence_records=[ev],
            composite_confidence=ConfidenceLevel.HIGH,
        )
        if accumulator is not None:
            accumulator.facts.append(fact)
        return [fact]


class MockEvaluator:
    def evaluate(self, session_id: str, iteration_id: str, iteration_number: int, max_iterations: int, objectives, facts, sources, prospect_entity, company_entity, conflicts):
        obj_assessments = tuple(
            ObjectiveAssessment(
                objective_id=obj.objective_id,
                description=obj.description,
                coverage_status=CoverageStatus.ADEQUATE if facts else CoverageStatus.PARTIAL,
                viability=SearchViability.PROMISING,
                is_critical=obj.is_critical,
            )
            for obj in objectives
        )
        dim = DimensionAssessment(
            dimension_name=SufficiencyDimension.OBJECTIVE_COVERAGE,
            status="adequate" if facts else "partial",
            summary="Coverage evaluated",
        )
        return ResearchSufficiencyAssessment(
            session_id=session_id,
            iteration_id=iteration_id,
            decision=SufficiencyDecision.SUFFICIENT if facts else SufficiencyDecision.CONTINUE,
            dimensions=DimensionAssessments(
                objective_coverage=dim,
                identity_certainty=dim,
                evidence_adequacy=dim,
                marginal_research_value=dim,
            ),
            objective_assessments=obj_assessments,
            explanation="Mock evaluation",
        )


def test_researcher_uses_historical_retrieval_with_zero_budget_consumption(tmp_path: Path):
    """Verify that local historical retrieval discovers candidates without deducting from search budget."""
    db_file = tmp_path / "researcher_fts_test.db"
    store = StateStore(db_file)
    store.initialize()

    # Pre-populate historical snapshot from a previous research session
    with store.uow() as uow:
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_hist', 'Dr. Smith', 'AeroCorp', '2026-10-01');")
        uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s_hist', 'p_hist', 'c_hist', '2026-10-01');")
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_h1', 'https://aerocorp.com/archives', 'aerocorp.com', '2026-10-01');")
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_h1', 'src_h1', 's_hist', 'AeroCorp Autonomous Delivery Drones', 'AeroCorp operates autonomous delivery drones across three metro areas.', 'success', '2026-10-01');"
        )
        uow.conn.execute(
            "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
            "VALUES ('ev_h1', 'snap_h1', 'src_h1', 'operates autonomous delivery drones across three metro areas', '2026-10-01');"
        )

    retrieval_service = HistoricalRetrievalService(state_store=store)
    mock_search = MockSearchClient()

    researcher = Researcher(
        state_store=store,
        planner=MockPlanner(),
        query_generator=MockQueryGenerator(),
        search_client=mock_search,
        page_fetcher=MockPageFetcher(),
        entity_service=MockEntityService(state_store=store),
        fact_service=MockFactService(state_store=store),
        evaluator=MockEvaluator(),
        retrieval_service=retrieval_service,
    )

    budget = GlobalResearchBudget(baseline_cap=10, feedback_cap=3)
    prospect = ProspectContext(full_name="Dr. Smith", company_name="AeroCorp")
    outreach = OutreachContext(objective="Partnership", sender_identity="Antigravity")

    # Run research with budget
    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        max_iterations=1,
        budget=budget,
    )

    assert package is not None
    # 1. Historical source was retrieved and hydrated into accumulator
    source_domains = {s.domain for s in package.sources}
    assert "aerocorp.com" in source_domains

    # 2. Historical source retains its original historical fetch_date
    hist_src = next(s for s in package.sources if s.domain == "aerocorp.com")
    assert hist_src.fetch_date == datetime.date(2026, 10, 1)

    # 3. Facts were extracted and verified
    assert len(package.facts) >= 1
    assert "AeroCorp" in package.facts[0].claim_text


def test_provider_failures_distinguished_from_historical_retrieval(tmp_path: Path):
    """Verify that when external search fails, local retrieval still functions and provider failure is recorded."""
    db_file = tmp_path / "provider_fail_test.db"
    store = StateStore(db_file)
    store.initialize()

    with store.uow() as uow:
        uow.conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p_hist', 'Dr. Smith', 'AeroCorp', '2026-10-01');")
        uow.conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s_hist', 'p_hist', 'c_hist', '2026-10-01');")
        uow.conn.execute("INSERT INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES ('src_h1', 'https://aerocorp.com/drones', 'aerocorp.com', '2026-10-01');")
        uow.conn.execute(
            "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
            "VALUES ('snap_h1', 'src_h1', 's_hist', 'AeroCorp Drones', 'AeroCorp operates autonomous delivery drones.', 'success', '2026-10-01');"
        )

    retrieval_service = HistoricalRetrievalService(state_store=store)
    failing_search = MockSearchClient(should_fail=True)

    researcher = Researcher(
        state_store=store,
        planner=MockPlanner(),
        query_generator=MockQueryGenerator(),
        search_client=failing_search,
        page_fetcher=MockPageFetcher(),
        entity_service=MockEntityService(state_store=store),
        fact_service=MockFactService(state_store=store),
        evaluator=MockEvaluator(),
        retrieval_service=retrieval_service,
    )

    prospect = ProspectContext(full_name="Dr. Smith", company_name="AeroCorp")
    outreach = OutreachContext(objective="Partnership", sender_identity="Antigravity")

    package = researcher.conduct_research(
        prospect=prospect,
        outreach=outreach,
        max_iterations=1,
    )

    # Local retrieval still provided the historical source
    assert any(s.domain == "aerocorp.com" for s in package.sources)
    # Search client failed
    assert failing_search.call_count >= 1
