from __future__ import annotations

import datetime
import tempfile
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.researcher.execution import ResearchExecutionCoordinator
from src.researcher.extraction.fetcher import PageFetcher
from src.researcher.extraction.models import FetchOutcome, FetchStatus
from src.researcher.models import SourceRecord, SourceType
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    TargetEntity,
)
from src.researcher.planning.planner import ResearchPlanner
from src.researcher.search.base import BaseSearchProvider
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.query_generator import GeneratedQuery, QueryGenerator
from src.researcher.search.search_client import SearchClient


class MockSearchProvider(BaseSearchProvider):
    """Mock search provider returning controllable responses for tests."""

    def __init__(self):
        super().__init__()
        self.name = "mock_search"
        self.query_log = []

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        self.query_log.append(query)

        if "fail" in query.lower():
            return SearchResponse(
                status=SearchStatus.FAILED,
                error_message="Rate limit exceeded on mock provider",
                query=query,
            )
        if "empty" in query.lower():
            return SearchResponse(
                status=SearchStatus.EMPTY,
                results=[],
                query=query,
            )

        # Standard successful query
        # Intentionally produce a duplicate canonical URL across different queries to test occurrence preservation
        return SearchResponse(
            status=SearchStatus.SUCCESS,
            results=[
                SearchResult(
                    url="https://techcrunch.com/2026/01/shared-article",
                    original_url="https://techcrunch.com/2026/01/shared-article?utm_source=mock",
                    title=f"Result for {query}",
                    snippet=f"Snippet describing {query}",
                    domain="techcrunch.com",
                    position=1,
                    query=query,
                ),
                SearchResult(
                    url=f"https://acme.org/news/{abs(hash(query))}",
                    original_url=f"https://acme.org/news/{abs(hash(query))}",
                    title=f"Acme News for {query}",
                    snippet=f"Company post on {query}",
                    domain="acme.org",
                    position=2,
                    query=query,
                ),
            ],
            query=query,
        )


class MockPageFetcher(PageFetcher):
    """Mock fetcher returning synthetic pages without network calls."""

    def __init__(self):
        pass

    def fetch(self, url: str) -> FetchOutcome:
        if "fail" in url.lower():
            return FetchOutcome(
                url=url,
                status=FetchStatus.CONNECTION_ERROR,
                error_message="Host unreachable",
                http_status_code=None,
                duration_ms=45.0,
            )

        record = SourceRecord(
            url=url,
            original_url=url,
            domain="mockdomain.com",
            title=f"Page title for {url}",
            raw_text=f"Full article text for {url}. Company announced significant operational expansion.",
            source_type=SourceType.NEWS,
            credibility_score=0.85,
            fetch_date=datetime.date(2026, 11, 7),
        )
        return FetchOutcome(
            url=url,
            status=FetchStatus.SUCCESS,
            source_record=record,
            http_status_code=200,
            duration_ms=120.0,
        )


class MockPlanner(ResearchPlanner):
    def __init__(self, plan: ResearchPlan):
        self._stub_plan = plan

    def plan(self, prospect, outreach, product=None, session_id=None):
        return ResearchPlan(
            session_id=session_id or self._stub_plan.session_id,
            objectives=self._stub_plan.objectives,
            total_query_budget=self._stub_plan.total_query_budget,
            max_iterations=self._stub_plan.max_iterations,
        )


class MockQueryGenerator(QueryGenerator):
    def __init__(self, queries_map: dict[str, list[str]]):
        self._map = queries_map

    def generate_queries(self, objective, prospect, existing_queries=None):
        raw = self._map.get(objective.objective_id, ["default query"])
        return [
            GeneratedQuery(
                query_text=q,
                objective_id=objective.objective_id,
                target_entity=objective.target_entity,
            )
            for q in raw
        ]


@pytest.fixture
def test_state_store():
    with tempfile.NamedTemporaryFile() as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


def test_research_execution_coordinator_lifecycle(test_state_store: StateStore):
    prospect = ProspectContext(
        full_name="Samantha Ray",
        company_name="Helios Systems",
        company_domain="helios.io",
        designation="VP of Product",
    )
    outreach = OutreachContext(
        objective="Invite to enterprise beta",
        sender_identity="Antigravity Team",
    )

    obj1 = ResearchObjective(
        objective_id="obj-id-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Samantha Ray's role at Helios",
    )
    obj2 = ResearchObjective(
        objective_id="obj-id-2",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Discover Helios Systems expansion",
    )
    plan = ResearchPlan(objectives=[obj1, obj2])

    queries_map = {
        "obj-id-1": ["Samantha Ray Helios Systems role", "Samantha Ray empty search"],
        "obj-id-2": ["Helios Systems Series B fail", "Helios Systems product release"],
    }

    mock_search = MockSearchProvider()
    coordinator = ResearchExecutionCoordinator(
        state_store=test_state_store,
        planner=MockPlanner(plan),
        query_generator=MockQueryGenerator(queries_map),
        search_client=SearchClient(provider=mock_search),
        page_fetcher=MockPageFetcher(),
    )

    result = coordinator.execute_discovery(
        prospect=prospect,
        outreach=outreach,
        session_id="session-discovery-1",
        max_results_per_query=3,
        max_pages_to_fetch=5,
    )

    assert result["session_id"] == "session-discovery-1"
    assert result["objectives_count"] == 2
    assert result["queries_count"] == 4

    # Verify persistence in database
    with test_state_store.uow() as uow:
        # 1. Session & Objectives
        sess = uow.research.get_session("session-discovery-1")
        assert sess is not None
        assert sess["status"] == "completed"

        objectives = uow.research.get_objectives("session-discovery-1")
        assert len(objectives) == 2

        # 2. Queries
        queries = uow.research.get_queries(session_id="session-discovery-1")
        assert len(queries) == 4

        # 3. Search Executions (Verify distinction between success, empty, and failed)
        failed_query = next(q for q in queries if "fail" in q["query_text"])
        empty_query = next(q for q in queries if "empty" in q["query_text"])
        success_query = next(q for q in queries if "product release" in q["query_text"])

        cursor = uow.conn.execute("SELECT * FROM search_executions;")
        all_execs = [dict(r) for r in cursor.fetchall()]
        assert len(all_execs) == 4

        failed_exec = next(e for e in all_execs if e["query_id"] == failed_query["query_id"])
        assert failed_exec["status"] == "failed"
        assert "Rate limit" in failed_exec["error_message"]

        empty_exec = next(e for e in all_execs if e["query_id"] == empty_query["query_id"])
        assert empty_exec["status"] == "empty"

        success_exec = next(e for e in all_execs if e["query_id"] == success_query["query_id"])
        assert success_exec["status"] == "success"

        # 4. Search Result Occurrences & Canonical Source Deduplication
        # Query 1 and Query 4 both returned: 'https://techcrunch.com/2026/01/shared-article'
        occurrences = uow.conn.execute("SELECT * FROM search_result_occurrences;").fetchall()
        assert len(occurrences) > 0

        shared_source = uow.sources.get_source_by_url("https://techcrunch.com/2026/01/shared-article")
        assert shared_source is not None

        # Occurrence-level provenance: Must have multiple occurrences pointing to the SAME source
        shared_occurrences = uow.research.get_occurrences_for_source(shared_source["source_id"])
        assert len(shared_occurrences) == 2
        # Different executions and positions preserved
        assert shared_occurrences[0]["execution_id"] != shared_occurrences[1]["execution_id"]

        # 5. Source Snapshots
        snapshots = uow.sources.get_snapshots_for_session("session-discovery-1")
        assert len(snapshots) > 0
        successful_snaps = [s for s in snapshots if s["fetch_status"] == "success"]
        assert len(successful_snaps) > 0
        assert "Full article text" in successful_snaps[0]["raw_text"]


def test_interrupted_execution_preserves_prior_commit_stages(test_state_store: StateStore):
    """Verify that if an error halts execution midway, prior stages (plan, queries, search executions) remain persisted."""
    prospect = ProspectContext(
        full_name="Interrupted User",
        company_name="Crash Corp",
    )
    outreach = OutreachContext(
        objective="Crash test",
        sender_identity="Antigravity Team",
    )
    plan = ResearchPlan(
        objectives=[
            ResearchObjective(
                objective_id="obj-crash-1",
                type=ResearchObjectiveType.IDENTITY_RESOLUTION,
                target_entity=TargetEntity.PROSPECT,
                description="Crash test objective",
            )
        ]
    )

    class CrashingPageFetcher(PageFetcher):
        def fetch(self, url: str):
            raise ConnectionResetError("Fatal network crash during page fetch")

    coordinator = ResearchExecutionCoordinator(
        state_store=test_state_store,
        planner=MockPlanner(plan),
        query_generator=MockQueryGenerator({"obj-crash-1": ["Crash Corp query"]}),
        search_client=SearchClient(provider=MockSearchProvider()),
        page_fetcher=CrashingPageFetcher(),
    )

    with pytest.raises(ConnectionResetError):
        coordinator.execute_discovery(
            prospect=prospect,
            outreach=outreach,
            session_id="session-interrupted-1",
        )

    # Verify that prior stages remain committed in DB even though the coordinator crashed
    with test_state_store.uow() as uow:
        sess = uow.research.get_session("session-interrupted-1")
        assert sess is not None
        assert sess["status"] == "in_progress"  # Not finalized because it crashed

        objs = uow.research.get_objectives("session-interrupted-1")
        assert len(objs) == 1

        queries = uow.research.get_queries("session-interrupted-1")
        assert len(queries) == 1

        # Search execution was committed
        execs = uow.conn.execute(
            "SELECT * FROM search_executions WHERE query_id = ?;",
            (queries[0]["query_id"],),
        ).fetchall()
        assert len(execs) == 1
