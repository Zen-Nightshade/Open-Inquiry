import datetime
import tempfile
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from src.db.state_store import StateStore
from src.researcher.evidence_request.executor import EvidenceRequestExecutor
from src.researcher.evidence_request.models import (
    EvidenceFeedbackBudget,
    EvidenceGapReason,
    EvidenceRequest,
    EvidenceRequestStatus,
)
from src.researcher.extraction.models import FetchOutcome, FetchStatus
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    ResolutionStatus,
    SourceRecord,
    SourceType,
    VerificationStatus,
)
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus


def _make_sample_request(
    entity_id: str = "ent_target",
    predicate: str = "operates_cloud",
) -> EvidenceRequest:
    return EvidenceRequest(
        run_id="run_researcher_test",
        target_entity_id=entity_id,
        target_entity_name="Target Org",
        requested_predicate=predicate,
        neutral_question="Does Target Org operate AWS infrastructure?",
        neutral_search_terms=["Target Org AWS deployment"],
        disconfirming_search_terms=["Target Org on-premise datacenter private cloud"],
        gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
        provenance_stage="strategy_product_gate",
    )


def test_neutral_and_disconfirming_query_execution():
    """Verify that both neutral and counter-evidence queries are generated and searched."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        captured_queries: List[str] = []

        class MockSearchClient:
            def search(self, query: str, max_results: int = 3):
                captured_queries.append(query)
                return SearchResponse(
                    status=SearchStatus.SUCCESS,
                    query=query,
                    results=[],
                    total_results=0,
                )

        mock_researcher = MagicMock()
        mock_researcher.search_client = MockSearchClient()
        mock_researcher.page_fetcher = None
        mock_researcher.fact_service = None

        executor = EvidenceRequestExecutor(researcher=mock_researcher, state_store=store)
        req = _make_sample_request()
        budget = EvidenceFeedbackBudget.create(baseline_budget_units=20)
        acc = ResearchAccumulator()

        res = executor.execute_requests(
            requests=[req],
            session_id=None,
            run_id="run_test",
            budget=budget,
            accumulator=acc,
            round_number=1,
        )

        # Both neutral and disconfirming search terms must be executed
        assert len(captured_queries) == 2
        assert "Target Org AWS deployment" in captured_queries
        assert "Target Org on-premise datacenter private cloud" in captured_queries

        # With empty search results, request status must be UNRESOLVED (not NO_MATCH)
        assert len(res.requests) == 1
        completed_req = res.requests[0]
        assert completed_req.status == EvidenceRequestStatus.UNRESOLVED
        assert completed_req.status != "no_match"
        assert set(completed_req.searched_scope) == set(captured_queries)


def test_m24_ambiguous_entity_attribution_remains_unresolved():
    """Verify that facts are never bound to an entity with ambiguous or unresolved status."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        # Seed an ambiguous entity in the database
        ambiguous_entity_id = "ent_ambig_999"
        with store.uow() as uow:
            uow.entities.create_entity(
                entity_id=ambiguous_entity_id,
                entity_type="organization",
                raw_name="Acme Global Inc",
                resolution_status="ambiguous",
            )

        mock_researcher = MagicMock()
        mock_researcher.search_client = MagicMock()
        executor = EvidenceRequestExecutor(researcher=mock_researcher, state_store=store)

        req = _make_sample_request(entity_id=ambiguous_entity_id)
        budget = EvidenceFeedbackBudget.create(baseline_budget_units=20)
        acc = ResearchAccumulator()

        res = executor.execute_requests(
            requests=[req],
            session_id=None,
            run_id="run_test",
            budget=budget,
            accumulator=acc,
            round_number=1,
        )

        assert len(res.requests) == 1
        req_res = res.requests[0]
        assert req_res.status == EvidenceRequestStatus.UNRESOLVED
        assert "ambiguous" in (req_res.rejection_reason or "").lower()
        # Search was not executed for ambiguous entity
        mock_researcher.search_client.search.assert_not_called()


def test_contradictory_evidence_survives_extraction_and_resolution():
    """Verify that counter-evidence and contradictions survive extraction into the accumulator."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        # Create mock search client returning counter-evidence URL
        class MockSearchClient:
            def search(self, query: str, max_results: int = 3):
                return SearchResponse(
                    status=SearchStatus.SUCCESS,
                    query=query,
                    results=[
                        SearchResult(
                            title="Acme migrates off AWS to on-premise datacenter",
                            url="https://targetorg.com/press/datacenter-migration",
                            original_url="https://targetorg.com/press/datacenter-migration",
                            domain="targetorg.com",
                            snippet="Target Org has decommissioned AWS in favor of self-hosted infrastructure.",
                            position=1,
                            query=query,
                        )
                    ],
                )

        # Create mock page fetcher
        class MockPageFetcher:
            def fetch_page(self, url: str) -> FetchOutcome:
                source = SourceRecord(
                    source_id="src_counter_1",
                    url=url,
                    domain="targetorg.com",
                    raw_text="Target Org completed migration off cloud to proprietary datacenters.",
                    fetch_date=datetime.date(2026, 11, 1),
                    source_type=SourceType.COMPANY_WEBSITE,
                    credibility_score=0.95,
                )
                return FetchOutcome(
                    status=FetchStatus.SUCCESS,
                    url=url,
                    http_status_code=200,
                    source_record=source,
                )

        # Create mock fact service extracting contradictory fact
        class MockFactService:
            def extract_and_persist_facts_for_session(self, session_id, snapshot_id, accumulator):
                ev = EvidenceRecord(
                    evidence_id="ev_counter_1",
                    source_id="src_counter_1",
                    verbatim_excerpt="Target Org completed migration off cloud to proprietary datacenters.",
                    credibility_score=0.95,
                )
                fact = CanonicalFact(
                    fact_id="fact_counter_1",
                    claim_text="Target Org does not use AWS and operates proprietary datacenters.",
                    fact_category=FactCategory.STRATEGY,
                    confidence=ConfidenceLevel.HIGH,
                    evidence_records=[ev],
                )
                if any(f.fact_id == "fact_counter_1" for f in accumulator.facts):
                    accumulator.merge_or_add_fact(fact, match_fact_id="fact_counter_1")
                else:
                    accumulator.merge_or_add_fact(fact)
                return [fact]

        mock_researcher = MagicMock()
        mock_researcher.search_client = MockSearchClient()
        mock_researcher.page_fetcher = MockPageFetcher()
        mock_researcher.fact_service = MockFactService()
        mock_researcher.syndication_detector = None

        executor = EvidenceRequestExecutor(researcher=mock_researcher, state_store=store)
        req = _make_sample_request()
        budget = EvidenceFeedbackBudget.create(baseline_budget_units=20)
        acc = ResearchAccumulator()

        res = executor.execute_requests(
            requests=[req],
            session_id=None,
            run_id="run_test",
            budget=budget,
            accumulator=acc,
            round_number=1,
        )

        assert len(res.requests) == 1
        req_res = res.requests[0]
        assert req_res.status == EvidenceRequestStatus.COMPLETED
        assert "fact_counter_1" in req_res.resulting_fact_ids
        assert len(acc.facts) == 1
        assert "proprietary datacenters" in acc.facts[0].claim_text


def test_m23_syndication_clustering_during_feedback():
    """Verify that multiple syndicated sources fetched during feedback are clustered via SyndicationDetector."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        class MockSearchClient:
            def search(self, query: str, max_results: int = 3):
                return SearchResponse(
                    status=SearchStatus.SUCCESS,
                    query=query,
                    results=[
                        SearchResult(
                            title="Wire 1",
                            url="https://wire1.com/story",
                            original_url="https://wire1.com/story",
                            domain="wire1.com",
                            snippet="press release",
                            position=1,
                            query=query,
                        ),
                        SearchResult(
                            title="Wire 2",
                            url="https://wire2.com/story",
                            original_url="https://wire2.com/story",
                            domain="wire2.com",
                            snippet="press release",
                            position=2,
                            query=query,
                        ),
                    ],
                )

        fetch_idx = [0]
        class MockPageFetcher:
            def fetch_page(self, url: str) -> FetchOutcome:
                fetch_idx[0] += 1
                source = SourceRecord(
                    source_id=f"src_wire_{fetch_idx[0]}",
                    url=url,
                    domain="wire.com",
                    raw_text="Identical press release syndicated text across wire services.",
                    fetch_date=datetime.date(2026, 11, 1),
                    source_type=SourceType.NEWS,
                    credibility_score=0.8,
                )
                return FetchOutcome(
                    status=FetchStatus.SUCCESS,
                    url=url,
                    http_status_code=200,
                    source_record=source,
                )

        synd_detector_called = [False]
        class MockSyndicationDetector:
            def detect_syndication(self, sources):
                synd_detector_called[0] = True
                res = MagicMock()
                res.clusters = []
                return res

        mock_researcher = MagicMock()
        mock_researcher.search_client = MockSearchClient()
        mock_researcher.page_fetcher = MockPageFetcher()
        mock_researcher.fact_service = None
        mock_researcher.syndication_detector = MockSyndicationDetector()

        executor = EvidenceRequestExecutor(researcher=mock_researcher, state_store=store)
        req = _make_sample_request()
        budget = EvidenceFeedbackBudget.create(baseline_budget_units=20)
        acc = ResearchAccumulator()

        res = executor.execute_requests(
            requests=[req],
            session_id=None,
            run_id="run_test",
            budget=budget,
            accumulator=acc,
            round_number=1,
        )

        assert synd_detector_called[0] is True
        assert len(acc.sources) >= 2
