import datetime
import json
import tempfile
from typing import List, Optional
import uuid
from unittest.mock import MagicMock

import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.schema import EXPECTED_DOMAIN_TABLES, is_fts_table
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator
from src.orchestration.models import RunStatus, StageIdentifier, StageStatus
from src.researcher.evidence_request.models import (
    EvidenceFeedbackBudget,
    EvidenceGapReason,
    EvidenceRequest,
    EvidenceRequestStatus,
    FeedbackRoundResult,
)
from src.researcher.extraction.models import FetchOutcome, FetchStatus
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
    SourceType,
)
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.signal_strategy.models import (
    CandidateSignal,
    EligibilityStatus,
    GateIdentifier,
    GateResult,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    SignalEligibility,
    SignalScore,
    StrategyResult,
)


def _make_dummy_strategy(primary_signal: RankedSignal, condition: Optional[str] = None) -> OutreachStrategy:
    is_cond = condition is not None
    unresolved = [condition] if condition else []
    return OutreachStrategy(
        strategy_id=f"strat_{uuid.uuid4().hex[:8]}",
        primary_signal=primary_signal,
        supporting_signals=[],
        alignment=ProspectProductAlignment(
            prospect_role_relevance="Oversees analytics infrastructure",
            business_implications="Potential compute cost escalation",
            capability_mapping=["Cluster rightsizing"],
        ),
        messaging_approach="Infrastructure Efficiency",
        central_message="Optimize container costs as you scale pipelines.",
        suggested_subject_line="Kubernetes scaling at Apex Data",
        cta="Open to comparing cluster costs?",
        tone_guidance="Collaborative, technical, consultative",
        is_conditional=is_cond,
        unresolved_conditions=unresolved,
    )


def _make_context():
    prospect = ProspectContext(
        full_name="Jordan Lee",
        company_name="Apex Data",
        company_domain="apexdata.com",
        designation="VP of Engineering",
    )
    product = ProductContext(
        product_name="CloudOptimizer",
        description="Kubernetes cost optimization engine.",
        capabilities=["Automated pod autoscaling", "Cluster rightsizing"],
        value_proposition="Cut Kubernetes cloud compute costs by 40%.",
    )
    outreach = OutreachContext(
        objective="Introduce CloudOptimizer for cluster efficiency",
        sender_identity="Sam Taylor, Principal Cloud Architect",
        sender_name="Sam Taylor",
        sender_company="OptimizerSystems",
    )
    return prospect, product, outreach


def _make_initial_package() -> ResearchPackage:
    source = SourceRecord(
        source_id="src_init_1",
        url="https://apexdata.com/news/expansion",
        domain="apexdata.com",
        raw_text="Apex Data is expanding engineering teams to build new high-throughput analytics pipelines.",
        fetch_date=datetime.date(2026, 11, 1),
        source_type=SourceType.COMPANY_WEBSITE,
        credibility_score=0.9,
    )
    evidence = EvidenceRecord(
        evidence_id="ev_init_1",
        source_id="src_init_1",
        verbatim_excerpt="Apex Data is expanding engineering teams to build new analytics pipelines.",
        credibility_score=0.9,
    )
    fact = CanonicalFact(
        fact_id="fact_init_1",
        claim_text="Apex Data is expanding engineering teams for new pipelines.",
        fact_category=FactCategory.HIRING,
        confidence=ConfidenceLevel.HIGH,
        evidence_records=[evidence],
    )
    return ResearchPackage(
        prospect_entity=EntityRecord(
            entity_id="ent_prospect_1",
            entity_type=EntityType.PERSON,
            raw_name="Jordan Lee",
            canonical_name="Jordan Lee",
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
        ),
        company_entity=EntityRecord(
            entity_id="ent_company_1",
            entity_type=EntityType.ORGANIZATION,
            raw_name="Apex Data",
            canonical_name="Apex Data Inc",
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
        ),
        sources=[source],
        evidence=[evidence],
        facts=[fact],
        is_sufficient=True,
        is_complete=True,
    )


def _make_dummy_score(composite: float = 0.82) -> SignalScore:
    return SignalScore(
        evidence_strength=0.85,
        recency_score=0.8,
        prospect_relevance=0.85,
        product_alignment=0.8,
        business_significance=0.8,
        composite_score=composite,
    )


def _make_dummy_eligibility(condition: str) -> SignalEligibility:
    return SignalEligibility(
        is_eligible=True,
        status=EligibilityStatus.CONDITIONAL,
        identity_gate=GateResult(gate=GateIdentifier.IDENTITY, passed=True, reason="Valid identity"),
        evidence_gate=GateResult(gate=GateIdentifier.EVIDENCE, passed=True, reason="Adequate evidence"),
        product_gate=GateResult(
            gate=GateIdentifier.PRODUCT_COMPATIBILITY,
            passed=True,
            reason=f"Conditional on {condition}.",
            details={"missing_prerequisites": [condition]},
        ),
        persona_gate=GateResult(gate=GateIdentifier.PERSONA, passed=True, reason="Target buyer"),
        unresolved_conditions=[condition],
    )


def test_evidence_feedback_gap_resolution_makes_strategy_viable():
    """Verify end-to-end feedback round where gap is detected, researched, and resolved, updating strategy viability."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        prospect, product, outreach = _make_context()
        initial_package = _make_initial_package()

        # Step 1: Strategy with missing prerequisite
        candidate_sig = CandidateSignal(
            signal_id="sig_pipeline_expansion",
            label="Analytics Pipeline Scaling",
            description="Apex Data scaling analytics pipelines.",
            supporting_fact_ids=["fact_init_1"],
            fact_category=FactCategory.HIRING,
        )
        eligibility = _make_dummy_eligibility("kubernetes")
        ranked_sig = RankedSignal(
            signal=candidate_sig,
            score=_make_dummy_score(0.82),
            rationale="Strong alignment with Kubernetes capabilities.",
            eligibility=eligibility,
        )

        initial_strat = _make_dummy_strategy(ranked_sig, condition="kubernetes")
        initial_strat_result = StrategyResult(
            is_viable=True,
            strategy=initial_strat,
            ranked_signals=[ranked_sig],
        )

        # Viable strategy after feedback finds Kubernetes
        resolved_strat = initial_strat.model_copy(update={
            "is_conditional": False,
            "unresolved_conditions": [],
        })
        resolved_strat_result = StrategyResult(
            is_viable=True,
            strategy=resolved_strat,
            ranked_signals=[ranked_sig],
        )

        call_counts = {"strat": 0}

        class MockStrategyService:
            def develop_strategy(self, research_package, **kwargs):
                call_counts["strat"] += 1
                # If research package now contains Kubernetes fact, return resolved viable strategy
                has_k8s = any("kubernetes" in f.claim_text.lower() for f in research_package.facts)
                if has_k8s:
                    return resolved_strat_result
                return initial_strat_result

        # Mock researcher with search, fetch, and extraction capabilities
        class MockSearchClient:
            def search(self, query: str, max_results: int = 3):
                return SearchResponse(
                    status=SearchStatus.SUCCESS,
                    query=query,
                    results=[
                        SearchResult(
                            title="Apex Data Kubernetes Engineering Blog",
                            url="https://apexdata.com/blog/migrating-to-eks-kubernetes",
                            original_url="https://apexdata.com/blog/migrating-to-eks-kubernetes",
                            domain="apexdata.com",
                            snippet="Apex Data completed migration to AWS EKS Kubernetes clusters.",
                            position=1,
                            query=query,
                        )
                    ],
                )

        class MockPageFetcher:
            def fetch_page(self, url: str) -> FetchOutcome:
                source = SourceRecord(
                    source_id="src_k8s_feed",
                    url=url,
                    domain="apexdata.com",
                    raw_text="Apex Data runs all real-time analytics workloads on production Kubernetes (EKS).",
                    fetch_date=datetime.date(2026, 11, 2),
                    source_type=SourceType.COMPANY_WEBSITE,
                    credibility_score=0.95,
                )
                return FetchOutcome(
                    status=FetchStatus.SUCCESS,
                    url=url,
                    http_status_code=200,
                    source_record=source,
                )

        class MockFactService:
            def extract_and_persist_facts_for_session(self, session_id, snapshot_id, accumulator):
                ev = EvidenceRecord(
                    evidence_id="ev_k8s_1",
                    source_id="src_k8s_feed",
                    verbatim_excerpt="Apex Data runs all real-time analytics workloads on production Kubernetes (EKS).",
                    credibility_score=0.95,
                )
                fact = CanonicalFact(
                    fact_id="fact_k8s_verified",
                    claim_text="Apex Data operates production Kubernetes clusters on AWS EKS.",
                    fact_category=FactCategory.STRATEGY,
                    confidence=ConfidenceLevel.HIGH,
                    evidence_records=[ev],
                )
                accumulator.merge_or_add_fact(fact)
                return [fact]

        mock_researcher = MagicMock()
        mock_researcher.conduct_research.return_value = initial_package
        mock_researcher.search_client = MockSearchClient()
        mock_researcher.page_fetcher = MockPageFetcher()
        mock_researcher.fact_service = MockFactService()
        mock_researcher.syndication_detector = None

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=mock_researcher,
            strategy_service=MockStrategyService(),
        )

        res = orchestrator.run(
            prospect=prospect,
            product=product,
            outreach=outreach,
        )

        # 1. Strategy was re-evaluated once new facts arrived
        assert call_counts["strat"] == 2

        # 2. Feedback round result was recorded
        assert res.feedback_round is not None
        assert res.feedback_round.round_number == 1
        assert res.feedback_round.strategy_revised is True
        assert len(res.feedback_round.requests) == 2
        assert res.feedback_round.requests[0].status == EvidenceRequestStatus.COMPLETED
        assert "fact_k8s_verified" in res.feedback_round.requests[0].resulting_fact_ids

        # 3. Stage records contain RESEARCH_FEEDBACK
        stage_names = [r.stage for r in res.stage_records]
        assert StageIdentifier.RESEARCH_FEEDBACK in stage_names
        fb_stage = next(r for r in res.stage_records if r.stage == StageIdentifier.RESEARCH_FEEDBACK)
        assert fb_stage.status == StageStatus.COMPLETED
        assert fb_stage.details["requests_count"] == 2

        # 4. Persistence verification: feedback_round_json is persisted in DB
        db_run = store.get_run(res.run_id)
        assert db_run is not None
        assert db_run.get("feedback_round_json") is not None
        persisted_fb = json.loads(db_run["feedback_round_json"])
        assert persisted_fb["round_number"] == 1
        assert persisted_fb["requests"][0]["status"] == "completed"

        # 5. Snapshot retrieval re-hydrates feedback_round correctly
        snapshot = orchestrator.get_run(res.run_id)
        assert snapshot is not None
        assert snapshot.feedback_round is not None
        assert snapshot.feedback_round.requests[0].resulting_fact_ids == ["fact_k8s_verified"]


def test_evidence_feedback_unresolved_maintains_history_and_invalidates():
    """Verify that when feedback research cannot find facts (UNRESOLVED), strategy remains unviable/invalidated without crashing."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        prospect, product, outreach = _make_context()
        initial_package = _make_initial_package()

        candidate_sig = CandidateSignal(
            signal_id="sig_pipeline_expansion",
            label="Analytics Scaling",
            description="Apex Data scaling analytics.",
            supporting_fact_ids=["fact_init_1"],
            fact_category=FactCategory.HIRING,
        )
        ranked_sig = RankedSignal(
            signal=candidate_sig,
            score=_make_dummy_score(0.8),
            rationale="Database scaling rationale.",
            eligibility=_make_dummy_eligibility("on_premise_oracle_db"),
        )
        initial_strat = _make_dummy_strategy(ranked_sig, condition="on_premise_oracle_db")
        initial_strat_result = StrategyResult(
            is_viable=True,
            strategy=initial_strat,
            ranked_signals=[ranked_sig],
        )

        class MockEmptySearchClient:
            def search(self, query: str, max_results: int = 3):
                return SearchResponse(
                    status=SearchStatus.EMPTY,
                    query=query,
                    results=[],
                    total_results=0,
                )

        mock_researcher = MagicMock()
        mock_researcher.conduct_research.return_value = initial_package
        mock_researcher.search_client = MockEmptySearchClient()
        mock_researcher.page_fetcher = None
        mock_researcher.fact_service = None
        mock_researcher.syndication_detector = None

        strat_calls = [0]
        class MockStrategyService:
            def develop_strategy(self, **kwargs):
                strat_calls[0] += 1
                if strat_calls[0] == 1:
                    return initial_strat_result
                return StrategyResult(
                    is_viable=False,
                    rejection_reason="Prerequisite 'on_premise_oracle_db' unverified in research.",
                    unviable_reasons=["Prerequisite 'on_premise_oracle_db' unverified in research."],
                )

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=mock_researcher,
            strategy_service=MockStrategyService(),
        )

        res = orchestrator.run(
            prospect=prospect,
            product=product,
            outreach=outreach,
        )

        # Unresolved search does not produce new facts, so strategy becomes invalidated
        assert res.feedback_round is not None
        assert res.feedback_round.requests[0].status == EvidenceRequestStatus.UNRESOLVED
        assert res.feedback_round.strategy_invalidated is True

        # Database still contains run with persisted audit trail
        db_run = store.get_run(res.run_id)
        assert db_run is not None
        assert db_run.get("feedback_round_json") is not None


def test_schema_36_tables_preserved_invariant():
    """Verify that exactly 36 physical domain tables are maintained in EXPECTED_DOMAIN_TABLES and SQLite database."""
    assert len(EXPECTED_DOMAIN_TABLES) == 36

    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        with store.uow() as uow:
            cursor = uow.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = {row[0] for row in cursor.fetchall()}
            domain_tables = {t for t in tables if not is_fts_table(t)}
            fts_tables = {t for t in tables if is_fts_table(t)}

        assert domain_tables == EXPECTED_DOMAIN_TABLES
        assert len(domain_tables) == 36
        assert "source_snapshots_fts" in fts_tables
        assert "evidence_passages_fts" in fts_tables
