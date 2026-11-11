from __future__ import annotations

import tempfile
import uuid
from typing import Any, Dict, List, Optional
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration import (
    ApplicationOrchestrator,
    RunResult,
    RunStatus,
    StageIdentifier,
    StageStatus,
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
from src.signal_strategy.models import (
    CandidateSignal,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    SignalScore,
    StrategyResult,
)


def _make_dummy_prospect() -> ProspectContext:
    return ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
    )


def _make_dummy_product() -> ProductContext:
    return ProductContext(
        product_name="Open Inquiry Engine",
        description="Autonomous outreach research engine.",
        capabilities=["Real-time company intelligence", "Verifiable fact extraction"],
        value_proposition="Automate personalized, evidence-grounded research outreach.",
    )


def _make_dummy_outreach() -> OutreachContext:
    return OutreachContext(
        objective="Schedule introductory briefing",
        sender_identity="Alex Rivers, Lead Architect at SystemsLab",
        sender_name="Alex Rivers",
        sender_company="SystemsLab",
    )



import datetime


def _make_dummy_package(
    is_sufficient: bool = True,
    facts_count: int = 2,
    resolution_status: ResolutionStatus = ResolutionStatus.RESOLVED,
) -> ResearchPackage:
    source = SourceRecord(
        source_id="src_1",
        url="https://vanguardai.io/news/1",
        canonical_url="https://vanguardai.io/news/1",
        domain="vanguardai.io",
        raw_text="Vanguard AI expanded its distributed infrastructure in Q3.",
        fetch_date=datetime.date(2026, 11, 1),
        source_type=SourceType.COMPANY_WEBSITE,
        credibility_score=0.9,
    )
    evidence = EvidenceRecord(
        evidence_id="ev_1",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI expanded its distributed infrastructure.",
        credibility_score=0.9,
    )
    facts = []
    if facts_count > 0:
        facts.append(
            CanonicalFact(
                fact_id="fact_1",
                claim_text="Vanguard AI expanded infrastructure in Q3.",
                fact_category=FactCategory.PRODUCT_LAUNCH,
                confidence=ConfidenceLevel.HIGH,
                evidence_records=[evidence],
                primary_event_date=datetime.date(2026, 9, 1),
            )
        )
    return ResearchPackage(
        prospect_entity=EntityRecord(
            entity_id="ent_1",
            entity_type=EntityType.PERSON,
            raw_name="Elena Rostova",
            canonical_name="Elena Rostova",
            resolution_status=resolution_status,
            confidence=ConfidenceLevel.HIGH,
        ),
        sources=[source],
        evidence=[evidence],
        facts=facts,
        is_sufficient=is_sufficient,
        is_complete=True,
        missing_information=[] if is_sufficient else ["Current employment role unconfirmed"],
        sufficiency_notes="Sufficient coverage." if is_sufficient else "Critical gap remaining.",
    )



def _make_dummy_strategy_result(is_viable: bool = True) -> StrategyResult:
    if not is_viable:
        return StrategyResult(
            is_viable=False,
            rejection_reason="No commercial signals aligned with product capabilities.",
            unviable_reasons=["Alignment score 0.15 below minimum threshold 0.30."],
        )

    cand = CandidateSignal(
        signal_id="sig_1",
        label="Infrastructure Expansion",
        description="Vanguard AI expanded distributed infrastructure.",
        supporting_fact_ids=["fact_1"],
        fact_category=FactCategory.PRODUCT_LAUNCH,
        observed_event="Infrastructure cluster launch",
        business_implication="Scaling training workloads",
        commercial_opportunity="Needs high throughput data pipeline",
    )
    score = SignalScore(
        evidence_strength=0.9,
        recency_score=0.85,
        prospect_relevance=0.80,
        product_alignment=0.85,
        business_significance=0.80,
        composite_score=0.84,
    )
    ranked = RankedSignal(
        signal=cand,
        score=score,
        rationale="Strong recency and high alignment with AI engine capabilities.",
    )
    alignment = ProspectProductAlignment(
        prospect_role_relevance="Directly impacts research infrastructure roadmap.",
        business_implications="Increases model training speed while controlling compute cost.",
        capability_mapping=["Real-time company intelligence"],
    )
    strategy = OutreachStrategy(
        strategy_id="strat_1",
        primary_signal=ranked,
        supporting_signals=[],
        alignment=alignment,
        messaging_approach="Expansion Congratulation & Infrastructure Acceleration",
        central_message="Scale compute with automated verifiable intelligence.",
        suggested_subject_line="Scaling Vanguard AI infrastructure",
        cta="Open to a 10-minute briefing on compute scaling?",
        tone_guidance="Collaborative, executive, data-driven",
    )
    return StrategyResult(
        is_viable=True,
        strategy=strategy,
        ranked_signals=[ranked],
    )


class MockResearcher:
    def __init__(self, package: ResearchPackage, should_raise: bool = False):
        self.package = package
        self.should_raise = should_raise
        self.call_count = 0

    def conduct_research(self, **kwargs) -> ResearchPackage:
        self.call_count += 1
        if self.should_raise:
            raise RuntimeError("External network connection timeout during search.")
        return self.package


class MockStrategyService:
    def __init__(self, result: StrategyResult):
        self.result = result
        self.call_count = 0

    def develop_strategy(self, **kwargs) -> StrategyResult:
        self.call_count += 1
        return self.result


def test_app_orchestrator_successful_flow():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True)
        strat_res = _make_dummy_strategy_result(is_viable=True)

        researcher = MockResearcher(package)
        strat_service = MockStrategyService(strat_res)

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res: RunResult = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.COMPLETED
        assert researcher.call_count == 1
        assert strat_service.call_count == 1
        assert res.strategy_result is not None
        assert res.strategy_result.is_viable is True
        assert len(res.warnings) == 0

        # Verify stages
        stage_names = [s.stage for s in res.stage_records]
        assert StageIdentifier.VALIDATION in stage_names
        assert StageIdentifier.INITIALIZATION in stage_names
        assert StageIdentifier.RESEARCH in stage_names
        assert StageIdentifier.SUFFICIENCY_GATE in stage_names
        assert StageIdentifier.SIGNAL_STRATEGY in stage_names


def test_app_orchestrator_insufficient_research_halts_before_strategy():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        # Insufficient research package
        package = _make_dummy_package(is_sufficient=False)
        researcher = MockResearcher(package)
        strat_service = MockStrategyService(_make_dummy_strategy_result(True))

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.INSUFFICIENT_RESEARCH
        assert researcher.call_count == 1
        # Strategy service must NOT have been called
        assert strat_service.call_count == 0
        assert res.strategy_result is None
        assert any("insufficient" in w.lower() for w in res.warnings)


def test_app_orchestrator_zero_facts_halts_before_strategy():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True, facts_count=0)
        researcher = MockResearcher(package)
        strat_service = MockStrategyService(_make_dummy_strategy_result(True))

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.INSUFFICIENT_RESEARCH
        assert strat_service.call_count == 0
        assert any("zero" in w.lower() for w in res.warnings)


def test_app_orchestrator_unresolved_identity_halts():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True, resolution_status=ResolutionStatus.UNRESOLVED)
        researcher = MockResearcher(package)
        strat_service = MockStrategyService(_make_dummy_strategy_result(True))

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.INSUFFICIENT_RESEARCH
        assert strat_service.call_count == 0


def test_app_orchestrator_no_viable_signal():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True)
        unviable_strat = _make_dummy_strategy_result(is_viable=False)

        researcher = MockResearcher(package)
        strat_service = MockStrategyService(unviable_strat)

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.NO_VIABLE_SIGNAL
        assert researcher.call_count == 1
        assert strat_service.call_count == 1
        assert res.error_message is not None


def test_app_orchestrator_recovery_and_retrieval_after_restart():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True)
        strat_res = _make_dummy_strategy_result(is_viable=True)

        researcher = MockResearcher(package)
        strat_service = MockStrategyService(strat_res)

        orch1 = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res1 = orch1.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )
        run_id = res1.run_id

        # Persist fact in database matching the session
        with store.uow() as uow:
            uow.assertions.create_canonical_fact(
                fact_id="fact_1",
                session_id=res1.research_session_id,
                claim_text="Vanguard AI expanded infrastructure in Q3.",
                fact_category="product_launch",
                primary_event_date="2026-09-01",
                composite_confidence="HIGH",
            )

        # Simulate process termination by constructing a brand new orchestrator
        store2 = StateStore(tf.name)
        orch2 = ApplicationOrchestrator(state_store=store2)

        snapshot = orch2.get_run(run_id)
        assert snapshot is not None
        assert snapshot.run_id == run_id
        assert snapshot.status == RunStatus.COMPLETED
        assert snapshot.prospect.full_name == "Elena Rostova"
        assert snapshot.product.product_name == "Open Inquiry Engine"
        assert snapshot.strategy_result is not None
        assert "Congratulation" in snapshot.strategy_result.strategy.messaging_approach

        # Verify stages inspection
        stages = orch2.get_run_stages(run_id)
        assert len(stages) >= 5

        # Verify run listing
        runs = orch2.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id

        # Verify explainability
        explain = orch2.get_strategy_explainability(run_id)
        assert explain["run_id"] == run_id
        assert explain["primary_signal"]["signal"]["label"] == "Infrastructure Expansion"
        assert len(explain["supporting_facts"]) == 1


def test_app_orchestrator_failure_handling():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True)
        researcher = MockResearcher(package, should_raise=True)
        strat_service = MockStrategyService(_make_dummy_strategy_result(True))

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.FAILED
        assert "timeout" in res.error_message
        assert strat_service.call_count == 0

        # Must be persisted as failed
        snapshot = orchestrator.get_run(res.run_id)
        assert snapshot.status == RunStatus.FAILED
        assert "timeout" in snapshot.error_message


def test_app_orchestrator_collision_resistant_identifiers():
    """Verify session_id and job_id use full collision-resistant hex UUIDs, not 8-char prefixes."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        package = _make_dummy_package(is_sufficient=True)
        researcher = MockResearcher(package)
        strat_service = MockStrategyService(_make_dummy_strategy_result(True))

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
        )

        res = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.research_session_id is not None
        assert res.research_session_id.startswith("session_")
        session_hex = res.research_session_id.replace("session_", "")
        assert len(session_hex) == 32  # Full collision-resistant 128-bit hex UUID
        # Must be valid hex
        int(session_hex, 16)

        # Inspect snapshot and job record in db
        snapshot = orchestrator.get_run(res.run_id)
        assert snapshot is not None
        assert snapshot.job_id is not None
        assert snapshot.job_id.startswith("job_")
        job_hex = snapshot.job_id.replace("job_", "")
        assert len(job_hex) == 32
        int(job_hex, 16)

