from __future__ import annotations

import tempfile
from typing import Any, List, Optional
import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CompatibilityStatus,
    OutreachContext,
    ProductContext,
    ProspectContext,
)
from src.db.state_store import StateStore
from src.email_generation.generator import EmailGenerator, EmailGenerationError
from src.email_generation.linter import ReadabilityLinter
from src.email_generation.models import (
    HumanReviewStatus,
    LinterSeverity,
    RawEmailDraftOutput,
    VerifiedOutreachDraft,
)
from src.email_generation.service import (
    DraftPersistenceError,
    EmailGenerationService,
)
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
    EvidenceRecord,
    FactCategory,
    ResearchPackage,
    VerificationStatus,
)
from src.signal_strategy.models import (
    CandidateSignal,
    CTAType,
    EligibilityStatus,
    GateIdentifier,
    GateResult,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    SignalEligibility,
    SignalScore,
    StrategyResult,
    ToulminArgument,
    ToulminBacking,
    ToulminClaim,
    ToulminGrounds,
    ToulminQualifier,
    ToulminRebuttal,
    ToulminWarrant,
)


class MockGenerator(EmailGenerator):
    def __init__(self, outputs: Optional[List[Any]] = None):
        self.outputs = list(outputs) if outputs else []
        self.call_count = 0
        self.last_rewrite_feedback: Optional[List[str]] = None

    def generate(
        self,
        prospect: ProspectContext,
        product: ProductContext,
        strategy: OutreachStrategy,
        outreach: OutreachContext,
        research_package: Optional[ResearchPackage] = None,
        rewrite_feedback: Optional[List[str]] = None,
        attempt_num: int = 1,
    ) -> RawEmailDraftOutput:
        self.call_count += 1
        self.last_rewrite_feedback = rewrite_feedback
        if not self.outputs:
            return RawEmailDraftOutput(
                subject="Cluster Telemetry Inquiry",
                body=(
                    "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
                    "When clusters grow fast, tracking can create heavy system lag. We built PulseRadar to cut node latency by 50% without adding extra overhead to your stack.\n\n"
                    "Are you open to comparing cluster telemetry benchmarks this week?"
                ),
            )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _make_dummy_prospect() -> ProspectContext:
    return ProspectContext(
        full_name="Elena Vance",
        company_name="Black Mesa Research",
        company_domain="blackmesa.gov",
        designation="Chief Technology Officer",
    )


def _make_dummy_product() -> ProductContext:
    return ProductContext(
        product_name="PulseRadar",
        description="Real-time signal intelligence for high-throughput distributed architectures",
        value_proposition="Identify and eliminate distributed bottleneck anomalies instantly",
    )


def _make_dummy_outreach() -> OutreachContext:
    return OutreachContext(
        objective="Initiate discovery conversation on cluster observability",
        sender_identity="Alex Rivers, VP of Technical Partnerships at Vanguard Labs",
        sender_name="Alex Rivers",
        sender_company="Vanguard Labs",
        constraints=["Keep below 100 words"],
        tone="direct, peer-to-peer",
    )


def _make_dummy_toulmin(proof_points=None, fact_ids=None) -> ToulminArgument:
    return ToulminArgument(
        grounds=ToulminGrounds(
            fact_ids=fact_ids or ["fact-101"],
            observed_event="Black Mesa deployed 10,000 distributed compute nodes",
            verification_status=VerificationStatus.SUPPORTED,
            composite_confidence=ConfidenceLevel.HIGH,
        ),
        warrant=ToulminWarrant(
            capability_id="cap-dist-telemetry",
            capability_name="Real-time Node Telemetry",
            reasoning="Large-scale node deployment introduces severe observability bottlenecks",
        ),
        backing=ToulminBacking(
            capability_id="cap-dist-telemetry",
            whitelisted_proof_points=proof_points or ["50% reduction in telemetry lag"],
            has_whitelisted_proof=True,
            provenance=OutputProvenance(origin=ProvenanceOrigin.MANUAL_INPUT, component="test"),
        ),
        qualifier=ToulminQualifier(
            eligibility_status=EligibilityStatus.ELIGIBLE,
            compatibility_status=CompatibilityStatus.VERIFIED_COMPATIBLE,
            verification_status=VerificationStatus.SUPPORTED,
            composite_confidence=ConfidenceLevel.HIGH,
            qualification_language="Directly applicable based on verified expansion",
        ),
        claim=ToulminClaim(
            claim_type="value_assertion",
            text="Cut cluster latency by 50% with verified telemetry probes",
            target_capability="cap-dist-telemetry",
            is_qualified=True,
        ),
        rebuttal=ToulminRebuttal(),
    )


def _make_dummy_strategy_result(
    is_viable: bool = True,
    has_strategy: bool = True,
    has_toulmin: bool = True,
) -> StrategyResult:
    if not is_viable:
        return StrategyResult(
            is_viable=False,
            strategy=None,
            rejection_reason="No viable signals met eligibility thresholds",
            unviable_reasons=["Hard exclusion triggered"],
        )

    if not has_strategy:
        return StrategyResult(
            is_viable=True,
            strategy=None,
        )

    cand = CandidateSignal(
        signal_id="sig-test-1",
        label="Cluster Rollout",
        description="Massive distributed cluster deployment",
        supporting_fact_ids=["fact-101"],
        fact_category=FactCategory.CHALLENGE,
        observed_event="Massive cluster expansion",
        target_capability="cap-dist-telemetry",
    )
    score = SignalScore(
        evidence_strength=0.9,
        recency_score=0.9,
        prospect_relevance=0.9,
        product_alignment=0.9,
        business_significance=0.85,
        composite_score=0.88,
    )
    elig = SignalEligibility(
        status=EligibilityStatus.ELIGIBLE,
        is_eligible=True,
        identity_gate=GateResult(gate=GateIdentifier.IDENTITY, passed=True, reason="Valid"),
        evidence_gate=GateResult(gate=GateIdentifier.EVIDENCE, passed=True, reason="Adequate"),
        product_gate=GateResult(gate=GateIdentifier.PRODUCT_COMPATIBILITY, passed=True, reason="Compatible"),
        persona_gate=GateResult(gate=GateIdentifier.PERSONA, passed=True, reason="Relevant"),
    )
    ranked = RankedSignal(signal=cand, score=score, rationale="Strong fit", eligibility=elig)

    strategy = OutreachStrategy(
        strategy_id="strat-test-1",
        primary_signal=ranked,
        alignment=ProspectProductAlignment(
            prospect_role_relevance="Telemetry aligns directly with CTO priorities",
            business_implications="Cluster deployment creates telemetry latency bottlenecks",
            capability_mapping=["cap-dist-telemetry"],
        ),
        messaging_approach="value-driven-expansion",
        central_message="Optimizing multi-node telemetry overhead across distributed compute clusters",
        suggested_subject_line="Scaling cluster telemetry at Black Mesa",
        cta="Open to comparing cluster telemetry benchmarks this week?",
        cta_type=CTAType.VALUE_EXCHANGE,
        cta_rationale="Low friction technical conversation",
        tone_guidance="Direct, engineer-to-engineer, zero corporate fluff",
        constraint_checklist=["Under 100 words"],
        toulmin_argument=_make_dummy_toulmin() if has_toulmin else None,
    )

    return StrategyResult(
        is_viable=True,
        strategy=strategy,
        ranked_signals=[ranked],
    )


def test_service_generate_draft_happy_path_with_persistence():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        generator = MockGenerator()
        linter = ReadabilityLinter()
        service = EmailGenerationService(generator=generator, linter=linter, state_store=store)

        prospect = _make_dummy_prospect()
        product = _make_dummy_product()
        outreach = _make_dummy_outreach()
        strat_res = _make_dummy_strategy_result()

        draft = service.generate_draft(
            prospect=prospect,
            product=product,
            strategy_result=strat_res,
            outreach=outreach,
        )

        assert isinstance(draft, VerifiedOutreachDraft)
        assert draft.draft.subject == "Cluster Telemetry Inquiry"
        assert draft.generation_attempts == 1
        assert draft.human_review_status == HumanReviewStatus.PENDING_REVIEW
        assert draft.linter_result is not None
        assert draft.linter_result.has_errors is False
        assert draft.selected_cta == CTAType.VALUE_EXCHANGE

        # Verify database persistence
        with store.uow() as uow:
            db_draft = uow.outreach.get_outreach_draft(draft.draft_id)
            assert db_draft is not None
            assert db_draft["subject"] == "Cluster Telemetry Inquiry"
            assert db_draft["human_review_status"] == "pending_review"


def test_service_rejects_unviable_strategy():
    service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter())
    strat_res = _make_dummy_strategy_result(is_viable=False)

    with pytest.raises(ValueError, match="not viable"):
        service.generate_draft(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            strategy_result=strat_res,
            outreach=_make_dummy_outreach(),
        )


def test_service_rejects_missing_strategy():
    service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter())
    strat_res = _make_dummy_strategy_result(is_viable=True, has_strategy=False)

    with pytest.raises(ValueError, match="contains no OutreachStrategy"):
        service.generate_draft(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            strategy_result=strat_res,
            outreach=_make_dummy_outreach(),
        )


def test_service_rejects_missing_toulmin():
    service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter())
    strat_res = _make_dummy_strategy_result(is_viable=True, has_strategy=True, has_toulmin=False)

    with pytest.raises(ValueError, match="lacks a validated ToulminArgument"):
        service.generate_draft(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            strategy_result=strat_res,
            outreach=_make_dummy_outreach(),
        )


def test_service_rewrite_policy_triggers_on_error_finding():
    # Attempt 1 has missing subject (ERROR)
    # Attempt 2 is valid
    err_draft = RawEmailDraftOutput(
        subject="",
        body="Elena, saw your recent 10,000 node cluster expansion at Black Mesa. Are you open to comparing benchmarks this week?",
    )
    clean_draft = RawEmailDraftOutput(
        subject="Cluster Telemetry at Black Mesa",
        body=(
            "Elena, saw your recent 10,000 node cluster expansion at Black Mesa. "
            "As distributed infrastructure scales, telemetry bottlenecks often create major observability lag across compute clusters. "
            "PulseRadar cuts cluster latency by 50% using kernel-level probes to provide real-time node visibility without overhead. "
            "Are you open to comparing cluster telemetry benchmarks this week?"
        ),
    )

    generator = MockGenerator(outputs=[err_draft, clean_draft])
    service = EmailGenerationService(generator=generator, linter=ReadabilityLinter())

    draft = service.generate_draft(
        prospect=_make_dummy_prospect(),
        product=_make_dummy_product(),
        strategy_result=_make_dummy_strategy_result(),
        outreach=_make_dummy_outreach(),
    )

    assert generator.call_count == 2
    assert generator.last_rewrite_feedback is not None
    assert any("subject" in fb.lower() for fb in generator.last_rewrite_feedback)
    assert draft.generation_attempts == 2
    assert draft.draft.subject == "Cluster Telemetry at Black Mesa"
    assert draft.linter_result.has_errors is False


def test_service_rewrite_policy_does_not_trigger_on_warnings_alone():
    # Draft has corporate jargon warning but no errors
    jargon_body = (
        "Elena, saw your recent cluster rollout. Let's maximize synergies and utilize next-gen paradigms. "
        "PulseRadar cuts cluster latency by 50% using kernel-level probes. "
        "Are you open to comparing telemetry benchmarks this week?"
    )
    warn_draft = RawEmailDraftOutput(
        subject="Cluster Telemetry",
        body=jargon_body,
    )

    generator = MockGenerator(outputs=[warn_draft])
    service = EmailGenerationService(generator=generator, linter=ReadabilityLinter())

    draft = service.generate_draft(
        prospect=_make_dummy_prospect(),
        product=_make_dummy_product(),
        strategy_result=_make_dummy_strategy_result(),
        outreach=_make_dummy_outreach(),
    )

    # Should only call generator once because warnings alone do not trigger rewrite
    assert generator.call_count == 1
    assert draft.generation_attempts == 1
    assert draft.linter_result.has_warnings is True
    assert draft.linter_result.has_errors is False
    assert any("corporate_jargon" in r for r in draft.human_review_reasons)


def test_service_rewrite_failure_preserves_original_draft():
    # Attempt 1 has error, Attempt 2 raises EmailGenerationError
    err_draft = RawEmailDraftOutput(
        subject="",
        body="Elena, saw your recent 10,000 node cluster expansion at Black Mesa. Are you open to comparing benchmarks this week?",
    )

    generator = MockGenerator(outputs=[err_draft, EmailGenerationError("Provider rate limited")])
    service = EmailGenerationService(generator=generator, linter=ReadabilityLinter())

    draft = service.generate_draft(
        prospect=_make_dummy_prospect(),
        product=_make_dummy_product(),
        strategy_result=_make_dummy_strategy_result(),
        outreach=_make_dummy_outreach(),
    )

    # Catches the rewrite failure and preserves attempt 1 draft with lint errors noted
    assert generator.call_count == 2
    assert draft.generation_attempts in (1, 2)
    assert draft.linter_result.has_errors is True
    assert draft.human_review_status == HumanReviewStatus.PENDING_REVIEW


def test_service_persistence_failure_raises_draft_persistence_error():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        # Deliberately NOT initializing store so table does not exist
        service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter(), state_store=store)

        with pytest.raises(DraftPersistenceError):
            service.generate_draft(
                prospect=_make_dummy_prospect(),
                product=_make_dummy_product(),
                strategy_result=_make_dummy_strategy_result(),
                outreach=_make_dummy_outreach(),
            )


# =============================================================================
# ApplicationOrchestrator Stage 7 Integration Tests
# =============================================================================

class MockResearcher:
    def __init__(self, is_sufficient: bool = True):
        self.is_sufficient = is_sufficient

    def conduct_research(self, **kwargs) -> ResearchPackage:
        ev = EvidenceRecord(
            evidence_id="ev-101",
            source_id="src-1",
            verbatim_excerpt="10,000 node deployment completed.",
        )
        fact = CanonicalFact(
            fact_id="fact-101",
            claim_text="10,000 node deployment completed.",
            evidence_records=[ev],
        )
        return ResearchPackage(
            facts=(fact,),
            sources=(),
            is_sufficient=self.is_sufficient,
            is_complete=True,
            missing_information=() if self.is_sufficient else ("Lack of technical details",),
        )


class MockStrategyService:
    def __init__(self, strategy_result: StrategyResult):
        self.strategy_result = strategy_result

    def develop_strategy(self, **kwargs) -> StrategyResult:
        return self.strategy_result


def test_orchestrator_stage_7_completed_with_valid_toulmin():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        researcher = MockResearcher(is_sufficient=True)
        strat_service = MockStrategyService(_make_dummy_strategy_result(is_viable=True, has_toulmin=True))
        email_service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter(), state_store=store)

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
            email_service=email_service,
        )

        res: RunResult = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.COMPLETED
        assert res.draft is not None
        assert res.draft_id == res.draft.draft_id

        # Verify Stage 7 record
        stage7 = next((s for s in res.stage_records if s.stage == StageIdentifier.EMAIL_GENERATION), None)
        assert stage7 is not None
        assert stage7.status == StageStatus.COMPLETED
        assert stage7.details["draft_id"] == res.draft.draft_id

        # Verify DB run record has draft_id
        db_run = store.get_run(res.run_id)
        assert db_run is not None
        assert db_run["draft_id"] == res.draft.draft_id

        snapshot = orchestrator.get_run(res.run_id)
        assert snapshot is not None
        assert snapshot.draft_id == res.draft.draft_id


def test_orchestrator_stage_7_skipped_when_strategy_not_viable():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        researcher = MockResearcher(is_sufficient=True)
        strat_service = MockStrategyService(_make_dummy_strategy_result(is_viable=False))
        email_service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter(), state_store=store)

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
            email_service=email_service,
        )

        res: RunResult = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.NO_VIABLE_SIGNAL
        assert res.draft is None
        assert res.draft_id is None

        stage7 = next((s for s in res.stage_records if s.stage == StageIdentifier.EMAIL_GENERATION), None)
        assert stage7 is not None
        assert stage7.status == StageStatus.SKIPPED


def test_orchestrator_stage_7_skipped_when_toulmin_missing():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        researcher = MockResearcher(is_sufficient=True)
        strat_service = MockStrategyService(_make_dummy_strategy_result(is_viable=True, has_toulmin=False))
        email_service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter(), state_store=store)

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
            email_service=email_service,
        )

        res: RunResult = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        assert res.status == RunStatus.COMPLETED
        assert res.draft is None
        assert res.draft_id is None

        stage7 = next((s for s in res.stage_records if s.stage == StageIdentifier.EMAIL_GENERATION), None)
        assert stage7 is not None
        assert stage7.status == StageStatus.SKIPPED
        assert "lacks validated ToulminArgument" in stage7.details["reason"]


def test_orchestrator_stage_7_skipped_when_research_insufficient():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        researcher = MockResearcher(is_sufficient=False)
        strat_service = MockStrategyService(_make_dummy_strategy_result(is_viable=True, has_toulmin=True))
        email_service = EmailGenerationService(generator=MockGenerator(), linter=ReadabilityLinter(), state_store=store)

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
            email_service=email_service,
        )

        res: RunResult = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
            allow_insufficient_research=False,
        )

        assert res.status == RunStatus.INSUFFICIENT_RESEARCH
        assert res.draft is None
        assert res.draft_id is None

        stage7 = next((s for s in res.stage_records if s.stage == StageIdentifier.EMAIL_GENERATION), None)
        assert stage7 is not None
        assert stage7.status == StageStatus.SKIPPED


def test_orchestrator_preserves_research_and_strategy_on_generator_failure():
    """Verify that when email generator raises an error, research and strategy are preserved."""
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        researcher = MockResearcher(is_sufficient=True)
        strat_service = MockStrategyService(_make_dummy_strategy_result(is_viable=True, has_toulmin=True))
        # Generator configured to raise EmailGenerationError
        failing_generator = MockGenerator(outputs=[EmailGenerationError("Provider rate limit timeout")])
        email_service = EmailGenerationService(
            generator=failing_generator,
            linter=ReadabilityLinter(),
            state_store=store,
        )

        orchestrator = ApplicationOrchestrator(
            state_store=store,
            researcher=researcher,
            strategy_service=strat_service,
            email_service=email_service,
        )

        res: RunResult = orchestrator.run(
            prospect=_make_dummy_prospect(),
            product=_make_dummy_product(),
            outreach=_make_dummy_outreach(),
        )

        # Run fails due to stage 7 failure
        assert res.status == RunStatus.FAILED
        assert res.draft is None
        assert res.draft_id is None

        # Stage 7 recorded as FAILED
        stage7 = next((s for s in res.stage_records if s.stage == StageIdentifier.EMAIL_GENERATION), None)
        assert stage7 is not None
        assert stage7.status == StageStatus.FAILED
        assert "Provider rate limit timeout" in str(stage7.error_message)

        # Invariant: Research package and strategy result are strictly preserved
        assert res.research_package is not None
        assert len(res.research_package.facts) > 0
        assert res.strategy_result is not None
        assert res.strategy_result.is_viable is True
        assert res.strategy_result.strategy is not None

        # DB run record also preserves strategy_result_json
        db_run = store.get_run(res.run_id)
        assert db_run is not None
        assert db_run["status"] == RunStatus.FAILED.value
        assert db_run["strategy_result_json"] is not None
