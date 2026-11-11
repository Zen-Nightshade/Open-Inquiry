from __future__ import annotations

import datetime
from datetime import timezone
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CapabilityNode,
    CompatibilityStatus,
    ProductContract,
    ProductContext,
    ProspectContext,
    OutreachContext,
)
from src.db.state_store import StateStore
from src.email_generation.claim_verifier.capability import ProductCapabilityBoundsGate
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    ModalityLevel,
    VerificationFinding,
    VerifierFindingSeverity,
    VerifierResult,
)
from src.email_generation.claim_verifier.service import ClaimVerifierService
from src.email_generation.generator import EmailGenerator
from src.email_generation.models import (
    EmailDraft,
    HumanReviewStatus,
    RawEmailDraftOutput,
    VerifiedOutreachDraft,
)
from src.email_generation.review_models import (
    InvalidReviewTransitionError,
    RejectionReason,
)
from src.email_generation.review_service import ReviewService
from src.email_generation.service import DraftPersistenceError, EmailGenerationService
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.orchestration import (
    ApplicationOrchestrator,
    RunResult,
    RunStatus,
    StageIdentifier,
    StageStatus,
)
from src.researcher.entity.service import EntityResolutionService
from src.researcher.epistemic_dag.models import (
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    QuestionLifecycleStatus,
    ResolutionCriterion,
    StoppingReason,
)
from src.researcher.epistemic_dag.stopping import (
    StoppingEvaluator,
    fact_matches_predicate,
    is_fact_attributable_to_node,
    normalize_predicate_token,
)
from src.researcher.entity.probabilistic_models import ResolutionOutcome
from src.researcher.entity.probabilistic_resolver import ProbabilisticResolutionResult
from src.researcher.entity.resolver import IdentityResolutionResult
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    MolecularFact,
    ResearchPackage,
    ResolutionStatus,
    SourceRecord,
    SourceType,
    VerificationStatus,
)
from src.signal_strategy.identifier import SignalIdentifier
from src.signal_strategy.models import (
    CandidateSignal,
    CTAType,
    EligibilityStatus,
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
from src.signal_strategy.service import SignalStrategyService
from src.signal_strategy.strategy_builder import StrategyBuilder


# =============================================================================
# Fixtures and Helpers
# =============================================================================

@pytest.fixture
def test_state_store():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_phase1.db"
        store = StateStore(db_path=db_path)
        store.initialize()
        yield store


def _make_dummy_prospect(company_name: str = "Apex Logistics") -> ProspectContext:
    return ProspectContext(
        full_name="Sarah Connor",
        company_name=company_name,
        company_domain="apexlogistics.example",
        designation="VP Supply Chain",
    )


def _make_dummy_product(
    capabilities: Optional[List[CapabilityNode]] = None,
    is_exhaustive: bool = False,
) -> ProductContext:
    caps = capabilities or [
        CapabilityNode(
            capability_id="cap-routing",
            name="Automated Fleet Routing",
            target_operational_problems=["fuel waste", "dispatch delays"],
            supported_use_cases=["Fleet logistics", "Multi-stop routing"],
            hard_exclusions=["Air freight dispatch"],
            verified_proof_points=["50% reduction in dispatch overhead"],
        )
    ]
    contract = ProductContract(
        product_name="FleetPulse",
        contract_version="1.0.0",
        description="Dynamic routing and fleet optimization engine.",
        value_proposition="Reduces fleet dispatch overhead by 50%.",
        capabilities=caps,
        hard_exclusions=["Air freight dispatch"],
        is_exhaustive=is_exhaustive,
    )
    return contract.to_context()


def _make_dummy_outreach() -> OutreachContext:
    return OutreachContext(
        objective="Schedule introductory evaluation",
        sender_identity="Dan Vance, Enterprise Solutions",
        sender_name="Dan Vance",
        sender_company="LogiTech Inc.",
    )


def _make_dummy_package(
    prospect_status: ResolutionStatus = ResolutionStatus.RESOLVED,
    company_status: ResolutionStatus = ResolutionStatus.RESOLVED,
    facts: Optional[List[CanonicalFact]] = None,
    is_sufficient: bool = True,
) -> ResearchPackage:
    ev = EvidenceRecord(
        evidence_id="ev-1",
        source_id="src-1",
        verbatim_excerpt="Apex Logistics announced addition of 50 new distribution hubs.",
        confidence=ConfidenceLevel.HIGH,
    )
    default_facts = facts if facts is not None else [
        CanonicalFact(
            fact_id="fact-1",
            claim_text="Apex Logistics expanded fleet with 50 new regional hubs.",
            fact_category=FactCategory.PRODUCT_LAUNCH,
            confidence=ConfidenceLevel.HIGH,
            evidence_records=[ev],
            primary_event_date=datetime.date(2026, 10, 15),
        )
    ]
    prospect_ent = EntityRecord(
        entity_id="ent-person-1",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        canonical_name="Sarah Connor",
        resolution_status=prospect_status,
        confidence=ConfidenceLevel.HIGH if prospect_status == ResolutionStatus.RESOLVED else ConfidenceLevel.LOW,
    )
    company_ent = EntityRecord(
        entity_id="ent-comp-1",
        entity_type=EntityType.ORGANIZATION,
        raw_name="Apex Logistics",
        canonical_name="Apex Logistics Inc.",
        resolution_status=company_status,
        confidence=ConfidenceLevel.HIGH if company_status == ResolutionStatus.RESOLVED else ConfidenceLevel.LOW,
    )
    return ResearchPackage(
        prospect_entity=prospect_ent,
        company_entity=company_ent,
        evidence=[ev],
        facts=default_facts,
        is_sufficient=is_sufficient,
        is_complete=True,
    )


def _make_dummy_strategy_result(is_viable: bool = True) -> StrategyResult:
    if not is_viable:
        return StrategyResult(
            is_viable=False,
            rejection_reason="No qualifying commercial signals identified.",
            unviable_reasons=["Candidate signals list empty."],
        )

    cand = CandidateSignal(
        signal_id="sig-fleet-1",
        label="Fleet Expansion",
        description="Apex Logistics expanded fleet with 50 new hubs.",
        supporting_fact_ids=["fact-1"],
        fact_category=FactCategory.PRODUCT_LAUNCH,
        observed_event="Fleet expansion announcement",
        target_capability="cap-routing",
    )
    score = SignalScore(
        evidence_strength=0.9,
        recency_score=0.9,
        prospect_relevance=0.85,
        product_alignment=0.9,
        business_significance=0.85,
        composite_score=0.88,
    )
    ranked = RankedSignal(
        signal=cand,
        score=score,
        rationale="Strong alignment with fleet routing capability.",
    )
    warrant = ToulminWarrant(
        capability_id="cap-routing",
        capability_name="Automated Fleet Routing",
        reasoning="Connects fleet hub expansion functionally to automated routing.",
        is_inferred=True,
        limitations=["Air freight dispatch"],
    )
    grounds = ToulminGrounds(
        fact_ids=["fact-1"],
        evidence_ids=["ev-1"],
        observed_event="Fleet expansion announcement",
        source_domains=["apexlogistics.example"],
    )
    backing = ToulminBacking(
        capability_id="cap-routing",
        whitelisted_proof_points=["50% reduction in dispatch overhead"],
        has_whitelisted_proof=True,
        provenance=OutputProvenance(
            origin=ProvenanceOrigin.DETERMINISTIC_RULE,
            component="product_contract",
        ),
    )
    qualifier = ToulminQualifier(
        eligibility_status=EligibilityStatus.ELIGIBLE,
        compatibility_status=CompatibilityStatus.VERIFIED_COMPATIBLE,
        verification_status=VerificationStatus.SUPPORTED,
        composite_confidence=ConfidenceLevel.HIGH,
        unresolved_prerequisites=[],
        qualification_language="Verified capability alignment with documented proof points.",
    )
    claim = ToulminClaim(
        claim_type="value_assertion",
        text="Automate fleet routing across new distribution hubs.",
        target_capability="cap-routing",
        is_qualified=True,
    )
    rebuttal = ToulminRebuttal(hard_exclusions=["Air freight dispatch"])
    toulmin = ToulminArgument(
        grounds=grounds,
        warrant=warrant,
        backing=backing,
        qualifier=qualifier,
        claim=claim,
        rebuttal=rebuttal,
    )
    alignment = ProspectProductAlignment(
        prospect_role_relevance="Direct relevance to fleet logistics oversight.",
        business_implications="Increases fleet efficiency across expanding hubs.",
        capability_mapping=["Automated Fleet Routing"],
    )
    strategy = OutreachStrategy(
        strategy_id="strat-1",
        primary_signal=ranked,
        supporting_signals=[],
        alignment=alignment,
        messaging_approach="Expansion Alignment & Fleet Routing",
        central_message="Scale dispatch routing across expanding hubs.",
        suggested_subject_line="Fleet dispatch routing across new hubs",
        cta="Open to a brief conversation next week?",
        tone_guidance="Direct, executive, data-driven",
        toulmin_argument=toulmin,
    )
    return StrategyResult(
        is_viable=True,
        strategy=strategy,
        ranked_signals=[ranked],
    )


# =============================================================================
# Workstream A: Verification Hard Gating & Review Quarantine (FL-04)
# =============================================================================

def test_draft_verification_failure_quarantines_draft(test_state_store: StateStore):
    """When claim verifier returns error findings, draft is quarantined immediately."""
    prospect = _make_dummy_prospect()
    product = _make_dummy_product()
    strat_res = _make_dummy_strategy_result(is_viable=True)
    outreach = _make_dummy_outreach()
    research_pkg = _make_dummy_package()

    # Create prerequisite database context records
    context_id = "ctx-quarantine-test"
    with test_state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id="p-test-1",
            full_name=prospect.full_name,
            company_name=prospect.company_name,
        )
        uow.products.create_product_version(
            version_id="pv-test-1",
            product_name=product.product_name,
            snapshot_json="{}",
        )
        uow.outreach.create_outreach_context(
            context_id=context_id,
            prospect_id="p-test-1",
            version_id="pv-test-1",
            context_json="{}",
        )

    # Mock ClaimVerifierService returning ERROR finding
    class FailingVerifierService:
        def verify(self, **kwargs) -> VerifierResult:
            finding = VerificationFinding(
                gate="product_capability_bounds",
                claim_id="c-fail",
                code="UNSUPPORTED_PROOF_POINT",
                message="Fabricated 99% cost reduction metric",
                severity=VerifierFindingSeverity.ERROR,
            )
            return VerifierResult(
                is_valid=False,
                passed=False,
                has_errors=True,
                has_warnings=False,
                findings=[finding],
                unverified_claim_ids=["c-fail"],
            )

    class MockGeneratorEngine(GeneratorEngine):
        def __init__(self):
            super().__init__(default_provider="mock")

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raw_output = RawEmailDraftOutput(
                subject="FleetPulse Optimization",
                body="FleetPulse guarantees a 99% reduction in total operational cost.",
            )
            raw_text = raw_output.model_dump_json()
            return GenerationResponse(
                text=raw_text,
                provider_name="mock",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=raw_output,
            )

    email_service = EmailGenerationService(
        state_store=test_state_store,
        generator=EmailGenerator(engine=MockGeneratorEngine()),
        verifier=FailingVerifierService(),
    )

    draft = email_service.generate_draft(
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        outreach=outreach,
        research_package=research_pkg,
        context_id=context_id,
        research_session_id="session-quarantine-1",
    )

    assert draft.human_review_status == HumanReviewStatus.QUARANTINED_VERIFICATION_FAILED

    # Check persistence in database
    with test_state_store.uow() as uow:
        persisted = uow.outreach.get_outreach_draft(draft.draft_id)
        assert persisted is not None
        assert persisted["human_review_status"] == "quarantined_verification_failed"


def test_run_status_on_verifier_error(test_state_store: StateStore):
    """When draft verification fails, orchestrator marks run and email stage as FAILED."""
    prospect = _make_dummy_prospect()
    product = _make_dummy_product()
    outreach = _make_dummy_outreach()
    research_pkg = _make_dummy_package()
    strat_res = _make_dummy_strategy_result(is_viable=True)

    class MockResearcher:
        def conduct_research(self, **kwargs):
            return research_pkg

    class MockStrategyService:
        def develop_strategy(self, **kwargs):
            return strat_res

    class MockFailingEmailService:
        def generate_draft(self, **kwargs):
            finding = VerificationFinding(
                gate="product_capability_bounds",
                claim_id="c-fail",
                code="UNSUPPORTED_PROOF_POINT",
                message="Unverified metric claim",
                severity=VerifierFindingSeverity.ERROR,
            )
            verifier_result = VerifierResult(
                is_valid=False,
                passed=False,
                has_errors=True,
                has_warnings=False,
                findings=[finding],
                unverified_claim_ids=["c-fail"],
            )
            draft_inner = EmailDraft(
                subject="Test Subject",
                body="Test body with 99% fabricated metric.",
                word_count=7,
            )
            context_id = kwargs.get("context_id")
            with test_state_store.uow() as uow:
                uow.outreach.create_outreach_draft(
                    draft_id="draft-failing-1",
                    context_id=context_id,
                    research_session_id=kwargs.get("research_session_id"),
                    subject="Test Subject",
                    body="Test body with 99% fabricated metric.",
                    word_count=7,
                    product_snapshot_json="{}",
                    claim_verifications="[]",
                    quality_evaluation="{}",
                    human_review_status="quarantined_verification_failed",
                    current_revision=1,
                )
            return VerifiedOutreachDraft(
                draft_id="draft-failing-1",
                draft=draft_inner,
                outreach_strategy=strat_res.strategy,
                human_review_status=HumanReviewStatus.QUARANTINED_VERIFICATION_FAILED,
                verifier_result=verifier_result,
                claim_verifications=[],
            )

    orchestrator = ApplicationOrchestrator(
        state_store=test_state_store,
        researcher=MockResearcher(),
        strategy_service=MockStrategyService(),
        email_service=MockFailingEmailService(),
    )

    res: RunResult = orchestrator.run(
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert res.status == RunStatus.FAILED
    assert any("quarantined" in w.lower() for w in res.warnings)

    email_stages = [s for s in res.stage_records if s.stage == StageIdentifier.EMAIL_GENERATION]
    assert len(email_stages) == 1
    assert email_stages[0].status == StageStatus.FAILED
    assert "quarantined" in email_stages[0].error_message.lower()


def test_review_operations_blocked_on_quarantined_draft(test_state_store: StateStore):
    """Review operations (edit, approve, reject) must raise InvalidReviewTransitionError on quarantined drafts."""
    service = ReviewService(test_state_store)
    draft_id = "draft-quarantined-ops"
    context_id = f"ctx-{draft_id}"

    with test_state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id=f"p-{draft_id}",
            full_name="Quarantined User",
            company_name="Apex Corp",
        )
        uow.products.create_product_version(
            version_id=f"pv-{draft_id}",
            product_name="PulseRadar",
            snapshot_json="{}",
        )
        uow.outreach.create_outreach_context(
            context_id=context_id,
            prospect_id=f"p-{draft_id}",
            version_id=f"pv-{draft_id}",
            context_json="{}",
        )
        uow.outreach.create_outreach_draft(
            draft_id=draft_id,
            context_id=context_id,
            research_session_id=f"sess-{draft_id}",
            subject="Quarantined Subject",
            body="Quarantined body with unverified claims.",
            word_count=5,
            product_snapshot_json="{}",
            claim_verifications="[]",
            quality_evaluation="{}",
            human_review_status="quarantined_verification_failed",
            current_revision=1,
        )

    # 1. Edit draft must fail
    with pytest.raises(InvalidReviewTransitionError) as exc_edit:
        service.edit_draft(
            draft_id=draft_id,
            expected_revision=1,
            reviewer_id="reviewer-1",
            new_subject="Attempted Edit",
            new_body="Attempted edit body.",
        )
    assert "quarantined_verification_failed" in str(exc_edit.value).lower()

    # 2. Approve draft must fail
    with pytest.raises(InvalidReviewTransitionError) as exc_app:
        service.approve_draft(
            draft_id=draft_id,
            expected_revision=1,
            reviewer_id="reviewer-1",
        )
    assert "quarantined_verification_failed" in str(exc_app.value).lower()

    # 3. Reject draft must fail
    with pytest.raises(InvalidReviewTransitionError) as exc_rej:
        service.reject_draft(
            draft_id=draft_id,
            expected_revision=1,
            reviewer_id="reviewer-1",
            reason_code=RejectionReason.OTHER,
        )
    assert "quarantined_verification_failed" in str(exc_rej.value).lower()

    # Verify status in database remains quarantined_verification_failed
    with test_state_store.uow() as uow:
        persisted = uow.outreach.get_outreach_draft(draft_id)
        assert persisted["human_review_status"] == "quarantined_verification_failed"


# =============================================================================
# Workstream B: Ambiguous & Unresolved Entity Resolution Circuit-Breaking (FL-10, REL-01)
# =============================================================================

def test_company_entity_ambiguity_halts_pipeline(test_state_store: StateStore):
    """When company entity resolution is AMBIGUOUS, pipeline halts with INSUFFICIENT_RESEARCH."""
    package = _make_dummy_package(
        prospect_status=ResolutionStatus.RESOLVED,
        company_status=ResolutionStatus.AMBIGUOUS,
    )

    strategy_called = False

    class MockResearcher:
        def conduct_research(self, **kwargs):
            return package

    class MockStrategyService:
        def develop_strategy(self, **kwargs):
            nonlocal strategy_called
            strategy_called = True
            return _make_dummy_strategy_result(is_viable=True)

    orchestrator = ApplicationOrchestrator(
        state_store=test_state_store,
        researcher=MockResearcher(),
        strategy_service=MockStrategyService(),
    )

    res = orchestrator.run(
        prospect=_make_dummy_prospect(),
        product=_make_dummy_product(),
        outreach=_make_dummy_outreach(),
        allow_insufficient_research=False,
    )

    assert res.status == RunStatus.INSUFFICIENT_RESEARCH
    assert not strategy_called
    assert any("company identity resolution failed with status: ambiguous" in w.lower() for w in res.warnings)


def test_company_entity_unresolved_halts_pipeline(test_state_store: StateStore):
    """When company entity resolution is UNRESOLVED, pipeline halts with INSUFFICIENT_RESEARCH."""
    package = _make_dummy_package(
        prospect_status=ResolutionStatus.RESOLVED,
        company_status=ResolutionStatus.UNRESOLVED,
    )

    strategy_called = False

    class MockResearcher:
        def conduct_research(self, **kwargs):
            return package

    class MockStrategyService:
        def develop_strategy(self, **kwargs):
            nonlocal strategy_called
            strategy_called = True
            return _make_dummy_strategy_result(is_viable=True)

    orchestrator = ApplicationOrchestrator(
        state_store=test_state_store,
        researcher=MockResearcher(),
        strategy_service=MockStrategyService(),
    )

    res = orchestrator.run(
        prospect=_make_dummy_prospect(),
        product=_make_dummy_product(),
        outreach=_make_dummy_outreach(),
        allow_insufficient_research=False,
    )

    assert res.status == RunStatus.INSUFFICIENT_RESEARCH
    assert not strategy_called
    assert any("company identity resolution failed with status: unresolved" in w.lower() for w in res.warnings)


def test_signal_strategy_rejects_ambiguous_company():
    """SignalStrategyService.develop_strategy rejects when company entity is AMBIGUOUS."""
    prospect = _make_dummy_prospect()
    product = _make_dummy_product()
    outreach = _make_dummy_outreach()
    pkg = _make_dummy_package(
        prospect_status=ResolutionStatus.RESOLVED,
        company_status=ResolutionStatus.AMBIGUOUS,
    )

    service = SignalStrategyService()
    res = service.develop_strategy(
        research_package=pkg,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert res.is_viable is False
    assert "ambiguous" in res.rejection_reason.lower()
    assert "company" in res.rejection_reason.lower()


def test_signal_strategy_rejects_unresolved_prospect():
    """SignalStrategyService.develop_strategy rejects when prospect entity is UNRESOLVED."""
    prospect = _make_dummy_prospect()
    product = _make_dummy_product()
    outreach = _make_dummy_outreach()
    pkg = _make_dummy_package(
        prospect_status=ResolutionStatus.UNRESOLVED,
        company_status=ResolutionStatus.RESOLVED,
    )

    service = SignalStrategyService()
    res = service.develop_strategy(
        research_package=pkg,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert res.is_viable is False
    assert "unresolved" in res.rejection_reason.lower()
    assert "prospect" in res.rejection_reason.lower()


def test_entity_service_reflects_company_ambiguity(test_state_store: StateStore):
    """EntityResolutionService reflects company entity ambiguity in overall result status."""
    prospect = _make_dummy_prospect()
    session_id = "session-ambig-comp-test"
    with test_state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id="p-ambig-test",
            full_name=prospect.full_name,
            company_name=prospect.company_name,
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id="p-ambig-test",
            context_id="ctx-ambig-test",
            status="in_progress",
        )

    source1 = SourceRecord(
        source_id="src-ambig-1",
        url="https://apexlogistics.example",
        canonical_url="https://apexlogistics.example",
        domain="apexlogistics.example",
        raw_text="Apex Logistics overview and operations.",
        source_type=SourceType.COMPANY_WEBSITE,
        fetch_date=datetime.date(2026, 11, 1),
    )

    class MockProbResolver:
        def resolve_candidates(self, candidates, target_prospect, entity_type=EntityType.PERSON):
            if entity_type == EntityType.ORGANIZATION:
                return ProbabilisticResolutionResult(
                    entity_type=EntityType.ORGANIZATION,
                    outcome=ResolutionOutcome.AMBIGUOUS,
                    resolution_status=ResolutionStatus.AMBIGUOUS,
                    confidence=ConfidenceLevel.LOW,
                    rationale="Multiple competing company candidate entities detected.",
                    competing_candidate_ids=["cand-c1", "cand-c2"],
                )
            # Person matches
            return ProbabilisticResolutionResult(
                entity_type=EntityType.PERSON,
                outcome=ResolutionOutcome.MATCH,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                selected_candidate_id="cand-p1",
                rationale="Exact name match.",
            )

    class MockIdentityResolver:
        def resolve(self, prospect, candidates, sources, session_id=None):
            p_entity = EntityRecord(
                entity_id="ent-p1",
                entity_type=EntityType.PERSON,
                raw_name=prospect.full_name,
                canonical_name=prospect.full_name,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
            )
            c_entity = EntityRecord(
                entity_id="ent-c1",
                entity_type=EntityType.ORGANIZATION,
                raw_name=prospect.company_name,
                canonical_name=prospect.company_name,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
            )
            return IdentityResolutionResult(
                target_prospect=prospect,
                resolution_status=ResolutionStatus.RESOLVED,
                confidence=ConfidenceLevel.HIGH,
                matched_candidate_ids=["cand-p1", "cand-c1"],
                prospect_entity=p_entity,
                company_entity=c_entity,
                rationale="Initial resolver output.",
            )

    service = EntityResolutionService(
        state_store=test_state_store,
        resolver=MockIdentityResolver(),
        probabilistic_resolver=MockProbResolver(),
    )

    result = service.resolve_entities_for_session(
        session_id=session_id,
        prospect=prospect,
        sources=[source1],
    )

    # Company ambiguity must not be masked by prospect match
    assert result.company_entity.resolution_status == ResolutionStatus.AMBIGUOUS
    assert result.resolution_status == ResolutionStatus.AMBIGUOUS


# =============================================================================
# Workstream C: Canonical Fact Attribution & Predicate Relevance (FL-03)
# =============================================================================

def test_fact_matches_predicate_semantics():
    """Normalize predicates and enforce predicate compatibility."""
    assert fact_matches_predicate("headcount", {"predicate": "employee_count"}) is True
    assert fact_matches_predicate("team_size", {"predicate": "headcount"}) is True
    assert fact_matches_predicate("headcount", {"predicate": "funding_raised"}) is False
    assert fact_matches_predicate("leadership_appointed", {"predicate": "executive_hiring"}) is True
    assert fact_matches_predicate("headquartered_in", {"predicate": "facility_opened"}) is False


def test_is_fact_attributable_to_node_criteria():
    """is_fact_attributable_to_node verifies predicate match, target entity, and lack of ambiguity."""
    node = EpistemicQuestionNode(
        question_id="q-headcount",
        run_id="run-1",
        requested_predicate="headcount",
        neutral_question="What is the headcount?",
        observability=ObservabilityClass.PUBLICLY_OBSERVABLE,
        target_entity_id="ent-comp-1",
    )

    ev = EvidenceRecord(
        evidence_id="ev-att-1",
        source_id="s1",
        verbatim_excerpt="Apex Logistics has 500 employees.",
        confidence=ConfidenceLevel.HIGH,
    )

    # 1. Matching predicate and entity
    fact_valid = CanonicalFact(
        fact_id="f-valid",
        claim_text="Apex Logistics has 500 employees.",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        predicate="employee_count",
        entity_ids=["ent-comp-1"],
        evidence_records=[ev],
    )
    assert is_fact_attributable_to_node(fact_valid, node) is True

    # 2. Mismatched predicate
    fact_wrong_pred = CanonicalFact(
        fact_id="f-wrong-pred",
        claim_text="Apex Logistics raised $10M Series A.",
        fact_category=FactCategory.FUNDING,
        predicate="funding_raised",
        entity_ids=["ent-comp-1"],
        evidence_records=[ev],
    )
    assert is_fact_attributable_to_node(fact_wrong_pred, node) is False

    # 3. Ambiguous entity resolution fact
    fact_ambig = CanonicalFact(
        fact_id="f-ambig",
        claim_text="Company has 500 employees.",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        predicate="employee_count",
        entity_ids=["ent-comp-1"],
        entity_resolution_status="ambiguous",
        evidence_records=[ev],
    )
    assert is_fact_attributable_to_node(fact_ambig, node) is False

    # 4. Target entity mismatch
    fact_wrong_entity = CanonicalFact(
        fact_id="f-wrong-ent",
        claim_text="Other Corp has 500 employees.",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        predicate="employee_count",
        entity_ids=["ent-comp-other"],
        evidence_records=[ev],
    )
    assert is_fact_attributable_to_node(fact_wrong_entity, node) is False


def test_stopping_evaluator_predicate_mismatch_prevents_resolution():
    """StoppingEvaluator does not mark node RESOLVED when facts do not match requested predicate."""
    node = EpistemicQuestionNode(
        question_id="q-headcount",
        run_id="run-1",
        requested_predicate="headcount",
        neutral_question="What is the headcount?",
        observability=ObservabilityClass.PUBLICLY_OBSERVABLE,
        estimated_research_cost=1,
        required_resolution_criteria=[
            ResolutionCriterion(
                criterion_type="min_sources",
                description="Requires >= 1 corroborated sources",
                is_mandatory=True,
                min_sources=1,
            )
        ],
    )

    ev_fund = EvidenceRecord(
        evidence_id="ev-funding-1",
        source_id="s-1",
        verbatim_excerpt="Apex raised $15M Series B funding.",
        confidence=ConfidenceLevel.HIGH,
    )
    # Only funding facts returned
    funding_facts = [
        CanonicalFact(
            fact_id="f-funding",
            claim_text="Apex raised $15M Series B funding.",
            fact_category=FactCategory.FUNDING,
            predicate="funding_raised",
            evidence_records=[ev_fund],
        )
    ]

    decision = StoppingEvaluator.evaluate_post_search(
        node=node,
        facts_found=funding_facts,
        sources_found=[{"source_id": "s-1"}],
        executed_queries=["Apex headcount query 1"],
    )

    # Must NOT be resolved; since executed_queries == estimated_cost, it exhausts scope
    assert decision.new_status != QuestionLifecycleStatus.RESOLVED
    assert decision.reason == StoppingReason.EXHAUSTED_SCOPE
    assert decision.new_status == QuestionLifecycleStatus.EXHAUSTED


def test_heterogeneous_batch_outcome_recording():
    """In a heterogeneous batch, Node 1 resolves with matching facts, while Node 2 receives zero facts and exhausts scope."""
    node1 = EpistemicQuestionNode(
        question_id="q1-headcount",
        run_id="run-1",
        requested_predicate="headcount",
        neutral_question="What is the headcount?",
        observability=ObservabilityClass.PUBLICLY_OBSERVABLE,
        estimated_research_cost=1,
        required_resolution_criteria=[
            ResolutionCriterion(
                criterion_type="min_sources",
                description="Requires >= 1 corroborated sources",
                is_mandatory=True,
                min_sources=1,
            )
        ],
    )
    node2 = EpistemicQuestionNode(
        question_id="q2-techstack",
        run_id="run-1",
        requested_predicate="technology_stack",
        neutral_question="What is their primary tech stack?",
        observability=ObservabilityClass.PUBLICLY_OBSERVABLE,
        estimated_research_cost=1,
        required_resolution_criteria=[
            ResolutionCriterion(
                criterion_type="min_sources",
                description="Requires >= 1 corroborated sources",
                is_mandatory=True,
                min_sources=1,
            )
        ],
    )

    ev_hc = EvidenceRecord(
        evidence_id="ev-hc-1",
        source_id="s-1",
        verbatim_excerpt="Apex Logistics employs 350 full-time staff.",
        confidence=ConfidenceLevel.HIGH,
    )
    # Batch produces only headcount facts
    headcount_fact = CanonicalFact(
        fact_id="f-hc",
        claim_text="Apex Logistics employs 350 full-time staff.",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        predicate="employee_count",
        evidence_records=[ev_hc],
    )
    batch_facts = [headcount_fact]

    # Evaluate Node 1 with partitioned attributable facts
    n1_facts = [f for f in batch_facts if is_fact_attributable_to_node(f, node1)]
    assert len(n1_facts) == 1
    d1 = StoppingEvaluator.evaluate_post_search(
        node=node1,
        facts_found=n1_facts,
        sources_found=[{"source_id": "s-1"}],
        executed_queries=["query 1"],
    )
    assert d1.reason == StoppingReason.RESOLVED
    assert d1.new_status == QuestionLifecycleStatus.RESOLVED

    # Evaluate Node 2 with partitioned attributable facts
    n2_facts = [f for f in batch_facts if is_fact_attributable_to_node(f, node2)]
    assert len(n2_facts) == 0
    d2 = StoppingEvaluator.evaluate_post_search(
        node=node2,
        facts_found=n2_facts,
        sources_found=[{"source_id": "s-1"}],
        executed_queries=["query 2"],
    )
    assert d2.new_status != QuestionLifecycleStatus.RESOLVED
    assert d2.reason == StoppingReason.EXHAUSTED_SCOPE


# =============================================================================
# Workstream D: Elimination of Synthetic Signals & Warrant Grounding (FL-11, REL-05)
# =============================================================================

def test_signal_identifier_zero_synthetic_fallbacks_on_failure():
    """SignalIdentifier returns [] on LLM failure; _deterministic_fallback_signals is eliminated."""
    prospect = _make_dummy_prospect()
    product = _make_dummy_product()

    ev = EvidenceRecord(
        evidence_id="ev-real-1",
        source_id="s1",
        verbatim_excerpt="Apex announced new distribution hub in Dallas",
        confidence=ConfidenceLevel.HIGH,
    )
    fact = CanonicalFact(
        fact_id="fact-real-1",
        claim_text="Apex announced new distribution hub in Dallas",
        fact_category=FactCategory.PRODUCT_LAUNCH,
        primary_event_date=datetime.date(2026, 11, 1),
        evidence_records=[ev],
    )

    class FailingEngine(GeneratorEngine):
        def __init__(self):
            super().__init__(default_provider="mock")

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise RuntimeError("LLM connection timeout")

    identifier = SignalIdentifier(engine=FailingEngine())
    assert not hasattr(identifier, "_deterministic_fallback_signals")

    signals = identifier.identify_signals(facts=[fact], prospect=prospect, product=product)
    assert len(signals) == 0


def test_signal_strategy_rejects_empty_signals():
    """SignalStrategyService returns is_viable=False when zero signals are identified."""
    prospect = _make_dummy_prospect()
    product = _make_dummy_product()
    outreach = _make_dummy_outreach()
    pkg = _make_dummy_package()

    class EmptySignalIdentifier:
        def identify_signals(self, **kwargs):
            return []

    service = SignalStrategyService(identifier=EmptySignalIdentifier())
    res = service.develop_strategy(
        research_package=pkg,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert res.is_viable is False
    assert "No qualifying commercial signals identified" in res.rejection_reason


def test_toulmin_warrant_reasoning_bounds_empirical_metrics():
    """Toulmin warrant reasoning sanitizes ungrounded empirical metrics to qualitative phrasing."""
    builder = StrategyBuilder()
    product = _make_dummy_product()
    prospect = _make_dummy_prospect()
    pkg = _make_dummy_package()
    top_signal = _make_dummy_strategy_result().ranked_signals[0]

    # Reasoning with ungrounded metrics not in proof points ("85% savings" and "$5M")
    raw_reasoning = "Adopting routing yields an 85% savings and $5M in operational cost."

    argument, err = builder._bind_and_validate_toulmin_argument(
        primary_signal=top_signal,
        product=product,
        prospect=prospect,
        research_package=pkg,
        raw_warrant_reasoning=raw_reasoning,
        candidate_capabilities=["cap-routing"],
    )

    assert argument is not None
    # 85% and $5M must NOT appear in warrant.reasoning; replaced with "significantly"
    assert "85%" not in argument.warrant.reasoning
    assert "$5M" not in argument.warrant.reasoning
    assert "significantly" in argument.warrant.reasoning


def test_unsupported_metric_in_claim_triggers_verifier_error_and_quarantine(test_state_store: StateStore):
    """Claim with ungrounded empirical metric produces an ERROR finding and quarantines draft."""
    contract = ProductContract(
        product_name="FleetPulse",
        contract_version="1.0.0",
        description="Routing engine",
        value_proposition="Reduces dispatch overhead by 50%",
        capabilities=[
            CapabilityNode(
                capability_id="cap-routing",
                name="Automated Fleet Routing",
                target_operational_problems=["fuel waste"],
                supported_use_cases=["Fleet logistics"],
                hard_exclusions=["Air freight dispatch"],
                verified_proof_points=["50% reduction in dispatch overhead"],
            )
        ],
    )
    product_ctx = contract.to_context()
    prospect = _make_dummy_prospect()
    outreach = _make_dummy_outreach()
    research_pkg = _make_dummy_package()
    strat_res = _make_dummy_strategy_result(is_viable=True)

    # 1. Direct CapabilityBoundsGate verification
    gate = ProductCapabilityBoundsGate()
    claim_with_fabricated_metric = DecomposedClaim(
        claim_id="c-unsupported-metric",
        text="FleetPulse guarantees a 95% reduction in total operational cost.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=63,
        category=ClaimCategory.METRIC_IMPACT,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim_with_fabricated_metric], contract, research_package=research_pkg)
    assert any(f.code == "UNSUPPORTED_PROOF_POINT" and f.severity == VerifierFindingSeverity.ERROR for f in findings)

    # 2. End-to-end through EmailGenerationService
    context_id = "ctx-unsupported-metric-test"
    with test_state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id="p-metric-1",
            full_name=prospect.full_name,
            company_name=prospect.company_name,
        )
        uow.products.create_product_version(
            version_id="pv-metric-1",
            product_name=product_ctx.product_name,
            snapshot_json="{}",
        )
        uow.outreach.create_outreach_context(
            context_id=context_id,
            prospect_id="p-metric-1",
            version_id="pv-metric-1",
            context_json="{}",
        )

    class MockGeneratorEngine(GeneratorEngine):
        def __init__(self):
            super().__init__(default_provider="mock")

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raw_output = RawEmailDraftOutput(
                subject="FleetPulse Optimization",
                body="FleetPulse guarantees a 95% reduction in total operational cost.",
            )
            raw_text = raw_output.model_dump_json()
            return GenerationResponse(
                text=raw_text,
                provider_name="mock",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=raw_output,
            )

    email_service = EmailGenerationService(
        state_store=test_state_store,
        generator=EmailGenerator(engine=MockGeneratorEngine()),
    )

    draft = email_service.generate_draft(
        prospect=prospect,
        product=product_ctx,
        strategy_result=strat_res,
        outreach=outreach,
        research_package=research_pkg,
        context_id=context_id,
        research_session_id="session-metric-1",
    )

    assert draft.human_review_status == HumanReviewStatus.QUARANTINED_VERIFICATION_FAILED
    assert draft.verifier_result is not None
    assert draft.verifier_result.has_errors is True


def test_grounded_warrant_passes_verification_and_reaches_pending_review(test_state_store: StateStore):
    """Claim with verified whitelisted proof point passes verification and reaches PENDING_REVIEW."""
    contract = ProductContract(
        product_name="FleetPulse",
        contract_version="1.0.0",
        description="Routing engine",
        value_proposition="Reduces dispatch overhead by 50%",
        capabilities=[
            CapabilityNode(
                capability_id="cap-routing",
                name="Automated Fleet Routing",
                target_operational_problems=["fuel waste"],
                supported_use_cases=["Fleet logistics"],
                hard_exclusions=["Air freight dispatch"],
                verified_proof_points=["50% reduction in dispatch overhead"],
            )
        ],
    )
    product_ctx = contract.to_context()
    prospect = _make_dummy_prospect()
    outreach = _make_dummy_outreach()
    research_pkg = _make_dummy_package()
    strat_res = _make_dummy_strategy_result(is_viable=True)

    gate = ProductCapabilityBoundsGate()
    claim_with_grounded_metric = DecomposedClaim(
        claim_id="c-grounded-metric",
        text="FleetPulse delivers a 50% reduction in dispatch overhead across logistics hubs.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=79,
        category=ClaimCategory.METRIC_IMPACT,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim_with_grounded_metric], contract, research_package=research_pkg)
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)
