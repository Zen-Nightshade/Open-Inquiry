from __future__ import annotations

import tempfile
from typing import Any, List, Optional, Tuple
import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import CapabilityNode, CompatibilityStatus, ProductContract
from tests.unit.test_claim_verifier_service import _make_strategy_result
from src.email_generation.claim_verifier.capability import ProductCapabilityBoundsGate
from src.email_generation.claim_verifier.decomposer import ClaimDecomposer
from src.email_generation.claim_verifier.entailment import (
    FactEntailmentGate,
    RawEntailmentOutput,
)
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    EntailmentVerdict,
    ModalityLevel,
    VerifierFindingSeverity,
)
from src.email_generation.claim_verifier.service import ClaimVerifierService
from src.email_generation.generator import EmailGenerator, RawEmailDraftOutput
from src.email_generation.linter import ReadabilityLinter
from src.email_generation.models import HumanReviewStatus
from src.email_generation.service import EmailGenerationService
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider import ProviderResponse
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
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


class MockEngine(GeneratorEngine):
    def __init__(self, raw_output: Optional[RawEntailmentOutput] = None, raise_error: Optional[Exception] = None):
        self.raw_output = raw_output
        self.raise_error = raise_error
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_count += 1
        if self.raise_error:
            raise self.raise_error
        return GenerationResponse(
            text="{}",
            provider_name="groq",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=self.raw_output or RawEntailmentOutput(verdict="NEUTRAL", rationale="Mock output"),
            metadata={},
        )


def _make_sample_research_package() -> Tuple[ResearchPackage, CanonicalFact, EvidenceRecord]:
    ev = EvidenceRecord(
        evidence_id="ev-infra-1",
        source_id="src-1",
        verbatim_excerpt="Black Mesa added 10,000 new compute nodes to their primary cluster.",
        confidence=ConfidenceLevel.HIGH,
    )
    fact = CanonicalFact(
        fact_id="fact-infra-1",
        claim_text="Black Mesa added 10,000 compute nodes to their cluster",
        evidence_records=[ev],
    )
    pkg = ResearchPackage(facts=(fact,), is_sufficient=True, is_complete=True)
    return pkg, fact, ev


def _make_test_strategy_result(
    is_conditional: bool = False,
    hard_exclusions: Optional[List[str]] = None,
) -> StrategyResult:
    cand = CandidateSignal(
        signal_id="sig-1",
        label="Cluster Rollout",
        description="10,000 node cluster expansion",
        supporting_fact_ids=["fact-infra-1"],
        observed_event="10,000 node cluster expansion",
        target_capability="cap-1",
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
    ranked = RankedSignal(signal=cand, score=score, rationale="Fit", eligibility=elig)

    qualifier = ToulminQualifier(
        eligibility_status=EligibilityStatus.ELIGIBLE,
        compatibility_status=(
            CompatibilityStatus.CONDITIONAL_ON_UNKNOWN if is_conditional else CompatibilityStatus.VERIFIED_COMPATIBLE
        ),
        verification_status=VerificationStatus.SUPPORTED,
        composite_confidence=ConfidenceLevel.HIGH,
        qualification_language="if clusters face telemetry overhead",
        unresolved_prerequisites=["AWS deployment"] if is_conditional else [],
    )
    warrant = ToulminWarrant(
        capability_id="cap-1",
        capability_name="Latency Tracking",
        reasoning="Rapid cluster node expansion can introduce telemetry sync delays.",
    )
    backing = ToulminBacking(
        capability_id="cap-1",
        whitelisted_proof_points=["50% reduction in telemetry lag"],
        provenance=OutputProvenance(origin=ProvenanceOrigin.MANUAL_INPUT, component="test"),
    )
    toulmin = ToulminArgument(
        grounds=ToulminGrounds(fact_ids=["fact-infra-1"], observed_event="10,000 node expansion"),
        warrant=warrant,
        backing=backing,
        qualifier=qualifier,
        claim=ToulminClaim(
            claim_type="diagnostic_inquiry" if is_conditional else "value_assertion",
            text="PulseRadar cuts telemetry lag without extra overhead",
            target_capability="Latency Tracking",
        ),
        rebuttal=ToulminRebuttal(),
    )
    strategy = OutreachStrategy(
        strategy_id="strat-1",
        primary_signal=ranked,
        alignment=ProspectProductAlignment(
            prospect_role_relevance="Direct telemetry relevance",
            business_implications="Cluster deployment creates telemetry latency",
            capability_mapping=["cap-1"],
        ),
        suggested_subject_line="Cluster Telemetry Inquiry",
        signal_id="sig-1",
        target_capability_id="cap-1",
        messaging_approach="technical_inquiry",
        angle="cluster_scaling_telemetry",
        central_message="PulseRadar helps manage cluster telemetry overhead",
        cta="Are you open to comparing cluster telemetry benchmarks this week?",
        tone_guidance="technical",
        is_conditional=is_conditional,
        unresolved_conditions=["AWS deployment"] if is_conditional else [],
        toulmin_argument=toulmin,
    )
    return StrategyResult(is_viable=True, strategy=strategy)


# 1. A claim that shares keywords with a fact but is not entailed
def test_claim_sharing_keywords_is_not_entailed():
    pkg, _, _ = _make_sample_research_package()
    gate = FactEntailmentGate()  # deterministic mode

    # Fact is about adding 10,000 nodes to cluster.
    # Claim mentions "cluster" and "nodes" and "mesa" but asserts a completely unentailed claim: shutting down.
    claim = DecomposedClaim(
        claim_id="c-unentailed-keywords",
        text="saw Black Mesa will decommission legacy compute cluster nodes next month",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=72,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    # Must NOT be ENTAILED solely because keywords ("cluster", "nodes", "mesa", "compute") overlap!
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "UNSUPPORTED_PROSPECT_ASSERTION" for f in findings)


# 2. A genuinely contradicted claim
def test_genuinely_contradicted_claim():
    ev = EvidenceRecord(
        evidence_id="ev-mig-delayed",
        source_id="src-mig",
        verbatim_excerpt="Wayne Enterprises cancelled all cloud migration initiatives.",
        confidence=ConfidenceLevel.HIGH,
    )
    fact = CanonicalFact(
        fact_id="fact-mig-delayed",
        claim_text="Wayne Enterprises cancelled cloud migration initiatives",
        evidence_records=[ev],
    )
    pkg = ResearchPackage(facts=(fact,))
    gate = FactEntailmentGate()

    claim = DecomposedClaim(
        claim_id="c-contra",
        text="congratulations on your completed cloud migration initiatives at Wayne Enterprises",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=84,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.CONTRADICTED
    assert any(f.code == "CONTRADICTED_FACT" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


# 3. A neutral claim with insufficient evidence
def test_neutral_claim_with_insufficient_evidence():
    pkg, _, _ = _make_sample_research_package()
    gate = FactEntailmentGate()

    claim = DecomposedClaim(
        claim_id="c-neutral",
        text="saw your engineering team adopted Kafka for distributed messaging",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=64,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "UNSUPPORTED_PROSPECT_ASSERTION" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


# 4. Missing or invalid fact IDs in LLM response
def test_invalid_fact_ids_rejected():
    pkg, fact, ev = _make_sample_research_package()
    # LLM returns a hallucinated fact ID
    engine = MockEngine(
        raw_output=RawEntailmentOutput(
            verdict="ENTAILED",
            supporting_fact_ids=["nonexistent-fact-999"],
            supporting_evidence_ids=[ev.evidence_id],
            confidence=0.9,
            rationale="Grounded in invalid fact ID",
        )
    )
    gate = FactEntailmentGate(engine=engine)

    claim = DecomposedClaim(
        claim_id="c-invalid-fact-id",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    # Downgraded to NEUTRAL because fact ID is invalid
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "FABRICATED_EVIDENCE_REFERENCE" for f in findings)
    assert any(f.code == "MISSING_EVIDENCE_REFERENCE" for f in findings)


# 5. Evidence IDs that do not belong to the referenced fact
def test_evidence_id_not_belonging_to_fact_rejected():
    ev1 = EvidenceRecord(evidence_id="ev-belonging", source_id="s1", verbatim_excerpt="Black Mesa added 10,000 new compute nodes")
    ev2_foreign = EvidenceRecord(evidence_id="ev-foreign", source_id="s2", verbatim_excerpt="Text 2")

    fact1 = CanonicalFact(fact_id="fact-1", claim_text="Black Mesa added 10,000 new compute nodes", evidence_records=[ev1])
    fact2 = CanonicalFact(fact_id="fact-2", claim_text="Fact 2", evidence_records=[ev2_foreign])
    pkg = ResearchPackage(facts=(fact1, fact2))

    # Model claims fact-1 is supported by ev-foreign (which belongs to fact-2)
    engine = MockEngine(
        raw_output=RawEntailmentOutput(
            verdict="ENTAILED",
            supporting_fact_ids=["fact-1"],
            supporting_evidence_ids=["ev-foreign"],
            confidence=0.9,
            rationale="Cross-referencing mismatched evidence",
        )
    )
    gate = FactEntailmentGate(engine=engine)

    claim = DecomposedClaim(
        claim_id="c-mismatched-ev",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    # Must downgrade to NEUTRAL because ev-foreign does not belong to fact-1
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "MISSING_EVIDENCE_REFERENCE" for f in findings)


# 6. Toulmin fallback references without validated fact/evidence backing
def test_toulmin_fallback_without_validated_research_fails():
    gate = FactEntailmentGate()

    toulmin = ToulminArgument(
        grounds=ToulminGrounds(
            fact_ids=["nonexistent-fact-xyz"],
            observed_event="Acme deployed 500 servers",
            verification_status=VerificationStatus.SUPPORTED,
            composite_confidence=ConfidenceLevel.HIGH,
        ),
        warrant=ToulminWarrant(capability_id="c1", capability_name="Observability", reasoning="scale"),
        backing=ToulminBacking(
            capability_id="c1",
            whitelisted_proof_points=["test"],
            provenance=OutputProvenance(origin=ProvenanceOrigin.MANUAL_INPUT, component="test"),
        ),
        qualifier=ToulminQualifier(
            eligibility_status=EligibilityStatus.ELIGIBLE,
            compatibility_status=CompatibilityStatus.VERIFIED_COMPATIBLE,
            verification_status=VerificationStatus.SUPPORTED,
            composite_confidence=ConfidenceLevel.HIGH,
            qualification_language="Directly applicable",
        ),
        claim=ToulminClaim(claim_type="value", text="claim", target_capability="c1"),
        rebuttal=ToulminRebuttal(),
    )

    claim = DecomposedClaim(
        claim_id="c-toulmin-fake",
        text="saw Acme deployed 500 servers recently",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=38,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    # Calling with research_package=None: Toulmin text must NOT become ungrounded fact!
    results, findings = gate.verify([claim], research_package=None, toulmin_argument=toulmin)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "MISSING_RESEARCH_EVIDENCE" for f in findings)


# 7. Factual subject-line assertions
def test_factual_subject_line_assertions_are_material():
    decomposer = ClaimDecomposer()

    # A subject asserting a factual claim about prospect expansion
    subject = "Black Mesa 10,000 node cluster expansion"
    claims, findings = decomposer.decompose(
        subject=subject,
        body="Elena, following up on our chat.",
        prospect_name="Elena Vance",
        company_name="Black Mesa",
    )

    subj_claim = next((c for c in claims if c.section == ClaimSection.SUBJECT), None)
    assert subj_claim is not None
    # Must be marked is_material=True because it contains factual content, not generic filler!
    assert subj_claim.is_material is True
    assert subj_claim.category == ClaimCategory.METRIC_IMPACT or subj_claim.category == ClaimCategory.PROSPECT_OBSERVATION


# 8. Negation and modality escalation
def test_modality_escalation_flagged():
    pkg, _, _ = _make_sample_research_package()
    service = ClaimVerifierService()

    from src.context.models import ProductContext, ProspectContext
    prospect = ProspectContext(full_name="Elena Vance", company_name="Black Mesa")
    product = ProductContext(product_name="PulseRadar", description="Telemetry", value_proposition="Telemetry")

    # Strategy has conditional qualifier ("if clusters face telemetry overhead")
    strat_res = _make_test_strategy_result(is_conditional=True)

    # Draft asserts definitive capability without conditional qualifier
    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
        "PulseRadar guarantees complete elimination of node latency bottlenecks.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    res = service.verify(
        subject=subject,
        body=body,
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        research_package=pkg,
    )

    assert res.passed is False
    assert any(f.code == "MODALITY_ESCALATION" for f in res.findings)


# 9. Product capability absent from a non-exhaustive contract
def test_unknown_capability_non_exhaustive_contract():
    gate = ProductCapabilityBoundsGate()

    cap = CapabilityNode(
        capability_id="cap-known",
        name="Distributed Tracing",
        verified_proof_points=["50% reduction in telemetry lag"],
    )
    contract = ProductContract(
        product_name="PulseRadar",
        contract_version="1.0.0",
        description="Tracing tool",
        value_proposition="Trace distributed nodes",
        capabilities=[cap],
        is_exhaustive=False,  # NON-EXHAUSTIVE
    )

    claim = DecomposedClaim(
        claim_id="c-unknown-cap",
        text="We built PulseRadar to optimize database disk caching natively.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=64,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        is_product_claim=True,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    # Under non-exhaustive contract, absence is unknown/conditional, not proof of impossibility
    assert results[0].is_whitelisted is False
    assert results[0].violates_exclusion is False
    # Finding must be a WARNING (not an ERROR)
    assert any(f.code == "CONDITIONAL_CAPABILITY_UNKNOWN" and f.severity == VerifierFindingSeverity.WARNING for f in findings)


# 10. Explicit product hard exclusions
def test_explicit_hard_exclusion_violation():
    gate = ProductCapabilityBoundsGate()

    cap = CapabilityNode(
        capability_id="cap-1",
        name="Distributed Tracing",
        hard_exclusions=["on-premise OpenShift"],
        verified_proof_points=["50% reduction"],
    )
    contract = ProductContract(
        product_name="PulseRadar",
        contract_version="1.0.0",
        description="Tracing",
        value_proposition="Tracing",
        capabilities=[cap],
        hard_exclusions=["on-premise OpenShift"],
    )

    claim = DecomposedClaim(
        claim_id="c-exclusion",
        text="We built PulseRadar to support on-premise OpenShift clusters.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=62,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        is_product_claim=True,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    assert results[0].violates_exclusion is True
    assert any(f.code == "HARD_EXCLUSION_VIOLATION" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


# 11. LLM provider timeout or failure
def test_llm_provider_timeout_fails_closed():
    pkg, _, _ = _make_sample_research_package()
    mock_engine = MockEngine(raise_error=TimeoutError("API connection timed out after 3000ms"))
    gate = FactEntailmentGate(engine=mock_engine)

    claim = DecomposedClaim(
        claim_id="c-timeout-check",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    # Must NOT be marked as ENTAILED and NOT as CONTRADICTED
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "VERIFIER_SERVICE_UNAVAILABLE" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


# 12. One-rewrite maximum, including re-verification after rewriting
class MockGeneratorOutputs(EmailGenerator):
    def __init__(self, drafts: List[RawEmailDraftOutput]):
        self.drafts = list(drafts)
        self.call_count = 0

    def generate(self, **kwargs) -> RawEmailDraftOutput:
        self.call_count += 1
        return self.drafts.pop(0)


def test_one_rewrite_maximum_including_reverification():
    # Attempt 1 has fabricated proof point (99%)
    bad_draft_1 = RawEmailDraftOutput(
        subject="Cluster Telemetry Inquiry",
        body="Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\nWe built PulseRadar to cut node latency by 99%.\n\nAre you open to comparing benchmarks this week?",
    )
    # Attempt 2 has corporate jargon warning and hard exclusion
    bad_draft_2 = RawEmailDraftOutput(
        subject="Cluster Telemetry Inquiry",
        body="Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\nPulseRadar supports on-premise OpenShift.\n\nAre you open to comparing benchmarks this week?",
    )
    # Attempt 3 should never be called
    unused_draft_3 = RawEmailDraftOutput(
        subject="Cluster Telemetry Inquiry",
        body="Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\nPulseRadar cuts node latency by 50%.\n\nAre you open to comparing benchmarks this week?",
    )

    generator = MockGeneratorOutputs([bad_draft_1, bad_draft_2, unused_draft_3])
    service = EmailGenerationService(generator=generator, linter=ReadabilityLinter())

    pkg, _, _ = _make_sample_research_package()

    from src.context.models import CapabilityNode, ProductContract, ProductContext, ProspectContext, OutreachContext
    prospect = ProspectContext(full_name="Elena Vance", company_name="Black Mesa")
    cap = CapabilityNode(
        capability_id="cap-1",
        name="Latency",
        hard_exclusions=["on-premise OpenShift"],
        verified_proof_points=["50% reduction in telemetry lag"],
    )
    contract = ProductContract(
        product_name="PulseRadar",
        contract_version="1.0.0",
        description="Telemetry",
        value_proposition="Reduces latency",
        capabilities=[cap],
        hard_exclusions=["on-premise OpenShift"],
    )
    product = contract.to_context()
    outreach = OutreachContext(objective="Discovery", sender_identity="Alex", sender_name="Alex", sender_company="Vanguard")

    strat_res = _make_test_strategy_result(is_conditional=False)

    draft = service.generate_draft(
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        outreach=outreach,
        research_package=pkg,
    )

    # Exactly 2 calls (1 original + 1 rewrite maximum)
    assert generator.call_count == 2
    assert draft.generation_attempts == 2
    # Second attempt was re-verified and produced error finding, so passed is False
    assert draft.verifier_result.passed is False
    # Draft is quarantined due to verification failure (circuit breaker protects review queue)
    assert draft.human_review_status == HumanReviewStatus.QUARANTINED_VERIFICATION_FAILED
