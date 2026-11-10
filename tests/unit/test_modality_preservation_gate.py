from __future__ import annotations

import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import CompatibilityStatus
from src.email_generation.claim_verifier.modality import ModalityPreservationGate
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    ModalityLevel,
    VerifierFindingSeverity,
)
from src.researcher.models import ConfidenceLevel, VerificationStatus
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
    ToulminArgument,
    ToulminBacking,
    ToulminClaim,
    ToulminGrounds,
    ToulminQualifier,
    ToulminRebuttal,
    ToulminWarrant,
)


def _make_dummy_strategy(
    is_conditional: bool = False,
    is_inferred: bool = False,
    cta_type: CTAType = CTAType.DIAGNOSTIC_QUESTION,
    unresolved_prerequisites: Optional[list[str]] = None,
) -> OutreachStrategy:
    cand = CandidateSignal(
        signal_id="sig-test-1",
        label="Cluster Rollout",
        description="Massive distributed cluster deployment",
        supporting_fact_ids=["fact-101"],
        observed_event="Massive cluster expansion",
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
    ranked = RankedSignal(signal=cand, score=score, rationale="Strong fit", eligibility=elig)

    qualifier = ToulminQualifier(
        eligibility_status=EligibilityStatus.ELIGIBLE,
        compatibility_status=(
            CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
            if is_conditional
            else CompatibilityStatus.VERIFIED_COMPATIBLE
        ),
        verification_status=VerificationStatus.SUPPORTED,
        composite_confidence=ConfidenceLevel.HIGH,
        qualification_language="if clusters face telemetry overhead",
        unresolved_prerequisites=unresolved_prerequisites or (["AWS deployment"] if is_conditional else []),
    )
    warrant = ToulminWarrant(
        capability_id="cap-1",
        capability_name="Node Latency Reduction",
        reasoning="Rapid cluster node expansion can introduce telemetry sync delays.",
        is_inferred=is_inferred,
    )
    backing = ToulminBacking(
        capability_id="cap-1",
        whitelisted_proof_points=["50% latency reduction in Kubernetes clusters"],
        has_whitelisted_proof=True,
        provenance=OutputProvenance(origin=ProvenanceOrigin.MANUAL_INPUT, component="test"),
    )
    toulmin = ToulminArgument(
        grounds=ToulminGrounds(
            fact_ids=["fact-1"],
            observed_event="10,000 node expansion",
        ),
        warrant=warrant,
        backing=backing,
        qualifier=qualifier,
        claim=ToulminClaim(
            claim_type="diagnostic_inquiry" if is_conditional else "value_assertion",
            text="PulseRadar cuts telemetry lag without extra overhead",
            target_capability="Node Latency Reduction",
        ),
        rebuttal=ToulminRebuttal(),
    )
    return OutreachStrategy(
        strategy_id="strat-101",
        primary_signal=ranked,
        alignment=ProspectProductAlignment(
            prospect_role_relevance="Telemetry aligns directly with engineering priorities",
            business_implications="Cluster deployment creates telemetry latency bottlenecks",
            capability_mapping=["cap-1"],
        ),
        suggested_subject_line="Scaling cluster telemetry at Black Mesa",
        signal_id="sig-1",
        target_capability_id="cap-1",
        messaging_approach="technical_inquiry",
        angle="cluster_scaling_telemetry",
        central_message="PulseRadar helps manage cluster telemetry overhead",
        cta="Are you open to comparing cluster telemetry benchmarks this week?",
        tone_guidance="technical",
        is_conditional=is_conditional,
        unresolved_conditions=unresolved_prerequisites or (["AWS deployment"] if is_conditional else []),
        toulmin_argument=toulmin,
        cta_type=cta_type,
    )



def test_modality_conditional_strategy_preserved_with_hedges():
    strategy = _make_dummy_strategy(is_conditional=True)
    gate = ModalityPreservationGate()

    claim = DecomposedClaim(
        claim_id="c-hedge",
        text="If your clusters face high synchronization overhead, PulseRadar could help reduce queue bottlenecks.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=104,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.CONDITIONAL,
        is_material=True,
    )

    results, findings = gate.verify([claim], strategy, strategy.toulmin_argument)
    assert len(results) == 1
    assert not results[0].is_escalated
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_modality_conditional_strategy_escalated_to_definitive():
    strategy = _make_dummy_strategy(is_conditional=True)
    gate = ModalityPreservationGate()

    # Makes definitive, unconditional promise under conditional strategy
    claim = DecomposedClaim(
        claim_id="c-escalated",
        text="PulseRadar will eliminate your cluster latency bottlenecks completely.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=74,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], strategy, strategy.toulmin_argument)
    assert len(results) == 1
    assert results[0].is_escalated
    assert any(f.code == "MODALITY_ESCALATION" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_modality_inferred_opportunity_represented_as_fact():
    strategy = _make_dummy_strategy(is_conditional=False, is_inferred=True)
    gate = ModalityPreservationGate()

    claim = DecomposedClaim(
        claim_id="c-inferred",
        text="Because your team is struggling with severe telemetry lag",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], strategy, strategy.toulmin_argument)
    assert len(results) == 1
    assert results[0].is_escalated
    assert any(f.code == "INFERRED_OPPORTUNITY_AS_FACT" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_modality_diagnostic_cta_escalated_to_presumptive_meeting():
    strategy = _make_dummy_strategy(is_conditional=False, cta_type=CTAType.DIAGNOSTIC_QUESTION)
    gate = ModalityPreservationGate()

    claim = DecomposedClaim(
        claim_id="c-cta-escalated",
        text="Let's book a 30-minute demo call tomorrow at 2 PM to walk through this.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=72,
        category=ClaimCategory.RELATIONAL_CTA,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], strategy, strategy.toulmin_argument)
    assert len(results) == 1
    assert results[0].is_escalated
    assert any(f.code == "CTA_MODALITY_ESCALATION" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_modality_safe_diagnostic_question_passes():
    strategy = _make_dummy_strategy(is_conditional=False, cta_type=CTAType.DIAGNOSTIC_QUESTION)
    gate = ModalityPreservationGate()

    claim = DecomposedClaim(
        claim_id="c-cta-safe",
        text="Are you open to comparing cluster telemetry benchmarks this week?",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=66,
        category=ClaimCategory.RELATIONAL_CTA,
        modality=ModalityLevel.DIAGNOSTIC,
        is_material=True,
    )

    results, findings = gate.verify([claim], strategy, strategy.toulmin_argument)
    assert len(results) == 1
    assert not results[0].is_escalated
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)
