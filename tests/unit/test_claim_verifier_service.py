from __future__ import annotations

import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CapabilityNode,
    CompatibilityStatus,
    ProductContract,
    ProductContext,
    ProspectContext,
)
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    EntailmentVerdict,
    ModalityLevel,
    VerificationFinding,
    VerifierFindingSeverity,
    VerifierResult,
)
from src.email_generation.claim_verifier.service import ClaimVerifierService
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


def _make_prospect() -> ProspectContext:
    return ProspectContext(
        full_name="Elena Vance",
        company_name="Black Mesa",
        company_domain="blackmesa.example",
        designation="VP Infrastructure",
    )


def _make_product(is_exhaustive: bool = False) -> ProductContext:
    cap = CapabilityNode(
        capability_id="cap-1",
        name="Node Latency Telemetry",
        target_operational_problems=["telemetry lag", "queue bottlenecks"],
        supported_use_cases=["Kubernetes node monitoring"],
        hard_exclusions=["on-premise OpenShift"],
        verified_proof_points=["50% latency reduction in Kubernetes clusters"],
    )
    contract = ProductContract(
        product_name="PulseRadar",
        contract_version="1.0.0",
        description="Distributed telemetry intelligence",
        value_proposition="Cuts node latency by 50% without adding extra overhead",
        capabilities=[cap],
        hard_exclusions=["on-premise OpenShift"],
        is_exhaustive=is_exhaustive,
    )
    return contract.to_context()


def _make_research_package() -> ResearchPackage:
    ev = EvidenceRecord(
        evidence_id="ev-101",
        source_id="src-001",
        verbatim_excerpt="Black Mesa added 10,000 new compute nodes to their primary cluster.",
        confidence=ConfidenceLevel.HIGH,
    )
    fact = CanonicalFact(
        fact_id="fact-201",
        claim_text="Black Mesa added 10,000 compute nodes to their cluster",
        evidence_records=[ev],
    )
    return ResearchPackage(facts=(fact,))


def _make_strategy_result(
    is_conditional: bool = False,
    is_inferred: bool = False,
    cta_type: CTAType = CTAType.DIAGNOSTIC_QUESTION,
) -> StrategyResult:
    cand = CandidateSignal(
        signal_id="sig-1",
        label="Cluster Rollout",
        description="10,000 node cluster expansion",
        supporting_fact_ids=["fact-201"],
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
        capability_name="Node Latency Telemetry",
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
        grounds=ToulminGrounds(fact_ids=["fact-201"], observed_event="10,000 node expansion"),
        warrant=warrant,
        backing=backing,
        qualifier=qualifier,
        claim=ToulminClaim(
            claim_type="diagnostic_inquiry" if is_conditional else "value_assertion",
            text="PulseRadar cuts telemetry lag without extra overhead",
            target_capability="Node Latency Telemetry",
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
        cta_type=cta_type,
    )
    return StrategyResult(
        is_viable=True,
        strategy=strategy,
        ranked_signals=[ranked],
    )


def test_claim_verifier_service_clean_draft_passes_all_gates():
    service = ClaimVerifierService()
    prospect = _make_prospect()
    product = _make_product()
    strat_res = _make_strategy_result()
    research_pkg = _make_research_package()

    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
        "When clusters grow fast, tracking can create heavy system lag. We built PulseRadar to cut node latency by 50% without adding extra overhead to your stack.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    verifier_result: VerifierResult = service.verify(
        subject=subject,
        body=body,
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        research_package=research_pkg,
    )

    assert verifier_result.passed is True
    assert verifier_result.has_errors is False
    assert len(verifier_result.decomposed_claims) > 0

    # Test conversion to ClaimVerification
    claim_verifs = verifier_result.to_claim_verifications()
    assert len(claim_verifs) == len(verifier_result.decomposed_claims)
    assert any(cv.status == VerificationStatus.SUPPORTED for cv in claim_verifs)


def test_claim_verifier_service_catches_unsupported_fact():
    service = ClaimVerifierService()
    prospect = _make_prospect()
    product = _make_product()
    strat_res = _make_strategy_result()
    research_pkg = _make_research_package()

    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your recent acquisition of Cyberdyne Systems.\n\n"
        "We built PulseRadar to cut node latency by 50%.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    verifier_result = service.verify(
        subject=subject,
        body=body,
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        research_package=research_pkg,
    )

    assert verifier_result.passed is False
    assert verifier_result.has_errors is True
    assert any(f.code == "UNSUPPORTED_PROSPECT_ASSERTION" for f in verifier_result.findings)


def test_claim_verifier_service_catches_modality_escalation():
    service = ClaimVerifierService()
    prospect = _make_prospect()
    product = _make_product()
    # Strategy is conditional on AWS deployment
    strat_res = _make_strategy_result(is_conditional=True)
    research_pkg = _make_research_package()

    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
        "PulseRadar will eliminate all cluster latency bottlenecks completely.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    verifier_result = service.verify(
        subject=subject,
        body=body,
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        research_package=research_pkg,
    )

    assert verifier_result.passed is False
    assert verifier_result.has_errors is True
    assert any(f.code == "MODALITY_ESCALATION" for f in verifier_result.findings)


def test_claim_verifier_service_catches_hard_exclusion_violation():
    service = ClaimVerifierService()
    prospect = _make_prospect()
    product = _make_product()
    strat_res = _make_strategy_result()
    research_pkg = _make_research_package()

    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
        "We built PulseRadar to support on-premise OpenShift clusters.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    verifier_result = service.verify(
        subject=subject,
        body=body,
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        research_package=research_pkg,
    )

    assert verifier_result.passed is False
    assert verifier_result.has_errors is True
    assert any(f.code == "HARD_EXCLUSION_VIOLATION" for f in verifier_result.findings)


def test_claim_verifier_service_catches_fabricated_metric():
    service = ClaimVerifierService()
    prospect = _make_prospect()
    product = _make_product()
    strat_res = _make_strategy_result()
    research_pkg = _make_research_package()

    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
        "We built PulseRadar to cut node latency by 99% with 10x throughput.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    verifier_result = service.verify(
        subject=subject,
        body=body,
        prospect=prospect,
        product=product,
        strategy_result=strat_res,
        research_package=research_pkg,
    )

    assert verifier_result.passed is False
    assert verifier_result.has_errors is True
    assert any(f.code == "UNSUPPORTED_PROOF_POINT" for f in verifier_result.findings)
