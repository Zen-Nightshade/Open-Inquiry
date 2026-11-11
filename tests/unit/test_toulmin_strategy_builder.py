from __future__ import annotations

import datetime
from typing import Any, List
import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CapabilityNode,
    CompatibilityResult,
    CompatibilityStatus,
    OutreachContext,
    ProductContract,
    ProductContext,
    ProspectContext,
)
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
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
    VerificationStatus,
)
from src.signal_strategy.models import (
    CTAType,
    CandidateSignal,
    EligibilityStatus,
    GateIdentifier,
    GateResult,
    OutreachStrategy,
    RankedSignal,
    SignalEligibility,
    SignalScore,
    StrategyResult,
    ToulminArgument,
)
from src.signal_strategy.strategy_builder import (
    RawStrategyOutput,
    StrategyBuilder,
)


class MockEngine(GeneratorEngine):
    def __init__(self, response_data: Any):
        super().__init__(default_provider="mock")
        self.response_data = response_data

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="{}",
            provider_name="mock",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=self.response_data,
        )


@pytest.fixture
def toulmin_fixtures():
    today = datetime.date.today()
    prospect = ProspectContext(
        full_name="Dr. Elena Vance",
        company_name="Black Mesa Research",
        company_domain="blackmesa.gov",
        designation="Chief Research Officer",
    )
    prospect_entity = EntityRecord(
        entity_id="entity-prospect-elena",
        entity_type=EntityType.PERSON,
        raw_name="Dr. Elena Vance",
        canonical_name="Elena Vance, Ph.D.",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )
    company_entity = EntityRecord(
        entity_id="entity-company-blackmesa",
        entity_type=EntityType.ORGANIZATION,
        raw_name="Black Mesa Research",
        canonical_name="Black Mesa Research Facility",
        domain="blackmesa.gov",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )

    ev1 = EvidenceRecord(
        evidence_id="ev-lambda-exp",
        source_id="src-lambda-report",
        verbatim_excerpt="Black Mesa commissioned the Lambda Complex supercomputing array.",
        event_date=today - datetime.timedelta(days=15),
        confidence=ConfidenceLevel.HIGH,
    )
    ev2 = EvidenceRecord(
        evidence_id="ev-resonance-issue",
        source_id="src-lambda-report",
        verbatim_excerpt="Anomalous materials experiments in Sector C experienced thermal containment fluctuations.",
        event_date=today - datetime.timedelta(days=10),
        confidence=ConfidenceLevel.HIGH,
    )
    source1 = SourceRecord(
        source_id="src-lambda-report",
        url="https://blackmesa.gov/reports/annual-facility-overview",
        domain="blackmesa.gov",
        title="Black Mesa Sector C Report",
        raw_text="Facility expansion report",
        source_type=SourceType.COMPANY_WEBSITE,
        fetch_date=today,
    )

    fact1 = CanonicalFact(
        fact_id="fact-lambda-expansion",
        claim_text="Black Mesa commissioned the Lambda Complex supercomputing array.",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        entity_ids=["entity-company-blackmesa"],
        evidence_records=[ev1],
        primary_event_date=today - datetime.timedelta(days=15),
        composite_confidence=ConfidenceLevel.HIGH,
    )
    fact2 = CanonicalFact(
        fact_id="fact-thermal-containment",
        claim_text="Anomalous materials experiments in Sector C experienced thermal containment fluctuations.",
        fact_category=FactCategory.CHALLENGE,
        entity_ids=["entity-company-blackmesa"],
        evidence_records=[ev2],
        primary_event_date=today - datetime.timedelta(days=10),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    research_package = ResearchPackage(
        prospect_entity=prospect_entity,
        company_entity=company_entity,
        sources=(source1,),
        facts=(fact1, fact2),
        search_queries_used=("black mesa sector c", "lambda complex expansion"),
        is_sufficient=True,
        is_complete=True,
    )

    cap1 = CapabilityNode(
        capability_id="cap-supercooled-containment",
        name="Supercooled Containment Field",
        description="Sub-Kelvin thermal barrier isolation for unstable materials",
        target_operational_problems=["Thermal runaway", "Cryogenic containment failure"],
        supported_use_cases=["High-energy anomaly containment", "Lambda core stabilization"],
        required_prerequisites=["Liquid nitrogen pipeline"],
        hard_exclusions=["Unshielded radioactive waste dumps"],
        verified_proof_points=[
            "Reduces containment breach probability by 95% under thermal stress",
            "Validated by MIT Materials Laboratory benchmark audit",
        ],
        provenance=OutputProvenance(
            origin=ProvenanceOrigin.MANUAL_INPUT,
            component="product_contract",
        ),
    )
    cap2 = CapabilityNode(
        capability_id="cap-resonance-array",
        name="Resonance Dampening Array",
        description="Multi-frequency electromagnetic harmonic dampening",
        target_operational_problems=["Anti-mass resonance cascade"],
        supported_use_cases=["Sub-atomic resonance suppression"],
        verified_proof_points=[],
        provenance=OutputProvenance(
            origin=ProvenanceOrigin.MANUAL_INPUT,
            component="product_contract",
        ),
    )

    contract = ProductContract(
        product_name="Aperture Containment Suite",
        description="Advanced physical and resonance containment hardware",
        capabilities=[cap1, cap2],
        value_proposition="Eliminates laboratory containment failures through autonomous isolation.",
        hard_exclusions=["Unshielded antimatter reactors"],
        is_exhaustive=True,
    )
    product = contract.to_context()

    outreach = OutreachContext(
        objective="Schedule introductory briefing on Lambda containment protocols",
        sender_identity="Dr. Isaac Kleiner",
        desired_cta="Would you be open to a 15-minute briefing next Thursday?",
        tone="peer-to-peer scientific",
        constraints=["Under 90 words", "Do not cite competitor pricing"],
    )

    return prospect, product, outreach, research_package, cap1, cap2


def make_ranked_signal(
    signal_id: str,
    label: str,
    fact_ids: List[str],
    target_cap: str = "Supercooled Containment Field",
    is_conditional: bool = False,
    unresolved_conditions: List[str] = None,
    composite_score: float = 0.90,
    product_alignment: float = 0.90,
    associated_entity_ids: List[str] = None,
) -> RankedSignal:
    cand = CandidateSignal(
        signal_id=signal_id,
        label=label,
        description=f"Observed event for {label}",
        supporting_fact_ids=fact_ids,
        fact_category=FactCategory.CHALLENGE,
        observed_event=f"Observed {label}",
        business_implication=f"Operational implications of {label}",
        target_capability=target_cap,
        associated_entity_ids=associated_entity_ids or [],
    )
    score = SignalScore(
        evidence_strength=0.95,
        recency_score=0.90,
        prospect_relevance=0.90,
        product_alignment=product_alignment,
        business_significance=0.85,
        composite_score=composite_score,
    )
    gate_res = GateResult(gate=GateIdentifier.PRODUCT_COMPATIBILITY, passed=True, reason="Compatible")
    status = EligibilityStatus.CONDITIONAL if is_conditional else EligibilityStatus.ELIGIBLE
    elig = SignalEligibility(
        status=status,
        is_eligible=not is_conditional,
        identity_gate=GateResult(gate=GateIdentifier.IDENTITY, passed=True, reason="Valid"),
        evidence_gate=GateResult(gate=GateIdentifier.EVIDENCE, passed=True, reason="Adequate"),
        product_gate=gate_res,
        persona_gate=GateResult(gate=GateIdentifier.PERSONA, passed=True, reason="Relevant"),
        unresolved_conditions=unresolved_conditions or [],
        matched_capability_name=target_cap,
    )
    return RankedSignal(signal=cand, score=score, rationale="Strong fit", eligibility=elig)


def test_valid_grounds_reference_facts_in_research_package(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-valid-grounds", "Lambda Thermal Surge", ["fact-lambda-expansion"])

    raw = RawStrategyOutput(
        prospect_role_relevance="Elena Vance is CRO.",
        business_implications="Thermal expansion threatens Lambda core.",
        capability_mapping=["Supercooled Containment Field"],
        messaging_approach="challenge-led",
        central_message="Aperture Supercooled Containment Field secures Lambda core.",
        suggested_subject_line="Lambda containment protocol",
        cta="15-minute briefing next week?",
        tone_guidance="Scientific peer",
        constraint_checklist=[],
    )
    builder = StrategyBuilder(engine=MockEngine(raw))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    assert result.strategy is not None
    arg = result.strategy.toulmin_argument
    assert arg is not None
    assert arg.grounds.fact_ids == ["fact-lambda-expansion"]
    assert "ev-lambda-exp" in arg.grounds.evidence_ids
    assert "blackmesa.gov" in arg.grounds.source_domains
    assert arg.grounds.verification_status == VerificationStatus.SUPPORTED


def test_nonexistent_fact_references_are_rejected(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-bad-fact", "Phantom Signal", ["fact-does-not-exist-999"])

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert "does not exist in research package" in (result.rejection_reason or "")


def test_evidence_references_must_exist(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-ev-test", "Lambda Event", ["fact-lambda-expansion"])
    builder = StrategyBuilder(engine=MockEngine(None))

    arg, err = builder._bind_and_validate_toulmin_argument(
        primary_signal=ranked,
        product=product,
        prospect=prospect,
        research_package=pkg,
        explicit_evidence_ids=["ev-nonexistent-777"],
    )
    assert arg is None
    assert err is not None
    assert "does not exist in research package" in err


def test_evidence_must_support_the_fact_it_is_attached_to(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-ev-mismatch", "Lambda Event", ["fact-lambda-expansion"])
    builder = StrategyBuilder(engine=MockEngine(None))

    arg, err = builder._bind_and_validate_toulmin_argument(
        primary_signal=ranked,
        product=product,
        prospect=prospect,
        research_package=pkg,
        explicit_evidence_ids=["ev-resonance-issue"],
    )
    assert arg is None
    assert err is not None
    assert "does not support the fact it is attached to" in err


def test_references_cannot_cross_prospect_entity_boundaries_incorrectly(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures

    alien_ev = EvidenceRecord(
        evidence_id="ev-aperture-competitor",
        source_id="src-lambda-report",
        verbatim_excerpt="Aperture Science acquired rival facility.",
        confidence=ConfidenceLevel.HIGH,
    )
    alien_fact = CanonicalFact(
        fact_id="fact-alien-competitor",
        claim_text="Aperture Science acquired rival facility.",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        entity_ids=["entity-aperture-laboratories"],
        evidence_records=[alien_ev],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    pkg_with_alien = ResearchPackage(
        prospect_entity=pkg.prospect_entity,
        company_entity=pkg.company_entity,
        sources=pkg.sources,
        facts=pkg.facts + (alien_fact,),
        is_sufficient=True,
        is_complete=True,
    )

    ranked = make_ranked_signal("sig-alien", "Competitor Acquisition", ["fact-alien-competitor"])
    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg_with_alien,
    )

    assert result.is_viable is False
    assert "do not match target company entity" in (result.rejection_reason or "")


def test_capability_resolution_uses_authoritative_ids(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-cap-res", "Thermal Expansion", ["fact-lambda-expansion"], target_cap="Supercooled Containment Field")

    raw = RawStrategyOutput(
        prospect_role_relevance="Elena Vance is CRO.",
        business_implications="Thermal expansion threatens core.",
        capability_mapping=["Supercooled Containment Field"],
        messaging_approach="challenge-led",
        central_message="Supercooled Containment Field isolates Lambda core.",
        suggested_subject_line="Lambda containment protocol",
        cta="15-minute briefing next week?",
        tone_guidance="Scientific peer",
        constraint_checklist=[],
    )
    builder = StrategyBuilder(engine=MockEngine(raw))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    warrant = result.strategy.toulmin_argument.warrant
    assert warrant.capability_id == cap1.capability_id == "cap-supercooled-containment"
    assert warrant.capability_name == cap1.name


def test_ambiguous_or_undeclared_capabilities_are_not_silently_substituted(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-hallucinated", "Gravity Distortion", ["fact-lambda-expansion"], target_cap="Quantum Gravity Repulsor")

    raw = RawStrategyOutput(
        prospect_role_relevance="Elena Vance is CRO.",
        business_implications="Gravity anomaly.",
        capability_mapping=["Quantum Gravity Repulsor"],
        messaging_approach="challenge-led",
        central_message="Quantum Gravity Repulsor eliminates gravity fluctuations.",
        suggested_subject_line="Gravity protocol",
        cta="Chat next week?",
        tone_guidance="Scientific peer",
        constraint_checklist=[],
    )
    builder = StrategyBuilder(engine=MockEngine(raw))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert "could not be resolved" in (result.rejection_reason or "")


def test_proof_points_are_restricted_to_the_capability_whitelist(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-proof-pts", "Thermal Surge", ["fact-lambda-expansion"], target_cap="Supercooled Containment Field")

    raw = RawStrategyOutput(
        prospect_role_relevance="Elena Vance is CRO.",
        business_implications="Thermal stress.",
        capability_mapping=["Supercooled Containment Field"],
        messaging_approach="challenge-led",
        central_message="Aperture Supercooled Containment Field secures Lambda core.",
        suggested_subject_line="Lambda containment protocol",
        cta="15-minute briefing next week?",
        tone_guidance="Scientific peer",
        constraint_checklist=[],
        proposed_proof_points=[
            "Validated by MIT Materials Laboratory benchmark audit",
            "Fabricated 100% Zero Bug Guarantee",
        ],
    )
    builder = StrategyBuilder(engine=MockEngine(raw))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    backing = result.strategy.toulmin_argument.backing
    assert "Validated by MIT Materials Laboratory benchmark audit" in backing.whitelisted_proof_points
    assert "Fabricated 100% Zero Bug Guarantee" not in backing.whitelisted_proof_points
    assert backing.has_whitelisted_proof is True


def test_manual_proof_point_provenance_is_preserved_without_implying_independent_verification(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-prov-test", "Thermal Surge", ["fact-lambda-expansion"])

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    backing = result.strategy.toulmin_argument.backing
    assert backing.provenance.origin == ProvenanceOrigin.MANUAL_INPUT
    assert backing.provenance.component == "product_contract"


def test_llm_reasoning_is_not_represented_as_verified_evidence(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-reasoning-test", "Thermal Surge", ["fact-lambda-expansion"])

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    arg = result.strategy.toulmin_argument
    assert arg.warrant.is_inferred is True
    assert arg.grounds.verification_status == VerificationStatus.SUPPORTED


def test_verified_compatible_arguments_can_produce_appropriately_qualified_claims(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-verified-claim", "Thermal Surge", ["fact-lambda-expansion"], is_conditional=False)

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    arg = result.strategy.toulmin_argument
    assert arg.claim.claim_type == "value_assertion"
    assert arg.qualifier.compatibility_status == CompatibilityStatus.VERIFIED_COMPATIBLE
    assert arg.qualifier.eligibility_status == EligibilityStatus.ELIGIBLE


def test_conditional_compatibility_produces_diagnostic_claims(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal(
        "sig-conditional-claim",
        "Thermal Surge",
        ["fact-lambda-expansion"],
        is_conditional=True,
        unresolved_conditions=["Liquid nitrogen pipeline"],
    )

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    arg = result.strategy.toulmin_argument
    assert arg.claim.claim_type == "diagnostic_inquiry"
    assert arg.qualifier.compatibility_status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
    assert "Exploring whether" in arg.claim.text


def test_unsupported_outcome_claims_are_rejected_or_qualified(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-unsupported-outcome", "Resonance Cascade", ["fact-thermal-containment"], target_cap="Resonance Dampening Array")

    raw = RawStrategyOutput(
        prospect_role_relevance="Elena Vance is CRO.",
        business_implications="Resonance cascade risk.",
        capability_mapping=["Resonance Dampening Array"],
        messaging_approach="challenge-led",
        central_message="Guarantees 40% cost reduction for Black Mesa.",
        suggested_subject_line="Resonance dampening",
        cta="15-minute briefing next week?",
        tone_guidance="Scientific peer",
        constraint_checklist=[],
    )
    builder = StrategyBuilder(engine=MockEngine(raw))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    arg = result.strategy.toulmin_argument
    assert arg.backing.has_whitelisted_proof is False
    assert "40%" not in arg.claim.text
    assert "guarantee" not in arg.claim.text.lower()


def test_contradictions_and_exclusions_are_represented_using_valid_references(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-rebuttal-test", "Thermal Surge", ["fact-lambda-expansion"])

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    rebuttal = result.strategy.toulmin_argument.rebuttal
    assert "Unshielded radioactive waste dumps" in rebuttal.hard_exclusions
    assert "Unshielded antimatter reactors" in rebuttal.hard_exclusions


def test_missing_backing_is_represented_explicitly_without_automatically_invalidating_an_otherwise_viable_strategy(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-no-backing", "Resonance Cascade", ["fact-thermal-containment"], target_cap="Resonance Dampening Array")

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    arg = result.strategy.toulmin_argument
    assert arg.backing.has_whitelisted_proof is False
    assert arg.rebuttal.missing_backing is True
    assert any("backing proof is unavailable" in c for c in result.strategy.constraint_checklist)


def test_cta_selection_follows_the_deterministic_policy(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    builder = StrategyBuilder(engine=MockEngine(None))

    # Case 1: Verified-compatible signal -> VALUE_EXCHANGE
    sig_verified = make_ranked_signal("sig-v", "Thermal Event", ["fact-lambda-expansion"], is_conditional=False)
    cta_v, text_v, rat_v = builder.evaluate_cta_policy(sig_verified, is_conditional=False, unresolved_conditions=[], outreach=outreach)
    assert cta_v == CTAType.VALUE_EXCHANGE

    # Case 2: Conditional signal with 1 prerequisite -> BINARY_CHECK
    sig_single = make_ranked_signal("sig-c1", "Thermal Event", ["fact-lambda-expansion"], is_conditional=True, unresolved_conditions=["Liquid nitrogen pipeline"])
    cta_c1, text_c1, rat_c1 = builder.evaluate_cta_policy(sig_single, is_conditional=True, unresolved_conditions=["Liquid nitrogen pipeline"], outreach=outreach)
    assert cta_c1 == CTAType.BINARY_CHECK
    assert "Liquid nitrogen pipeline" in text_c1

    # Case 3: Conditional signal with 2 prerequisites -> DIAGNOSTIC_QUESTION
    sig_multi = make_ranked_signal("sig-c2", "Thermal Event", ["fact-lambda-expansion"], is_conditional=True, unresolved_conditions=["Liquid nitrogen pipeline", "Cryo-compressor"])
    cta_c2, text_c2, rat_c2 = builder.evaluate_cta_policy(sig_multi, is_conditional=True, unresolved_conditions=["Liquid nitrogen pipeline", "Cryo-compressor"], outreach=outreach)
    assert cta_c2 == CTAType.DIAGNOSTIC_QUESTION


def test_unsupported_high_commitment_ctas_are_replaced(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    high_commitment_outreach = OutreachContext(
        objective="Book a demo",
        sender_identity="Dr. Isaac Kleiner",
        desired_cta="Schedule a 45-minute live demonstration and formal architectural audit",
        tone="peer-to-peer scientific",
    )
    sig_conditional = make_ranked_signal(
        "sig-high-cta",
        "Thermal Event",
        ["fact-lambda-expansion"],
        is_conditional=True,
        unresolved_conditions=["Liquid nitrogen pipeline"],
    )

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[sig_conditional],
        prospect=prospect,
        product=product,
        outreach=high_commitment_outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    assert result.strategy.cta_type == CTAType.BINARY_CHECK
    assert result.strategy.cta != high_commitment_outreach.desired_cta
    assert "Liquid nitrogen pipeline" in result.strategy.cta


def test_invalid_essential_references_cause_strategy_abstention(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    builder = StrategyBuilder(engine=MockEngine(None))

    # Test A: Non-existent fact reference causes abstention
    sig_bad_fact = make_ranked_signal("sig-bad-fact", "Nonexistent Fact", ["fact-unknown-999"])
    res_a = builder.build_strategy(
        ranked_signals=[sig_bad_fact],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )
    assert res_a.is_viable is False

    # Test B: Undeclared capability causes abstention
    sig_bad_cap = make_ranked_signal("sig-bad-cap", "Undeclared Cap", ["fact-lambda-expansion"], target_cap="Dark Matter Fusion Reactor")
    res_b = builder.build_strategy(
        ranked_signals=[sig_bad_cap],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )
    assert res_b.is_viable is False


def test_empty_optional_toulmin_components_are_handled_correctly(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-empty-opt", "Thermal Surge", ["fact-lambda-expansion"])

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    arg = result.strategy.toulmin_argument
    # Verify that optional lists and fields serialize cleanly
    dumped = arg.model_dump()
    assert isinstance(dumped, dict)
    reconstructed = ToulminArgument.model_validate(dumped)
    assert reconstructed.warrant.capability_id == arg.warrant.capability_id


def test_existing_strategy_consumers_remain_compatible_with_transition_adapter(toulmin_fixtures):
    prospect, product, outreach, pkg, cap1, cap2 = toulmin_fixtures
    ranked = make_ranked_signal("sig-adapter-compat", "Thermal Surge", ["fact-lambda-expansion"])

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
        research_package=pkg,
    )

    assert result.is_viable is True
    # Test StrategyResult property delegates
    assert result.toulmin_argument is not None
    assert result.toulmin_argument == result.strategy.toulmin_argument
    assert result.allowed_cta_friction == result.strategy.cta_type.value
    assert result.target_capability_id == result.strategy.toulmin_argument.warrant.capability_id

    # Test legacy OutreachStrategy fields
    strat = result.strategy
    assert isinstance(strat.central_message, str) and len(strat.central_message) > 0
    assert isinstance(strat.cta, str) and len(strat.cta) > 0
    assert strat.cta_type == CTAType.VALUE_EXCHANGE
    assert isinstance(strat.cta_rationale, str)
    assert len(strat.constraint_checklist) > 0
    assert strat.messaging_approach == "event-hook"
