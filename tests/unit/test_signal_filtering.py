from __future__ import annotations

import datetime
from typing import Any
import pytest

from src.common.provenance import ProvenanceOrigin
from src.context.models import (
    CapabilityNode,
    CompatibilityResult,
    CompatibilityStatus,
    ConstraintType,
    ProductConstraint,
    ProductContract,
    ProductContext,
    ProspectContext,
    OutreachContext,
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
    VerificationStatus,
)
from src.signal_strategy.eligibility import SignalEligibilityEvaluator
from src.signal_strategy.models import (
    CandidateSignal,
    EligibilityStatus,
    GateIdentifier,
    SignalEligibility,
)
from src.signal_strategy.ranker import (
    RawEvaluatedSignal,
    RawSignalEvaluationOutput,
    SignalRanker,
)
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.signal_strategy.strategy_builder import (
    RawStrategyOutput,
    StrategyBuilder,
)
from src.signal_strategy.service import SignalStrategyService
from src.signal_strategy.identifier import (
    RawCandidateSignal,
    RawSignalIdentificationOutput,
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


class MultiMockEngine(GeneratorEngine):
    def __init__(self, responses: list[Any]):
        super().__init__(default_provider="mock")
        self.responses = list(responses)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        output = self.responses.pop(0) if self.responses else None
        return GenerationResponse(
            text="{}",
            provider_name="mock",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=output,
        )


def _build_test_contract(
    is_exhaustive: bool = False,
    hard_exclusions: list[str] | None = None,
) -> ProductContract:
    """Helper to construct a deterministic ProductContract for unit tests."""
    return ProductContract(
        product_name="CloudAudit Pro",
        contract_version="2.1.0",
        description="Automated continuous cloud compliance auditing and remediation",
        value_proposition="Reduces SOC2 audit prep time from months to continuous real-time monitoring.",
        capabilities=[
            CapabilityNode(
                name="Continuous SOC2 Monitoring",
                description="Real-time scanning of AWS/GCP infrastructure for SOC2 controls",
                supported_use_cases=["SOC2 Type II prep", "Cloud security audit"],
                required_prerequisites=["aws_or_gcp_cloud_environment"],
                target_operational_problems=["audit evidence manual collection"],
                target_buyer_personas=["Head of Security", "CISO", "VP Engineering"],
            ),
            CapabilityNode(
                name="Kubernetes RBAC Enforcement",
                description="Cluster-level RBAC role policy auditing and enforcement",
                supported_use_cases=["K8s security", "Container compliance"],
                required_prerequisites=["kubernetes_cluster_v1_24_plus"],
                target_operational_problems=["privilege escalation in containers"],
                target_buyer_personas=["DevOps Lead", "Head of Infrastructure"],
            ),
        ],
        global_constraints=[
            ProductConstraint(
                name="cloud_only",
                constraint_type=ConstraintType.SCOPE_LIMITATION,
                description="Requires cloud-native infrastructure; on-premise legacy mainframes unsupported",
            )
        ],
        hard_exclusions=hard_exclusions or ["on-premise mainframe", "cryptocurrency mining"],
        is_exhaustive=is_exhaustive,
    )


@pytest.fixture
def prospect() -> ProspectContext:
    return ProspectContext(
        full_name="Sarah Chen",
        company_name="Fintech Velocity",
        company_domain="fintechvelocity.io",
        designation="CISO",
    )


@pytest.fixture
def outreach() -> OutreachContext:
    return OutreachContext(
        objective="Schedule technical discovery on automated SOC2 audit prep",
        sender_identity="Compliance Architect at CloudAudit",
        desired_cta="Open to a 15-minute briefing next week?",
        tone="professional peer",
        constraints=["Keep under 100 words"],
    )


# ==============================================================================
# Scenario 1: Compatible signal passes product gate
# ==============================================================================

def test_compatible_signal_passes_product_gate(prospect: ProspectContext):
    """Signal matching contract capability with satisfied prerequisites passes product gate."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="fact-1",
        claim_text="Fintech Velocity operates its core payment stack on aws_or_gcp_cloud_environment",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Fintech Velocity migrates to AWS")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-soc2",
        label="SOC2 Audit Preparation",
        description="Fintech Velocity announced upcoming SOC2 Type II audit",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-1"],
        fact_category=FactCategory.STRATEGY,
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(
        signal=sig,
        facts=[fact],
        prospect=prospect,
        product=product,
    )

    assert eligibility.status == EligibilityStatus.ELIGIBLE
    assert eligibility.is_eligible is True
    assert eligibility.product_gate.passed is True
    assert eligibility.compatibility_result is not None
    assert eligibility.compatibility_result.status == CompatibilityStatus.VERIFIED_COMPATIBLE
    assert eligibility.failed_gates == []
    assert eligibility.unresolved_conditions == []


# ==============================================================================
# Scenario 2: Hard exclusion vetoes signal (composite_score = 0.0, rejected)
# ==============================================================================

def test_hard_exclusion_vetoes_signal(prospect: ProspectContext):
    """Signal involving a hard-excluded technology is rejected by product gate with score 0.0."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="fact-crypto",
        claim_text="Fintech Velocity deployed on-premise mainframe to handle legacy batch records",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Mainframe deployed")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-mainframe",
        label="Mainframe Deployment",
        description="Fintech Velocity deployed on-premise mainframe infrastructure",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-crypto"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=product)

    assert eligibility.status == EligibilityStatus.INELIGIBLE
    assert eligibility.is_eligible is False
    assert eligibility.product_gate.passed is False
    assert GateIdentifier.PRODUCT_COMPATIBILITY in eligibility.failed_gates
    assert eligibility.compatibility_result.status == CompatibilityStatus.INCOMPATIBLE
    assert "on-premise mainframe" in eligibility.compatibility_result.violated_exclusions

    # In ranker, composite score must be strictly 0.0
    ranker = SignalRanker(eligibility_evaluator=evaluator)
    ranked = ranker.rank_signals([sig], [fact], prospect, product)
    assert len(ranked) == 1
    assert ranked[0].score.composite_score == 0.0
    assert "Ineligible for outreach" in ranked[0].rationale


# ==============================================================================
# Scenario 3: Unsatisfied prerequisite vetoes signal
# ==============================================================================

def test_unsatisfied_prerequisite_vetoes_signal(prospect: ProspectContext):
    """Directly conflicting or unsatisfied prerequisite vetoes the signal."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Kubernetes RBAC Enforcement"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    # Fact confirms environment does NOT meet prerequisite
    fact = CanonicalFact(
        fact_id="fact-k8s",
        claim_text="Fintech Velocity operates bare-metal Docker without kubernetes_cluster_v1_24_plus",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Docker standalone without k8s")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-rbac",
        label="RBAC Cluster Enforcement",
        description="Fintech Velocity auditing container access",
        target_capability="Kubernetes RBAC Enforcement",
        supporting_fact_ids=["fact-k8s"],
    )

    # When evaluate_compatibility detects missing prerequisites, status is INCOMPATIBLE
    compat = contract.evaluate_compatibility(
        required_capability="Kubernetes RBAC Enforcement",
        known_conditions={"kubernetes_cluster_v1_24_plus": False},
    )
    assert compat.status == CompatibilityStatus.INCOMPATIBLE
    assert "kubernetes_cluster_v1_24_plus" in compat.missing_prerequisites


# ==============================================================================
# Scenario 4: Unknown prerequisite produces conditional compatibility
# ==============================================================================

def test_unknown_prerequisite_produces_conditional_compatibility(prospect: ProspectContext):
    """Unverified prerequisite allows signal to pass as CONDITIONAL with unresolved conditions."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    # Fact does NOT mention cloud environment (aws_or_gcp_cloud_environment is unknown)
    fact = CanonicalFact(
        fact_id="fact-expansion",
        claim_text="Fintech Velocity expanding engineering team by 50 engineers",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Fintech Velocity hiring spree")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-expansion",
        label="Engineering Team Expansion",
        description="Fintech Velocity expanding engineering team",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-expansion"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=product)

    assert eligibility.status == EligibilityStatus.CONDITIONAL
    assert eligibility.is_eligible is False
    assert eligibility.is_verified_eligible is False
    assert eligibility.requires_diagnostic_strategy is True
    assert eligibility.is_ineligible is False
    assert eligibility.product_gate.passed is True
    assert eligibility.compatibility_result.status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
    assert "aws_or_gcp_cloud_environment" in eligibility.unresolved_conditions

    # In ranker, product_alignment is preserved as None without fabrication
    ranker = SignalRanker(eligibility_evaluator=evaluator)
    ranked = ranker.rank_signals([sig], [fact], prospect, product)
    assert len(ranked) == 1
    assert ranked[0].score.product_alignment is None
    # Because it is conditional, composite score is computed using neutral weight, NOT capped at 0.20
    assert ranked[0].score.composite_score >= 0.35


# ==============================================================================
# Scenario 5: Undeclared capability in non-exhaustive contract is conditional
# ==============================================================================

def test_undeclared_capability_in_non_exhaustive_contract_is_conditional(prospect: ProspectContext):
    """Undeclared capability in non-exhaustive contract passes as CONDITIONAL."""
    contract = _build_test_contract(is_exhaustive=False)
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="fact-pci",
        claim_text="Fintech Velocity requires PCI-DSS payment tokenization certification",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="PCI-DSS tokenization audit")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-pci",
        label="PCI-DSS Tokenization",
        description="Fintech Velocity pursuing PCI-DSS payment compliance",
        target_capability="Automated PCI-DSS Tokenization",
        supporting_fact_ids=["fact-pci"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=product)

    assert eligibility.status == EligibilityStatus.CONDITIONAL
    assert eligibility.is_eligible is False
    assert eligibility.is_verified_eligible is False
    assert eligibility.requires_diagnostic_strategy is True
    assert eligibility.is_ineligible is False
    assert eligibility.product_gate.passed is True
    assert any("product_support_for_Automated PCI-DSS Tokenization" in cond for cond in eligibility.unresolved_conditions)


# ==============================================================================
# Scenario 6: Undeclared capability in exhaustive contract is rejected
# ==============================================================================

def test_undeclared_capability_in_exhaustive_contract_is_rejected(prospect: ProspectContext):
    """Closed-world enforcement: undeclared capability in exhaustive contract is strictly INCOMPATIBLE."""
    contract = _build_test_contract(is_exhaustive=True)
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="fact-pci",
        claim_text="Fintech Velocity requires PCI-DSS payment tokenization",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="PCI-DSS tokenization")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-pci",
        label="PCI-DSS Tokenization",
        description="Fintech Velocity pursuing PCI-DSS compliance",
        target_capability="Automated PCI-DSS Tokenization",
        supporting_fact_ids=["fact-pci"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=product)

    assert eligibility.status == EligibilityStatus.INELIGIBLE
    assert eligibility.is_eligible is False
    assert eligibility.product_gate.passed is False
    assert eligibility.compatibility_result.status == CompatibilityStatus.INCOMPATIBLE
    assert "not supported under exhaustive product contract" in eligibility.compatibility_result.reason

    ranker = SignalRanker(eligibility_evaluator=evaluator)
    ranked = ranker.rank_signals([sig], [fact], prospect, product)
    assert ranked[0].score.composite_score == 0.0


# ==============================================================================
# Scenario 7: High recency (1.0) cannot rescue an incompatible signal
# ==============================================================================

def test_high_recency_cannot_rescue_incompatible_signal(prospect: ProspectContext):
    """Non-compensatory rule: recency score of 1.0 cannot compensate for failed product gate."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    today = datetime.date.today()
    fact = CanonicalFact(
        fact_id="fact-today",
        claim_text="Fintech Velocity announced immediate migration to on-premise mainframe today",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[
            EvidenceRecord(source_id="s1", verbatim_excerpt="Today mainframe announcement"),
            EvidenceRecord(source_id="s2", verbatim_excerpt="Confirmed deployment"),
        ],
        primary_event_date=today,
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-recent-incompat",
        label="Breaking Mainframe Migration",
        description="Fintech Velocity deployed on-premise mainframe infrastructure today",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-today"],
        event_date=today,
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig], [fact], prospect, product)

    assert len(ranked) == 1
    # Recency score was computed as 1.0
    assert ranked[0].score.recency_score == 1.0
    # But composite score is strictly 0.0 due to non-compensatory veto
    assert ranked[0].score.composite_score == 0.0
    assert ranked[0].eligibility.status == EligibilityStatus.INELIGIBLE


# ==============================================================================
# Scenario 8: High commercial significance (0.99) cannot rescue evidence-gate failure
# ==============================================================================

def test_high_commercial_significance_cannot_rescue_evidence_gate_failure(prospect: ProspectContext):
    """Non-compensatory rule: massive commercial impact cannot rescue ungrounded fact."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    # Signal cites a fact ID that does NOT exist in research
    sig = CandidateSignal(
        signal_id="sig-phantom",
        label="Hundred Million Dollar Expansion",
        description="Fintech Velocity raised $100M Series D to conquer European cloud banking",
        commercial_opportunity="Massive enterprise compliance opportunity",
        supporting_fact_ids=["non-existent-fact-999"],
        event_date=datetime.date.today(),
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[], prospect=prospect, product=product)

    assert eligibility.status == EligibilityStatus.INELIGIBLE
    assert eligibility.evidence_gate.passed is False
    assert GateIdentifier.EVIDENCE in eligibility.failed_gates

    ranker = SignalRanker(eligibility_evaluator=evaluator)
    ranked = ranker.rank_signals([sig], [], prospect, product)
    assert ranked[0].score.composite_score == 0.0
    assert "non-existent research fact IDs" in ranked[0].rationale


# ==============================================================================
# Scenario 9: Conditional compatibility produces qualified claim & non-assertion constraint
# ==============================================================================

def test_conditional_compatibility_produces_qualified_claim(prospect: ProspectContext, outreach: OutreachContext):
    """StrategyBuilder qualifies claims and adds non-assertion constraints for conditional signals."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="fact-team",
        claim_text="Fintech Velocity doubled engineering team",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Engineering doubled")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-cond",
        label="Engineering Team Scaling",
        description="Fintech Velocity expanded engineering headcount",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-team"],
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig], [fact], prospect, product)
    assert ranked[0].eligibility.status == EligibilityStatus.CONDITIONAL

    builder = StrategyBuilder()
    result = builder.build_strategy(
        ranked_signals=ranked,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is True
    assert result.strategy is not None
    strategy = result.strategy
    assert strategy.is_conditional is True
    assert "aws_or_gcp_cloud_environment" in strategy.unresolved_conditions
    assert strategy.alignment.is_conditional is True

    # Must contain explicit non-assertion constraint
    constraint_text = " ".join(strategy.constraint_checklist)
    assert "Do not assert verified support" in constraint_text
    assert "aws_or_gcp_cloud_environment" in constraint_text
    assert "exploratory inquiry" in constraint_text

    # Diagnostic messaging approach
    assert strategy.messaging_approach == "diagnostic"
    assert "Exploring whether" in strategy.central_message


# ==============================================================================
# Scenario 10: Ineligible signals cannot reach strategy selection
# ==============================================================================

def test_ineligible_signals_cannot_reach_strategy_selection(prospect: ProspectContext, outreach: OutreachContext):
    """Even if an ineligible signal has a high ranker input score, StrategyBuilder selects the top eligible signal."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact_bad = CanonicalFact(
        fact_id="fact-bad",
        claim_text="Fintech Velocity acquired on-premise mainframe legacy center",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Mainframe acquisition")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    fact_good = CanonicalFact(
        fact_id="fact-good",
        claim_text="Fintech Velocity operates aws_or_gcp_cloud_environment infrastructure and announced SOC2 prep",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s2", verbatim_excerpt="AWS SOC2 prep")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig_incompatible = CandidateSignal(
        signal_id="sig-bad",
        label="Mainframe Legacy Acquisition",
        description="Mainframe acquisition",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-bad"],
    )
    sig_compatible = CandidateSignal(
        signal_id="sig-good",
        label="AWS SOC2 Prep",
        description="AWS SOC2 prep",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-good"],
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals(
        signals=[sig_incompatible, sig_compatible],
        facts=[fact_bad, fact_good],
        prospect=prospect,
        product=product,
    )

    # Compatible signal must be ranked #1
    assert ranked[0].signal.signal_id == "sig-good"
    assert ranked[0].eligibility.is_eligible is True
    assert ranked[1].signal.signal_id == "sig-bad"
    assert ranked[1].score.composite_score == 0.0

    builder = StrategyBuilder()
    result = builder.build_strategy(ranked_signals=ranked, prospect=prospect, product=product, outreach=outreach)
    assert result.is_viable is True
    assert result.strategy.primary_signal.signal.signal_id == "sig-good"


# ==============================================================================
# Scenario 11: Strategy abstains when no eligible signals exist
# ==============================================================================

def test_strategy_abstains_when_no_eligible_signals(prospect: ProspectContext, outreach: OutreachContext):
    """When all candidate signals fail eligibility gates, StrategyResult is unviable."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="fact-mf",
        claim_text="Fintech Velocity deployed on-premise mainframe system",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Mainframe only")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    sig = CandidateSignal(
        signal_id="sig-mf",
        label="Mainframe Deployment",
        description="Fintech Velocity on-premise mainframe",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["fact-mf"],
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig], [fact], prospect, product)
    assert ranked[0].eligibility.is_eligible is False

    builder = StrategyBuilder()
    result = builder.build_strategy(ranked_signals=ranked, prospect=prospect, product=product, outreach=outreach)

    assert result.is_viable is False
    assert result.strategy is None
    assert result.provenance.rule_id == "no_eligible_signals_abstention"
    assert len(result.unviable_reasons) > 0


# ==============================================================================
# Scenario 12: Decision traceability and serialization
# ==============================================================================

def test_decision_traceability_and_serialization(prospect: ProspectContext):
    """SignalEligibility contains complete four-gate audit trail and serializes losslessly to JSON."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="f1",
        claim_text="Fintech Velocity uses aws_or_gcp_cloud_environment for cloud payments",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Cloud payment")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    sig = CandidateSignal(
        signal_id="sig-audit",
        label="SOC2 Audit",
        description="SOC2 audit",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f1"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=product)

    # Verify all four gates recorded
    assert eligibility.identity_gate.gate == GateIdentifier.IDENTITY
    assert eligibility.identity_gate.passed is True
    assert eligibility.evidence_gate.gate == GateIdentifier.EVIDENCE
    assert eligibility.evidence_gate.passed is True
    assert eligibility.product_gate.gate == GateIdentifier.PRODUCT_COMPATIBILITY
    assert eligibility.product_gate.passed is True
    assert eligibility.persona_gate.gate == GateIdentifier.PERSONA
    assert eligibility.persona_gate.passed is True

    # JSON serialization and deserialization roundtrip
    json_str = eligibility.model_dump_json()
    reconstructed = SignalEligibility.model_validate_json(json_str)

    assert reconstructed.status == eligibility.status
    assert reconstructed.identity_gate.reason == eligibility.identity_gate.reason
    assert reconstructed.evidence_gate.details == eligibility.evidence_gate.details
    assert reconstructed.compatibility_result.status == eligibility.compatibility_result.status
    assert reconstructed.provenance.rule_id == "lexicographic_four_gate_eval"


# ==============================================================================
# Scenario 13: Legacy ProductContext input compatibility
# ==============================================================================

def test_legacy_product_context_input_compatibility(prospect: ProspectContext, outreach: OutreachContext):
    """Pipeline operates seamlessly when passed legacy ProductContext without explicit contract."""
    legacy_product = ProductContext(
        product_name="LegacyCloudApp",
        description="Legacy cloud monitoring",
        capabilities=["Automated Alerting", "Log Aggregation"],
        value_proposition="Real-time alerts",
        limitations=["Requires cloud log forwarder"],
    )
    assert legacy_product.contract is None

    fact = CanonicalFact(
        fact_id="f1",
        claim_text="Fintech Velocity deployed cloud log forwarder across all nodes",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Log forwarder installed")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    sig = CandidateSignal(
        signal_id="sig-alert",
        label="Log Infrastructure Upgrade",
        description="Fintech Velocity upgraded log ingestion pipeline",
        target_capability="Automated Alerting",
        supporting_fact_ids=["f1"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=legacy_product)
    assert eligibility.is_eligible is True

    raw_eval = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="sig-alert",
                prospect_relevance=0.85,
                product_alignment=0.80,
                business_significance=0.75,
                rationale="Matches automated alerting and log aggregation capabilities.",
            )
        ]
    )
    ranker = SignalRanker(engine=MockEngine(raw_eval), eligibility_evaluator=evaluator)
    ranked = ranker.rank_signals([sig], [fact], prospect, legacy_product)
    assert len(ranked) == 1
    assert ranked[0].score.product_alignment == 0.80

    builder = StrategyBuilder()
    result = builder.build_strategy(
        ranked_signals=ranked,
        prospect=prospect,
        product=legacy_product,
        outreach=outreach,
    )
    assert result.is_viable is True
    assert result.strategy is not None


# ==============================================================================
# Milestone 1.2.1: 13 Adversarial Tests
# ==============================================================================

def test_adversarial_1_conditional_signals_reach_diagnostic_strategy_path(prospect: ProspectContext, outreach: OutreachContext):
    """1. Conditional signals can reach the diagnostic strategy path and produce exploratory inquiries."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    # Prerequisite aws_or_gcp_cloud_environment is unknown (not mentioned in claim)
    fact = CanonicalFact(
        fact_id="f-unknown",
        claim_text="Fintech Velocity announced upcoming SOC2 compliance initiative",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="SOC2 initiative announced")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    sig = CandidateSignal(
        signal_id="sig-soc2-unknown",
        label="SOC2 Compliance Initiative",
        description="Upcoming compliance effort",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-unknown"],
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig], [fact], prospect, product)

    assert len(ranked) == 1
    assert ranked[0].eligibility.requires_diagnostic_strategy is True
    assert ranked[0].eligibility.is_verified_eligible is False
    assert ranked[0].score.product_alignment is None

    builder = StrategyBuilder()
    result = builder.build_strategy(ranked_signals=ranked, prospect=prospect, product=product, outreach=outreach)

    assert result.is_viable is True
    assert result.strategy is not None
    assert result.strategy.is_conditional is True
    assert result.strategy.messaging_approach == "diagnostic"
    assert "aws_or_gcp_cloud_environment" in result.strategy.unresolved_conditions
    assert any("Do not assert verified support" in c for c in result.strategy.constraint_checklist)


def test_adversarial_2_conditional_signals_cannot_reach_normal_assertion_path(prospect: ProspectContext, outreach: OutreachContext):
    """2. Conditional signals cannot reach the normal assertion-based strategy path under any circumstances."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="f-cond",
        claim_text="Fintech Velocity announced SOC2 preparation without detailing infrastructure",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="SOC2 prep")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    sig = CandidateSignal(
        signal_id="sig-cond",
        label="SOC2 Preparation",
        description="SOC2 prep",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-cond"],
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig], [fact], prospect, product)

    assert ranked[0].eligibility.is_verified_eligible is False

    builder = StrategyBuilder()
    verified_candidates = [s for s in ranked if s.eligibility and s.eligibility.is_verified_eligible]
    assert len(verified_candidates) == 0

    result = builder.build_strategy(ranked_signals=ranked, prospect=prospect, product=product, outreach=outreach)
    assert result.strategy.is_conditional is True


def test_adversarial_3_verified_signals_take_precedence_over_conditional_signals(prospect: ProspectContext, outreach: OutreachContext):
    """3. Verified-compatible signals take precedence over conditional signals under normal selection policy."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact_verified = CanonicalFact(
        fact_id="f-ver",
        claim_text="Fintech Velocity operates aws_or_gcp_cloud_environment infrastructure and completed audit scope",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="aws infrastructure verified")],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    fact_cond = CanonicalFact(
        fact_id="f-cond",
        claim_text="Fintech Velocity planning SOC2 audit without cloud environment specified",
        fact_category=FactCategory.STRATEGY,
        evidence_records=[EvidenceRecord(source_id="s2", verbatim_excerpt="SOC2 planning")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig_verified = CandidateSignal(
        signal_id="sig-ver",
        label="Verified SOC2 Monitoring",
        description="Verified SOC2 monitoring",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-ver"],
    )
    sig_cond = CandidateSignal(
        signal_id="sig-cond",
        label="Conditional SOC2 Planning",
        description="SOC2 planning",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-cond"],
    )

    raw_eval = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="sig-ver",
                prospect_relevance=0.80,
                product_alignment=0.85,
                business_significance=0.70,
                rationale="Verified Continuous SOC2 Monitoring capability match",
            ),
            RawEvaluatedSignal(
                signal_id="sig-cond",
                prospect_relevance=0.99,
                product_alignment=0.99,
                business_significance=0.99,
                rationale="Highly relevant but infrastructure prerequisite unknown",
            ),
        ]
    )

    ranker = SignalRanker(engine=MockEngine(raw_eval))
    ranked = ranker.rank_signals([sig_cond, sig_verified], [fact_verified, fact_cond], prospect, product)

    assert ranked[0].signal.signal_id == "sig-ver"
    assert ranked[0].eligibility.is_verified_eligible is True
    assert ranked[1].signal.signal_id == "sig-cond"
    assert ranked[1].eligibility.requires_diagnostic_strategy is True

    builder = StrategyBuilder()
    result = builder.build_strategy(ranked_signals=ranked, prospect=prospect, product=product, outreach=outreach)
    assert result.is_viable is True
    assert result.strategy.is_conditional is False
    assert result.strategy.primary_signal.signal.signal_id == "sig-ver"


def test_adversarial_4_high_recency_cannot_promote_conditional_signal_into_verified_pool(prospect: ProspectContext):
    """4. High recency (1.0) cannot promote conditional signals into the verified-compatible pool."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    today = datetime.date.today()
    fact_verified_old = CanonicalFact(
        fact_id="f-old",
        claim_text="Fintech Velocity operates aws_or_gcp_cloud_environment infrastructure for years",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="aws cloud")],
        primary_event_date=today - datetime.timedelta(days=200),
        composite_confidence=ConfidenceLevel.HIGH,
    )
    fact_cond_today = CanonicalFact(
        fact_id="f-new",
        claim_text="Fintech Velocity launched massive SOC2 project today without infrastructure disclosure",
        evidence_records=[EvidenceRecord(source_id="s2", verbatim_excerpt="today")],
        primary_event_date=today,
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig_verified_old = CandidateSignal(
        signal_id="sig-ver-old",
        label="Verified SOC2 Monitoring",
        description="Verified SOC2 monitoring",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-old"],
        event_date=fact_verified_old.primary_event_date,
    )
    sig_cond_today = CandidateSignal(
        signal_id="sig-cond-today",
        label="Today SOC2 Project",
        description="today soc2",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-new"],
        event_date=fact_cond_today.primary_event_date,
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig_cond_today, sig_verified_old], [fact_verified_old, fact_cond_today], prospect, product)

    assert ranked[0].signal.signal_id == "sig-ver-old"
    assert ranked[0].eligibility.is_verified_eligible is True
    assert ranked[1].signal.signal_id == "sig-cond-today"
    assert ranked[1].score.recency_score == 1.0
    assert ranked[1].eligibility.requires_diagnostic_strategy is True
    assert ranked[1].score.product_alignment is None


def test_adversarial_5_high_business_significance_cannot_promote_conditional_signal_into_verified_pool(prospect: ProspectContext):
    """5. High business significance (1.0) cannot promote conditional signals into the verified-compatible pool."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact_ver = CanonicalFact(
        fact_id="f-ver",
        claim_text="Fintech Velocity operates aws_or_gcp_cloud_environment infrastructure",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="aws")],
    )
    fact_cond = CanonicalFact(
        fact_id="f-cond",
        claim_text="Fintech Velocity announced $100M Series D round and SOC2 preparation without infrastructure details",
        evidence_records=[EvidenceRecord(source_id="s2", verbatim_excerpt="funding")],
    )

    sig_ver = CandidateSignal(
        signal_id="sig-ver",
        label="Verified SOC2 Monitoring",
        description="Verified SOC2 monitoring",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-ver"],
    )
    sig_cond = CandidateSignal(
        signal_id="sig-cond",
        label="Series D SOC2 Initiative",
        description="series d",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-cond"],
    )

    raw_eval = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="sig-ver",
                prospect_relevance=0.60,
                product_alignment=0.70,
                business_significance=0.50,
                rationale="Moderate significance verified capability match",
            ),
            RawEvaluatedSignal(
                signal_id="sig-cond",
                prospect_relevance=1.0,
                product_alignment=1.0,
                business_significance=1.0,  # Maximum commercial magnitude
                rationale="Massive $100M round but infrastructure is unspecified",
            ),
        ]
    )

    ranker = SignalRanker(engine=MockEngine(raw_eval))
    ranked = ranker.rank_signals([sig_cond, sig_ver], [fact_ver, fact_cond], prospect, product)

    assert ranked[0].signal.signal_id == "sig-ver"
    assert ranked[0].eligibility.is_verified_eligible is True
    assert ranked[1].signal.signal_id == "sig-cond"
    assert ranked[1].score.business_significance == 1.0
    assert ranked[1].score.product_alignment is None
    assert ranked[1].eligibility.requires_diagnostic_strategy is True


def test_adversarial_6_ineligible_signals_cannot_enter_either_ranking_pool(prospect: ProspectContext):
    """6. Ineligible signals cannot enter either ranking pool (verified or conditional)."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact_bad = CanonicalFact(
        fact_id="f-bad",
        claim_text="Fintech Velocity deployed on-premise mainframe legacy core",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="mainframe")],
    )
    sig_bad = CandidateSignal(
        signal_id="sig-bad",
        label="Mainframe Migration",
        description="on-premise mainframe",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-bad"],
    )

    ranker = SignalRanker()
    ranked = ranker.rank_signals([sig_bad], [fact_bad], prospect, product)

    assert len(ranked) == 1
    assert ranked[0].eligibility.is_ineligible is True
    assert ranked[0].score.composite_score == 0.0
    assert ranked[0].score.prospect_relevance == 0.0
    assert ranked[0].score.business_significance == 0.0


def test_adversarial_7_ineligible_candidates_retained_for_audit_do_not_influence_ranking(prospect: ProspectContext):
    """7. Ineligible candidates retained for audit do not influence ranking of eligible signals."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact_ver = CanonicalFact(
        fact_id="f-ver",
        claim_text="Fintech Velocity operates aws_or_gcp_cloud_environment infrastructure",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="aws")],
    )
    fact_cond = CanonicalFact(
        fact_id="f-cond",
        claim_text="Fintech Velocity planning SOC2 audit without infrastructure details",
        evidence_records=[EvidenceRecord(source_id="s2", verbatim_excerpt="soc2")],
    )
    fact_ineligible = CanonicalFact(
        fact_id="f-inel",
        claim_text="Fintech Velocity acquired on-premise mainframe legacy core",
        evidence_records=[EvidenceRecord(source_id="s3", verbatim_excerpt="on-premise mainframe")],
    )

    sig_ver = CandidateSignal(
        signal_id="sig-ver",
        label="Verified SOC2 Monitoring",
        description="Verified SOC2 monitoring",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-ver"],
    )
    sig_cond = CandidateSignal(
        signal_id="sig-cond",
        label="Conditional SOC2",
        description="SOC2 audit",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-cond"],
    )
    sig_inel = CandidateSignal(
        signal_id="sig-inel",
        label="Mainframe Core",
        description="on-premise mainframe",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-inel"],
    )

    ranker = SignalRanker()
    ranked_clean = ranker.rank_signals([sig_ver, sig_cond], [fact_ver, fact_cond], prospect, product)
    ranked_with_ineligible = ranker.rank_signals(
        [sig_inel, sig_ver, sig_cond], [fact_ver, fact_cond, fact_ineligible], prospect, product
    )

    assert ranked_clean[0].signal.signal_id == ranked_with_ineligible[0].signal.signal_id == "sig-ver"
    assert ranked_clean[0].score.composite_score == ranked_with_ineligible[0].score.composite_score
    assert ranked_clean[1].signal.signal_id == ranked_with_ineligible[1].signal.signal_id == "sig-cond"
    assert ranked_clean[1].score.composite_score == ranked_with_ineligible[1].score.composite_score
    assert ranked_with_ineligible[2].signal.signal_id == "sig-inel"
    assert ranked_with_ineligible[2].score.composite_score == 0.0


def test_adversarial_8_legacy_flat_string_product_inputs_use_deterministic_contract_eval(prospect: ProspectContext):
    """8. Legacy flat-string product inputs use deterministic contract evaluation."""
    legacy_product = ProductContext(
        product_name="LegacyGuard",
        description="Legacy flat string product",
        capabilities=["Database Auditing"],
        value_proposition="Real-time db auditing",
        limitations=["on-premise"],
    )
    assert legacy_product.contract is None

    fact = CanonicalFact(
        fact_id="f1",
        claim_text="Fintech Velocity deployed on-premise oracle database",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="on-premise oracle")],
    )
    sig = CandidateSignal(
        signal_id="sig-db",
        label="Database Audit",
        description="on-premise db",
        target_capability="Database Auditing",
        supporting_fact_ids=["f1"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=legacy_product)

    assert eligibility.status == EligibilityStatus.INELIGIBLE
    assert eligibility.is_ineligible is True
    assert eligibility.product_gate.passed is False
    assert "on-premise" in eligibility.compatibility_result.violated_exclusions


def test_adversarial_9_structured_and_legacy_representations_produce_consistent_decisions(prospect: ProspectContext):
    """9. Structured and legacy representations produce consistent compatibility decisions."""
    structured_contract = ProductContract(
        product_name="TestProduct",
        description="Test description",
        capabilities=[CapabilityNode(capability_id="c1", name="Data Encryption")],
        value_proposition="Data encryption",
        hard_exclusions=["mainframe"],
    )
    structured_product = ProductContext(
        product_name="TestProduct",
        description="Test description",
        capabilities=["Data Encryption"],
        value_proposition="Data encryption",
        contract=structured_contract,
    )
    legacy_product = ProductContext(
        product_name="TestProduct",
        description="Test description",
        capabilities=["Data Encryption"],
        value_proposition="Data encryption",
        limitations=["mainframe"],
    )

    fact_compat = CanonicalFact(
        fact_id="f1",
        claim_text="Fintech Velocity deployed cloud database",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="cloud db")],
    )
    fact_incompat = CanonicalFact(
        fact_id="f2",
        claim_text="Fintech Velocity operating legacy mainframe core",
        evidence_records=[EvidenceRecord(source_id="s2", verbatim_excerpt="mainframe")],
    )

    sig_compat = CandidateSignal(
        signal_id="s1",
        label="Cloud DB Encryption",
        description="encryption",
        target_capability="Data Encryption",
        supporting_fact_ids=["f1"],
    )
    sig_incompat = CandidateSignal(
        signal_id="s2",
        label="Mainframe Encryption",
        description="mainframe",
        target_capability="Data Encryption",
        supporting_fact_ids=["f2"],
    )

    evaluator = SignalEligibilityEvaluator()
    elig_struct_compat = evaluator.evaluate(signal=sig_compat, facts=[fact_compat], prospect=prospect, product=structured_product)
    elig_legacy_compat = evaluator.evaluate(signal=sig_compat, facts=[fact_compat], prospect=prospect, product=legacy_product)
    assert elig_struct_compat.status == elig_legacy_compat.status == EligibilityStatus.ELIGIBLE

    elig_struct_incompat = evaluator.evaluate(signal=sig_incompat, facts=[fact_incompat], prospect=prospect, product=structured_product)
    elig_legacy_incompat = evaluator.evaluate(signal=sig_incompat, facts=[fact_incompat], prospect=prospect, product=legacy_product)
    assert elig_struct_incompat.status == elig_legacy_incompat.status == EligibilityStatus.INELIGIBLE


def test_adversarial_10_fact_attributed_via_resolved_entity_relationship_passes_identity_gate(prospect: ProspectContext):
    """10. Fact attributed via resolved entity relationship passes identity gate without company string in text."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="f-entity",
        claim_text="European subsidiary operating aws_or_gcp_cloud_environment infrastructure opened Zurich center",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Zurich center")],
        entity_ids=["ent-org-123"],
    )
    sig = CandidateSignal(
        signal_id="sig-entity",
        label="Zurich Engineering Expansion",
        description="Engineering expansion",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-entity"],
        associated_entity_ids=["ent-org-123"],
    )

    company_entity = EntityRecord(
        entity_id="ent-org-123",
        raw_name="Fintech Velocity",
        name="Fintech Velocity",
        canonical_name="Fintech Velocity Inc.",
        entity_type=EntityType.ORGANIZATION,
        resolution_status=ResolutionStatus.RESOLVED,
    )
    package = ResearchPackage(
        facts=(fact,),
        company_entity=company_entity,
        is_sufficient=True,
        is_complete=True,
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(
        signal=sig,
        facts=[fact],
        prospect=prospect,
        product=product,
        research_package=package,
    )

    assert eligibility.identity_gate.passed is True
    assert "resolved entity relationship" in eligibility.identity_gate.reason


def test_adversarial_11_single_authoritative_primary_source_passes_evidence_gate(prospect: ProspectContext):
    """11. Single authoritative primary source passes evidence gate (corroboration is not a veto)."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="f-single-source",
        claim_text="Fintech Velocity operates aws_or_gcp_cloud_environment infrastructure per SEC 10-K annual report",
        evidence_records=[
            EvidenceRecord(
                source_id="src-sec-10k",
                verbatim_excerpt="We host our infrastructure primarily on AWS.",
                confidence=ConfidenceLevel.HIGH,
            )
        ],
        composite_confidence=ConfidenceLevel.HIGH,
    )
    sig = CandidateSignal(
        signal_id="sig-single",
        label="SEC 10-K Filing AWS Disclosure",
        description="SEC disclosure",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-single-source"],
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(signal=sig, facts=[fact], prospect=prospect, product=product)

    assert eligibility.evidence_gate.passed is True
    assert eligibility.evidence_gate.details.get("single_source_sufficient") is True
    assert eligibility.evidence_gate.details.get("corroborated_evidence_count") == 1


def test_adversarial_12_ambiguous_identity_remains_unresolved_and_fails_identity_gate(prospect: ProspectContext):
    """12. Ambiguous identity remains unresolved rather than being forced into a match, failing identity gate."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="f-ambig",
        claim_text="Fintech Velocity operating aws_or_gcp_cloud_environment infrastructure",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="infra")],
    )
    sig = CandidateSignal(
        signal_id="sig-ambig",
        label="AWS Infrastructure",
        description="infra",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-ambig"],
    )

    ambiguous_company_entity = EntityRecord(
        entity_id="ent-ambig-999",
        raw_name="Fintech Velocity",
        name="Fintech Velocity",
        entity_type=EntityType.ORGANIZATION,
        resolution_status=ResolutionStatus.AMBIGUOUS,
        disambiguation_notes="Multiple conflicting entities found: Fintech Velocity UK vs Fintech Velocity US",
    )
    package = ResearchPackage(
        facts=(fact,),
        company_entity=ambiguous_company_entity,
        is_sufficient=True,
        is_complete=True,
    )

    evaluator = SignalEligibilityEvaluator()
    eligibility = evaluator.evaluate(
        signal=sig,
        facts=[fact],
        prospect=prospect,
        product=product,
        research_package=package,
    )

    assert eligibility.identity_gate.passed is False
    assert eligibility.status == EligibilityStatus.INELIGIBLE
    assert "ambiguous or unresolved" in eligibility.identity_gate.reason


def test_adversarial_13_strategy_abstention_works_when_neither_verified_nor_diagnostic_justified(
    prospect: ProspectContext, outreach: OutreachContext
):
    """13. Strategy abstention works when neither verified nor diagnostic strategy selection is justified."""
    contract = _build_test_contract()
    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=contract,
    )

    fact = CanonicalFact(
        fact_id="f-party",
        claim_text="Fintech Velocity held annual team dinner",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="dinner")],
    )
    package = ResearchPackage(facts=(fact,), is_sufficient=True, is_complete=True)

    sig_out = RawSignalIdentificationOutput(
        candidate_signals=[
            RawCandidateSignal(
                label="Annual Team Dinner",
                description="Team dinner",
                supporting_fact_ids=["f-party"],
                fact_category="other",
            )
        ]
    )
    rank_out = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="",
                prospect_relevance=0.1,
                product_alignment=0.01,
                business_significance=0.1,
                rationale="Zero relevance to cloud audit",
            )
        ]
    )

    service = SignalStrategyService(engine=MultiMockEngine([sig_out, rank_out]))
    result = service.develop_strategy(
        research_package=package,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert len(result.unviable_reasons) > 0
