from __future__ import annotations

import pytest

from src.context.models import CapabilityNode, ProductConstraint, ProductContract, ConstraintType
from src.email_generation.claim_verifier.capability import ProductCapabilityBoundsGate
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    ModalityLevel,
    VerifierFindingSeverity,
)


def _make_dummy_contract(is_exhaustive: bool = False) -> ProductContract:
    cap = CapabilityNode(
        capability_id="cap-telemetry",
        name="Node Latency Telemetry",
        description="Real-time telemetry and latency reduction for container nodes",
        target_operational_problems=["telemetry lag", "queue bottlenecks"],
        supported_use_cases=["Kubernetes node monitoring"],
        hard_exclusions=["on-premise OpenShift", "legacy mainframes"],
        verified_proof_points=["50% latency reduction in Kubernetes clusters", "10,000 node benchmark"],
    )
    return ProductContract(
        product_name="PulseRadar",
        contract_version="1.0.0",
        description="Distributed telemetry intelligence",
        value_proposition="Eliminates telemetry overhead at massive scale",
        capabilities=[cap],
        hard_exclusions=["on-premise OpenShift"],
        is_exhaustive=is_exhaustive,
    )


def test_capability_valid_claim_with_whitelisted_metric():
    contract = _make_dummy_contract()
    gate = ProductCapabilityBoundsGate()

    claim = DecomposedClaim(
        claim_id="c-valid",
        text="We built PulseRadar to cut node latency by 50% without adding extra overhead.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=80,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    assert results[0].is_whitelisted
    assert not results[0].violates_exclusion
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_capability_hard_exclusion_violated_produces_error():
    contract = _make_dummy_contract()
    gate = ProductCapabilityBoundsGate()

    # Mentions on-premise OpenShift which is explicitly in hard_exclusions
    claim = DecomposedClaim(
        claim_id="c-excl",
        text="PulseRadar integrates natively with your on-premise OpenShift clusters.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=74,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    assert results[0].violates_exclusion
    assert any(f.code == "HARD_EXCLUSION_VIOLATION" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_capability_unsupported_fabricated_metric_produces_error():
    contract = _make_dummy_contract()
    gate = ProductCapabilityBoundsGate()

    # Claims 99% and 10x, which are not in verified_proof_points (50% is the only one)
    claim = DecomposedClaim(
        claim_id="c-metric",
        text="PulseRadar guarantees a 99% reduction in latency and 10x throughput.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=70,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    assert not results[0].is_whitelisted
    assert len(results[0].unsupported_proof_points) > 0
    assert any(f.code == "UNSUPPORTED_PROOF_POINT" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_capability_unknown_under_exhaustive_contract_produces_error():
    contract = _make_dummy_contract(is_exhaustive=True)
    gate = ProductCapabilityBoundsGate()

    # Claims database sharding optimization, totally absent from capabilities
    claim = DecomposedClaim(
        claim_id="c-unsupported-cap",
        text="We provide automated database sharding and index rebalancing.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=62,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    assert any(f.code == "INCOMPATIBLE_CAPABILITY" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_capability_unknown_under_non_exhaustive_contract_produces_warning():
    contract = _make_dummy_contract(is_exhaustive=False)
    gate = ProductCapabilityBoundsGate()

    claim = DecomposedClaim(
        claim_id="c-unknown-cap",
        text="We provide automated database sharding and index rebalancing.",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=62,
        category=ClaimCategory.PRODUCT_CAPABILITY,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], contract)
    assert len(results) == 1
    # Under non-exhaustive, unknown capabilities produce a WARNING, not an ERROR
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)
    assert any(f.code == "CONDITIONAL_CAPABILITY_UNKNOWN" and f.severity == VerifierFindingSeverity.WARNING for f in findings)
