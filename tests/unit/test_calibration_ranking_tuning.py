from __future__ import annotations

import pytest

from src.evaluation.calibration import (
    CalibrationArtifact,
    CalibrationService,
    CalibrationTarget,
    RankingQualityAnalyzer,
    ScoreFamily,
)
from src.signal_strategy.models import (
    CandidateSignal,
    EligibilityStatus,
    GateIdentifier,
    GateResult,
    RankedSignal,
    SignalEligibility,
    SignalScore,
)


def _create_mock_ranked_signal(
    signal_id: str,
    composite_score: float,
    status: EligibilityStatus,
) -> RankedSignal:
    gate_res = GateResult(
        gate=GateIdentifier.IDENTITY,
        passed=status != EligibilityStatus.INELIGIBLE,
        reason="Test gate evaluation",
    )
    elig = SignalEligibility(
        status=status,
        is_eligible=status == EligibilityStatus.ELIGIBLE,
        identity_gate=gate_res,
        evidence_gate=gate_res,
        product_gate=gate_res,
        persona_gate=gate_res,
    )
    score = SignalScore(
        evidence_strength=0.8,
        recency_score=0.9,
        prospect_relevance=0.85,
        product_alignment=0.8 if status != EligibilityStatus.INELIGIBLE else 0.0,
        business_significance=0.9,
        composite_score=composite_score,
    )
    candidate = CandidateSignal(
        signal_id=signal_id,
        label=f"Signal {signal_id}",
        description="Test commercial signal",
        supporting_fact_ids=["fact_1"],
    )
    return RankedSignal(
        signal=candidate,
        score=score,
        rationale="Test rationale",
        eligibility=elig,
    )


def test_authoritative_eligibility_invariants():
    """Verify that RankingQualityAnalyzer enforces composite_score == 0.0 for all ineligible signals."""
    # 1. Clean valid set: eligible signals have positive scores, ineligible has 0.0
    sig_eligible = _create_mock_ranked_signal("sig_1", 0.85, EligibilityStatus.ELIGIBLE)
    sig_conditional = _create_mock_ranked_signal("sig_2", 0.65, EligibilityStatus.CONDITIONAL)
    sig_ineligible = _create_mock_ranked_signal("sig_3", 0.0, EligibilityStatus.INELIGIBLE)

    valid_res = RankingQualityAnalyzer.verify_eligibility_invariants([sig_eligible, sig_conditional, sig_ineligible])
    assert valid_res["is_valid"] is True
    assert valid_res["authoritative_gates_preserved"] is True
    assert len(valid_res["violations"]) == 0
    assert valid_res["ineligible_count"] == 1

    # 2. Corrupted set: ineligible signal has non-zero score (invariant violation!)
    sig_corrupted = _create_mock_ranked_signal("sig_bad", 0.75, EligibilityStatus.INELIGIBLE)
    corrupted_res = RankingQualityAnalyzer.verify_eligibility_invariants([sig_eligible, sig_corrupted])
    assert corrupted_res["is_valid"] is False
    assert corrupted_res["authoritative_gates_preserved"] is False
    assert len(corrupted_res["violations"]) == 1
    assert "violates non-compensatory gate invariant" in corrupted_res["violations"][0]


def test_rejection_reason_taxonomy_analysis():
    """Verify reviewer rejection reason distribution and gate attribution."""
    events = [
        {"reason_code": "MISALIGNED_PRODUCT"},
        {"reason_code": "MISALIGNED_PRODUCT"},
        {"reason_code": "WEAK_EVIDENCE"},
        {"reason_code": "WRONG_PERSONA"},
        {"reason_code": "ENTITY_MISMATCH"},
        {"reason_code": "BAD_TIMING"},  # ungated reason
    ]

    analysis = RankingQualityAnalyzer.analyze_rejection_reasons(events)

    assert analysis["total_rejections"] == 6
    assert analysis["reason_distribution"]["MISALIGNED_PRODUCT"] == 2
    assert analysis["reason_distribution"]["WEAK_EVIDENCE"] == 1
    assert analysis["gate_attribution"]["product_compatibility"] == 2
    assert analysis["gate_attribution"]["evidence"] == 1
    assert analysis["gate_attribution"]["persona"] == 1
    assert analysis["gate_attribution"]["identity"] == 1
    assert "BAD_TIMING" in analysis["ungated_reasons"]


def test_calibration_service_uncalibrated_fallback():
    """Verify that CalibrationService preserves raw scores and returns uncalibrated when gate is not met."""
    service = CalibrationService()

    # Query draft acceptance when uncalibrated
    pred = service.calibrate_draft_acceptance(raw_score=0.742)
    assert pred.is_calibrated is False
    assert pred.calibrated_probability is None
    assert pred.raw_score == 0.742
    assert "No production calibration model fitted" in (pred.explanation or "")

    # Query signal score when uncalibrated
    sig_pred = service.calibrate_signal_score(raw_score=0.88)
    assert sig_pred.is_calibrated is False
    assert sig_pred.calibrated_probability is None
    assert sig_pred.raw_score == 0.88


def test_calibration_service_with_registered_artifact():
    """Verify calibration inference when a fitted artifact is active."""
    service = CalibrationService()

    artifact = CalibrationArtifact(
        target=CalibrationTarget.DRAFT_ACCEPTANCE,
        score_family=ScoreFamily.SIGNAL_COMPOSITE,
        a_parameter=-4.0,
        b_parameter=2.0,
        is_calibrated=True,
        trained_at="2026-11-11T00:00:00",
        train_sample_count=250,
        eval_sample_count=50,
    )
    service.register_artifact(artifact)

    pred = service.calibrate_draft_acceptance(raw_score=0.8)
    assert pred.is_calibrated is True
    assert pred.raw_score == 0.8
    assert pred.calibrated_probability is not None
    assert 0.0 <= pred.calibrated_probability <= 1.0
