from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration.evaluator import CalibrationEvaluator
from src.evaluation.calibration.platt_scaler import PlattScaler


def test_brier_and_log_loss_computation():
    """Verify Brier score and log loss computation on known distributions."""
    # Perfect predictions: p = y
    probs_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    res_perf = CalibrationEvaluator.evaluate(probs_perfect, labels)
    assert res_perf.brier_score == 0.0
    assert res_perf.log_loss < 1e-10

    # Totally inaccurate predictions
    probs_wrong = np.array([1.0, 1.0, 0.0, 0.0])
    res_wrong = CalibrationEvaluator.evaluate(probs_wrong, labels)
    assert res_wrong.brier_score == 1.0
    assert res_wrong.log_loss > 30.0  # clipped at 1e-15


def test_ece_and_reliability_diagram_bins():
    """Verify that ECE and 10 equal-width reliability bins are calculated correctly."""
    # 20 samples uniformly distributed
    probs = np.linspace(0.05, 0.95, 20)
    labels = np.array([0] * 10 + [1] * 10)

    res = CalibrationEvaluator.evaluate(probs, labels, num_bins=10)

    assert len(res.reliability_bins) == 10
    assert res.sample_count == 20
    assert res.positive_count == 10
    assert res.negative_count == 10
    assert 0.0 <= res.ece <= 1.0

    # Check bin boundaries
    for i, b in enumerate(res.reliability_bins):
        assert b.bin_index == i
        assert b.bin_lower == round(i * 0.1, 3)
        assert b.bin_upper == round((i + 1) * 0.1, 3)
        assert 0.0 <= b.mean_predicted_prob <= 1.0
        assert 0.0 <= b.observed_accuracy <= 1.0


def test_discrimination_metrics_roc_pr_auc():
    """Verify that ROC-AUC and PR-AUC are calculated for multi-class labels."""
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])

    res = CalibrationEvaluator.evaluate(probs, labels)
    assert res.roc_auc == 1.0
    assert res.pr_auc == 1.0

    # Test single-class handling: ROC-AUC and PR-AUC must be None, not crash
    single_class_res = CalibrationEvaluator.evaluate(probs, np.array([1, 1, 1, 1, 1, 1]))
    assert single_class_res.roc_auc is None
    assert single_class_res.pr_auc is None


def test_compare_raw_vs_calibrated():
    """Verify comparison on synthetic uncalibrated scores vs Platt calibrated probabilities."""
    # Synthetic uncalibrated scores (e.g. overconfident raw scores [0.6, 0.7, 0.8...])
    train_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    scaler = PlattScaler(min_samples=5)
    scaler.fit(train_scores, train_labels)

    eval_scores = np.array([0.15, 0.35, 0.65, 0.88])
    eval_labels = np.array([0, 0, 1, 1])
    cal_probs = scaler.predict_proba(eval_scores)

    comparison = CalibrationEvaluator.compare_raw_vs_calibrated(
        raw_scores=eval_scores,
        calibrated_probs=cal_probs,
        labels=eval_labels,
    )

    assert "raw" in comparison
    assert "calibrated" in comparison
    assert "brier_score_diff" in comparison
    assert "ece_diff" in comparison
    assert isinstance(comparison["improved"], bool)
