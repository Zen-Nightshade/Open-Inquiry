from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration.models import (
    CalibrationArtifact,
    CalibrationTarget,
    ScoreFamily,
    SplitStrategy,
)
from src.evaluation.calibration.platt_scaler import PlattScaler


def test_platt_scaler_fit_and_monotonicity():
    """Verify that PlattScaler fits sigmoid parameters and preserves monotonicity."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    scaler = PlattScaler(min_samples=5)
    scaler.fit(scores, labels)

    assert scaler.is_fitted is True
    assert scaler.a_parameter is not None
    assert scaler.b_parameter is not None

    # Higher score should yield higher probability of acceptance
    p_low = scaler.predict_proba(0.15)
    p_high = scaler.predict_proba(0.85)

    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low


def test_platt_scaler_probability_bounds():
    """Verify that predicted probabilities are strictly bounded within [0.0, 1.0]."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    scaler = PlattScaler(min_samples=5)
    scaler.fit(scores, labels)

    # Test extreme negative and extreme positive score inputs
    extreme_scores = np.array([-1000.0, -10.0, 0.0, 0.5, 1.0, 10.0, 1000.0])
    probs = scaler.predict_proba(extreme_scores)

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_platt_scaler_single_class_rejection():
    """Verify that training data with only one class raises ValueError."""
    scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    all_positive = np.array([1, 1, 1, 1, 1])

    scaler = PlattScaler(min_samples=5)
    with pytest.raises(ValueError, match="requires both positive and negative classes"):
        scaler.fit(scores, all_positive)


def test_platt_scaler_below_min_samples_rejection():
    """Verify that sample count below min_samples raises ValueError."""
    scores = np.array([0.2, 0.8])
    labels = np.array([0, 1])

    scaler = PlattScaler(min_samples=10)
    with pytest.raises(ValueError, match="below minimum required sample size"):
        scaler.fit(scores, labels)


def test_platt_scaler_artifact_serialization():
    """Verify round-trip serialization to and from CalibrationArtifact."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    scaler = PlattScaler(min_samples=5)
    scaler.fit(scores, labels)

    artifact = scaler.to_artifact(
        target=CalibrationTarget.DRAFT_ACCEPTANCE,
        score_family=ScoreFamily.SIGNAL_COMPOSITE,
        train_sample_count=len(scores),
        eval_sample_count=4,
        eval_metrics={"brier_score_calibrated": 0.08, "ece_calibrated": 0.05},
        split_strategy=SplitStrategy.PROSPECT_GROUPED,
        notes="Platt test model",
    )

    assert artifact.is_calibrated is True
    assert artifact.a_parameter == scaler.a_parameter
    assert artifact.b_parameter == scaler.b_parameter
    assert artifact.target == CalibrationTarget.DRAFT_ACCEPTANCE
    assert artifact.brier_score_calibrated == 0.08

    # Reconstruct scaler from artifact
    reconstructed = PlattScaler.from_artifact(artifact)
    assert reconstructed.is_fitted is True

    test_input = np.array([0.25, 0.75])
    orig_p = scaler.predict_proba(test_input)
    recon_p = reconstructed.predict_proba(test_input)

    np.testing.assert_allclose(orig_p, recon_p, rtol=1e-6)
