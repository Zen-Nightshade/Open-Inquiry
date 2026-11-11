from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.evaluation.calibration.models import (
    CalibrationArtifact,
    CalibrationTarget,
    ScoreFamily,
    SplitStrategy,
)

logger = logging.getLogger(__name__)


class PlattScaler:
    """Logistic sigmoid calibration model via Platt scaling.

    Maps uncalibrated scores to well-calibrated posterior probabilities:
        P(y=1 | s) = 1 / (1 + exp(A * s + B))

    Follows Platt (1999) with out-of-sample regularized target probabilities
    to prevent overconfident probability estimates on small cohorts.
    """

    def __init__(
        self,
        a_parameter: Optional[float] = None,
        b_parameter: Optional[float] = None,
        min_samples: int = 10,
        use_platt_smoothing: bool = True,
    ) -> None:
        self.a_parameter = a_parameter
        self.b_parameter = b_parameter
        self.min_samples = min_samples
        self.use_platt_smoothing = use_platt_smoothing
        self.is_fitted = a_parameter is not None and b_parameter is not None

    def fit(
        self,
        scores: Union[Sequence[float], np.ndarray],
        labels: Union[Sequence[int], np.ndarray],
    ) -> PlattScaler:
        """Fit parameters A and B using deterministic L-BFGS optimization.

        Args:
            scores: Array of uncalibrated continuous scores.
            labels: Array of binary ground truth labels in {0, 1}.

        Raises:
            ValueError: If sample count < min_samples, labels contain only a single
                        class, or inputs contain NaNs/Infs.
        """
        s_arr = np.asarray(scores, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.float64)

        if s_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Scores ({len(s_arr)}) and labels ({len(y_arr)}) dimension mismatch.")

        if s_arr.shape[0] < self.min_samples:
            raise ValueError(
                f"Sample count ({len(s_arr)}) is below minimum required sample size ({self.min_samples})."
            )

        if not np.all(np.isfinite(s_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Input scores and labels must not contain NaN or Inf values.")

        unique_labels = np.unique(y_arr)
        if not np.all(np.isin(unique_labels, [0.0, 1.0])):
            raise ValueError(f"Labels must be binary in {{0, 1}}, got: {unique_labels.tolist()}")

        if len(unique_labels) < 2:
            raise ValueError(
                "Platt scaling requires both positive and negative classes in training data. "
                f"Found only class: {unique_labels.tolist()}"
            )

        n_pos = float(np.sum(y_arr == 1.0))
        n_neg = float(np.sum(y_arr == 0.0))

        if self.use_platt_smoothing:
            # Platt regularized targets to avoid extreme overconfidence
            t_pos = (n_pos + 1.0) / (n_pos + 2.0)
            t_neg = 1.0 / (n_neg + 2.0)
            targets = np.where(y_arr == 1.0, t_pos, t_neg)
        else:
            targets = y_arr

        # Initial parameters: A=0, B=log((N_- + 1)/(N_+ + 1))
        initial_b = np.log((n_neg + 1.0) / (n_pos + 1.0))
        initial_params = np.array([0.0, initial_b], dtype=np.float64)

        eps = 1e-15

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = params[0], params[1]
            z = a * s_arr + b
            # p = 1 / (1 + exp(z)) = expit(-z)
            p = expit(-z)
            p_safe = np.clip(p, eps, 1.0 - eps)

            # Negative log-likelihood: - sum( t*log(p) + (1-t)*log(1-p) )
            loss = -np.sum(targets * np.log(p_safe) + (1.0 - targets) * np.log(1.0 - p_safe))

            # d(loss)/dz_i = targets_i - p_i
            # dz_i/da = s_i, dz_i/db = 1
            diff = targets - p
            grad_a = np.sum(diff * s_arr)
            grad_b = np.sum(diff)
            grad = np.array([grad_a, grad_b], dtype=np.float64)

            return loss, grad

        opt_res = minimize(
            fun=loss_and_grad,
            x0=initial_params,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
        )

        if not opt_res.success and "CONVERGENCE" not in opt_res.message.upper():
            logger.warning("PlattScaler optimization warning: %s", opt_res.message)

        self.a_parameter = float(opt_res.x[0])
        self.b_parameter = float(opt_res.x[1])
        self.is_fitted = True

        return self

    def predict_proba(
        self,
        scores: Union[float, Sequence[float], np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute calibrated probability P(y=1 | s) in [0.0, 1.0]."""
        if not self.is_fitted or self.a_parameter is None or self.b_parameter is None:
            raise RuntimeError("PlattScaler must be fitted before predict_proba can be called.")

        is_scalar = isinstance(scores, (int, float, np.floating, np.integer))
        s_arr = np.asarray([scores] if is_scalar else scores, dtype=np.float64)

        z = self.a_parameter * s_arr + self.b_parameter
        probs = expit(-z)
        probs_clipped = np.clip(probs, 0.0, 1.0)

        if is_scalar:
            return float(probs_clipped[0])
        return probs_clipped

    def to_artifact(
        self,
        target: CalibrationTarget,
        score_family: ScoreFamily,
        train_sample_count: int,
        eval_sample_count: int,
        eval_metrics: Optional[Dict[str, Any]] = None,
        split_strategy: SplitStrategy = SplitStrategy.PROSPECT_GROUPED,
        notes: Optional[str] = None,
    ) -> CalibrationArtifact:
        """Serialize fitted model to a versioned CalibrationArtifact."""
        if not self.is_fitted or self.a_parameter is None or self.b_parameter is None:
            raise RuntimeError("Cannot export unfitted PlattScaler to artifact.")

        eval_metrics = eval_metrics or {}
        now_iso = datetime.now(timezone.utc).isoformat()

        return CalibrationArtifact(
            target=target,
            score_family=score_family,
            method="platt_scaling",
            a_parameter=self.a_parameter,
            b_parameter=self.b_parameter,
            is_calibrated=True,
            trained_at=now_iso,
            train_sample_count=train_sample_count,
            eval_sample_count=eval_sample_count,
            brier_score_raw=eval_metrics.get("brier_score_raw"),
            brier_score_calibrated=eval_metrics.get("brier_score_calibrated"),
            log_loss_raw=eval_metrics.get("log_loss_raw"),
            log_loss_calibrated=eval_metrics.get("log_loss_calibrated"),
            ece_raw=eval_metrics.get("ece_raw"),
            ece_calibrated=eval_metrics.get("ece_calibrated"),
            roc_auc=eval_metrics.get("roc_auc"),
            pr_auc=eval_metrics.get("pr_auc"),
            split_strategy=split_strategy,
            notes=notes,
        )

    @classmethod
    def from_artifact(cls, artifact: CalibrationArtifact) -> PlattScaler:
        """Instantiate a fitted PlattScaler from a serialized CalibrationArtifact."""
        if not artifact.is_calibrated:
            scaler = cls()
            scaler.is_fitted = False
            return scaler

        return cls(
            a_parameter=artifact.a_parameter,
            b_parameter=artifact.b_parameter,
        )
