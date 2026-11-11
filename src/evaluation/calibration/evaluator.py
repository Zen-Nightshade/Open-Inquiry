from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.calibration.models import (
    CalibrationEvaluationResult,
    ReliabilityBin,
)

logger = logging.getLogger(__name__)


class CalibrationEvaluator:
    """Computes empirical calibration and discrimination metrics."""

    @classmethod
    def evaluate(
        cls,
        probabilities: Union[Sequence[float], np.ndarray],
        labels: Union[Sequence[int], np.ndarray],
        num_bins: int = 10,
    ) -> CalibrationEvaluationResult:
        """Compute comprehensive calibration metrics against binary ground truth labels.

        Args:
            probabilities: Model predicted probabilities or continuous confidence scores in [0, 1].
            labels: Ground truth binary labels in {0, 1}.
            num_bins: Number of equal-width probability bins (default 10).
        """
        p_arr = np.asarray(probabilities, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.float64)

        if len(p_arr) != len(y_arr):
            raise ValueError(f"Probabilities ({len(p_arr)}) and labels ({len(y_arr)}) dimension mismatch.")

        if len(p_arr) == 0:
            raise ValueError("Cannot evaluate an empty dataset.")

        p_safe = np.clip(p_arr, 0.0, 1.0)
        n_total = len(p_safe)
        n_pos = int(np.sum(y_arr == 1.0))
        n_neg = int(np.sum(y_arr == 0.0))

        # 1. Brier score
        brier = float(np.mean((p_safe - y_arr) ** 2))

        # 2. Log loss
        eps = 1e-15
        p_log_safe = np.clip(p_safe, eps, 1.0 - eps)
        log_loss = float(-np.mean(y_arr * np.log(p_log_safe) + (1.0 - y_arr) * np.log(1.0 - p_log_safe)))

        # 3. Reliability diagram & ECE
        bins_data, ece = cls.compute_reliability_bins(p_safe, y_arr, num_bins=num_bins)

        # 4. Discrimination: ROC-AUC and PR-AUC
        roc_auc: Optional[float] = None
        pr_auc: Optional[float] = None

        if n_pos > 0 and n_neg > 0:
            try:
                roc_auc = float(roc_auc_score(y_arr, p_safe))
            except Exception as e:
                logger.debug("Could not compute ROC-AUC: %s", e)
            try:
                pr_auc = float(average_precision_score(y_arr, p_safe))
            except Exception as e:
                logger.debug("Could not compute PR-AUC: %s", e)

        return CalibrationEvaluationResult(
            sample_count=n_total,
            positive_count=n_pos,
            negative_count=n_neg,
            brier_score=round(brier, 6),
            log_loss=round(log_loss, 6),
            ece=round(ece, 6),
            reliability_bins=bins_data,
            roc_auc=round(roc_auc, 6) if roc_auc is not None else None,
            pr_auc=round(pr_auc, 6) if pr_auc is not None else None,
        )

    @classmethod
    def compute_reliability_bins(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ) -> tuple[List[ReliabilityBin], float]:
        """Compute equal-width reliability diagram bins and expected calibration error."""
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bins_list: List[ReliabilityBin] = []
        weighted_ece_sum = 0.0
        n_total = len(probs)

        for i in range(num_bins):
            b_low = float(bin_boundaries[i])
            b_high = float(bin_boundaries[i + 1])

            if i == num_bins - 1:
                in_bin = (probs >= b_low) & (probs <= b_high)
            else:
                in_bin = (probs >= b_low) & (probs < b_high)

            count = int(np.sum(in_bin))

            if count > 0:
                mean_prob = float(np.mean(probs[in_bin]))
                acc = float(np.mean(labels[in_bin]))
                err = float(abs(acc - mean_prob))
                weighted_ece_sum += (count / n_total) * err
            else:
                mean_prob = (b_low + b_high) / 2.0
                acc = 0.0
                err = 0.0

            bins_list.append(
                ReliabilityBin(
                    bin_index=i,
                    bin_lower=round(b_low, 3),
                    bin_upper=round(b_high, 3),
                    sample_count=count,
                    mean_predicted_prob=round(mean_prob, 4),
                    observed_accuracy=round(acc, 4),
                    calibration_error=round(err, 4),
                )
            )

        return bins_list, weighted_ece_sum

    @classmethod
    def compare_raw_vs_calibrated(
        cls,
        raw_scores: Union[Sequence[float], np.ndarray],
        calibrated_probs: Union[Sequence[float], np.ndarray],
        labels: Union[Sequence[int], np.ndarray],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """Compare calibration performance of raw continuous scores vs calibrated probabilities."""
        raw_res = cls.evaluate(raw_scores, labels, num_bins=num_bins)
        cal_res = cls.evaluate(calibrated_probs, labels, num_bins=num_bins)

        return {
            "raw": raw_res,
            "calibrated": cal_res,
            "brier_score_diff": round(raw_res.brier_score - cal_res.brier_score, 6),
            "log_loss_diff": round(raw_res.log_loss - cal_res.log_loss, 6),
            "ece_diff": round(raw_res.ece - cal_res.ece, 6),
            "improved": cal_res.brier_score <= raw_res.brier_score and cal_res.ece <= raw_res.ece,
        }
