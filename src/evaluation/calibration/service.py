from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from src.evaluation.calibration.models import (
    CalibrationArtifact,
    CalibrationPrediction,
    CalibrationTarget,
    ScoreFamily,
)
from src.evaluation.calibration.platt_scaler import PlattScaler

logger = logging.getLogger(__name__)


class CalibrationService:
    """Manages empirical calibration artifacts and probability inferences."""

    def __init__(self) -> None:
        self._artifacts: Dict[Tuple[CalibrationTarget, ScoreFamily], CalibrationArtifact] = {}
        self._scalers: Dict[Tuple[CalibrationTarget, ScoreFamily], PlattScaler] = {}

    def register_artifact(self, artifact: CalibrationArtifact) -> None:
        """Register a serialized calibration artifact and initialize its PlattScaler."""
        key = (artifact.target, artifact.score_family)
        self._artifacts[key] = artifact

        if artifact.is_calibrated:
            self._scalers[key] = PlattScaler.from_artifact(artifact)
            logger.info(
                "Registered active calibration model for target=%s score_family=%s (A=%.4f, B=%.4f)",
                artifact.target.value,
                artifact.score_family.value,
                artifact.a_parameter,
                artifact.b_parameter,
            )
        else:
            self._scalers.pop(key, None)
            logger.info(
                "Registered explicitly uncalibrated artifact for target=%s score_family=%s",
                artifact.target.value,
                artifact.score_family.value,
            )

    def load_artifact_from_file(self, file_path: Union[str, Path]) -> CalibrationArtifact:
        """Load and register a calibration artifact from a JSON file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        artifact = CalibrationArtifact.model_validate(data)
        self.register_artifact(artifact)
        return artifact

    def save_artifact_to_file(self, artifact: CalibrationArtifact, file_path: Union[str, Path]) -> None:
        """Serialize a calibration artifact to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2))

    def calibrate(
        self,
        raw_score: float,
        target: CalibrationTarget,
        score_family: ScoreFamily,
    ) -> CalibrationPrediction:
        """Calibrate an uncalibrated raw score into a posterior probability.

        If no calibrated model is registered or the gate was not met, returns an
        explicitly uncalibrated CalibrationPrediction without altering raw_score.
        """
        key = (target, score_family)
        scaler = self._scalers.get(key)
        artifact = self._artifacts.get(key)

        if scaler is None or artifact is None or not artifact.is_calibrated:
            return CalibrationPrediction(
                raw_score=raw_score,
                calibrated_probability=None,
                is_calibrated=False,
                target=target,
                score_family=score_family,
                explanation=(
                    "No production calibration model fitted (available dataset is below threshold gate "
                    "or uncalibrated fallback active). Raw score preserved without pseudo-probability conversion."
                ),
            )

        try:
            prob = float(scaler.predict_proba(raw_score))
            return CalibrationPrediction(
                raw_score=raw_score,
                calibrated_probability=round(prob, 6),
                is_calibrated=True,
                target=target,
                score_family=score_family,
                explanation=f"Calibrated via Platt scaling (A={artifact.a_parameter:.4f}, B={artifact.b_parameter:.4f})",
            )
        except Exception as e:
            logger.error("Error during calibration inference: %s", e)
            return CalibrationPrediction(
                raw_score=raw_score,
                calibrated_probability=None,
                is_calibrated=False,
                target=target,
                score_family=score_family,
                explanation=f"Calibration inference failed ({e}); falling back to uncalibrated raw score.",
            )

    def calibrate_draft_acceptance(
        self,
        raw_score: float,
        score_family: ScoreFamily = ScoreFamily.SIGNAL_COMPOSITE,
    ) -> CalibrationPrediction:
        """Convenience method for draft acceptance probability calibration."""
        return self.calibrate(
            raw_score=raw_score,
            target=CalibrationTarget.DRAFT_ACCEPTANCE,
            score_family=score_family,
        )

    def calibrate_signal_score(
        self,
        raw_score: float,
    ) -> CalibrationPrediction:
        """Convenience method for signal ranking calibration."""
        return self.calibrate(
            raw_score=raw_score,
            target=CalibrationTarget.SIGNAL_RANKING,
            score_family=ScoreFamily.SIGNAL_COMPOSITE,
        )

    def calibrate_claim_verification(
        self,
        raw_score: float,
    ) -> CalibrationPrediction:
        """Convenience method for claim verifier confidence calibration."""
        return self.calibrate(
            raw_score=raw_score,
            target=CalibrationTarget.CLAIM_VERIFICATION,
            score_family=ScoreFamily.CLAIM_VERIFIER,
        )
