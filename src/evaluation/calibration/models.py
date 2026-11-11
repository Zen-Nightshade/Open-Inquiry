from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CalibrationTarget(str, Enum):
    """Explicitly isolated calibration decision targets."""

    DRAFT_ACCEPTANCE = "draft_acceptance"
    SIGNAL_RANKING = "signal_ranking"
    CLAIM_VERIFICATION = "claim_verification"


class ScoreFamily(str, Enum):
    """Identifier for the source model/heuristic score family being calibrated."""

    SIGNAL_COMPOSITE = "signal_composite"
    CLAIM_VERIFIER = "claim_verifier"
    READABILITY_LINTER = "readability_linter"
    DRAFT_ACCEPTANCE_COMPOSITE = "draft_acceptance_composite"


class SplitStrategy(str, Enum):
    """Evaluation dataset partitioning strategy to prevent train/test contamination."""

    PROSPECT_GROUPED = "prospect_grouped"
    COMPANY_GROUPED = "company_grouped"
    TEMPORAL_HOLDOUT = "temporal_holdout"


class CalibrationDatasetRecord(BaseModel):
    """Immutable data record capturing a single draft revision outcome and its raw score."""

    model_config = ConfigDict(frozen=True)

    draft_id: str = Field(description="Unique identifier of the outreach draft")
    revision_number: int = Field(default=1, description="Revision number of this draft observation")
    prospect_id: Optional[str] = Field(default=None, description="Prospect identifier for grouping")
    company_name: Optional[str] = Field(default=None, description="Company name for cross-leakage prevention")
    raw_score: float = Field(description="Raw heuristic or model score prior to calibration")
    label: int = Field(description="Binary outcome label: 1 for approved, 0 for rejected")
    outcome: str = Field(description="Final review outcome action: APPROVED or REJECTED")
    rejection_reason: Optional[str] = Field(default=None, description="Reviewer rejection reason code if rejected")
    char_edit_distance: Optional[int] = Field(default=None, description="Levenshtein char edit distance from revision 1")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of review event")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary audit metadata")


class ReliabilityBin(BaseModel):
    """Calibration diagram bin metrics for a discrete probability interval."""

    model_config = ConfigDict(frozen=True)

    bin_index: int = Field(description="0-indexed bin position")
    bin_lower: float = Field(description="Inclusive lower probability boundary")
    bin_upper: float = Field(description="Inclusive upper probability boundary")
    sample_count: int = Field(description="Number of predictions falling in this bin")
    mean_predicted_prob: float = Field(description="Average predicted probability within bin")
    observed_accuracy: float = Field(description="Proportion of true positive labels in this bin")
    calibration_error: float = Field(description="Absolute difference between accuracy and confidence")


class CalibrationEvaluationResult(BaseModel):
    """Standardized metrics container for calibration and discrimination performance."""

    model_config = ConfigDict(frozen=True)

    sample_count: int = Field(description="Total observations evaluated")
    positive_count: int = Field(description="Total positive (y=1) observations")
    negative_count: int = Field(description="Total negative (y=0) observations")
    brier_score: float = Field(description="Mean squared probability error in [0, 1]")
    log_loss: float = Field(description="Mean binary cross-entropy loss")
    ece: float = Field(description="Expected Calibration Error across 10 bins")
    reliability_bins: List[ReliabilityBin] = Field(default_factory=list, description="Binned calibration diagram data")
    roc_auc: Optional[float] = Field(default=None, description="Area under ROC curve if computable")
    pr_auc: Optional[float] = Field(default=None, description="Area under Precision-Recall curve if computable")


class CalibrationArtifact(BaseModel):
    """Serialized parameter artifact for a fitted Platt scaling calibration model."""

    model_config = ConfigDict(frozen=True)

    artifact_version: str = Field(default="1.0.0", description="Artifact schema version")
    target: CalibrationTarget = Field(description="Calibration target type")
    score_family: ScoreFamily = Field(description="Score family calibrated")
    method: str = Field(default="platt_scaling", description="Calibration algorithm")
    a_parameter: float = Field(description="Platt scaling slope parameter A in sigmoid P=1/(1+exp(A*s + B))")
    b_parameter: float = Field(description="Platt scaling intercept parameter B in sigmoid P=1/(1+exp(A*s + B))")
    is_calibrated: bool = Field(default=True, description="Whether calibrated model was fitted and active")
    trained_at: str = Field(description="ISO timestamp when artifact was fitted")
    train_sample_count: int = Field(description="Number of training examples used")
    eval_sample_count: int = Field(description="Number of evaluation examples used")
    brier_score_raw: Optional[float] = Field(default=None, description="Brier score before calibration on eval set")
    brier_score_calibrated: Optional[float] = Field(default=None, description="Brier score after calibration on eval set")
    log_loss_raw: Optional[float] = Field(default=None, description="Log loss before calibration on eval set")
    log_loss_calibrated: Optional[float] = Field(default=None, description="Log loss after calibration on eval set")
    ece_raw: Optional[float] = Field(default=None, description="ECE before calibration on eval set")
    ece_calibrated: Optional[float] = Field(default=None, description="ECE after calibration on eval set")
    roc_auc: Optional[float] = Field(default=None, description="Held-out ROC-AUC")
    pr_auc: Optional[float] = Field(default=None, description="Held-out PR-AUC")
    split_strategy: SplitStrategy = Field(default=SplitStrategy.PROSPECT_GROUPED, description="Partitioning used")
    notes: Optional[str] = Field(default=None, description="Audit or provenance notes")


class CalibrationPrediction(BaseModel):
    """Output envelope for a score passing through the calibration service."""

    model_config = ConfigDict(frozen=True)

    raw_score: float = Field(description="Original, unmodified input score")
    calibrated_probability: Optional[float] = Field(default=None, description="Calibrated posterior probability in [0, 1]")
    is_calibrated: bool = Field(default=False, description="Whether calibrated_probability is backed by a valid model")
    target: CalibrationTarget
    score_family: ScoreFamily
    explanation: Optional[str] = Field(default=None, description="Context regarding calibration status or fallback")
