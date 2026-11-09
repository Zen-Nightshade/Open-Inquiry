from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.models import ResearchPackage, VerificationStatus
from src.signal_strategy.models import OutreachStrategy



class ClaimType(str, Enum):
    FACT_ABOUT_PROSPECT = "fact_about_prospect"
    FACT_ABOUT_COMPANY = "fact_about_company"
    PRODUCT_CLAIM = "product_claim"
    MARKET_INFERENCE = "market_inference"
    VALUE_PROPOSITION = "value_proposition"


class HumanReviewStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class AtomicClaim(BaseModel):
    """An individual atomic claim extracted from an email draft."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = Field(description="Exact statement extracted from draft")
    claim_type: ClaimType = Field(default=ClaimType.FACT_ABOUT_COMPANY)
    sentence_index: Optional[int] = Field(default=None, description="Position of the claim in the draft text")


class ClaimVerification(BaseModel):
    """Grounded entailment verification linking an atomic claim to research evidence."""

    model_config = ConfigDict(frozen=True)

    claim: AtomicClaim
    status: VerificationStatus
    supporting_fact_ids: List[str] = Field(
        default_factory=list,
        description="Foreign keys to CanonicalFact.fact_id",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Specific EvidenceRecord IDs corroborating this claim",
    )
    explanation: str = Field(description="Detailed verification rationale")
    suggested_replacement: Optional[str] = Field(
        default=None,
        description="Remediation text if claim is unsupported or contradicted",
    )


class QualityDimension(BaseModel):
    """Score and feedback for an individual quality dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="Name of quality dimension, e.g. personalization, conciseness")
    score: int = Field(ge=1, le=5, description="Rubric score from 1 to 5")
    justification: str = Field(description="Detailed reasoning for assigned score")
    suggestion: Optional[str] = Field(default=None, description="Actionable improvement suggestion")


class QualityEvaluation(BaseModel):
    """Multi-dimensional quality assessment of an outreach draft."""

    model_config = ConfigDict(frozen=True)

    dimensions: List[QualityDimension] = Field(
        description="Scores across all 7 evaluation dimensions",
    )
    overall_score: float = Field(ge=1.0, le=5.0, description="Mean composite quality score")
    critical_issues: List[str] = Field(default_factory=list, description="Critical failures blocking delivery")
    is_acceptable: bool = Field(description="Whether draft meets quality bar for human review")


class EmailDraft(BaseModel):
    """The subject and body text of an outreach email draft."""

    model_config = ConfigDict(frozen=True)

    subject: str = Field(description="Email subject line")
    body: str = Field(description="Plain text email body")
    word_count: int = Field(ge=0, description="Body word count")
    generation_attempt: int = Field(default=1, ge=1, description="Attempt number if rewritten")


class VerifiedOutreachDraft(BaseModel):
    """Complete, verified outreach draft contract for human review.
    
    Includes full provenance back to source evidence and explicit review flags.
    """

    model_config = ConfigDict(frozen=True)

    draft_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    draft: EmailDraft
    claim_verifications: List[ClaimVerification] = Field(
        description="Verification results for all claims in the draft",
    )
    quality_evaluation: QualityEvaluation = Field(
        description="Rubric evaluation across 7 dimensions",
    )
    research_package: ResearchPackage = Field(
        description="Full research package providing complete evidence provenance",
    )
    outreach_strategy: OutreachStrategy = Field(
        description="Strategy blueprint that guided this draft",
    )
    generation_attempts: int = Field(default=1, ge=1)
    human_review_status: HumanReviewStatus = Field(
        default=HumanReviewStatus.PENDING_REVIEW,
        description="Current workflow review status",
    )
    human_review_reasons: List[str] = Field(
        default_factory=list,
        description="Explicit system-flagged reasons or warnings requiring human attention",
    )
    reviewer_notes: Optional[str] = Field(default=None, description="Notes from the human reviewer")
