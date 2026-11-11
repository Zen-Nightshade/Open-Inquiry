from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance
from src.researcher.models import ResearchPackage, VerificationStatus
from src.signal_strategy.models import CTAType, OutreachStrategy



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
    QUARANTINED_VERIFICATION_FAILED = "quarantined_verification_failed"


class LinterSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LinterFinding(BaseModel):
    """An individual finding from the deterministic readability and quality linter."""

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(description="Unique identifier for the linter rule")
    severity: LinterSeverity = Field(description="Severity level of the finding")
    explanation: str = Field(description="Human-readable explanation of the issue")
    location: Optional[str] = Field(default=None, description="Relevant text excerpt or location in draft")


class LinterResult(BaseModel):
    """Consolidated outcome of the deterministic readability and email-quality linter."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(description="True if no ERROR severity findings are present")
    findings: List[LinterFinding] = Field(default_factory=list, description="All linter findings")
    word_count: int = Field(ge=0, description="Total word count analyzed")
    sentence_count: int = Field(ge=0, description="Total sentence count analyzed")
    paragraph_count: int = Field(ge=0, description="Total paragraph count analyzed")
    flesch_kincaid_grade: Optional[float] = Field(default=None, description="Computed Flesch-Kincaid readability grade")
    has_errors: bool = Field(default=False, description="True if any findings have severity ERROR")
    has_warnings: bool = Field(default=False, description="True if any findings have severity WARNING")


class RawEmailDraftOutput(BaseModel):
    """Pydantic model for structured LLM decoding of email drafts."""

    subject: str = Field(description="Subject line of the cold outreach email")
    body: str = Field(description="Plain text body of the cold outreach email")


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
    """Complete outreach draft contract for human review.
    
    Contains the generated and linted email draft, strategy alignment, selected CTA,
    and linter results. In M1.4, semantic claim verification (M1.5) has not yet been performed.
    Class name is preserved for backward compatibility across downstream modules.
    """

    model_config = ConfigDict(frozen=True)

    draft_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    draft: EmailDraft
    claim_verifications: List[ClaimVerification] = Field(
        default_factory=list,
        description="Verification results for all claims in the draft",
    )
    quality_evaluation: Optional[QualityEvaluation] = Field(
        default=None,
        description="Rubric evaluation across 7 dimensions",
    )
    research_package: Optional[ResearchPackage] = Field(
        default=None,
        description="Full research package providing complete evidence provenance",
    )
    outreach_strategy: OutreachStrategy = Field(
        description="Strategy blueprint that guided this draft",
    )
    linter_result: Optional[LinterResult] = Field(
        default=None,
        description="Authoritative outcome and findings of the readability and quality linter",
    )
    verifier_result: Optional[Any] = Field(
        default=None,
        description="Authoritative outcome and findings of the four-gate claim and capability verifier",
    )
    selected_cta: Optional[CTAType] = Field(
        default=None,
        description="Authoritative CTA type selected from strategy",
    )
    cta_text: Optional[str] = Field(
        default=None,
        description="Exact CTA sentence or closing inquiry used in draft",
    )
    provenance: Optional[OutputProvenance] = Field(
        default=None,
        description="Generation and linting provenance metadata",
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
