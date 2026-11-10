from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.email_generation.models import AtomicClaim, ClaimType, ClaimVerification
from src.researcher.models import VerificationStatus


class ClaimCategory(str, Enum):
    """Semantic category of an atomic proposition in an outreach draft."""

    PROSPECT_OBSERVATION = "prospect_observation"  # Facts about prospect/company
    PRODUCT_CAPABILITY = "product_capability"      # Claims about product features/value
    METRIC_IMPACT = "metric_impact"                # Quantitative metrics or proof points
    MODALITY_HEDGE = "modality_hedge"              # Explicit conditions or epistemic hedges
    RELATIONAL_CTA = "relational_cta"              # Call-to-action question or request


class ClaimSection(str, Enum):
    """Section of the email draft where the claim originates."""

    SUBJECT = "subject"
    BODY = "body"


class EntailmentVerdict(str, Enum):
    """Three-valued semantic fact entailment outcome."""

    ENTAILED = "entailed"          # Claim is logically/factually entailed by research facts
    CONTRADICTED = "contradicted"  # Claim is directly contradicted by research facts
    NEUTRAL = "neutral"            # Claim is unsupported or unobservable from research facts


class ModalityLevel(str, Enum):
    """Epistemic strength and qualification of a proposition."""

    DEFINITIVE = "definitive"      # Unconditional certainty: "is", "will", "guarantees"
    CONDITIONAL = "conditional"    # Prerequisite-dependent: "if", "when configured", "assuming"
    HYPOTHETICAL = "hypothetical"  # Probabilistic: "might", "could", "may experience"
    DIAGNOSTIC = "diagnostic"      # Exploratory inquiry: "are you seeing", "how do you track"


class VerifierFindingSeverity(str, Enum):
    """Severity classification for verifier gate findings."""

    ERROR = "error"      # Hard violation: invalidates verification, triggers rewrite
    WARNING = "warning"  # Soft issue: flags human attention, does not trigger rewrite alone


class VerificationFinding(BaseModel):
    """An individual finding or violation recorded by a verifier gate."""

    model_config = ConfigDict(frozen=True)

    gate: str = Field(description="Name of the verifier gate that produced the finding")
    severity: VerifierFindingSeverity = Field(description="Severity classification")
    code: str = Field(description="Stable programmatic error/warning code")
    message: str = Field(description="Detailed human-readable explanation")
    claim_id: Optional[str] = Field(default=None, description="Associated decomposed claim ID if applicable")
    section: Optional[ClaimSection] = Field(default=None, description="Subject or body")
    text_span: Optional[Tuple[int, int]] = Field(default=None, description="Character offsets (start, end) in section text")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic payload")


class DecomposedClaim(BaseModel):
    """An atomic proposition decomposed from draft text with exact character spans."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(min_length=1, description="Exact substring proposition")
    section: ClaimSection = Field(description="Source draft section (subject or body)")
    span_start: int = Field(ge=0, description="Start character offset in section string")
    span_end: int = Field(ge=0, description="End character offset in section string")
    category: ClaimCategory = Field(description="Semantic category")
    subject: str = Field(default="", description="Subject of proposition")
    predicate: str = Field(default="", description="Predicate or core assertion")
    modality: ModalityLevel = Field(default=ModalityLevel.DEFINITIVE, description="Epistemic modality")
    is_material: bool = Field(default=True, description="Whether assertion is material for verification")
    is_product_claim: bool = Field(default=False, description="True if claim asserts product capability/metric, False if prospect observation")


    def to_atomic_claim(self, sentence_index: Optional[int] = None) -> AtomicClaim:
        """Convert to backwards-compatible AtomicClaim."""
        category_map = {
            ClaimCategory.PROSPECT_OBSERVATION: ClaimType.FACT_ABOUT_COMPANY,
            ClaimCategory.PRODUCT_CAPABILITY: ClaimType.PRODUCT_CLAIM,
            ClaimCategory.METRIC_IMPACT: ClaimType.PRODUCT_CLAIM,
            ClaimCategory.MODALITY_HEDGE: ClaimType.MARKET_INFERENCE,
            ClaimCategory.RELATIONAL_CTA: ClaimType.VALUE_PROPOSITION,
        }
        return AtomicClaim(
            claim_id=self.claim_id,
            claim_text=self.text,
            claim_type=category_map.get(self.category, ClaimType.FACT_ABOUT_COMPANY),
            sentence_index=sentence_index,
        )


class EntailmentCheckResult(BaseModel):
    """Result of evaluating a prospect claim against research facts and evidence."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    verdict: EntailmentVerdict
    matched_fact_ids: List[str] = Field(default_factory=list, description="IDs of corroborating CanonicalFact")
    matched_evidence_ids: List[str] = Field(default_factory=list, description="IDs of corroborating EvidenceRecord")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = Field(description="Entailment reasoning")


class ModalityCheckResult(BaseModel):
    """Result of evaluating epistemic modality against the approved Toulmin argument."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    approved_modality: ModalityLevel
    draft_modality: ModalityLevel
    is_escalated: bool
    rationale: str


class CapabilityCheckResult(BaseModel):
    """Result of evaluating a product claim against the ProductContract."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    capability_id: Optional[str] = None
    capability_name: Optional[str] = None
    is_whitelisted: bool = True
    is_precondition_satisfied: bool = True
    violates_exclusion: bool = False
    unsupported_proof_points: List[str] = Field(default_factory=list)
    rationale: str


class VerifierResult(BaseModel):
    """Consolidated outcome of the four-gate claim and capability verifier."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(description="True if zero ERROR findings exist")
    findings: List[VerificationFinding] = Field(default_factory=list, description="All gate findings")
    decomposed_claims: List[DecomposedClaim] = Field(default_factory=list, description="All extracted atomic claims")
    entailment_results: List[EntailmentCheckResult] = Field(default_factory=list)
    modality_results: List[ModalityCheckResult] = Field(default_factory=list)
    capability_results: List[CapabilityCheckResult] = Field(default_factory=list)
    unverified_claim_ids: List[str] = Field(default_factory=list, description="Claims that could not be verified")
    provenance: Optional[OutputProvenance] = Field(default=None)

    @property
    def has_errors(self) -> bool:
        """True if any finding has ERROR severity."""
        return any(f.severity == VerifierFindingSeverity.ERROR for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        """True if any finding has WARNING severity."""
        return any(f.severity == VerifierFindingSeverity.WARNING for f in self.findings)

    @property
    def error_messages(self) -> List[str]:
        """List of all error finding messages."""
        return [f.message for f in self.findings if f.severity == VerifierFindingSeverity.ERROR]

    def to_claim_verifications(self) -> List[ClaimVerification]:
        """Convert verifier results to backwards-compatible ClaimVerification models."""
        verifications: List[ClaimVerification] = []
        entailment_map = {res.claim_id: res for res in self.entailment_results}
        capability_map = {res.claim_id: res for res in self.capability_results}

        for idx, claim in enumerate(self.decomposed_claims):
            atomic = claim.to_atomic_claim(sentence_index=idx)
            ent = entailment_map.get(claim.claim_id)
            cap = capability_map.get(claim.claim_id)

            status = VerificationStatus.UNSUPPORTED
            fact_ids: List[str] = []
            ev_ids: List[str] = []
            explanation = "Unverified claim"

            if ent:
                if ent.verdict == EntailmentVerdict.ENTAILED:
                    status = VerificationStatus.SUPPORTED
                elif ent.verdict == EntailmentVerdict.CONTRADICTED:
                    status = VerificationStatus.CONTRADICTED
                else:
                    status = VerificationStatus.UNSUPPORTED
                fact_ids = ent.matched_fact_ids
                ev_ids = ent.matched_evidence_ids
                explanation = ent.rationale
            elif cap:
                if cap.violates_exclusion or not cap.is_whitelisted or cap.unsupported_proof_points:
                    status = VerificationStatus.CONTRADICTED if cap.violates_exclusion else VerificationStatus.UNSUPPORTED
                else:
                    status = VerificationStatus.SUPPORTED
                explanation = cap.rationale
            else:
                if claim.category == ClaimCategory.RELATIONAL_CTA:
                    status = VerificationStatus.SUPPORTED
                    explanation = "Relational CTA / closing inquiry"
                elif claim.category == ClaimCategory.MODALITY_HEDGE:
                    status = VerificationStatus.SUPPORTED
                    explanation = "Epistemic condition or hedge"

            verifications.append(
                ClaimVerification(
                    claim=atomic,
                    status=status,
                    supporting_fact_ids=fact_ids,
                    supporting_evidence_ids=ev_ids,
                    explanation=explanation,
                )
            )
        return verifications
