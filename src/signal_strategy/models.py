from __future__ import annotations
import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance
from src.context.models import CompatibilityResult, CompatibilityStatus
from src.researcher.models import ConfidenceLevel, FactCategory, VerificationStatus


class GateIdentifier(str, Enum):
    """The four non-compensatory eligibility gates."""

    IDENTITY = "identity"
    EVIDENCE = "evidence"
    PRODUCT_COMPATIBILITY = "product_compatibility"
    PERSONA = "persona"


class EligibilityStatus(str, Enum):
    """Outcome of evaluating non-compensatory signal eligibility."""

    ELIGIBLE = "eligible"
    CONDITIONAL = "conditional"
    INELIGIBLE = "ineligible"


class GateResult(BaseModel):
    """Result of evaluating an individual non-compensatory gate."""

    model_config = ConfigDict(frozen=True)

    gate: GateIdentifier = Field(description="Gate classification")
    passed: bool = Field(description="Whether the signal satisfied this gate")
    reason: str = Field(description="Explanation of gate determination")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic details or conditions")


class SignalEligibility(BaseModel):
    """Authoritative non-compensatory eligibility evaluation for a candidate signal."""

    model_config = ConfigDict(frozen=True)

    status: EligibilityStatus = Field(description="Overall eligibility classification")
    is_eligible: bool = Field(description="True if ELIGIBLE; False if CONDITIONAL or INELIGIBLE")
    identity_gate: GateResult = Field(description="Prospect/entity identity validity evaluation")
    evidence_gate: GateResult = Field(description="Supporting evidence adequacy evaluation")
    product_gate: GateResult = Field(description="Product compatibility evaluation via ProductContract")
    persona_gate: GateResult = Field(description="Prospect persona / buyer relevance evaluation")
    compatibility_result: Optional[CompatibilityResult] = Field(default=None, description="Detailed product compatibility result")
    failed_gates: List[GateIdentifier] = Field(default_factory=list, description="List of gates that failed")
    rejection_reasons: List[str] = Field(default_factory=list, description="Detailed reasons for failure")
    unresolved_conditions: List[str] = Field(default_factory=list, description="Unresolved conditions for conditional eligibility")
    matched_capability_name: Optional[str] = Field(default=None, description="Matched product capability name if applicable")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Provenance of eligibility evaluation")

    @property
    def is_verified_eligible(self) -> bool:
        """True if signal passed all gates and is verified compatible."""
        return self.status == EligibilityStatus.ELIGIBLE

    @property
    def requires_diagnostic_strategy(self) -> bool:
        """True if signal has no hard incompatibilities but depends on unverified conditions."""
        return self.status == EligibilityStatus.CONDITIONAL

    @property
    def is_ineligible(self) -> bool:
        """True if signal failed one or more hard eligibility gates."""
        return self.status == EligibilityStatus.INELIGIBLE

    @property
    def matched_capability_id(self) -> Optional[str]:
        """Matched capability ID from compatibility evaluation if available."""
        if self.compatibility_result and self.compatibility_result.matched_capability_id:
            return self.compatibility_result.matched_capability_id
        return None


class CandidateSignal(BaseModel):
    """A synthesized commercial signal derived from one or more supporting facts."""

    model_config = ConfigDict(frozen=True)

    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(description="Short identifier, e.g. 'Series B Expansion'")
    description: str = Field(description="Detailed description of the observed commercial event")
    supporting_fact_ids: List[str] = Field(min_length=1, description="IDs of CanonicalFact supporting this signal")
    fact_category: FactCategory = Field(default=FactCategory.OTHER)
    associated_entity_ids: List[str] = Field(default_factory=list)
    observed_event: Optional[str] = Field(default=None, description="The concrete factual event that took place")
    business_implication: Optional[str] = Field(default=None, description="Operational or business consequences for the company")
    commercial_opportunity: Optional[str] = Field(default=None, description="Why this creates an opening for the product")
    target_capability: Optional[str] = Field(default=None, description="Specific ProductContract capability targeted by this signal")
    event_date: Optional[datetime.date] = Field(default=None, description="Consensus or extracted date of the event")
    is_inferred: bool = Field(default=False, description="Whether the signal is an inference rather than direct factual statement")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")


class SignalScore(BaseModel):
    """Multi-dimensional score evaluating the strength and relevance of a commercial signal."""

    model_config = ConfigDict(frozen=True)

    evidence_strength: float = Field(ge=0.0, le=1.0, description="Confidence and breadth of corroborating evidence")
    recency_score: float = Field(ge=0.0, le=1.0, description="Freshness and timeliness of the signal")
    prospect_relevance: float = Field(ge=0.0, le=1.0, description="Direct relevance to the prospect's role")
    product_alignment: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Alignment with offering capabilities (None if unevaluated)")
    business_significance: float = Field(ge=0.0, le=1.0, description="Commercial scale/impact for target company")
    composite_score: float = Field(ge=0.0, le=1.0, description="Weighted composite ranking score")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")


class RankedSignal(BaseModel):
    """A scored commercial signal with ranking justification and eligibility determination."""

    model_config = ConfigDict(frozen=True)

    signal: CandidateSignal
    score: SignalScore
    rationale: str = Field(description="Detailed reasoning for this signal's score and priority")
    eligibility: Optional[SignalEligibility] = Field(default=None, description="Deterministic eligibility gate assessment")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")


class ProspectProductAlignment(BaseModel):
    """Analysis mapping target company signals and prospect role to product capabilities."""

    model_config = ConfigDict(frozen=True)

    prospect_role_relevance: str = Field(description="Narrative explaining why the prospect cares")
    business_implications: str = Field(description="Commercial and operational impacts on the company")
    capability_mapping: List[str] = Field(min_length=1, description="Product capabilities addressing this opportunity")
    is_conditional: bool = Field(default=False, description="Whether alignment is conditional on unverified conditions")
    unresolved_conditions: List[str] = Field(default_factory=list, description="Unresolved conditions if alignment is conditional")


class CTAType(str, Enum):
    """Deterministic call-to-action classification based on epistemic certainty."""

    VALUE_EXCHANGE = "VALUE_EXCHANGE"
    BINARY_CHECK = "BINARY_CHECK"
    DIAGNOSTIC_QUESTION = "DIAGNOSTIC_QUESTION"


class ToulminGrounds(BaseModel):
    """Verified observations and facts about the prospect or organization."""

    model_config = ConfigDict(frozen=True)

    fact_ids: List[str] = Field(min_length=1, description="IDs of CanonicalFact from ResearchPackage")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of EvidenceRecord supporting the facts")
    observed_event: str = Field(min_length=1, description="Concrete factual event observed")
    verification_status: VerificationStatus = Field(default=VerificationStatus.SUPPORTED, description="Verification status of supporting facts")
    composite_confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Composite confidence level of supporting facts")
    source_domains: List[str] = Field(default_factory=list, description="Corroborating source domains (descriptive metadata only)")


class ToulminWarrant(BaseModel):
    """Reasoning connecting prospect evidence to a possible business problem and product relevance."""

    model_config = ConfigDict(frozen=True)

    capability_id: str = Field(min_length=1, description="Authoritative CapabilityNode ID from ProductContract")
    capability_name: str = Field(min_length=1, description="Unambiguous name of the product capability")
    reasoning: str = Field(min_length=1, description="Logical bridge explaining why the observed event creates operational/business need")
    is_inferred: bool = Field(default=False, description="Distinguishes documented functionality from inferred relevance")
    limitations: List[str] = Field(default_factory=list, description="Preserved reasoning boundaries and limitations")


class ToulminBacking(BaseModel):
    """Evidence supporting the product capability or the warrant."""

    model_config = ConfigDict(frozen=True)

    capability_id: str = Field(min_length=1, description="Authoritative CapabilityNode ID from ProductContract")
    whitelisted_proof_points: List[str] = Field(default_factory=list, description="Whitelisted proof points from CapabilityNode.verified_proof_points")
    has_whitelisted_proof: bool = Field(default=False, description="Explicit indicator whether whitelisted proof exists")
    provenance: OutputProvenance = Field(description="Preserved provenance and verification state of proof points")


class ToulminQualifier(BaseModel):
    """Strength, epistemic state, and limitations of the conclusion."""

    model_config = ConfigDict(frozen=True)

    eligibility_status: EligibilityStatus = Field(description="Eligibility status of the signal")
    compatibility_status: CompatibilityStatus = Field(description="Product compatibility evaluation status")
    verification_status: VerificationStatus = Field(description="Fact verification state")
    composite_confidence: ConfidenceLevel = Field(description="Supporting evidence quality/confidence")
    unresolved_prerequisites: List[str] = Field(default_factory=list, description="Unresolved technical conditions or prerequisites")
    qualification_language: str = Field(min_length=1, description="Explicit linguistic hedge matching epistemic state")


class ToulminClaim(BaseModel):
    """Proposed commercial proposition or diagnostic question for outreach."""

    model_config = ConfigDict(frozen=True)

    claim_type: str = Field(description="'value_assertion' for verified, 'diagnostic_inquiry' for conditional")
    text: str = Field(min_length=1, description="Core proposition statement or exploratory question")
    target_capability: str = Field(min_length=1, description="Product capability addressed")
    is_qualified: bool = Field(default=True, description="Whether claim reflects qualifier limitations")


class ToulminRebuttal(BaseModel):
    """Actual limitations, counterarguments, and boundary conditions."""

    model_config = ConfigDict(frozen=True)

    contradictory_fact_ids: List[str] = Field(default_factory=list, description="References to contradictory CanonicalFact IDs")
    unresolved_prerequisites: List[str] = Field(default_factory=list, description="References to unestablished prerequisite conditions")
    hard_exclusions: List[str] = Field(default_factory=list, description="References to violated or boundary hard exclusions")
    missing_backing: bool = Field(default=False, description="Explicit indicator if whitelisted proof points are missing")
    limitations: List[str] = Field(default_factory=list, description="Documented product or operational limitations")


class ToulminArgument(BaseModel):
    """Structured 6-part Toulmin commercial argument blueprint."""

    model_config = ConfigDict(frozen=True)

    grounds: ToulminGrounds
    warrant: ToulminWarrant
    backing: ToulminBacking
    qualifier: ToulminQualifier
    claim: ToulminClaim
    rebuttal: ToulminRebuttal
    provenance: Optional[OutputProvenance] = Field(default=None, description="Producing component provenance")


class OutreachStrategy(BaseModel):
    """Structured, immutable blueprint directing the email generation stage."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    primary_signal: RankedSignal = Field(description="Lead hook and reason for contact")
    supporting_signals: List[RankedSignal] = Field(default_factory=list, description="Secondary context signals")
    alignment: ProspectProductAlignment = Field(description="Prospect-product capability alignment")
    messaging_approach: str = Field(description="Strategic framing (e.g. challenge-led, event-hook, peer-expansion)")
    central_message: str = Field(description="One-sentence core proposition")
    suggested_subject_line: str = Field(description="Recommended email subject")
    cta: str = Field(description="Specific, low-friction next action")
    tone_guidance: str = Field(description="Tone guidelines matching context")
    constraint_checklist: List[str] = Field(default_factory=list, description="Explicit constraints to enforce")
    is_conditional: bool = Field(default=False, description="Whether strategy is based on conditional product compatibility")
    unresolved_conditions: List[str] = Field(default_factory=list, description="Conditions that remain unverified and must be qualified")
    toulmin_argument: Optional[ToulminArgument] = Field(default=None, description="Authoritative 6-part Toulmin commercial argument")
    cta_type: Optional[CTAType] = Field(default=None, description="Deterministic CTA type")
    cta_rationale: Optional[str] = Field(default=None, description="Deterministic rationale for the selected CTA type")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")


class StrategyResult(BaseModel):
    """Overall outcome of the signal identification and strategy formulation process."""

    model_config = ConfigDict(frozen=True)

    is_viable: bool = Field(description="Whether a viable, evidence-backed strategy could be formulated")
    strategy: Optional[OutreachStrategy] = Field(default=None, description="The formulated strategy blueprint if viable")
    ranked_signals: List[RankedSignal] = Field(default_factory=list, description="All evaluated candidate signals ranked by priority")
    rejection_reason: Optional[str] = Field(default=None, description="Primary reason why strategy could not be formulated")
    unviable_reasons: List[str] = Field(default_factory=list, description="Detailed diagnostic reasons for non-viability")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")

    @property
    def toulmin_argument(self) -> Optional[ToulminArgument]:
        """Authoritative Toulmin commercial argument delegated directly from strategy."""
        return self.strategy.toulmin_argument if self.strategy else None

    @property
    def allowed_cta_friction(self) -> Optional[str]:
        """Allowed CTA type string matching contract."""
        if self.strategy and self.strategy.cta_type:
            return self.strategy.cta_type.value
        return None

    @property
    def target_capability_id(self) -> Optional[str]:
        """Matched capability ID from Toulmin argument."""
        if self.strategy and self.strategy.toulmin_argument:
            return self.strategy.toulmin_argument.warrant.capability_id
        return None
