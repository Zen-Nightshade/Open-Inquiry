from __future__ import annotations
import datetime
import uuid
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.models import FactCategory


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
    event_date: Optional[datetime.date] = Field(default=None, description="Consensus or extracted date of the event")
    is_inferred: bool = Field(default=False, description="Whether the signal is an inference rather than direct factual statement")


class SignalScore(BaseModel):
    """Multi-dimensional score evaluating the strength and relevance of a commercial signal."""

    model_config = ConfigDict(frozen=True)

    evidence_strength: float = Field(ge=0.0, le=1.0, description="Confidence and breadth of corroborating evidence")
    recency_score: float = Field(ge=0.0, le=1.0, description="Freshness and timeliness of the signal")
    prospect_relevance: float = Field(ge=0.0, le=1.0, description="Direct relevance to the prospect's role")
    product_alignment: float = Field(ge=0.0, le=1.0, description="Alignment with offering capabilities")
    business_significance: float = Field(ge=0.0, le=1.0, description="Commercial scale/impact for target company")
    composite_score: float = Field(ge=0.0, le=1.0, description="Weighted composite ranking score")


class RankedSignal(BaseModel):
    """A scored commercial signal with ranking justification."""

    model_config = ConfigDict(frozen=True)

    signal: CandidateSignal
    score: SignalScore
    rationale: str = Field(description="Detailed reasoning for this signal's score and priority")


class ProspectProductAlignment(BaseModel):
    """Analysis mapping target company signals and prospect role to product capabilities."""

    model_config = ConfigDict(frozen=True)

    prospect_role_relevance: str = Field(description="Narrative explaining why the prospect cares")
    business_implications: str = Field(description="Commercial and operational impacts on the company")
    capability_mapping: List[str] = Field(min_length=1, description="Product capabilities addressing this opportunity")


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


class StrategyResult(BaseModel):
    """Overall outcome of the signal identification and strategy formulation process."""

    model_config = ConfigDict(frozen=True)

    is_viable: bool = Field(description="Whether a viable, evidence-backed strategy could be formulated")
    strategy: Optional[OutreachStrategy] = Field(default=None, description="The formulated strategy blueprint if viable")
    ranked_signals: List[RankedSignal] = Field(default_factory=list, description="All evaluated candidate signals ranked by priority")
    rejection_reason: Optional[str] = Field(default=None, description="Primary reason why strategy could not be formulated")
    unviable_reasons: List[str] = Field(default_factory=list, description="Detailed diagnostic reasons for non-viability")
