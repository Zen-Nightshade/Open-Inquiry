from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SignalScoreResponse(BaseModel):
    evidence_strength: float = Field(..., description="Corroboration strength")
    recency_score: float = Field(..., description="Recency score")
    prospect_relevance: float = Field(..., description="Relevance to prospect role")
    product_alignment: float = Field(..., description="Alignment with offering capabilities")
    business_significance: float = Field(..., description="Scale/impact for target company")
    composite_score: float = Field(..., description="Weighted composite score")


class CandidateSignalResponse(BaseModel):
    signal_id: str = Field(..., description="Unique signal ID")
    label: str = Field(..., description="Signal label")
    description: str = Field(..., description="Detailed description")
    supporting_fact_ids: List[str] = Field(..., description="Supporting fact IDs")
    fact_category: str = Field(..., description="Primary fact category")
    associated_entity_ids: List[str] = Field(default_factory=list, description="Associated entities")
    observed_event: Optional[str] = Field(None, description="Underlying event")
    business_implication: Optional[str] = Field(None, description="Company implication")
    commercial_opportunity: Optional[str] = Field(None, description="Product opening")
    event_date: Optional[str] = Field(None, description="Date of observed event")
    is_inferred: bool = Field(False, description="Whether signal was inferred")


class RankedSignalResponse(BaseModel):
    signal: CandidateSignalResponse = Field(..., description="The candidate signal")
    score: SignalScoreResponse = Field(..., description="Signal multidimensional score")
    rationale: str = Field(..., description="Scoring and ranking rationale")


class AlignmentResponse(BaseModel):
    prospect_role_relevance: str = Field(..., description="Why the prospect cares")
    business_implications: str = Field(..., description="Impact on the company")
    capability_mapping: List[str] = Field(default_factory=list, description="Product capabilities addressing this")


class StrategyInspectionResponse(BaseModel):
    run_id: str = Field(..., description="Application run identifier")
    session_id: Optional[str] = Field(None, description="Associated research session ID")
    prospect_name: str = Field(..., description="Prospect name")
    company_name: str = Field(..., description="Company name")
    messaging_approach: str = Field(..., description="Selected strategic framing")
    central_message: str = Field(..., description="Core one-sentence message")
    suggested_subject_line: str = Field(..., description="Recommended subject line")
    cta: str = Field(..., description="Low-friction call to action")
    tone_guidance: str = Field(..., description="Tone guidance")
    primary_signal: Dict[str, Any] = Field(..., description="Lead signal details")
    supporting_signals: List[Dict[str, Any]] = Field(default_factory=list, description="Secondary context signals")
    alignment: Dict[str, Any] = Field(..., description="Prospect-product capability alignment")
    supporting_facts: List[Dict[str, Any]] = Field(default_factory=list, description="Underlying canonical facts")
    evidence_chain: List[Dict[str, Any]] = Field(default_factory=list, description="Passage-level provenance chain")
    rejection_reasons: List[str] = Field(default_factory=list, description="Reasons if strategy was not viable")
