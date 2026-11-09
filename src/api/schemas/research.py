from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceInfoResponse(BaseModel):
    source_id: str = Field(..., description="Unique source identifier")
    canonical_url: str = Field(..., description="Canonical source URL")
    domain: Optional[str] = Field(None, description="Web domain of source")
    source_type: Optional[str] = Field(None, description="Type or classification of source")


class EvidencePassageResponse(BaseModel):
    evidence_id: str = Field(..., description="Unique evidence passage ID")
    verbatim_excerpt: str = Field(..., description="Verbatim text quote from source document")
    excerpt_validated: bool = Field(..., description="Whether excerpt was verified against raw source content")
    credibility_score: float = Field(..., description="Assessed credibility score of this evidence passage")
    event_date: Optional[str] = Field(None, description="Extracted date of event described in passage")
    source: Optional[SourceInfoResponse] = Field(None, description="Originating source document metadata")


class FactWithEvidenceResponse(BaseModel):
    fact_id: str = Field(..., description="Canonical fact identifier")
    fact_category: str = Field(..., description="Categorization of fact (e.g. company_trajectory, funding)")
    claim_text: str = Field(..., description="Extracted factual assertion statement")
    composite_confidence: str = Field(..., description="Confidence rating ('low', 'medium', 'high')")
    primary_event_date: Optional[str] = Field(None, description="Primary date associated with fact")
    created_at: Optional[str] = Field(None, description="Timestamp fact was recorded")
    evidence: List[EvidencePassageResponse] = Field(default_factory=list, description="Linked corroborating evidence passages")


class IdentityResponse(BaseModel):
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(..., description="Type of entity ('person', 'organization', etc.)")
    raw_name: str = Field(..., description="Extracted name variation")
    canonical_name: Optional[str] = Field(None, description="Resolved canonical identity name")
    resolution_status: str = Field(..., description="Resolution status ('MATCHED', 'AMBIGUOUS', etc.)")
    confidence: float = Field(..., description="Resolution confidence score")
    rationale: str = Field(..., description="Justification for entity resolution decision")
    assessed_by: Optional[str] = Field(None, description="Agent or module that performed assessment")
    assessed_at: Optional[str] = Field(None, description="Assessment timestamp")


class ConflictResponse(BaseModel):
    conflict_id: str = Field(..., description="Conflict identifier")
    severity: Optional[str] = Field(None, description="Conflict severity level")
    description: str = Field(..., description="Description of the contradiction")
    status: str = Field(..., description="Resolution status ('open', 'resolved', etc.)")
    created_at: Optional[str] = Field(None, description="Timestamp conflict was detected")


class ResearchInspectionResponse(BaseModel):
    session_id: str = Field(..., description="Research session identifier")
    run_id: Optional[str] = Field(None, description="Associated application run identifier")
    status: str = Field(..., description="Session status")
    is_sufficient: bool = Field(..., description="Whether research achieved sufficiency criteria")
    is_complete: bool = Field(..., description="Whether research session finished completely")
    sufficiency_notes: Optional[str] = Field(None, description="Sufficiency evaluation explanation")
    started_at: Optional[str] = Field(None, description="Research start timestamp")
    completed_at: Optional[str] = Field(None, description="Research completion timestamp")
    objectives: List[Dict[str, Any]] = Field(default_factory=list, description="Target research objectives")
    facts: List[FactWithEvidenceResponse] = Field(default_factory=list, description="Canonical facts with provenance chain")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="Detected contradictory statements")
    identities: List[IdentityResponse] = Field(default_factory=list, description="Resolved entities and identities")
    iterations: List[Dict[str, Any]] = Field(default_factory=list, description="Research iteration history")
    latest_assessment: Optional[Dict[str, Any]] = Field(None, description="Latest orchestration assessment")
