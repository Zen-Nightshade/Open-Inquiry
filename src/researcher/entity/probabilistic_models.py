from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance
from src.researcher.models import (
    ConfidenceLevel,
    EntityType,
    ResolutionStatus,
    VerificationStatus,
)


class ResolutionOutcome(str, Enum):
    """Explicit four-state identity resolution decision outcome."""

    MATCH = "match"                # Confirmed match with sufficient margin
    NO_MATCH = "no_match"          # Confirmed non-match due to disqualifying or contradictory evidence
    AMBIGUOUS = "ambiguous"        # Multiple viable candidates competing without sufficient margin
    UNRESOLVED = "unresolved"      # Insufficient evidence to decide (not proven distinct)

    def to_resolution_status(self) -> ResolutionStatus:
        """Map resolution outcome to backward-compatible ResolutionStatus."""
        if self == ResolutionOutcome.MATCH:
            return ResolutionStatus.RESOLVED
        elif self == ResolutionOutcome.AMBIGUOUS:
            return ResolutionStatus.AMBIGUOUS
        elif self == ResolutionOutcome.NO_MATCH:
            return ResolutionStatus.UNRESOLVED
        return ResolutionStatus.UNRESOLVED


class FeatureScore(BaseModel):
    """Explainable evaluation of an individual observable feature comparison.

    Represents raw values, calculated similarity, signed contribution to the composite
    match score, and whether the feature is missing (unobserved) or represents negative evidence.
    """

    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(description="Name of the feature, e.g. name_exact, org_domain_match")
    feature_group: str = Field(
        default="general",
        description="Evidence group: name, organization, context, negative",
    )
    raw_target_value: Optional[str] = Field(default=None, description="Observed value from target prospect/entity")
    raw_candidate_value: Optional[str] = Field(default=None, description="Observed value from candidate entity")
    similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized similarity metric in [0.0, 1.0]",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Configured importance weight for this feature",
    )
    contribution: float = Field(
        default=0.0,
        description="Signed contribution to the composite score (+positive or -penalty)",
    )
    is_missing: bool = Field(
        default=False,
        description="True if attribute was unobserved on target or candidate; treated neutrally, NOT as negative evidence",
    )
    is_negative_penalty: bool = Field(
        default=False,
        description="True if feature provides contradictory evidence penalizing candidate score",
    )
    rationale: str = Field(default="", description="Human-readable explanation of the score")


class CandidateMatchAssessment(BaseModel):
    """Auditable candidate pair match evaluation.

    Preserves feature breakdowns, positive and negative evidence, missing features,
    composite match score, and decision rationale.
    """

    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id: str = Field(description="DiscoveredEntityCandidate ID being evaluated")
    target_id: Optional[str] = Field(default=None, description="Target entity ID or prospect identifier")
    candidate_name: str = Field(description="Raw or normalized candidate entity name")
    target_name: Optional[str] = Field(default=None, description="Target entity name")
    entity_type: EntityType = Field(description="Entity type: person, organization, product, location")
    match_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Heuristic composite match score in [0.0, 1.0]; not a calibrated posterior probability",
    )
    feature_scores: List[FeatureScore] = Field(
        default_factory=list,
        description="Individual feature evaluation results",
    )
    positive_evidence: List[str] = Field(
        default_factory=list,
        description="Summary of positive corroborating evidence points",
    )
    negative_evidence: List[str] = Field(
        default_factory=list,
        description="Summary of contradictory or disqualifying evidence points",
    )
    missing_features: List[str] = Field(
        default_factory=list,
        description="Attributes unobserved on one or both records",
    )
    outcome: ResolutionOutcome = Field(description="Individual candidate resolution outcome")
    is_disqualified: bool = Field(
        default=False,
        description="True if candidate has disqualifying negative evidence (e.g. incompatible entity type or conflicting verified domain)",
    )
    rationale: str = Field(description="Explainable rationale for assessment and score")
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RelationshipPredicate(str, Enum):
    """Controlled vocabulary of typed inter-entity relationships."""

    WORKS_FOR = "works_for"
    FOUNDED = "founded"
    SUBSIDIARY_OF = "subsidiary_of"
    PARTNER_OF = "partner_of"
    INVESTOR_IN = "investor_in"
    BOARD_MEMBER = "board_member"
    LEADERSHIP_ROLE = "leadership_role"
    LOCATED_IN = "located_in"


class TypedEntityRelationship(BaseModel):
    """Auditable, typed relationship between two entities with temporal bounds and provenance."""

    model_config = ConfigDict(frozen=True)

    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_entity_id: str = Field(description="Subject entity ID")
    from_entity_name: Optional[str] = Field(default=None, description="Subject entity name")
    to_entity_id: str = Field(description="Object entity ID")
    to_entity_name: Optional[str] = Field(default=None, description="Object entity name")
    predicate: str = Field(description="Typed predicate, e.g. works_for, founded, subsidiary_of")
    valid_from: Optional[str] = Field(
        default=None,
        description="Beginning of validity interval (ISO date or partial YYYY / YYYY-MM)",
    )
    valid_until: Optional[str] = Field(
        default=None,
        description="End of validity interval if terminated; None if ongoing or unknown",
    )
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    verification_status: VerificationStatus = Field(default=VerificationStatus.SUPPORTED)
    source_id: Optional[str] = Field(default=None, description="Source provenance ID")
    supporting_fact_ids: List[str] = Field(default_factory=list, description="IDs of CanonicalFacts evidencing this relationship")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and rule metadata")
    observed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
