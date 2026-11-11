from __future__ import annotations

from src.researcher.entity.candidates import (
    CandidateObservation,
    DiscoveredEntityCandidate,
    EntityCandidateGenerator,
    clean_domain,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.entity.probabilistic_models import (
    CandidateMatchAssessment,
    FeatureScore,
    RelationshipPredicate,
    ResolutionOutcome,
    TypedEntityRelationship,
)
from src.researcher.entity.probabilistic_resolver import (
    ProbabilisticEntityResolver,
    ProbabilisticResolutionResult,
)
from src.researcher.entity.probabilistic_scorer import (
    ProbabilisticCandidateScorer,
    are_locations_compatible,
    are_names_nickname_match,
    compute_levenshtein_similarity,
    compute_token_jaccard,
    is_acronym_match,
)
from src.researcher.entity.resolver import (
    AffiliationData,
    IdentifierData,
    IdentityAssessmentOutput,
    IdentityResolutionResult,
    IdentityResolver,
    RelationshipData,
)
from src.researcher.entity.service import EntityResolutionService

__all__ = [
    "CandidateObservation",
    "DiscoveredEntityCandidate",
    "EntityCandidateGenerator",
    "clean_domain",
    "normalize_person_name",
    "normalize_org_name",
    "AffiliationData",
    "IdentifierData",
    "RelationshipData",
    "IdentityAssessmentOutput",
    "IdentityResolutionResult",
    "IdentityResolver",
    "EntityResolutionService",
    "ResolutionOutcome",
    "FeatureScore",
    "CandidateMatchAssessment",
    "RelationshipPredicate",
    "TypedEntityRelationship",
    "ProbabilisticCandidateScorer",
    "ProbabilisticEntityResolver",
    "ProbabilisticResolutionResult",
    "are_names_nickname_match",
    "compute_levenshtein_similarity",
    "compute_token_jaccard",
    "is_acronym_match",
    "are_locations_compatible",
]
