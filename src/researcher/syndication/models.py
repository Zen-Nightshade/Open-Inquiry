from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
from pydantic import BaseModel, ConfigDict, Field


class DependenceType(str, Enum):
    """Classification of the relationship between two sources."""
    EXACT_DUPLICATE = "exact_duplicate"        # 100% identical raw or normalized content
    NEAR_DUPLICATE = "near_duplicate"          # Strong textual overlap (Jaccard >= threshold, e.g. 0.70)
    POSSIBLE_DEPENDENCE = "possible_dependence" # Sub-threshold similarity or same domain, ambiguous
    INDEPENDENT = "independent"                # Distinct reporting across independent sources/domains


class DetectionMethod(str, Enum):
    """Method used to detect or verify source dependence."""
    EXACT_RAW_HASH = "exact_raw_hash"
    EXACT_NORMALIZED_HASH = "exact_normalized_hash"
    MINHASH_LSH = "minhash_lsh"
    SAME_DOMAIN = "same_domain"
    MANUAL = "manual"


class SourceFingerprint(BaseModel):
    """Deterministic cryptographic and locality-sensitive fingerprints for a source document."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    snapshot_id: Optional[str] = None
    raw_content_hash: str = Field(description="SHA-256 of raw text/content")
    normalized_content_hash: str = Field(description="SHA-256 of normalized text")
    minhash_signature: Tuple[int, ...] = Field(
        default=(), description="MinHash signature integers for LSH candidate retrieval"
    )
    shingle_count: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "raw_content_hash": self.raw_content_hash,
            "normalized_content_hash": self.normalized_content_hash,
            "minhash_signature": list(self.minhash_signature),
            "shingle_count": self.shingle_count,
            "token_count": self.token_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceFingerprint:
        sig = data.get("minhash_signature", ())
        if isinstance(sig, list):
            sig = tuple(sig)
        return cls(
            source_id=data["source_id"],
            snapshot_id=data.get("snapshot_id"),
            raw_content_hash=data["raw_content_hash"],
            normalized_content_hash=data["normalized_content_hash"],
            minhash_signature=sig,
            shingle_count=data.get("shingle_count", 0),
            token_count=data.get("token_count", 0),
            created_at=data.get("created_at"),
        )


class SourceDependenceRelationship(BaseModel):
    """Evaluated pairwise dependence relationship between two sources."""

    model_config = ConfigDict(frozen=True)

    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id_a: str
    source_id_b: str
    dependence_type: DependenceType
    detection_method: DetectionMethod
    similarity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Similarity metric (1.0 for exact, verified Jaccard for near-duplicate)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    detected_at: Optional[str] = None


class SyndicationCluster(BaseModel):
    """Cluster of syndicated or duplicate sources sharing a common publication origin."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str = Field(default_factory=lambda: f"syn_{uuid.uuid4().hex[:12]}")
    canonical_source_id: str = Field(
        description="Deterministic representative source ID for the cluster (not an epistemic claim of original wire publisher)"
    )
    member_source_ids: Set[str] = Field(
        description="All source IDs belonging to this syndication cluster (including canonical_source_id)"
    )
    dependence_type: DependenceType = Field(default=DependenceType.NEAR_DUPLICATE)
    primary_detection_method: DetectionMethod = Field(default=DetectionMethod.MINHASH_LSH)
    average_similarity: float = Field(default=1.0, ge=0.0, le=1.0)
    description: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "canonical_source_id": self.canonical_source_id,
            "member_source_ids": sorted(list(self.member_source_ids)),
            "dependence_type": self.dependence_type.value,
            "primary_detection_method": self.primary_detection_method.value,
            "average_similarity": self.average_similarity,
            "description": self.description,
            "created_at": self.created_at,
        }


class SyndicationDetectionResult(BaseModel):
    """Comprehensive output of a syndication detection run over a collection of sources."""

    model_config = ConfigDict(frozen=True)

    clusters: List[SyndicationCluster] = Field(default_factory=list)
    relationships: List[SourceDependenceRelationship] = Field(default_factory=list)
    fingerprints: Dict[str, SourceFingerprint] = Field(default_factory=dict)
    total_sources_evaluated: int = 0
    duplicate_sources_count: int = 0
    independent_sources_count: int = 0

    @property
    def syndication_groups(self) -> Dict[str, Set[str]]:
        """Map of cluster_id -> set of member source_ids for consumer compatibility."""
        return {c.cluster_id: set(c.member_source_ids) for c in self.clusters}

    @property
    def source_to_cluster_map(self) -> Dict[str, str]:
        """Map of source_id -> cluster_id."""
        mapping: Dict[str, str] = {}
        for c in self.clusters:
            for s_id in c.member_source_ids:
                mapping[s_id] = c.cluster_id
        return mapping
