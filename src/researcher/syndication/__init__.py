from __future__ import annotations

from src.researcher.syndication.detector import SyndicationDetector
from src.researcher.syndication.minhash import (
    LSHIndex,
    compute_minhash_signature,
    estimate_minhash_similarity,
)
from src.researcher.syndication.models import (
    DependenceType,
    DetectionMethod,
    SourceDependenceRelationship,
    SourceFingerprint,
    SyndicationCluster,
    SyndicationDetectionResult,
)
from src.researcher.syndication.normalizer import (
    compute_normalized_content_hash,
    compute_raw_content_hash,
    normalize_source_text,
    strip_html_tags,
)
from src.researcher.syndication.shingling import (
    compute_jaccard_similarity,
    generate_5gram_shingles,
    tokenize,
)

__all__ = [
    "SyndicationDetector",
    "DependenceType",
    "DetectionMethod",
    "SourceFingerprint",
    "SourceDependenceRelationship",
    "SyndicationCluster",
    "SyndicationDetectionResult",
    "normalize_source_text",
    "strip_html_tags",
    "compute_raw_content_hash",
    "compute_normalized_content_hash",
    "tokenize",
    "generate_5gram_shingles",
    "compute_jaccard_similarity",
    "compute_minhash_signature",
    "estimate_minhash_similarity",
    "LSHIndex",
]
