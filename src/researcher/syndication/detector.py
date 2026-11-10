from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

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
)
from src.researcher.syndication.shingling import (
    compute_jaccard_similarity,
    generate_5gram_shingles,
    tokenize,
)

logger = logging.getLogger(__name__)


class SyndicationDetector:
    """Three-stage cascade syndication detector:
    1. Deterministic SHA-256 exact matching (raw & normalized).
    2. 5-token-gram MinHash / LSH near-duplicate candidate retrieval.
    3. Exact pairwise Jaccard verification & cluster assignment.
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.70,
        possible_threshold: float = 0.50,
        min_shingles: int = 10,
        num_hashes: int = 64,
        num_bands: int = 16,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        self.jaccard_threshold = (
            similarity_threshold if similarity_threshold is not None else jaccard_threshold
        )
        self.possible_threshold = possible_threshold
        self.min_shingles = min_shingles
        self.num_hashes = num_hashes
        self.num_bands = num_bands

    def fingerprint(
        self,
        source_id: str,
        text: str,
        snapshot_id: Optional[str] = None,
    ) -> SourceFingerprint:
        """Compute and return the SourceFingerprint for a document."""
        return self.fingerprint_document(source_id, text, snapshot_id)[0]

    def fingerprint_document(
        self,
        source_id: str,
        text: str,
        snapshot_id: Optional[str] = None,
    ) -> Tuple[SourceFingerprint, Set[str]]:
        """Compute all cryptographic and locality-sensitive fingerprints for a document.
        
        Returns:
            (SourceFingerprint, Set[5gram_shingles])
        """
        now = datetime.now(timezone.utc).isoformat()
        raw_hash = compute_raw_content_hash(text)
        normalized = normalize_source_text(text)
        norm_hash = compute_normalized_content_hash(text)

        tokens = tokenize(normalized)
        shingles = generate_5gram_shingles(tokens)
        minhash_sig = compute_minhash_signature(shingles, num_hashes=self.num_hashes)

        fp = SourceFingerprint(
            source_id=source_id,
            snapshot_id=snapshot_id,
            raw_content_hash=raw_hash,
            normalized_content_hash=norm_hash,
            minhash_signature=minhash_sig,
            shingle_count=len(shingles),
            token_count=len(tokens),
            created_at=now,
        )
        return fp, shingles

    def detect_syndication(
        self,
        source_texts: Any,
        source_domains: Optional[Dict[str, str]] = None,
        snapshot_ids: Optional[Dict[str, str]] = None,
    ) -> SyndicationDetectionResult:
        """Execute the full detection cascade over a collection of sources.
        
        Args:
            source_texts: Mapping from source_id -> raw or extracted document text,
                or a sequence of dicts/SourceRecords.
            source_domains: Optional mapping from source_id -> domain string.
            snapshot_ids: Optional mapping from source_id -> snapshot_id.
            
        Returns:
            SyndicationDetectionResult with clusters, relationships, and fingerprints.
        """
        if isinstance(source_texts, (list, tuple)):
            converted_texts: Dict[str, str] = {}
            converted_domains: Dict[str, str] = {}
            converted_snapshots: Dict[str, str] = {}
            for item in source_texts:
                if isinstance(item, dict):
                    s_id = item.get("source_id", "")
                    txt = item.get("text") or item.get("raw_text") or ""
                    dom = item.get("domain") or ""
                    snap = item.get("snapshot_id") or ""
                else:
                    s_id = getattr(item, "source_id", "")
                    txt = getattr(item, "text", "") or getattr(item, "raw_text", "")
                    dom = getattr(item, "domain", "")
                    snap = getattr(item, "snapshot_id", "")
                if s_id:
                    converted_texts[s_id] = txt
                    if dom:
                        converted_domains[s_id] = dom
                    if snap:
                        converted_snapshots[s_id] = snap
            source_texts = converted_texts
            if not source_domains and converted_domains:
                source_domains = converted_domains
            if not snapshot_ids and converted_snapshots:
                snapshot_ids = converted_snapshots

        source_ids = sorted(list(source_texts.keys()))
        if not source_ids:
            return SyndicationDetectionResult()

        fingerprints: Dict[str, SourceFingerprint] = {}
        shingles_by_source: Dict[str, Set[str]] = {}

        # 1. Fingerprint all documents
        for s_id in source_ids:
            text = source_texts[s_id]
            snap_id = snapshot_ids.get(s_id) if snapshot_ids else None
            fp, shingles = self.fingerprint_document(s_id, text, snap_id)
            fingerprints[s_id] = fp
            shingles_by_source[s_id] = shingles

        relationships: List[SourceDependenceRelationship] = []
        confirmed_duplicate_pairs: Set[Tuple[str, str]] = set()
        evaluated_pairs: Set[Tuple[str, str]] = set()

        # 2. Stage 1: Exact Duplicate Detection via Hashes
        # A. Raw Content Hash Collisions
        raw_hash_to_sources: Dict[str, List[str]] = {}
        for s_id, fp in fingerprints.items():
            raw_hash_to_sources.setdefault(fp.raw_content_hash, []).append(s_id)

        for raw_h, group in raw_hash_to_sources.items():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair = (group[i], group[j])
                        evaluated_pairs.add(pair)
                        confirmed_duplicate_pairs.add(pair)
                        rel = SourceDependenceRelationship(
                            source_id_a=pair[0],
                            source_id_b=pair[1],
                            dependence_type=DependenceType.EXACT_DUPLICATE,
                            detection_method=DetectionMethod.EXACT_RAW_HASH,
                            similarity_score=1.0,
                            metadata={"hash": raw_h, "scope": "raw_content"},
                            detected_at=datetime.now(timezone.utc).isoformat(),
                        )
                        relationships.append(rel)

        # B. Normalized Content Hash Collisions (ignoring pairs already flagged raw exact)
        norm_hash_to_sources: Dict[str, List[str]] = {}
        for s_id, fp in fingerprints.items():
            norm_hash_to_sources.setdefault(fp.normalized_content_hash, []).append(s_id)

        for norm_h, group in norm_hash_to_sources.items():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair = (group[i], group[j])
                        if pair not in evaluated_pairs:
                            evaluated_pairs.add(pair)
                            confirmed_duplicate_pairs.add(pair)
                            rel = SourceDependenceRelationship(
                                source_id_a=pair[0],
                                source_id_b=pair[1],
                                dependence_type=DependenceType.EXACT_DUPLICATE,
                                detection_method=DetectionMethod.EXACT_NORMALIZED_HASH,
                                similarity_score=1.0,
                                metadata={"hash": norm_h, "scope": "normalized_content"},
                                detected_at=datetime.now(timezone.utc).isoformat(),
                            )
                            relationships.append(rel)

        # 3. Stage 2: Near-Duplicate Candidate Retrieval via LSH
        lsh = LSHIndex(num_bands=self.num_bands, rows_per_band=self.num_hashes // self.num_bands)
        for s_id, fp in fingerprints.items():
            lsh.insert(s_id, fp.minhash_signature)

        candidate_pairs = lsh.get_all_candidate_pairs()

        # 4. Stage 3: Exact Jaccard Verification of Candidates
        for pair in candidate_pairs:
            if pair in evaluated_pairs:
                continue
            evaluated_pairs.add(pair)

            id_a, id_b = pair
            shingles_a = shingles_by_source[id_a]
            shingles_b = shingles_by_source[id_b]

            # Short document guardrail:
            # If both documents are very short, 5-gram Jaccard has high variance
            if len(shingles_a) < self.min_shingles or len(shingles_b) < self.min_shingles:
                # Compare word tokens directly
                tokens_a = set(tokenize(normalize_source_text(source_texts[id_a])))
                tokens_b = set(tokenize(normalize_source_text(source_texts[id_b])))
                word_jaccard = (
                    len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                    if (tokens_a and tokens_b) else 0.0
                )
                if word_jaccard >= self.jaccard_threshold:
                    rel = SourceDependenceRelationship(
                        source_id_a=id_a,
                        source_id_b=id_b,
                        dependence_type=DependenceType.NEAR_DUPLICATE,
                        detection_method=DetectionMethod.MINHASH_LSH,
                        similarity_score=word_jaccard,
                        metadata={
                            "guardrail": "short_document_token_jaccard",
                            "token_jaccard": word_jaccard,
                            "shingles_a": len(shingles_a),
                            "shingles_b": len(shingles_b),
                        },
                        detected_at=datetime.now(timezone.utc).isoformat(),
                    )
                    relationships.append(rel)
                    confirmed_duplicate_pairs.add(pair)
                continue

            # Full 5-gram Jaccard verification
            jaccard = compute_jaccard_similarity(shingles_a, shingles_b)
            tokens_a = set(tokenize(normalize_source_text(source_texts[id_a])))
            tokens_b = set(tokenize(normalize_source_text(source_texts[id_b])))
            word_jaccard = (
                len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                if (tokens_a and tokens_b) else 0.0
            )

            is_confirmed = (jaccard >= self.jaccard_threshold) or (
                jaccard >= self.possible_threshold and word_jaccard >= 0.75
            )

            if is_confirmed:
                rel = SourceDependenceRelationship(
                    source_id_a=id_a,
                    source_id_b=id_b,
                    dependence_type=DependenceType.NEAR_DUPLICATE,
                    detection_method=DetectionMethod.MINHASH_LSH,
                    similarity_score=jaccard,
                    metadata={"jaccard_5gram": jaccard, "word_jaccard": word_jaccard},
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
                relationships.append(rel)
                confirmed_duplicate_pairs.add(pair)
            elif jaccard >= self.possible_threshold or word_jaccard >= 0.60:
                # Sub-threshold similarity: record as possible dependence without forcing into confirmed cluster
                rel = SourceDependenceRelationship(
                    source_id_a=id_a,
                    source_id_b=id_b,
                    dependence_type=DependenceType.POSSIBLE_DEPENDENCE,
                    detection_method=DetectionMethod.MINHASH_LSH,
                    similarity_score=jaccard,
                    metadata={"jaccard_5gram": jaccard, "word_jaccard": word_jaccard, "note": "sub_threshold_candidate"},
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
                relationships.append(rel)

        # 5. Connected Component Cluster Formation
        clusters = self._form_clusters(confirmed_duplicate_pairs, source_ids, relationships)

        # Count metrics
        syndicated_source_ids: Set[str] = set()
        for c in clusters:
            syndicated_source_ids.update(c.member_source_ids)

        return SyndicationDetectionResult(
            clusters=clusters,
            relationships=relationships,
            fingerprints=fingerprints,
            total_sources_evaluated=len(source_ids),
            duplicate_sources_count=len(syndicated_source_ids),
            independent_sources_count=len(source_ids) - len(syndicated_source_ids),
        )

    def _form_clusters(
        self,
        confirmed_pairs: Set[Tuple[str, str]],
        all_source_ids: List[str],
        relationships: List[SourceDependenceRelationship],
    ) -> List[SyndicationCluster]:
        """Group confirmed duplicate pairs into connected components."""
        parent: Dict[str, str] = {s_id: s_id for s_id in all_source_ids}

        def find(u: str) -> str:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u: str, v: str) -> None:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                # Deterministic root: pick lexicographically smaller root
                if root_u < root_v:
                    parent[root_v] = root_u
                else:
                    parent[root_u] = root_v

        for u, v in confirmed_pairs:
            union(u, v)

        # Collect members by root
        groups_by_root: Dict[str, Set[str]] = {}
        for s_id in all_source_ids:
            root = find(s_id)
            groups_by_root.setdefault(root, set()).add(s_id)

        # Filter to multi-member clusters (size > 1)
        clusters: List[SyndicationCluster] = []
        now = datetime.now(timezone.utc).isoformat()

        # Build lookup for pair similarities
        sim_lookup: Dict[Tuple[str, str], float] = {}
        dep_type_lookup: Dict[Tuple[str, str], DependenceType] = {}
        det_method_lookup: Dict[Tuple[str, str], DetectionMethod] = {}
        for r in relationships:
            key = (min(r.source_id_a, r.source_id_b), max(r.source_id_a, r.source_id_b))
            sim_lookup[key] = r.similarity_score
            dep_type_lookup[key] = r.dependence_type
            det_method_lookup[key] = r.detection_method

        for root, members in sorted(groups_by_root.items()):
            if len(members) <= 1:
                continue

            sorted_members = sorted(list(members))
            canonical_id = sorted_members[0]

            # Compute average pairwise similarity in cluster
            sims: List[float] = []
            cluster_dep_type = DependenceType.EXACT_DUPLICATE
            cluster_method = DetectionMethod.EXACT_RAW_HASH

            for i in range(len(sorted_members)):
                for j in range(i + 1, len(sorted_members)):
                    pair_key = (sorted_members[i], sorted_members[j])
                    sim = sim_lookup.get(pair_key, 1.0)
                    sims.append(sim)
                    d_type = dep_type_lookup.get(pair_key, DependenceType.NEAR_DUPLICATE)
                    if d_type == DependenceType.NEAR_DUPLICATE:
                        cluster_dep_type = DependenceType.NEAR_DUPLICATE
                        cluster_method = DetectionMethod.MINHASH_LSH

            avg_sim = sum(sims) / len(sims) if sims else 1.0

            clusters.append(
                SyndicationCluster(
                    canonical_source_id=canonical_id,
                    member_source_ids=members,
                    dependence_type=cluster_dep_type,
                    primary_detection_method=cluster_method,
                    average_similarity=round(avg_sim, 4),
                    description=f"Syndication cluster of {len(members)} documents ({cluster_dep_type.value})",
                    created_at=now,
                )
            )

        return clusters

    def evaluate_pair(
        self,
        source_id_a: str,
        text_a: str,
        source_id_b: str,
        text_b: str,
    ) -> SourceDependenceRelationship:
        """Compare two individual source documents and return their evaluated dependence relationship."""
        now = datetime.now(timezone.utc).isoformat()
        if source_id_a == source_id_b:
            return SourceDependenceRelationship(
                source_id_a=source_id_a,
                source_id_b=source_id_b,
                dependence_type=DependenceType.EXACT_DUPLICATE,
                detection_method=DetectionMethod.EXACT_RAW_HASH,
                similarity_score=1.0,
                metadata={"reason": "identical_source_id"},
                detected_at=now,
            )

        # 1. Raw Hash
        raw_a = compute_raw_content_hash(text_a)
        raw_b = compute_raw_content_hash(text_b)
        if raw_a == raw_b:
            return SourceDependenceRelationship(
                source_id_a=source_id_a,
                source_id_b=source_id_b,
                dependence_type=DependenceType.EXACT_DUPLICATE,
                detection_method=DetectionMethod.EXACT_RAW_HASH,
                similarity_score=1.0,
                metadata={"hash": raw_a},
                detected_at=now,
            )

        # 2. Normalized Hash
        norm_a = compute_normalized_content_hash(text_a)
        norm_b = compute_normalized_content_hash(text_b)
        if norm_a == norm_b:
            return SourceDependenceRelationship(
                source_id_a=source_id_a,
                source_id_b=source_id_b,
                dependence_type=DependenceType.EXACT_DUPLICATE,
                detection_method=DetectionMethod.EXACT_NORMALIZED_HASH,
                similarity_score=1.0,
                metadata={"hash": norm_a},
                detected_at=now,
            )

        # 3. 5-Gram Jaccard Verification
        tokens_a = tokenize(normalize_source_text(text_a))
        tokens_b = tokenize(normalize_source_text(text_b))
        shingles_a = generate_5gram_shingles(tokens_a)
        shingles_b = generate_5gram_shingles(tokens_b)

        if len(shingles_a) < self.min_shingles or len(shingles_b) < self.min_shingles:
            word_jaccard = (
                len(set(tokens_a) & set(tokens_b)) / len(set(tokens_a) | set(tokens_b))
                if (tokens_a and tokens_b) else 0.0
            )
            if word_jaccard >= self.jaccard_threshold:
                return SourceDependenceRelationship(
                    source_id_a=source_id_a,
                    source_id_b=source_id_b,
                    dependence_type=DependenceType.NEAR_DUPLICATE,
                    detection_method=DetectionMethod.MINHASH_LSH,
                    similarity_score=word_jaccard,
                    metadata={"guardrail": "short_document_token_jaccard"},
                    detected_at=now,
                )
            return SourceDependenceRelationship(
                source_id_a=source_id_a,
                source_id_b=source_id_b,
                dependence_type=DependenceType.INDEPENDENT,
                detection_method=DetectionMethod.MINHASH_LSH,
                similarity_score=word_jaccard,
                metadata={"guardrail": "short_document_insufficient_shingles"},
                detected_at=now,
            )

        jaccard = compute_jaccard_similarity(shingles_a, shingles_b)
        tok_set_a = set(tokens_a)
        tok_set_b = set(tokens_b)
        word_jaccard = (
            len(tok_set_a & tok_set_b) / len(tok_set_a | tok_set_b)
            if (tok_set_a and tok_set_b) else 0.0
        )

        is_near_dup = (jaccard >= self.jaccard_threshold) or (
            jaccard >= self.possible_threshold and word_jaccard >= 0.75
        )

        if is_near_dup:
            dep_type = DependenceType.NEAR_DUPLICATE
        elif jaccard >= self.possible_threshold or word_jaccard >= 0.60:
            dep_type = DependenceType.POSSIBLE_DEPENDENCE
        else:
            dep_type = DependenceType.INDEPENDENT

        return SourceDependenceRelationship(
            source_id_a=source_id_a,
            source_id_b=source_id_b,
            dependence_type=dep_type,
            detection_method=DetectionMethod.MINHASH_LSH,
            similarity_score=jaccard,
            metadata={"jaccard_5gram": jaccard, "word_jaccard": word_jaccard},
            detected_at=now,
        )
