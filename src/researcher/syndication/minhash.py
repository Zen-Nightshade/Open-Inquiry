from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple


# Fixed prime for 32-bit MinHash permutations (largest 32-bit prime < 2^32)
_MERSENNE_PRIME = 4294967311
_NUM_HASHES = 64
_NUM_BANDS = 16
_ROWS_PER_BAND = 4  # 16 * 4 = 64

# Deterministically generate random coefficients using a fixed seed for 100% reproducibility
_RNG = random.Random(42)
_A_COEFFS: Tuple[int, ...] = tuple(
    _RNG.randint(1, _MERSENNE_PRIME - 1) | 1 for _ in range(_NUM_HASHES)
)
_B_COEFFS: Tuple[int, ...] = tuple(
    _RNG.randint(0, _MERSENNE_PRIME - 1) for _ in range(_NUM_HASHES)
)


def hash_shingle(shingle: str) -> int:
    """Hash a shingle string to a 32-bit integer."""
    digest = hashlib.md5(shingle.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little")


def compute_minhash_signature(
    shingles: Set[str],
    num_hashes: int = _NUM_HASHES,
) -> Tuple[int, ...]:
    """Compute deterministic MinHash signature for a set of shingles.
    
    If shingles is empty, returns all zeros.
    """
    if not shingles:
        return tuple(0 for _ in range(num_hashes))

    # Initialize signature array with maximum possible 32-bit integers
    signature = [_MERSENNE_PRIME] * num_hashes

    for shingle in shingles:
        x = hash_shingle(shingle)
        for i in range(num_hashes):
            val = (_A_COEFFS[i] * x + _B_COEFFS[i]) % _MERSENNE_PRIME
            if val < signature[i]:
                signature[i] = val

    return tuple(signature)


def estimate_minhash_similarity(
    sig_a: Tuple[int, ...],
    sig_b: Tuple[int, ...]
) -> float:
    """Estimate Jaccard similarity by measuring fraction of matching MinHash signature values."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b) or len(sig_a) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b and a != 0)
    return matches / len(sig_a)


class LSHIndex:
    """Locality-Sensitive Hashing (LSH) index for retrieving near-duplicate candidate pairs."""

    def __init__(
        self,
        num_bands: int = _NUM_BANDS,
        rows_per_band: int = _ROWS_PER_BAND,
    ) -> None:
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        # Mapping from (band_idx, bucket_hash) -> Set[source_id]
        self._buckets: Dict[Tuple[int, int], Set[str]] = {}
        # Mapping from source_id -> signature
        self._signatures: Dict[str, Tuple[int, ...]] = {}

    def insert(self, source_id: str, signature: Tuple[int, ...]) -> None:
        """Insert a document's MinHash signature into the LSH index."""
        self._signatures[source_id] = signature

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            
            # Hash the band slice
            bucket_hash = hash(band_slice)
            key = (band_idx, bucket_hash)
            
            if key not in self._buckets:
                self._buckets[key] = set()
            self._buckets[key].add(source_id)

    def query_candidates(self, source_id: str, signature: Tuple[int, ...]) -> Set[str]:
        """Retrieve all candidate source IDs that collide with signature in at least one band."""
        candidates: Set[str] = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = signature[start:end]
            
            bucket_hash = hash(band_slice)
            key = (band_idx, bucket_hash)
            
            if key in self._buckets:
                candidates.update(self._buckets[key])

        # Remove the source itself from candidates
        candidates.discard(source_id)
        return candidates

    def get_all_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """Return all unique candidate pairs (id_a, id_b) that collide in at least one bucket."""
        candidate_pairs: Set[Tuple[str, str]] = set()

        for bucket_members in self._buckets.values():
            if len(bucket_members) > 1:
                members_list = sorted(list(bucket_members))
                for i in range(len(members_list)):
                    for j in range(i + 1, len(members_list)):
                        candidate_pairs.add((members_list[i], members_list[j]))

        return candidate_pairs
