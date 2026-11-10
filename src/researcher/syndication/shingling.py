from __future__ import annotations

import re
from typing import List, Set


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric and metric word tokens."""
    if not text:
        return []
    # Match words, numbers, currencies, percentages
    tokens = re.findall(r"\$?\b\w+(?:[.,]\w+)*%?", text.lower())
    return [t for t in tokens if t]


def generate_5gram_shingles(tokens: List[str]) -> Set[str]:
    """Generate 5-token-gram shingles using a sliding window.
    
    If text contains fewer than 5 tokens, safely falls back to a single shingle
    encompassing the entire token sequence to avoid losing short texts.
    """
    if not tokens:
        return set()

    if len(tokens) < 5:
        return {" ".join(tokens)}

    shingles: Set[str] = set()
    for i in range(len(tokens) - 4):
        shingles.add(" ".join(tokens[i : i + 5]))
    return shingles


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate the exact Jaccard similarity between two shingle sets: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union
