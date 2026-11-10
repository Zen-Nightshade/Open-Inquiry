from __future__ import annotations

import re
from typing import Any, List, Sequence
from pydantic import BaseModel, ConfigDict, Field


class EditDistanceMetrics(BaseModel):
    """Calculated character-level and word-level Levenshtein edit distance metrics."""

    model_config = ConfigDict(frozen=True)

    char_distance: int = Field(ge=0, description="Minimum Unicode character edit distance")
    word_distance: int = Field(ge=0, description="Minimum word token edit distance")
    normalized_char_ratio: float = Field(ge=0.0, le=1.0, description="char_distance / max(len(orig), len(res))")
    normalized_word_ratio: float = Field(ge=0.0, le=1.0, description="word_distance / max(len(orig), len(res))")


WORD_TOKEN_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def tokenize_words(text: str) -> List[str]:
    """Extract deterministic word tokens using Unicode word boundaries."""
    if not text:
        return []
    return WORD_TOKEN_PATTERN.findall(text)


def levenshtein_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """Compute standard minimum Levenshtein edit distance between two sequences with unit costs."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    if seq_a == seq_b:
        return 0

    if len_a < len_b:
        seq_a, seq_b = seq_b, seq_a
        len_a, len_b = len_b, len_a

    previous_row = list(range(len_b + 1))
    current_row = [0] * (len_b + 1)

    for i, elem_a in enumerate(seq_a):
        current_row[0] = i + 1
        for j, elem_b in enumerate(seq_b):
            cost = 0 if elem_a == elem_b else 1
            current_row[j + 1] = min(
                previous_row[j + 1] + 1,  # deletion
                current_row[j] + 1,      # insertion
                previous_row[j] + cost,  # substitution
            )
        previous_row, current_row = current_row, previous_row

    return previous_row[len_b]


def compute_edit_distance(original: str, resulting: str) -> EditDistanceMetrics:
    """Compute character and word Levenshtein distance and normalized ratios between two strings."""
    # Character distance over Unicode code points
    chars_orig = list(original)
    chars_res = list(resulting)
    char_dist = levenshtein_distance(chars_orig, chars_res)
    max_char_len = max(len(chars_orig), len(chars_res))
    norm_char = (char_dist / max_char_len) if max_char_len > 0 else 0.0

    # Word distance over deterministic token list
    words_orig = tokenize_words(original)
    words_res = tokenize_words(resulting)
    word_dist = levenshtein_distance(words_orig, words_res)
    max_word_len = max(len(words_orig), len(words_res))
    norm_word = (word_dist / max_word_len) if max_word_len > 0 else 0.0

    return EditDistanceMetrics(
        char_distance=char_dist,
        word_distance=word_dist,
        normalized_char_ratio=round(norm_char, 6),
        normalized_word_ratio=round(norm_word, 6),
    )


def compute_draft_edit_distance(
    original_subject: str,
    original_body: str,
    resulting_subject: str,
    resulting_body: str,
) -> EditDistanceMetrics:
    """Compute edit distance metrics across full draft text (subject and body)."""
    orig_text = f"{original_subject}\n\n{original_body}".strip()
    res_text = f"{resulting_subject}\n\n{resulting_body}".strip()
    return compute_edit_distance(orig_text, res_text)
