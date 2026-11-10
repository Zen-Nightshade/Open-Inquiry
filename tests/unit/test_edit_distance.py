from __future__ import annotations

import pytest

from src.email_generation.edit_distance import (
    EditDistanceMetrics,
    compute_draft_edit_distance,
    compute_edit_distance,
    levenshtein_distance,
    tokenize_words,
)


def test_identical_strings():
    text = "The quick brown fox jumps over the lazy dog."
    metrics = compute_edit_distance(text, text)
    assert metrics.char_distance == 0
    assert metrics.word_distance == 0
    assert metrics.normalized_char_ratio == 0.0
    assert metrics.normalized_word_ratio == 0.0


def test_both_empty_strings():
    metrics = compute_edit_distance("", "")
    assert metrics.char_distance == 0
    assert metrics.word_distance == 0
    assert metrics.normalized_char_ratio == 0.0
    assert metrics.normalized_word_ratio == 0.0


def test_one_empty_string():
    text = "Hello world"
    metrics = compute_edit_distance("", text)
    assert metrics.char_distance == len(text)
    assert metrics.word_distance == 2
    assert metrics.normalized_char_ratio == 1.0
    assert metrics.normalized_word_ratio == 1.0

    metrics_rev = compute_edit_distance(text, "")
    assert metrics_rev.char_distance == len(text)
    assert metrics_rev.word_distance == 2
    assert metrics_rev.normalized_char_ratio == 1.0
    assert metrics_rev.normalized_word_ratio == 1.0


def test_single_character_substitution():
    orig = "kitten"
    res = "sitten"
    metrics = compute_edit_distance(orig, res)
    assert metrics.char_distance == 1
    assert metrics.normalized_char_ratio == round(1 / 6, 6)
    assert metrics.word_distance == 1  # 1 word substituted
    assert metrics.normalized_word_ratio == 1.0


def test_word_insertion_and_deletion():
    orig = "We can improve latency."
    res = "We can drastically improve cluster latency."
    # Words in orig: ["We", "can", "improve", "latency"] (len 4)
    # Words in res: ["We", "can", "drastically", "improve", "cluster", "latency"] (len 6)
    # Inserted 2 words: "drastically", "cluster"
    metrics = compute_edit_distance(orig, res)
    assert metrics.word_distance == 2
    assert metrics.normalized_word_ratio == round(2 / 6, 6)


def test_unicode_and_accents():
    orig = "Café au lait"
    res = "Cafe au lait"
    metrics = compute_edit_distance(orig, res)
    assert metrics.char_distance == 1  # 'é' vs 'e'
    assert metrics.word_distance == 1  # 'Café' vs 'Cafe'


def test_contractions_and_punctuation():
    orig = "Don't hesitate to check our team's report."
    res = "Do not hesitate to check our team's report."
    words_orig = tokenize_words(orig)
    words_res = tokenize_words(res)
    # Don't is 1 token with contraction pattern \b[\w']+\b
    assert "Don't" in words_orig
    assert "team's" in words_orig
    assert "team's" in words_res
    metrics = compute_edit_distance(orig, res)
    assert metrics.word_distance >= 1


def test_compute_draft_edit_distance():
    orig_sub = "Meeting tomorrow"
    orig_body = "Hi Alice, would love to connect about latency."
    res_sub = "Discussion tomorrow"
    res_body = "Hi Alice, would love to share benchmarks about cluster latency."

    metrics = compute_draft_edit_distance(orig_sub, orig_body, res_sub, res_body)
    assert metrics.char_distance > 0
    assert metrics.word_distance > 0
    assert 0.0 < metrics.normalized_char_ratio < 1.0
    assert 0.0 < metrics.normalized_word_ratio < 1.0


def test_levenshtein_distance_empty_and_disjoint():
    assert levenshtein_distance([], []) == 0
    assert levenshtein_distance([1, 2, 3], []) == 3
    assert levenshtein_distance([], [1, 2, 3]) == 3
    assert levenshtein_distance([1, 2, 3], [4, 5, 6]) == 3
