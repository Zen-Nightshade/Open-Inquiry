from __future__ import annotations

import pytest

from src.researcher.extraction.excerpt_validator import (
    ExcerptValidator,
    normalize_whitespace,
    validate_excerpt,
)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("line1\n\n\tline2 \r\n line3") == "line1 line2 line3"
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""


def test_exact_excerpt_accepted():
    raw_text = "Acme Corp announced an expansion into Europe today with a new London headquarters."
    excerpt = "Acme Corp announced an expansion into Europe today"
    assert validate_excerpt(excerpt, raw_text) is True


def test_whitespace_normalized_excerpt_accepted():
    raw_text = """
    Acme Corp announced a $50M Series B
    financing round led by Tech Capital Partners,
    with participation from existing investors.
    """
    excerpt = "Acme Corp announced a $50M Series B financing round led by Tech Capital Partners,"
    assert validate_excerpt(excerpt, raw_text) is True


def test_fabricated_excerpt_rejected():
    raw_text = "Cyberdyne Systems reported strong revenue growth in the third quarter of 2026."
    # Fabricated / altered word ("exceptional" instead of "strong")
    fabricated = "Cyberdyne Systems reported exceptional revenue growth in the third quarter"
    assert validate_excerpt(fabricated, raw_text) is False

    # Entirely made up text
    completely_fake = "Cyberdyne Systems was acquired by Skynet for $10 billion"
    assert validate_excerpt(completely_fake, raw_text) is False


def test_validator_class_snapshot_validation():
    validator = ExcerptValidator(min_excerpt_length=10)
    snapshot = {
        "snapshot_id": "snap_1",
        "source_id": "src_1",
        "raw_text": "Helios Technologies released its Q2 earnings report showing 40% growth.",
    }

    # Valid excerpt and correct source_id
    assert (
        validator.validate_snapshot_evidence(
            excerpt="released its Q2 earnings report showing 40% growth",
            snapshot=snapshot,
            expected_source_id="src_1",
        )
        is True
    )

    # Nonexistent snapshot rejected
    assert (
        validator.validate_snapshot_evidence(
            excerpt="released its Q2 earnings report",
            snapshot=None,
        )
        is False
    )

    # Excerpt from different source_id rejected
    assert (
        validator.validate_snapshot_evidence(
            excerpt="released its Q2 earnings report",
            snapshot=snapshot,
            expected_source_id="different_source_999",
        )
        is False
    )

    # Empty raw_text rejected
    assert (
        validator.validate_snapshot_evidence(
            excerpt="released its Q2 earnings report",
            snapshot={"source_id": "src_1", "raw_text": ""},
        )
        is False
    )


def test_empty_and_short_excerpts_rejected():
    validator = ExcerptValidator(min_excerpt_length=5)
    raw = "Valid long document text goes here for testing."

    assert validator.is_valid_excerpt("", raw) is False
    assert validator.is_valid_excerpt("   ", raw) is False
    assert validator.is_valid_excerpt("hi", raw) is False  # below min length
    assert validator.is_valid_excerpt("Valid long document", "") is False
