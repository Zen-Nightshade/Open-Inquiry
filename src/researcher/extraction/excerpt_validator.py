from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing all whitespace sequences to a single space and stripping edges."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def validate_excerpt(verbatim_excerpt: str, raw_text: str) -> bool:
    """Deterministically check if verbatim_excerpt is contained in raw_text after whitespace normalization.
    
    Guarantees:
      - Rejects empty, whitespace-only, or fabricated excerpts.
      - Exact substring matching only. Zero fuzzy matching or LLM-based hallucinated acceptance.
    """
    if not verbatim_excerpt or not verbatim_excerpt.strip():
        return False
    if not raw_text or not raw_text.strip():
        return False

    norm_excerpt = normalize_whitespace(verbatim_excerpt)
    norm_raw = normalize_whitespace(raw_text)

    if not norm_excerpt:
        return False

    return norm_excerpt in norm_raw


class ExcerptValidator:
    """Validates extracted evidence excerpts against source snapshot text content."""

    def __init__(self, min_excerpt_length: int = 5) -> None:
        self.min_excerpt_length = min_excerpt_length

    def is_valid_excerpt(self, excerpt: str, raw_text: Optional[str]) -> bool:
        """Validate whether an excerpt exists verbatim within the raw text."""
        if not excerpt or len(excerpt.strip()) < self.min_excerpt_length:
            return False
        if not raw_text:
            return False
        return validate_excerpt(excerpt, raw_text)

    def validate_snapshot_evidence(
        self,
        excerpt: str,
        snapshot: Optional[Dict[str, Any]],
        expected_source_id: Optional[str] = None,
    ) -> bool:
        """Verify that excerpt exists in the snapshot and matches expected source ID if provided."""
        if not snapshot:
            logger.warning("Excerpt validation failed: snapshot is None")
            return False

        # Validate source ownership if source_id is specified
        if expected_source_id and snapshot.get("source_id") != expected_source_id:
            logger.warning(
                "Excerpt validation failed: snapshot source_id %s does not match expected %s",
                snapshot.get("source_id"),
                expected_source_id,
            )
            return False

        raw_text = snapshot.get("raw_text")
        if not raw_text:
            logger.warning("Excerpt validation failed: snapshot raw_text is empty")
            return False

        return self.is_valid_excerpt(excerpt, raw_text)
