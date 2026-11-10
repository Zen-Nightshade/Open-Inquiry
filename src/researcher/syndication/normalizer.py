from __future__ import annotations

import hashlib
import html
import re
from typing import Optional


# Common syndication wire boilerplate patterns
_WIRE_PREFIX_PATTERNS = [
    r"^(?:\(?(?:pr\s*newswire|business\s*wire|globenewswire|marketwired|accesswire|prnewswire)\)?\s*(?:\([a-z\s]+\))?\s*[-—–:/|]+\s*)+",
    r"^(?:[a-z\s,]+(?:/\s*(?:pr\s*newswire|business\s*wire|globenewswire|prnewswire)\s*/)?\s*[-—–]+\s*)+",
    r"^(?:(?:pr\s*newswire|business\s*wire|globenewswire|marketwired|accesswire|prnewswire)\s*[-—–:/|]\s*)+",
    r"^(?:[a-z\s,]+(?:[-—–]\s*))+",
]

_BOILERPLATE_SECTION_PATTERNS = [
    r"(?i)\babout\s+[a-z0-9\s.,&-]+:\s*.*$",
    r"(?i)\bforward-looking\s+statements\b.*$",
    r"(?i)\bsafe\s+harbor\s+statement\b.*$",
    r"(?i)\bmedia\s+contacts?:\s*.*$",
    r"(?i)\binvestor\s+relations?:\s*.*$",
    r"(?i)\ball\s+rights\s+reserved\.?.*$",
    r"(?i)\bcopyright\s+(?:©|\(c\)?|\d{4}).*$",
]


def strip_html_tags(text: str) -> str:
    """Remove HTML tags while preserving their enclosed text content."""
    # Remove script and style blocks entirely
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities (e.g. &amp;, &quot;, &#39;)
    return html.unescape(text)


def normalize_source_text(text: Optional[str]) -> str:
    """Normalize textual content by stripping formatting noise while strictly preserving factual tokens.
    
    Invariants:
      - Numbers, currency values, dates, entity names, and negations are strictly preserved.
      - Irrelevant whitespace, HTML artifacts, smart quotes, and standard wire headers are removed.
    """
    if not text:
        return ""

    # 1. Strip HTML tags and unescape entities
    cleaned = strip_html_tags(text)

    # 2. Normalize smart quotes, dashes, and Unicode typography to standard ASCII
    replacements = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "—": "-", "–": "-", "―": "-", "−": "-",
        "…": "...", "\u00a0": " ", "\u200b": "", "\ufeff": "",
    }
    for orig, repl in replacements.items():
        cleaned = cleaned.replace(orig, repl)

    # 3. Strip wire footer sections (Forward-Looking Statements, Media Contacts, etc.)
    for pattern in _BOILERPLATE_SECTION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    # 4. Lowercase for case-insensitive content comparison
    lowered = cleaned.lower()

    # 5. Strip wire datelines / service prefixes (e.g. "pr newswire --")
    for pattern in _WIRE_PREFIX_PATTERNS:
        lowered = re.sub(pattern, "", lowered.strip())

    # 6. Normalize punctuation: remove decorative symbols but keep factual punctuation in numbers
    # (e.g., keep periods/commas within digits like $50.5m or 1,000, keep dollar signs and percentages)
    # Replace non-word/non-number characters with space unless they are part of numbers/words
    tokens = re.findall(r"\$?\b\w+(?:[.,]\w+)*%?|\S", lowered)
    
    # Filter out standalone punctuation noise (brackets, bullets, pipe symbols, quotes)
    cleaned_tokens = [
        t for t in tokens
        if not re.fullmatch(r"[-—–_.,;:!?'\"/\\|*&^%$#@~`+=()[\]{}<>]+", t)
    ]

    # 7. Collapse all whitespace
    normalized = " ".join(cleaned_tokens)
    return normalized.strip()


def compute_raw_content_hash(text: Optional[str]) -> str:
    """Compute deterministic SHA-256 hex digest of raw source content."""
    if not text:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def compute_normalized_content_hash(text: Optional[str]) -> str:
    """Compute deterministic SHA-256 hex digest of normalized textual content."""
    normalized = normalize_source_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
