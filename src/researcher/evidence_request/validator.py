from __future__ import annotations

import re
from typing import List, Optional, Tuple


class BiasedEvidenceRequestError(ValueError):
    """Raised when an EvidenceRequest violates the hypothesis-blind contract by including commercial or persuasive language."""
    pass


# Commercial persuasion vocabulary that steers research toward sales/outreach outcomes
PROHIBITED_COMMERCIAL_TERMS = {
    # 1. Product/Vendor ownership
    "our product",
    "our solution",
    "our platform",
    "our tool",
    "our software",
    "our company",
    "our service",
    "our services",
    "we sell",
    "we offer",
    "we provide",
    # 2. Sales & Pitching terminology
    "pitch",
    "pitching",
    "cold email",
    "outreach message",
    "outreach email",
    "call to action",
    "closing line",
    "sales opportunity",
    "commercial hook",
    "sales hook",
    "buying signal",
    # 3. Commercial persuasion & buying motivation
    "needs our",
    "need our",
    "buy our",
    "purchase our",
    "reasons to buy",
    "reason to buy",
    "reasons to purchase",
    "reason to purchase",
    "why they should buy",
    "why they should purchase",
    "why they should switch to us",
    "proof that they need",
    "prove that they need",
    "prove they need",
    "convince them",
    # 4. Fabricated pain / sentiment steering
    "dissatisfied with",
    "dissatisfaction with",
    "unhappy with",
    "frustrated with",
    "pain point",
    "pain points",
    "struggling with current vendor",
    "complaining about vendor",
}

# Regex patterns catching persuasive or commercial steering sentence structures
PROHIBITED_PATTERNS = [
    re.compile(r"\b(?:prove|confirm|show)\s+(?:that\s+)?.*(?:\bneeds?\b|\bwants?\b|\bshould\s+buy\b)", re.IGNORECASE),
    re.compile(r"\b(?:find|search\s+for)\s+(?:evidence|reasons)\s+(?:proving|supporting|that\s+supports?)\s+.*(?:outreach|pitch|message|proposal)", re.IGNORECASE),
    re.compile(r"\b(?:why|how)\s+.*should\s+(?:buy|purchase|switch\s+to|choose)\s+.*", re.IGNORECASE),
    re.compile(r"\b(?:confirm|prove)\s+that\s+.*(?:dissatisfied|unhappy|frustrated)", re.IGNORECASE),
]


SORTED_PROHIBITED_COMMERCIAL_TERMS = sorted(PROHIBITED_COMMERCIAL_TERMS, key=len, reverse=True)


def check_commercial_bias(text: str) -> Tuple[bool, Optional[str]]:
    """Deterministically inspect text for commercial persuasion or sales-biased framing.
    
    Returns:
        (is_biased, matched_reason)
    """
    if not text:
        return False, None

    normalized = text.strip().lower()

    # 1. Check prohibited exact phrases (longer phrases prioritized)
    for term in SORTED_PROHIBITED_COMMERCIAL_TERMS:
        if term in normalized:
            return True, f"Prohibited commercial term detected: '{term}'"

    # 2. Check regex patterns
    for pattern in PROHIBITED_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return True, f"Prohibited commercial persuasion pattern detected: '{match.group(0)}'"

    return False, None


def validate_hypothesis_blindness(
    neutral_question: str,
    neutral_search_terms: List[str],
    disconfirming_search_terms: List[str],
) -> None:
    """Enforce hypothesis blindness across question and search terms.
    
    Raises:
        BiasedEvidenceRequestError: If any commercial persuasion or sales steering is detected.
    """
    # 1. Inspect neutral question
    is_biased, reason = check_commercial_bias(neutral_question)
    if is_biased:
        raise BiasedEvidenceRequestError(f"Neutral question violates hypothesis blindness: {reason}")

    # 2. Inspect neutral search terms
    for term in neutral_search_terms:
        is_biased, reason = check_commercial_bias(term)
        if is_biased:
            raise BiasedEvidenceRequestError(f"Neutral search term '{term}' violates hypothesis blindness: {reason}")

    # 3. Inspect disconfirming search terms
    for term in disconfirming_search_terms:
        is_biased, reason = check_commercial_bias(term)
        if is_biased:
            raise BiasedEvidenceRequestError(f"Disconfirming search term '{term}' violates hypothesis blindness: {reason}")
