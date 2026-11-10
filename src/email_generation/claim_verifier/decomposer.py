from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    ModalityLevel,
    VerificationFinding,
    VerifierFindingSeverity,
)


class ClaimDecomposer:
    """Gate 1: Decomposes email subject and body into atomic propositions with exact spans.

    Preserves exact character offsets such that:
        source_text[claim.span_start:claim.span_end] == claim.text
    Distinguishes greetings/fillers, prospect observations, product assertions,
    numerical claims, modality hedges, and diagnostic questions/CTAs.
    """

    GREETING_PATTERN = re.compile(
        r"^(?:(?:hi|hello|hey|dear)\s+[a-z0-9_\-\.\']+(?:\s+[a-z0-9_\-\.\']+)?)\s*[,!:]\s*",
        re.IGNORECASE,
    )

    SIGNOFF_PATTERN = re.compile(
        r"^(?:best|thanks|thank you|regards|sincerely|cheers)[,!]?\s*(?:[a-z0-9_\-\.\'\s]+)?$",
        re.IGNORECASE,
    )
    CTA_QUESTION_PATTERN = re.compile(
        r"(?:\?|are you open|would you be open|how are you|are you seeing|interested in|worth a quick|worth exploring|open to|let's connect|schedule|calendar)",
        re.IGNORECASE,
    )
    NUMERICAL_PATTERN = re.compile(r"\b(?:\d+[\d,\.]*|\$\d+[\d,\.]*|\d+%\s*|\d+x)\b")
    CONDITIONAL_PATTERN = re.compile(r"\b(?:if|assuming|when configured|depending on|in the event that)\b", re.IGNORECASE)
    HYPOTHETICAL_PATTERN = re.compile(r"\b(?:might|could|may|perhaps|potentially|possibly)\b", re.IGNORECASE)
    DEFINITIVE_PATTERN = re.compile(r"\b(?:will|guarantees|ensures|eliminates|solves|is|are|has|added|cut|reduced)\b", re.IGNORECASE)

    def decompose(
        self,
        subject: str,
        body: str,
        prospect_name: Optional[str] = None,
        company_name: Optional[str] = None,
        product_name: Optional[str] = None,
    ) -> Tuple[List[DecomposedClaim], List[VerificationFinding]]:
        """Decompose draft subject and body into discrete atomic claims with exact spans."""
        claims: List[DecomposedClaim] = []
        findings: List[VerificationFinding] = []

        # 1. Decompose subject
        subj_claims, subj_findings = self._decompose_section(
            text=subject,
            section=ClaimSection.SUBJECT,
            prospect_name=prospect_name,
            company_name=company_name,
            product_name=product_name,
        )
        claims.extend(subj_claims)
        findings.extend(subj_findings)

        # 2. Decompose body
        body_claims, body_findings = self._decompose_section(
            text=body,
            section=ClaimSection.BODY,
            prospect_name=prospect_name,
            company_name=company_name,
            product_name=product_name,
        )
        claims.extend(body_claims)
        findings.extend(body_findings)

        # Invariant check on all claims
        for claim in claims:
            src = subject if claim.section == ClaimSection.SUBJECT else body
            extracted = src[claim.span_start : claim.span_end]
            if extracted != claim.text:
                findings.append(
                    VerificationFinding(
                        gate="ClaimDecomposition",
                        severity=VerifierFindingSeverity.ERROR,
                        code="SPAN_MISMATCH",
                        message=f"Character span [{claim.span_start}:{claim.span_end}] does not match text '{claim.text}' (got '{extracted}')",
                        claim_id=claim.claim_id,
                        section=claim.section,
                        text_span=(claim.span_start, claim.span_end),
                    )
                )

        return claims, findings

    def _decompose_section(
        self,
        text: str,
        section: ClaimSection,
        prospect_name: Optional[str],
        company_name: Optional[str],
        product_name: Optional[str],
    ) -> Tuple[List[DecomposedClaim], List[VerificationFinding]]:
        claims: List[DecomposedClaim] = []
        findings: List[VerificationFinding] = []

        if not text or not text.strip():
            return claims, findings

        # Split text into sentences/lines while tracking exact character offsets
        sentences = self._split_into_sentences_with_spans(text)

        for sent_idx, (sent_text, s_start, s_end) in enumerate(sentences):
            trimmed_text = sent_text.strip()
            if not trimmed_text:
                continue

            # Check if signoff (e.g. "Best,", "Thanks,", "Cheers, Alex")
            if self.SIGNOFF_PATTERN.match(trimmed_text):
                continue

            # Skip single-word or short signoff name fragments (e.g. "Alex", "Sattar")
            words = trimmed_text.split()
            if len(words) <= 2 and not any(w.endswith("?") for w in words):
                has_verb_or_metric = bool(re.search(r"\b(?:\d+|is|are|was|were|have|has|saw|added|built|cut|help|need|check|open)\b", trimmed_text, re.IGNORECASE))
                if not has_verb_or_metric:
                    continue

            # Check for leading greeting in sentence (e.g. "Hi Elena," or "Elena, saw your team...")
            remaining_text = sent_text
            offset = s_start

            greeting_match = self.GREETING_PATTERN.match(remaining_text)
            if not greeting_match and prospect_name and sent_idx == 0 and section == ClaimSection.BODY:
                first_name = prospect_name.split()[0]
                name_pattern = re.compile(rf"^(?:{re.escape(prospect_name)}|{re.escape(first_name)})\s*[,!:]\s*", re.IGNORECASE)
                greeting_match = name_pattern.match(remaining_text)

            if greeting_match:
                greeting_end = greeting_match.end()
                after_greeting = remaining_text[greeting_end:]
                if not after_greeting.strip():
                    # Just a greeting line (e.g. "Hi Elena,") -> skip
                    continue
                # Advance past greeting to the proposition
                leading_spaces = len(after_greeting) - len(after_greeting.lstrip())
                actual_start = offset + greeting_end + leading_spaces
                remaining_text = after_greeting.lstrip()
                offset = actual_start

            # Split compound sentence by coordinate conjunctions if present (e.g. ", and ", "; ")
            compound_parts = self._split_compound_sentence(remaining_text, offset)

            for prop_text, p_start, p_end in compound_parts:
                clean_prop = prop_text.strip()
                if not clean_prop:
                    continue

                # Adjust start/end to strip boundaries cleanly
                l_strip = len(prop_text) - len(prop_text.lstrip())
                r_strip = len(prop_text) - len(prop_text.rstrip())
                exact_start = p_start + l_strip
                exact_end = p_end - r_strip
                exact_text = text[exact_start:exact_end]

                category, modality, is_material, is_product_claim = self._classify_proposition(
                    text=exact_text,
                    section=section,
                    prospect_name=prospect_name,
                    company_name=company_name,
                    product_name=product_name,
                )

                claim = DecomposedClaim(
                    claim_id=str(uuid.uuid4()),
                    text=exact_text,
                    section=section,
                    span_start=exact_start,
                    span_end=exact_end,
                    category=category,
                    subject=company_name or prospect_name or "",
                    predicate=exact_text,
                    modality=modality,
                    is_material=is_material,
                    is_product_claim=is_product_claim,
                )
                claims.append(claim)

        return claims, findings

    def _split_into_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences/paragraphs while tracking exact character offsets."""
        results: List[Tuple[str, int, int]] = []
        # Pattern matching sentence terminators (. ! ?) followed by whitespace, or double newlines
        pattern = re.compile(r"([^.!?\n]+(?:[.!?]+|\n+|$))")
        for match in pattern.finditer(text):
            sent = match.group(1)
            start, end = match.start(1), match.end(1)
            if sent.strip():
                results.append((sent, start, end))
        return results

    def _split_compound_sentence(self, sentence_text: str, base_offset: int) -> List[Tuple[str, int, int]]:
        """Split a compound sentence on strong coordinate conjunctions while tracking offsets."""
        # Look for ", and ", ", but ", "; "
        conjunction_pattern = re.compile(r"(?:,\s+(?:and|but|while)\s+|;\s+)", re.IGNORECASE)
        splits: List[Tuple[str, int, int]] = []
        last_idx = 0

        for match in conjunction_pattern.finditer(sentence_text):
            m_start, m_end = match.start(), match.end()
            part = sentence_text[last_idx:m_start]
            # Only split if part has substantial words (>3 words)
            if len(part.strip().split()) >= 3:
                splits.append((part, base_offset + last_idx, base_offset + m_start))
                last_idx = m_end

        tail = sentence_text[last_idx:]
        if tail.strip():
            splits.append((tail, base_offset + last_idx, base_offset + len(sentence_text)))

        return splits if splits else [(sentence_text, base_offset, base_offset + len(sentence_text))]

    PROD_INDICATOR_PATTERN = re.compile(
        r"\b(?:we built|we developed|our platform|our tool|our solution|we provide|we help|we can|our engine|our system|our product|we enable|we allow|our team|we designed)\b",
        re.IGNORECASE,
    )

    def _classify_proposition(
        self,
        text: str,
        section: ClaimSection,
        prospect_name: Optional[str],
        company_name: Optional[str],
        product_name: Optional[str],
    ) -> Tuple[ClaimCategory, ModalityLevel, bool, bool]:
        """Classify category, modality, materiality, and product-claim status of a proposition."""
        text_lower = text.lower()

        # 1. Determine whether this proposition is about the product vs prospect
        is_product = False
        if product_name and re.search(rf"\b{re.escape(product_name.lower())}\b", text_lower):
            is_product = True
        elif self.PROD_INDICATOR_PATTERN.search(text):
            is_product = True

        # 2. Check if diagnostic CTA / closing question
        if text.strip().endswith("?") or self.CTA_QUESTION_PATTERN.search(text):
            return ClaimCategory.RELATIONAL_CTA, ModalityLevel.DIAGNOSTIC, True, is_product

        # 3. Check modality level
        if self.CONDITIONAL_PATTERN.search(text):
            modality = ModalityLevel.CONDITIONAL
        elif self.HYPOTHETICAL_PATTERN.search(text):
            modality = ModalityLevel.HYPOTHETICAL
        else:
            modality = ModalityLevel.DEFINITIVE

        # 4. Check for explicit conditional hedge clause
        if text_lower.startswith(("if ", "assuming ", "depending on ", "when ", "as ")):
            return ClaimCategory.MODALITY_HEDGE, ModalityLevel.CONDITIONAL, True, is_product

        # 5. Product capability vs metric
        if is_product:
            if self.NUMERICAL_PATTERN.search(text):
                return ClaimCategory.METRIC_IMPACT, modality, True, True
            return ClaimCategory.PRODUCT_CAPABILITY, modality, True, True

        # 6. Prospect observation vs metric
        if self.NUMERICAL_PATTERN.search(text):
            return ClaimCategory.METRIC_IMPACT, modality, True, False
        return ClaimCategory.PROSPECT_OBSERVATION, modality, True, False
