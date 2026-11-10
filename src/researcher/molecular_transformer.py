from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.researcher.models import CanonicalFact, ConfidenceLevel, VerificationStatus
from src.researcher.molecular_models import (
    Modality,
    MolecularFact,
    MolecularValue,
    NormalizedPredicate,
    Polarity,
    ValueType,
)
from src.researcher.predicate_normalizer import PredicateNormalizer


def parse_molecular_value(text: str) -> MolecularValue:
    """Parse a textual object/complement into a structured MolecularValue with unit and normalized value."""
    cleaned = text.strip()
    if not cleaned:
        return MolecularValue(value_type=ValueType.UNKNOWN, raw_value="")

    # 1. Currency: e.g. $50 million, $50M, $30.5M, 50 million USD, €10M, £5M
    curr_match = re.search(
        r"([\$€£])?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(billion|million|thousand|b|m|k)?\s*(usd|eur|gbp)?\b",
        cleaned,
        re.IGNORECASE,
    )
    if curr_match and (curr_match.group(1) or curr_match.group(4)):
        symbol = curr_match.group(1) or ""
        num_str = curr_match.group(2).replace(",", "")
        scale = (curr_match.group(3) or "").lower()
        curr_code = (curr_match.group(4) or "").upper()

        unit = curr_code
        if not unit:
            if symbol == "$":
                unit = "USD"
            elif symbol == "€":
                unit = "EUR"
            elif symbol == "£":
                unit = "GBP"
            else:
                unit = "CURRENCY"

        val = float(num_str)
        if scale in ("billion", "b"):
            val *= 1_000_000_000
        elif scale in ("million", "m"):
            val *= 1_000_000
        elif scale in ("thousand", "k"):
            val *= 1_000

        return MolecularValue(
            value_type=ValueType.CURRENCY,
            raw_value=cleaned,
            normalized_value=val,
            unit=unit,
        )

    # 2. Number with unit: e.g. 500 engineers, 1,200 employees, 25%
    num_match = re.search(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(%|percent|[a-zA-Z_\-]+)?\b", cleaned, re.IGNORECASE)
    if num_match:
        num_str = num_match.group(1).replace(",", "")
        raw_unit = (num_match.group(2) or "").strip()
        val = float(num_str)
        if raw_unit in ("%", "percent"):
            return MolecularValue(
                value_type=ValueType.NUMBER,
                raw_value=cleaned,
                normalized_value=val,
                unit="%",
            )
        elif raw_unit:
            return MolecularValue(
                value_type=ValueType.NUMBER,
                raw_value=cleaned,
                normalized_value=val,
                unit=raw_unit.lower(),
            )

    # 3. Location: if looks like "in <City>, <Country/State>"
    loc_match = re.search(r"\b(?:in|at)\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)?)\b", cleaned)
    if loc_match and len(cleaned.split()) <= 5:
        return MolecularValue(
            value_type=ValueType.LOCATION,
            raw_value=cleaned,
            normalized_value=loc_match.group(1).strip(),
            unit=None,
        )

    # 4. Standard string
    return MolecularValue(
        value_type=ValueType.STRING,
        raw_value=cleaned,
        normalized_value=cleaned,
        unit=None,
    )


class MolecularTransformer:
    """Transforms and decomposes CanonicalFacts into atomic MolecularFacts."""

    def __init__(self, normalizer: Optional[PredicateNormalizer] = None) -> None:
        self.normalizer = normalizer or PredicateNormalizer()

    # Bounded coordinating conjunction regex splitting two independent verb-object clauses
    # Pattern: Subject + Clause 1 + " and " / " as well as " + Clause 2
    _COORDINATED_CLAUSE_REGEX = re.compile(
        r"^(?P<subject>[A-Z][A-Za-z0-9_\s\.\-]{1,50})\s+"
        r"(?P<clause1>[a-z][a-z0-9\s,\$€£\.\-]+?)\s+"
        r"(?:\band\b|\bas well as\b)\s+"
        r"(?P<clause2>(?:plans to|planning to|targets|aims to|expects to|hired|opened|closed|acquired|raised|appointed|named|serves as|launched)\b[a-z0-9\s,\$€£\.\-]+?)\.?$",
        re.IGNORECASE,
    )

    def transform(
        self,
        canonical_fact: CanonicalFact,
        subject_name_override: Optional[str] = None,
        subject_entity_id: Optional[str] = None,
    ) -> Tuple[MolecularFact, ...]:
        """Decompose a CanonicalFact into one or more atomic MolecularFacts."""
        claim_text = canonical_fact.claim_text.strip()
        sub_id = subject_entity_id or (canonical_fact.entity_ids[0] if canonical_fact.entity_ids else None)
        sub_name = subject_name_override or self._extract_subject_name(claim_text)

        # 1. Attempt bounded decomposition on coordinated clauses
        decomposed = self._try_decompose_coordinated_clauses(
            claim_text=claim_text,
            parent_id=canonical_fact.fact_id,
            subject_name=sub_name,
            subject_entity_id=sub_id,
            event_date=canonical_fact.primary_event_date,
            confidence=canonical_fact.composite_confidence,
        )
        if decomposed:
            return decomposed

        # 2. Fallback: treat as a single atomic proposition
        atom = self._create_single_atom(
            claim_text=claim_text,
            parent_id=canonical_fact.fact_id,
            subject_name=sub_name,
            subject_entity_id=sub_id,
            event_date=canonical_fact.primary_event_date,
            confidence=canonical_fact.composite_confidence,
            is_decomposed=False,
        )
        return (atom,)

    def _try_decompose_coordinated_clauses(
        self,
        claim_text: str,
        parent_id: str,
        subject_name: str,
        subject_entity_id: Optional[str],
        event_date: Any,
        confidence: ConfidenceLevel,
    ) -> Optional[Tuple[MolecularFact, ...]]:
        """Bounded decomposition splitting high-confidence coordinated clauses with shared subject."""
        # Conservative check: do not split if relative pronouns or subordinate conjunctions exist
        subordinating_words = (" who ", " which ", " that ", " because ", " although ", " while ", " after ", " before ")
        if any(w in claim_text.lower() for w in subordinating_words):
            return None

        # Determine subject and strip it from the start of claim_text
        subj = subject_name
        text_without_sub = claim_text
        if text_without_sub.lower().startswith(subj.lower()):
            text_without_sub = text_without_sub[len(subj):].strip()
        else:
            extracted_subj = self._extract_subject_name(claim_text)
            if extracted_subj and text_without_sub.lower().startswith(extracted_subj.lower()):
                subj = extracted_subj
                text_without_sub = text_without_sub[len(extracted_subj):].strip()

        # Split on coordinating conjunction: " and " or " as well as "
        parts = re.split(r"\s+(?:and|as well as)\s+", text_without_sub, maxsplit=1)
        if len(parts) != 2:
            return None

        clause1_text = parts[0].strip().rstrip(".")
        clause2_text = parts[1].strip().rstrip(".")
        if not clause1_text or not clause2_text:
            return None

        pred1, obj1 = self._split_predicate_and_object(clause1_text)
        pred2, obj2 = self._split_predicate_and_object(clause2_text)

        norm1 = self.normalizer.normalize(pred1)
        norm2 = self.normalizer.normalize(pred2)

        # Conservative check: BOTH clauses must have unambiguous recognized predicates to split!
        if not norm1.is_normalized or not norm2.is_normalized:
            return None

        atom1 = self._parse_clause(
            subject=subj,
            clause_text=clause1_text,
            parent_id=parent_id,
            subject_entity_id=subject_entity_id,
            event_date=event_date,
            confidence=confidence,
            is_decomposed=True,
        )

        atom2 = self._parse_clause(
            subject=subj,
            clause_text=clause2_text,
            parent_id=parent_id,
            subject_entity_id=subject_entity_id,
            event_date=event_date,
            confidence=confidence,
            is_decomposed=True,
        )

        if atom1 and atom2:
            return (atom1, atom2)

        return None

    def _parse_clause(
        self,
        subject: str,
        clause_text: str,
        parent_id: str,
        subject_entity_id: Optional[str],
        event_date: Any,
        confidence: ConfidenceLevel,
        is_decomposed: bool,
    ) -> MolecularFact:
        """Parse an individual verb-object clause into an atomic MolecularFact."""
        # Find leading verb phrase
        pred, obj = self._split_predicate_and_object(clause_text)
        norm_res = self.normalizer.normalize(pred)
        mol_value = parse_molecular_value(obj)
        qualifiers = self._extract_qualifiers(obj, norm_res.normalized_predicate)

        atom_text = f"{subject} {pred} {obj}".strip()
        return MolecularFact(
            molecular_id=str(uuid.uuid4()),
            parent_canonical_id=parent_id,
            atom_text=atom_text,
            subject_entity_id=subject_entity_id,
            subject_name=subject,
            raw_predicate=pred,
            normalized_predicate=norm_res.normalized_predicate,
            is_normalized=norm_res.is_normalized,
            object_value=mol_value,
            modality=norm_res.modality,
            polarity=norm_res.polarity,
            qualifiers=qualifiers,
            event_date=event_date,
            is_decomposed=is_decomposed,
            confidence=confidence,
            verification_status=VerificationStatus.SUPPORTED,
        )

    def _create_single_atom(
        self,
        claim_text: str,
        parent_id: str,
        subject_name: str,
        subject_entity_id: Optional[str],
        event_date: Any,
        confidence: ConfidenceLevel,
        is_decomposed: bool,
    ) -> MolecularFact:
        """Create a single MolecularFact when decomposition is not applicable."""
        pred, obj = self._split_predicate_and_object(claim_text, subject_name=subject_name)
        norm_res = self.normalizer.normalize(pred)
        mol_value = parse_molecular_value(obj)
        qualifiers = self._extract_qualifiers(obj, norm_res.normalized_predicate)

        return MolecularFact(
            molecular_id=str(uuid.uuid4()),
            parent_canonical_id=parent_id,
            atom_text=claim_text,
            subject_entity_id=subject_entity_id,
            subject_name=subject_name,
            raw_predicate=pred,
            normalized_predicate=norm_res.normalized_predicate,
            is_normalized=norm_res.is_normalized,
            object_value=mol_value,
            modality=norm_res.modality,
            polarity=norm_res.polarity,
            qualifiers=qualifiers,
            event_date=event_date,
            is_decomposed=is_decomposed,
            confidence=confidence,
            verification_status=VerificationStatus.SUPPORTED,
        )

    def _split_predicate_and_object(
        self, clause: str, subject_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """Split a clause into verb phrase (predicate) and complement (object)."""
        text = clause.strip()
        if subject_name and text.lower().startswith(subject_name.lower()):
            text = text[len(subject_name):].strip()

        # Check multi-word predicate prefixes first
        multi_word_verbs = [
            "plans to hire", "plan to hire", "planning to hire", "plans to add",
            "targeting hiring", "targets hiring", "aims to recruit", "aims to hire",
            "expects to add", "looking to hire",
            "plans to open", "planning to open", "plans to raise", "plans to acquire",
            "secured funding", "closed round", "bagged funding", "bagged investment",
            "raised funding", "secured investment", "announced funding", "closed series",
            "raised series", "secured series",
            "set up office", "established office", "opened office", "opened facility",
            "launched facility", "closed office", "closed facility", "shut down facility",
            "shuttered office", "headquartered in", "based in", "headquarters in",
            "operates in", "operating in", "present in",
            "completed acquisition of", "partnered with", "formed partnership with",
            "entered agreement with", "teamed up with", "collaborated with",
            "has headcount of", "headcount of", "workforce of", "team of", "staff of",
            "added employees", "brought on",
            "joined as", "promoted to", "serves as", "serving as", "elected as",
            "appointed as", "hired as", "stepped down", "stepped down as", "resigned as",
            "rolled out", "introduced product",
        ]
        for verb in multi_word_verbs:
            if text.lower().startswith(verb):
                remainder = text[len(verb):].strip()
                return verb, remainder

        # Single word verb split
        tokens = text.split(maxsplit=1)
        if len(tokens) == 2:
            return tokens[0], tokens[1]
        elif len(tokens) == 1:
            return tokens[0], ""
        return "", text

    def _extract_subject_name(self, text: str) -> str:
        """Heuristic to extract the leading subject noun phrase from claim text."""
        tokens = text.split()
        if not tokens:
            return "Unknown Entity"
        # Take leading capitalized words
        cap_tokens = []
        for t in tokens:
            cleaned = re.sub(r"[^\w]", "", t)
            if cleaned and cleaned[0].isupper():
                cap_tokens.append(cleaned)
            else:
                break
        return " ".join(cap_tokens) if cap_tokens else tokens[0]

    def _extract_qualifiers(self, obj_text: str, norm_pred: Optional[NormalizedPredicate]) -> Dict[str, Any]:
        """Extract structured qualifiers (e.g. series/round, location, role) from complement text."""
        qualifiers: Dict[str, Any] = {}
        lower = obj_text.lower()

        # Funding round / series
        series_match = re.search(r"\b(series\s+[a-z]|seed|pre-seed|ipo|growth|debt|bridge)\b", lower)
        if series_match:
            qualifiers["round"] = series_match.group(1).title()

        # Location qualifier: e.g. "in Munich", "in Hyderabad, India"
        loc_match = re.search(r"\b(?:in|at)\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z][a-zA-Z\s]+)?)\b", obj_text)
        if loc_match:
            qualifiers["location"] = loc_match.group(1).strip()

        # Role qualifier: e.g. "VP of Engineering", "Chief Technology Officer"
        role_match = re.search(r"\b(?:as|for)\s+([A-Z][a-zA-Z0-9\s/]+)\b", obj_text)
        if role_match and norm_pred in (NormalizedPredicate.LEADERSHIP_APPOINTED, NormalizedPredicate.LEADERSHIP_DEPARTED):
            qualifiers["role"] = role_match.group(1).strip()

        return qualifiers
