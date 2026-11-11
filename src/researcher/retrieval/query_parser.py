from __future__ import annotations

import re
from typing import List, Tuple


class FTSQueryParser:
    """Sanitizes user and LLM queries into safe, parameterized SQLite FTS5 MATCH expressions."""

    OPERATORS = {"AND", "OR", "NOT"}

    @classmethod
    def sanitize(cls, query: str) -> str:
        """Sanitize a raw search string for SQLite FTS5.

        - Extracts and preserves balanced double-quoted phrases.
        - Splits text outside quotes on whitespace and special punctuation.
        - Strips control characters and special FTS5 operators (: * ^ ( ) { } [ ] ~ etc.).
        - Preserves binary operators (AND, OR, NOT) only when strictly situated between
          two valid search terms/phrases.
        - Encloses individual terms in double quotes to prevent syntax errors on reserved keywords.
        - Returns empty string for empty/whitespace-only input.
        """
        if not query or not query.strip():
            return ""

        # Remove control characters and non-printable characters
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", query)

        # Extract balanced double-quoted phrases and intervening text
        raw_tokens: List[Tuple[str, str]] = []
        pos = 0
        pattern = re.compile(r'"([^"]+)"')
        for m in pattern.finditer(cleaned):
            start, end = m.span()
            if start > pos:
                raw_tokens.append(("text", cleaned[pos:start]))
            raw_tokens.append(("phrase", m.group(1)))
            pos = end
        if pos < len(cleaned):
            raw_tokens.append(("text", cleaned[pos:]))

        elements: List[str] = []
        for kind, val in raw_tokens:
            if kind == "phrase":
                # Clean punctuation inside phrase
                phrase_clean = re.sub(r'["*^:(){}\[\]]', " ", val).strip()
                phrase_clean = " ".join(phrase_clean.split())
                if phrase_clean:
                    elements.append(f'"{phrase_clean}"')
            else:
                # Replace special FTS punctuation with space
                subbed = re.sub(r"[:\^\(\)\{\}\[\]\?\!\,\.\;\*\+\~\/\\\#\$\%\&\|\<\>\=\@\']", " ", val)
                for w in subbed.split():
                    w_clean = re.sub(r"[^\w\-]", "", w).strip()
                    if w_clean:
                        elements.append(w_clean)

        if not elements:
            return ""

        # Process binary operators vs terms
        final_terms: List[str] = []
        for i, elem in enumerate(elements):
            elem_upper = elem.upper()
            if elem_upper in cls.OPERATORS:
                has_prev = (i > 0 and bool(final_terms) and final_terms[-1] not in cls.OPERATORS)
                has_next = (i < len(elements) - 1 and elements[i + 1].upper() not in cls.OPERATORS)
                if has_prev and has_next:
                    final_terms.append(elem_upper)
                else:
                    # Treat solitary or misplaced operator as a quoted search term
                    final_terms.append(f'"{elem_upper}"')
            else:
                if elem.startswith('"') and elem.endswith('"'):
                    final_terms.append(elem)
                else:
                    final_terms.append(f'"{elem}"')

        # Safety cleanup: cannot start or end with a bare operator
        while final_terms and final_terms[0] in cls.OPERATORS:
            final_terms[0] = f'"{final_terms[0]}"'
        while final_terms and final_terms[-1] in cls.OPERATORS:
            final_terms[-1] = f'"{final_terms[-1]}"'

        return " ".join(final_terms)
