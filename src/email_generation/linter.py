from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.context.models import OutreachContext
from src.email_generation.models import (
    LinterFinding,
    LinterResult,
    LinterSeverity,
)
from src.signal_strategy.models import OutreachStrategy, ToulminArgument


# Common abbreviations that should not trigger sentence boundaries
_ABBREVIATIONS = (
    "mr.", "ms.", "mrs.", "dr.", "prof.", "inc.", "ltd.", "co.",
    "corp.", "e.g.", "i.e.", "vs.", "etc.", "u.s.", "ai.", "jan.",
    "feb.", "mar.", "apr.", "aug.", "sept.", "oct.", "nov.", "dec.",
)

# Generic opening clichés
_GENERIC_OPENINGS = (
    "i hope this email finds you well",
    "i hope you are doing well",
    "i hope you're doing well",
    "hope you are doing well",
    "hope you're doing well",
    "hope all is well",
    "hope you are well",
    "my name is",
    "i am reaching out",
    "i'm reaching out",
    "reaching out because",
    "allow me to introduce",
    "hope your week is going well",
    "i know you are busy",
    "hope this note finds you well",
)

# Corporate buzzwords and empty marketing jargon
_CORPORATE_JARGON = (
    "synergy", "synergies", "paradigm shift", "disruptive",
    "game changer", "game-changer", "revolutionary", "best-in-class",
    "best in class", "cutting-edge", "cutting edge", "next-gen",
    "next gen", "holistic", "streamline your workflow", "leverage",
    "wheelhouse", "low-hanging fruit", "boil the ocean", "move the needle",
    "hyper-growth", "hypergrowth",
)

# Compounded hedge words
_HEDGE_WORDS = (
    "perhaps", "maybe", "possibly", "potentially", "somewhat",
    "sort of", "kind of",
)

# Weak or banned CTA patterns
_WEAK_CTA_PATTERNS = (
    "thoughts?",
    "let me know your thoughts",
    "pick your brain",
    "steal 15 minutes",
    "grab a coffee",
    "jump on a call",
    "hop on a call",
    "jump on a quick call",
    "hop on a quick call",
)


def _count_syllables(word: str) -> int:
    """Approximate syllable count using heuristic vowel clustering."""
    clean = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not clean:
        return 1
    if len(clean) <= 3:
        return 1

    # Discard silent trailing e
    if clean.endswith("e") and not clean.endswith("le"):
        clean = clean[:-1]

    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in clean:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    return max(1, count)


def _split_into_sentences(text: str) -> List[str]:
    """Split body text into sentences while respecting common abbreviations and decimals."""
    if not text or not text.strip():
        return []

    # Replace newlines with spaces for sentence extraction
    normalized = re.sub(r"\s+", " ", text.strip())

    # Protect decimals like 3.5 or 0.84
    protected = re.sub(r"(\d)\.(\d)", r"\1__DECIMAL__\2", normalized)

    # Protect common abbreviations
    for abbr in _ABBREVIATIONS:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        protected = pattern.sub(abbr.replace(".", "__DOT__"), protected)

    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)
    results = []
    for s in raw_sentences:
        clean = s.replace("__DECIMAL__", ".").replace("__DOT__", ".").strip()
        if clean:
            results.append(clean)
    return results


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs."""
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [p.strip() for p in text.splitlines() if p.strip()]
    return paras


def _tokenize_words(text: str) -> List[str]:
    """Extract alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text)


class ReadabilityLinter:
    """Deterministic readability and email-quality linter.

    Evaluates cold outreach drafts against cold email benchmarks (conciseness, sentence length,
    opening directness, CTA focus, and explicit constraint violations) without calling external LLMs.
    """

    def __init__(
        self,
        max_words_warning: int = 100,
        max_words_error: int = 150,
        max_sentence_words: int = 25,
        max_paragraph_sentences: int = 3,
        max_paragraph_words: int = 50,
        max_reading_grade: float = 10.0,
    ) -> None:
        self.max_words_warning = max_words_warning
        self.max_words_error = max_words_error
        self.max_sentence_words = max_sentence_words
        self.max_paragraph_sentences = max_paragraph_sentences
        self.max_paragraph_words = max_paragraph_words
        self.max_reading_grade = max_reading_grade

    def lint(
        self,
        subject: Optional[str],
        body: Optional[str],
        strategy: Optional[OutreachStrategy] = None,
        outreach: Optional[OutreachContext] = None,
        toulmin_argument: Optional[ToulminArgument] = None,
    ) -> LinterResult:
        """Execute deterministic linting rules across subject and body."""
        findings: List[LinterFinding] = []

        sub_clean = (subject or "").strip()
        body_clean = (body or "").strip()

        # Rule 1: Missing subject
        if not sub_clean:
            findings.append(
                LinterFinding(
                    rule_id="missing_subject",
                    severity=LinterSeverity.ERROR,
                    explanation="Email subject line is missing or empty.",
                    location="subject",
                )
            )

        # Rule 2: Missing body
        if not body_clean:
            findings.append(
                LinterFinding(
                    rule_id="missing_body",
                    severity=LinterSeverity.ERROR,
                    explanation="Email body is missing or empty.",
                    location="body",
                )
            )
            return LinterResult(
                passed=False,
                findings=findings,
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                flesch_kincaid_grade=None,
                has_errors=True,
                has_warnings=False,
            )

        words = _tokenize_words(body_clean)
        word_count = len(words)
        sentences = _split_into_sentences(body_clean)
        sentence_count = len(sentences)
        paragraphs = _split_into_paragraphs(body_clean)
        paragraph_count = len(paragraphs)

        # Rule 3: Word count checks
        if word_count > self.max_words_error:
            findings.append(
                LinterFinding(
                    rule_id="word_count_error",
                    severity=LinterSeverity.ERROR,
                    explanation=f"Email body ({word_count} words) exceeds maximum allowable limit of {self.max_words_error} words.",
                    location=f"{word_count} words",
                )
            )
        elif word_count > self.max_words_warning:
            findings.append(
                LinterFinding(
                    rule_id="word_count_warning",
                    severity=LinterSeverity.WARNING,
                    explanation=f"Email body ({word_count} words) exceeds recommended cold outreach length of {self.max_words_warning} words.",
                    location=f"{word_count} words",
                )
            )

        # Rule 4: Sentence length checks
        for idx, sentence in enumerate(sentences, 1):
            s_words = _tokenize_words(sentence)
            if len(s_words) > self.max_sentence_words:
                findings.append(
                    LinterFinding(
                        rule_id="sentence_length",
                        severity=LinterSeverity.WARNING,
                        explanation=f"Sentence #{idx} exceeds {self.max_sentence_words} words ({len(s_words)} words). Shorten for readability.",
                        location=sentence[:60] + ("..." if len(sentence) > 60 else ""),
                    )
                )

        # Rule 5: Paragraph length checks
        for p_idx, para in enumerate(paragraphs, 1):
            p_words = _tokenize_words(para)
            p_sentences = _split_into_sentences(para)
            if len(p_words) > self.max_paragraph_words or len(p_sentences) > self.max_paragraph_sentences:
                findings.append(
                    LinterFinding(
                        rule_id="paragraph_length",
                        severity=LinterSeverity.WARNING,
                        explanation=(
                            f"Paragraph #{p_idx} exceeds recommended limits ({len(p_words)} words, {len(p_sentences)} sentences). "
                            "Break into shorter visual blocks for mobile reading."
                        ),
                        location=para[:60] + ("..." if len(para) > 60 else ""),
                    )
                )

        # Rule 6: CTA evaluation (missing, multiple, weak/banned)
        cta_candidates = [s for s in sentences if "?" in s]
        if not cta_candidates:
            cta_keywords = ("open to", "let me know", "worth a", "worth exploring", "briefing", "connect this week")
            for s in sentences:
                if any(kw in s.lower() for kw in cta_keywords):
                    cta_candidates.append(s)

        if len(cta_candidates) == 0:
            findings.append(
                LinterFinding(
                    rule_id="missing_cta",
                    severity=LinterSeverity.ERROR,
                    explanation="Email body contains no clear closing call to action or question.",
                    location="body closing",
                )
            )
        elif len(cta_candidates) > 1:
            findings.append(
                LinterFinding(
                    rule_id="multiple_ctas",
                    severity=LinterSeverity.WARNING,
                    explanation=f"Multiple calls to action detected ({len(cta_candidates)} found). Cold outreach should have a single clear CTA.",
                    location="; ".join(s[:40] for s in cta_candidates[:2]),
                )
            )

        # Rule 7: Weak or banned CTA patterns
        approved_cta_text = ""
        if strategy:
            cta_candidate = getattr(strategy, "cta", None) or getattr(strategy, "call_to_action", "") or ""
            approved_cta_text = cta_candidate.lower()

        lower_body = body_clean.lower()
        for banned_pattern in _WEAK_CTA_PATTERNS:
            if banned_pattern in lower_body and (not approved_cta_text or banned_pattern not in approved_cta_text):
                findings.append(
                    LinterFinding(
                        rule_id="weak_banned_cta",
                        severity=LinterSeverity.WARNING,
                        explanation=f"Draft contains weak or presumptive CTA pattern: '{banned_pattern}'. Use approved value-exchange or diagnostic closing.",
                        location=banned_pattern,
                    )
                )

        # Rule 8: Generic opening
        if sentences:
            first_sentence_lower = sentences[0].lower().strip()
            for generic in _GENERIC_OPENINGS:
                if generic in first_sentence_lower:
                    findings.append(
                        LinterFinding(
                            rule_id="generic_opening",
                            severity=LinterSeverity.WARNING,
                            explanation=f"Draft opens with generic cliché: '{generic}'. Lead with specific prospect context instead.",
                            location=sentences[0][:60],
                        )
                    )
                    break

        # Rule 9: Excessive first-person usage
        first_person_tokens = re.findall(r"\b(i|me|my|mine|we|us|our|ours)\b", lower_body)
        fp_count = len(first_person_tokens)
        if fp_count > 4 or (word_count > 0 and (fp_count / word_count) > 0.08):
            findings.append(
                LinterFinding(
                    rule_id="excessive_first_person",
                    severity=LinterSeverity.WARNING,
                    explanation=f"Excessive first-person pronouns ({fp_count} found). Keep the narrative centered on the prospect.",
                    location=f"{fp_count} occurrences",
                )
            )

        # Rule 10: Corporate jargon / buzzwords
        for jargon in _CORPORATE_JARGON:
            pattern = re.compile(rf"\b{re.escape(jargon)}\b", re.IGNORECASE)
            match = pattern.search(body_clean)
            if match:
                findings.append(
                    LinterFinding(
                        rule_id="corporate_jargon",
                        severity=LinterSeverity.WARNING,
                        explanation=f"Recognized corporate jargon detected: '{jargon}'. Use plain, specific language instead.",
                        location=match.group(0),
                    )
                )

        # Rule 11: Excessive hedging
        hedge_matches = []
        for hedge in _HEDGE_WORDS:
            matches = re.findall(rf"\b{re.escape(hedge)}\b", lower_body)
            hedge_matches.extend(matches)
        if len(hedge_matches) >= 3:
            findings.append(
                LinterFinding(
                    rule_id="excessive_hedging",
                    severity=LinterSeverity.WARNING,
                    explanation=f"Excessive hedging detected ({len(hedge_matches)} hedge terms found). Express uncertainty cleanly without compounding hedges.",
                    location=f"{', '.join(hedge_matches[:4])}",
                )
            )

        # Rule 12: Excessive exclamation marks
        excl_count = body_clean.count("!")
        has_double_excl = "!!" in body_clean
        if excl_count > 1 or has_double_excl:
            findings.append(
                LinterFinding(
                    rule_id="excessive_exclamations",
                    severity=LinterSeverity.WARNING,
                    explanation=f"Exclamation marks used excessively ({excl_count} found). Maintain a calm, professional tone.",
                    location=f"{excl_count} exclamation marks",
                )
            )

        # Rule 13: Unsupported explicit numerical/metric tokens
        self._check_numerical_claims(
            body_clean=body_clean,
            strategy=strategy,
            toulmin_argument=toulmin_argument,
            findings=findings,
        )

        # Rule 14: Explicit outreach constraints
        if outreach:
            self._check_explicit_constraints(
                subject_clean=sub_clean,
                body_clean=body_clean,
                word_count=word_count,
                outreach=outreach,
                findings=findings,
            )

        # Rule 15: Flesch-Kincaid Grade level
        fk_grade = None
        if word_count > 0 and sentence_count > 0:
            total_syllables = sum(_count_syllables(w) for w in words)
            fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59
            fk_grade = round(max(0.0, fk_grade), 2)
            if fk_grade > self.max_reading_grade:
                findings.append(
                    LinterFinding(
                        rule_id="reading_grade_above_10",
                        severity=LinterSeverity.WARNING,
                        explanation=f"Flesch-Kincaid reading grade level ({fk_grade}) exceeds target of {self.max_reading_grade}. Simplify syntax and terms.",
                        location=f"Grade {fk_grade}",
                    )
                )

        has_errors = any(f.severity == LinterSeverity.ERROR for f in findings)
        has_warnings = any(f.severity == LinterSeverity.WARNING for f in findings)
        passed = not has_errors

        return LinterResult(
            passed=passed,
            findings=findings,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            flesch_kincaid_grade=fk_grade,
            has_errors=has_errors,
            has_warnings=has_warnings,
        )

    def _check_numerical_claims(
        self,
        body_clean: str,
        strategy: Optional[OutreachStrategy],
        toulmin_argument: Optional[ToulminArgument],
        findings: List[LinterFinding],
    ) -> None:
        """Check for explicit percentages, multipliers, currency values, and large numbers."""
        metric_patterns = [
            r"\b\d+(?:\.\d+)?%\b",
            r"\b\d+(?:\.\d+)?x\b",
            r"\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:k|m|b|million|billion))?\b",
        ]

        found_metrics: Set[str] = set()
        for pat in metric_patterns:
            matches = re.findall(pat, body_clean, re.IGNORECASE)
            for m in matches:
                found_metrics.add(m.strip().lower())

        if not found_metrics:
            return

        approved_corpus = ""
        toulmin = toulmin_argument
        if not toulmin and strategy and hasattr(strategy, "toulmin_argument") and strategy.toulmin_argument:
            toulmin = strategy.toulmin_argument

        if toulmin:
            if toulmin.backing:
                proofs = getattr(toulmin.backing, "whitelisted_proof_points", None) or getattr(toulmin.backing, "proof_points", [])
                approved_corpus += " " + " ".join(proofs)
            if toulmin.claim:
                claim_str = getattr(toulmin.claim, "text", "") or getattr(toulmin.claim, "claim_text", "")
                approved_corpus += " " + claim_str
            if toulmin.warrant:
                approved_corpus += " " + getattr(toulmin.warrant, "reasoning", "")
            if toulmin.grounds:
                obs = getattr(toulmin.grounds, "observed_event", "")
                fact_ids = getattr(toulmin.grounds, "fact_ids", [])
                approved_corpus += f" {' '.join(fact_ids)} {obs}"
        if strategy:
            approved_corpus += f" {getattr(strategy, 'central_message', '')} {getattr(strategy, 'messaging_approach', '')}"

        approved_corpus_lower = approved_corpus.lower()

        for metric in sorted(found_metrics):
            if metric not in approved_corpus_lower:
                findings.append(
                    LinterFinding(
                        rule_id="unsupported_numerical_claims",
                        severity=LinterSeverity.WARNING,
                        explanation=f"Explicit numerical/metric token '{metric}' not found in approved Toulmin grounds or whitelisted backing.",
                        location=metric,
                    )
                )

    def _check_explicit_constraints(
        self,
        subject_clean: str,
        body_clean: str,
        word_count: int,
        outreach: OutreachContext,
        findings: List[LinterFinding],
    ) -> None:
        """Verify deterministically detectable outreach constraints."""
        combined_text = f"{subject_clean} {body_clean}".lower()

        constraints = getattr(outreach, "constraints", None) or []
        forbidden_topics = getattr(outreach, "forbidden_topics", None) or []

        all_constraints = list(constraints) + [f"do not mention {t}" for t in forbidden_topics]

        for constraint in all_constraints:
            c_lower = constraint.lower()
            negative_match = re.search(r"(?:do not mention|no mention of|avoid mentioning|avoid)\s+([a-zA-Z0-9_-]+)", c_lower)
            if negative_match:
                term = negative_match.group(1).strip()
                if term and re.search(rf"\b{re.escape(term)}\b", combined_text):
                    findings.append(
                        LinterFinding(
                            rule_id="violated_outreach_constraints",
                            severity=LinterSeverity.ERROR,
                            explanation=f"Draft violates explicit outreach constraint: '{constraint}'.",
                            location=term,
                        )
                    )

            max_words_match = re.search(r"(?:max|limit to|under)\s+(\d+)\s*words?", c_lower)
            if max_words_match:
                limit = int(max_words_match.group(1))
                if word_count > limit:
                    findings.append(
                        LinterFinding(
                            rule_id="violated_outreach_constraints",
                            severity=LinterSeverity.ERROR,
                            explanation=f"Draft exceeds explicit word constraint of {limit} words (actual: {word_count} words).",
                            location=f"{word_count} words",
                        )
                    )
