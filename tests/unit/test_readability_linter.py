from __future__ import annotations

import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import CompatibilityStatus, OutreachContext
from src.email_generation.linter import ReadabilityLinter
from src.email_generation.models import LinterSeverity
from src.researcher.models import ConfidenceLevel, VerificationStatus
from src.signal_strategy.models import (
    CTAType,
    EligibilityStatus,
    OutreachStrategy,
    ToulminArgument,
    ToulminBacking,
    ToulminClaim,
    ToulminGrounds,
    ToulminQualifier,
    ToulminRebuttal,
    ToulminWarrant,
)


def _make_dummy_toulmin(proof_points=None, fact_ids=None) -> ToulminArgument:
    return ToulminArgument(
        grounds=ToulminGrounds(
            fact_ids=fact_ids or ["fact_1"],
            observed_event="Vanguard AI expanded distributed training clusters",
        ),
        warrant=ToulminWarrant(
            capability_id="cap_1",
            capability_name="Real-time company intelligence",
            reasoning="Enables scalable pipeline automation",
        ),
        backing=ToulminBacking(
            capability_id="cap_1",
            whitelisted_proof_points=proof_points or ["50% reduction in pipeline latency"],
            has_whitelisted_proof=True,
            provenance=OutputProvenance(origin=ProvenanceOrigin.MANUAL_INPUT, component="test"),
        ),
        qualifier=ToulminQualifier(
            eligibility_status=EligibilityStatus.ELIGIBLE,
            compatibility_status=CompatibilityStatus.VERIFIED_COMPATIBLE,
            verification_status=VerificationStatus.SUPPORTED,
            composite_confidence=ConfidenceLevel.HIGH,
            qualification_language="Directly applicable based on verified expansion",
        ),
        claim=ToulminClaim(
            claim_type="value_assertion",
            text="Automate data extraction with verified provenance",
            target_capability="cap_1",
            is_qualified=True,
        ),
        rebuttal=ToulminRebuttal(),
        allowed_cta_friction=CTAType.VALUE_EXCHANGE,
    )


def test_missing_subject():
    linter = ReadabilityLinter()
    res = linter.lint(subject="", body="Hi Alex, are you open to exploring compute optimization?")
    assert res.passed is False
    assert res.has_errors is True
    errs = [f for f in res.findings if f.rule_id == "missing_subject"]
    assert len(errs) == 1
    assert errs[0].severity == LinterSeverity.ERROR


def test_missing_body():
    linter = ReadabilityLinter()
    res = linter.lint(subject="Quick question", body="   ")
    assert res.passed is False
    assert res.has_errors is True
    errs = [f for f in res.findings if f.rule_id == "missing_body"]
    assert len(errs) == 1
    assert errs[0].severity == LinterSeverity.ERROR


def test_word_count_thresholds():
    linter = ReadabilityLinter(max_words_warning=100, max_words_error=150)

    # 1. Compliant word count (< 100)
    res_ok = linter.lint(subject="Hello", body="This is a short, punchy email. " * 5 + " Are you free?")
    assert not any(f.rule_id.startswith("word_count") for f in res_ok.findings)

    # 2. Warning threshold (101 - 150)
    body_warn = "Word " * 110 + " Are you free?"
    res_warn = linter.lint(subject="Hello", body=body_warn)
    warns = [f for f in res_warn.findings if f.rule_id == "word_count_warning"]
    assert len(warns) == 1
    assert warns[0].severity == LinterSeverity.WARNING
    assert res_warn.passed is True  # Warnings do not fail 'passed'

    # 3. Error threshold (> 150)
    body_err = "Word " * 160 + " Are you free?"
    res_err = linter.lint(subject="Hello", body=body_err)
    errs = [f for f in res_err.findings if f.rule_id == "word_count_error"]
    assert len(errs) == 1
    assert errs[0].severity == LinterSeverity.ERROR
    assert res_err.passed is False


def test_excessive_sentence_length():
    linter = ReadabilityLinter(max_sentence_words=25)
    long_sentence = (
        "This sentence is intentionally constructed with far too many unnecessary descriptive words "
        "that drag on and on without making any concise point whatsoever for executive communication."
    )
    body = f"{long_sentence} Are you open to discussing this?"
    res = linter.lint(subject="Hello", body=body)
    findings = [f for f in res.findings if f.rule_id == "sentence_length"]
    assert len(findings) >= 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_long_paragraphs():
    linter = ReadabilityLinter(max_paragraph_words=50, max_paragraph_sentences=3)
    long_para = "Sentence one is brief. Sentence two adds context. Sentence three deepens the technical point. Sentence four is too much for one block."
    body = f"{long_para}\n\nAre you free to connect?"
    res = linter.lint(subject="Hello", body=body)
    findings = [f for f in res.findings if f.rule_id == "paragraph_length"]
    assert len(findings) >= 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_missing_and_multiple_ctas():
    linter = ReadabilityLinter()

    # Missing CTA
    res_missing = linter.lint(subject="Hello", body="We noticed you expanded your data clusters in Frankfurt last quarter. Our product is fast.")
    missing_findings = [f for f in res_missing.findings if f.rule_id == "missing_cta"]
    assert len(missing_findings) == 1
    assert missing_findings[0].severity == LinterSeverity.ERROR
    assert res_missing.passed is False

    # Multiple CTAs
    res_multiple = linter.lint(
        subject="Hello",
        body="Would you like to review our benchmarks? Also, are you free for a call this Thursday?",
    )
    multi_findings = [f for f in res_multiple.findings if f.rule_id == "multiple_ctas"]
    assert len(multi_findings) == 1
    assert multi_findings[0].severity == LinterSeverity.WARNING


def test_generic_opening_flagged():
    linter = ReadabilityLinter()
    generic_openings = [
        "I hope this email finds you well. I saw your recent launch.",
        "My name is Alex Rivers and I am reaching out to discuss compute.",
        "Hope you are doing well today. Saw the update.",
    ]
    for opening in generic_openings:
        res = linter.lint(subject="Hello", body=f"{opening} Are you free?")
        findings = [f for f in res.findings if f.rule_id == "generic_opening"]
        assert len(findings) == 1, f"Failed for {opening}"
        assert findings[0].severity == LinterSeverity.WARNING


def test_excessive_first_person_usage():
    linter = ReadabilityLinter()
    body = "I wanted to share what my team and I built, because we think our product can help me and our company. Are you open to a briefing?"
    res = linter.lint(subject="Hello", body=body)
    findings = [f for f in res.findings if f.rule_id == "excessive_first_person"]
    assert len(findings) >= 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_corporate_jargon_flagged():
    linter = ReadabilityLinter()
    body = "We can leverage our best-in-class, cutting-edge technology to create true synergy. Are you open to exploring this?"
    res = linter.lint(subject="Hello", body=body)
    findings = [f for f in res.findings if f.rule_id == "corporate_jargon"]
    assert len(findings) >= 2  # multiple jargon matches
    assert all(f.severity == LinterSeverity.WARNING for f in findings)


def test_excessive_hedging():
    linter = ReadabilityLinter()
    body = "Perhaps we could maybe discuss whether this potentially fits your roadmap. Are you open to a conversation?"
    res = linter.lint(subject="Hello", body=body)
    findings = [f for f in res.findings if f.rule_id == "excessive_hedging"]
    assert len(findings) == 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_excessive_exclamations():
    linter = ReadabilityLinter()
    body = "Congratulations on the launch!! We are super excited! Are you free?"
    res = linter.lint(subject="Hello", body=body)
    findings = [f for f in res.findings if f.rule_id == "excessive_exclamations"]
    assert len(findings) == 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_supported_vs_unsupported_numerical_claims():
    linter = ReadabilityLinter()
    toulmin = _make_dummy_toulmin(proof_points=["50% reduction in pipeline latency"])

    # 1. Supported metric "50%" in body
    body_supported = "Our customers see a 50% reduction in pipeline latency. Are you open to a brief discussion?"
    res_sup = linter.lint(subject="Hello", body=body_supported, toulmin_argument=toulmin)
    assert not any(f.rule_id == "unsupported_numerical_claims" for f in res_sup.findings)

    # 2. Unsupported metric "10x" or "$2M"
    body_unsupported = "Our customers achieve a 10x ROI and $2M in savings. Are you open to a brief discussion?"
    res_unsup = linter.lint(subject="Hello", body=body_unsupported, toulmin_argument=toulmin)
    findings = [f for f in res_unsup.findings if f.rule_id == "unsupported_numerical_claims"]
    assert len(findings) >= 2
    assert all(f.severity == LinterSeverity.WARNING for f in findings)


def test_explicit_outreach_constraints():
    linter = ReadabilityLinter()
    outreach = OutreachContext(
        objective="Initial briefing",
        sender_identity="Alex Rivers",
        constraints=["do not mention AWS", "limit to 40 words"],
    )

    # Violates competitor mention and word limit
    long_body = "We noticed your team is migrating clusters from AWS to bare metal. " * 3 + "Are you open to discussing this?"
    res = linter.lint(subject="AWS Migration", body=long_body, outreach=outreach)
    errs = [f for f in res.findings if f.rule_id == "violated_outreach_constraints"]
    assert len(errs) >= 1
    assert any(f.severity == LinterSeverity.ERROR for f in errs)
    assert res.passed is False


def test_weak_banned_cta():
    linter = ReadabilityLinter()
    body = "I noticed your infrastructure expansion. Thoughts?"
    res = linter.lint(subject="Infrastructure", body=body)
    findings = [f for f in res.findings if f.rule_id == "weak_banned_cta"]
    assert len(findings) >= 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_reading_grade_calculation():
    linter = ReadabilityLinter(max_reading_grade=10.0)
    # Extremely dense academic/bureaucratic text
    dense_body = (
        "The epistemological reconceptualization of multidimensional organizational interoperability "
        "necessitates uncompromisingly rigorous algorithmic characterization of infrastructural paradigms. "
        "Are you open to evaluating this empirical proposition?"
    )
    res = linter.lint(subject="Evaluation", body=dense_body)
    assert res.flesch_kincaid_grade is not None
    assert res.flesch_kincaid_grade > 10.0
    findings = [f for f in res.findings if f.rule_id == "reading_grade_above_10"]
    assert len(findings) == 1
    assert findings[0].severity == LinterSeverity.WARNING


def test_clean_draft_passes_with_zero_errors():
    linter = ReadabilityLinter()
    toulmin = _make_dummy_toulmin(proof_points=["50% reduction in pipeline latency"])
    body = (
        "Noticed Vanguard AI recently expanded its distributed training clusters.\n\n"
        "Teams running similar workloads cut pipeline latency by 50% using automated verification.\n\n"
        "Open to exploring whether this aligns with your Q4 infrastructure goals?"
    )
    res = linter.lint(
        subject="Scaling Vanguard AI infrastructure",
        body=body,
        toulmin_argument=toulmin,
    )
    assert res.passed is True
    assert res.has_errors is False
    assert res.word_count <= 50
