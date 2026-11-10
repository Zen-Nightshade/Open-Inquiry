from __future__ import annotations

import pytest

from src.email_generation.claim_verifier.decomposer import ClaimDecomposer
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    ModalityLevel,
)


@pytest.fixture
def decomposer() -> ClaimDecomposer:
    return ClaimDecomposer()


def test_decomposer_exact_character_spans(decomposer: ClaimDecomposer):
    subject = "Cluster Telemetry Inquiry"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa.\n\n"
        "When clusters grow fast, tracking can create heavy system lag. We built PulseRadar to cut node latency by 50% without adding extra overhead to your stack.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    claims, findings = decomposer.decompose(
        subject=subject,
        body=body,
        prospect_name="Elena Vance",
        company_name="Black Mesa",
        product_name="PulseRadar",
    )

    assert len(findings) == 0, f"Found unexpected decomposition findings: {findings}"
    assert len(claims) > 0

    # Verify exact character spans against source texts
    for claim in claims:
        src = subject if claim.section == ClaimSection.SUBJECT else body
        extracted = src[claim.span_start : claim.span_end]
        assert extracted == claim.text, (
            f"Span mismatch: [{claim.span_start}:{claim.span_end}] extracted '{extracted}' != claim '{claim.text}'"
        )


def test_decomposer_distinguishes_subject_and_body_sections(decomposer: ClaimDecomposer):
    subject = "Black Mesa Node Infrastructure"
    body = "Saw your recent infrastructure scaling."

    claims, findings = decomposer.decompose(
        subject=subject,
        body=body,
        company_name="Black Mesa",
    )

    subj_claims = [c for c in claims if c.section == ClaimSection.SUBJECT]
    body_claims = [c for c in claims if c.section == ClaimSection.BODY]

    assert len(subj_claims) == 1
    assert subj_claims[0].text == "Black Mesa Node Infrastructure"
    assert subj_claims[0].span_start == 0
    assert subj_claims[0].span_end == len(subject)

    assert len(body_claims) == 1
    assert body_claims[0].text == "Saw your recent infrastructure scaling."
    assert body_claims[0].span_start == 0
    assert body_claims[0].span_end == len(body)


def test_decomposer_handles_greeting_and_multiple_propositions(decomposer: ClaimDecomposer):
    subject = "Telemetry Review"
    body = (
        "Elena, saw your team added 10,000 new compute nodes at Black Mesa, "
        "and we know cluster tracking creates heavy system lag."
    )

    claims, findings = decomposer.decompose(
        subject=subject,
        body=body,
        prospect_name="Elena Vance",
        company_name="Black Mesa",
    )

    body_claims = [c for c in claims if c.section == ClaimSection.BODY]
    # Greeting "Elena, " should be skipped and compound sentence split into two propositions
    assert len(body_claims) == 2

    c1 = body_claims[0]
    assert c1.text == "saw your team added 10,000 new compute nodes at Black Mesa"
    assert body[c1.span_start : c1.span_end] == c1.text
    assert c1.category in (ClaimCategory.PROSPECT_OBSERVATION, ClaimCategory.METRIC_IMPACT)

    c2 = body_claims[1]
    assert "cluster tracking creates heavy system lag" in c2.text
    assert body[c2.span_start : c2.span_end] == c2.text


def test_decomposer_identifies_diagnostic_cta_and_modality(decomposer: ClaimDecomposer):
    subject = "Telemetry benchmarks"
    body = (
        "We built PulseRadar to cut latency.\n\n"
        "Are you open to comparing cluster telemetry benchmarks this week?"
    )

    claims, _ = decomposer.decompose(
        subject=subject,
        body=body,
        product_name="PulseRadar",
    )

    cta_claims = [c for c in claims if c.category == ClaimCategory.RELATIONAL_CTA]
    assert len(cta_claims) == 1
    assert cta_claims[0].modality == ModalityLevel.DIAGNOSTIC
    assert cta_claims[0].text == "Are you open to comparing cluster telemetry benchmarks this week?"


def test_decomposer_identifies_conditional_hedges(decomposer: ClaimDecomposer):
    subject = "Infrastructure questions"
    body = (
        "If your clusters face high synchronization overhead, PulseRadar could help reduce queue bottlenecks."
    )

    claims, _ = decomposer.decompose(subject=subject, body=body, product_name="PulseRadar")
    body_claims = [c for c in claims if c.section == ClaimSection.BODY]
    assert len(body_claims) >= 1
    assert any(c.modality in (ModalityLevel.CONDITIONAL, ModalityLevel.HYPOTHETICAL) for c in body_claims)


def test_decomposer_ignores_signoff_lines(decomposer: ClaimDecomposer):
    subject = "Node metrics"
    body = (
        "Saw your team added new compute nodes.\n\n"
        "Best,\n"
        "Alex"
    )

    claims, _ = decomposer.decompose(subject=subject, body=body)
    body_claims = [c for c in claims if c.section == ClaimSection.BODY]
    # "Best,\nAlex" should not be treated as a factual claim
    assert not any("Best" in c.text for c in body_claims)
    assert not any(c.text == "Alex" for c in body_claims)
