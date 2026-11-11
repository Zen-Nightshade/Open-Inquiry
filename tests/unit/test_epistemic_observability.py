from __future__ import annotations

import pytest
from src.researcher.epistemic_dag.models import ObservabilityClass
from src.researcher.epistemic_dag.observability import ObservabilityClassifier


def test_publicly_observable_predicates():
    """Verify official announcements, press releases, funding, and hiring are classified PUBLICLY_OBSERVABLE."""
    preds = [
        ("office_expansion", "Did Acme Corp publicly announce a new London office?"),
        ("announced_funding", "Has the startup closed a Series B funding round?"),
        ("press_release", "Did the company issue a press release regarding the acquisition?"),
        ("headcount_growth", "Are there published hiring notices for sales engineers?"),
    ]
    for pred, text in preds:
        obs, rationale, proxy = ObservabilityClassifier.classify(pred, text)
        assert obs == ObservabilityClass.PUBLICLY_OBSERVABLE
        assert proxy is None


def test_conditionally_observable_predicates():
    """Verify tech stack, vendor usage, and employment status are CONDITIONALLY_OBSERVABLE."""
    preds = [
        ("uses_vendor", "Does Acme Corp currently use Snowflake for data warehousing?"),
        ("works_for", "Is Jane Doe currently employed as VP Engineering?"),
        ("tech_stack", "What frontend framework is used on their customer portal?"),
    ]
    for pred, text in preds:
        obs, rationale, proxy = ObservabilityClassifier.classify(pred, text)
        assert obs == ObservabilityClass.CONDITIONALLY_OBSERVABLE
        assert proxy is None


def test_private_facts_classified_with_proxy():
    """Verify internal budgets, dissatisfaction, and secret roadmaps are classified PRIVATE_OR_NONPUBLIC with distinct proxies."""
    private_cases = [
        (
            "internal_budget",
            "What is Acme Corp's actual internal cloud infrastructure budget?",
            "hiring_volume_for_infrastructure",
        ),
        (
            "vendor_dissatisfaction",
            "Is the prospect privately dissatisfied with their current CRM vendor?",
            "evaluates_alternative_rfp",
        ),
        (
            "unannounced_roadmap",
            "What is the confidential unannounced product roadmap for next year?",
            "trademark_or_patent_filing",
        ),
    ]
    for pred, text, expected_proxy in private_cases:
        obs, rationale, proxy = ObservabilityClassifier.classify(pred, text)
        assert obs == ObservabilityClass.PRIVATE_OR_NONPUBLIC
        assert "private" in rationale.lower() or "internal" in rationale.lower()
        assert proxy is not None
        assert proxy["proxy_predicate"] == expected_proxy
        # The proxy question is distinct and does not overwrite the original
        assert proxy["proxy_question"] != text


def test_unknown_observability_fallback():
    """Verify ambiguous or novel predicates fall back to UNKNOWN and do NOT silently become PRIVATE_OR_NONPUBLIC."""
    obs, rationale, proxy = ObservabilityClassifier.classify("novel_predicate_xyz", "Does the team prefer coffee?")
    assert obs == ObservabilityClass.UNKNOWN
    assert proxy is None


def test_smb_and_private_company_non_bias():
    """Verify an entity being an SMB or private does NOT make its public facts unobservable."""
    obs, rationale, _ = ObservabilityClassifier.classify(
        predicate="office_expansion",
        question_text="Did the 15-person private agency open a new workspace?",
        is_smb=True,
        is_private=True,
    )
    assert obs == ObservabilityClass.PUBLICLY_OBSERVABLE
    assert "SMB or private company status does not restrict" in rationale


def test_empty_search_and_provider_failure_do_not_imply_unobservability():
    """Verify empty search results or network failures do NOT cause unobservable classification."""
    obs_empty, rationale_empty, _ = ObservabilityClassifier.classify(
        predicate="uses_vendor",
        question_text="Does the company use Datadog?",
        search_status="empty",
    )
    assert obs_empty == ObservabilityClass.UNKNOWN
    assert "Search outcome 'empty' indicates search execution status" in rationale_empty

    obs_fail, rationale_fail, _ = ObservabilityClassifier.classify(
        predicate="uses_vendor",
        question_text="Does the company use Datadog?",
        search_status="failed",
    )
    assert obs_fail == ObservabilityClass.UNKNOWN
    assert "Search outcome 'failed' indicates search execution status" in rationale_fail
