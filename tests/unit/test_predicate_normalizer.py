import pytest

from src.researcher.molecular_models import Modality, NormalizedPredicate, Polarity
from src.researcher.predicate_normalizer import PredicateNormalizer


@pytest.fixture
def normalizer():
    return PredicateNormalizer()


def test_planned_vs_executed_hiring_distinction(normalizer):
    """Critical Invariant: Planned and executed actions must NEVER normalize to the same predicate or modality."""
    res_planned = normalizer.normalize("plans to hire")
    res_executed = normalizer.normalize("hired")

    assert res_planned.is_normalized is True
    assert res_planned.normalized_predicate == NormalizedPredicate.HIRING_PLANNED
    assert res_planned.modality == Modality.PLANNED
    assert res_planned.polarity == Polarity.POSITIVE

    assert res_executed.is_normalized is True
    assert res_executed.normalized_predicate == NormalizedPredicate.HIRING_EXECUTED
    assert res_executed.modality == Modality.DEFINITIVE
    assert res_executed.polarity == Polarity.POSITIVE

    # Distinct tokens
    assert res_planned.normalized_predicate != res_executed.normalized_predicate
    assert res_planned.modality != res_executed.modality


def test_core_controlled_business_predicates(normalizer):
    """Verify controlled vocabulary maps standard business event verb phrases."""
    # Corporate
    assert normalizer.normalize("opened facility").normalized_predicate == NormalizedPredicate.FACILITY_OPENED
    assert normalizer.normalize("closed office").normalized_predicate == NormalizedPredicate.FACILITY_CLOSED
    assert normalizer.normalize("headquartered in").normalized_predicate == NormalizedPredicate.HEADQUARTERED_IN
    assert normalizer.normalize("operating in").normalized_predicate == NormalizedPredicate.OPERATING_IN

    # Capital
    assert normalizer.normalize("raised series").normalized_predicate == NormalizedPredicate.FUNDING_RAISED
    assert normalizer.normalize("secured funding").normalized_predicate == NormalizedPredicate.FUNDING_RAISED
    assert normalizer.normalize("acquired").normalized_predicate == NormalizedPredicate.ACQUISITION_ACQUIRED
    assert normalizer.normalize("partnered with").normalized_predicate == NormalizedPredicate.PARTNERSHIP_FORMED

    # Headcount & Leadership
    assert normalizer.normalize("headcount of").normalized_predicate == NormalizedPredicate.EMPLOYEE_COUNT
    assert normalizer.normalize("appointed").normalized_predicate == NormalizedPredicate.LEADERSHIP_APPOINTED
    assert normalizer.normalize("resigned").normalized_predicate == NormalizedPredicate.LEADERSHIP_DEPARTED

    # Product
    assert normalizer.normalize("launched").normalized_predicate == NormalizedPredicate.PRODUCT_LAUNCHED
    assert normalizer.normalize("unveiled").normalized_predicate == NormalizedPredicate.PRODUCT_LAUNCHED


def test_unmapped_unknown_predicates_are_first_class_supported(normalizer):
    """Unknown verbs are preserved cleanly without data loss or false categorization."""
    custom_pred = "arbitrarily restructured into autonomous squads"
    res = normalizer.normalize(custom_pred)

    assert res.is_normalized is False
    assert res.normalized_predicate is None
    assert res.raw_predicate == custom_pred
    assert res.modality == Modality.DEFINITIVE
    assert res.polarity == Polarity.POSITIVE


def test_explicit_negation_inverts_polarity(normalizer):
    """Explicit negation prefixes invert polarity without destroying predicate recognition."""
    res = normalizer.normalize("did not open")
    assert res.is_normalized is True
    assert res.normalized_predicate == NormalizedPredicate.FACILITY_OPENED
    assert res.polarity == Polarity.NEGATIVE

    res_no_plans = normalizer.normalize("no plans to hire")
    assert res_no_plans.is_normalized is True
    assert res_no_plans.normalized_predicate == NormalizedPredicate.HIRING_PLANNED
    assert res_no_plans.polarity == Polarity.NEGATIVE


def test_conditional_markers_set_conditional_modality(normalizer):
    """Explicit conditional hedges mark Modality.CONDITIONAL."""
    res = normalizer.normalize("acquired subject to regulatory approval")
    assert res.is_normalized is True
    assert res.normalized_predicate == NormalizedPredicate.ACQUISITION_ACQUIRED
    assert res.modality == Modality.CONDITIONAL


def test_empty_or_whitespace_predicate(normalizer):
    """Empty or whitespace strings return safe unnormalized result."""
    res = normalizer.normalize("   ")
    assert res.is_normalized is False
    assert res.normalized_predicate is None
