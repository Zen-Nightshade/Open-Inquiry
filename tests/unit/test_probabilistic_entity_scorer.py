from __future__ import annotations

import pytest

from src.context.models import ProspectContext
from src.researcher.entity.candidates import (
    DiscoveredEntityCandidate,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.entity.probabilistic_models import (
    FeatureScore,
    ResolutionOutcome,
)
from src.researcher.entity.probabilistic_scorer import (
    ProbabilisticCandidateScorer,
    are_names_nickname_match,
    compute_levenshtein_similarity,
)
from src.researcher.models import EntityRecord, EntityType


def test_nickname_matching_and_levenshtein_similarity():
    # Bidirectional nickname pairs
    assert are_names_nickname_match("Robert", "Bob") is True
    assert are_names_nickname_match("bob", "robert") is True
    assert are_names_nickname_match("William", "Bill") is True
    assert are_names_nickname_match("Elizabeth", "Liz") is True
    assert are_names_nickname_match("Alexander", "Alex") is True
    assert are_names_nickname_match("John", "Alice") is False
    assert are_names_nickname_match("Same", "Same") is False

    # Levenshtein similarity
    assert compute_levenshtein_similarity("John Doe", "John Doe") == 1.0
    sim_minor = compute_levenshtein_similarity("Jonathon", "Jonathan")
    assert sim_minor >= 0.85
    sim_distinct = compute_levenshtein_similarity("Apple", "Microsoft")
    assert sim_distinct < 0.20


def test_person_exact_name_and_org_scoring():
    scorer = ProbabilisticCandidateScorer()

    target = ProspectContext(
        prospect_id="p-1",
        first_name="Sarah",
        last_name="Connor",
        full_name="Sarah Connor",
        designation="Chief Information Security Officer",
        company_name="Cyberdyne Systems",
        company_domain="cyberdyne.com",
    )

    candidate = DiscoveredEntityCandidate(
        candidate_id="c-1",
        raw_name="Sarah Connor",
        normalized_name=normalize_person_name("Sarah Connor"),
        entity_type=EntityType.PERSON,
        source_ids=["s-1"],
        organizations=["Cyberdyne Systems"],
        domains=["cyberdyne.com"],
        roles=["CISO", "Chief Information Security Officer"],
    )

    assessment = scorer.score_candidate(candidate=candidate, target_prospect=target)

    assert assessment.match_score >= 0.85
    assert assessment.outcome == ResolutionOutcome.MATCH
    assert assessment.is_disqualified is False
    assert any("Sarah Connor" in e for e in assessment.positive_evidence)

    # Verify feature breakdown includes name, org, and domain
    feat_names = {f.feature_name for f in assessment.feature_scores}
    assert "person_name_group" in feat_names
    assert "person_employer_group" in feat_names


def test_nickname_scored_positively_without_penalization():
    scorer = ProbabilisticCandidateScorer()

    target = ProspectContext(
        prospect_id="p-2",
        first_name="Robert",
        last_name="Smith",
        full_name="Robert Smith",
        designation="VP Engineering",
        company_name="Acme Corp",
        company_domain="acme.com",
    )

    # Candidate has nickname 'Bob Smith'
    candidate = DiscoveredEntityCandidate(
        candidate_id="c-2",
        raw_name="Bob Smith",
        normalized_name=normalize_person_name("Bob Smith"),
        entity_type=EntityType.PERSON,
        source_ids=["s-2"],
        organizations=["Acme Corp"],
        domains=["acme.com"],
        roles=["VP of Engineering"],
    )

    assessment = scorer.score_candidate(candidate=candidate, target_prospect=target)

    assert assessment.match_score >= 0.75
    # Should identify nickname match in rationale
    feat_name = next((f for f in assessment.feature_scores if f.feature_name == "person_name_group"), None)
    assert feat_name is not None
    assert "nickname match" in feat_name.rationale
    assert feat_name.similarity == 0.90
    assert feat_name.contribution > 0


def test_acronym_org_matching():
    scorer = ProbabilisticCandidateScorer()

    target_org = EntityRecord(
        entity_id="e-org-1",
        raw_name="International Business Machines",
        canonical_name="International Business Machines",
        entity_type=EntityType.ORGANIZATION,
    )

    candidate_org = DiscoveredEntityCandidate(
        candidate_id="c-org-1",
        raw_name="IBM",
        normalized_name=normalize_org_name("IBM"),
        entity_type=EntityType.ORGANIZATION,
        source_ids=["s-3"],
        domains=["ibm.com"],
    )

    assessment = scorer.score_candidate(candidate=candidate_org, target_entity=target_org)

    # Should detect acronym match in org_name_group
    feat_org = next((f for f in assessment.feature_scores if f.feature_name == "org_name_group"), None)
    assert feat_org is not None
    assert "acronym match" in feat_org.rationale
    assert feat_org.similarity == 0.90
    assert assessment.match_score >= 0.40


def test_conflicting_verified_domain_penalizes_and_disqualifies():
    scorer = ProbabilisticCandidateScorer()

    target = ProspectContext(
        prospect_id="p-3",
        first_name="Jane",
        last_name="Doe",
        full_name="Jane Doe",
        company_name="Global Tech",
        company_domain="globaltech.com",
    )

    # Same person name, but candidate is at a verified conflicting domain/company
    candidate = DiscoveredEntityCandidate(
        candidate_id="c-3",
        raw_name="Jane Doe",
        normalized_name=normalize_person_name("Jane Doe"),
        entity_type=EntityType.PERSON,
        source_ids=["s-4"],
        organizations=["Competitor Inc"],
        domains=["competitor.com"],
    )

    assessment = scorer.score_candidate(candidate=candidate, target_prospect=target)

    # Conflicting verified domain should produce a negative penalty
    penalties = [f for f in assessment.feature_scores if f.is_negative_penalty]
    assert len(penalties) > 0
    assert any(p.feature_name in ("conflicting_employer_penalty", "conflicting_verified_domain") for p in penalties)
    assert assessment.is_disqualified is True
    assert assessment.match_score < 0.40


def test_missing_attributes_treated_neutrally():
    scorer = ProbabilisticCandidateScorer()

    # Target has minimal info (no location, no role)
    target = ProspectContext(
        prospect_id="p-4",
        first_name="Alex",
        last_name="Rivera",
        full_name="Alex Rivera",
        company_name="Nova Labs",
    )

    # Candidate has no role or location
    candidate = DiscoveredEntityCandidate(
        candidate_id="c-4",
        raw_name="Alex Rivera",
        normalized_name=normalize_person_name("Alex Rivera"),
        entity_type=EntityType.PERSON,
        source_ids=["s-5"],
        organizations=["Nova Labs"],
    )

    assessment = scorer.score_candidate(candidate=candidate, target_prospect=target)

    # Missing features are recorded, but have 0 penalty
    assert len(assessment.missing_features) > 0
    for feat in assessment.feature_scores:
        if feat.is_missing:
            assert feat.contribution == 0.0
            assert feat.is_negative_penalty is False


def test_correlated_features_not_double_counted():
    scorer = ProbabilisticCandidateScorer()

    target = ProspectContext(
        prospect_id="p-5",
        first_name="David",
        last_name="Miller",
        full_name="David Miller",
        company_name="Acme Corp",
    )

    candidate = DiscoveredEntityCandidate(
        candidate_id="c-5",
        raw_name="David Miller",
        normalized_name=normalize_person_name("David Miller"),
        entity_type=EntityType.PERSON,
        source_ids=["s-6"],
    )

    assessment = scorer.score_candidate(candidate=candidate, target_prospect=target)

    # Even though exact name, token Jaccard, and Levenshtein all match,
    # the total name contribution cannot exceed the name group weight cap (0.45)
    name_contributions = sum(
        f.contribution for f in assessment.feature_scores
        if f.feature_group == "name"
    )
    assert name_contributions <= 0.451


def test_entity_type_incompatibility_hard_disqualification():
    scorer = ProbabilisticCandidateScorer()

    target = ProspectContext(
        prospect_id="p-6",
        first_name="Alice",
        last_name="Walker",
        full_name="Alice Walker",
        company_name="Alpha Tech",
    )

    # Candidate is an Organization with name 'Alice Walker Foundation'
    candidate = DiscoveredEntityCandidate(
        candidate_id="c-6",
        raw_name="Alice Walker Foundation",
        normalized_name=normalize_org_name("Alice Walker Foundation"),
        entity_type=EntityType.ORGANIZATION,
        source_ids=["s-7"],
    )

    assessment = scorer.score_candidate(candidate=candidate, target_prospect=target)

    assert assessment.match_score == 0.0
    assert assessment.is_disqualified is True
    assert assessment.outcome == ResolutionOutcome.NO_MATCH
    assert "Incompatible entity types" in assessment.negative_evidence[0]
