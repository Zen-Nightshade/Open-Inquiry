from __future__ import annotations

import pytest

from src.context.models import ProspectContext
from src.researcher.entity.candidates import (
    DiscoveredEntityCandidate,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.entity.probabilistic_models import (
    CandidateMatchAssessment,
    ResolutionOutcome,
)
from src.researcher.entity.probabilistic_resolver import (
    ProbabilisticEntityResolver,
    ProbabilisticResolutionResult,
)
from src.researcher.models import EntityRecord, EntityType, ResolutionStatus


def test_empty_candidates_returns_unresolved():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-empty",
        first_name="Jane",
        last_name="Doe",
        full_name="Jane Doe",
        company_name="Apex Global",
    )

    result = resolver.resolve_candidates(candidates=[], target_prospect=target)

    assert result.outcome == ResolutionOutcome.UNRESOLVED
    assert result.resolution_status == ResolutionStatus.UNRESOLVED
    assert result.selected_candidate_id is None
    assert result.best_score == 0.0
    assert "No candidate entities were discovered" in result.rationale


def test_clear_single_match_with_sufficient_margin():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-clear",
        first_name="Marcus",
        last_name="Vance",
        full_name="Marcus Vance",
        designation="VP Technology",
        company_name="Vance Enterprises",
        company_domain="vance.com",
    )

    c_best = DiscoveredEntityCandidate(
        candidate_id="c-vance-1",
        raw_name="Marcus Vance",
        normalized_name=normalize_person_name("Marcus Vance"),
        entity_type=EntityType.PERSON,
        source_ids=["src-1"],
        organizations=["Vance Enterprises"],
        domains=["vance.com"],
        roles=["VP Technology"],
    )

    c_runner = DiscoveredEntityCandidate(
        candidate_id="c-vance-2",
        raw_name="Mark Vance",
        normalized_name=normalize_person_name("Mark Vance"),
        entity_type=EntityType.PERSON,
        source_ids=["src-2"],
        organizations=["Vance Enterprises"],
        domains=["vance.com"],
        roles=["Junior Developer"],
    )

    result = resolver.resolve_candidates(
        candidates=[c_best, c_runner],
        target_prospect=target,
    )

    assert result.outcome == ResolutionOutcome.MATCH
    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.selected_candidate_id == "c-vance-1"
    assert result.best_score >= 0.75
    assert result.score_margin is not None
    assert result.score_margin >= 0.15
    assert len(result.competing_candidate_ids) == 0


def test_close_viable_competing_candidates_produces_ambiguous():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-ambig",
        first_name="Michael",
        last_name="Chang",
        full_name="Michael Chang",
        designation="Director of Data",
        company_name="DataCorp Global",
        company_domain="datacorp.com",
    )

    # Candidate 1: Mike Chang at DataCorp
    c1 = DiscoveredEntityCandidate(
        candidate_id="c-chang-1",
        raw_name="Mike Chang",
        normalized_name=normalize_person_name("Mike Chang"),
        entity_type=EntityType.PERSON,
        source_ids=["s-1"],
        organizations=["DataCorp Global"],
        domains=["datacorp.com"],
        roles=["Director of Data Science"],
    )

    # Candidate 2: Michael Chang at DataCorp
    c2 = DiscoveredEntityCandidate(
        candidate_id="c-chang-2",
        raw_name="Michael Chang",
        normalized_name=normalize_person_name("Michael Chang"),
        entity_type=EntityType.PERSON,
        source_ids=["s-2"],
        organizations=["DataCorp Global"],
        domains=["datacorp.com"],
        roles=["Data Director"],
    )

    result = resolver.resolve_candidates(
        candidates=[c1, c2],
        target_prospect=target,
    )

    # Margin between the two is < 0.15 -> must be AMBIGUOUS!
    assert result.outcome == ResolutionOutcome.AMBIGUOUS
    assert result.resolution_status == ResolutionStatus.AMBIGUOUS
    assert result.selected_candidate_id is None
    assert len(result.competing_candidate_ids) == 2
    assert "c-chang-1" in result.competing_candidate_ids
    assert "c-chang-2" in result.competing_candidate_ids
    assert "Ambiguous match" in result.rationale


def test_disqualified_candidate_does_not_create_artificial_ambiguity():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-disq",
        first_name="Samantha",
        last_name="Reed",
        full_name="Samantha Reed",
        designation="Head of Security",
        company_name="Fortress Bio",
        company_domain="fortressbio.com",
    )

    # Valid candidate matching Fortress Bio
    c_valid = DiscoveredEntityCandidate(
        candidate_id="c-valid",
        raw_name="Samantha Reed",
        normalized_name=normalize_person_name("Samantha Reed"),
        entity_type=EntityType.PERSON,
        source_ids=["s-1"],
        organizations=["Fortress Bio"],
        domains=["fortressbio.com"],
        roles=["Head of Security"],
    )

    # Disqualified candidate with same name at a conflicting verified employer
    c_disqualified = DiscoveredEntityCandidate(
        candidate_id="c-disq",
        raw_name="Samantha Reed",
        normalized_name=normalize_person_name("Samantha Reed"),
        entity_type=EntityType.PERSON,
        source_ids=["s-2"],
        organizations=["Other Rival Corp"],
        domains=["otherrival.com"],
        roles=["Head of Security"],
    )

    result = resolver.resolve_candidates(
        candidates=[c_valid, c_disqualified],
        target_prospect=target,
    )

    # Disqualified candidate must be filtered out of viable candidates and not cause ambiguity
    assert result.outcome == ResolutionOutcome.MATCH
    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.selected_candidate_id == "c-valid"


def test_strong_non_match_with_affirmative_contradictory_evidence():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-nomatch",
        first_name="Laura",
        last_name="Croft",
        full_name="Laura Croft",
        company_name="Archeo Labs",
        company_domain="archeolabs.com",
    )

    # Discovered candidate has contradictory employer and conflicting domain
    c_contradictory = DiscoveredEntityCandidate(
        candidate_id="c-nomatch-1",
        raw_name="Laura Croft",
        normalized_name=normalize_person_name("Laura Croft"),
        entity_type=EntityType.PERSON,
        source_ids=["s-1"],
        organizations=["Tomb Raiders LLC"],
        domains=["tombraiders.com"],
    )

    result = resolver.resolve_candidates(
        candidates=[c_contradictory],
        target_prospect=target,
    )

    assert result.outcome == ResolutionOutcome.NO_MATCH
    assert result.selected_candidate_id is None
    assert "contradictory" in result.rationale.lower() or "non-match" in result.rationale.lower()


def test_insufficient_evidence_produces_unresolved():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-unres",
        first_name="David",
        last_name="Kim",
        full_name="David Kim",
        company_name="Horizon AI",
    )

    # Only a common name match, without employer, domain, or role evidence
    c_weak = DiscoveredEntityCandidate(
        candidate_id="c-weak-1",
        raw_name="David Kim",
        normalized_name=normalize_person_name("David Kim"),
        entity_type=EntityType.PERSON,
        source_ids=["s-1"],
    )

    result = resolver.resolve_candidates(
        candidates=[c_weak],
        target_prospect=target,
    )

    # Name alone is capped at 0.45 (between 0.30 and 0.75) with missing attributes -> UNRESOLVED
    assert result.outcome == ResolutionOutcome.UNRESOLVED
    assert result.resolution_status == ResolutionStatus.UNRESOLVED
    assert result.selected_candidate_id is None
    assert "Insufficient evidence" in result.rationale


def test_entity_type_isolation_and_filtering():
    resolver = ProbabilisticEntityResolver()
    target = ProspectContext(
        prospect_id="p-type",
        first_name="Wayne",
        last_name="Enterprises",
        full_name="Wayne Enterprises",
        company_name="Wayne Enterprises",
    )

    # Only Organization candidate, but resolver target is PERSON
    c_org = DiscoveredEntityCandidate(
        candidate_id="c-org-1",
        raw_name="Wayne Enterprises Inc",
        normalized_name=normalize_org_name("Wayne Enterprises Inc"),
        entity_type=EntityType.ORGANIZATION,
        source_ids=["s-1"],
    )

    result = resolver.resolve_candidates(
        candidates=[c_org],
        target_prospect=target,
        entity_type=EntityType.PERSON,
    )

    # Should report NO_MATCH due to type incompatibility
    assert result.outcome == ResolutionOutcome.NO_MATCH
    assert "incompatible entity types" in result.rationale.lower()
