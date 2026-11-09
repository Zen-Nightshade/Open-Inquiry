import datetime
import pytest

from src.researcher.extraction.assessment import EvidenceAssessment, EvidenceAssessor
from src.researcher.models import (
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    ResolutionStatus,
    SourceRecord,
    VerificationStatus,
)


@pytest.fixture
def assessor():
    return EvidenceAssessor()


def test_assess_directly_supported_claim(assessor):
    evidence = EvidenceRecord(
        evidence_id="ev_1",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI raised a 50 million series B funding round led by Sequoia.",
        confidence=ConfidenceLevel.HIGH,
        credibility_score=0.9,
        event_date=datetime.date(2026, 5, 1),
    )
    claim = "Vanguard AI raised 50 million series B funding round."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    assert assessment.confidence == ConfidenceLevel.HIGH
    assert not assessment.conflict_detected
    assert "directly entails" in assessment.support_rationale.lower()


def test_strong_semantic_support_from_weak_source_does_not_yield_high_confidence(assessor):
    """Strong semantic overlap from a low-confidence source must NOT yield HIGH confidence."""
    evidence = EvidenceRecord(
        evidence_id="ev_weak",
        source_id="src_weak",
        verbatim_excerpt="Vanguard AI raised a 50 million series B funding round led by Sequoia.",
        confidence=ConfidenceLevel.LOW,
        credibility_score=0.3,
    )
    claim = "Vanguard AI raised 50 million series B funding round."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    # Evidence confidence is LOW -> composite assessment confidence must be LOW, not HIGH
    assert assessment.confidence == ConfidenceLevel.LOW
    assert assessment.confidence != ConfidenceLevel.HIGH


def test_high_credibility_source_with_inferred_support_does_not_yield_high_confidence(assessor):
    """High channel credibility (0.95) cannot elevate an INFERRED semantic support to HIGH confidence."""
    evidence = EvidenceRecord(
        evidence_id="ev_inf_sec",
        source_id="src_sec",
        verbatim_excerpt="Vanguard AI expanded their enterprise sales team across Europe.",
        confidence=ConfidenceLevel.HIGH,
        credibility_score=0.95,
    )
    # Claim requires domain inference
    claim = "Vanguard AI expanded international enterprise operations in Europe and Asia."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.INFERRED
    # Inferred support caps at MEDIUM, even from an official high-credibility filing
    assert assessment.confidence == ConfidenceLevel.MEDIUM
    assert assessment.confidence != ConfidenceLevel.HIGH


def test_assess_unsupported_numerical_qualifier(assessor):
    """When the excerpt is present but the extracted claim adds an unsupported numerical metric, it must be UNSUPPORTED."""
    evidence = EvidenceRecord(
        evidence_id="ev_num",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI raised a Series B round led by Sequoia Capital.",
        credibility_score=0.9,
    )
    # The claim fabricates '$50 million' which is absent from the excerpt
    claim = "Vanguard AI raised $50 million Series B round led by Sequoia Capital."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.UNSUPPORTED
    assert assessment.confidence == ConfidenceLevel.LOW
    assert "numerical metrics" in assessment.support_rationale.lower()


def test_assess_negation_and_polarity_mismatch(assessor):
    """When an excerpt negates what the claim affirms, it must be flagged as CONTRADICTED."""
    evidence = EvidenceRecord(
        evidence_id="ev_neg",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI denied rumors that it raised a Series B round.",
        credibility_score=0.9,
    )
    claim = "Vanguard AI raised a Series B round."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.CONTRADICTED
    assert assessment.conflict_detected
    assert "refutes or negates" in assessment.support_rationale.lower()


def test_assess_inferred_claim(assessor):
    evidence = EvidenceRecord(
        evidence_id="ev_2",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI expanded their enterprise sales team across Europe.",
        credibility_score=0.8,
    )
    # Claim contains partial overlap requiring domain inference
    claim = "Vanguard AI expanded international enterprise operations in Europe and Asia."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.INFERRED
    assert assessment.confidence == ConfidenceLevel.MEDIUM
    assert not assessment.conflict_detected


def test_assess_future_announcement_preserves_supported(assessor):
    """An announcement or planned release with a future event date is SUPPORTED if the announcement itself is evidenced."""
    future_date = datetime.date.today() + datetime.timedelta(days=90)
    evidence = EvidenceRecord(
        evidence_id="ev_future_announce",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI announced plans to release its neural compiler in December.",
        credibility_score=0.9,
        event_date=future_date,
    )
    claim = "Vanguard AI announced plans to release neural compiler in December."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    # The announcement did happen, so the claim about the announcement is SUPPORTED
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    assert assessment.temporal_notes is not None
    assert "announcement/plan" in assessment.temporal_notes.lower()


def test_assess_future_completed_claim_is_unsupported(assessor):
    """An extracted claim asserting that a future event has already completed cannot be SUPPORTED."""
    future_date = datetime.date.today() + datetime.timedelta(days=90)
    evidence = EvidenceRecord(
        evidence_id="ev_future_complete",
        source_id="src_1",
        verbatim_excerpt="Vanguard AI announced plans to release its neural compiler in December.",
        credibility_score=0.9,
        event_date=future_date,
    )
    # Claim falsely asserts completion of the future event
    claim = "Vanguard AI launched and shipped its neural compiler."

    assessment = assessor.assess_evidence_support(claim_text=claim, evidence=evidence)
    assert assessment.assessment_status == VerificationStatus.UNSUPPORTED
    assert "claim asserts completed outcome" in assessment.temporal_notes.lower()


def test_assess_historical_event_without_arbitrary_threshold(assessor):
    """Historical events (e.g. founded in 2018) are recorded factually without an arbitrary 5-year penalty."""
    past_event = datetime.date(2018, 1, 15)
    pub_date = datetime.date(2026, 2, 1)

    source = SourceRecord(
        source_id="src_pub",
        url="https://example.com/history",
        domain="example.com",
        raw_text="Vanguard AI was originally founded in 2018.",
        publication_date=pub_date,
        fetch_date=datetime.date(2026, 2, 2),
    )
    evidence = EvidenceRecord(
        evidence_id="ev_hist",
        source_id="src_pub",
        verbatim_excerpt="Vanguard AI was originally founded in 2018.",
        credibility_score=0.9,
        event_date=past_event,
    )
    claim = "Vanguard AI was founded in 2018."

    assessment = assessor.assess_evidence_support(
        claim_text=claim,
        evidence=evidence,
        source=source,
    )
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    assert assessment.temporal_notes is not None
    assert "Historical event" in assessment.temporal_notes


def test_entity_attribution_wrong_person(assessor):
    """When an assertion refers to a different candidate entity, target prospect attribution must be excluded."""
    target_prospect_id = "ent_target_prospect"
    other_person_entity = EntityRecord(
        entity_id="ent_other_doctor",
        raw_name="Elena Rostova",
        entity_type=EntityType.PERSON,
        resolution_status=ResolutionStatus.RESOLVED,
    )

    evidence = EvidenceRecord(
        evidence_id="ev_wrong",
        source_id="src_1",
        verbatim_excerpt="Elena Rostova was appointed Chief of Surgery at St. Jude Hospital.",
        credibility_score=0.9,
    )
    claim = "Elena Rostova was appointed Chief of Surgery."

    assessment = assessor.assess_evidence_support(
        claim_text=claim,
        evidence=evidence,
        entity=other_person_entity,
        target_entity_id=target_prospect_id,
    )
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    # Must NOT be attributed to the target prospect
    assert not assessment.target_prospect_attributed
    assert "NOT target prospect" in assessment.entity_notes


def test_entity_attribution_ambiguous_candidate(assessor):
    """When an entity is AMBIGUOUS, candidate reference is preserved, but target prospect attribution is False."""
    ambiguous_entity = EntityRecord(
        entity_id="ent_ambig",
        raw_name="Elena Rostova",
        entity_type=EntityType.PERSON,
        resolution_status=ResolutionStatus.AMBIGUOUS,
    )

    evidence = EvidenceRecord(
        evidence_id="ev_ambig",
        source_id="src_1",
        verbatim_excerpt="Elena Rostova spoke at the conference on neural architecture search.",
        credibility_score=0.9,
    )
    claim = "Elena Rostova spoke at conference on neural architecture search."

    assessment = assessor.assess_evidence_support(
        claim_text=claim,
        evidence=evidence,
        entity=ambiguous_entity,
    )
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    assert not assessment.target_prospect_attributed
    assert assessment.confidence == ConfidenceLevel.MEDIUM
    assert "AMBIGUOUS" in assessment.entity_notes.upper()


def test_entity_attribution_resolved_target_prospect(assessor):
    """When an entity is RESOLVED to the target prospect, target_prospect_attributed is True."""
    target_id = "ent_target"
    resolved_entity = EntityRecord(
        entity_id=target_id,
        raw_name="Elena Rostova",
        entity_type=EntityType.PERSON,
        resolution_status=ResolutionStatus.RESOLVED,
    )

    evidence = EvidenceRecord(
        evidence_id="ev_resolved",
        source_id="src_1",
        verbatim_excerpt="Elena Rostova serves as Head of Research at Vanguard AI.",
        credibility_score=0.9,
    )
    claim = "Elena Rostova serves as Head of Research at Vanguard AI."

    assessment = assessor.assess_evidence_support(
        claim_text=claim,
        evidence=evidence,
        entity=resolved_entity,
        target_entity_id=target_id,
    )
    assert assessment.assessment_status == VerificationStatus.SUPPORTED
    assert assessment.target_prospect_attributed


def test_syndication_independence_checks(assessor):
    domains = {
        "src_1": "techcrunch.com",
        "src_2": "techcrunch.com",
        "src_3": "bloomberg.com",
        "src_4": "reuters.com",
    }
    # 1. Identical source IDs are never independent
    assert not assessor.check_independence("src_1", "src_1", source_domains=domains)

    # 2. Belonging to same syndication group is not independent (confirmed common origin)
    syndication_groups = {
        "group_wire_1": {"src_1", "src_3"},
    }
    assert not assessor.check_independence(
        "src_1", "src_3", syndication_groups=syndication_groups, source_domains=domains
    )

    # 3. Same domain is not independent
    assert not assessor.check_independence("src_1", "src_2", source_domains=domains)

    # 4. Unknown domain / provenance is handled conservatively (NOT independent)
    assert not assessor.check_independence("src_1", "src_unknown", source_domains=domains)
    assert not assessor.check_independence("src_1", "src_3", source_domains=None)

    # 5. Distinct domains, distinct origin groups, distinct excerpts -> independent
    assert assessor.check_independence(
        "src_1", "src_4",
        syndication_groups=syndication_groups,
        source_domains=domains,
        excerpt_a="Vanguard AI raised $50 million Series B led by Sequoia.",
        excerpt_b="In exclusive reporting, Reuters confirmed Vanguard secured $50 million.",
    )

    # 6. Suspected syndication: different domains but identical wire excerpt -> NOT independent
    assert not assessor.check_independence(
        "src_1", "src_4",
        source_domains=domains,
        excerpt_a="SAN FRANCISCO, May 1 (PRNEWSWIRE) -- Vanguard AI today announced $50M Series B.",
        excerpt_b="SAN FRANCISCO, May 1 (PRNEWSWIRE) -- Vanguard AI today announced $50M Series B.",
    )
