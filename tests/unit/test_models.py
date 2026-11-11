from __future__ import annotations

import datetime
import pytest
from pydantic import ValidationError

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.email_generation.models import (
    AtomicClaim,
    ClaimType,
    ClaimVerification,
    EmailDraft,
    HumanReviewStatus,
    QualityDimension,
    QualityEvaluation,
    VerificationStatus,
    VerifiedOutreachDraft,
)
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    ResearchPackage,
    ResolutionStatus,
    SourceRecord,
    SourceType,
    SufficiencyResult,
)
from src.signal_strategy.models import (
    CandidateSignal,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    SignalScore,
)


def test_context_models_validation_and_immutability():
    prospect = ProspectContext(
        full_name="Sarah Connor",
        company_name="Cyberdyne Systems",
        company_domain="cyberdyne.com",
        designation="VP Engineering",
        location="Los Angeles, CA",
    )
    assert prospect.full_name == "Sarah Connor"
    assert prospect.company_domain == "cyberdyne.com"

    # Verify immutability (frozen=True)
    with pytest.raises(ValidationError):
        prospect.full_name = "John Connor"  # type: ignore

    product = ProductContext(
        product_name="Open Inquiry Automator",
        description="Automated outreach intelligence engine",
        capabilities=["Signal extraction", "Grounded email synthesis"],
        value_proposition="10x SDR productivity",
    )
    assert len(product.capabilities) == 2
    with pytest.raises(ValidationError):
        product.value_proposition = "New value"  # type: ignore

    outreach = OutreachContext(
        objective="Book discovery call",
        sender_identity="Alex Mercer, Head of Sales at AlphaCorp",
    )
    assert outreach.tone == "professional"
    assert outreach.relationship == "cold"


def test_unresolved_entity_record():
    # Supports completely unresolved entity
    unresolved_person = EntityRecord(
        entity_type=EntityType.PERSON,
        raw_name="Dr. Smith",
    )
    assert unresolved_person.canonical_name is None
    assert unresolved_person.resolution_status == ResolutionStatus.UNRESOLVED
    assert unresolved_person.confidence == ConfidenceLevel.UNRESOLVED

    # Supports resolved entity
    resolved_company = EntityRecord(
        entity_type=EntityType.ORGANIZATION,
        raw_name="Stark Ind.",
        canonical_name="Stark Industries Inc.",
        domain="starkindustries.com",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )
    assert resolved_company.canonical_name == "Stark Industries Inc."
    assert resolved_company.domain == "starkindustries.com"


def test_source_provenance_and_credibility():
    source = SourceRecord(
        url="https://techcrunch.com/article1",
        domain="techcrunch.com",
        title="Cyberdyne secures Series B",
        raw_text="Cyberdyne systems announced an investment round...",
        source_type=SourceType.NEWS,
        credibility_score=0.9,
        fetch_date=datetime.date(2026, 11, 1),
        publication_date=datetime.date(2026, 10, 28),
    )
    assert source.credibility_score == 0.9
    assert source.source_type == SourceType.NEWS

    # Credibility score must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        SourceRecord(
            url="https://invalid.com",
            domain="invalid.com",
            raw_text="text",
            credibility_score=1.5,
            fetch_date=datetime.date(2026, 11, 1),
        )


def test_canonical_fact_multiple_evidence_preservation():
    ev1 = EvidenceRecord(
        source_id="src_1",
        verbatim_excerpt="Cyberdyne raised $50M in Series B funding led by Tech Ventures.",
        event_date=datetime.date(2026, 10, 15),
        extracted_date=datetime.date(2026, 11, 1),
        confidence=ConfidenceLevel.HIGH,
        credibility_score=0.95,
    )
    ev2 = EvidenceRecord(
        source_id="src_2",
        verbatim_excerpt="SEC filing confirms $50,000,000 equity round for Cyberdyne Systems.",
        event_date=datetime.date(2026, 10, 16),
        extracted_date=datetime.date(2026, 11, 2),
        confidence=ConfidenceLevel.HIGH,
        credibility_score=0.99,
    )

    fact = CanonicalFact(
        claim_text="Cyberdyne Systems raised $50M Series B funding in October 2026",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_cyberdyne"],
        evidence_records=[ev1, ev2],
        primary_event_date=datetime.date(2026, 10, 15),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    assert len(fact.evidence_records) == 2
    assert fact.source_ids == ["src_1", "src_2"]

    # CanonicalFact requires at least one evidence record
    with pytest.raises(ValidationError):
        CanonicalFact(
            claim_text="Unsupported claim",
            evidence_records=[],
        )


def test_research_accumulator_and_incomplete_research_package():
    accumulator = ResearchAccumulator()

    src = SourceRecord(
        url="https://cyberdyne.com/about",
        domain="cyberdyne.com",
        raw_text="Official site",
        fetch_date=datetime.date(2026, 11, 1),
    )
    accumulator.add_source(src)
    assert "https://cyberdyne.com/about" in accumulator.seen_urls

    ev1 = EvidenceRecord(
        source_id=src.source_id,
        verbatim_excerpt="Our AI division grew 200% this quarter.",
        extracted_date=datetime.date(2026, 11, 1),
    )
    fact1 = CanonicalFact(
        claim_text="Cyberdyne AI division grew 200%",
        evidence_records=[ev1],
    )
    accumulator.merge_or_add_fact(fact1)
    assert len(accumulator.facts) == 1

    # Simulate deduplication where an independent second source corroborates the same fact
    ev2 = EvidenceRecord(
        source_id="src_press_release",
        verbatim_excerpt="AI division experienced 200% headcount expansion.",
        extracted_date=datetime.date(2026, 11, 2),
    )
    fact2_duplicate = CanonicalFact(
        claim_text="Cyberdyne AI division expanded headcount by 200%",
        evidence_records=[ev2],
    )
    # Merging preserves both independent evidence records
    accumulator.merge_or_add_fact(fact2_duplicate, match_fact_id=fact1.fact_id)

    assert len(accumulator.facts) == 1
    merged = accumulator.facts[0]
    assert len(merged.evidence_records) == 2
    assert merged.composite_confidence == ConfidenceLevel.HIGH

    # Freeze into ResearchPackage supporting incomplete research
    pkg = accumulator.to_package(
        is_sufficient=False,
        is_complete=False,
        missing_information=["Prospect email address", "Q4 revenue target"],
        sufficiency_notes="Prospect contact info could not be resolved from public sources",
    )
    assert pkg.is_sufficient is False
    assert pkg.is_complete is False
    assert len(pkg.missing_information) == 2
    assert len(pkg.facts) == 1
    assert isinstance(pkg.facts, tuple)
    assert isinstance(pkg.sources, tuple)
    assert isinstance(pkg.missing_information, tuple)
    assert isinstance(pkg.search_queries_used, tuple)

    # Verify frozen immutability of ResearchPackage
    with pytest.raises(ValidationError):
        pkg.is_sufficient = True  # type: ignore


def test_signal_strategy_and_email_models():
    candidate = CandidateSignal(
        label="AI Division Expansion",
        description="Company expanded AI division by 200%",
        supporting_fact_ids=["fact_1"],
    )
    score = SignalScore(
        evidence_strength=0.9,
        recency_score=0.85,
        prospect_relevance=0.8,
        product_alignment=0.95,
        business_significance=0.85,
        composite_score=0.88,
    )
    ranked = RankedSignal(
        signal=candidate,
        score=score,
        rationale="Strong alignment with our AI infrastructure optimization platform.",
    )
    alignment = ProspectProductAlignment(
        prospect_role_relevance="As VP Engineering, she is scaling the new AI infrastructure.",
        business_implications="Rapid expansion increases compute overhead and reliability risk.",
        capability_mapping=["Automated cluster scaling", "Cost intelligence dashboard"],
    )
    strategy = OutreachStrategy(
        primary_signal=ranked,
        alignment=alignment,
        messaging_approach="growth-scaling-pain",
        central_message="Scale your new AI clusters without runaway compute bills.",
        suggested_subject_line="Scaling Cyberdyne's AI cluster efficiency",
        cta="15-min benchmark chat next Tuesday?",
        tone_guidance="Direct, peer-to-peer technical executive tone",
    )

    ev_record = EvidenceRecord(
        source_id="src_1",
        verbatim_excerpt="AI division grew 200%",
        extracted_date=datetime.date(2026, 11, 1),
    )
    fact = CanonicalFact(
        claim_text="AI division grew 200%",
        evidence_records=[ev_record],
    )
    research_pkg = ResearchPackage(
        sources=[],
        facts=[fact],
        is_sufficient=True,
    )

    draft = EmailDraft(
        subject="Scaling Cyberdyne's AI cluster efficiency",
        body="Sarah, noticed your AI division expanded 200% recently. Are you hitting compute bottlenecks?",
        word_count=16,
    )

    claim = AtomicClaim(
        claim_text="AI division expanded 200% recently",
        claim_type=ClaimType.FACT_ABOUT_COMPANY,
    )
    verification = ClaimVerification(
        claim=claim,
        status=VerificationStatus.SUPPORTED,
        supporting_fact_ids=[fact.fact_id],
        supporting_evidence_ids=[ev_record.evidence_id],
        explanation="Directly entailed by TechCrunch announcement excerpt.",
    )

    evaluation = QualityEvaluation(
        dimensions=[
            QualityDimension(dimension="personalization", score=5, justification="Specific signal hook"),
            QualityDimension(dimension="relevance", score=5, justification="Matches VP Engineering priority"),
            QualityDimension(dimension="product_alignment", score=4, justification="Clear solution link"),
            QualityDimension(dimension="coherence", score=5, justification="Natural flow"),
            QualityDimension(dimension="conciseness", score=5, justification="Under 50 words"),
            QualityDimension(dimension="tone", score=5, justification="Respectful and executive"),
            QualityDimension(dimension="cta_quality", score=4, justification="Specific time ask"),
        ],
        overall_score=4.7,
        is_acceptable=True,
    )

    # Verified draft contract with human review status and reasons
    verified_draft = VerifiedOutreachDraft(
        draft=draft,
        claim_verifications=[verification],
        quality_evaluation=evaluation,
        research_package=research_pkg,
        outreach_strategy=strategy,
        generation_attempts=1,
        human_review_status=HumanReviewStatus.PENDING_REVIEW,
        human_review_reasons=["First-time outreach to executive prospect", "Verify calendar availability"],
    )

    assert verified_draft.human_review_status == HumanReviewStatus.PENDING_REVIEW
    assert len(verified_draft.human_review_reasons) == 2
    assert verified_draft.claim_verifications[0].status == VerificationStatus.SUPPORTED
    assert verified_draft.claim_verifications[0].supporting_evidence_ids == [ev_record.evidence_id]

    # Verify JSON serialization works seamlessly
    json_data = verified_draft.model_dump_json()
    assert "Scaling Cyberdyne" in json_data
    assert "pending_review" in json_data
