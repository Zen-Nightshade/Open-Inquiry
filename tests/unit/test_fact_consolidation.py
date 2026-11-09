import datetime
import pytest

from src.researcher.extraction.assessment import EvidenceAssessor
from src.researcher.extraction.consolidator import (
    ConflictSeverity,
    DetectedConflict,
    FactConsolidator,
    extract_numerical_tokens,
)
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
)


@pytest.fixture
def consolidator():
    return FactConsolidator(assessor=EvidenceAssessor())


def test_numerical_token_extraction():
    assert extract_numerical_tokens("Vanguard AI raised $50 million in Series B") == {"$50 million"}
    assert extract_numerical_tokens("Growth of 25% across 3 regions") == {"25%", "3"}
    assert extract_numerical_tokens("No metrics here") == set()


def test_merge_same_fact_independent_sources_upgrades_confidence(consolidator):
    """When the same fact is corroborated by independent sources, confidence upgrades to HIGH."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_1",
        claim_text="Vanguard AI raised $50 million in Series B funding.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_techcrunch",
                verbatim_excerpt="Vanguard AI raised $50 million in Series B funding.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    fact2 = CanonicalFact(
        fact_id="fact_2",
        claim_text="Vanguard AI announced a $50 million Series B round.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_reuters",
                verbatim_excerpt="Vanguard AI announced a $50 million Series B round led by Sequoia.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    source_domains = {
        "src_techcrunch": "techcrunch.com",
        "src_reuters": "reuters.com",
    }

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
        source_domains=source_domains,
    )

    assert len(result.consolidated_facts) == 1
    consolidated = result.consolidated_facts[0]
    assert consolidated.fact_id == "fact_1"
    # Independent confirmation upgrades confidence to HIGH
    assert consolidated.composite_confidence == ConfidenceLevel.HIGH
    # Both evidence records must be preserved
    assert len(consolidated.evidence_records) == 2
    ev_ids = {ev.evidence_id for ev in consolidated.evidence_records}
    assert ev_ids == {"ev_1", "ev_2"}
    assert len(result.conflicts) == 0


def test_merge_same_fact_syndicated_sources_preserves_confidence(consolidator):
    """When a fact appears across syndicated copies of the same wire release, confidence is NOT inflated."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_1",
        claim_text="Vanguard AI launched its new distributed training engine.",
        fact_category=FactCategory.PRODUCT_LAUNCH,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_pr_newswire",
                verbatim_excerpt="Vanguard AI launched its new distributed training engine.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    fact2 = CanonicalFact(
        fact_id="fact_2",
        claim_text="Vanguard AI launched its new distributed training engine.",
        fact_category=FactCategory.PRODUCT_LAUNCH,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_syndicated_repost",
                verbatim_excerpt="Vanguard AI launched its new distributed training engine.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    # Both sources belong to the same syndication origin group
    syndication_groups = {
        "wire_group_42": {"src_pr_newswire", "src_syndicated_repost"}
    }

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
        syndication_groups=syndication_groups,
    )

    assert len(result.consolidated_facts) == 1
    consolidated = result.consolidated_facts[0]
    # Deduplicated into 1 fact, retaining all source evidence records
    assert len(consolidated.evidence_records) == 2
    # Confidence must NOT inflate to HIGH because sources are syndicated
    assert consolidated.composite_confidence == ConfidenceLevel.MEDIUM
    assert len(result.conflicts) == 0


def test_repeated_evidence_same_source_does_not_inflate(consolidator):
    """Multiple mentions from the exact same source ID do not increase confidence."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_1",
        claim_text="Vanguard AI opened an office in Munich.",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_blog",
                verbatim_excerpt="In January, Vanguard AI opened an office in Munich.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    fact2 = CanonicalFact(
        fact_id="fact_2",
        claim_text="Vanguard AI opened an office in Munich.",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_blog",  # Same source ID!
                verbatim_excerpt="The Munich office of Vanguard AI will host 50 engineers.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
    )

    assert len(result.consolidated_facts) == 1
    consolidated = result.consolidated_facts[0]
    assert len(consolidated.evidence_records) == 2
    # Same source repeated -> confidence stays MEDIUM
    assert consolidated.composite_confidence == ConfidenceLevel.MEDIUM


def test_distinct_facts_remain_separate(consolidator):
    """Facts with different categories, temporal dates, or predicates must remain distinct."""
    accumulator = ResearchAccumulator()

    # Fact A: Funding
    fact_funding = CanonicalFact(
        fact_id="fact_fund",
        claim_text="Vanguard AI raised $50 million in Series B.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        primary_event_date=datetime.date(2025, 6, 1),
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_fund",
                source_id="src_1",
                verbatim_excerpt="Vanguard AI raised $50 million in Series B.",
            )
        ],
    )

    # Fact B: Hiring (different category)
    fact_hiring = CanonicalFact(
        fact_id="fact_hire",
        claim_text="Vanguard AI hired 100 research engineers in 2026.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_vanguard"],
        primary_event_date=datetime.date(2026, 1, 1),
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_hire",
                source_id="src_2",
                verbatim_excerpt="Vanguard AI hired 100 research engineers.",
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact_funding, fact_hiring],
        accumulator=accumulator,
    )

    assert len(result.consolidated_facts) == 2
    assert {f.fact_id for f in result.consolidated_facts} == {"fact_fund", "fact_hire"}
    assert len(result.conflicts) == 0


def test_numerical_conflict_detected_and_preserved(consolidator):
    """When two facts state conflicting amounts for the same round/event, both are kept and a conflict is flagged."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_50m",
        claim_text="Vanguard AI closed a $50 million Series B round.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_1",
                verbatim_excerpt="Vanguard AI closed a $50 million Series B round.",
            )
        ],
    )

    fact2 = CanonicalFact(
        fact_id="fact_30m",
        claim_text="Vanguard AI raised a $30 million Series B round.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_2",
                verbatim_excerpt="Vanguard AI raised a $30 million Series B round.",
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
    )

    # Both facts preserved separately (no destructive overwrite)
    assert len(result.consolidated_facts) == 2
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.fact_id_a == "fact_50m"
    assert conflict.fact_id_b == "fact_30m"
    assert "conflicting metrics for series b" in conflict.description.lower()


def test_employment_conflict_detected_and_preserved(consolidator):
    """When two facts state concurrent contradictory employment statuses, a confirmed conflict is detected."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_joined",
        claim_text="Sarah Connor currently serves as VP of Engineering at Vanguard AI.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_1",
                verbatim_excerpt="Sarah Connor currently serves as VP of Engineering at Vanguard AI.",
            )
        ],
    )

    fact2 = CanonicalFact(
        fact_id="fact_resigned",
        claim_text="Sarah Connor resigned as VP of Engineering.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_2",
                verbatim_excerpt="Sarah Connor resigned as VP of Engineering.",
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
    )

    assert len(result.consolidated_facts) == 2
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.severity == ConflictSeverity.CONFIRMED
    assert "[confirmed]" in conflict.description.lower()
    assert "contradictory employment status" in conflict.description.lower()


def test_employment_missing_dates_detected_as_potential_conflict(consolidator):
    """When two employment milestones lack dates and neither asserts current status, it is a POTENTIAL conflict."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_joined_undated",
        claim_text="Sarah Connor joined Vanguard AI as VP of Engineering.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_1",
                verbatim_excerpt="Sarah Connor joined Vanguard AI as VP of Engineering.",
            )
        ],
    )

    fact2 = CanonicalFact(
        fact_id="fact_resigned_undated",
        claim_text="Sarah Connor resigned as VP of Engineering.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_2",
                verbatim_excerpt="Sarah Connor resigned as VP of Engineering.",
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
    )

    assert len(result.consolidated_facts) == 2
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.severity == ConflictSeverity.POTENTIAL
    assert "[potential]" in conflict.description.lower()
    assert "potential conflict (missing dates)" in conflict.description.lower()


def test_employment_identical_dates_detected_as_confirmed_conflict(consolidator):
    """When two employment milestones have the exact same event date, they represent a CONFIRMED conflict."""
    accumulator = ResearchAccumulator()
    event_date = datetime.date(2024, 6, 1)

    fact1 = CanonicalFact(
        fact_id="fact_j_same",
        claim_text="Sarah Connor joined Vanguard AI on June 1, 2024.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        primary_event_date=event_date,
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_1",
                source_id="src_1",
                verbatim_excerpt="Sarah Connor joined Vanguard AI on June 1, 2024.",
                event_date=event_date,
            )
        ],
    )

    fact2 = CanonicalFact(
        fact_id="fact_l_same",
        claim_text="Sarah Connor stepped down from Vanguard AI on June 1, 2024.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        primary_event_date=event_date,
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_2",
                source_id="src_2",
                verbatim_excerpt="Sarah Connor stepped down from Vanguard AI on June 1, 2024.",
                event_date=event_date,
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
    )

    assert len(result.consolidated_facts) == 2
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.severity == ConflictSeverity.CONFIRMED
    assert "[confirmed]" in conflict.description.lower()
    assert "contradictory employment status" in conflict.description.lower()


def test_multiple_low_quality_sources_cannot_upgrade_to_high(consolidator):
    """Corroboration among low-quality/low-confidence sources cannot establish HIGH confidence."""
    accumulator = ResearchAccumulator()

    fact1 = CanonicalFact(
        fact_id="fact_low1",
        claim_text="Vanguard AI is opening an office in Tokyo.",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_l1",
                source_id="src_forum_rumor",
                verbatim_excerpt="Rumors on Reddit claim Vanguard AI is opening an office in Tokyo.",
                confidence=ConfidenceLevel.LOW,
                credibility_score=0.3,
            )
        ],
        composite_confidence=ConfidenceLevel.LOW,
    )

    fact2 = CanonicalFact(
        fact_id="fact_low2",
        claim_text="Vanguard AI is opening an office in Tokyo.",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_l2",
                source_id="src_scraper_blog",
                verbatim_excerpt="Scraper blog reports Vanguard AI is opening an office in Tokyo.",
                confidence=ConfidenceLevel.LOW,
                credibility_score=0.4,
            )
        ],
        composite_confidence=ConfidenceLevel.LOW,
    )

    domains = {"src_forum_rumor": "forum.net", "src_scraper_blog": "blogscraper.io"}
    result = consolidator.consolidate(
        new_facts=[fact1, fact2],
        accumulator=accumulator,
        source_domains=domains,
    )

    assert len(result.consolidated_facts) == 1
    consolidated = result.consolidated_facts[0]
    # Corroboration of low-quality sources upgrades LOW to MEDIUM, but CANNOT reach HIGH
    assert consolidated.composite_confidence == ConfidenceLevel.MEDIUM
    assert consolidated.composite_confidence != ConfidenceLevel.HIGH


def test_sequential_employment_milestones_not_conflicting(consolidator):
    """Joining in 2021 and leaving in 2024 are sequential career milestones, not a contradiction."""
    accumulator = ResearchAccumulator()

    fact_joined = CanonicalFact(
        fact_id="fact_j",
        claim_text="Sarah Connor joined Vanguard AI in 2021.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        primary_event_date=datetime.date(2021, 5, 1),
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_j",
                source_id="src_1",
                verbatim_excerpt="Sarah Connor joined Vanguard AI in 2021 as Lead Engineer.",
                event_date=datetime.date(2021, 5, 1),
            )
        ],
    )

    fact_left = CanonicalFact(
        fact_id="fact_l",
        claim_text="Sarah Connor resigned from Vanguard AI in 2024.",
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_sarah"],
        primary_event_date=datetime.date(2024, 6, 1),
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_l",
                source_id="src_2",
                verbatim_excerpt="Sarah Connor resigned from Vanguard AI in 2024 to launch a startup.",
                event_date=datetime.date(2024, 6, 1),
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact_joined, fact_left],
        accumulator=accumulator,
    )

    # Both facts preserved, zero conflict reported
    assert len(result.consolidated_facts) == 2
    assert len(result.conflicts) == 0


def test_compatible_funding_scopes_total_vs_round_not_conflicting(consolidator):
    """Total funding to date ($50M) and Series B round size ($30M) have compatible scopes and are not in conflict."""
    accumulator = ResearchAccumulator()

    fact_total = CanonicalFact(
        fact_id="fact_tot",
        claim_text="Vanguard AI raised $50 million total funding to date.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_tot",
                source_id="src_1",
                verbatim_excerpt="Vanguard AI raised $50 million total funding to date across three rounds.",
            )
        ],
    )

    fact_round = CanonicalFact(
        fact_id="fact_round",
        claim_text="Vanguard AI raised a $30 million Series B round.",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_round",
                source_id="src_2",
                verbatim_excerpt="Vanguard AI raised a $30 million Series B round led by Sequoia.",
            )
        ],
    )

    result = consolidator.consolidate(
        new_facts=[fact_total, fact_round],
        accumulator=accumulator,
    )

    # Both facts preserved, zero conflict reported because scope is compatible
    assert len(result.consolidated_facts) == 2
    assert len(result.conflicts) == 0


def test_idempotent_repeated_processing_does_not_inflate(consolidator):
    """Processing identical facts and evidence records repeatedly does not add duplicate records or inflate confidence."""
    accumulator = ResearchAccumulator()

    fact = CanonicalFact(
        fact_id="fact_idempotent",
        claim_text="Vanguard AI launched its neural compiler.",
        fact_category=FactCategory.PRODUCT_LAUNCH,
        entity_ids=["ent_vanguard"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_stable_id",
                source_id="src_1",
                verbatim_excerpt="Vanguard AI launched its neural compiler.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    # First run
    result1 = consolidator.consolidate(new_facts=[fact], accumulator=accumulator)
    assert len(result1.consolidated_facts) == 1
    assert len(result1.consolidated_facts[0].evidence_records) == 1
    assert result1.consolidated_facts[0].composite_confidence == ConfidenceLevel.MEDIUM

    # Second run with identical fact and evidence
    result2 = consolidator.consolidate(new_facts=[fact], accumulator=accumulator)
    assert len(result2.consolidated_facts) == 1
    # Evidence must NOT be duplicated
    assert len(result2.consolidated_facts[0].evidence_records) == 1
    # Confidence must NOT inflate
    assert result2.consolidated_facts[0].composite_confidence == ConfidenceLevel.MEDIUM

