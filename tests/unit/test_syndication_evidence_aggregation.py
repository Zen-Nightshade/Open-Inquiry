from __future__ import annotations

import datetime
import pytest

from src.researcher.extraction.assessment import EvidenceAssessor
from src.researcher.extraction.consolidator import FactConsolidator
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    SourceRecord,
    SourceType,
    VerificationStatus,
)


def _make_evidence(
    source_id: str,
    excerpt: str = "Acme Corp raised $50M in Series B funding.",
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    credibility: float = 0.7,
) -> EvidenceRecord:
    return EvidenceRecord(
        source_id=source_id,
        verbatim_excerpt=excerpt,
        confidence=confidence,
        credibility_score=credibility,
        event_date=datetime.date(2026, 3, 15),
    )


def test_five_syndicated_copies_do_not_inflate_confidence_to_high():
    """Five syndicated copies from one press release must be grouped into one cluster and NOT upgrade confidence to HIGH."""
    accumulator = ResearchAccumulator()

    # Define 5 sources originating from the same wire release
    syndication_groups = {
        "wire_cluster_1": {"src_wire", "src_reprint1", "src_reprint2", "src_reprint3", "src_reprint4"}
    }

    # Initial fact from the primary wire
    fact_1 = CanonicalFact(
        claim_text="Acme Corp raised $50M in Series B funding",
        fact_category=FactCategory.FUNDING,
        evidence_records=[_make_evidence("src_wire", confidence=ConfidenceLevel.MEDIUM)],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )
    accumulator.merge_or_add_fact(fact_1, syndication_groups=syndication_groups)

    # Merge 4 subsequent syndicated copies
    for i in range(1, 5):
        sid = f"src_reprint{i}"
        next_fact = CanonicalFact(
            claim_text="Acme Corp raised $50M in Series B funding",
            fact_category=FactCategory.FUNDING,
            evidence_records=[_make_evidence(sid, confidence=ConfidenceLevel.MEDIUM)],
            composite_confidence=ConfidenceLevel.MEDIUM,
        )
        accumulator.merge_or_add_fact(
            next_fact,
            match_fact_id=accumulator.facts[0].fact_id,
            is_independent=False,  # Detected as non-independent / syndicated
            syndication_groups=syndication_groups,
        )

    assert len(accumulator.facts) == 1
    consolidated = accumulator.facts[0]

    # Verify all 5 evidence records and source IDs are preserved
    assert len(consolidated.evidence_records) == 5
    assert len(consolidated.source_ids) == 5

    # Verification: independent cluster count must be exactly 1
    assert consolidated.independent_source_count == 1

    # Invariant: composite confidence must NOT have upgraded to HIGH
    assert consolidated.composite_confidence == ConfidenceLevel.MEDIUM

    # Verify evidence aggregation summary
    summary = consolidated.get_evidence_aggregation_summary(syndication_groups=syndication_groups)
    assert summary["total_evidence_records"] == 5
    assert summary["total_source_records"] == 5
    assert summary["independent_cluster_count"] == 1
    assert summary["is_corroborated_independently"] is False


def test_two_independent_sources_upgrade_confidence_to_high():
    """Two genuinely independent sources reporting the fact DO count as independent corroboration and upgrade confidence."""
    accumulator = ResearchAccumulator()

    syndication_groups = {
        "cluster_wire": {"src_wire_1", "src_wire_2"},
        "cluster_independent": {"src_sec_filing"},
    }

    fact_wire = CanonicalFact(
        claim_text="Acme Corp raised $50M in Series B funding",
        fact_category=FactCategory.FUNDING,
        evidence_records=[_make_evidence("src_wire_1", confidence=ConfidenceLevel.MEDIUM)],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )
    accumulator.merge_or_add_fact(fact_wire, syndication_groups=syndication_groups)

    # Genuinely independent regulatory filing
    fact_sec = CanonicalFact(
        claim_text="Acme Corp raised $50M in Series B funding",
        fact_category=FactCategory.FUNDING,
        evidence_records=[_make_evidence("src_sec_filing", confidence=ConfidenceLevel.MEDIUM)],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )
    accumulator.merge_or_add_fact(
        fact_sec,
        match_fact_id=accumulator.facts[0].fact_id,
        is_independent=True,
        syndication_groups=syndication_groups,
    )

    consolidated = accumulator.facts[0]
    assert len(consolidated.evidence_records) == 2
    assert consolidated.independent_source_count == 2
    assert consolidated.composite_confidence == ConfidenceLevel.HIGH

    summary = consolidated.get_evidence_aggregation_summary(syndication_groups=syndication_groups)
    assert summary["independent_cluster_count"] == 2
    assert summary["is_corroborated_independently"] is True


def test_source_credibility_distinct_from_independence():
    """Low credibility sources cannot achieve HIGH confidence even with multiple syndicated copies."""
    accumulator = ResearchAccumulator()

    syndication_groups = {
        "spam_ring": {"spam_1", "spam_2", "spam_3"}
    }

    # Low credibility evidence
    fact_spam = CanonicalFact(
        claim_text="Acme Corp opening office in Tokyo",
        fact_category=FactCategory.OTHER,
        evidence_records=[_make_evidence("spam_1", confidence=ConfidenceLevel.LOW, credibility=0.2)],
        composite_confidence=ConfidenceLevel.LOW,
    )
    accumulator.merge_or_add_fact(fact_spam, syndication_groups=syndication_groups)

    for sid in ["spam_2", "spam_3"]:
        next_fact = CanonicalFact(
            claim_text="Acme Corp opening office in Tokyo",
            fact_category=FactCategory.OTHER,
            evidence_records=[_make_evidence(sid, confidence=ConfidenceLevel.LOW, credibility=0.2)],
            composite_confidence=ConfidenceLevel.LOW,
        )
        accumulator.merge_or_add_fact(
            next_fact,
            match_fact_id=accumulator.facts[0].fact_id,
            is_independent=False,
            syndication_groups=syndication_groups,
        )

    consolidated = accumulator.facts[0]
    assert consolidated.composite_confidence == ConfidenceLevel.LOW
    assert consolidated.independent_source_count == 1


def test_consolidator_integration_with_syndication_groups():
    """FactConsolidator should evaluate independence using syndication_groups without dropping sources."""
    consolidator = FactConsolidator()
    accumulator = ResearchAccumulator()

    syndication_groups = {
        "cluster_pr": {"pr_wire", "yahoo_finance", "benzinga"}
    }
    source_domains = {
        "pr_wire": "prnewswire.com",
        "yahoo_finance": "finance.yahoo.com",
        "benzinga": "benzinga.com",
        "independent_tech": "techcrunch.com",
    }

    # First fact
    fact_1 = CanonicalFact(
        claim_text="Acme Corp appoints Dr. Aris Thorne as CTO",
        fact_category=FactCategory.HIRING,
        entity_ids=["entity_acme"],
        evidence_records=[_make_evidence("pr_wire", excerpt="Acme Corp appoints Dr. Aris Thorne as CTO")],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )
    # Second fact from syndication copy
    fact_2 = CanonicalFact(
        claim_text="Acme Corp appoints Dr. Aris Thorne as CTO",
        fact_category=FactCategory.HIRING,
        entity_ids=["entity_acme"],
        evidence_records=[_make_evidence("yahoo_finance", excerpt="Acme Corp appoints Dr. Aris Thorne as CTO")],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )
    # Third fact from independent journalist
    fact_3 = CanonicalFact(
        claim_text="Acme Corp appoints Dr. Aris Thorne as CTO",
        fact_category=FactCategory.HIRING,
        entity_ids=["entity_acme"],
        evidence_records=[_make_evidence("independent_tech", excerpt="Acme Corp announced Dr. Aris Thorne as their new Chief Technology Officer.")],
        composite_confidence=ConfidenceLevel.MEDIUM,
    )

    res = consolidator.consolidate(
        [fact_1, fact_2, fact_3],
        accumulator,
        syndication_groups=syndication_groups,
        source_domains=source_domains,
    )

    assert len(res.consolidated_facts) == 1
    fact = res.consolidated_facts[0]

    # All 3 evidence records preserved
    assert len(fact.evidence_records) == 3
    # Two independent clusters: cluster_pr and independent_tech
    assert fact.independent_source_count == 2
    # Upgraded to HIGH because independent corroboration with substantial evidence exists
    assert fact.composite_confidence == ConfidenceLevel.HIGH
