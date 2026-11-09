from __future__ import annotations

import datetime
import json
import pytest

from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.extraction.extractor import (
    FactExtractionOutput,
    FactExtractor,
    RawExtractedEvidence,
    RawExtractedFact,
)
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    FactCategory,
    ResolutionStatus,
    SourceRecord,
)


@pytest.fixture
def sample_known_entities():
    prospect = EntityRecord(
        entity_id="ent-prospect-1",
        entity_type=EntityType.PERSON,
        raw_name="Elena Rostova",
        canonical_name="Elena Rostova",
        resolution_status=ResolutionStatus.RESOLVED,
    )
    company = EntityRecord(
        entity_id="ent-company-1",
        entity_type=EntityType.ORGANIZATION,
        raw_name="Vanguard AI",
        canonical_name="Vanguard AI Inc.",
        resolution_status=ResolutionStatus.RESOLVED,
    )
    return [prospect, company]


def test_atomic_fact_extraction_and_attribution(sample_known_entities):
    raw_text = (
        "Vanguard AI closed a $25M Series A round on 2026-04-10 led by Horizon Capital. "
        "Separately, Elena Rostova was named Head of Research at Vanguard AI."
    )

    source = SourceRecord(
        source_id="src_1",
        url="https://tech-news.org/vanguard-series-a",
        domain="tech-news.org",
        title="Vanguard AI Series A and Leadership",
        raw_text=raw_text,
        fetch_date=datetime.date(2026, 11, 7),
        credibility_score=0.9,
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI closed a $25M Series A funding round led by Horizon Capital.",
                fact_category=FactCategory.FUNDING,
                subject_entity_name="Vanguard AI",
                predicate="closed",
                object_value="$25M Series A",
                primary_event_date="2026-04-10",
                confidence=ConfidenceLevel.HIGH,
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI closed a $25M Series A round on 2026-04-10 led by Horizon Capital.",
                        event_date="2026-04-10",
                    )
                ],
            ),
            RawExtractedFact(
                claim_text="Elena Rostova was appointed Head of Research at Vanguard AI.",
                fact_category=FactCategory.HIRING,
                subject_entity_name="Elena Rostova",
                predicate="named",
                object_value="Head of Research",
                confidence=ConfidenceLevel.HIGH,
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Elena Rostova was named Head of Research at Vanguard AI.",
                    )
                ],
            ),
        ]
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_json = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_json),
                parsed_output=llm_output,
            )

    extractor = FactExtractor(engine=MockEngine())
    facts = extractor.extract(
        source=source,
        known_entities=sample_known_entities,
        entity_context={"Elena Rostova": "ent-prospect-1", "Vanguard AI": "ent-company-1"},
    )

    assert len(facts) == 2

    # Check fact 1 (Funding)
    funding_fact = next(f for f in facts if f.fact_category == FactCategory.FUNDING)
    assert "25M" in funding_fact.claim_text
    assert funding_fact.primary_event_date == datetime.date(2026, 4, 10)
    assert "ent-company-1" in funding_fact.entity_ids
    assert len(funding_fact.evidence_records) == 1
    assert funding_fact.evidence_records[0].source_id == "src_1"
    assert funding_fact.evidence_records[0].credibility_score == 0.9

    # Check fact 2 (Hiring / Role)
    hiring_fact = next(f for f in facts if f.fact_category == FactCategory.HIRING)
    assert "Elena Rostova" in hiring_fact.claim_text
    assert "ent-prospect-1" in hiring_fact.entity_ids
    assert len(hiring_fact.evidence_records) == 1


def test_unsupported_and_hallucinated_excerpt_rejected(sample_known_entities):
    """If an LLM invents a claim or hallucinates an excerpt not in raw_text, it must be discarded."""
    raw_text = "Vanguard AI released its annual open source compiler toolchain."

    source = SourceRecord(
        source_id="src_1",
        url="https://vanguardai.io/blog",
        domain="vanguardai.io",
        raw_text=raw_text,
        fetch_date=datetime.date(2026, 11, 7),
    )

    llm_output = FactExtractionOutput(
        facts=[
            # Fact with real excerpt
            RawExtractedFact(
                claim_text="Vanguard AI released an open source compiler toolchain.",
                fact_category=FactCategory.PRODUCT_LAUNCH,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI released its annual open source compiler toolchain.",
                    )
                ],
            ),
            # Fact with completely fabricated excerpt
            RawExtractedFact(
                claim_text="Vanguard AI is currently planning an initial public offering in London.",
                fact_category=FactCategory.STRATEGY,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI is currently planning an initial public offering in London.",
                    )
                ],
            ),
        ]
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_json = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_json,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_json),
                parsed_output=llm_output,
            )

    extractor = FactExtractor(engine=MockEngine())
    facts = extractor.extract(source=source, known_entities=sample_known_entities)

    # Only the first fact with a genuine verbatim excerpt should survive!
    assert len(facts) == 1
    assert facts[0].fact_category == FactCategory.PRODUCT_LAUNCH
    assert "compiler toolchain" in facts[0].claim_text


def test_empty_or_unusable_source_text():
    source = SourceRecord(
        source_id="src_empty",
        url="https://empty.org",
        domain="empty.org",
        raw_text="   ",
        fetch_date=datetime.date(2026, 11, 7),
    )

    extractor = FactExtractor()
    facts = extractor.extract(source=source)
    assert facts == []
