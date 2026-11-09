from __future__ import annotations

import json
import pytest

from src.context.models import ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.entity.candidates import (
    CandidateObservation,
    DiscoveredEntityCandidate,
)
from src.researcher.entity.resolver import (
    AffiliationData,
    IdentifierData,
    IdentityAssessmentOutput,
    IdentityResolutionResult,
    IdentityResolver,
)
from src.researcher.models import (
    ConfidenceLevel,
    EntityType,
    ResolutionStatus,
)


@pytest.fixture
def target_prospect() -> ProspectContext:
    return ProspectContext(
        full_name="Sarah Connor",
        company_name="Cyberdyne Systems",
        company_domain="cyberdyne.com",
        designation="VP of Engineering",
        linkedin_url="https://linkedin.com/in/sarahconnor",
        location="Los Angeles, CA",
    )


def test_empty_candidates_returns_unresolved(target_prospect):
    resolver = IdentityResolver()
    result = resolver.resolve(prospect=target_prospect, candidates=[])

    assert isinstance(result, IdentityResolutionResult)
    assert result.resolution_status == ResolutionStatus.UNRESOLVED
    assert result.confidence == ConfidenceLevel.UNRESOLVED
    assert result.matched_candidate_ids == []
    assert result.prospect_entity.resolution_status == ResolutionStatus.UNRESOLVED
    assert result.company_entity.resolution_status == ResolutionStatus.UNRESOLVED


def test_successful_multi_attribute_resolution(target_prospect):
    candidate = DiscoveredEntityCandidate(
        candidate_id="cand-1",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        normalized_name="sarah connor",
        domains=["cyberdyne.com"],
        roles=["VP of Engineering"],
        organizations=["Cyberdyne Systems"],
        linkedin_urls=["https://linkedin.com/in/sarahconnor"],
        locations=["Los Angeles, CA"],
        observations=[
            CandidateObservation(
                source_id="src_1",
                url="https://cyberdyne.com/leadership",
                snippet="Sarah Connor is VP of Engineering at Cyberdyne Systems.",
            )
        ],
    )

    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=["cand-1"],
        prospect_canonical_name="Sarah Connor",
        company_canonical_name="Cyberdyne Systems Inc.",
        prospect_aliases=[],
        company_aliases=["Cyberdyne"],
        affiliations=[
            AffiliationData(
                role="VP of Engineering",
                organization_name="Cyberdyne Systems",
                is_current=True,
                start_date="2022-01-01",
                end_date=None,
            )
        ],
        identifiers=[
            IdentifierData(id_type="linkedin_url", id_value="https://linkedin.com/in/sarahconnor"),
            IdentifierData(id_type="company_domain", id_value="cyberdyne.com"),
        ],
        relationships=[],
        rationale="Exact name, matching domain (cyberdyne.com), matching role, and verified LinkedIn URL.",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    resolver = IdentityResolver(engine=MockEngine())
    result = resolver.resolve(prospect=target_prospect, candidates=[candidate])

    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.confidence == ConfidenceLevel.HIGH
    assert result.matched_candidate_ids == ["cand-1"]
    assert result.prospect_entity.canonical_name == "Sarah Connor"
    assert result.prospect_entity.resolution_status == ResolutionStatus.RESOLVED
    assert len(result.affiliations) == 1
    assert result.affiliations[0].is_current is True
    assert result.affiliations[0].end_date is None


def test_name_only_match_rejected_by_guardrail(target_prospect):
    """Guardrail must demote RESOLVED to UNRESOLVED if only the name matches without corroborating attributes."""
    candidate = DiscoveredEntityCandidate(
        candidate_id="cand-name-only",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        normalized_name="sarah connor",
        domains=["unrelated-news.org"],
        roles=["Author"],
        organizations=["Random Gazette"],
        linkedin_urls=[],
        locations=["London, UK"],
    )

    # LLM mistakenly tries to declare RESOLVED
    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=["cand-name-only"],
        prospect_canonical_name="Sarah Connor",
        rationale="Name matches target prospect.",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    resolver = IdentityResolver(engine=MockEngine())
    result = resolver.resolve(prospect=target_prospect, candidates=[candidate])

    # Guardrail must enforce demotion: name similarity alone cannot resolve
    assert result.resolution_status == ResolutionStatus.UNRESOLVED
    assert result.confidence == ConfidenceLevel.LOW
    assert "Guardrail Enforced" in (result.disambiguation_notes or "")
    assert "Name similarity alone" in (result.disambiguation_notes or "")


def test_multiple_conflicting_candidates_produces_ambiguous(target_prospect):
    """Multiple candidates matching target name with conflicting organizations must resolve to AMBIGUOUS."""
    cand1 = DiscoveredEntityCandidate(
        candidate_id="cand-1",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        normalized_name="sarah connor",
        domains=["cyberdyne.com"],
        organizations=["Cyberdyne Systems"],
        roles=["VP of Engineering"],
    )
    cand2 = DiscoveredEntityCandidate(
        candidate_id="cand-2",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        normalized_name="sarah connor",
        domains=["skynet-defense.org"],
        organizations=["Skynet Defense"],
        roles=["Director of Systems"],
    )

    # LLM tries to pick one or declare resolved
    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.MEDIUM,
        matched_candidate_ids=["cand-1"],
        rationale="Picked cand-1",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    resolver = IdentityResolver(engine=MockEngine())
    result = resolver.resolve(prospect=target_prospect, candidates=[cand1, cand2])

    # Guardrail must force status to AMBIGUOUS
    assert result.resolution_status == ResolutionStatus.AMBIGUOUS
    assert result.confidence == ConfidenceLevel.LOW
    assert "Guardrail Enforced: Multiple conflicting candidates discovered" in (result.disambiguation_notes or "")
    assert any("Multiple conflicting candidates" in c for c in result.conflicts_or_ambiguities)


def test_candidate_at_different_company_produces_partially_resolved(target_prospect):
    """Person candidate confirmed at a different company must resolve to PARTIALLY_RESOLVED."""
    candidate = DiscoveredEntityCandidate(
        candidate_id="cand-diff-company",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        normalized_name="sarah connor",
        domains=["acme-robotics.com"],
        organizations=["Acme Robotics"],
        roles=["VP of Engineering"],
        linkedin_urls=["https://linkedin.com/in/sarahconnor"],
    )

    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=["cand-diff-company"],
        prospect_canonical_name="Sarah Connor",
        affiliations=[
            AffiliationData(
                role="VP of Engineering",
                organization_name="Acme Robotics",
                is_current=True,
                start_date="2024-01-01",
                end_date=None,
            )
        ],
        rationale="Person confirmed via LinkedIn, but now at Acme Robotics instead of Cyberdyne Systems.",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    resolver = IdentityResolver(engine=MockEngine())
    result = resolver.resolve(prospect=target_prospect, candidates=[candidate])

    # Must be PARTIALLY_RESOLVED, not silently overwriting target company as RESOLVED
    assert result.resolution_status == ResolutionStatus.PARTIALLY_RESOLVED
    assert "different organization" in (result.disambiguation_notes or "")
    assert len(result.affiliations) == 1
    assert result.affiliations[0].organization_name == "Acme Robotics"


def test_hallucinated_candidate_id_dropped(target_prospect):
    """Hallucinated candidate IDs returned by the LLM must be rejected."""
    real_candidate = DiscoveredEntityCandidate(
        candidate_id="real-cand-1",
        entity_type=EntityType.PERSON,
        raw_name="Sarah Connor",
        normalized_name="sarah connor",
        domains=["cyberdyne.com"],
        organizations=["Cyberdyne Systems"],
    )

    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=["real-cand-1", "hallucinated-id-999"],
        prospect_canonical_name="Sarah Connor",
        rationale="Matched candidate",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    resolver = IdentityResolver(engine=MockEngine())
    result = resolver.resolve(prospect=target_prospect, candidates=[real_candidate])

    assert "real-cand-1" in result.matched_candidate_ids
    assert "hallucinated-id-999" not in result.matched_candidate_ids
