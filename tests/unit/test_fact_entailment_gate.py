from __future__ import annotations

import uuid
from typing import Any, List, Optional
import pytest

from src.email_generation.claim_verifier.entailment import (
    FactEntailmentGate,
    RawEntailmentOutput,
)
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    ClaimSection,
    DecomposedClaim,
    EntailmentVerdict,
    ModalityLevel,
    VerifierFindingSeverity,
)
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider import ProviderResponse
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    ResearchPackage,
)


class MockEngine(GeneratorEngine):
    def __init__(self, raw_output: Optional[RawEntailmentOutput] = None, raise_error: Optional[Exception] = None):
        self.raw_output = raw_output
        self.raise_error = raise_error
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_count += 1
        if self.raise_error:
            raise self.raise_error
        return GenerationResponse(
            text="{}",
            provider_name="groq",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=self.raw_output or RawEntailmentOutput(verdict="NEUTRAL", rationale="Default mock"),
            metadata={},
        )




def _make_dummy_research_package() -> Tuple[ResearchPackage, CanonicalFact, EvidenceRecord]:
    ev = EvidenceRecord(
        evidence_id="ev-101",
        source_id="src-001",
        verbatim_excerpt="Black Mesa added 10,000 new compute nodes to their primary cluster.",
        confidence=ConfidenceLevel.HIGH,
    )
    fact = CanonicalFact(
        fact_id="fact-201",
        claim_text="Black Mesa added 10,000 compute nodes to their cluster",
        evidence_records=[ev],
    )
    pkg = ResearchPackage(facts=(fact,))
    return pkg, fact, ev


def test_entailment_supported_claim_deterministic():
    pkg, fact, ev = _make_dummy_research_package()
    gate = FactEntailmentGate()  # Deterministic mode

    claim = DecomposedClaim(
        claim_id="c-1",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.ENTAILED
    assert fact.fact_id in results[0].matched_fact_ids
    assert ev.evidence_id in results[0].matched_evidence_ids
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_entailment_negation_inversion_detected():
    # Fact says delayed, claim asserts completed
    ev = EvidenceRecord(
        evidence_id="ev-delayed",
        source_id="src-001",
        verbatim_excerpt="Acme Corp delayed their cloud migration indefinitely.",
    )
    fact = CanonicalFact(
        fact_id="fact-delayed",
        claim_text="Acme Corp delayed cloud migration due to budget cuts",
        evidence_records=[ev],
    )
    pkg = ResearchPackage(facts=(fact,))
    gate = FactEntailmentGate()

    claim = DecomposedClaim(
        claim_id="c-neg",
        text="congratulations on completing your cloud migration at Acme",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.CONTRADICTED
    assert any(f.code == "CONTRADICTED_FACT" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_entailment_unsupported_assertion_produces_error():
    pkg, _, _ = _make_dummy_research_package()
    gate = FactEntailmentGate()

    # Claim about totally unmentioned acquisition
    claim = DecomposedClaim(
        claim_id="c-unsupported",
        text="saw your recent acquisition of Cyberdyne Systems",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=48,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "UNSUPPORTED_PROSPECT_ASSERTION" and f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_entailment_llm_supported_with_valid_ids():
    pkg, fact, ev = _make_dummy_research_package()
    mock_engine = MockEngine(
        raw_output=RawEntailmentOutput(
            verdict="ENTAILED",
            supporting_fact_ids=[fact.fact_id],
            supporting_evidence_ids=[ev.evidence_id],
            confidence=0.95,
            rationale="Verbatim excerpt directly confirms the 10,000 node addition.",
        )
    )
    gate = FactEntailmentGate(engine=mock_engine)

    claim = DecomposedClaim(
        claim_id="c-llm",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.ENTAILED
    assert results[0].matched_fact_ids == [fact.fact_id]
    assert results[0].matched_evidence_ids == [ev.evidence_id]
    assert not any(f.severity == VerifierFindingSeverity.ERROR for f in findings)


def test_entailment_llm_fabricated_ids_rejected_and_downgraded():
    pkg, fact, ev = _make_dummy_research_package()
    # Model returns fabricated IDs that do not exist
    mock_engine = MockEngine(
        raw_output=RawEntailmentOutput(
            verdict="ENTAILED",
            supporting_fact_ids=["fact-fake-999"],
            supporting_evidence_ids=["ev-fake-888"],
            confidence=0.95,
            rationale="Hallucinated evidence reference.",
        )
    )
    gate = FactEntailmentGate(engine=mock_engine)

    claim = DecomposedClaim(
        claim_id="c-fab",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    # Downgraded to NEUTRAL because fabricated IDs were stripped
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "FABRICATED_EVIDENCE_REFERENCE" for f in findings)
    assert any(f.code == "MISSING_EVIDENCE_REFERENCE" for f in findings)


def test_entailment_llm_service_timeout_fails_closed():
    pkg, _, _ = _make_dummy_research_package()
    mock_engine = MockEngine(raise_error=TimeoutError("Connection to LLM provider timed out"))
    gate = FactEntailmentGate(engine=mock_engine)

    claim = DecomposedClaim(
        claim_id="c-timeout",
        text="saw your team added 10,000 new compute nodes at Black Mesa",
        section=ClaimSection.BODY,
        span_start=0,
        span_end=58,
        category=ClaimCategory.PROSPECT_OBSERVATION,
        modality=ModalityLevel.DEFINITIVE,
        is_material=True,
    )

    results, findings = gate.verify([claim], pkg)
    assert len(results) == 1
    assert results[0].verdict == EntailmentVerdict.NEUTRAL
    assert any(f.code == "VERIFIER_SERVICE_UNAVAILABLE" and f.severity == VerifierFindingSeverity.ERROR for f in findings)
