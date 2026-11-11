from __future__ import annotations

import datetime
from unittest.mock import MagicMock
import pytest

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.engine.response import GenerationResponse
from src.researcher.entity.candidates import DiscoveredEntityCandidate
from src.researcher.entity.resolver import (
    IdentityAssessmentOutput,
    IdentityResolver,
)
from src.researcher.extraction.extractor import (
    FactExtractionOutput,
    FactExtractor,
    RawExtractedEvidence,
    RawExtractedFact,
)
from src.researcher.models import (
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    FactCategory,
    ResolutionStatus,
)
from src.signal_strategy.identifier import (
    RawCandidateSignal,
    RawSignalIdentificationOutput,
    SignalIdentifier,
)
from src.signal_strategy.models import CandidateSignal, RankedSignal, SignalScore
from src.signal_strategy.ranker import (
    RawEvaluatedSignal,
    RawSignalEvaluationOutput,
    SignalRanker,
)
from src.signal_strategy.strategy_builder import (
    RawStrategyOutput,
    StrategyBuilder,
)


def test_provenance_model_and_origins():
    prov = OutputProvenance(
        origin=ProvenanceOrigin.MODEL,
        component="test_component",
        model_name="test-model",
        provider="groq",
        model_execution_id="exec-12345",
    )
    assert prov.origin == ProvenanceOrigin.MODEL
    assert prov.component == "test_component"
    assert prov.model_execution_id == "exec-12345"
    assert not prov.is_fallback


def test_entity_resolver_attaches_provenance():
    mock_engine = MagicMock()
    mock_response = GenerationResponse(
        text='{"resolution_status": "resolved"}',
        provider_name="groq",
        raw_response=None,
        parsed_output=IdentityAssessmentOutput(
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
            matched_candidate_ids=["cand_1"],
            rationale="Exact match with verified domain.",
        ),
        metadata={"execution_id": "exec_res_99"},
    )
    mock_engine.generate.return_value = mock_response

    resolver = IdentityResolver(engine=mock_engine, provider_name="groq")
    prospect = ProspectContext(
        full_name="Jane Doe",
        company_name="Acme Corp",
        company_domain="acme.com",
    )
    candidate = DiscoveredEntityCandidate(
        candidate_id="cand_1",
        entity_type=EntityType.PERSON,
        raw_name="Jane Doe",
        normalized_name="jane doe",
        domains=["acme.com"],
        organizations=["Acme Corp"],
    )

    # 1. With candidate matching: model provenance attached
    res = resolver.resolve(prospect, [candidate])
    assert res.prospect_entity.provenance is not None
    assert res.prospect_entity.provenance.origin == ProvenanceOrigin.MODEL
    assert res.prospect_entity.provenance.model_execution_id == "exec_res_99"

    # 2. Fast path with zero candidates: deterministic rule fallback provenance attached
    empty_res = resolver.resolve(prospect, [])
    assert empty_res.prospect_entity.provenance is not None
    assert empty_res.prospect_entity.provenance.origin == ProvenanceOrigin.DETERMINISTIC_RULE
    assert empty_res.prospect_entity.provenance.is_fallback is True
    assert empty_res.prospect_entity.provenance.rule_id == "empty_candidates_fallback"


def test_fact_extractor_attaches_provenance():
    mock_engine = MagicMock()
    raw_fact = RawExtractedFact(
        claim_text="Acme Corp closed Series B funding.",
        fact_category=FactCategory.FUNDING,
        subject_entity_name="Acme Corp",
        evidence=[
            RawExtractedEvidence(
                verbatim_excerpt="Acme Corp closed Series B funding",
                event_date="2026-02-01",
            )
        ],
    )
    mock_response = GenerationResponse(
        text="{}",
        provider_name="groq",
        raw_response=None,
        parsed_output=FactExtractionOutput(facts=[raw_fact]),
        metadata={"execution_id": "exec_fact_88"},
    )
    mock_engine.generate.return_value = mock_response

    extractor = FactExtractor(engine=mock_engine, provider_name="groq")
    facts = extractor.extract({
        "source_id": "src-1",
        "title": "Funding News",
        "raw_text": "Acme Corp closed Series B funding in February 2026.",
        "credibility_score": 0.85,
    })

    assert len(facts) == 1
    assert facts[0].provenance is not None
    assert facts[0].provenance.origin == ProvenanceOrigin.MODEL
    assert facts[0].provenance.model_execution_id == "exec_fact_88"


def test_signal_identifier_provenance_model_and_fallback():
    mock_engine = MagicMock()
    raw_sig = RawCandidateSignal(
        label="Funding Growth",
        description="Series B funding event",
        supporting_fact_ids=["fact-1"],
        fact_category="funding",
    )
    mock_response = GenerationResponse(
        text="{}",
        provider_name="groq",
        raw_response=None,
        parsed_output=RawSignalIdentificationOutput(candidate_signals=[raw_sig]),
        metadata={"execution_id": "exec_sig_77"},
    )
    mock_engine.generate.return_value = mock_response

    identifier = SignalIdentifier(engine=mock_engine, provider_name="groq")
    prospect = ProspectContext(full_name="Jane Doe", company_name="Acme Corp")
    product = ProductContext(
        product_name="ProdX",
        description="Enterprise automation tool",
        capabilities=["Cap A"],
        value_proposition="Accelerates workflows",
    )
    fact = MagicMock()
    fact.fact_id = "fact-1"
    fact.fact_category = FactCategory.FUNDING
    fact.entity_ids = []
    fact.primary_event_date = datetime.date(2026, 2, 1)
    fact.claim_text = "Acme Corp closed Series B funding."

    # Model path
    signals = identifier.identify_signals([fact], prospect, product)
    assert len(signals) == 1
    assert signals[0].provenance is not None
    assert signals[0].provenance.origin == ProvenanceOrigin.MODEL
    assert signals[0].provenance.model_execution_id == "exec_sig_77"

    # On LLM error, zero synthetic signals are fabricated (Phase 1 FL-11)
    mock_engine.generate.side_effect = RuntimeError("LLM failure")
    fallback_signals = identifier.identify_signals([fact], prospect, product)
    assert len(fallback_signals) == 0


def test_signal_ranker_provenance_model_and_fallback():
    mock_engine = MagicMock()
    cand_sig = CandidateSignal(
        label="Funding Growth",
        description="Series B funding event for Acme Corp",
        target_capability="Cap A",
        supporting_fact_ids=["fact-1"],
    )
    raw_eval = RawEvaluatedSignal(
        signal_id=cand_sig.signal_id,
        prospect_relevance=0.8,
        product_alignment=0.75,
        business_significance=0.9,
        rationale="Strong fit",
    )
    mock_response = GenerationResponse(
        text="{}",
        provider_name="groq",
        raw_response=None,
        parsed_output=RawSignalEvaluationOutput(evaluations=[raw_eval]),
        metadata={"execution_id": "exec_rank_66"},
    )
    mock_engine.generate.return_value = mock_response

    ranker = SignalRanker(engine=mock_engine, provider_name="groq")
    prospect = ProspectContext(full_name="Jane Doe", company_name="Acme Corp")
    product = ProductContext(
        product_name="ProdX",
        description="Enterprise automation tool",
        capabilities=["Cap A"],
        value_proposition="Accelerates workflows",
    )
    fact = MagicMock()
    fact.fact_id = "fact-1"
    fact.fact_category = FactCategory.FUNDING
    fact.entity_ids = []
    fact.primary_event_date = datetime.date(2026, 2, 1)

    # 1. Model ranking path
    ranked = ranker.rank_signals([cand_sig], [fact], prospect, product)
    assert len(ranked) == 1
    assert ranked[0].provenance is not None
    assert ranked[0].provenance.origin == ProvenanceOrigin.MODEL
    assert ranked[0].provenance.model_execution_id == "exec_rank_66"
    assert ranked[0].score.provenance is not None
    assert ranked[0].score.provenance.origin == ProvenanceOrigin.MODEL

    # 2. Heuristic fallback when LLM fails
    mock_engine.generate.side_effect = RuntimeError("Ranking LLM error")
    fallback_ranked = ranker.rank_signals([cand_sig], [fact], prospect, product)
    assert len(fallback_ranked) == 1
    assert fallback_ranked[0].provenance is not None
    assert fallback_ranked[0].provenance.origin == ProvenanceOrigin.HEURISTIC_FALLBACK
    assert fallback_ranked[0].provenance.is_fallback is True
    assert fallback_ranked[0].score.provenance.origin == ProvenanceOrigin.HEURISTIC_FALLBACK


def test_strategy_builder_provenance():
    mock_engine = MagicMock()
    raw_strat = RawStrategyOutput(
        prospect_role_relevance="Direct relevance to role",
        business_implications="Operational scaling pressure",
        capability_mapping=["Cap A"],
        messaging_approach="value-led",
        central_message="ProdX solves operational scaling",
        suggested_subject_line="Acme scaling",
        cta="Open for a quick chat?",
        tone_guidance="Direct",
    )
    mock_response = GenerationResponse(
        text="{}",
        provider_name="groq",
        raw_response=None,
        parsed_output=raw_strat,
        metadata={"execution_id": "exec_strat_55"},
    )
    mock_engine.generate.return_value = mock_response

    builder = StrategyBuilder(engine=mock_engine, provider_name="groq")
    prospect = ProspectContext(full_name="Jane Doe", company_name="Acme Corp")
    product = ProductContext(
        product_name="ProdX",
        description="Enterprise automation tool",
        capabilities=["Cap A"],
        value_proposition="Accelerates workflows",
    )
    outreach = OutreachContext(
        objective="Schedule demo",
        sender_identity="Jane Sales, ProdX AE",
    )

    cand = CandidateSignal(label="Expansion", description="Acme is expanding", supporting_fact_ids=["f1"])
    score = SignalScore(
        evidence_strength=0.9,
        recency_score=0.8,
        prospect_relevance=0.85,
        product_alignment=0.8,
        business_significance=0.8,
        composite_score=0.82,
    )
    ranked = RankedSignal(signal=cand, score=score, rationale="High fit")

    # Viable strategy with model
    res = builder.build_strategy([ranked], prospect, product, outreach)
    assert res.is_viable is True
    assert res.provenance is not None
    assert res.provenance.origin == ProvenanceOrigin.DETERMINISTIC_RULE
    assert res.provenance.rule_id == "strategy_viability_confirmed"
    assert res.strategy is not None
    assert res.strategy.provenance is not None
    assert res.strategy.provenance.origin == ProvenanceOrigin.MODEL
    assert res.strategy.provenance.model_execution_id == "exec_strat_55"

    # Unviable check
    score_unviable = SignalScore(
        evidence_strength=0.9,
        recency_score=0.8,
        prospect_relevance=0.85,
        product_alignment=None,  # unevaluated!
        business_significance=0.8,
        composite_score=0.20,
    )
    ranked_unviable = RankedSignal(signal=cand, score=score_unviable, rationale="Unevaluated")
    unviable_res = builder.build_strategy([ranked_unviable], prospect, product, outreach)
    assert unviable_res.is_viable is False
    assert unviable_res.provenance is not None
    assert unviable_res.provenance.origin == ProvenanceOrigin.DETERMINISTIC_RULE
    assert unviable_res.provenance.rule_id == "strategy_viability_check"
