from __future__ import annotations

from typing import Any
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.models import FactCategory
from src.signal_strategy.models import (
    CandidateSignal,
    RankedSignal,
    SignalScore,
)
from src.signal_strategy.strategy_builder import (
    RawStrategyOutput,
    StrategyBuilder,
)


class MockEngine(GeneratorEngine):
    def __init__(self, response_data: Any):
        super().__init__(default_provider="mock")
        self.response_data = response_data

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="{}",
            provider_name="mock",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=self.response_data,
        )


@pytest.fixture
def strategy_fixtures():
    prospect = ProspectContext(
        full_name="Maya Lin",
        company_name="CloudScale Systems",
        company_domain="cloudscale.io",
        designation="VP of Infrastructure",
    )
    product = ProductContext(
        product_name="Kubex",
        description="Kubernetes cost governance and automated cluster rightsizing engine",
        capabilities=[
            "Automated Kubernetes cluster rightsizing",
            "Real-time cloud billing anomaly detection",
            "Continuous multi-cloud cost allocation tags",
        ],
        value_proposition="Reduces cloud infrastructure waste by 40% with zero engineering overhead.",
    )
    outreach = OutreachContext(
        objective="Schedule introductory technical demo",
        sender_identity="Antigravity Alliances",
        desired_cta="15-minute introductory engineering chat next Thursday?",
        tone="peer-to-peer technical",
        constraints=["Under 80 words", "Do not cite competitor pricing"],
    )
    return prospect, product, outreach


def test_strategy_builder_happy_path_and_capability_bounding(strategy_fixtures):
    """Strategy is formulated successfully and hallucinated capabilities are strictly stripped."""
    prospect, product, outreach = strategy_fixtures

    sig = CandidateSignal(
        signal_id="sig-k8s-surge",
        label="Cluster Sprawl and Cost Surge",
        description="CloudScale launched 5 new microservice clusters leading to 45% compute surge",
        supporting_fact_ids=["fact-k8s"],
        fact_category=FactCategory.CHALLENGE,
        observed_event="Launch of 5 microservice clusters with 45% compute increase",
        business_implication="Severe cloud bill overrun and manual cluster configuration burnout",
        commercial_opportunity="Kubex automated cluster rightsizing and real-time anomaly detection",
    )
    score = SignalScore(
        evidence_strength=0.95,
        recency_score=0.90,
        prospect_relevance=0.90,
        product_alignment=0.95,
        business_significance=0.85,
        composite_score=0.91,
    )
    ranked = RankedSignal(
        signal=sig,
        score=score,
        rationale="Strong direct fit with Kubex core capabilities.",
    )

    # LLM returns 2 genuine product capabilities + 1 hallucinated capability
    raw_output = RawStrategyOutput(
        prospect_role_relevance="As VP Infrastructure, Maya owns the compute budget and cluster efficiency.",
        business_implications="Rapid cluster expansion drives uncontrolled infrastructure spend.",
        capability_mapping=[
            "Automated Kubernetes cluster rightsizing",  # Valid
            "Real-time cloud billing anomaly detection",  # Valid
            "Quantum Multi-Cloud AI Teleportation Engine",  # Hallucinated!
        ],
        messaging_approach="challenge-led",
        central_message="Kubex automatically rightsizes CloudScale's expanding clusters to curb compute overruns.",
        suggested_subject_line="CloudScale's new microservice cluster efficiency",
        cta="15-minute engineering chat next Thursday?",
        tone_guidance="Technical, respectful peer tone",
        constraint_checklist=["Focus strictly on container waste", "Do not assert unverified cluster metrics"],
    )

    builder = StrategyBuilder(engine=MockEngine(raw_output))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is True
    assert result.strategy is not None
    strat = result.strategy
    assert strat.primary_signal.signal.signal_id == "sig-k8s-surge"
    assert strat.messaging_approach == "challenge-led"
    assert "CloudScale" in strat.suggested_subject_line

    # Verify capability bounding: the hallucinated quantum engine must be removed
    assert len(strat.alignment.capability_mapping) == 2
    assert "Automated Kubernetes cluster rightsizing" in strat.alignment.capability_mapping
    assert "Real-time cloud billing anomaly detection" in strat.alignment.capability_mapping
    assert "Quantum Multi-Cloud AI Teleportation Engine" not in strat.alignment.capability_mapping

    # Verify constraints: outreach constraints merged with LLM constraints
    assert "Under 80 words" in strat.constraint_checklist
    assert "Do not cite competitor pricing" in strat.constraint_checklist
    assert "Focus strictly on container waste" in strat.constraint_checklist


def test_strategy_builder_unviable_when_composite_score_too_low(strategy_fixtures):
    """When the top signal has a low composite score, formulation returns is_viable=False."""
    prospect, product, outreach = strategy_fixtures

    sig = CandidateSignal(
        signal_id="sig-weak",
        label="Minor Event",
        description="Minor event",
        supporting_fact_ids=["fact-1"],
    )
    score = SignalScore(
        evidence_strength=0.30,
        recency_score=0.30,
        prospect_relevance=0.30,
        product_alignment=0.30,
        business_significance=0.20,
        composite_score=0.28,  # Below min_composite_score threshold of 0.35
    )
    ranked = RankedSignal(signal=sig, score=score, rationale="Weak signal")

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert "below minimum viability threshold" in (result.rejection_reason or "")
    assert len(result.unviable_reasons) >= 1


def test_strategy_builder_unviable_when_product_alignment_too_low(strategy_fixtures):
    """When top signal has negligible product alignment, formulation returns is_viable=False."""
    prospect, product, outreach = strategy_fixtures

    sig = CandidateSignal(
        signal_id="sig-no-fit",
        label="Irrelevant Event",
        description="Company hosted charity golf tournament",
        supporting_fact_ids=["fact-golf"],
    )
    score = SignalScore(
        evidence_strength=1.0,
        recency_score=1.0,
        prospect_relevance=0.8,
        product_alignment=0.10,  # Below min_product_alignment of 0.25
        business_significance=0.8,
        composite_score=0.55,  # Composite score passes, but alignment fails!
    )
    ranked = RankedSignal(signal=sig, score=score, rationale="Zero product fit")

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert "below minimum alignment threshold" in (result.rejection_reason or "")


def test_strategy_builder_empty_signals_handled(strategy_fixtures):
    """Empty ranked signals returns an explicit non-viable result."""
    prospect, product, outreach = strategy_fixtures
    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[],
        prospect=prospect,
        product=product,
        outreach=outreach,
    )
    assert result.is_viable is False
    assert result.strategy is None
    assert "No candidate signals" in (result.rejection_reason or "")


def test_strategy_builder_deterministic_fallback(strategy_fixtures):
    """If LLM fails, deterministic fallback builds a valid OutreachStrategy."""
    prospect, product, outreach = strategy_fixtures

    sig = CandidateSignal(
        signal_id="sig-fallback",
        label="Regional Expansion",
        description="CloudScale opened European data center",
        supporting_fact_ids=["fact-dc"],
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        observed_event="European data center opened",
        business_implication="Need to manage multi-region cluster costs",
    )
    score = SignalScore(
        evidence_strength=0.90,
        recency_score=0.90,
        prospect_relevance=0.85,
        product_alignment=0.80,
        business_significance=0.80,
        composite_score=0.84,
    )
    ranked = RankedSignal(signal=sig, score=score, rationale="Good fit")

    class FailingEngine(GeneratorEngine):
        def __init__(self):
            super().__init__(default_provider="mock")

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise RuntimeError("API timeout")

    builder = StrategyBuilder(engine=FailingEngine())
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is True
    assert result.strategy is not None
    assert result.strategy.primary_signal.signal.signal_id == "sig-fallback"
    assert len(result.strategy.alignment.capability_mapping) >= 1
    assert result.strategy.alignment.capability_mapping[0] in product.capabilities


def test_strategy_builder_unviable_when_product_alignment_is_unevaluated(strategy_fixtures):
    """When top signal has unevaluated product_alignment (None), strategy must be marked unviable."""
    prospect, product, outreach = strategy_fixtures

    sig = CandidateSignal(
        signal_id="sig-uneval",
        label="Event with Unevaluated Alignment",
        description="Event description",
        supporting_fact_ids=["fact-1"],
    )
    score = SignalScore(
        evidence_strength=0.90,
        recency_score=0.90,
        prospect_relevance=0.85,
        product_alignment=None,  # Unevaluated due to LLM failure
        business_significance=0.80,
        composite_score=0.20,
    )
    ranked = RankedSignal(signal=sig, score=score, rationale="Alignment was unevaluated")

    builder = StrategyBuilder(engine=MockEngine(None))
    result = builder.build_strategy(
        ranked_signals=[ranked],
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert "product alignment is unevaluated" in (result.rejection_reason or "")
    assert len(result.unviable_reasons) >= 1
    assert "product alignment is unevaluated" in result.unviable_reasons[0]

