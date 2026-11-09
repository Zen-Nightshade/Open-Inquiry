from __future__ import annotations

import datetime
from typing import Any
import pytest
from pydantic import ValidationError

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchPackage,
    ResolutionStatus,
    SourceRecord,
    SourceType,
)
from src.signal_strategy.identifier import (
    RawCandidateSignal,
    RawSignalIdentificationOutput,
    SignalIdentifier,
)
from src.signal_strategy.models import StrategyResult
from src.signal_strategy.ranker import (
    RawEvaluatedSignal,
    RawSignalEvaluationOutput,
    SignalRanker,
)
from src.signal_strategy.service import SignalStrategyService
from src.signal_strategy.strategy_builder import (
    RawStrategyOutput,
    StrategyBuilder,
)


class MultiMockEngine(GeneratorEngine):
    def __init__(self, responses: list[Any]):
        super().__init__(default_provider="mock")
        self.responses = list(responses)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        output = self.responses.pop(0) if self.responses else None
        return GenerationResponse(
            text="{}",
            provider_name="mock",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=output,
        )


@pytest.fixture
def workflow_context():
    prospect = ProspectContext(
        full_name="Dr. Aris Thorne",
        company_name="Vanguard Quantum Labs",
        company_domain="vanguardquantum.com",
        designation="Head of Quantum Hardware",
    )
    product = ProductContext(
        product_name="CryoShield Pro",
        description="Sub-Kelvin cryogenic thermal isolation and magnetic shielding systems",
        capabilities=[
            "Sub-10mK cryogenic temperature stabilization",
            "Multi-layer mu-metal ultra-low magnetic interference shielding",
            "Real-time cryogenic vibration damping analytics",
        ],
        value_proposition="Reduces quantum qubit decoherence rates by 70% in high-noise laboratory environments.",
        differentiators=["Patented passive vibration decoupling", "Compact footprint"],
        limitations=["Requires liquid helium supply"],
    )
    outreach = OutreachContext(
        objective="Invite Dr. Thorne to private executive roundtable on qubit stabilization",
        sender_identity="Antigravity Quantum Alliance",
        desired_cta="Are you open to joining our closed-door session in Boston next month?",
        tone="authoritative scientific peer",
        constraints=["Do not make unsupported decoherence claims", "Keep body text under 90 words"],
    )
    return prospect, product, outreach


def test_signal_strategy_end_to_end_workflow(workflow_context):
    """
    End-to-end integration test of SignalStrategyService:
    1. ResearchPackage with verified facts and entities is provided.
    2. SignalIdentifier extracts grounded candidate signals.
    3. SignalRanker scores and ranks signals across 5 dimensions.
    4. StrategyBuilder selects top signal, bounds capabilities, and builds OutreachStrategy.
    5. Returns viable StrategyResult with frozen immutability.
    """
    prospect, product, outreach = workflow_context

    today = datetime.date.today()

    ev1 = EvidenceRecord(
        source_id="src-mit-review",
        verbatim_excerpt="Vanguard Quantum Labs installed a 64-qubit testbed but reported severe decoherence from magnetic ambient noise.",
        event_date=today - datetime.timedelta(days=20),
        confidence=ConfidenceLevel.HIGH,
    )
    fact1 = CanonicalFact(
        fact_id="fact-qubit-noise",
        claim_text="Vanguard Quantum Labs reported severe decoherence from magnetic ambient noise on 64-qubit testbed",
        fact_category=FactCategory.CHALLENGE,
        evidence_records=[ev1],
        primary_event_date=today - datetime.timedelta(days=20),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    ev2 = EvidenceRecord(
        source_id="src-pr-newswire",
        verbatim_excerpt="Vanguard raised $40M in Series B financing to expand quantum computing hardware infrastructure.",
        event_date=today - datetime.timedelta(days=45),
        confidence=ConfidenceLevel.HIGH,
    )
    fact2 = CanonicalFact(
        fact_id="fact-funding-40m",
        claim_text="Vanguard raised $40M Series B for quantum hardware infrastructure expansion",
        fact_category=FactCategory.FUNDING,
        evidence_records=[ev2],
        primary_event_date=today - datetime.timedelta(days=45),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    source1 = SourceRecord(
        source_id="src-mit-review",
        url="https://technologyreview.com/vanguard-quantum",
        domain="technologyreview.com",
        title="Decoherence Challenges at Vanguard Quantum",
        raw_text="Full article text",
        source_type=SourceType.NEWS,
        fetch_date=today,
    )
    source2 = SourceRecord(
        source_id="src-pr-newswire",
        url="https://prnewswire.com/vanguard-series-b",
        domain="prnewswire.com",
        title="Vanguard Secures $40M",
        raw_text="Full article text",
        source_type=SourceType.PRESS_RELEASE,
        fetch_date=today,
    )

    prospect_entity = EntityRecord(
        entity_id="p-aris",
        entity_type=EntityType.PERSON,
        raw_name=prospect.full_name,
        canonical_name="Dr. Aris Thorne",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )
    company_entity = EntityRecord(
        entity_id="c-vanguard",
        entity_type=EntityType.ORGANIZATION,
        raw_name=prospect.company_name,
        canonical_name="Vanguard Quantum Labs Inc.",
        domain="vanguardquantum.com",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )

    research_package = ResearchPackage(
        prospect_entity=prospect_entity,
        company_entity=company_entity,
        sources=(source1, source2),
        facts=(fact1, fact2),
        search_queries_used=("vanguard quantum decoherence", "vanguard quantum funding"),
        is_sufficient=True,
        is_complete=True,
    )

    # Mock responses for the 3 stages:
    # 1. Identifier
    mock_signals = RawSignalIdentificationOutput(
        candidate_signals=[
            RawCandidateSignal(
                label="64-Qubit Ambient Decoherence Struggle",
                description="Vanguard's new 64-qubit testbed is encountering decoherence issues caused by magnetic noise",
                supporting_fact_ids=["fact-qubit-noise"],
                fact_category="challenge",
                observed_event="Installation of 64-qubit testbed experiencing severe magnetic decoherence",
                business_implication="High qubit error rates blocking quantum algorithm validation milestones",
                commercial_opportunity="CryoShield Pro mu-metal magnetic shielding directly stabilizes qubit coherence",
                is_inferred=False,
            ),
            RawCandidateSignal(
                label="Series B Quantum Hardware Scale-up",
                description="Raised $40M to expand hardware infrastructure",
                supporting_fact_ids=["fact-funding-40m"],
                fact_category="funding",
                observed_event="Closed $40M Series B financing round",
                business_implication="Capital allocation for laboratory buildout and cryogenic equipment",
                commercial_opportunity="CryoShield procurement for new test bays",
                is_inferred=False,
            ),
        ]
    )

    # 2. Ranker
    mock_ranking = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="",  # ranker matches by order or fills
                prospect_relevance=0.98,
                product_alignment=0.99,
                business_significance=0.90,
                rationale="Direct pain point match with Head of Quantum Hardware and CryoShield Pro magnetic shielding.",
            ),
            RawEvaluatedSignal(
                signal_id="",
                prospect_relevance=0.75,
                product_alignment=0.70,
                business_significance=0.85,
                rationale="Funding provides budget for hardware buildout.",
            ),
        ]
    )

    # 3. Strategy Builder
    mock_strategy = RawStrategyOutput(
        prospect_role_relevance="As Head of Quantum Hardware, Dr. Thorne is directly accountable for qubit fidelity and coherence times.",
        business_implications="Environmental magnetic interference in the new 64-qubit testbed threatens compute milestones.",
        capability_mapping=[
            "Multi-layer mu-metal ultra-low magnetic interference shielding",
            "Sub-10mK cryogenic temperature stabilization",
        ],
        messaging_approach="challenge-led",
        central_message="CryoShield Pro isolates Vanguard's 64-qubit testbed from ambient magnetic noise to restore target coherence times.",
        suggested_subject_line="Vanguard's 64-qubit testbed magnetic shielding",
        cta="Are you open to joining our closed-door session in Boston next month?",
        tone_guidance="Scientific peer executive tone",
        constraint_checklist=[
            "Keep technical and reference specific mu-metal specs",
            "Do not cite unverified competitor benchmark numbers",
        ],
    )

    engine = MultiMockEngine([mock_signals, mock_ranking, mock_strategy])
    service = SignalStrategyService(engine=engine)

    result = service.develop_strategy(
        research_package=research_package,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    # Assertions on StrategyResult
    assert isinstance(result, StrategyResult)
    assert result.is_viable is True
    assert result.strategy is not None

    strategy = result.strategy
    assert strategy.primary_signal.signal.label == "64-Qubit Ambient Decoherence Struggle"
    assert "fact-qubit-noise" in strategy.primary_signal.signal.supporting_fact_ids
    assert strategy.primary_signal.score.composite_score >= 0.85
    assert strategy.primary_signal.score.product_alignment >= 0.90
    assert strategy.primary_signal.score.recency_score == 1.0  # 20 days ago

    # Alignment and capabilities
    assert len(strategy.alignment.capability_mapping) == 2
    for cap in strategy.alignment.capability_mapping:
        assert cap in product.capabilities

    # Constraints merged
    assert "Do not make unsupported decoherence claims" in strategy.constraint_checklist
    assert "Keep body text under 90 words" in strategy.constraint_checklist
    assert "Keep technical and reference specific mu-metal specs" in strategy.constraint_checklist

    # Deep immutability verification
    with pytest.raises((ValidationError, TypeError)):
        strategy.primary_signal = None  # type: ignore


def test_signal_strategy_aborts_on_empty_research_package(workflow_context):
    """If research package has zero facts, service aborts cleanly with is_viable=False."""
    prospect, product, outreach = workflow_context

    empty_package = ResearchPackage(
        sources=(),
        facts=(),
        is_sufficient=False,
        is_complete=True,
    )

    service = SignalStrategyService(engine=MultiMockEngine([]))
    result = service.develop_strategy(
        research_package=empty_package,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert "no factual evidence" in (result.rejection_reason or "")


def test_signal_strategy_aborts_when_facts_have_zero_product_relevance(workflow_context):
    """If facts describe events with zero product relevance, service returns is_viable=False."""
    prospect, product, outreach = workflow_context

    fact = CanonicalFact(
        fact_id="f-party",
        claim_text="Vanguard Quantum Labs hosted company summer barbecue in city park",
        fact_category=FactCategory.OTHER,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Annual barbecue held")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    package = ResearchPackage(
        facts=(fact,),
        is_sufficient=True,
        is_complete=True,
    )

    # Identifier outputs party signal
    sig_out = RawSignalIdentificationOutput(
        candidate_signals=[
            RawCandidateSignal(
                label="Company Summer Barbecue",
                description="Hosted annual company picnic",
                supporting_fact_ids=["f-party"],
                fact_category="other",
            )
        ]
    )

    # Ranker assigns very low product alignment
    rank_out = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="",
                prospect_relevance=0.2,
                product_alignment=0.02,  # No cryogenic relevance!
                business_significance=0.1,
                rationale="Barbecue has zero technical relevance to cryogenic hardware.",
            )
        ]
    )

    engine = MultiMockEngine([sig_out, rank_out])
    service = SignalStrategyService(engine=engine)

    result = service.develop_strategy(
        research_package=package,
        prospect=prospect,
        product=product,
        outreach=outreach,
    )

    assert result.is_viable is False
    assert result.strategy is None
    assert any("minimum alignment threshold" in r for r in result.unviable_reasons)
