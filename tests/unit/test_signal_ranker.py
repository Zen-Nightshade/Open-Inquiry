from __future__ import annotations

import datetime
from typing import Any
import pytest

from src.context.models import ProductContext, ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EvidenceRecord,
    FactCategory,
)
from src.signal_strategy.models import CandidateSignal
from src.signal_strategy.ranker import (
    RawEvaluatedSignal,
    RawSignalEvaluationOutput,
    SignalRanker,
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
def ranking_context():
    prospect = ProspectContext(
        full_name="Elena Vance",
        company_name="Nexus Health",
        company_domain="nexushealth.com",
        designation="VP of Clinical Infrastructure",
    )
    product = ProductContext(
        product_name="MedSync",
        description="HIPAA-compliant EHR data pipeline and real-time FHIR synchronization engine",
        capabilities=[
            "Real-time FHIR v4 bidirectional synchronization",
            "Automated HIPAA compliance auditing",
            "Zero-latency clinical database integration",
        ],
        value_proposition="Eliminates patient record sync lag between hospital EHR systems and clinical apps.",
    )
    return prospect, product


def test_signal_ranker_multi_dimensional_scoring_and_ordering(ranking_context):
    """Signals are scored across all 5 dimensions and sorted descending by composite score."""
    prospect, product = ranking_context

    fact1 = CanonicalFact(
        fact_id="fact-1",
        claim_text="Nexus Health acquired 3 regional hospital clinics requiring EHR data migration",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        evidence_records=[
            EvidenceRecord(source_id="s1", verbatim_excerpt="Nexus acquires 3 clinics"),
            EvidenceRecord(source_id="s2", verbatim_excerpt="Clinics EHR migration announced"),
        ],
        primary_event_date=datetime.date.today() - datetime.timedelta(days=15),
        composite_confidence=ConfidenceLevel.HIGH,
    )
    fact2 = CanonicalFact(
        fact_id="fact-2",
        claim_text="Nexus Health sponsored annual local community 5k charity walk",
        fact_category=FactCategory.OTHER,
        evidence_records=[EvidenceRecord(source_id="s3", verbatim_excerpt="5k charity sponsorship")],
        primary_event_date=datetime.date.today() - datetime.timedelta(days=5),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig1 = CandidateSignal(
        signal_id="sig-ehr-migration",
        label="Hospital Acquisition EHR Migration",
        description="Nexus Health acquired 3 clinics and must migrate disparate patient record systems",
        supporting_fact_ids=["fact-1"],
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        event_date=fact1.primary_event_date,
    )
    sig2 = CandidateSignal(
        signal_id="sig-charity-walk",
        label="Local 5k Charity Walk",
        description="Nexus Health sponsored local 5k run",
        supporting_fact_ids=["fact-2"],
        fact_category=FactCategory.OTHER,
        event_date=fact2.primary_event_date,
    )

    raw_eval = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="sig-ehr-migration",
                prospect_relevance=0.95,
                product_alignment=0.95,
                business_significance=0.85,
                rationale="Direct match with MedSync's FHIR EHR synchronization capabilities and VP Clinical Infrastructure scope.",
            ),
            RawEvaluatedSignal(
                signal_id="sig-charity-walk",
                prospect_relevance=0.10,
                product_alignment=0.05,
                business_significance=0.20,
                rationale="Charity sponsorship has no technical or operational relevance to MedSync EHR synchronization.",
            ),
        ]
    )

    ranker = SignalRanker(engine=MockEngine(raw_eval))
    ranked = ranker.rank_signals(
        signals=[sig2, sig1],  # passed out of order
        facts=[fact1, fact2],
        prospect=prospect,
        product=product,
    )

    assert len(ranked) == 2
    # sig-ehr-migration must be #1
    top = ranked[0]
    assert top.signal.signal_id == "sig-ehr-migration"
    assert top.score.composite_score >= 0.85
    assert top.score.product_alignment == 0.95
    assert top.score.prospect_relevance == 0.95
    assert top.score.recency_score == 1.0  # within 15 days
    assert top.score.evidence_strength >= 0.90

    # sig-charity-walk must be #2 and penalized by gating
    bottom = ranked[1]
    assert bottom.signal.signal_id == "sig-charity-walk"
    assert bottom.score.composite_score <= 0.10  # gated by product_alignment (0.05)


def test_signal_ranker_product_alignment_gating(ranking_context):
    """Signals with product_alignment < 0.25 cannot masquerade as viable outreach hooks."""
    prospect, product = ranking_context

    fact = CanonicalFact(
        fact_id="f1",
        claim_text="Nexus Health renovated main campus cafeteria",
        fact_category=FactCategory.OTHER,
        evidence_records=[
            EvidenceRecord(source_id="s1", verbatim_excerpt="Cafeteria open"),
            EvidenceRecord(source_id="s2", verbatim_excerpt="Renovation complete"),
        ],
        primary_event_date=datetime.date.today(),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    sig = CandidateSignal(
        signal_id="sig-cafe",
        label="Cafeteria Renovation",
        description="Renovated campus dining hall",
        supporting_fact_ids=["f1"],
        event_date=fact.primary_event_date,
    )

    # Even if an LLM hallucinates high business significance or recency
    raw_eval = RawSignalEvaluationOutput(
        evaluations=[
            RawEvaluatedSignal(
                signal_id="sig-cafe",
                prospect_relevance=0.80,
                product_alignment=0.10,  # Irrelevant to healthcare IT
                business_significance=0.90,
                rationale="Cafeteria is nice, but irrelevant to clinical IT.",
            )
        ]
    )

    ranker = SignalRanker(engine=MockEngine(raw_eval))
    ranked = ranker.rank_signals(signals=[sig], facts=[fact], prospect=prospect, product=product)

    assert len(ranked) == 1
    # Gate caps composite score to min(raw_composite, product_alignment)
    assert ranked[0].score.composite_score <= 0.10


def test_signal_ranker_deterministic_evidence_strength(ranking_context):
    """Evidence strength reflects fact confidence, multi-source corroboration, and inference penalty."""
    prospect, product = ranking_context

    # Fact A: High confidence + 2 independent sources
    fact_high = CanonicalFact(
        fact_id="f-high",
        claim_text="High confidence claim",
        evidence_records=[
            EvidenceRecord(source_id="s1", verbatim_excerpt="Quote 1"),
            EvidenceRecord(source_id="s2", verbatim_excerpt="Quote 2"),
        ],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    # Fact B: Low confidence + 1 source
    fact_low = CanonicalFact(
        fact_id="f-low",
        claim_text="Low confidence unverified rumor",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Rumor excerpt")],
        composite_confidence=ConfidenceLevel.LOW,
    )

    sig_high = CandidateSignal(
        signal_id="sig-high",
        label="High Confidence Signal",
        description="High confidence",
        supporting_fact_ids=["f-high"],
        is_inferred=False,
    )
    sig_low = CandidateSignal(
        signal_id="sig-low",
        label="Low Confidence Signal",
        description="Low confidence",
        supporting_fact_ids=["f-low"],
        is_inferred=False,
    )
    sig_inferred = CandidateSignal(
        signal_id="sig-inf",
        label="Inferred High Signal",
        description="Inferred",
        supporting_fact_ids=["f-high"],
        is_inferred=True,
    )

    ranker = SignalRanker(engine=MockEngine(RawSignalEvaluationOutput()))
    ranked = ranker.rank_signals(
        signals=[sig_high, sig_low, sig_inferred],
        facts=[fact_high, fact_low],
        prospect=prospect,
        product=product,
    )

    ranked_by_id = {r.signal.signal_id: r for r in ranked}

    # Corroborated high confidence fact gets >= 0.95
    assert ranked_by_id["sig-high"].score.evidence_strength >= 0.95

    # Low confidence fact gets <= 0.40
    assert ranked_by_id["sig-low"].score.evidence_strength <= 0.40

    # Inferred signal gets discount compared to direct high confidence
    assert ranked_by_id["sig-inf"].score.evidence_strength < ranked_by_id["sig-high"].score.evidence_strength


def test_signal_ranker_deterministic_recency_score(ranking_context):
    """Recency score drops predictably based on elapsed days from event date."""
    prospect, product = ranking_context
    today = datetime.date.today()

    f1 = CanonicalFact(
        fact_id="f1",
        claim_text="Recent event",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="q")],
        primary_event_date=today - datetime.timedelta(days=10),
    )
    f2 = CanonicalFact(
        fact_id="f2",
        claim_text="Older event",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="q")],
        primary_event_date=today - datetime.timedelta(days=120),
    )
    f3 = CanonicalFact(
        fact_id="f3",
        claim_text="Very old event",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="q")],
        primary_event_date=today - datetime.timedelta(days=450),
    )
    f4 = CanonicalFact(
        fact_id="f4",
        claim_text="Undated event",
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="q")],
        primary_event_date=None,
    )

    s1 = CandidateSignal(signal_id="s1", label="10 days", description="", supporting_fact_ids=["f1"], event_date=f1.primary_event_date)
    s2 = CandidateSignal(signal_id="s2", label="120 days", description="", supporting_fact_ids=["f2"], event_date=f2.primary_event_date)
    s3 = CandidateSignal(signal_id="s3", label="450 days", description="", supporting_fact_ids=["f3"], event_date=f3.primary_event_date)
    s4 = CandidateSignal(signal_id="s4", label="undated", description="", supporting_fact_ids=["f4"], event_date=None)

    ranker = SignalRanker(engine=MockEngine(RawSignalEvaluationOutput()))
    ranked = ranker.rank_signals(signals=[s1, s2, s3, s4], facts=[f1, f2, f3, f4], prospect=prospect, product=product)
    scores = {r.signal.signal_id: r.score.recency_score for r in ranked}

    assert scores["s1"] == 1.0
    assert scores["s2"] == 0.70
    assert scores["s3"] == 0.30
    assert scores["s4"] == 0.45
