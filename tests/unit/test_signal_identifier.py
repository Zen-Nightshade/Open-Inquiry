from __future__ import annotations

import datetime
from typing import Any, List
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
from src.signal_strategy.identifier import (
    RawCandidateSignal,
    RawSignalIdentificationOutput,
    SignalIdentifier,
)
from src.signal_strategy.models import CandidateSignal


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
def sample_context():
    prospect = ProspectContext(
        full_name="Alex Rivera",
        company_name="Apex Logistics",
        company_domain="apexlogistics.com",
        designation="Chief Operating Officer",
    )
    product = ProductContext(
        product_name="FleetPulse AI",
        description="Predictive maintenance and route optimization platform for commercial fleets",
        capabilities=[
            "Predictive vehicle breakdown alerting",
            "Dynamic fuel-saving route optimization",
            "Automated fleet compliance audit logs",
        ],
        value_proposition="Reduces unscheduled fleet downtime by 35% and cuts fuel expenses.",
    )
    return prospect, product


def test_signal_identifier_empty_facts(sample_context):
    """If no facts are provided, identifier returns empty list without calling engine."""
    prospect, product = sample_context
    identifier = SignalIdentifier(engine=MockEngine(None))
    signals = identifier.identify_signals(facts=[], prospect=prospect, product=product)
    assert signals == []


def test_signal_identifier_grounding_validation_and_event_separation(sample_context):
    """Signals must be grounded in valid fact IDs, and distinguish event, implication, and opportunity."""
    prospect, product = sample_context

    fact1 = CanonicalFact(
        fact_id="fact-expansion-101",
        claim_text="Apex Logistics expanded fleet by 250 heavy-duty trucks across Texas",
        fact_category=FactCategory.ORGANIZATION_CHANGE,
        evidence_records=[
            EvidenceRecord(source_id="src-1", verbatim_excerpt="Apex Logistics announced addition of 250 trucks.")
        ],
        primary_event_date=datetime.date(2026, 10, 1),
        composite_confidence=ConfidenceLevel.HIGH,
    )
    fact2 = CanonicalFact(
        fact_id="fact-cost-102",
        claim_text="Apex reported 18% increase in diesel maintenance costs in Q3",
        fact_category=FactCategory.CHALLENGE,
        evidence_records=[
            EvidenceRecord(source_id="src-2", verbatim_excerpt="Apex quarterly report shows 18% maintenance surge.")
        ],
        primary_event_date=datetime.date(2026, 10, 15),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    # LLM outputs 2 signals: one validly grounded, one referencing hallucinated fact IDs
    raw_output = RawSignalIdentificationOutput(
        candidate_signals=[
            RawCandidateSignal(
                label="Fleet Expansion Maintenance Pressure",
                description="Apex expanded fleet by 250 trucks while experiencing rising maintenance expenses",
                supporting_fact_ids=["fact-expansion-101", "fact-cost-102"],
                fact_category="challenge",
                observed_event="Addition of 250 heavy-duty trucks alongside 18% maintenance cost increase",
                business_implication="Severe operational overhead and heightened risk of unexpected roadside breakdowns",
                commercial_opportunity="FleetPulse predictive maintenance and route optimization directly tackles downtime",
                is_inferred=True,
            ),
            RawCandidateSignal(
                label="Hallucinated Acquisition",
                description="Apex acquired European freight carrier",
                supporting_fact_ids=["fake-fact-999"],  # Hallucinated!
                fact_category="strategy",
                observed_event="Acquisition",
            ),
        ]
    )

    identifier = SignalIdentifier(engine=MockEngine(raw_output))
    signals = identifier.identify_signals(facts=[fact1, fact2], prospect=prospect, product=product)

    # The ungrounded hallucinated signal must be discarded
    assert len(signals) == 1
    sig = signals[0]
    assert sig.label == "Fleet Expansion Maintenance Pressure"
    assert sig.supporting_fact_ids == ["fact-expansion-101", "fact-cost-102"]
    assert sig.observed_event == "Addition of 250 heavy-duty trucks alongside 18% maintenance cost increase"
    assert "roadside breakdowns" in (sig.business_implication or "")
    assert "FleetPulse" in (sig.commercial_opportunity or "")
    assert sig.event_date == datetime.date(2026, 10, 15)  # latest date
    assert sig.is_inferred is True


def test_signal_identifier_partial_hallucination_filtered(sample_context):
    """If a signal references one valid fact and one fake fact, the fake fact is stripped and valid kept."""
    prospect, product = sample_context

    fact1 = CanonicalFact(
        fact_id="fact-real-1",
        claim_text="Apex appointed Alex Rivera as Chief Operating Officer",
        fact_category=FactCategory.HIRING,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Rivera named COO")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    raw_output = RawSignalIdentificationOutput(
        candidate_signals=[
            RawCandidateSignal(
                label="New COO Appointment",
                description="Alex Rivera joined as COO to modernize dispatch",
                supporting_fact_ids=["fact-real-1", "fake-ghost-fact"],
                fact_category="hiring",
            )
        ]
    )

    identifier = SignalIdentifier(engine=MockEngine(raw_output))
    signals = identifier.identify_signals(facts=[fact1], prospect=prospect, product=product)

    assert len(signals) == 1
    assert signals[0].supporting_fact_ids == ["fact-real-1"]


def test_signal_identifier_fallback_on_llm_failure(sample_context):
    """If LLM fails or errors, identifier executes deterministic fallback from canonical facts."""
    prospect, product = sample_context

    fact = CanonicalFact(
        fact_id="fact-fb-1",
        claim_text="Apex announced new regional distribution hub in Dallas",
        fact_category=FactCategory.PRODUCT_LAUNCH,
        evidence_records=[EvidenceRecord(source_id="s1", verbatim_excerpt="Hub opened in Dallas")],
        primary_event_date=datetime.date(2026, 11, 1),
        composite_confidence=ConfidenceLevel.HIGH,
    )

    class FailingEngine(GeneratorEngine):
        def __init__(self):
            super().__init__(default_provider="mock")

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise RuntimeError("API timeout")

    identifier = SignalIdentifier(engine=FailingEngine())
    signals = identifier.identify_signals(facts=[fact], prospect=prospect, product=product)

    assert len(signals) == 1
    assert signals[0].supporting_fact_ids == ["fact-fb-1"]
    assert "Apex announced new regional distribution hub" in signals[0].description
    assert signals[0].event_date == datetime.date(2026, 11, 1)
