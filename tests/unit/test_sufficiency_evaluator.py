from __future__ import annotations

import datetime
from typing import Any, List, Optional
import pytest

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
    ResolutionStatus,
    SourceRecord,
)
from src.researcher.orchestration.evaluator import (
    RawObjectiveAssessment,
    RawRecommendedAction,
    RawSufficiencyEvaluation,
    ResearchSufficiencyEvaluator,
)
from src.researcher.orchestration.models import (
    ActionType,
    ConflictSeverity,
    CoverageStatus,
    PropositionType,
    SearchViability,
    SufficiencyDecision,
)
from src.researcher.planning.models import ResearchObjective, ResearchObjectiveType, TargetEntity


class MockEngine(GeneratorEngine):
    def __init__(self, raw_output: RawSufficiencyEvaluation):
        self.raw_output = raw_output

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="{}",
            provider_name="mock_groq",
            raw_response=ProviderResponse(text="{}"),
            parsed_output=self.raw_output,
        )


def _make_sample_entities():
    prospect = EntityRecord(
        entity_id="p-1",
        entity_type=EntityType.PERSON,
        raw_name="Elena Rostova",
        canonical_name="Elena Rostova",
        domain="vanguardai.io",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )
    company = EntityRecord(
        entity_id="c-1",
        entity_type=EntityType.ORGANIZATION,
        raw_name="Vanguard AI",
        canonical_name="Vanguard AI Inc.",
        domain="vanguardai.io",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )
    return prospect, company


def test_full_coverage_scenario_sufficient():
    """All critical objectives ADEQUATE and identity confirmed yields SUFFICIENT."""
    prospect, company = _make_sample_entities()
    obj1 = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Elena's current role",
        is_critical=True,
    )
    obj2 = ResearchObjective(
        objective_id="obj-2",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify Vanguard funding history",
        is_critical=True,
    )

    ev1 = EvidenceRecord(source_id="src-1", verbatim_excerpt="Elena is Head of Research at Vanguard AI")
    fact1 = CanonicalFact(fact_id="fact-1", claim_text="Elena is Head of Research", evidence_records=[ev1])

    ev2 = EvidenceRecord(source_id="src-2", verbatim_excerpt="Vanguard raised $50M in Series B")
    fact2 = CanonicalFact(fact_id="fact-2", claim_text="Vanguard raised Series B", evidence_records=[ev2])

    source1 = SourceRecord(source_id="src-1", url="https://vanguardai.io/team", domain="vanguardai.io", raw_text="Team page", fetch_date=datetime.date(2026, 11, 1))
    source2 = SourceRecord(source_id="src-2", url="https://techcrunch.com/vanguard", domain="techcrunch.com", raw_text="News article", fetch_date=datetime.date(2026, 11, 1))

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="speech_act_announcement",
                supporting_fact_ids=["fact-1"],
            ),
            RawObjectiveAssessment(
                objective_id="obj-2",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="commercial_realization",
                supporting_fact_ids=["fact-2"],
            ),
        ],
        explanation="All core objectives verified",
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj1, obj2],
        facts=[fact1, fact2],
        sources=[source1, source2],
        prospect_entity=prospect,
        company_entity=company,
    )

    assert result.decision == SufficiencyDecision.SUFFICIENT
    assert result.is_sufficient is True
    assert result.dimensions.objective_coverage.status == "adequate"
    assert result.dimensions.identity_certainty.status == "confirmed"


def test_speech_act_vs_commercial_realization_authority():
    """Primary company source satisfies speech acts, but caps commercial realization at PARTIAL without 3rd-party source."""
    prospect, company = _make_sample_entities()
    obj_comm = ResearchObjective(
        objective_id="obj-comm",
        type=ResearchObjectiveType.PRODUCT_ORIENTED,
        target_entity=TargetEntity.COMPANY,
        description="Verify commercial deployment across 50 enterprises",
        is_critical=True,
    )

    ev_blog = EvidenceRecord(source_id="src-blog", verbatim_excerpt="We deployed to 50 enterprises")
    fact_deploy = CanonicalFact(fact_id="fact-deploy", claim_text="Deployed to 50 enterprise customers", evidence_records=[ev_blog])
    source_blog = SourceRecord(
        source_id="src-blog",
        url="https://vanguardai.io/blog/50-customers",
        domain="vanguardai.io",
        raw_text="Blog post",
        fetch_date=datetime.date(2026, 11, 1),
    )

    # LLM says adequate commercial realization
    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-comm",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="commercial_realization",
                supporting_fact_ids=["fact-deploy"],
            )
        ],
        remaining_gaps=[],
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj_comm],
        facts=[fact_deploy],
        sources=[source_blog],
        prospect_entity=prospect,
        company_entity=company,
    )

    # Deterministic policy must cap at PARTIAL and emit CONTINUE
    assert result.decision == SufficiencyDecision.CONTINUE
    obj_result = result.objective_assessments[0]
    assert obj_result.coverage_status == CoverageStatus.PARTIAL
    assert obj_result.viability == SearchViability.PROMISING
    # Check that a recommended action for 3rd party corroboration was injected
    assert any("Independent corroboration required" in gap for gap in result.remaining_gaps)


def test_grounding_validation_discards_hallucinated_facts():
    """LLM citing non-existent fact UUIDs has those IDs stripped; if none remain, objective reverts to UNADDRESSED."""
    prospect, company = _make_sample_entities()
    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Discover recent patent filings",
        is_critical=True,
    )

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="speech_act_announcement",
                supporting_fact_ids=["non-existent-fact-uuid-1234"],
            )
        ]
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj],
        facts=[],  # No facts in session!
        sources=[],
        prospect_entity=prospect,
        company_entity=company,
    )

    obj_res = result.objective_assessments[0]
    assert obj_res.coverage_status == CoverageStatus.UNADDRESSED
    assert obj_res.supporting_fact_ids == ()
    assert result.decision == SufficiencyDecision.CONTINUE


def test_symmetric_identity_gate():
    """If prospect is resolved HIGH but company is UNRESOLVED, sufficiency is blocked and decision is CONTINUE."""
    prospect = EntityRecord(
        entity_id="p-1",
        entity_type=EntityType.PERSON,
        raw_name="Elena Rostova",
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
    )
    company_unresolved = EntityRecord(
        entity_id="c-unresolved",
        entity_type=EntityType.ORGANIZATION,
        raw_name="Vanguard AI",
        resolution_status=ResolutionStatus.UNRESOLVED,
        confidence=ConfidenceLevel.UNRESOLVED,
    )

    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.COMPANY,
        description="Verify company identity",
        is_critical=True,
    )
    ev = EvidenceRecord(source_id="s1", verbatim_excerpt="Company exists")
    fact = CanonicalFact(fact_id="f1", claim_text="Company verified", evidence_records=[ev])
    src = SourceRecord(source_id="s1", url="https://example.com", domain="example.com", raw_text="text", fetch_date=datetime.date(2026, 11, 1))

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                supporting_fact_ids=["f1"],
            )
        ],
        identity_resolution_viable=True,
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj],
        facts=[fact],
        sources=[src],
        prospect_entity=prospect,
        company_entity=company_unresolved,
    )

    assert result.decision == SufficiencyDecision.CONTINUE
    assert result.dimensions.identity_certainty.status == "uncertain"


def test_conflict_severity_handling():
    """Confirmed conflict blocks sufficiency; unknown conflict is treated conservatively as evidence deficiency."""
    prospect, company = _make_sample_entities()
    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify current title",
        is_critical=True,
    )
    ev = EvidenceRecord(source_id="s1", verbatim_excerpt="Elena is VP of Product")
    fact = CanonicalFact(fact_id="f1", claim_text="Elena is VP of Product", evidence_records=[ev])
    src = SourceRecord(source_id="s1", url="https://example.com", domain="example.com", raw_text="text", fetch_date=datetime.date(2026, 11, 1))

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                supporting_fact_ids=["f1"],
            )
        ]
    )

    conflicts = [
        {
            "conflict_id": "conf-1",
            "description": "Contradiction regarding Elena current position",
            "severity": "unknown",
            "fact_ids": ["f1"],
        }
    ]

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj],
        facts=[fact],
        sources=[src],
        prospect_entity=prospect,
        company_entity=company,
        conflicts=conflicts,
    )

    # Unknown conflict conservative handling: capped at PARTIAL, decision is CONTINUE
    assert result.decision == SufficiencyDecision.CONTINUE
    assert result.objective_assessments[0].coverage_status == CoverageStatus.PARTIAL
    assert any("Unresolved conflict with unknown severity" in d for d in result.evidence_deficiencies)


def test_budget_ceiling_termination():
    """When iteration reaches max_iterations and is not SUFFICIENT, decision is INSUFFICIENT."""
    prospect, company = _make_sample_entities()
    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Find unlisted patent applications",
        is_critical=True,
    )

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="partial",
                viability="promising",
                supporting_fact_ids=[],
            )
        ]
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-3",
        iteration_number=3,
        max_iterations=3,
        objectives=[obj],
        facts=[],
        sources=[],
        prospect_entity=prospect,
        company_entity=company,
    )

    assert result.decision == SufficiencyDecision.INSUFFICIENT
    assert result.is_sufficient is False


def test_evaluative_claim_capping_without_third_party():
    """Evaluative claims like 'industry leading' are capped at PARTIAL if sourced solely from company blog."""
    prospect, company = _make_sample_entities()
    obj = ResearchObjective(
        objective_id="obj-eval",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify Vanguard is the top AI provider in Europe",
        is_critical=True,
    )
    ev = EvidenceRecord(source_id="s-blog", verbatim_excerpt="Vanguard is Europe's number one AI platform")
    fact = CanonicalFact(fact_id="f-eval", claim_text="Vanguard is #1 AI platform in Europe", evidence_records=[ev])
    src = SourceRecord(source_id="s-blog", url="https://vanguardai.io/about", domain="vanguardai.io", raw_text="About us", fetch_date=datetime.date(2026, 11, 1))

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-eval",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="evaluative_claim",
                supporting_fact_ids=["f-eval"],
            )
        ]
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj],
        facts=[fact],
        sources=[src],
        prospect_entity=prospect,
        company_entity=company,
    )

    assert result.decision == SufficiencyDecision.CONTINUE
    assert result.objective_assessments[0].coverage_status == CoverageStatus.PARTIAL


def test_partial_critical_objective_viability_mix():
    """If Obj 1 is EXHAUSTED but Obj 2 is PARTIAL with PROMISING viability, decision is CONTINUE."""
    prospect, company = _make_sample_entities()
    obj1 = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Elena history",
        is_critical=True,
    )
    obj2 = ResearchObjective(
        objective_id="obj-2",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify European enterprise customers",
        is_critical=True,
    )

    ev1 = EvidenceRecord(source_id="s1", verbatim_excerpt="Elena is Head of Research")
    fact1 = CanonicalFact(fact_id="f1", claim_text="Elena is Head of Research", evidence_records=[ev1])
    src1 = SourceRecord(source_id="s1", url="https://vanguardai.io/team", domain="vanguardai.io", raw_text="Team", fetch_date=datetime.date(2026, 11, 1))

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="adequate",
                viability="exhausted",
                proposition_type="speech_act_announcement",
                supporting_fact_ids=["f1"],
            ),
            RawObjectiveAssessment(
                objective_id="obj-2",
                coverage_status="partial",
                viability="promising",
                proposition_type="commercial_realization",
                supporting_fact_ids=[],
                missing_aspects=["Specific customer names"],
            ),
        ]
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj1, obj2],
        facts=[fact1],
        sources=[src1],
        prospect_entity=prospect,
        company_entity=company,
    )

    assert result.decision == SufficiencyDecision.CONTINUE
    assert result.is_sufficient is False


def test_all_incomplete_critical_objectives_exhausted_yields_insufficient():
    """If all incomplete critical objectives have viability EXHAUSTED, decision terminates INSUFFICIENT even before max_iterations."""
    prospect, company = _make_sample_entities()
    obj = ResearchObjective(
        objective_id="obj-1",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Verify confidential board members",
        is_critical=True,
    )

    mock_raw = RawSufficiencyEvaluation(
        objective_assessments=[
            RawObjectiveAssessment(
                objective_id="obj-1",
                coverage_status="partial",
                viability="exhausted",
                supporting_fact_ids=[],
            )
        ],
        identity_resolution_viable=False,
        conflict_resolution_viable=False,
    )

    evaluator = ResearchSufficiencyEvaluator(engine=MockEngine(mock_raw))
    result = evaluator.evaluate(
        session_id="s1",
        iteration_id="it-1",
        iteration_number=1,
        max_iterations=3,
        objectives=[obj],
        facts=[],
        sources=[],
        prospect_entity=prospect,
        company_entity=company,
    )

    assert result.decision == SufficiencyDecision.INSUFFICIENT
    assert result.is_sufficient is False
