from __future__ import annotations

import pytest
from src.researcher.epistemic_dag.adapter import DAGAdapter
from src.researcher.epistemic_dag.models import (
    DecisionRelevance,
    ObservabilityClass,
    QuestionLifecycleStatus,
)
from src.researcher.evidence_request.models import EvidenceGapReason, EvidenceRequest
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    TargetEntity,
)


def test_convert_objectives_to_dag_preserves_intent_and_dependencies():
    """Verify ResearchObjectives convert accurately without false commercial assumptions."""
    obj1 = ResearchObjective(
        objective_id="obj_ident",
        type=ResearchObjectiveType.IDENTITY_RESOLUTION,
        target_entity=TargetEntity.PROSPECT,
        description="Verify Elena Rostova is Head of Research at Vanguard AI",
        priority=1,
        is_critical=True,
        max_queries=2,
    )
    obj2 = ResearchObjective(
        objective_id="obj_company",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.COMPANY,
        description="Investigate Vanguard AI's announced funding and office expansions",
        priority=2,
        is_critical=True,
        max_queries=3,
    )
    obj3 = ResearchObjective(
        objective_id="obj_prospect_initiative",
        type=ResearchObjectiveType.NEUTRAL_DISCOVERY,
        target_entity=TargetEntity.PROSPECT,
        description="Evaluate technical initiatives led by Elena Rostova",
        priority=3,
        is_critical=False,
        max_queries=2,
    )

    dag = DAGAdapter.convert_objectives_to_dag(
        objectives=[obj1, obj2, obj3],
        run_id="run_100",
        prospect_name="Elena Rostova",
        company_name="Vanguard AI",
    )

    assert len(dag.nodes) == 3
    node_ident = dag.get_node(f"qn_{obj1.objective_id.replace('-', '')[:10]}")
    node_company = dag.get_node(f"qn_{obj2.objective_id.replace('-', '')[:10]}")
    node_initiative = dag.get_node(f"qn_{obj3.objective_id.replace('-', '')[:10]}")

    # Company fact does NOT depend on prospect identity
    assert node_company.dependencies == []
    assert node_company.status == QuestionLifecycleStatus.READY
    assert node_company.observability in (ObservabilityClass.PUBLICLY_OBSERVABLE, ObservabilityClass.CONDITIONALLY_OBSERVABLE)

    # Prospect-specific initiative DOES depend on prospect identity
    assert node_ident.question_id in node_initiative.dependencies
    assert node_initiative.status == QuestionLifecycleStatus.BLOCKED

    # Provenance preserved
    assert node_ident.provenance == f"research_objective:{obj1.objective_id}"
    assert node_company.provenance == f"research_objective:{obj2.objective_id}"


def test_convert_evidence_request_to_dag_node():
    """Verify M3.1 EvidenceRequest converts to a DAG node preserving IDs, neutrality, and budget."""
    req = EvidenceRequest(
        request_id="evr_abc123",
        run_id="run_100",
        target_entity_id="ent_vanguard",
        target_entity_name="Vanguard AI",
        requested_predicate="announced_funding",
        neutral_question="Did Vanguard AI announce a Series A funding round in Q3 2026?",
        neutral_search_terms=["Vanguard AI Series A funding Q3 2026"],
        disconfirming_search_terms=["Vanguard AI self funded bootstrap", "Vanguard AI seed round only"],
        valid_from="2026-07-01",
        valid_to="2026-09-30",
        gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
        provenance_stage="strategy_validation",
        budget_cost_units=1,
    )

    node = DAGAdapter.convert_evidence_request_to_node(req, run_id="run_100")

    assert node.question_id == f"qn_{req.request_id.replace('-', '')[:10]}"
    assert node.neutral_question == req.neutral_question
    assert node.supporting_search_directions == req.neutral_search_terms
    assert node.disconfirming_search_directions == req.disconfirming_search_terms
    assert node.decision_relevance == DecisionRelevance.CRITICAL
    assert node.estimated_research_cost == 1
    assert node.provenance == f"evidence_request:{req.request_id}:{req.provenance_stage}"
    assert node.valid_from == "2026-07-01"
    assert node.valid_to == "2026-09-30"
