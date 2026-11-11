from __future__ import annotations

import pytest
from src.researcher.epistemic_dag.budget import GlobalResearchBudget
from src.researcher.epistemic_dag.models import (
    DecisionRelevance,
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    QuestionLifecycleStatus,
    ResolutionLikelihood,
    VoIAction,
)
from src.researcher.epistemic_dag.voi import ValueOfInformationEvaluator


def _make_node(
    question_id: str,
    obs: ObservabilityClass = ObservabilityClass.PUBLICLY_OBSERVABLE,
    relevance: DecisionRelevance = DecisionRelevance.HIGH,
    cost: int = 1,
    dependencies: list[str] | None = None,
) -> EpistemicQuestionNode:
    return EpistemicQuestionNode(
        question_id=question_id,
        run_id="run_1",
        requested_predicate="test_predicate",
        neutral_question=f"Question {question_id}?",
        observability=obs,
        decision_relevance=relevance,
        estimated_research_cost=cost,
        dependencies=dependencies or [],
    )


def test_voi_high_relevance_publicly_observable():
    """Verify high relevance + publicly observable facts yield RESEARCH action."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    node = _make_node("q1", obs=ObservabilityClass.PUBLICLY_OBSERVABLE, relevance=DecisionRelevance.HIGH)
    dag = dag.add_node(node)
    budget = GlobalResearchBudget(total_budget_units=20)

    ass = ValueOfInformationEvaluator.evaluate(node, dag, budget)
    assert ass.action == VoIAction.RESEARCH
    assert ass.voi_score >= ValueOfInformationEvaluator.RESEARCH_THRESHOLD
    assert "Sufficient net epistemic yield" in ass.explanation


def test_voi_private_facts_stop_immediately():
    """Verify private/non-public facts receive action STOP and zero VoI to protect budget."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    node = _make_node("q_private", obs=ObservabilityClass.PRIVATE_OR_NONPUBLIC, relevance=DecisionRelevance.CRITICAL)
    dag = dag.add_node(node)
    budget = GlobalResearchBudget(total_budget_units=20)

    ass = ValueOfInformationEvaluator.evaluate(node, dag, budget)
    assert ass.action == VoIAction.STOP
    assert ass.voi_score == 0.0
    assert ass.resolution_likelihood == ResolutionLikelihood.NEGLIGIBLE
    assert "PRIVATE_OR_NONPUBLIC" in ass.explanation


def test_voi_dependency_benefit_unlocks_elevated_priority():
    """Verify a root question unlocking multiple critical questions receives elevated VoI score."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    root = _make_node("root", obs=ObservabilityClass.CONDITIONALLY_OBSERVABLE, relevance=DecisionRelevance.MEDIUM)
    child1 = _make_node("c1", dependencies=["root"], relevance=DecisionRelevance.CRITICAL)
    child2 = _make_node("c2", dependencies=["root"], relevance=DecisionRelevance.CRITICAL)
    dag = dag.add_node(root).add_node(child1).add_node(child2)

    budget = GlobalResearchBudget(total_budget_units=20)
    ass = ValueOfInformationEvaluator.evaluate(root, dag, budget)

    # Dependency benefit should be 0.4 + 0.4 = 0.8
    assert ass.dependency_benefit >= 0.8
    assert ass.action == VoIAction.RESEARCH


def test_voi_cost_penalty_and_budget_exhaustion():
    """Verify when budget is exhausted or cost exceeds remaining units, action is STOP."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    node = _make_node("q1", cost=5)
    dag = dag.add_node(node)

    # Budget with only 2 queries remaining
    exhausted_budget = GlobalResearchBudget(
        total_budget_units=20,
        baseline_units_consumed=18,
    )
    assert exhausted_budget.remaining_total_units == 2

    ass = ValueOfInformationEvaluator.evaluate(node, dag, exhausted_budget)
    assert ass.action == VoIAction.STOP
    assert "Global research budget exhausted" in ass.explanation
