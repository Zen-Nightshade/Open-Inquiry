from __future__ import annotations

import pytest
from src.researcher.epistemic_dag.budget import GlobalResearchBudget
from src.researcher.epistemic_dag.models import (
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    QuestionLifecycleStatus,
    ResolutionCriterion,
    StoppingReason,
)
from src.researcher.epistemic_dag.stopping import StoppingEvaluator


def _make_node(
    question_id: str = "q1",
    obs: ObservabilityClass = ObservabilityClass.PUBLICLY_OBSERVABLE,
    cost: int = 1,
    dependencies: list[str] | None = None,
) -> EpistemicQuestionNode:
    return EpistemicQuestionNode(
        question_id=question_id,
        run_id="run_1",
        requested_predicate="test_predicate",
        neutral_question="Is this factual?",
        observability=obs,
        estimated_research_cost=cost,
        dependencies=dependencies or [],
        required_resolution_criteria=[
            ResolutionCriterion(
                criterion_type="min_sources",
                description="Requires >= 1 corroborated sources",
                is_mandatory=True,
                min_sources=1,
            )
        ],
    )


def test_stopping_resolved_criteria():
    """Verify evidence satisfying criteria triggers StoppingReason.RESOLVED."""
    node = _make_node()
    mock_facts = [{"fact_id": "f1", "claim_text": "Corroborated fact"}]
    mock_sources = [{"source_id": "s1"}]

    decision = StoppingEvaluator.evaluate_post_search(
        node=node,
        facts_found=mock_facts,
        sources_found=mock_sources,
        executed_queries=["query 1"],
    )

    assert decision.should_stop is True
    assert decision.reason == StoppingReason.RESOLVED
    assert decision.new_status == QuestionLifecycleStatus.RESOLVED


def test_stopping_unobservable():
    """Verify private/non-public questions stop pre-search with StoppingReason.UNOBSERVABLE."""
    node = _make_node(obs=ObservabilityClass.PRIVATE_OR_NONPUBLIC)
    dag = EpistemicQuestionDAG(run_id="run_1").add_node(node)
    budget = GlobalResearchBudget(total_budget_units=20)

    decision = StoppingEvaluator.evaluate_pre_search(node, dag, budget)
    assert decision.should_stop is True
    assert decision.reason == StoppingReason.UNOBSERVABLE
    assert decision.new_status == QuestionLifecycleStatus.UNOBSERVABLE
    assert "private/non-public" in decision.rationale


def test_stopping_exhausted_scope_distinguished_from_absence():
    """Verify exhausting allocated search scope records EXHAUSTED_SCOPE and explicit scope metadata without asserting global absence."""
    node = _make_node(cost=2)
    decision = StoppingEvaluator.evaluate_post_search(
        node=node,
        facts_found=[],
        sources_found=[],
        executed_queries=["query 1", "query 2"],
    )

    assert decision.should_stop is True
    assert decision.reason == StoppingReason.EXHAUSTED_SCOPE
    assert decision.new_status == QuestionLifecycleStatus.EXHAUSTED
    assert "Exhausted allocated search scope" in decision.rationale
    assert "This does NOT imply the fact is false or absent globally" in decision.rationale
    assert len(decision.details["executed_queries"]) == 2


def test_stopping_operational_failure_distinguished_from_factual_outcomes():
    """Verify external provider failure records FAILED, distinct from EXHAUSTED or UNOBSERVABLE."""
    node = _make_node()
    decision = StoppingEvaluator.evaluate_post_search(
        node=node,
        facts_found=[],
        sources_found=[],
        executed_queries=["query 1"],
        had_provider_failure=True,
    )

    assert decision.should_stop is True
    assert decision.reason == StoppingReason.OPERATIONAL_FAILURE
    assert decision.new_status == QuestionLifecycleStatus.FAILED
    assert "Recorded as FAILED; this does NOT represent unobservability, absence" in decision.rationale


def test_stopping_dependency_blocked():
    """Verify questions with failed or unobservable prerequisites are marked DEPENDENCY_BLOCKED."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    q_root = _make_node("root", obs=ObservabilityClass.PRIVATE_OR_NONPUBLIC)
    q_child = _make_node("child", dependencies=["root"])
    dag = dag.add_node(q_root).add_node(q_child)

    # Transition root to UNOBSERVABLE
    dag = dag.transition_node("root", QuestionLifecycleStatus.UNOBSERVABLE, stop_reason=StoppingReason.UNOBSERVABLE)

    budget = GlobalResearchBudget(total_budget_units=20)
    decision = StoppingEvaluator.evaluate_pre_search(dag.get_node("child"), dag, budget)

    assert decision.should_stop is True
    assert decision.reason == StoppingReason.DEPENDENCY_BLOCKED
    assert decision.new_status == QuestionLifecycleStatus.BLOCKED


def test_stopping_budget_exhausted():
    """Verify questions stop when global budget is completely spent."""
    node = _make_node()
    dag = EpistemicQuestionDAG(run_id="run_1").add_node(node)
    empty_budget = GlobalResearchBudget(total_budget_units=20, baseline_units_consumed=20)

    decision = StoppingEvaluator.evaluate_pre_search(node, dag, empty_budget)
    assert decision.should_stop is True
    assert decision.reason == StoppingReason.BUDGET_EXHAUSTED
    assert decision.new_status == QuestionLifecycleStatus.SKIPPED
