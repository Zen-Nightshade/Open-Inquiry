from __future__ import annotations

import pytest
from src.researcher.epistemic_dag.models import (
    DecisionRelevance,
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    QuestionLifecycleStatus,
    ResolutionCriterion,
    StoppingReason,
)


def _make_node(
    question_id: str,
    run_id: str = "run_1",
    predicate: str = "works_for",
    neutral_question: str = "Does Alice work at Acme Corp?",
    dependencies: list[str] | None = None,
    priority: int = 5,
    status: QuestionLifecycleStatus = QuestionLifecycleStatus.PENDING,
    cost: int = 1,
) -> EpistemicQuestionNode:
    return EpistemicQuestionNode(
        question_id=question_id,
        run_id=run_id,
        requested_predicate=predicate,
        neutral_question=neutral_question,
        dependencies=dependencies or [],
        priority=priority,
        status=status,
        estimated_research_cost=cost,
    )


def test_dag_valid_construction_and_topological_sort():
    """Verify DAG constructs cleanly and yields a valid topological order."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    n1 = _make_node("q1", priority=1)
    n2 = _make_node("q2", dependencies=["q1"], priority=2)
    n3 = _make_node("q3", dependencies=["q2"], priority=3)

    dag = dag.add_node(n1).add_node(n2).add_node(n3)
    order = dag.get_topological_order()

    assert order == ["q1", "q2", "q3"]
    assert len(dag.nodes) == 3


def test_dag_cycle_detection_kahn():
    """Verify that circular dependencies (A -> B -> A) raise a clear cycle error."""
    n1 = _make_node("q1", dependencies=["q2"])
    n2 = _make_node("q2", dependencies=["q1"])

    with pytest.raises(ValueError, match="Cycle detected in EpistemicQuestionDAG"):
        EpistemicQuestionDAG(run_id="run_1", nodes={"q1": n1, "q2": n2}).validate()


def test_dag_self_dependency_detection():
    """Verify self-dependencies are caught during validation."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    n1 = _make_node("q1", dependencies=["q1"])

    with pytest.raises(ValueError, match="Self-dependency detected"):
        dag.add_node(n1)


def test_dag_missing_dependency_detection():
    """Verify that referencing a non-existent question ID raises a validation error."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    n1 = _make_node("q1", dependencies=["ghost_q"])

    with pytest.raises(ValueError, match="Missing dependency: question 'q1' depends on non-existent question 'ghost_q'"):
        dag.add_node(n1)


def test_dag_duplicate_id_detection():
    """Verify duplicate question IDs are rejected."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    n1 = _make_node("q1")
    dag = dag.add_node(n1)

    with pytest.raises(ValueError, match="Duplicate question ID 'q1'"):
        dag.add_node(n1)


def test_dag_cross_run_ownership_rejection():
    """Verify that nodes belonging to a different run_id cannot be added."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    n_other = _make_node("q1", run_id="run_2")

    with pytest.raises(ValueError, match="Cross-run ownership violation"):
        dag.add_node(n_other)


def test_independent_branch_scheduling_and_dependency_unblocking():
    """Verify independent branches are READY immediately, while dependent branches unblock when parents resolve."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    # Branch 1 (Independent): Company milestone
    q_company = _make_node("q_company", predicate="announced_funding", neutral_question="Did company announce funding?")
    # Branch 2: Identity (root) -> Leadership authority (dependent)
    q_ident = _make_node("q_ident", predicate="works_for", neutral_question="Does Bob work at company?")
    q_auth = _make_node("q_auth", predicate="has_authority", dependencies=["q_ident"])

    dag = dag.add_node(q_company).add_node(q_ident).add_node(q_auth)

    # Initial state after dependency evaluation
    dag = dag.reevaluate_dependencies()
    ready_ids = {n.question_id for n in dag.get_ready_nodes()}
    blocked_ids = {n.question_id for n in dag.get_blocked_nodes()}

    assert ready_ids == {"q_company", "q_ident"}
    assert blocked_ids == {"q_auth"}

    # Resolve identity prerequisite
    dag = dag.transition_node("q_ident", QuestionLifecycleStatus.IN_PROGRESS)
    dag = dag.transition_node("q_ident", QuestionLifecycleStatus.RESOLVED, stop_reason=StoppingReason.RESOLVED)

    # Recheck ready and blocked
    ready_ids = {n.question_id for n in dag.get_ready_nodes()}
    assert "q_auth" in ready_ids
    assert dag.get_node("q_auth").status == QuestionLifecycleStatus.READY


def test_prerequisite_failure_blocks_downstream_question():
    """Verify that if a prerequisite fails or is unobservable, downstream dependent question transitions to BLOCKED."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    q_root = _make_node("q_root")
    q_child = _make_node("q_child", dependencies=["q_root"])
    dag = dag.add_node(q_root).add_node(q_child).reevaluate_dependencies()

    assert dag.get_node("q_child").status == QuestionLifecycleStatus.BLOCKED

    # Root ends in UNOBSERVABLE
    dag = dag.transition_node("q_root", QuestionLifecycleStatus.UNOBSERVABLE, stop_reason=StoppingReason.UNOBSERVABLE)

    child_node = dag.get_node("q_child")
    assert child_node.status == QuestionLifecycleStatus.BLOCKED
    assert child_node.stop_reason == StoppingReason.DEPENDENCY_BLOCKED
    assert "ended in unobservable" in child_node.stop_rationale


def test_invalid_lifecycle_transition_rejected():
    """Verify illegal transitions (e.g. RESOLVED back to PENDING) raise ValueError."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    q1 = _make_node("q1", status=QuestionLifecycleStatus.READY)
    dag = dag.add_node(q1)

    dag = dag.transition_node("q1", QuestionLifecycleStatus.IN_PROGRESS)
    dag = dag.transition_node("q1", QuestionLifecycleStatus.RESOLVED, stop_reason=StoppingReason.RESOLVED)

    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        dag.transition_node("q1", QuestionLifecycleStatus.PENDING)


def test_explicit_reopen_with_provenance():
    """Verify resolved nodes can be reopened via explicit reopen API with audit history."""
    dag = EpistemicQuestionDAG(run_id="run_1")
    q1 = _make_node("q1", status=QuestionLifecycleStatus.READY)
    dag = dag.add_node(q1)
    dag = dag.transition_node("q1", QuestionLifecycleStatus.IN_PROGRESS)
    dag = dag.transition_node("q1", QuestionLifecycleStatus.RESOLVED, stop_reason=StoppingReason.RESOLVED)

    # Reopen
    dag = dag.reopen_node(
        question_id="q1",
        reason="Discovered recent press release stating executive departed",
        provenance="contradiction_detector:conf_12",
    )

    reopened_node = dag.get_node("q1")
    assert reopened_node.status == QuestionLifecycleStatus.READY
    assert len(reopened_node.reopen_history) == 1
    assert reopened_node.reopen_history[0]["reason"] == "Discovered recent press release stating executive departed"
    assert reopened_node.reopen_history[0]["prior_status"] == "resolved"
