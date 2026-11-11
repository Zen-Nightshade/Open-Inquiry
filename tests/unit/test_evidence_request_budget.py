import tempfile
import pytest

from src.db.state_store import StateStore
from src.researcher.evidence_request.executor import EvidenceRequestExecutor
from src.researcher.evidence_request.models import (
    EvidenceFeedbackBudget,
    EvidenceGapReason,
    EvidenceRequest,
    EvidenceRequestStatus,
)
from src.researcher.models import ResearchAccumulator


def _make_dummy_request(
    predicate: str = "infrastructure_requirement::kubernetes",
    question: str = "Does Acme use Kubernetes?",
    budget_cost: int = 1,
) -> EvidenceRequest:
    return EvidenceRequest(
        run_id="run_budget_test",
        target_entity_id="ent_acme",
        target_entity_name="Acme Corp",
        requested_predicate=predicate,
        neutral_question=question,
        neutral_search_terms=["Acme Kubernetes cluster"],
        disconfirming_search_terms=["Acme no Kubernetes on-premise"],
        gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
        provenance_stage="strategy_product_gate",
        budget_cost_units=budget_cost,
    )


def test_budget_cap_calculation_and_consumption():
    # 20 units baseline -> 15% is 3 units
    b20 = EvidenceFeedbackBudget.create(baseline_budget_units=20)
    assert b20.feedback_budget_cap == 3
    assert b20.can_execute_round() is True
    assert b20.can_consume(1) is True

    # 10 units baseline -> 15% is 1.5 floored to 1 unit
    b10 = EvidenceFeedbackBudget.create(baseline_budget_units=10)
    assert b10.feedback_budget_cap == 1

    # 40 units baseline -> 15% is 6 units
    b40 = EvidenceFeedbackBudget.create(baseline_budget_units=40)
    assert b40.feedback_budget_cap == 6

    # Test consumption tracking
    b = b20.with_consumption(cost_units=2)
    assert b.feedback_units_consumed == 2
    assert b.requests_executed_count == 1
    assert b.can_consume(1) is True
    assert b.can_consume(2) is False

    b_full = b.with_consumption(cost_units=1)
    assert b_full.feedback_units_consumed == 3
    assert b_full.can_consume(1) is False

    with pytest.raises(ValueError, match="Budget exhaustion"):
        b_full.with_consumption(cost_units=1)


def test_no_recursive_feedback_rounds_deferred():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        budget = EvidenceFeedbackBudget.create(baseline_budget_units=20)
        executor = EvidenceRequestExecutor(researcher=None, state_store=store)
        acc = ResearchAccumulator()

        req = _make_dummy_request()

        # Round 2 must be rejected immediately without execution
        res = executor.execute_requests(
            requests=[req],
            session_id="sess_1",
            run_id="run_1",
            budget=budget,
            accumulator=acc,
            round_number=2,
        )

        assert res.status == "deferred"
        assert len(res.requests) == 1
        assert res.requests[0].status == EvidenceRequestStatus.DEFERRED
        assert "Maximum feedback rounds exceeded" in (res.requests[0].rejection_reason or "")
        assert res.budget.rounds_executed == 0
        assert res.budget.feedback_units_consumed == 0


def test_max_three_requests_and_budget_exhaustion():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        # Cap is 2 units
        budget = EvidenceFeedbackBudget(
            baseline_budget_units=20,
            feedback_budget_cap=2,
        )
        executor = EvidenceRequestExecutor(researcher=None, state_store=store)
        acc = ResearchAccumulator()

        reqs = [
            _make_dummy_request(predicate=f"pred_{i}", question=f"Question {i}?", budget_cost=1)
            for i in range(5)
        ]

        res = executor.execute_requests(
            requests=reqs,
            session_id="sess_1",
            run_id="run_1",
            budget=budget,
            accumulator=acc,
            round_number=1,
        )

        # 5 requests provided:
        # req 0 (cost 1): executed -> status FAILED (researcher is None)
        # req 1 (cost 1): executed -> status FAILED (researcher is None)
        # req 2 (cost 1): BUDGET_EXHAUSTED (consumed 2/2)
        # req 3, 4: DEFERRED (> 3 requests per round cap)
        assert len(res.requests) == 5
        assert res.requests[0].status == EvidenceRequestStatus.FAILED
        assert res.requests[1].status == EvidenceRequestStatus.FAILED
        assert res.requests[2].status == EvidenceRequestStatus.BUDGET_EXHAUSTED
        assert res.requests[3].status == EvidenceRequestStatus.DEFERRED
        assert res.requests[4].status == EvidenceRequestStatus.DEFERRED
        assert res.budget.feedback_units_consumed == 2
        assert res.budget.rounds_executed == 1
