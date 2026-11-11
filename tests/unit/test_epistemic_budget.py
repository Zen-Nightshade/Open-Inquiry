from __future__ import annotations

import pytest
from src.researcher.epistemic_dag.budget import GlobalResearchBudget


def test_budget_consumption_and_remaining():
    """Verify single shared budget tracks baseline consumption accurately."""
    b = GlobalResearchBudget(total_budget_units=20)
    assert b.remaining_total_units == 20
    assert b.feedback_budget_cap == 3  # floor(0.15 * 20)

    # Consume 5 baseline units
    b2 = b.consume_baseline(cost_units=5)
    assert b2.baseline_units_consumed == 5
    assert b2.feedback_units_consumed == 0
    assert b2.remaining_total_units == 15
    assert b2.remaining_feedback_units == 3


def test_feedback_cap_sublimit_within_global_pool():
    """Verify feedback research respects the 15% cap and cannot exceed global remaining units."""
    b = GlobalResearchBudget(total_budget_units=20, baseline_units_consumed=18)
    # Remaining total is 2, even though 15% cap is 3!
    assert b.remaining_total_units == 2
    assert b.remaining_feedback_units == 2  # min(2, 3) = 2

    assert b.can_consume_feedback(2) is True
    assert b.can_consume_feedback(3) is False  # Cannot exceed remaining total budget


def test_no_budget_reset_across_phases():
    """Verify expenditure accumulates without resets."""
    b = GlobalResearchBudget(total_budget_units=20)
    b_baseline = b.consume_baseline(10)
    b_feedback = b_baseline.consume_feedback(2)

    assert b_feedback.total_units_consumed == 12
    assert b_feedback.baseline_units_consumed == 10
    assert b_feedback.feedback_units_consumed == 2
    assert b_feedback.remaining_total_units == 8


def test_consumption_past_exhaustion_raises_error():
    """Verify consuming past remaining budget raises ValueError."""
    b = GlobalResearchBudget(total_budget_units=5, baseline_units_consumed=5)
    assert b.is_exhausted is True

    with pytest.raises(ValueError, match="Global budget exhaustion"):
        b.consume_baseline(1)

    with pytest.raises(ValueError, match="Budget violation"):
        b.consume_feedback(1)
