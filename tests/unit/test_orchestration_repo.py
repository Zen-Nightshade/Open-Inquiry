from __future__ import annotations

import tempfile
import sqlite3
import pytest

from src.db.connection import get_connection
from src.db.repositories.orchestration_repo import OrchestrationRepository
from src.db.schema import init_schema
from src.db.unit_of_work import UnitOfWork


@pytest.fixture
def db_conn():
    with tempfile.NamedTemporaryFile() as tf:
        conn = get_connection(tf.name)
        init_schema(conn)

        # Seed prerequisite prospect and session
        conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Alice', 'Acme', '2026-11-07T00:00:00Z');")
        conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s1', 'p1', 'c1', '2026-11-07T00:00:00Z');")
        conn.commit()

        yield conn
        conn.close()


def test_iteration_lifecycle_and_queries(db_conn):
    repo = OrchestrationRepository(db_conn)

    # 1. Create iteration 1
    it1 = repo.create_iteration(
        iteration_id="iter-1",
        session_id="s1",
        iteration_number=1,
        status="in_progress",
    )
    assert it1["iteration_id"] == "iter-1"
    assert it1["iteration_number"] == 1
    assert it1["status"] == "in_progress"

    # 2. Get iteration
    assert repo.get_iteration("iter-1")["iteration_id"] == "iter-1"
    assert repo.get_iteration_by_number("s1", 1)["iteration_id"] == "iter-1"
    assert repo.get_iteration_by_number("s1", 2) is None

    # 3. Update iteration metrics
    updated = repo.update_iteration(
        iteration_id="iter-1",
        status="completed",
        queries_executed=3,
        snapshots_fetched=2,
        facts_extracted=5,
        conflicts_detected=1,
        completed_at="2026-11-07T00:05:00Z",
    )
    assert updated is True

    it1_updated = repo.get_iteration("iter-1")
    assert it1_updated["status"] == "completed"
    assert it1_updated["queries_executed"] == 3
    assert it1_updated["facts_extracted"] == 5

    # 4. Create iteration 2
    repo.create_iteration(
        iteration_id="iter-2",
        session_id="s1",
        iteration_number=2,
        status="in_progress",
    )

    latest = repo.get_latest_iteration("s1")
    assert latest["iteration_id"] == "iter-2"
    assert latest["iteration_number"] == 2

    iterations = repo.list_iterations("s1")
    assert len(iterations) == 2
    assert [i["iteration_number"] for i in iterations] == [1, 2]


def test_iteration_constraints(db_conn):
    repo = OrchestrationRepository(db_conn)
    repo.create_iteration(iteration_id="iter-1", session_id="s1", iteration_number=1)

    # Duplicate iteration number for same session violates UNIQUE constraint
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_iteration(iteration_id="iter-dup", session_id="s1", iteration_number=1)

    # Invalid status violates CHECK constraint
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_iteration(iteration_id="iter-bad", session_id="s1", iteration_number=2, status="invalid_status")


def test_assessment_crud_and_integrity(db_conn):
    repo = OrchestrationRepository(db_conn)
    repo.create_iteration(iteration_id="iter-1", session_id="s1", iteration_number=1, status="completed")

    # 1. Create assessment
    ass1 = repo.create_assessment(
        assessment_id="ass-1",
        session_id="s1",
        iteration_id="iter-1",
        decision="continue",
        coverage_status="partial",
        identity_status="confirmed",
        evidence_status="adequate",
        marginal_value_status="viable",
        explanation="Need deeper corroboration on product deployment.",
        objective_details_json={"obj1": "partial"},
        remaining_gaps_json=["gap1"],
        recommended_actions_json=[{"action": "search"}],
        dimension_details_json={"coverage": "partial"},
    )
    assert ass1["assessment_id"] == "ass-1"
    assert ass1["decision"] == "continue"

    # 2. Get assessment queries
    assert repo.get_assessment("ass-1")["assessment_id"] == "ass-1"
    assert repo.get_assessment_for_iteration("iter-1")["assessment_id"] == "ass-1"
    assert repo.get_latest_assessment("s1")["assessment_id"] == "ass-1"

    # 3. Exactly one assessment per iteration (UNIQUE constraint on iteration_id)
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_assessment(
            assessment_id="ass-dup",
            session_id="s1",
            iteration_id="iter-1",
            decision="sufficient",
            coverage_status="complete",
            identity_status="confirmed",
            evidence_status="adequate",
            marginal_value_status="diminished",
            explanation="Dup",
            objective_details_json={},
            remaining_gaps_json=[],
            recommended_actions_json=[],
            dimension_details_json={},
        )

    # 4. Invalid decision violates CHECK constraint
    repo.create_iteration(iteration_id="iter-2", session_id="s1", iteration_number=2)
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_assessment(
            assessment_id="ass-bad",
            session_id="s1",
            iteration_id="iter-2",
            decision="invalid_decision",
            coverage_status="complete",
            identity_status="confirmed",
            evidence_status="adequate",
            marginal_value_status="diminished",
            explanation="Bad decision",
            objective_details_json={},
            remaining_gaps_json=[],
            recommended_actions_json=[],
            dimension_details_json={},
        )


def test_unit_of_work_orchestration_transaction(db_conn):
    # Verify UnitOfWork exposes orchestration and commits/rolls back atomically
    with tempfile.NamedTemporaryFile() as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.execute("INSERT INTO prospects (prospect_id, full_name, company_name, created_at) VALUES ('p1', 'Alice', 'Acme', '2026-11-07T00:00:00Z');")
        conn.execute("INSERT INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES ('s1', 'p1', 'c1', '2026-11-07T00:00:00Z');")
        conn.commit()

        # Success commit test
        with UnitOfWork(conn) as uow:
            uow.orchestration.create_iteration(iteration_id="it-uow-1", session_id="s1", iteration_number=1)

        row = conn.execute("SELECT * FROM research_iterations WHERE iteration_id = 'it-uow-1';").fetchone()
        assert row is not None

        # Rollback on exception test
        with pytest.raises(ValueError):
            with UnitOfWork(conn) as uow:
                uow.orchestration.create_iteration(iteration_id="it-uow-2", session_id="s1", iteration_number=2)
                raise ValueError("Simulated failure")

        row2 = conn.execute("SELECT * FROM research_iterations WHERE iteration_id = 'it-uow-2';").fetchone()
        assert row2 is None

        conn.close()
