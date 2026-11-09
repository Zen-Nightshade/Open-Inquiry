from __future__ import annotations

import tempfile
import pytest

from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.unit_of_work import UnitOfWork


@pytest.fixture
def test_db():
    with tempfile.NamedTemporaryFile() as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield tf.name


def test_uow_shared_connection(test_db):
    """Verify that all repositories within a UnitOfWork share the same SQLite connection."""
    with UnitOfWork(db_path=test_db) as uow:
        assert uow.outreach.conn is uow.conn
        assert uow.products.conn is uow.conn
        assert uow.research.conn is uow.conn
        assert uow.sources.conn is uow.conn
        assert uow.entities.conn is uow.conn
        assert uow.assertions.conn is uow.conn
        assert uow.jobs.conn is uow.conn


def test_uow_atomic_commit_across_repositories(test_db):
    """Verify that writes across multiple repositories commit atomically on clean context exit."""
    prospect_id = "p-100"
    version_id = "v-100"
    session_id = "s-100"

    with UnitOfWork(db_path=test_db) as uow:
        uow.outreach.create_prospect(
            prospect_id=prospect_id,
            full_name="Sarah Connor",
            company_name="Cyberdyne",
        )
        uow.products.create_product_version(
            version_id=version_id,
            product_name="Defense Net",
            snapshot_json="{}",
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id=prospect_id,
            context_id="ctx-100",
        )

    # In a new connection/transaction, all 3 entities must exist
    conn = get_connection(test_db)
    try:
        p = conn.execute("SELECT * FROM prospects WHERE prospect_id = ?;", (prospect_id,)).fetchone()
        assert p is not None
        assert p["full_name"] == "Sarah Connor"

        v = conn.execute("SELECT * FROM product_versions WHERE version_id = ?;", (version_id,)).fetchone()
        assert v is not None

        s = conn.execute("SELECT * FROM research_sessions WHERE session_id = ?;", (session_id,)).fetchone()
        assert s is not None
    finally:
        conn.close()


def test_uow_atomic_rollback_on_exception(test_db):
    """Verify that if an exception occurs inside the UoW block, all repository writes roll back."""
    prospect_id = "p-will-rollback"
    version_id = "v-will-rollback"

    with pytest.raises(RuntimeError):
        with UnitOfWork(db_path=test_db) as uow:
            uow.outreach.create_prospect(
                prospect_id=prospect_id,
                full_name="John Connor",
                company_name="Resistance",
            )
            uow.products.create_product_version(
                version_id=version_id,
                product_name="EMP Jammer",
                snapshot_json="{}",
            )
            raise RuntimeError("Simulated pipeline failure")

    # In a new connection, neither record must exist
    conn = get_connection(test_db)
    try:
        p = conn.execute("SELECT * FROM prospects WHERE prospect_id = ?;", (prospect_id,)).fetchone()
        assert p is None

        v = conn.execute("SELECT * FROM product_versions WHERE version_id = ?;", (version_id,)).fetchone()
        assert v is None
    finally:
        conn.close()


def test_uow_with_provided_connection(test_db):
    """Verify that UnitOfWork works with an externally provided connection and does not close it on exit."""
    conn = get_connection(test_db)
    prospect_id = "p-provided-conn"

    with UnitOfWork(connection=conn) as uow:
        uow.outreach.create_prospect(
            prospect_id=prospect_id,
            full_name="Kyle Reese",
            company_name="TechCom",
        )

    # Connection should still be open
    p = conn.execute("SELECT * FROM prospects WHERE prospect_id = ?;", (prospect_id,)).fetchone()
    assert p is not None
    assert p["full_name"] == "Kyle Reese"
    conn.close()
