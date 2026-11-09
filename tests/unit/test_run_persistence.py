from __future__ import annotations

import tempfile
import uuid
import pytest

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.connection import get_connection
from src.db.state_store import StateStore
from src.db.unit_of_work import UnitOfWork


def test_run_repository_crud_and_lifecycle():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        # Create prospect first
        prospect = ProspectContext(
            full_name="Sarah Connor",
            company_name="Cyberdyne Systems",
            company_domain="cyberdyne.com",
            designation="Director of Security",
        )
        prospect_id = store.save_prospect(prospect)

        # Create Run
        run_id = str(uuid.uuid4())
        run_row = store.create_run(
            run_id=run_id,
            prospect_id=prospect_id,
            prospect_snapshot_json=prospect.model_dump_json(),
            product_snapshot_json="{}",
            outreach_snapshot_json="{}",
            stage_statuses_json='{"validation": "completed"}',
            status="running",
        )

        assert run_row["run_id"] == run_id
        assert run_row["status"] == "running"
        assert run_row["prospect_id"] == prospect_id

        # Update Run
        updated = store.update_run(
            run_id=run_id,
            status="completed",
            strategy_result_json='{"is_viable": true}',
            stage_statuses_json='{"validation": "completed", "research": "completed"}',
            warnings_json='["Warning test"]',
            completed_at="2026-11-07T20:00:00Z",
        )
        assert updated is True

        # Retrieve Run
        retrieved = store.get_run(run_id)
        assert retrieved is not None
        assert retrieved["status"] == "completed"
        assert retrieved["strategy_result_json"] == '{"is_viable": true}'
        assert "Warning test" in retrieved["warnings_json"]

        # List Runs
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id

        filtered = store.list_runs(status="failed")
        assert len(filtered) == 0


def test_run_repository_foreign_key_enforcement():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(tf.name)
        store.initialize()

        # Attempt to create a run with non-existent prospect_id
        with pytest.raises(Exception):
            store.create_run(
                run_id=str(uuid.uuid4()),
                prospect_id="non-existent-prospect-id",
                prospect_snapshot_json="{}",
                product_snapshot_json="{}",
                outreach_snapshot_json="{}",
                stage_statuses_json="{}",
            )
