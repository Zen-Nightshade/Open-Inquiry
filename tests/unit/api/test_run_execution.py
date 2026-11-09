from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator
from src.orchestration.models import RunStatus


def test_background_run_execution():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_exec.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)

        # Mock the heavy run method to verify the background task is properly invoked and dispatches to orchestrator.run
        with patch.object(ApplicationOrchestrator, "run") as mock_run:
            mock_run.return_value = MagicMock()

            # TestClient executes FastAPI BackgroundTasks before completing the request block
            with TestClient(app) as client:
                res = client.post(
                    "/api/v1/runs",
                    json={
                        "prospect": {"full_name": "Elena Rostova", "company_name": "Vanguard AI"},
                        "product": {
                            "product_name": "ComputeScale",
                            "description": "Platform",
                            "capabilities": ["Fault recovery"],
                            "value_proposition": "Cut costs",
                        },
                        "outreach": {"objective": "Call", "sender_identity": "Alex"},
                    },
                )
                assert res.status_code == 202
                run_id = res.json()["run_id"]

            assert mock_run.called
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["run_id"] == run_id
            assert call_kwargs["prospect"].full_name == "Elena Rostova"


def test_startup_reconciliation_of_interrupted_runs():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_reconcile.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        orch = ApplicationOrchestrator(state_store=store)
        p = ProspectContext(full_name="Interrupted Prospect", company_name="Ghost Corp")
        prod = ProductContext(
            product_name="Tool",
            description="Desc",
            capabilities=["Cap"],
            value_proposition="Val",
        )
        out = OutreachContext(objective="Obj", sender_identity="Sender")
        run_snap = orch.create_pending_run(prospect=p, product=prod, outreach=out, run_id="run-interrupted-1")

        # Manually transition run into RUNNING state to simulate a crash during execution
        store.update_run(run_id="run-interrupted-1", status=RunStatus.RUNNING.value)
        assert store.get_run("run-interrupted-1")["status"] == "running"

        # Now start up the FastAPI application (lifespan runs)
        app = create_app(db_path=db_path)
        with TestClient(app) as client:
            res = client.get("/api/v1/runs/run-interrupted-1")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "failed"
            assert "interrupted by server shutdown or restart" in data["error_message"]
