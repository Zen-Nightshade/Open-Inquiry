from __future__ import annotations

import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator


def test_get_run_stages():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_stages.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        orch = ApplicationOrchestrator(state_store=store)
        p = ProspectContext(full_name="Elena Rostova", company_name="Vanguard AI")
        prod = ProductContext(
            product_name="ToolX",
            description="A tool",
            capabilities=["Fast sync"],
            value_proposition="Save time",
        )
        out = OutreachContext(
            objective="Intro",
            sender_identity="Seller",
        )
        run_snap = orch.create_pending_run(prospect=p, product=prod, outreach=out, run_id="run-stage-test")

        app = create_app(db_path=db_path)
        client = TestClient(app)

        res = client.get(f"/api/v1/runs/{run_snap.run_id}/stages")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_snap.run_id
        assert data["total_stages"] >= 1
        assert data["stages"][0]["stage"] == "validation"
        assert data["stages"][0]["status"] == "completed"

        # 404 for nonexistent run
        res_404 = client.get("/api/v1/runs/does-not-exist/stages")
        assert res_404.status_code == 404
        assert res_404.json()["error"]["code"] == "NOT_FOUND"
