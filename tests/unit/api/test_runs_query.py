from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator


def _seed_runs(store: StateStore):
    orch = ApplicationOrchestrator(state_store=store)
    p1 = ProspectContext(full_name="Elena Rostova", company_name="Vanguard AI")
    p2 = ProspectContext(full_name="John Doe", company_name="Acme Corp")
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
    r1 = orch.create_pending_run(prospect=p1, product=prod, outreach=out, run_id="run-1111")
    r2 = orch.create_pending_run(prospect=p2, product=prod, outreach=out, run_id="run-2222")
    return r1, r2


def test_list_runs():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_query.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        _seed_runs(store)

        app = create_app(db_path=db_path)
        client = TestClient(app)

        res = client.get("/api/v1/runs")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2
        assert len(data["runs"]) == 2

        # Filter by company
        res_comp = client.get("/api/v1/runs?company=vanguard")
        assert res_comp.status_code == 200
        data_comp = res_comp.json()
        assert data_comp["total"] == 1
        assert data_comp["runs"][0]["company_name"] == "Vanguard AI"

        # Filter by status
        res_stat = client.get("/api/v1/runs?status=pending")
        assert res_stat.status_code == 200
        assert res_stat.json()["total"] == 2

        res_none = client.get("/api/v1/runs?status=completed")
        assert res_none.status_code == 200
        assert res_none.json()["total"] == 0


def test_get_run_detail_and_not_found():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_detail.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        r1, _ = _seed_runs(store)

        app = create_app(db_path=db_path)
        client = TestClient(app)

        # Success detail
        res = client.get(f"/api/v1/runs/{r1.run_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == r1.run_id
        assert data["status"] == "pending"
        assert data["prospect"]["full_name"] == "Elena Rostova"
        assert data["product"]["product_name"] == "ToolX"
        assert len(data["links"]) > 0

        # Not found
        res_nf = client.get("/api/v1/runs/nonexistent-id")
        assert res_nf.status_code == 404
        err = res_nf.json()
        assert "error" in err
        assert err["error"]["code"] == "NOT_FOUND"
