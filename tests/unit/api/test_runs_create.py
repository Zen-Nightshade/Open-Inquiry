from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.db.state_store import StateStore


def _sample_payload():
    return {
        "prospect": {
            "full_name": "Elena Rostova",
            "company_name": "Vanguard AI",
            "company_domain": "vanguard.ai",
            "designation": "Head of Research",
            "linkedin_url": "https://linkedin.com/in/elena-rostova",
            "location": "San Francisco, CA",
        },
        "product": {
            "product_name": "ComputeScale",
            "description": "High throughput distributed GPU clustering platform",
            "capabilities": ["Distributed checkpointing", "Dynamic gang scheduling"],
            "value_proposition": "Cut model training costs by 40%",
            "icp": "High-growth AI labs",
            "differentiators": ["Near-zero overhead fault recovery"],
            "limitations": ["Requires Kubernetes"],
        },
        "outreach": {
            "objective": "Schedule exploratory call",
            "sender_identity": "Alex Chen, Founder at ComputeScale",
            "sender_name": "Alex Chen",
            "sender_company": "ComputeScale",
            "desired_cta": "Grab 15 minutes next Tuesday",
            "tone": "direct and technical",
            "relationship": "cold",
            "constraints": ["Keep under 120 words"],
        },
    }


def test_create_run_success():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_create.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        with patch("src.api.routes.runs._background_execute_run") as mock_bg:
            response = client.post("/api/v1/runs", json=_sample_payload())
            assert response.status_code == 202
            data = response.json()
            assert "run_id" in data
            assert data["status"] == "pending"
            assert "Executing" in data["message"]
            
            # Verify hypermedia links
            rels = {link["rel"]: link["href"] for link in data["links"]}
            assert "self" in rels
            assert "stages" in rels
            assert "research" in rels
            assert "strategy" in rels
            assert data["run_id"] in rels["self"]

            # Verify run exists in database in pending status
            saved_run = store.get_run(data["run_id"])
            assert saved_run is not None
            assert saved_run["status"] == "pending"


def test_create_run_missing_prospect_fields():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_create.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        bad_payload = _sample_payload()
        del bad_payload["prospect"]["full_name"]

        response = client.post("/api/v1/runs", json=bad_payload)
        assert response.status_code == 422
        err = response.json()
        assert "error" in err
        assert err["error"]["code"] == "VALIDATION_ERROR"


def test_create_run_empty_required_field():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_create.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        bad_payload = _sample_payload()
        bad_payload["prospect"]["full_name"] = ""

        response = client.post("/api/v1/runs", json=bad_payload)
        assert response.status_code in (400, 422)
        err = response.json()
        assert "error" in err
