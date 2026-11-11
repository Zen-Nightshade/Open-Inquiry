from __future__ import annotations

import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.db.state_store import StateStore


def test_health_uninitialized_database():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "uninit.db")
        app = create_app(db_path=db_path)
        client = TestClient(app)

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"]["file_exists"] is False
        assert "providers" in data


def test_health_initialized_database():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "healthy.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"]["file_exists"] is True
        assert data["database"]["is_initialized"] is True
        assert data["database"]["is_schema_valid"] is True
        assert data["version"] == "1.0.0"


def test_api_metadata_branding():
    app = create_app()
    assert app.title == "Open Inquiry Research & Outreach Strategy API"
    assert "Open Inquiry" in app.description
    client = TestClient(app)
    res = client.get("/openapi.json")
    assert res.status_code == 200
    openapi = res.json()
    assert openapi["info"]["title"] == "Open Inquiry Research & Outreach Strategy API"

