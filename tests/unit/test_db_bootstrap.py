from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest

from repo_paths import DATA_DIR, REPO_ROOT, get_db_path
from src.db.connection import get_connection
from src.db.schema import EXPECTED_DOMAIN_TABLES, init_schema, verify_schema
from src.db.state_store import StateStore


def test_state_store_bootstrap_and_database_status():
    with tempfile.TemporaryDirectory() as td:
        db_file = Path(td) / "test.db"
        store = StateStore(db_file)

        # Before initialization - file does not exist
        status_before = store.get_database_status()
        assert status_before["file_exists"] is False
        assert status_before["is_initialized"] is False
        assert status_before["is_schema_valid"] is False
        assert status_before["expected_tables_found"] == 0
        assert not db_file.exists()

        # Initialize
        store.initialize()
        assert db_file.exists()

        # After initialization
        status_after = store.get_database_status()
        assert status_after["file_exists"] is True
        assert status_after["is_initialized"] is True
        assert status_after["is_schema_valid"] is True
        assert status_after["expected_tables_found"] == 36
        assert status_after["total_expected_tables"] == 36
        assert status_after["table_count"] == 36
        assert "application_runs" in status_after["tables"]
        assert "migration_ledger" not in status_after["tables"]


def test_runtime_open_inquiry_db_path_environment_override(monkeypatch):
    with tempfile.NamedTemporaryFile() as tf:
        custom_path = str(Path(tf.name).resolve())
        monkeypatch.setenv("OPEN_INQUIRY_DB_PATH", custom_path)

        # Verify get_db_path resolves the environment variable
        resolved_path = get_db_path()
        assert str(resolved_path) == custom_path

        # Verify StateStore without explicit db_path uses the env override
        store = StateStore()
        assert str(store.db_path) == custom_path

        # Verify connection and schema initialization apply to the custom path
        store.initialize()
        status = store.get_database_status()
        assert status["db_path"] == custom_path
        assert status["is_initialized"] is True
        assert status["is_schema_valid"] is True
        assert status["expected_tables_found"] == 36


def test_runtime_legacy_zamp_db_path_fallback(monkeypatch):
    with tempfile.NamedTemporaryFile() as tf:
        custom_path = str(Path(tf.name).resolve())
        monkeypatch.delenv("OPEN_INQUIRY_DB_PATH", raising=False)
        monkeypatch.setenv("ZAMP_DB_PATH", custom_path)

        with pytest.deprecated_call():
            resolved_path = get_db_path()
        assert str(resolved_path) == custom_path

        with pytest.deprecated_call():
            store = StateStore()
        assert str(store.db_path) == custom_path
