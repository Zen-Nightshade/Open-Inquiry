from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.db.backup import backup_database, restore_database
from src.db.connection import get_connection
from src.db.schema import init_schema, verify_schema
from src.db.unit_of_work import UnitOfWork


def test_online_backup_and_restore():
    """Verify that an online backup captures schema and data, and can be restored cleanly."""
    with tempfile.TemporaryDirectory() as td:
        dir_path = Path(td)
        original_db = dir_path / "orig.db"
        backup_file = dir_path / "backup.db"
        restored_db = dir_path / "restored.db"

        # 1. Initialize original DB and populate records
        conn = get_connection(original_db)
        init_schema(conn)

        with UnitOfWork(connection=conn) as uow:
            uow.outreach.create_prospect(
                prospect_id="p-backup-1",
                full_name="Ada Lovelace",
                company_name="Babbage Analytical",
                company_domain="analytical.org",
            )
            uow.products.create_product_version(
                version_id="pv-backup-1",
                product_name="Analytical Engine",
                snapshot_json='{"version": 1}',
            )

        # 2. Perform hot online backup while connection is open
        backup_database(conn, backup_file)
        conn.close()

        assert backup_file.exists()
        assert backup_file.stat().st_size > 0

        # Verify backup database content
        backup_conn = get_connection(backup_file)
        assert verify_schema(backup_conn)["is_valid"] is True
        p = backup_conn.execute(
            "SELECT * FROM prospects WHERE prospect_id = 'p-backup-1';"
        ).fetchone()
        assert p is not None
        assert p["full_name"] == "Ada Lovelace"
        backup_conn.close()

        # 3. Restore to a new location
        restore_database(backup_file, restored_db)
        assert restored_db.exists()

        restored_conn = get_connection(restored_db)
        assert verify_schema(restored_conn)["is_valid"] is True
        restored_p = restored_conn.execute(
            "SELECT * FROM prospects WHERE prospect_id = 'p-backup-1';"
        ).fetchone()
        assert restored_p is not None
        assert restored_p["full_name"] == "Ada Lovelace"
        restored_conn.close()


def test_restore_missing_file_raises():
    with tempfile.TemporaryDirectory() as td:
        missing_backup = Path(td) / "does_not_exist.db"
        target = Path(td) / "target.db"

        with pytest.raises(FileNotFoundError):
            restore_database(missing_backup, target)
