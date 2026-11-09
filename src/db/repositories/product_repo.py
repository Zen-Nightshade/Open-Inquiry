from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ProductRepository:
    """Repository managing versioned snapshots of ProductContext."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    def create_product_version(
        self,
        version_id: str,
        product_name: str,
        snapshot_json: str,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO product_versions (version_id, product_name, snapshot_json, created_at)
        VALUES (?, ?, ?, ?);
        """
        self.conn.execute(sql, (version_id, product_name, snapshot_json, now))
        return self.get_product_version(version_id) or {}

    def get_product_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM product_versions WHERE version_id = ?;",
            (version_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_latest_version_for_product(self, product_name: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM product_versions WHERE product_name = ? ORDER BY created_at DESC LIMIT 1;",
            (product_name,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_product_versions(self, product_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if product_name:
            cursor = self.conn.execute(
                "SELECT * FROM product_versions WHERE product_name = ? ORDER BY created_at DESC;",
                (product_name,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM product_versions ORDER BY created_at DESC;"
            )
        return [dict(r) for r in cursor.fetchall()]
