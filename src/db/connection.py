from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

from repo_paths import REPO_ROOT, get_db_path

PathLike = Union[str, Path]


def get_connection(
    db_path: Optional[PathLike] = None,
    timeout: float = 5.0,
) -> sqlite3.Connection:
    """Create and configure a SQLite connection following Open Inquiry Phase B architecture.

    Configured PRAGMAs:
      - foreign_keys = ON (enforce referential integrity)
      - busy_timeout = 5000 (minimum 5s lock wait to prevent SQLITE_BUSY deadlocks)
      - synchronous = NORMAL (safe performance mode for WAL)
      - journal_mode = WAL (Write-Ahead Logging for non-memory databases)
    """
    target_path = get_db_path(db_path)

    is_memory = str(target_path) == ":memory:"

    if not is_memory:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target_path), timeout=timeout)
    else:
        conn = sqlite3.connect(":memory:", timeout=timeout)

    conn.row_factory = sqlite3.Row

    # Enforce foreign keys and timeout PRAGMAs
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
    conn.execute("PRAGMA synchronous = NORMAL;")

    if not is_memory:
        conn.execute("PRAGMA journal_mode = WAL;")

    return conn
