from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def backup_database(
    source: Union[sqlite3.Connection, PathLike],
    target_path: PathLike,
    pages: int = -1,
) -> None:
    """Create a hot online backup of a SQLite database using sqlite3.Connection.backup.

    This uses SQLite's official backup API rather than file copying, ensuring
    read consistency even when WAL mode or concurrent readers are active.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    owns_source = not isinstance(source, sqlite3.Connection)
    source_conn = sqlite3.connect(str(source)) if owns_source else source

    dest_conn = sqlite3.connect(str(target))
    try:
        source_conn.backup(dest_conn, pages=pages)
    finally:
        dest_conn.close()
        if owns_source:
            source_conn.close()


def restore_database(
    backup_path: PathLike,
    target_path: PathLike,
    pages: int = -1,
) -> None:
    """Restore a database from an existing backup file to a target location."""
    backup = Path(backup_path).resolve()
    if not backup.exists():
        raise FileNotFoundError(f"Backup file not found at: {backup}")

    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(backup))
    dest_conn = sqlite3.connect(str(target))
    try:
        src_conn.backup(dest_conn, pages=pages)
    finally:
        src_conn.close()
        dest_conn.close()
