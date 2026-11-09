from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

from src.db.connection import get_connection
from src.db.repositories.assertion_repo import AssertionRepository
from src.db.repositories.entity_repo import EntityRepository
from src.db.repositories.job_repo import JobRepository
from src.db.repositories.orchestration_repo import OrchestrationRepository
from src.db.repositories.outreach_repo import OutreachRepository
from src.db.repositories.product_repo import ProductRepository
from src.db.repositories.research_repo import ResearchRepository
from src.db.repositories.run_repo import RunRepository
from src.db.repositories.source_repo import SourceRepository

PathLike = Union[str, Path]


class UnitOfWork:
    """Manages transactional boundaries across domain repositories sharing a single SQLite connection.

    Rules:
      - Repositories never commit or rollback independently.
      - UnitOfWork begins an immediate transaction on entry and commits on clean exit.
      - Any unhandled exception causes an immediate rollback.
      - Transactions must never be held open across external I/O (network/LLM).
    """

    def __init__(
        self,
        connection: Optional[sqlite3.Connection] = None,
        db_path: Optional[PathLike] = None,
    ) -> None:
        self._owns_connection = connection is None
        self.conn = connection if connection is not None else get_connection(db_path)

        # Repositories share this connection and transaction boundary
        self.research = ResearchRepository(self.conn)
        self.sources = SourceRepository(self.conn)
        self.entities = EntityRepository(self.conn)
        self.assertions = AssertionRepository(self.conn)
        self.outreach = OutreachRepository(self.conn)
        self.products = ProductRepository(self.conn)
        self.jobs = JobRepository(self.conn)
        self.orchestration = OrchestrationRepository(self.conn)
        self.runs = RunRepository(self.conn)


    def __enter__(self) -> UnitOfWork:
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN IMMEDIATE;")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()
        finally:
            if self._owns_connection:
                self.conn.close()

    def commit(self) -> None:
        """Manually commit the active transaction."""
        self.conn.commit()

    def rollback(self) -> None:
        """Manually roll back the active transaction."""
        self.conn.rollback()
