from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from repo_paths import REPO_ROOT, get_db_path
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.connection import get_connection
from src.db.schema import (
    EXPECTED_DOMAIN_TABLES,
    init_schema,
    verify_schema,
)
from src.db.unit_of_work import UnitOfWork

PathLike = Union[str, Path]


class StateStore:
    """Composition facade providing high-level coordinated database operations.

    Ensures that every operation runs within explicit, short UnitOfWork transactions.
    """

    def __init__(self, db_path: Optional[PathLike] = None) -> None:
        resolved = get_db_path(db_path)
        self.db_path = ":memory:" if str(resolved) == ":memory:" else resolved

    def initialize(self) -> None:
        """Initialize database schema directly from schema.sql. Idempotent."""
        conn = get_connection(self.db_path)
        try:
            init_schema(conn)
        finally:
            conn.close()

    def get_database_status(self) -> Dict[str, Any]:
        """Inspect the database status without creating an empty database file if absent."""
        is_memory = str(self.db_path) == ":memory:"
        if not is_memory:
            p = Path(self.db_path)
            if not p.exists():
                return {
                    "db_path": str(self.db_path),
                    "file_exists": False,
                    "is_initialized": False,
                    "table_count": 0,
                    "expected_tables_found": 0,
                    "total_expected_tables": len(EXPECTED_DOMAIN_TABLES),
                    "is_schema_valid": False,
                    "missing_tables": sorted(list(EXPECTED_DOMAIN_TABLES)),
                    "incompatibility_details": "Database file does not exist",
                    "tables": [],
                }

        conn = get_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            )
            tables = [row["name"] for row in cursor.fetchall()]

            if not tables:
                return {
                    "db_path": str(self.db_path),
                    "file_exists": True,
                    "is_initialized": False,
                    "table_count": 0,
                    "expected_tables_found": 0,
                    "total_expected_tables": len(EXPECTED_DOMAIN_TABLES),
                    "is_schema_valid": False,
                    "missing_tables": sorted(list(EXPECTED_DOMAIN_TABLES)),
                    "incompatibility_details": "Database file is empty (0 tables)",
                    "tables": [],
                }

            verification = verify_schema(conn)
            is_valid = verification["is_valid"]
            missing_tables = verification["missing_tables"]
            expected_found = len(EXPECTED_DOMAIN_TABLES) - len(missing_tables)

            incompatibility_details = None
            if not is_valid:
                incompatibility_details = f"Missing {len(missing_tables)} expected domain tables: {missing_tables}"

            return {
                "db_path": str(self.db_path),
                "file_exists": True,
                "is_initialized": expected_found > 0,
                "table_count": len(tables),
                "expected_tables_found": expected_found,
                "total_expected_tables": len(EXPECTED_DOMAIN_TABLES),
                "is_schema_valid": is_valid,
                "missing_tables": missing_tables,
                "incompatibility_details": incompatibility_details,
                "tables": tables,
            }
        finally:
            conn.close()

    def uow(self) -> UnitOfWork:
        """Provide a UnitOfWork instance bound to this store's database."""
        return UnitOfWork(db_path=self.db_path)


    def save_prospect(
        self,
        prospect: ProspectContext,
        prospect_id: Optional[str] = None,
    ) -> str:
        """Save a ProspectContext domain model to the prospects table."""
        pid = prospect_id or str(uuid.uuid4())
        with self.uow() as uow:
            uow.outreach.create_prospect(
                prospect_id=pid,
                full_name=prospect.full_name,
                company_name=prospect.company_name,
                company_domain=prospect.company_domain,
                designation=prospect.designation,
                linkedin_url=prospect.linkedin_url,
                location=prospect.location,
            )
        return pid

    def get_prospect(self, prospect_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a prospect by ID."""
        with self.uow() as uow:
            return uow.outreach.get_prospect(prospect_id)

    def save_product_context(
        self,
        product: ProductContext,
        version_id: Optional[str] = None,
    ) -> str:
        """Save a snapshot of ProductContext to the product_versions table."""
        vid = version_id or str(uuid.uuid4())
        snapshot_json = product.model_dump_json()
        with self.uow() as uow:
            uow.products.create_product_version(
                version_id=vid,
                product_name=product.product_name,
                snapshot_json=snapshot_json,
            )
        return vid

    def get_product_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a product version snapshot by ID."""
        with self.uow() as uow:
            return uow.products.get_product_version(version_id)

    def get_product_context(self, version_id: str) -> Optional[ProductContext]:
        """Retrieve and deserialize a ProductContext (including structured contract) by version ID."""
        row = self.get_product_version(version_id)
        if not row or not row.get("snapshot_json"):
            return None
        return ProductContext.model_validate_json(row["snapshot_json"])

    def get_latest_version_for_product(self, product_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest product version snapshot by product name."""
        with self.uow() as uow:
            return uow.products.get_latest_version_for_product(product_name)

    def save_outreach_context(
        self,
        outreach: OutreachContext,
        prospect_id: str,
        version_id: str,
        context_id: Optional[str] = None,
    ) -> str:
        """Save an OutreachContext instance linked to prospect and product version."""
        cid = context_id or str(uuid.uuid4())
        context_json = outreach.model_dump_json()
        with self.uow() as uow:
            uow.outreach.create_outreach_context(
                context_id=cid,
                prospect_id=prospect_id,
                version_id=version_id,
                context_json=context_json,
            )
        return cid

    def get_outreach_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an outreach context by ID."""
        with self.uow() as uow:
            return uow.outreach.get_outreach_context(context_id)

    def create_run(
        self,
        run_id: str,
        prospect_id: str,
        prospect_snapshot_json: str,
        product_snapshot_json: str,
        outreach_snapshot_json: str,
        stage_statuses_json: str,
        status: str = "pending",
        job_id: Optional[str] = None,
        context_id: Optional[str] = None,
        research_session_id: Optional[str] = None,
        strategy_result_json: Optional[str] = None,
        warnings_json: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a new application run."""
        with self.uow() as uow:
            return uow.runs.create_run(
                run_id=run_id,
                prospect_id=prospect_id,
                prospect_snapshot_json=prospect_snapshot_json,
                product_snapshot_json=product_snapshot_json,
                outreach_snapshot_json=outreach_snapshot_json,
                stage_statuses_json=stage_statuses_json,
                status=status,
                job_id=job_id,
                context_id=context_id,
                research_session_id=research_session_id,
                strategy_result_json=strategy_result_json,
                warnings_json=warnings_json,
                error_message=error_message,
                started_at=started_at,
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an application run by ID."""
        with self.uow() as uow:
            return uow.runs.get_run(run_id)

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        research_session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        strategy_result_json: Optional[str] = None,
        stage_statuses_json: Optional[str] = None,
        warnings_json: Optional[str] = None,
        error_message: Optional[str] = None,
        completed_at: Optional[str] = None,
        draft_id: Optional[str] = None,
    ) -> bool:
        """Update an application run."""
        with self.uow() as uow:
            return uow.runs.update_run(
                run_id=run_id,
                status=status,
                research_session_id=research_session_id,
                context_id=context_id,
                strategy_result_json=strategy_result_json,
                stage_statuses_json=stage_statuses_json,
                warnings_json=warnings_json,
                error_message=error_message,
                completed_at=completed_at,
                draft_id=draft_id,
            )

    def list_runs(
        self,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List application runs ordered by start time descending."""
        with self.uow() as uow:
            return uow.runs.list_runs(limit=limit, status=status)

    def record_model_execution(self, **kwargs) -> Dict[str, Any]:
        """Record a model execution record."""
        with self.uow() as uow:
            return uow.jobs.record_model_execution(**kwargs)

    def get_model_executions(
        self,
        job_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve model executions for a job or run."""
        with self.uow() as uow:
            return uow.jobs.get_model_executions(job_id=job_id, run_id=run_id)
