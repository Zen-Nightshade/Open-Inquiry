from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from repo_paths import get_db_path
from src.api.routes.health import router as health_router
from src.api.routes.reviews import router as reviews_router
from src.api.routes.runs import router as runs_router
from src.api.schemas.common import ErrorDetail, ErrorEnvelope
from src.config.settings import init_env
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup reconciliation and shutdown cleanup."""
    init_env()
    active_db_path = getattr(app.state, "db_path", None) or str(get_db_path())
    logger.info("Starting Open Inquiry API Server with database: %s", active_db_path)

    try:
        store = StateStore(db_path=active_db_path)
        orchestrator = ApplicationOrchestrator(state_store=store)
        reconciled = orchestrator.reconcile_interrupted_runs()
        if reconciled > 0:
            logger.info("Startup reconciliation: marked %d interrupted run(s) as FAILED", reconciled)
    except Exception as exc:
        logger.warning("Startup database check/reconciliation warning: %s", exc)

    yield

    logger.info("Open Inquiry API Server shut down.")


def create_app(db_path: Optional[str] = None) -> FastAPI:
    """FastAPI application factory for Open Inquiry HTTP API."""
    app = FastAPI(
        title="Open Inquiry Research & Outreach Strategy API",
        description=(
            "HTTP interface for Open Inquiry: Autonomous Evidence-Grounded Research & Outreach Strategy Engine. "
            "Enables clients to launch asynchronous prospect research, monitor stage execution, inspect "
            "canonical facts with full passage-level provenance, and retrieve strategic outreach blueprints."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store active database path on app state via centralized resolution
    app.state.db_path = str(get_db_path(db_path))

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers returning unified ErrorEnvelope
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        code_name = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorEnvelope(
                error=ErrorDetail(
                    code=code_name,
                    message=str(exc.detail),
                )
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorEnvelope(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Request payload validation failed.",
                    details=exc.errors(),
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled API exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorEnvelope(
                error=ErrorDetail(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected server error occurred during request processing.",
                )
            ).model_dump(),
        )

    # Root redirect to interactive API documentation
    @app.get("/", include_in_schema=False)
    def root_redirect():
        return RedirectResponse(url="/docs")

    # Register routers
    app.include_router(health_router)
    app.include_router(runs_router)
    app.include_router(reviews_router)

    return app
