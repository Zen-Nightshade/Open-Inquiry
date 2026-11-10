from __future__ import annotations

from typing import Optional
from fastapi import Depends, Request

from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator


def get_state_store(request: Request) -> StateStore:
    """Dependency returning a StateStore initialized with the app's configured db_path."""
    db_path: Optional[str] = getattr(request.app.state, "db_path", None)
    return StateStore(db_path=db_path)


def get_orchestrator(store: StateStore = Depends(get_state_store)) -> ApplicationOrchestrator:
    """Dependency returning an ApplicationOrchestrator instance."""
    return ApplicationOrchestrator(state_store=store)


def get_review_service(store: StateStore = Depends(get_state_store)) -> Any:
    """Dependency returning a ReviewService instance."""
    from src.email_generation.review_service import ReviewService
    return ReviewService(state_store=store)

