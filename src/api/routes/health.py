from __future__ import annotations

import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from src.api.dependencies import get_state_store
from src.api.schemas.common import HealthResponse
from src.db.state_store import StateStore

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Check the readiness of the database, active schema validity, and external API credentials.",
)
def get_health(store: StateStore = Depends(get_state_store)) -> HealthResponse:
    db_status = store.get_database_status()
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    
    is_healthy = db_status.get("file_exists", False) and db_status.get("is_initialized", False) and db_status.get("is_schema_valid", False)
    status_str = "healthy" if is_healthy else "degraded"

    return HealthResponse(
        status=status_str,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        database=db_status,
        providers={
            "groq": has_groq,
        },
    )
