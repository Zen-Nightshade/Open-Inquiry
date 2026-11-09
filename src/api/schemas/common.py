from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Any] = Field(None, description="Optional diagnostic details or validation errors")


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class LinkItem(BaseModel):
    href: str = Field(..., description="Target URI")
    rel: str = Field(..., description="Link relationship name")
    method: str = Field("GET", description="HTTP method for interacting with link")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status ('healthy' or 'degraded')")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    version: str = Field("1.0.0", description="API version")
    database: Dict[str, Any] = Field(..., description="Database connection and table diagnostics")
    providers: Dict[str, bool] = Field(..., description="External API credentials availability")
