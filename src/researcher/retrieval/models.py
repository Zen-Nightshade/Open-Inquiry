from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RetrievedRecordType(str, Enum):
    """Classification of the indexed historical record type."""

    SOURCE_SNAPSHOT = "source_snapshot"
    EVIDENCE_RECORD = "evidence_record"


class RetrievedItem(BaseModel):
    """Immutable result item returned from historical lexical search."""

    model_config = ConfigDict(frozen=True)

    record_type: RetrievedRecordType
    record_id: str = Field(description="Unique identifier: snapshot_id or evidence_id")
    source_id: str = Field(description="Associated source ID")
    session_id: Optional[str] = Field(default=None, description="Research session where record was gathered")
    run_id: Optional[str] = Field(default=None, description="Application run ID if linked to a run")
    project_id: Optional[str] = Field(default=None, description="Project/prospect ID for tenant/project isolation")
    title: Optional[str] = Field(default=None, description="Source or document title")
    snippet: str = Field(description="Highlighted text snippet with [[MATCH]]...[[/MATCH]] markers")
    raw_text: str = Field(description="Full text or verbatim excerpt")
    rank: float = Field(description="Raw SQLite FTS5 BM25 score (more negative is better match)")
    score: float = Field(description="Normalized non-negative relevance score (higher is better)")
    captured_at: Optional[str] = Field(default=None, description="Original fetch_date or extracted_at timestamp")
    url: Optional[str] = Field(default=None, description="Canonical or original URL")
    domain: Optional[str] = Field(default=None, description="Source domain")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserving historical origin")


class RetrievalScope(BaseModel):
    """Scoping parameters to restrict search and prevent cross-tenant/project data leakage."""

    model_config = ConfigDict(frozen=True)

    project_id: Optional[str] = Field(default=None, description="Restrict to specific project/prospect ID")
    prospect_id: Optional[str] = Field(default=None, description="Alias for project_id")
    run_id: Optional[str] = Field(default=None, description="Restrict to specific application run ID")
    session_id: Optional[str] = Field(default=None, description="Restrict to specific research session ID")
    domain: Optional[str] = Field(default=None, description="Restrict to specific source domain")
    record_type: Optional[RetrievedRecordType] = Field(default=None, description="Restrict to snapshots or evidence")


class RetrievalResponse(BaseModel):
    """Response envelope for lexical retrieval queries."""

    model_config = ConfigDict(frozen=True)

    query: str
    sanitized_query: str
    total_matches: int
    items: List[RetrievedItem] = Field(default_factory=list)
    duration_ms: float
