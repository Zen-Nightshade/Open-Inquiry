from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SearchStatus(str, Enum):
    """Execution status of a search query."""

    SUCCESS = "success"  # Search succeeded and returned 1 or more results
    EMPTY = "empty"      # Search succeeded but found 0 matching results
    FAILED = "failed"    # Search execution failed (network, rate limit, timeout, provider error)


class SearchResult(BaseModel):
    """Normalized search result item.
    
    Note: Search snippets are preliminary discovery indicators, NOT verified evidence.
    """

    model_config = ConfigDict(frozen=True)

    url: str = Field(description="Canonical normalized URL")
    original_url: str = Field(description="Raw URL returned by search engine")
    title: str = Field(description="Result title")
    snippet: str = Field(description="Search snippet / teaser text")
    domain: str = Field(description="Cleaned domain name")
    position: int = Field(ge=1, description="Rank position in search results")
    query: str = Field(description="Query string that produced this result")


class SearchResponse(BaseModel):
    """Structured response for a search query distinguishing success, empty, and failure."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Query string executed")
    status: SearchStatus = Field(description="Outcome status of the search execution")
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    error_message: Optional[str] = Field(default=None, description="Detailed error description if failed")
    duration_ms: Optional[float] = Field(default=None, description="Execution duration in milliseconds")

    @property
    def is_success(self) -> bool:
        return self.status == SearchStatus.SUCCESS

    @property
    def is_empty(self) -> bool:
        return self.status == SearchStatus.EMPTY

    @property
    def is_failed(self) -> bool:
        return self.status == SearchStatus.FAILED
