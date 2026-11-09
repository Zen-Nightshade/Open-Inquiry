from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.models import SourceRecord


class FetchStatus(str, Enum):
    """Outcome status of an HTTP fetch and extraction attempt."""

    SUCCESS = "success"                      # Content successfully fetched, parsed, and extracted
    HTTP_ERROR = "http_error"                # Non-2xx HTTP status (403, 404, 500, etc.)
    TIMEOUT = "timeout"                      # Connection or read timeout
    CONNECTION_ERROR = "connection_error"    # DNS resolution failure, network unreachable, connection reset
    EMPTY_CONTENT = "empty_content"          # HTTP 200 succeeded but page body was empty or below length threshold
    PARSE_ERROR = "parse_error"              # Unrecoverable HTML extraction or encoding parsing failure


class FetchOutcome(BaseModel):
    """Detailed result of fetching and extracting a web page.
    
    Populates the existing SourceRecord contract upon success.
    """

    model_config = ConfigDict(frozen=True)

    url: str = Field(description="Requested URL")
    final_url: Optional[str] = Field(default=None, description="Final resolved URL after redirects")
    status: FetchStatus = Field(description="Outcome status of the fetch operation")
    http_status_code: Optional[int] = Field(default=None, description="HTTP response code if available")
    source_record: Optional[SourceRecord] = Field(
        default=None,
        description="Structured SourceRecord instance if extraction succeeded",
    )
    error_message: Optional[str] = Field(default=None, description="Descriptive error explanation if failed")
    duration_ms: Optional[float] = Field(default=None, description="Network and parsing duration in milliseconds")

    @property
    def is_success(self) -> bool:
        return self.status == FetchStatus.SUCCESS and self.source_record is not None
