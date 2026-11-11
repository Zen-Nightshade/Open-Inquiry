from __future__ import annotations

import logging
import time
from typing import Any, Optional
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from .base import BaseSearchProvider
from .models import SearchResponse, SearchResult, SearchStatus
from .url_utils import canonicalize_url, extract_domain

logger = logging.getLogger(__name__)


def is_transient_error(exc: Exception) -> bool:
    """Classify whether an exception represents a transient failure eligible for bounded retry."""
    if isinstance(exc, (RatelimitException, TimeoutException, TimeoutError, ConnectionError)):
        return True
    exc_type_name = type(exc).__name__
    if any(transient in exc_type_name for transient in ("Timeout", "ConnectError", "NetworkError", "RateLimit")):
        return True
    exc_str = str(exc).lower()
    if any(k in exc_str for k in ("ratelimit", "rate limit", "rate_limit", "202 ratelimit", "timeout", "timed out", "connection reset", "connection refused")):
        return True
    return False


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """DuckDuckGo search provider adapter using duckduckgo_search library with bounded retries."""

    def __init__(
        self,
        timeout: float = 10.0,
        backend: str = "auto",
        client: Optional[Any] = None,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
    ):
        self.timeout = timeout
        self.backend = backend
        self._client_factory = client
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _get_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory
        return DDGS(timeout=self.timeout)

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """Execute text query via DuckDuckGo and normalize into SearchResponse with bounded retries for transient errors."""
        cleaned_query = (query or "").strip()
        if not cleaned_query or max_results <= 0:
            return SearchResponse(
                query=query,
                status=SearchStatus.EMPTY,
                results=[],
                duration_ms=0.0,
                attempts=0,
            )

        start_time = time.time()
        last_error = None
        is_transient = False
        attempts = 0

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                client = self._get_client()
                raw_results = client.text(
                    keywords=cleaned_query,
                    max_results=max_results,
                    backend=self.backend,
                )
                elapsed_ms = (time.time() - start_time) * 1000

                if not raw_results:
                    return SearchResponse(
                        query=cleaned_query,
                        status=SearchStatus.EMPTY,
                        results=[],
                        duration_ms=elapsed_ms,
                        attempts=attempts,
                    )

                normalized_results: list[SearchResult] = []
                for idx, item in enumerate(raw_results):
                    raw_url = str(item.get("href") or item.get("url") or "").strip()
                    if not raw_url:
                        continue

                    canonical = canonicalize_url(raw_url)
                    if not canonical:
                        continue

                    title = str(item.get("title") or "").strip()
                    snippet = str(item.get("body") or item.get("snippet") or "").strip()
                    domain = extract_domain(canonical)

                    normalized_results.append(
                        SearchResult(
                            url=canonical,
                            original_url=raw_url,
                            title=title,
                            snippet=snippet,
                            domain=domain,
                            position=idx + 1,
                            query=cleaned_query,
                        )
                    )

                if not normalized_results:
                    return SearchResponse(
                        query=cleaned_query,
                        status=SearchStatus.EMPTY,
                        results=[],
                        duration_ms=elapsed_ms,
                        attempts=attempts,
                    )

                return SearchResponse(
                    query=cleaned_query,
                    status=SearchStatus.SUCCESS,
                    results=normalized_results,
                    duration_ms=elapsed_ms,
                    attempts=attempts,
                )

            except Exception as exc:
                last_error = exc
                is_transient = is_transient_error(exc)
                if is_transient and attempt < self.max_retries:
                    backoff = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Transient search error on attempt %d/%d for query '%s' (%s: %s). Backing off %.2fs...",
                        attempts,
                        self.max_retries + 1,
                        cleaned_query,
                        type(exc).__name__,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                else:
                    break

        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown search error"
        return SearchResponse(
            query=cleaned_query,
            status=SearchStatus.FAILED,
            results=[],
            error_message=error_msg,
            duration_ms=elapsed_ms,
            is_transient=is_transient,
            attempts=attempts,
        )
