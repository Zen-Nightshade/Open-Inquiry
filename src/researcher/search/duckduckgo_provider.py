from __future__ import annotations

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


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """DuckDuckGo search provider adapter using duckduckgo_search library."""

    def __init__(
        self,
        timeout: float = 10.0,
        backend: str = "auto",
        client: Optional[Any] = None,
    ):
        self.timeout = timeout
        self.backend = backend
        self._client_factory = client

    def _get_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory
        return DDGS(timeout=self.timeout)

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """Execute text query via DuckDuckGo and normalize into SearchResponse."""
        cleaned_query = (query or "").strip()
        if not cleaned_query or max_results <= 0:
            return SearchResponse(
                query=query,
                status=SearchStatus.EMPTY,
                results=[],
                duration_ms=0.0,
            )

        start_time = time.time()

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
                )

            return SearchResponse(
                query=cleaned_query,
                status=SearchStatus.SUCCESS,
                results=normalized_results,
                duration_ms=elapsed_ms,
            )

        except (RatelimitException, TimeoutException, DuckDuckGoSearchException) as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            return SearchResponse(
                query=cleaned_query,
                status=SearchStatus.FAILED,
                results=[],
                error_message=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            return SearchResponse(
                query=cleaned_query,
                status=SearchStatus.FAILED,
                results=[],
                error_message=f"Unexpected search failure ({type(exc).__name__}): {exc}",
                duration_ms=elapsed_ms,
            )
