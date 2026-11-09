from __future__ import annotations

import logging
from typing import List, Optional, Set

from .base import BaseSearchProvider
from .duckduckgo_provider import DuckDuckGoSearchProvider
from .models import SearchResponse, SearchResult, SearchStatus

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-independent search client coordinating execution, normalization, and deduplication.
    
    The underlying search provider (DuckDuckGo, Brave, etc.) is injected at initialization
    and can be swapped without modifying Researcher business logic.
    """

    def __init__(
        self,
        provider: Optional[BaseSearchProvider] = None,
        max_results_per_domain: int = 2,
    ):
        self.provider: BaseSearchProvider = provider or DuckDuckGoSearchProvider()
        self.max_results_per_domain = max_results_per_domain

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """Execute a single query through the configured search provider."""
        return self.provider.search(query=query, max_results=max_results)

    def deduplicate(
        self,
        results: List[SearchResult],
        seen_urls: Optional[Set[str]] = None,
        max_results_per_domain: Optional[int] = None,
        update_seen: bool = False,
    ) -> List[SearchResult]:
        """Deduplicate results against prior URLs and apply domain capping.
        
        Correctness Guarantees:
        1. URL deduplication occurs BEFORE domain capping. Duplicate URLs within the input
           are filtered out first and never consume domain quota.
        2. Domain capping is configurable via parameter or client default (0 or None disables capping).
        3. Caller-owned `seen_urls` sets are NOT modified unless `update_seen=True` is explicitly specified.
        4. Independent search results with distinct canonical URLs within the domain cap are preserved.
        """
        effective_cap = (
            self.max_results_per_domain
            if max_results_per_domain is None
            else max_results_per_domain
        )

        known_urls = set(seen_urls) if seen_urls is not None else set()

        # Step 1: URL Deduplication (unique canonical URLs within batch and against known_urls)
        unique_results: List[SearchResult] = []
        batch_seen: Set[str] = set()

        for item in results:
            if not item.url:
                continue

            canonical = item.url
            if canonical in known_urls or canonical in batch_seen:
                continue

            batch_seen.add(canonical)
            unique_results.append(item)

        # Step 2: Domain Capping on deduplicated results
        final_results: List[SearchResult] = []
        domain_counts: dict[str, int] = {}

        for item in unique_results:
            domain = item.domain
            current_count = domain_counts.get(domain, 0)
            if effective_cap > 0 and current_count >= effective_cap:
                continue

            domain_counts[domain] = current_count + 1
            final_results.append(item)

        # Step 3: Explicitly update caller's seen_urls only if requested
        if update_seen and seen_urls is not None:
            seen_urls.update(item.url for item in final_results)

        return final_results

    def search_and_deduplicate(
        self,
        queries: List[str],
        seen_urls: Set[str],
        max_results_per_query: int = 8,
        max_results_per_domain: Optional[int] = None,
    ) -> List[SearchResult]:
        """Execute a batch of queries, accumulating novel deduplicated results.
        
        Mutates `seen_urls` across queries as explicitly documented to track cross-query discovery.
        Failures on individual queries are logged and isolated; they are NOT treated
        as evidence that information does not exist.
        """
        accumulated: List[SearchResult] = []

        for q in queries:
            cleaned = (q or "").strip()
            if not cleaned:
                continue

            resp = self.search(cleaned, max_results=max_results_per_query)

            if resp.status == SearchStatus.FAILED:
                logger.warning(
                    "Search query '%s' failed (%s). Continuing with remaining queries.",
                    cleaned,
                    resp.error_message,
                )
                continue

            if resp.status == SearchStatus.EMPTY:
                logger.debug("Search query '%s' yielded 0 results.", cleaned)
                continue

            novel = self.deduplicate(
                resp.results,
                seen_urls=seen_urls,
                max_results_per_domain=max_results_per_domain,
                update_seen=True,
            )
            accumulated.extend(novel)

        return accumulated
