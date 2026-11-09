from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from duckduckgo_search.exceptions import RatelimitException, TimeoutException

from src.researcher.search.base import BaseSearchProvider
from src.researcher.search.duckduckgo_provider import DuckDuckGoSearchProvider
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.search_client import SearchClient


class DummyProvider(BaseSearchProvider):
    def __init__(self, response_map: dict[str, SearchResponse]):
        self.response_map = response_map

    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        return self.response_map.get(
            query,
            SearchResponse(query=query, status=SearchStatus.EMPTY, results=[]),
        )


def test_ddg_provider_success():
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {
            "title": "Cyberdyne Systems News",
            "href": "https://techcrunch.com/2026/cyberdyne?utm_source=rss",
            "body": "Cyberdyne Systems announced today that it raised a Series B round.",
        },
        {
            "title": "Forbes Profile",
            "href": "https://www.forbes.com/companies/cyberdyne/",
            "body": "Company overview and leadership breakdown.",
        },
    ]

    provider = DuckDuckGoSearchProvider(client=mock_ddgs)
    resp = provider.search("Cyberdyne Series B", max_results=5)

    assert resp.is_success
    assert resp.status == SearchStatus.SUCCESS
    assert len(resp.results) == 2

    first = resp.results[0]
    assert first.domain == "techcrunch.com"
    assert "utm_source" not in first.url
    assert first.original_url == "https://techcrunch.com/2026/cyberdyne?utm_source=rss"
    assert first.position == 1
    assert first.snippet.startswith("Cyberdyne Systems announced")


def test_ddg_provider_distinguishes_empty_from_failed():
    mock_ddgs = MagicMock()

    # 1. Empty results (no matches found)
    mock_ddgs.text.return_value = []
    provider = DuckDuckGoSearchProvider(client=mock_ddgs)
    empty_resp = provider.search("nonexistent query that finds nothing", max_results=5)

    assert empty_resp.status == SearchStatus.EMPTY
    assert empty_resp.is_empty
    assert not empty_resp.is_failed
    assert len(empty_resp.results) == 0

    # 2. Rate limit failure
    mock_ddgs.text.side_effect = RatelimitException("Rate limit exceeded")
    rl_resp = provider.search("Cyberdyne Systems", max_results=5)

    assert rl_resp.status == SearchStatus.FAILED
    assert rl_resp.is_failed
    assert not rl_resp.is_empty
    assert "RatelimitException" in (rl_resp.error_message or "")
    assert len(rl_resp.results) == 0

    # 3. Timeout failure
    mock_ddgs.text.side_effect = TimeoutException("Connection timed out")
    to_resp = provider.search("Cyberdyne Systems", max_results=5)

    assert to_resp.status == SearchStatus.FAILED
    assert "TimeoutException" in (to_resp.error_message or "")


def test_search_client_url_deduplication_before_domain_capping():
    """Verify URL deduplication runs first so duplicate URLs do not consume domain quota."""
    client = SearchClient(provider=None, max_results_per_domain=1)

    # Input has 2 copies of Article A, and 1 copy of Article B, both on domain techcrunch.com
    incoming = [
        SearchResult(
            url="https://techcrunch.com/article-a",
            original_url="https://techcrunch.com/article-a",
            title="Article A Copy 1",
            snippet="Snippet A",
            domain="techcrunch.com",
            position=1,
            query="test",
        ),
        SearchResult(
            url="https://techcrunch.com/article-a",
            original_url="https://techcrunch.com/article-a?utm_source=feed",
            title="Article A Copy 2",
            snippet="Snippet A",
            domain="techcrunch.com",
            position=2,
            query="test",
        ),
        SearchResult(
            url="https://techcrunch.com/article-b",
            original_url="https://techcrunch.com/article-b",
            title="Article B",
            snippet="Snippet B",
            domain="techcrunch.com",
            position=3,
            query="test",
        ),
        SearchResult(
            url="https://bloomberg.com/article-c",
            original_url="https://bloomberg.com/article-c",
            title="Article C",
            snippet="Snippet C",
            domain="bloomberg.com",
            position=4,
            query="test",
        ),
    ]

    deduped = client.deduplicate(incoming, max_results_per_domain=1)

    # With cap=1, techcrunch should yield article-a (deduplicated), bloomberg yields article-c
    assert len(deduped) == 2
    assert deduped[0].url == "https://techcrunch.com/article-a"
    assert deduped[1].url == "https://bloomberg.com/article-c"


def test_search_client_seen_urls_caller_immutability_and_explicit_update():
    """Verify that caller-owned seen_urls is not mutated unless update_seen=True."""
    client = SearchClient(provider=None, max_results_per_domain=5)
    caller_seen: set[str] = {"https://seen-before.com/page"}

    incoming = [
        SearchResult(
            url="https://techcrunch.com/new-article",
            original_url="https://techcrunch.com/new-article",
            title="New",
            snippet="Text",
            domain="techcrunch.com",
            position=1,
            query="test",
        ),
    ]

    # By default, update_seen is False: caller_seen MUST NOT be mutated
    res1 = client.deduplicate(incoming, seen_urls=caller_seen, update_seen=False)
    assert len(res1) == 1
    assert caller_seen == {"https://seen-before.com/page"}

    # When update_seen is True, caller_seen is explicitly updated
    res2 = client.deduplicate(incoming, seen_urls=caller_seen, update_seen=True)
    assert len(res2) == 1
    assert "https://techcrunch.com/new-article" in caller_seen


def test_search_client_configurable_domain_capping():
    client = SearchClient(provider=None, max_results_per_domain=1)

    incoming = [
        SearchResult(
            url="https://techcrunch.com/a1",
            original_url="https://techcrunch.com/a1",
            title="A1",
            snippet="S1",
            domain="techcrunch.com",
            position=1,
            query="test",
        ),
        SearchResult(
            url="https://techcrunch.com/a2",
            original_url="https://techcrunch.com/a2",
            title="A2",
            snippet="S2",
            domain="techcrunch.com",
            position=2,
            query="test",
        ),
    ]

    # With client default cap of 1
    res_cap_1 = client.deduplicate(incoming)
    assert len(res_cap_1) == 1

    # Overridden to cap of 2
    res_cap_2 = client.deduplicate(incoming, max_results_per_domain=2)
    assert len(res_cap_2) == 2

    # Overridden to 0 (disabled domain capping)
    res_cap_disabled = client.deduplicate(incoming, max_results_per_domain=0)
    assert len(res_cap_disabled) == 2


def test_search_and_deduplicate_batch_isolates_failures():
    """Verify that a failing query does not crash the entire batch."""
    q1 = "Cyberdyne funding"
    q2 = "Cyberdyne leadership error"
    q3 = "Cyberdyne hiring"

    success_item = SearchResult(
        url="https://techcrunch.com/cyberdyne-round",
        original_url="https://techcrunch.com/cyberdyne-round",
        title="Round announcement",
        snippet="Announced round",
        domain="techcrunch.com",
        position=1,
        query=q1,
    )
    hiring_item = SearchResult(
        url="https://jobs.lever.co/cyberdyne",
        original_url="https://jobs.lever.co/cyberdyne",
        title="Careers",
        snippet="Open roles",
        domain="lever.co",
        position=1,
        query=q3,
    )

    response_map = {
        q1: SearchResponse(query=q1, status=SearchStatus.SUCCESS, results=[success_item]),
        q2: SearchResponse(query=q2, status=SearchStatus.FAILED, results=[], error_message="503 Backend Error"),
        q3: SearchResponse(query=q3, status=SearchStatus.SUCCESS, results=[hiring_item]),
    }

    dummy_provider = DummyProvider(response_map)
    client = SearchClient(provider=dummy_provider)
    seen: set[str] = set()

    batch_results = client.search_and_deduplicate([q1, q2, q3], seen)

    assert len(batch_results) == 2
    assert batch_results[0].url == "https://techcrunch.com/cyberdyne-round"
    assert batch_results[1].url == "https://jobs.lever.co/cyberdyne"
    assert len(seen) == 2
