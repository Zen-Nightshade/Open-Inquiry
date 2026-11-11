from __future__ import annotations

from unittest.mock import MagicMock, call, patch
import pytest
from duckduckgo_search.exceptions import (
    DuckDuckGoSearchException,
    RatelimitException,
    TimeoutException,
)

from src.researcher.search.duckduckgo_provider import (
    DuckDuckGoSearchProvider,
    is_transient_error,
)
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.search_client import SearchClient


def test_transient_error_classification():
    # Transient errors
    assert is_transient_error(RatelimitException("202 Ratelimit")) is True
    assert is_transient_error(TimeoutException("Request timed out")) is True
    assert is_transient_error(TimeoutError("Connection timed out")) is True
    assert is_transient_error(ConnectionError("Connection reset by peer")) is True
    assert is_transient_error(DuckDuckGoSearchException("DuckDuckGo rate limit exceeded")) is True

    # Permanent errors
    assert is_transient_error(ValueError("Invalid argument")) is False
    assert is_transient_error(KeyError("missing_key")) is False
    assert is_transient_error(DuckDuckGoSearchException("404 Not Found")) is False


def test_ddg_provider_recovers_after_transient_retry():
    mock_ddgs = MagicMock()
    # First call raises RatelimitException; second call succeeds
    mock_ddgs.text.side_effect = [
        RatelimitException("202 Ratelimit"),
        [{"href": "https://example.com/a", "title": "A", "body": "Snippet A"}],
    ]

    provider = DuckDuckGoSearchProvider(client=mock_ddgs, max_retries=2, backoff_factor=0.01)
    resp = provider.search("test query")

    assert resp.status == SearchStatus.SUCCESS
    assert len(resp.results) == 1
    assert resp.results[0].url == "https://example.com/a"
    assert resp.attempts == 2
    assert mock_ddgs.text.call_count == 2


def test_ddg_provider_exhausts_transient_retries():
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = TimeoutException("Connection timed out")

    provider = DuckDuckGoSearchProvider(client=mock_ddgs, max_retries=2, backoff_factor=0.01)
    resp = provider.search("timeout query")

    assert resp.status == SearchStatus.FAILED
    assert resp.is_transient is True
    assert resp.attempts == 3  # 1 initial + 2 retries
    assert "TimeoutException" in resp.error_message
    assert mock_ddgs.text.call_count == 3


def test_ddg_provider_permanent_error_no_retry():
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = ValueError("Fatal search syntax error")

    provider = DuckDuckGoSearchProvider(client=mock_ddgs, max_retries=2, backoff_factor=0.01)
    resp = provider.search("bad syntax")

    assert resp.status == SearchStatus.FAILED
    assert resp.is_transient is False
    assert resp.attempts == 1  # No retries on permanent error
    assert "ValueError" in resp.error_message
    assert mock_ddgs.text.call_count == 1


def test_search_client_defensive_exception_handling():
    mock_provider = MagicMock()
    mock_provider.search.side_effect = RuntimeError("Catastrophic provider crash")

    client = SearchClient(provider=mock_provider)
    resp = client.search("crash query")

    assert resp.status == SearchStatus.FAILED
    assert "ProviderException" in resp.error_message
    assert "Catastrophic provider crash" in resp.error_message


def test_search_and_deduplicate_preserves_partial_results():
    mock_provider = MagicMock()

    # Query 1 succeeds with result 1
    # Query 2 fails with rate limit
    # Query 3 succeeds with result 2
    mock_provider.search.side_effect = [
        SearchResponse(
            query="q1",
            status=SearchStatus.SUCCESS,
            results=[
                SearchResult(
                    url="https://acme.com/about",
                    original_url="https://acme.com/about",
                    title="About Acme",
                    snippet="Acme overview",
                    domain="acme.com",
                    position=1,
                    query="q1",
                )
            ],
        ),
        SearchResponse(
            query="q2",
            status=SearchStatus.FAILED,
            error_message="RatelimitException: 429 Too Many Requests",
            is_transient=True,
        ),
        SearchResponse(
            query="q3",
            status=SearchStatus.SUCCESS,
            results=[
                SearchResult(
                    url="https://news.com/acme",
                    original_url="https://news.com/acme",
                    title="Acme News",
                    snippet="Acme raises funding",
                    domain="news.com",
                    position=1,
                    query="q3",
                )
            ],
        ),
    ]

    client = SearchClient(provider=mock_provider)
    seen_urls: set[str] = set()
    results = client.search_and_deduplicate(
        queries=["q1", "q2", "q3"],
        seen_urls=seen_urls,
    )

    # Both q1 and q3 results are preserved despite q2 failing
    assert len(results) == 2
    assert results[0].url == "https://acme.com/about"
    assert results[1].url == "https://news.com/acme"
    assert "https://acme.com/about" in seen_urls
    assert "https://news.com/acme" in seen_urls
