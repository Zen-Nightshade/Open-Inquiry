from __future__ import annotations

import datetime
import httpx
import pytest

from src.researcher.extraction.fetcher import PageFetcher
from src.researcher.extraction.models import FetchStatus
from src.researcher.models import SourceType


def create_mock_client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, follow_redirects=True)


def test_fetch_success_with_metadata_and_date():
    html = """<!DOCTYPE html>
    <html>
    <head>
        <title>Cyberdyne Announces $30M Series A Round</title>
        <meta property="og:title" content="Cyberdyne Announces $30M Series A Round" />
        <meta property="article:published_time" content="2026-07-20T10:00:00Z" />
        <link rel="canonical" href="https://techcrunch.com/2026/cyberdyne-series-a" />
    </head>
    <body>
        <main>
            <h1>Cyberdyne Secures $30M</h1>
            <p>Cyberdyne Systems announced today that it closed $30 million in Series A funding.</p>
            <p>The investment will accelerate global commercial deployment of autonomous systems.</p>
            <p>CEO Miles Dyson remarked that customer demand exceeded expectations across major accounts.</p>
        </main>
    </body>
    </html>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    mock_client = create_mock_client(handler)
    fetcher = PageFetcher(client=mock_client)

    outcome = fetcher.fetch("https://techcrunch.com/2026/cyberdyne-series-a")

    assert outcome.is_success
    assert outcome.status == FetchStatus.SUCCESS
    assert outcome.http_status_code == 200

    rec = outcome.source_record
    assert rec is not None
    assert rec.domain == "techcrunch.com"
    assert "Series A" in (rec.title or "")
    assert "Cyberdyne Systems announced today" in rec.raw_text
    assert rec.publication_date == datetime.date(2026, 7, 20)
    assert rec.publication_date_source == "meta:article:published_time"
    assert rec.fetch_date == datetime.date.today()
    assert rec.source_type == SourceType.NEWS
    assert rec.credibility_score >= 0.85
    assert rec.html_canonical_url == "https://techcrunch.com/2026/cyberdyne-series-a"


def test_fetch_success_with_missing_publication_date():
    html = """<!DOCTYPE html>
    <html>
    <head><title>Product Documentation Overview</title></head>
    <body>
        <article>
            <p>Open Inquiry automates personalized B2B outreach research, claim verification, and drafting.</p>
            <p>Built with reliable provenance tracking and evidence corroboration safeguards.</p>
        </article>
    </body>
    </html>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    mock_client = create_mock_client(handler)
    fetcher = PageFetcher(client=mock_client)

    outcome = fetcher.fetch("https://docs.openinquiry.ai/overview")

    assert outcome.is_success
    rec = outcome.source_record
    assert rec is not None
    assert rec.publication_date is None
    assert rec.publication_date_source is None
    assert "Open Inquiry automates personalized B2B outreach" in rec.raw_text


def test_fetch_malformed_html():
    malformed = """
    <div><h1>Unclosed Header
    <p>This page is severely broken with missing doctype, unclosed tags, and raw entities.
    <table border=1><tr><td>Col 1<td>Col 2<tr><td>Val A<td>Val B</table>
    <span>Still contains valid informational content for extraction.</span>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=malformed)

    mock_client = create_mock_client(handler)
    fetcher = PageFetcher(client=mock_client)

    outcome = fetcher.fetch("https://legacy-site.org/messy")

    assert outcome.is_success
    assert "severely broken with missing doctype" in outcome.source_record.raw_text


def test_fetch_empty_content():
    blank_html = "<html><head></head><body>    </body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=blank_html)

    mock_client = create_mock_client(handler)
    fetcher = PageFetcher(client=mock_client, min_content_length=20)

    outcome = fetcher.fetch("https://empty-shell.com/")

    assert outcome.status == FetchStatus.EMPTY_CONTENT
    assert not outcome.is_success
    assert outcome.source_record is None


def test_fetch_http_errors():
    def not_found_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    fetcher_404 = PageFetcher(client=create_mock_client(not_found_handler))
    res_404 = fetcher_404.fetch("https://example.com/missing")
    assert res_404.status == FetchStatus.HTTP_ERROR
    assert res_404.http_status_code == 404
    assert not res_404.is_success

    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Access Denied")

    fetcher_403 = PageFetcher(client=create_mock_client(forbidden_handler))
    res_403 = fetcher_403.fetch("https://example.com/blocked")
    assert res_403.status == FetchStatus.HTTP_ERROR
    assert res_403.http_status_code == 403


def test_fetch_timeout_and_connection_errors():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Server took too long to send data")

    fetcher_timeout = PageFetcher(client=create_mock_client(timeout_handler))
    outcome_to = fetcher_timeout.fetch("https://slow-server.com")
    assert outcome_to.status == FetchStatus.TIMEOUT
    assert "Timeout" in (outcome_to.error_message or "")

    def connect_error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to resolve host")

    fetcher_conn = PageFetcher(client=create_mock_client(connect_error_handler))
    outcome_conn = fetcher_conn.fetch("https://invalid-host.com")
    assert outcome_conn.status == FetchStatus.CONNECTION_ERROR


def test_fetch_text_truncation():
    long_html = f"<html><body><article><p>{'Information content block. ' * 500}</p></article></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=long_html)

    fetcher = PageFetcher(client=create_mock_client(handler), max_content_length=200)
    outcome = fetcher.fetch("https://verbose-blog.com")

    assert outcome.is_success
    assert len(outcome.source_record.raw_text) <= 250
    assert outcome.source_record.raw_text.endswith("... [truncated]")


def test_fetch_multiple_redirects_and_cross_domain():
    """Verify that multiple redirects across domains preserve original, final, and canonical URLs."""
    html = """<html><head><title>Destination Page</title></head>
    <body><article><p>Final destination article body after multi-hop redirect chain.</p></article></body></html>"""

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        # Hop 1: short.ly -> intermediate.org
        if "short.ly/xyz" in url_str:
            return httpx.Response(301, headers={"Location": "https://intermediate.org/jump?id=1"})
        # Hop 2: intermediate.org -> destination-news.com
        if "intermediate.org/jump" in url_str:
            return httpx.Response(302, headers={"Location": "https://destination-news.com/post?id=99&utm_source=twitter&utm_medium=feed"})
        # Final destination
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    fetcher = PageFetcher(client=client)

    outcome = fetcher.fetch("http://short.ly/xyz")

    assert outcome.is_success
    rec = outcome.source_record
    assert rec is not None
    # 1. original_url: exact URL requested from search
    assert rec.original_url == "http://short.ly/xyz"
    # 2. final_url: actual HTTP destination URL reached after all redirects
    assert rec.final_url == "https://destination-news.com/post?id=99&utm_source=twitter&utm_medium=feed"
    # 3. url: canonical normalized final URL (tracking stripped, article ID preserved)
    assert rec.url == "https://destination-news.com/post?id=99"
    assert "utm_source" not in rec.url
    assert "id=99" in rec.url
    # 4. domain: extracted from canonical target
    assert rec.domain == "destination-news.com"


def test_fetch_html_canonical_url_preserved_without_silent_overwrite():
    """HTML <link rel='canonical'> is captured as html_canonical_url but does NOT overwrite actual fetched URL."""
    html = """<html>
    <head>
        <title>Syndicated Article</title>
        <link rel="canonical" href="https://original-publisher.com/story/123" />
    </head>
    <body><article><p>Syndicated news content republished on partner site.</p></article></body>
    </html>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    fetcher = PageFetcher(client=create_mock_client(handler))
    outcome = fetcher.fetch("https://partner-portal.com/republished-story")

    assert outcome.is_success
    rec = outcome.source_record
    assert rec is not None
    # Actual fetched URL remains the primary resource URL
    assert rec.url == "https://partner-portal.com/republished-story"
    assert rec.domain == "partner-portal.com"
    # HTML canonical tag is preserved in dedicated metadata
    assert rec.html_canonical_url == "https://original-publisher.com/story/123"


def test_fetch_excessive_redirects_loop():
    """Verify that redirect loops exceeding max_redirects are caught and surfaced cleanly."""
    def loop_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "loop-a" in url_str:
            return httpx.Response(302, headers={"Location": "https://example.com/loop-b"})
        return httpx.Response(302, headers={"Location": "https://example.com/loop-a"})

    client = httpx.Client(transport=httpx.MockTransport(loop_handler), follow_redirects=True, max_redirects=3)
    fetcher = PageFetcher(client=client, max_redirects=3)

    outcome = fetcher.fetch("https://example.com/loop-a")

    assert outcome.status == FetchStatus.HTTP_ERROR
    assert not outcome.is_success
    assert "Too many redirects" in (outcome.error_message or "")
