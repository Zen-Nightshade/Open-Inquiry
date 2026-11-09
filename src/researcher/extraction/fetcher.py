from __future__ import annotations

import datetime
import logging
import time
from typing import Optional
from bs4 import BeautifulSoup
import httpx
import trafilatura

from src.researcher.models import SourceRecord
from src.researcher.search.url_utils import canonicalize_url, extract_domain
from .metadata_extractor import (
    classify_source_type,
    estimate_credibility,
    extract_html_canonical_url,
    extract_publication_date_with_source,
    extract_title,
)
from .models import FetchOutcome, FetchStatus

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


class PageFetcher:
    """Robust web page fetcher using HTTPX and Trafilatura.
    
    Extracts text, metadata, dates, and credibility, preserving complete
    URL provenance across redirects and canonical normalization.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        max_redirects: int = 5,
        user_agent: Optional[str] = None,
        max_content_length: int = 25000,
        min_content_length: int = 25,
        client: Optional[httpx.Client] = None,
    ):
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_content_length = max_content_length
        self.min_content_length = min_content_length
        self._user_agent = user_agent or DEFAULT_USER_AGENT

        self._headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        self._external_client = client
        self._client: Optional[httpx.Client] = client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers=self._headers,
                timeout=self.timeout,
                follow_redirects=True,
                max_redirects=self.max_redirects,
            )
        return self._client

    def fetch(self, url: str) -> FetchOutcome:
        """Fetch and extract page contents while maintaining strict URL provenance.
        
        Provenance contract:
        - original_url: exact URL requested (as provided by search or caller)
        - final_url: actual HTTP destination reached after following network redirects
        - url: canonical normalized version of final_url used for deduplication
        - html_canonical_url: declared <link rel="canonical"> in HTML if present
        """
        cleaned_url = (url or "").strip()
        if not cleaned_url:
            return FetchOutcome(
                url=url,
                status=FetchStatus.CONNECTION_ERROR,
                error_message="Empty URL provided",
                duration_ms=0.0,
            )

        start_time = time.time()
        client = self._get_client()

        try:
            resp = client.get(cleaned_url)
            elapsed_ms = (time.time() - start_time) * 1000
            final_url = str(resp.url)

            # Check HTTP status code
            if resp.status_code >= 400:
                return FetchOutcome(
                    url=cleaned_url,
                    final_url=final_url,
                    status=FetchStatus.HTTP_ERROR,
                    http_status_code=resp.status_code,
                    error_message=f"HTTP {resp.status_code}: {resp.reason_phrase}",
                    duration_ms=elapsed_ms,
                )

            html_text = resp.text
            if not html_text or not html_text.strip():
                return FetchOutcome(
                    url=cleaned_url,
                    final_url=final_url,
                    status=FetchStatus.EMPTY_CONTENT,
                    http_status_code=resp.status_code,
                    error_message="Received empty response body from server",
                    duration_ms=elapsed_ms,
                )

            # 1. Primary body text extraction via Trafilatura
            extracted_text = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=True,
                output_format="txt",
            )

            # 2. Fallback text extraction if trafilatura finds no main content
            if not extracted_text or len(extracted_text.strip()) < self.min_content_length:
                soup = BeautifulSoup(html_text, "html.parser")
                for s in soup(["script", "style", "nav", "footer", "header"]):
                    s.decompose()
                paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "article", "section"])]
                fallback_text = "\n\n".join(p for p in paragraphs if p)
                if fallback_text and len(fallback_text) >= self.min_content_length:
                    extracted_text = fallback_text

            if not extracted_text or len(extracted_text.strip()) < self.min_content_length:
                return FetchOutcome(
                    url=cleaned_url,
                    final_url=final_url,
                    status=FetchStatus.EMPTY_CONTENT,
                    http_status_code=resp.status_code,
                    error_message=f"Extracted content length below minimum threshold ({self.min_content_length} chars)",
                    duration_ms=elapsed_ms,
                )

            # Truncate text cleanly if exceeding maximum limit
            clean_text = extracted_text.strip()
            if len(clean_text) > self.max_content_length:
                clean_text = clean_text[: self.max_content_length] + " ... [truncated]"

            # Metadata extraction
            title = extract_title(html_text)
            pub_date, pub_date_source = extract_publication_date_with_source(html_text)
            html_canonical = extract_html_canonical_url(html_text)

            # Canonical normalized URL used for deduplication (based on actual final_url)
            canonical = canonicalize_url(final_url) or final_url
            domain = extract_domain(canonical)
            source_type = classify_source_type(canonical, title=title)
            credibility, rationale = estimate_credibility(source_type, domain)

            source_record = SourceRecord(
                url=canonical,
                original_url=cleaned_url,
                final_url=final_url,
                html_canonical_url=html_canonical,
                domain=domain,
                title=title,
                raw_text=clean_text,
                source_type=source_type,
                credibility_score=credibility,
                credibility_rationale=rationale,
                fetch_date=datetime.date.today(),
                publication_date=pub_date,
                publication_date_source=pub_date_source,
            )

            return FetchOutcome(
                url=cleaned_url,
                final_url=final_url,
                status=FetchStatus.SUCCESS,
                http_status_code=resp.status_code,
                source_record=source_record,
                duration_ms=elapsed_ms,
            )

        except httpx.TooManyRedirects as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            return FetchOutcome(
                url=cleaned_url,
                status=FetchStatus.HTTP_ERROR,
                error_message=f"Too many redirects (exceeded limit of {self.max_redirects}): {exc}",
                duration_ms=elapsed_ms,
            )
        except (httpx.TimeoutException, TimeoutError) as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            return FetchOutcome(
                url=cleaned_url,
                status=FetchStatus.TIMEOUT,
                error_message=f"Timeout connecting to {cleaned_url}: {exc}",
                duration_ms=elapsed_ms,
            )
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            return FetchOutcome(
                url=cleaned_url,
                status=FetchStatus.CONNECTION_ERROR,
                error_message=f"Connection error reaching {cleaned_url}: {exc}",
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            return FetchOutcome(
                url=cleaned_url,
                status=FetchStatus.PARSE_ERROR,
                error_message=f"Unexpected fetch failure ({type(exc).__name__}): {exc}",
                duration_ms=elapsed_ms,
            )

    def fetch_source(self, url: str) -> Optional[SourceRecord]:
        """Convenience helper returning the SourceRecord on success, or None on failure."""
        outcome = self.fetch(url)
        return outcome.source_record if outcome.is_success else None

    def close(self) -> None:
        """Close internal HTTPX client if owned."""
        if self._client is not None and self._client is not self._external_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> PageFetcher:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
