from __future__ import annotations

import datetime
from src.researcher.extraction.metadata_extractor import (
    classify_source_type,
    estimate_credibility,
    extract_html_canonical_url,
    extract_publication_date,
    extract_publication_date_with_source,
    extract_title,
)
from src.researcher.models import SourceType


def test_missing_and_malformed_publication_dates():
    # 1. No date in HTML
    html_none = "<html><head><title>No Dates</title></head><body><p>Text</p></body></html>"
    d, src = extract_publication_date_with_source(html_none)
    assert d is None
    assert src is None
    assert extract_publication_date(html_none) is None

    # 2. Malformed / garbage date in meta
    html_garbage = (
        '<html><head><meta property="article:published_time" content="garbage-not-a-date" /></head>'
        "<body><p>Text</p></body></html>"
    )
    d_g, src_g = extract_publication_date_with_source(html_garbage)
    assert d_g is None
    assert src_g is None


def test_future_publication_dates_rejected():
    # Future placeholder date e.g. 2099-12-31 must be rejected
    html_future = (
        '<html><head><meta property="article:published_time" content="2099-12-31T00:00:00Z" /></head>'
        "<body><p>Text</p></body></html>"
    )
    d_f, src_f = extract_publication_date_with_source(html_future)
    assert d_f is None
    assert src_f is None


def test_modification_only_dates_not_confused_with_publication_date():
    # Document with only modification metadata must NOT return the modification date
    html_mod_only = (
        '<html><head><title>Updated Article</title>'
        '<meta property="article:modified_time" content="2026-05-15T12:00:00Z" />'
        '<meta property="og:updated_time" content="2026-05-15T12:00:00Z" />'
        "</head><body><p>Updated text</p></body></html>"
    )
    d_mod, src_mod = extract_publication_date_with_source(html_mod_only)
    assert d_mod is None
    assert src_mod is None
    assert extract_publication_date(html_mod_only) is None


def test_both_published_and_modified_dates_returns_publication_date():
    # When both exist, publication date is returned, NOT modification date
    html_both = (
        '<html><head>'
        '<meta property="article:published_time" content="2026-03-01T08:00:00Z" />'
        '<meta property="article:modified_time" content="2026-06-15T14:30:00Z" />'
        "</head><body><p>Article body</p></body></html>"
    )
    d, src = extract_publication_date_with_source(html_both)
    assert d == datetime.date(2026, 3, 1)
    assert src == "meta:article:published_time"


def test_json_ld_publication_date_extraction():
    html_json_ld = """<html><head>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": "Cyberdyne Growth",
        "datePublished": "2026-04-12T09:00:00Z",
        "dateModified": "2026-04-15T10:00:00Z"
    }
    </script>
    </head><body><p>Content</p></body></html>"""

    d, src = extract_publication_date_with_source(html_json_ld)
    assert d == datetime.date(2026, 4, 12)
    assert src == "json_ld:datePublished"


def test_extract_html_canonical_url():
    html_with_canon = '<html><head><link rel="canonical" href="https://example.com/canonical-url" /></head></html>'
    assert extract_html_canonical_url(html_with_canon) == "https://example.com/canonical-url"

    html_without_canon = "<html><head><title>No canonical link</title></head></html>"
    assert extract_html_canonical_url(html_without_canon) is None


def test_credibility_score_semantics_boundary():
    # Verify credibility ratings are assigned by domain/channel authority heuristic
    score_sec, rationale_sec = estimate_credibility(SourceType.SEC_FILING, "sec.gov")
    assert score_sec == 0.95
    assert "regulatory" in rationale_sec.lower()

    score_news, rationale_news = estimate_credibility(SourceType.NEWS, "techcrunch.com")
    assert score_news == 0.90
    assert "journalism" in rationale_news.lower()

    score_blog, _ = estimate_credibility(SourceType.BLOG, "medium.com")
    assert score_blog == 0.60

    # Ensure all scores are bounded between 0.0 and 1.0
    for st in SourceType:
        score, rat = estimate_credibility(st, "example.com")
        assert 0.0 <= score <= 1.0
        assert isinstance(rat, str) and len(rat) > 0
