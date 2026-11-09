from __future__ import annotations

import datetime
import json
import logging
from typing import Optional, Tuple
from bs4 import BeautifulSoup
import dateparser
import htmldate

from src.researcher.models import SourceType
from src.researcher.search.url_utils import extract_domain

logger = logging.getLogger(__name__)

# Known domain groupings for source classification
PRESS_RELEASE_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "accesswire.com",
    "newsfilecorp.com",
}

MAJOR_NEWS_DOMAINS = {
    "techcrunch.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "forbes.com",
    "ft.com",
    "cnbc.com",
    "theverge.com",
    "venturebeat.com",
    "wired.com",
    "fortune.com",
    "nytimes.com",
}

JOB_BOARD_DOMAINS = {
    "greenhouse.io",
    "lever.co",
    "workday.com",
    "indeed.com",
    "glassdoor.com",
    "jobs.lever.co",
    "boards.greenhouse.io",
}


def extract_html_canonical_url(html: str) -> Optional[str]:
    """Extract canonical URL declared within HTML <link rel='canonical' href='...'> if present."""
    if not html or not html.strip():
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        link = soup.find("link", rel=lambda val: val and "canonical" in val.lower())
        if link and link.get("href"):
            return str(link["href"]).strip()
    except Exception as exc:
        logger.debug("Failed extracting HTML canonical link: %s", exc)
    return None


def extract_publication_date_with_source(
    html: str,
) -> Tuple[Optional[datetime.date], Optional[str]]:
    """Extract page publication date and the metadata origin that provided it.
    
    Correctness Rules:
    1. Returns (None, None) when publication date is missing or malformed.
    2. Does NOT confuse modification/update dates with publication dates.
       If only modification metadata is present, returns (None, None).
    3. Rejects future dates (e.g. placeholder dates > today + 1 day).
    4. Handles malformed or unparseable dates safely without raising exceptions.
    5. Prioritizes explicit publication metadata over fallback body text extraction.
    """
    if not html or not html.strip():
        return None, None

    max_allowed_date = datetime.date.today() + datetime.timedelta(days=1)

    try:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Check explicit publication metadata tags
        pub_meta_candidates = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "article:published_time"}),
            ("meta", {"name": "publication_date"}),
            ("meta", {"name": "date"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"property": "og:published_time"}),
            ("meta", {"itemprop": "datePublished"}),
        ]
        for tag, attrs in pub_meta_candidates:
            el = soup.find(tag, attrs=attrs)
            if el and el.get("content"):
                raw_val = str(el["content"]).strip()
                parsed = dateparser.parse(raw_val)
                if parsed and parsed.date() <= max_allowed_date:
                    attr_val = list(attrs.values())[0]
                    return parsed.date(), f"meta:{attr_val}"

        # 2. Check JSON-LD structured data for datePublished
        for script in soup.find_all("script", type="application/ld+json"):
            if script.string and "datePublished" in script.string:
                try:
                    data = json.loads(script.string)
                    # Handle both dict and list of items
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if isinstance(item, dict) and item.get("datePublished"):
                            parsed = dateparser.parse(str(item["datePublished"]))
                            if parsed and parsed.date() <= max_allowed_date:
                                return parsed.date(), "json_ld:datePublished"
                except Exception:
                    pass

        # 3. Check <time> tag with explicit pubdate or datePublished itemprop
        for t in soup.find_all("time"):
            if t.has_attr("pubdate") or t.get("itemprop") == "datePublished":
                raw_time = t.get("datetime") or t.get_text()
                if raw_time:
                    parsed = dateparser.parse(str(raw_time).strip())
                    if parsed and parsed.date() <= max_allowed_date:
                        return parsed.date(), "time:pubdate"

        # 4. Check if ONLY modification tags exist (no publication date)
        mod_meta_candidates = [
            ("meta", {"property": "article:modified_time"}),
            ("meta", {"name": "article:modified_time"}),
            ("meta", {"property": "og:updated_time"}),
            ("meta", {"name": "last-modified"}),
            ("meta", {"name": "updated_at"}),
        ]
        has_modification_only = any(soup.find(t, a) for t, a in mod_meta_candidates)
        if has_modification_only:
            # We explicitly do not treat modification dates as publication dates
            return None, None

        # 5. Fallback: htmldate with original_date=True
        raw_date = htmldate.find_date(html, original_date=True, outputformat="%Y-%m-%d")
        if raw_date:
            parsed_date = datetime.date.fromisoformat(raw_date)
            if parsed_date <= max_allowed_date:
                return parsed_date, "htmldate:original"

    except Exception as exc:
        logger.debug("Publication date extraction encountered non-fatal error: %s", exc)

    return None, None


def extract_publication_date(html: str) -> Optional[datetime.date]:
    """Extract article or page publication date using metadata fallbacks and htmldate."""
    date_val, _ = extract_publication_date_with_source(html)
    return date_val


def extract_title(html: str) -> Optional[str]:
    """Extract page title with fallback hierarchy (og:title, title tag, h1)."""
    if not html or not html.strip():
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")

        # 1. OpenGraph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = str(og_title["content"]).strip()
            if title:
                return title

        # 2. Standard HTML <title>
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            if title:
                return title

        # 3. Twitter card title
        tw_title = soup.find("meta", attrs={"name": "twitter:title"})
        if tw_title and tw_title.get("content"):
            title = str(tw_title["content"]).strip()
            if title:
                return title

        # 4. First <h1> tag
        h1 = soup.find("h1")
        if h1 and h1.get_text():
            title = h1.get_text().strip()
            if title:
                return title

    except Exception as exc:
        logger.debug("HTML title extraction failed: %s", exc)

    return None


def classify_source_type(url: str, title: Optional[str] = None) -> SourceType:
    """Heuristically determine source category from URL structure, domain, and title."""
    domain = extract_domain(url).lower()
    url_lower = url.lower()
    title_lower = (title or "").lower()

    if "linkedin.com" in domain:
        return SourceType.LINKEDIN

    if domain == "sec.gov" or "/edgar/" in url_lower:
        return SourceType.SEC_FILING

    if domain in PRESS_RELEASE_DOMAINS or "press-release" in url_lower or "press release" in title_lower:
        return SourceType.PRESS_RELEASE

    if any(jb in domain for jb in JOB_BOARD_DOMAINS) or "/careers/" in url_lower or "/jobs/" in url_lower:
        return SourceType.JOB_BOARD

    if domain in MAJOR_NEWS_DOMAINS or "/news/" in url_lower or "/article/" in url_lower:
        return SourceType.NEWS

    if "blog." in domain or "/blog/" in url_lower:
        return SourceType.BLOG

    return SourceType.OTHER


def estimate_credibility(source_type: SourceType, domain: str) -> Tuple[float, str]:
    """Compute a heuristic baseline rating representing source channel authority (0.0 to 1.0).
    
    Important Semantic Boundaries:
    - This score evaluates publication channel reputation and institutional governance; it is NOT claim confidence.
    - A high credibility score does NOT establish that any factual assertion in the source is true.
    - These scores are coarse engineering heuristics for prioritization, NOT calibrated statistical probabilities.
    - Specific evidence strength and factual entailment are evaluated separately in later verification phases.
    """
    domain_clean = domain.lower()

    if source_type == SourceType.SEC_FILING:
        return 0.95, "Official regulatory disclosure filing subject to legal penalties"

    if domain_clean in MAJOR_NEWS_DOMAINS:
        return 0.90, "Established financial/technology journalism with editorial oversight"

    if source_type == SourceType.PRESS_RELEASE:
        return 0.85, "Primary corporate press release distribution wire"

    if source_type == SourceType.JOB_BOARD:
        return 0.85, "Verified enterprise hiring portal"

    if source_type == SourceType.COMPANY_WEBSITE:
        return 0.80, "Primary organizational web domain"

    if source_type == SourceType.NEWS:
        return 0.75, "General editorial or industry news source"

    if source_type == SourceType.LINKEDIN:
        return 0.70, "Professional social network profile or authored post"

    if source_type == SourceType.BLOG:
        return 0.60, "Corporate or independent blog publication"

    return 0.65, "Standard public web content"
