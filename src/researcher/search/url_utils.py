from __future__ import annotations

import re
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

TRACKING_PARAM_PATTERNS = {
    # Google Analytics / Urchin
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_reader",
    "utm_place",
    # Ad network click identifiers
    "gclid",
    "gclsrc",
    "fbclid",
    "msclkid",
    "dclid",
    "yclid",
    "twclid",
    "wbraid",
    "gbraid",
    # Referral and tracking tags (specific tracking names, not generic words)
    "ref_src",
    "ref_url",
    "trk",
    "trackingid",
    # Marketing automation / newsletters
    "_hsenc",
    "_hsmi",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "vero_id",
    "wickedid",
}


def extract_domain(url: str) -> str:
    """Extract clean domain name without scheme, port, or 'www.' prefix.
    
    Used for domain-level capping, credibility heuristics, and entity grouping.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        netloc = netloc.strip()
        if not netloc or " " in netloc or not re.match(r"^[a-z0-9.-]+$", netloc):
            return ""
        if "." not in netloc and netloc != "localhost":
            return ""
        return netloc
    except Exception:
        return ""


def canonicalize_url(url: str) -> str:
    """Normalize and canonicalize a URL for reliable deduplication.
    
    Correctness Rules:
    1. Trims whitespace. Rejects empty strings, non-HTTP schemes (mailto, javascript, ftp, file),
       and malformed URLs without network location.
    2. Lowercases scheme and netloc.
    3. Strips default ports (:80 for http, :443 for https).
    4. Does NOT assume www and non-www are equivalent (preserved as distinct hostnames unless redirected).
    5. Strips only verified tracking/analytics query parameters, preserving article IDs,
       search parameters, filters, and repeated parameters.
    6. Normalizes query parameter ordering by (key, value) deterministically.
    7. Preserves percent-encoded path elements and query parameter values.
    8. Normalizes redundant slashes and strips trailing slash on non-root paths.
    9. Strips URL fragments (#...).
    """
    if not url or not isinstance(url, str):
        return ""

    raw = url.strip()
    if not raw:
        return ""

    # Reject dangerous or non-web schemes immediately
    if raw.lower().startswith(("javascript:", "mailto:", "data:", "file:", "ftp:", "about:")):
        return ""

    # Add https scheme only if URL appears to be a domain/path without scheme
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = f"https://{raw}"

    try:
        parsed = urlparse(raw)
    except Exception:
        return ""

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return ""

    netloc = parsed.netloc.lower()
    if not netloc:
        return ""

    # Reject invalid hostnames (must contain a dot or be localhost/IP)
    host = netloc.split(":")[0]
    if "." not in host and host != "localhost":
        return ""

    # Strip default ports (:80, :443)
    if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]

    # Normalize path: collapse unencoded multiple slashes, strip trailing slash unless root '/'
    path = parsed.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Filter tracking parameters and sort parameters deterministically
    filtered_params: list[tuple[str, str]] = []
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            k_lower = k.lower()
            if k_lower in TRACKING_PARAM_PATTERNS or k_lower.startswith("utm_"):
                continue
            filtered_params.append((k, v))
        # Stable deterministic sorting by (key, value) preserving repeated parameters
        filtered_params.sort(key=lambda item: (item[0], item[1]))

    new_query = urlencode(filtered_params, quote_via=quote) if filtered_params else ""

    # Reconstruct canonical URL without fragment
    return urlunparse((scheme, netloc, path, "", new_query, ""))
