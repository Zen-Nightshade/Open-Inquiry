from __future__ import annotations

from src.researcher.search.url_utils import canonicalize_url, extract_domain


def test_extract_domain():
    assert extract_domain("https://www.techcrunch.com/2026/article") == "techcrunch.com"
    assert extract_domain("http://WWW.Sub.Example.CO.UK:8080/path?q=1") == "sub.example.co.uk"
    assert extract_domain("company.com") == "company.com"
    assert extract_domain("") == ""
    assert extract_domain("not a valid url $$$") == ""


def test_canonicalize_url_tracking_parameters():
    url = (
        "https://www.example.com/article/?"
        "utm_source=twitter&utm_medium=social&utm_campaign=spring_launch&"
        "gclid=12345&fbclid=abcdef&ref_src=twsrc&trk=public_post&real_param=keep_this"
    )
    canonical = canonicalize_url(url)
    assert "utm_source" not in canonical
    assert "utm_medium" not in canonical
    assert "gclid" not in canonical
    assert "fbclid" not in canonical
    assert "ref_src" not in canonical
    assert "trk=" not in canonical
    assert "real_param=keep_this" in canonical


def test_canonicalize_url_preserves_meaningful_query_parameters():
    url = "https://news.example.com/item?article_id=98765&id=42&view=full&page=3&sort=desc"
    canonical = canonicalize_url(url)
    assert "article_id=98765" in canonical
    assert "id=42" in canonical
    assert "view=full" in canonical
    assert "page=3" in canonical
    assert "sort=desc" in canonical


def test_canonicalize_url_query_parameter_sorting_and_repeated_keys():
    # Deterministic sorting across keys
    url1 = "https://example.com/search?z=3&a=1&m=2"
    assert canonicalize_url(url1) == "https://example.com/search?a=1&m=2&z=3"

    # Repeated query parameters are preserved and sorted deterministically
    url2 = "https://example.com/filter?tag=python&tag=ai&category=tech"
    canonical2 = canonicalize_url(url2)
    assert canonical2 == "https://example.com/filter?category=tech&tag=ai&tag=python"


def test_canonicalize_url_encoded_paths_and_values():
    # Percent-encoded path elements preserved
    url_path = "https://example.com/news%20item/C%2B%2B%20Guide"
    assert canonicalize_url(url_path) == "https://example.com/news%20item/C%2B%2B%20Guide"

    # Percent-encoded query values preserved correctly
    url_query = "https://example.com/search?q=C%2B%2B%20Language&topic=AI%2FML"
    canonical_q = canonicalize_url(url_query)
    assert "q=C%2B%2B%20Language" in canonical_q
    assert "topic=AI%2FML" in canonical_q


def test_canonicalize_url_rejects_non_http_and_malformed_urls():
    assert canonicalize_url("javascript:alert(1)") == ""
    assert canonicalize_url("mailto:user@example.com") == ""
    assert canonicalize_url("data:text/html,<h1>Hello</h1>") == ""
    assert canonicalize_url("ftp://ftp.example.com/files.zip") == ""
    assert canonicalize_url("file:///etc/passwd") == ""
    assert canonicalize_url("http://") == ""
    assert canonicalize_url("https://") == ""
    assert canonicalize_url("http://:8080") == ""
    assert canonicalize_url("") == ""
    assert canonicalize_url("   ") == ""


def test_canonicalize_url_preserves_www_distinctness():
    # www and non-www are not assumed equivalent without redirect proof
    c_www = canonicalize_url("https://www.example.com/about")
    c_bare = canonicalize_url("https://example.com/about")
    assert c_www == "https://www.example.com/about"
    assert c_bare == "https://example.com/about"
    assert c_www != c_bare
