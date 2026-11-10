from __future__ import annotations

import pytest

from src.researcher.syndication import (
    DependenceType,
    DetectionMethod,
    LSHIndex,
    SourceDependenceRelationship,
    SourceFingerprint,
    SyndicationCluster,
    SyndicationDetector,
    compute_jaccard_similarity,
    compute_minhash_signature,
    compute_normalized_content_hash,
    compute_raw_content_hash,
    estimate_minhash_similarity,
    generate_5gram_shingles,
    normalize_source_text,
    strip_html_tags,
    tokenize,
)


def test_html_stripping_and_normalization():
    raw_html = """
    <html>
      <head><title>Wire Release</title></head>
      <body>
        <p>PR Newswire — San Francisco, CA — <b>Acme Corp</b> announced $50M in Series B funding today.</p>
        <p>Forward-Looking Statements: This press release contains forward-looking statements.</p>
      </body>
    </html>
    """
    stripped = strip_html_tags(raw_html)
    assert "<html>" not in stripped
    assert "<b>" not in stripped
    assert "Acme Corp" in stripped
    assert "$50M" in stripped

    norm = normalize_source_text(raw_html)
    assert "pr newswire" not in norm  # stripped wire prefix
    assert "forward-looking statements" not in norm  # stripped disclaimer
    assert "acme corp" in norm
    assert "$50m" in norm  # financial token preserved
    assert "series b" in norm


def test_exact_raw_content_hash_collision():
    detector = SyndicationDetector()
    text = "Acme Corp announces revolutionary AI platform for enterprise customers."

    fp_a = detector.fingerprint("source_1", text)
    fp_b = detector.fingerprint("source_2", text)

    assert fp_a.raw_content_hash == fp_b.raw_content_hash
    assert fp_a.normalized_content_hash == fp_b.normalized_content_hash

    rel = detector.evaluate_pair("source_1", text, "source_2", text)
    assert rel.dependence_type == DependenceType.EXACT_DUPLICATE
    assert rel.detection_method == DetectionMethod.EXACT_RAW_HASH
    assert rel.similarity_score == 1.0


def test_exact_normalized_content_hash_collision():
    detector = SyndicationDetector()
    text_wire = (
        "BUSINESS WIRE (BUSINESS WIRE) -- SAN FRANCISCO -- "
        "“Acme Corp” announced ‘$50M’ in Series B funding today.   "
        "Media Contact: press@acme.com"
    )
    text_blog = (
        "<p>Acme Corp announced $50M in Series B funding today.</p>\n"
        "<div>Copyright 2026 Acme Corp. All rights reserved.</div>"
    )

    rel = detector.evaluate_pair("source_wire", text_wire, "source_blog", text_blog)
    assert rel.dependence_type == DependenceType.EXACT_DUPLICATE
    assert rel.detection_method == DetectionMethod.EXACT_NORMALIZED_HASH
    assert rel.similarity_score == 1.0


def test_near_duplicate_minhash_and_jaccard_verification():
    detector = SyndicationDetector(similarity_threshold=0.70)

    original_text = (
        "Acme Corp, a high-growth autonomous robotics company based in San Francisco, "
        "has officially closed a fifty million dollar Series B financing round led by "
        "Pinnacle Ventures with participation from Horizon Capital and existing investors. "
        "The investment brings total capital raised to eighty million dollars since inception. "
        "The funds will be directed toward scaling hardware production, expanding software "
        "engineering teams, and accelerating go-to-market deployments across North America and Europe."
    )

    syndicated_copy = (
        "Acme Corp, a high-growth autonomous robotics company based in San Francisco, "
        "has officially closed a fifty million dollar Series B financing round led by "
        "Pinnacle Ventures with participation from Horizon Capital and existing investors. "
        "The investment brings total capital raised to eighty million dollars since inception. "
        "The funds will be directed toward scaling hardware production, expanding "
        "engineering teams, and accelerating go-to-market deployments across North America and Europe."
    )

    rel = detector.evaluate_pair("src_orig", original_text, "src_syn", syndicated_copy)
    assert rel.dependence_type == DependenceType.NEAR_DUPLICATE
    assert rel.detection_method == DetectionMethod.MINHASH_LSH
    assert rel.similarity_score >= 0.70


def test_paragraph_reordering_near_duplicate():
    detector = SyndicationDetector(similarity_threshold=0.70)

    p1 = (
        "Acme Corp closed a fifty million dollar Series B financing round led by Pinnacle Ventures. "
        "The investment will fund next-generation autonomous ground vehicles."
    )
    p2 = (
        "The company specializes in autonomous logistics robotics for enterprise fulfillment centers and warehouse operators. "
        "Its flagship robotic fleet operates across twenty customer facilities."
    )
    p3 = (
        "Leadership announced that the new capital will support international expansion and team scaling. "
        "Several strategic enterprise partners also participated in the round."
    )

    doc_a = f"{p1}\n\n{p2}\n\n{p3}"
    doc_b = f"{p2}\n\n{p1}\n\n{p3}"

    rel = detector.evaluate_pair("doc_a", doc_a, "doc_b", doc_b)
    assert rel.dependence_type == DependenceType.NEAR_DUPLICATE
    assert rel.similarity_score >= 0.70


def test_independent_reporting_on_same_event():
    """Two outlets independently writing articles about the same event should NOT be clustered."""
    detector = SyndicationDetector(similarity_threshold=0.70)

    techcrunch_style = (
        "Robotics startup Acme Corp has secured fifty million dollars in a new funding round. "
        "Speaking to reporters on Tuesday, chief executive Jane Doe explained that autonomous "
        "warehouse vehicles are experiencing unprecedented demand amid continuing supply chain friction. "
        "Pinnacle Ventures spearheaded the round, marking their third major logistics bet this fiscal year."
    )

    reuters_style = (
        "Autonomous vehicle developer Acme Corp raised $50 million in Series B financing, the company "
        "disclosed in a regulatory filing and press briefing. Investors led by Pinnacle Ventures valued "
        "the firm at an undisclosed premium. Supply chain automation investments have surged across "
        "e-commerce distribution hubs seeking to curb fulfillment labor shortages."
    )

    rel = detector.evaluate_pair("src_tc", techcrunch_style, "src_reuters", reuters_style)
    assert rel.dependence_type == DependenceType.INDEPENDENT
    assert rel.similarity_score < 0.50


def test_short_document_handling():
    detector = SyndicationDetector()
    short_a = "Acme Corp raised $50M."
    short_b = "Acme Corp raised $50M."

    rel = detector.evaluate_pair("short_1", short_a, "short_2", short_b)
    assert rel.dependence_type == DependenceType.EXACT_DUPLICATE

    short_c = "Beta Corp raised $10M."
    rel_diff = detector.evaluate_pair("short_1", short_a, "short_3", short_c)
    assert rel_diff.dependence_type == DependenceType.INDEPENDENT


def test_connected_component_clustering():
    detector = SyndicationDetector(similarity_threshold=0.70)

    base_text = (
        "Acme Corp announces the appointment of Dr. Aris Thorne as Chief Technology Officer. "
        "Dr. Thorne previously led distributed systems research at Global Tech Labs for eight years. "
        "In his new executive role, he will oversee global research and development for cloud orchestration systems. "
        "The appointment follows rapid enterprise customer adoption across European and North American markets."
    )
    copy_1 = (
        "Acme Corp announces the appointment of Dr. Aris Thorne as Chief Technology Officer. "
        "Dr. Thorne previously led distributed systems research at Global Tech Labs for eight years. "
        "In his new executive role, he will oversee research and development for cloud orchestration systems. "
        "The appointment follows rapid enterprise customer adoption across European and North American markets."
    )
    copy_2 = (
        "Acme Corp announces the appointment of Dr. Aris Thorne as Chief Technology Officer. "
        "Dr. Thorne previously led distributed systems research at Global Tech Labs for eight years. "
        "In his new executive role, he will oversee global research and development for cloud orchestration systems. "
        "The executive appointment follows rapid customer adoption across European and North American markets."
    )
    unrelated_text = (
        "Zeta Systems reported quarterly revenue growth of fifteen percent driven by enterprise "
        "database software subscriptions and expanded professional services in the European market."
    )

    sources = [
        {"source_id": "src_wire", "text": base_text},
        {"source_id": "src_reprint1", "text": copy_1},
        {"source_id": "src_reprint2", "text": copy_2},
        {"source_id": "src_independent", "text": unrelated_text},
    ]

    result = detector.detect_syndication(sources)

    # 3 copies should be clustered together into 1 cluster
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert len(cluster.member_source_ids) == 3
    assert "src_wire" in cluster.member_source_ids
    assert "src_reprint1" in cluster.member_source_ids
    assert "src_reprint2" in cluster.member_source_ids
    assert "src_independent" not in cluster.member_source_ids

    # Canonical source ID is deterministically one of the cluster members
    assert cluster.canonical_source_id in cluster.member_source_ids

    # Group mapping
    groups = result.syndication_groups
    assert len(groups) == 1
    assert cluster.cluster_id in groups
    assert groups[cluster.cluster_id] == cluster.member_source_ids
