from __future__ import annotations

import pytest

from src.researcher.syndication.detector import SyndicationDetector
from src.researcher.syndication.models import DependenceType


def get_labeled_evaluation_dataset() -> dict[str, str]:
    """A realistic labeled corpus containing:
    - Original wire press release
    - Exact HTML copy
    - Syndicated reprints with minor edits (MarketWatch, Yahoo, Benzinga)
    - Independently written reporting on the same event (TechCrunch, Reuters, Bloomberg)
    - Unrelated document with common legal/corporate boilerplate
    - Completely unrelated scientific document
    """
    return {
        # Cluster 1: Acme Corp $50M Series B Press Release and Syndicated Reprints
        "d1_wire": (
            "PR Newswire (SAN FRANCISCO, March 15, 2026) — Acme Corp, the category-defining autonomous "
            "logistics robotics platform, today announced that it has secured $50 million in Series B financing "
            "led by Pinnacle Ventures, with continued backing from Horizon Capital and Seed Founders Fund. "
            "The new capital will accelerate the commercial manufacturing of Acme’s flagship autonomous warehouse "
            "transporters and support the expansion of engineering operations across North America and Europe. "
            "Over the past twelve months, Acme Corp expanded customer deployments by three hundred percent across "
            "tier-one retail fulfillment centers. Forward-Looking Statements: This announcement contains projections."
        ),
        "d2_html": (
            "<html><body><h1>Acme Corp Raises $50 Million in Series B Financing</h1>"
            "<p>SAN FRANCISCO — Acme Corp, the category-defining autonomous logistics robotics platform, "
            "today announced that it has secured $50 million in Series B financing led by Pinnacle Ventures, "
            "with continued backing from Horizon Capital and Seed Founders Fund.</p>"
            "<p>The new capital will accelerate the commercial manufacturing of Acme’s flagship autonomous warehouse "
            "transporters and support the expansion of engineering operations across North America and Europe.</p>"
            "<p>Over the past twelve months, Acme Corp expanded customer deployments by three hundred percent across "
            "tier-one retail fulfillment centers.</p>"
            "<p>Media Contact: press@acme.com</p></body></html>"
        ),
        "d3_reprint1": (
            "Acme Corp, the category-defining autonomous logistics robotics platform, today announced that it has "
            "secured $50 million in Series B financing led by Pinnacle Ventures, with continued backing from Horizon "
            "Capital and Seed Founders Fund. The new capital will accelerate the commercial manufacturing of Acme’s "
            "flagship autonomous warehouse transporters and support the expansion of engineering operations across "
            "North America and Europe. Over the past twelve months, Acme Corp expanded customer deployments by three "
            "hundred percent across tier-one retail fulfillment centers."
        ),
        "d4_reprint2": (
            "Acme Corp, the category-defining autonomous logistics robotics platform, today announced that it has "
            "secured $50 million in Series B financing led by Pinnacle Ventures, with continued backing from Horizon "
            "Capital and Seed Founders Fund. The capital will accelerate commercial manufacturing of Acme’s "
            "flagship autonomous warehouse transporters and support expansion of engineering operations across "
            "North America and Europe. Over the past twelve months, Acme Corp expanded customer deployments by three "
            "hundred percent across tier-one retail fulfillment centers."
        ),
        "d5_reprint3": (
            "Acme Corp, the category-defining autonomous logistics robotics platform, today announced that it has "
            "secured $50 million in Series B financing led by Pinnacle Ventures, with continued backing from Horizon "
            "Capital and Seed Founders Fund. The new funds will accelerate the manufacturing of Acme’s flagship "
            "autonomous warehouse transporters and support the expansion of engineering operations across North "
            "America and Europe. Over the past twelve months, Acme Corp expanded customer deployments across tier-one "
            "retail fulfillment centers."
        ),

        # Independent articles on the same funding event (Jane Doe, Acme Corp, $50M Series B)
        "d6_techcrunch": (
            "Robotics maker Acme Corp has bagged $50 million in a Series B round as enterprise warehouses accelerate "
            "automation initiatives to counter ongoing fulfillment labor pinches. Silicon Valley venture powerhouse "
            "Pinnacle Ventures took the lead check in the round, accompanied by Horizon Capital. Founder Jane Doe "
            "emphasized in an interview that the capital infusion will bankroll hardware assembly lines in Ohio and "
            "double the distributed software engineering bench before year end."
        ),
        "d7_reuters": (
            "Autonomous mobile robot developer Acme Corp raised $50 million in private funding, valuation undisclosed, "
            "the company told Reuters on Tuesday. Venture firm Pinnacle Ventures anchored the round alongside prior backers. "
            "Supply chain disruptions and warehouse wage inflation have fueled an unprecedented surge in robotics investments, "
            "prompting logistics operators to automate pallet transit and shelf-picking workflows across major freight terminals."
        ),
        "d8_bloomberg": (
            "Acme Corp, which manufactures self-driving transport carts for e-commerce warehouses, concluded a $50 million "
            "capital raise as venture firms double down on industrial artificial intelligence applications. Lead investor "
            "Pinnacle Ventures cited triple-digit revenue growth and multi-year enterprise contracts with multinational "
            "retailers as primary investment drivers during an exclusive briefing with Bloomberg Technology."
        ),

        # Distinct article with common corporate boilerplate
        "d9_boilerplate_other": (
            "PR Newswire (AUSTIN, Texas) — OmniCorp, a developer of cloud database optimization software, today announced "
            "it has closed $20 million in Series A funding led by Lonestar Capital. The capital will fund relational query engine "
            "benchmarks and European sales channel partnerships. Over the past twelve months, OmniCorp added forty new enterprise accounts. "
            "Forward-Looking Statements: This announcement contains projections and forward-looking statements."
        ),

        # Completely unrelated article
        "d10_unrelated": (
            "Researchers at the University of Washington demonstrated a high-fidelity topological qubit memory architecture "
            "operating under cryogenic temperatures of twenty millikelvin. The experimental protocol achieved a quantum error "
            "suppression factor exceeding ninety-nine point nine percent across a two-dimensional superconducting resonator lattice."
        ),
    }


def test_syndication_evaluation_precision_and_recall():
    """Evaluate pairwise precision and recall across the labeled corpus.
    
    Target:
      - Precision >= 95% (zero false positives on independent reporting or boilerplate).
      - Recall >= 90% (detect all wire reprints and syndicated copies).
    """
    docs = get_labeled_evaluation_dataset()

    # Ground truth: d1 through d5 are dependent copies (10 pairs). All other pairs are independent.
    pos_keys = ["d1_wire", "d2_html", "d3_reprint1", "d4_reprint2", "d5_reprint3"]
    gt_positives = set()
    for i in range(len(pos_keys)):
        for j in range(i + 1, len(pos_keys)):
            gt_positives.add(tuple(sorted([pos_keys[i], pos_keys[j]])))

    detector = SyndicationDetector(similarity_threshold=0.70)
    keys = list(docs.keys())

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            pair = tuple(sorted([k1, k2]))
            is_gt_positive = pair in gt_positives

            rel = detector.evaluate_pair(k1, docs[k1], k2, docs[k2])
            is_pred_positive = rel.dependence_type in (
                DependenceType.EXACT_DUPLICATE,
                DependenceType.NEAR_DUPLICATE,
            )

            if is_pred_positive and is_gt_positive:
                tp += 1
            elif is_pred_positive and not is_gt_positive:
                fp += 1
            elif not is_pred_positive and not is_gt_positive:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    assert fp == 0, f"Expected 0 false positives, got {fp}"
    assert precision >= 0.95, f"Precision {precision:.4f} below 95% threshold"
    assert recall >= 0.90, f"Recall {recall:.4f} below 90% threshold"
    assert tp == 10, f"Expected all 10 syndicated pairs detected, got {tp}"


def test_cluster_level_isolation_of_independent_reporting():
    """Verify that detect_syndication clusters all 5 syndicated copies together and keeps independent articles isolated."""
    docs = get_labeled_evaluation_dataset()
    detector = SyndicationDetector(similarity_threshold=0.70)

    result = detector.detect_syndication(docs)

    # There must be exactly ONE cluster (the syndicated wire group)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]

    expected_members = {"d1_wire", "d2_html", "d3_reprint1", "d4_reprint2", "d5_reprint3"}
    assert cluster.member_source_ids == expected_members

    # None of the independent or boilerplate documents should be in the cluster
    for independent_id in ["d6_techcrunch", "d7_reuters", "d8_bloomberg", "d9_boilerplate_other", "d10_unrelated"]:
        assert independent_id not in cluster.member_source_ids
