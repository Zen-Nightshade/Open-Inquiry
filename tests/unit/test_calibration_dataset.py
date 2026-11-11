from __future__ import annotations

import sqlite3
import pytest

from src.db.schema import init_schema
from src.evaluation.calibration import (
    MINIMUM_ANNOTATED_DRAFTS_GATE,
    CalibrationDatasetBuilder,
    CalibrationDatasetRecord,
    CalibrationTarget,
    ScoreFamily,
    SplitStrategy,
)


@pytest.fixture
def test_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    return conn


def _setup_draft_environment(
    conn: sqlite3.Connection,
    prospect_id: str,
    company_name: str,
    draft_id: str,
) -> None:
    ctx_id = f"ctx_{draft_id}"
    ver_id = f"ver_{draft_id}"
    sess_id = f"sess_{draft_id}"

    conn.execute(
        "INSERT OR IGNORE INTO prospects (prospect_id, full_name, company_name, created_at) "
        "VALUES (?, ?, ?, '2026-01-01');",
        (prospect_id, f"Contact at {company_name}", company_name),
    )
    conn.execute(
        "INSERT OR IGNORE INTO product_versions (version_id, product_name, snapshot_json, created_at) "
        "VALUES (?, 'Product One', '{}', '2026-01-01');",
        (ver_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO outreach_contexts (context_id, prospect_id, version_id, context_json, created_at) "
        "VALUES (?, ?, ?, '{}', '2026-01-01');",
        (ctx_id, prospect_id, ver_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO outreach_drafts (draft_id, context_id, research_session_id, subject, body, word_count, "
        "product_snapshot_json, claim_verifications, quality_evaluation, current_revision, created_at) "
        "VALUES (?, ?, ?, 'Subject', 'Body text', 25, '{}', '[]', '{}', 1, '2026-01-01');",
        (draft_id, ctx_id, sess_id),
    )


def test_draft_revision_semantics_final_approved(test_db_conn: sqlite3.Connection):
    """Verify that multiple revisions of a draft are resolved by the final reviewed outcome (APPROVED)."""
    conn = test_db_conn
    _setup_draft_environment(conn, "p1", "Acme Corp", "draft_multi_rev")

    # Event 1: revision 1 EDITED
    conn.execute(
        "INSERT INTO human_review_events (event_id, draft_id, action, reviewed_revision, resulting_revision, char_edit_distance, decided_at) "
        "VALUES ('ev1', 'draft_multi_rev', 'EDITED', 1, 2, 45, '2026-01-01T10:00:00');"
    )
    # Event 2: revision 2 EDITED
    conn.execute(
        "INSERT INTO human_review_events (event_id, draft_id, action, reviewed_revision, resulting_revision, char_edit_distance, decided_at) "
        "VALUES ('ev2', 'draft_multi_rev', 'EDITED', 2, 3, 15, '2026-01-01T10:30:00');"
    )
    # Event 3: revision 3 APPROVED
    conn.execute(
        "INSERT INTO human_review_events (event_id, draft_id, action, reviewed_revision, resulting_revision, decided_at) "
        "VALUES ('ev3', 'draft_multi_rev', 'APPROVED', 3, 3, '2026-01-01T11:00:00');"
    )
    conn.commit()

    builder = CalibrationDatasetBuilder()
    records = builder.build_dataset_from_db(conn=conn)

    assert len(records) == 1
    rec = records[0]
    assert rec.draft_id == "draft_multi_rev"
    assert rec.revision_number == 3
    assert rec.label == 1
    assert rec.outcome == "APPROVED"
    assert rec.rejection_reason is None
    assert rec.metadata["event_count"] == 3


def test_draft_revision_semantics_final_rejected(test_db_conn: sqlite3.Connection):
    """Verify that a draft edited and then rejected has label 0 with reason preserved."""
    conn = test_db_conn
    _setup_draft_environment(conn, "p2", "Beta Inc", "draft_rej")

    # Event 1: revision 1 EDITED
    conn.execute(
        "INSERT INTO human_review_events (event_id, draft_id, action, reviewed_revision, resulting_revision, char_edit_distance, decided_at) "
        "VALUES ('ev_r1', 'draft_rej', 'EDITED', 1, 2, 30, '2026-01-01T10:00:00');"
    )
    # Event 2: revision 2 REJECTED with valid schema reason code
    conn.execute(
        "INSERT INTO human_review_events (event_id, draft_id, action, reviewed_revision, resulting_revision, reason_code, decided_at) "
        "VALUES ('ev_r2', 'draft_rej', 'REJECTED', 2, 2, 'WRONG_CAPABILITY', '2026-01-01T11:00:00');"
    )
    conn.commit()

    builder = CalibrationDatasetBuilder()
    records = builder.build_dataset_from_db(conn=conn)

    assert len(records) == 1
    rec = records[0]
    assert rec.draft_id == "draft_rej"
    assert rec.label == 0
    assert rec.outcome == "REJECTED"
    assert rec.rejection_reason == "WRONG_CAPABILITY"


def test_intermediate_edited_without_final_decision_excluded(test_db_conn: sqlite3.Connection):
    """Verify that drafts pending review (final action EDITED) are excluded from binary labels."""
    conn = test_db_conn
    _setup_draft_environment(conn, "p3", "Gamma LLC", "draft_pending")

    conn.execute(
        "INSERT INTO human_review_events (event_id, draft_id, action, reviewed_revision, resulting_revision, char_edit_distance, decided_at) "
        "VALUES ('ev_p1', 'draft_pending', 'EDITED', 1, 2, 20, '2026-01-01T10:00:00');"
    )
    conn.commit()

    builder = CalibrationDatasetBuilder()
    records = builder.build_dataset_from_db(conn=conn)
    assert len(records) == 0


def test_grouped_train_eval_split_prevents_leakage():
    """Verify that grouped splitting never divides revisions/drafts of the same company between train and eval."""
    records = []
    # Create 10 companies, 5 drafts each
    for c_idx in range(10):
        comp_name = f"Company_{c_idx}"
        prospect_id = f"p_{c_idx}"
        for d_idx in range(5):
            records.append(
                CalibrationDatasetRecord(
                    draft_id=f"draft_{c_idx}_{d_idx}",
                    revision_number=1,
                    prospect_id=prospect_id,
                    company_name=comp_name,
                    raw_score=0.5 + 0.05 * c_idx,
                    label=1 if (c_idx + d_idx) % 2 == 0 else 0,
                    outcome="APPROVED" if (c_idx + d_idx) % 2 == 0 else "REJECTED",
                )
            )

    train_recs, eval_recs = CalibrationDatasetBuilder.split_grouped(
        records=records,
        eval_ratio=0.20,
        group_by=SplitStrategy.COMPANY_GROUPED,
        seed=123,
    )

    assert len(train_recs) > 0
    assert len(eval_recs) > 0
    assert len(train_recs) + len(eval_recs) == len(records)

    train_companies = {r.company_name for r in train_recs}
    eval_companies = {r.company_name for r in eval_recs}

    # Strict zero-leakage check: the intersection must be completely empty!
    leakage = train_companies.intersection(eval_companies)
    assert len(leakage) == 0, f"Found company leakage across splits: {leakage}"


def test_threshold_readiness_gate():
    """Verify that check_dataset_readiness blocks fitting when below 200 annotated drafts."""
    # 1. 50 drafts (below gate)
    small_records = [
        CalibrationDatasetRecord(
            draft_id=f"d_{i}",
            revision_number=1,
            prospect_id=f"p_{i}",
            company_name=f"Comp_{i}",
            raw_score=0.7,
            label=1 if i % 2 == 0 else 0,
            outcome="APPROVED" if i % 2 == 0 else "REJECTED",
        )
        for i in range(50)
    ]

    is_ready, rationale, stats = CalibrationDatasetBuilder.check_dataset_readiness(
        small_records, gate_threshold=MINIMUM_ANNOTATED_DRAFTS_GATE
    )
    assert is_ready is False
    assert "below the mandatory threshold of 200" in rationale
    assert stats["total_annotated_drafts"] == 50

    # 2. 200 drafts with both classes (satisfies gate)
    adequate_records = [
        CalibrationDatasetRecord(
            draft_id=f"d_{i}",
            revision_number=1,
            prospect_id=f"p_{i}",
            company_name=f"Comp_{i}",
            raw_score=0.7,
            label=1 if i % 2 == 0 else 0,
            outcome="APPROVED" if i % 2 == 0 else "REJECTED",
        )
        for i in range(200)
    ]

    is_ready_200, rationale_200, stats_200 = CalibrationDatasetBuilder.check_dataset_readiness(
        adequate_records, gate_threshold=MINIMUM_ANNOTATED_DRAFTS_GATE
    )
    assert is_ready_200 is True
    assert "satisfies readiness gate" in rationale_200
    assert stats_200["total_annotated_drafts"] == 200
