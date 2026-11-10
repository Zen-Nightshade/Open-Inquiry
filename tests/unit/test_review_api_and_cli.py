from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from src.api.app import create_app
from src.cli.main import app as cli_app
from src.db.state_store import StateStore
from src.email_generation.review_models import RejectionReason

runner = CliRunner()


@pytest.fixture
def review_setup():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_cli_review_test.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        # Create 2 drafts
        with store.uow() as uow:
            uow.outreach.create_prospect(
                prospect_id="p-1",
                full_name="Jane Doe",
                company_name="Acme Corp",
            )
            uow.products.create_product_version(
                version_id="pv-1",
                product_name="PulseRadar",
                snapshot_json="{}",
            )
            uow.outreach.create_outreach_context(
                context_id="ctx-1",
                prospect_id="p-1",
                version_id="pv-1",
                context_json="{}",
            )
            uow.outreach.create_outreach_draft(
                draft_id="draft-101",
                context_id="ctx-1",
                research_session_id="sess-101",
                subject="Cluster Telemetry Inquiry",
                body="Initial body copy for testing review workflow.",
                word_count=7,
                product_snapshot_json="{}",
                claim_verifications="[]",
                quality_evaluation="{}",
                human_review_status="pending_review",
                current_revision=1,
            )
            uow.outreach.create_outreach_draft(
                draft_id="draft-102",
                context_id="ctx-1",
                research_session_id="sess-102",
                subject="Second Draft Subject",
                body="Second body copy.",
                word_count=3,
                product_snapshot_json="{}",
                claim_verifications="[]",
                quality_evaluation="{}",
                human_review_status="pending_review",
                current_revision=1,
            )

        app = create_app(db_path=db_path)
        client = TestClient(app)
        yield db_path, client


# =============================================================================
# API Endpoint Tests
# =============================================================================

def test_api_list_drafts(review_setup):
    _, client = review_setup
    res = client.get("/api/v1/drafts")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 2
    assert len(data["drafts"]) == 2
    assert data["limit"] == 50
    assert data["offset"] == 0


def test_api_get_draft(review_setup):
    _, client = review_setup
    res = client.get("/api/v1/drafts/draft-101")
    assert res.status_code == 200
    assert res.json()["draft_id"] == "draft-101"

    # Non-existent
    res_404 = client.get("/api/v1/drafts/non-existent-draft")
    assert res_404.status_code == 404


def test_api_edit_draft_lifecycle(review_setup):
    _, client = review_setup

    # Edit draft-101 (rev 1 -> rev 2)
    edit_payload = {
        "expected_revision": 1,
        "new_subject": "Revised Telemetry Subject",
        "new_body": "Updated body copy with benchmark numbers.",
        "reviewer_id": "api-reviewer",
        "notes": "Added benchmark metrics",
    }
    res = client.post("/api/v1/drafts/draft-101/edit", json=edit_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "EDITED"
    assert data["reviewed_revision"] == 1
    assert data["resulting_revision"] == 2
    assert data["char_edit_distance"] > 0
    assert data["word_edit_distance"] > 0

    # Draft remains pending_review
    res_draft = client.get("/api/v1/drafts/draft-101")
    assert res_draft.json()["human_review_status"] == "pending_review"
    assert res_draft.json()["current_revision"] == 2

    # Attempting to edit with stale revision 1 produces 409
    res_conflict = client.post("/api/v1/drafts/draft-101/edit", json=edit_payload)
    assert res_conflict.status_code == 409

    # Approve on revision 2
    approve_payload = {
        "expected_revision": 2,
        "reviewer_id": "api-approver",
        "notes": "Approved revised draft",
    }
    res_app = client.post("/api/v1/drafts/draft-101/approve", json=approve_payload)
    assert res_app.status_code == 200
    assert res_app.json()["action"] == "APPROVED"
    assert res_app.json()["reviewed_revision"] == 2

    # History contains 2 revisions and 2 events
    res_hist = client.get("/api/v1/drafts/draft-101/history")
    assert res_hist.status_code == 200
    hist = res_hist.json()
    assert hist["status"] == "approved"
    assert len(hist["revisions"]) == 2
    assert len(hist["events"]) == 2


def test_api_reject_draft_with_and_without_reason(review_setup):
    _, client = review_setup

    # Rejection without reason fails validation (422)
    bad_payload = {
        "expected_revision": 1,
        "notes": "No reason provided",
    }
    res_bad = client.post("/api/v1/drafts/draft-102/reject", json=bad_payload)
    assert res_bad.status_code == 422

    # Rejection with valid reason succeeds
    good_payload = {
        "expected_revision": 1,
        "reason_code": "HALLUCINATION",
        "reviewer_id": "qa-reviewer",
        "notes": "Exaggerated cluster size claim",
    }
    res_good = client.post("/api/v1/drafts/draft-102/reject", json=good_payload)
    assert res_good.status_code == 200
    assert res_good.json()["action"] == "REJECTED"
    assert res_good.json()["reason_code"] == "HALLUCINATION"


def test_api_telemetry_endpoint(review_setup):
    _, client = review_setup

    # Approve draft-101, reject draft-102
    client.post("/api/v1/drafts/draft-101/approve", json={"expected_revision": 1})
    client.post(
        "/api/v1/drafts/draft-102/reject",
        json={"expected_revision": 1, "reason_code": "TONE_MISMATCH"},
    )

    res = client.get("/api/v1/reviews/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["total_drafts_reviewed"] == 2
    assert data["approved_count"] == 1
    assert data["rejected_count"] == 1
    assert data["approval_rate"] == 0.5
    assert data["rejection_rate"] == 0.5
    assert data["percentile_method"] == "linear_interpolation"


# =============================================================================
# CLI Command Tests
# =============================================================================

def test_cli_review_list(review_setup):
    db_path, _ = review_setup
    result = runner.invoke(cli_app, ["review", "list", "--db-path", db_path])
    assert result.exit_code == 0
    assert "draft-101" in result.output
    assert "draft-102" in result.output
    assert "pending_review" in result.output


def test_cli_review_inspect(review_setup):
    db_path, _ = review_setup
    result = runner.invoke(cli_app, ["review", "inspect", "draft-101", "--db-path", db_path])
    assert result.exit_code == 0
    assert "Cluster Telemetry Inquiry" in result.output
    assert "Revision History" in result.output


def test_cli_review_edit_and_approve(review_setup):
    db_path, _ = review_setup

    # Edit
    res_edit = runner.invoke(
        cli_app,
        [
            "review",
            "edit",
            "draft-101",
            "--revision",
            "1",
            "--subject",
            "CLI Subject v2",
            "--body",
            "CLI Body copy v2",
            "--reviewer",
            "cli-user",
            "--db-path",
            db_path,
        ],
    )
    assert res_edit.exit_code == 0
    assert "successfully updated from revision v1 to v2" in res_edit.output
    assert "Char Edit Distance" in res_edit.output

    # Approve
    res_app = runner.invoke(
        cli_app,
        [
            "review",
            "approve",
            "draft-101",
            "--revision",
            "2",
            "--reviewer",
            "cli-approver",
            "--db-path",
            db_path,
        ],
    )
    assert res_app.exit_code == 0
    assert "successfully APPROVED on revision v2" in res_app.output


def test_cli_review_reject(review_setup):
    db_path, _ = review_setup
    res = runner.invoke(
        cli_app,
        [
            "review",
            "reject",
            "draft-102",
            "--revision",
            "1",
            "--reason",
            "WRONG_CAPABILITY",
            "--db-path",
            db_path,
        ],
    )
    assert res.exit_code == 0
    assert "REJECTED on revision v1 due to WRONG_CAPABILITY" in res.output


def test_cli_review_telemetry(review_setup):
    db_path, client = review_setup

    # Record some actions
    client.post("/api/v1/drafts/draft-101/approve", json={"expected_revision": 1})
    client.post("/api/v1/drafts/draft-102/reject", json={"expected_revision": 1, "reason_code": "OTHER"})

    res = runner.invoke(cli_app, ["review", "telemetry", "--db-path", db_path])
    assert res.exit_code == 0
    assert "Human Review Decision Telemetry" in res.output
    assert "Approval Rate" in res.output
    assert "Rejection Rate" in res.output
    assert "Edit Distance Statistics" in res.output
