from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator


def _seed_research_data(store: StateStore):
    orch = ApplicationOrchestrator(state_store=store)
    p = ProspectContext(full_name="Elena Rostova", company_name="Vanguard AI")
    prod = ProductContext(
        product_name="ComputeScale",
        description="Platform",
        capabilities=["GPU sync"],
        value_proposition="Save money",
    )
    out = OutreachContext(
        objective="Intro",
        sender_identity="Seller",
    )
    run_snap = orch.create_pending_run(prospect=p, product=prod, outreach=out, run_id="run-res-test")
    session_id = run_snap.research_session_id

    # Populate session details, fact, evidence, entity, conflict in SQLite
    with store.uow() as uow:
        # Mark session completed
        uow.research.conn.execute(
            "UPDATE research_sessions SET status = 'completed', is_sufficient = 1, is_complete = 1, sufficiency_notes = 'High confidence' WHERE session_id = ?",
            (session_id,),
        )
        # Add objective
        obj_id = str(uuid.uuid4())
        uow.research.create_objective(
            objective_id=obj_id,
            session_id=session_id,
            type="commercial_realization",
            target_entity="company",
            description="Investigate funding and scale",
            priority=1,
            is_critical=True,
        )
        # Add source and snapshot
        src_id = str(uuid.uuid4())
        uow.sources.upsert_source(
            source_id=src_id,
            canonical_url="https://techcrunch.com/vanguard-series-b",
            domain="techcrunch.com",
            source_type="press_release",
            credibility_score=0.95,
        )
        snap_id = str(uuid.uuid4())
        uow.sources.create_snapshot(
            snapshot_id=snap_id,
            source_id=src_id,
            session_id=session_id,
            fetch_status="success",
            raw_text="Vanguard AI secured $45M in Series B financing led by Horizon Ventures.",
        )
        # Add fact
        fact_id = str(uuid.uuid4())
        uow.assertions.create_canonical_fact(
            fact_id=fact_id,
            session_id=session_id,
            claim_text="Vanguard AI raised $45M in Series B funding",
            fact_category="funding",
            composite_confidence="high",
            primary_event_date="2026-03-01",
        )
        # Add evidence passage and link
        ev_id = str(uuid.uuid4())
        uow.sources.create_evidence_passage(
            evidence_id=ev_id,
            snapshot_id=snap_id,
            source_id=src_id,
            verbatim_excerpt="Vanguard AI secured $45M in Series B financing led by Horizon Ventures.",
            excerpt_validated=1,
            credibility_score=0.95,
            event_date="2026-03-01",
        )
        uow.assertions.link_fact_evidence(
            fact_id=fact_id,
            evidence_id=ev_id,
            support_type="supporting",
        )
        # Add entity & assessment
        ent_id = str(uuid.uuid4())
        uow.entities.create_entity(
            entity_id=ent_id,
            raw_name="Elena Rostova",
            entity_type="person",
            canonical_name="Elena Rostova",
        )
        uow.entities.record_identity_assessment(
            assessment_id=str(uuid.uuid4()),
            session_id=session_id,
            entity_id=ent_id,
            candidate_ids="",
            resolution_status="MATCHED",
            confidence="0.98",
            rationale="Exact match with target prospect and company executive roster",
            assessed_by="entity_resolver",
        )
        # Add conflict
        uow.assertions.create_conflict(
            conflict_id=str(uuid.uuid4()),
            session_id=session_id,
            severity="potential",
            description="Conflicting employee count: 120 vs 250",
            status="open",
        )

    return run_snap.run_id


def test_get_research_inspection():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_res.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        run_id = _seed_research_data(store)

        app = create_app(db_path=db_path)
        client = TestClient(app)

        res = client.get(f"/api/v1/runs/{run_id}/research")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        assert data["is_sufficient"] is True
        assert data["is_complete"] is True
        assert data["sufficiency_notes"] == "High confidence"

        # Check facts and passage provenance
        assert len(data["facts"]) == 1
        fact = data["facts"][0]
        assert "raised $45M" in fact["claim_text"]
        assert fact["composite_confidence"] == "high"
        assert len(fact["evidence"]) == 1
        ev = fact["evidence"][0]
        assert "secured $45M" in ev["verbatim_excerpt"]
        assert ev["credibility_score"] == 0.95
        assert ev["excerpt_validated"] is True
        assert ev["source"]["domain"] == "techcrunch.com"
        assert "techcrunch.com/vanguard-series-b" in ev["source"]["canonical_url"]

        # Check identities and conflicts
        assert len(data["identities"]) == 1
        assert data["identities"][0]["raw_name"] == "Elena Rostova"
        assert data["identities"][0]["resolution_status"] == "MATCHED"

        assert len(data["conflicts"]) == 1
        assert "Conflicting employee count" in data["conflicts"][0]["description"]


def test_get_research_not_found():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_res_nf.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        res = client.get("/api/v1/runs/nonexistent/research")
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "NOT_FOUND"
