from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator
from src.orchestration.models import RunStatus, StageExecutionRecord, StageIdentifier, StageStatus
from src.researcher.models import FactCategory
from src.signal_strategy.models import (
    CandidateSignal,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    SignalScore,
    StrategyResult,
)


def _seed_completed_run_with_strategy(store: StateStore):
    orch = ApplicationOrchestrator(state_store=store)
    p = ProspectContext(full_name="Elena Rostova", company_name="Vanguard AI")
    prod = ProductContext(
        product_name="ComputeScale",
        description="Platform",
        capabilities=["Distributed checkpointing", "Dynamic gang scheduling"],
        value_proposition="Cut training costs by 40%",
    )
    out = OutreachContext(
        objective="Intro",
        sender_identity="Seller",
    )
    run_snap = orch.create_pending_run(prospect=p, product=prod, outreach=out, run_id="run-strat-test")
    session_id = run_snap.research_session_id

    # Create fact, source, snapshot and evidence in DB
    fact_id = str(uuid.uuid4())
    src_id = str(uuid.uuid4())
    snap_id = str(uuid.uuid4())
    ev_id = str(uuid.uuid4())
    with store.uow() as uow:
        uow.sources.upsert_source(
            source_id=src_id,
            canonical_url="https://techcrunch.com/vanguard-scaling",
            domain="techcrunch.com",
            source_type="news",
            credibility_score=0.92,
        )
        uow.sources.create_snapshot(
            snapshot_id=snap_id,
            source_id=src_id,
            session_id=session_id,
            fetch_status="success",
            raw_text="Vanguard deployed 1,000 new H100 GPUs to scale foundation model pretraining.",
        )
        uow.assertions.create_canonical_fact(
            fact_id=fact_id,
            session_id=session_id,
            claim_text="Vanguard deployed 1,000 new H100 GPUs",
            fact_category="company_trajectory",
            composite_confidence="high",
            primary_event_date="2026-03-01",
        )
        uow.sources.create_evidence_passage(
            evidence_id=ev_id,
            snapshot_id=snap_id,
            source_id=src_id,
            verbatim_excerpt="Vanguard deployed 1,000 new H100 GPUs to scale foundation model pretraining.",
            excerpt_validated=1,
            credibility_score=0.92,
            event_date="2026-03-01",
        )
        uow.assertions.link_fact_evidence(
            fact_id=fact_id,
            evidence_id=ev_id,
            support_type="supporting",
        )

    # Build StrategyResult
    cand = CandidateSignal(
        signal_id="sig-1",
        label="GPU Cluster Expansion",
        description="Scaling compute cluster by 1,000 H100s",
        supporting_fact_ids=[fact_id],
        fact_category=FactCategory.STRATEGY,
        observed_event="Deployed 1,000 H100s",
        business_implication="Exponential compute cost and checkpoint overhead",
        commercial_opportunity="Need high efficiency fault tolerance",
    )
    score = SignalScore(
        evidence_strength=0.9,
        recency_score=0.95,
        prospect_relevance=0.88,
        product_alignment=0.92,
        business_significance=0.85,
        composite_score=0.91,
    )
    ranked = RankedSignal(
        signal=cand,
        score=score,
        rationale="Top priority due to direct alignment with distributed GPU training offering",
    )
    alignment = ProspectProductAlignment(
        prospect_role_relevance="Elena oversees research infrastructure efficiency",
        business_implications="GPU idle time during crashes costs thousands per hour",
        capability_mapping=["Distributed checkpointing", "Dynamic gang scheduling"],
    )
    strat = OutreachStrategy(
        strategy_id="strat-1",
        primary_signal=ranked,
        supporting_signals=[],
        alignment=alignment,
        messaging_approach="challenge-led",
        central_message="Protect multi-million dollar H100 pretraining runs from cluster fault overhead.",
        suggested_subject_line="H100 cluster checkpoint overhead at Vanguard AI",
        cta="Open to comparing checkpoint recovery benchmarks?",
        tone_guidance="Technical, peer-to-peer",
    )
    strat_result = StrategyResult(
        is_viable=True,
        strategy=strat,
        ranked_signals=[ranked],
    )

    # Finalize run in state store
    stage_rec = StageExecutionRecord(
        stage=StageIdentifier.FINALIZATION,
        status=StageStatus.COMPLETED,
        started_at="2026-03-01T10:00:00Z",
        completed_at="2026-03-01T10:00:01Z",
        duration_ms=1000.0,
    )
    orch._finalize_run(
        run_id=run_snap.run_id,
        job_id=run_snap.job_id,
        status=RunStatus.COMPLETED,
        prospect=p,
        product=prod,
        outreach=out,
        research_session_id=session_id,
        research_package=None,
        strategy_result=strat_result,
        stage_records=[stage_rec],
        warnings=[],
        error_message=None,
        started_at="2026-03-01T10:00:00Z",
    )

    return run_snap.run_id


def test_get_strategy_inspection():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_strat.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        run_id = _seed_completed_run_with_strategy(store)

        app = create_app(db_path=db_path)
        client = TestClient(app)

        res = client.get(f"/api/v1/runs/{run_id}/strategy")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        assert data["messaging_approach"] == "challenge-led"
        assert "Protect multi-million dollar H100" in data["central_message"]
        assert "H100 cluster checkpoint overhead" in data["suggested_subject_line"]
        assert "Open to comparing" in data["cta"]
        assert data["primary_signal"]["signal"]["label"] == "GPU Cluster Expansion"
        assert data["alignment"]["prospect_role_relevance"] == "Elena oversees research infrastructure efficiency"

        # Check evidence chain connecting strategy to source document
        assert len(data["evidence_chain"]) == 1
        ev_item = data["evidence_chain"][0]
        assert "deployed 1,000 new H100 GPUs" in ev_item["claim_text"]
        assert ev_item["domain"] == "techcrunch.com"


def test_get_strategy_not_found():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_strat_nf.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        res = client.get("/api/v1/runs/nonexistent/strategy")
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "NOT_FOUND"
