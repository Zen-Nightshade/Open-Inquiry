from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from src.cli.main import app
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration import (
    ApplicationOrchestrator,
    RunResult,
    RunStatus,
    StageExecutionRecord,
    StageIdentifier,
    StageStatus,
)
from src.researcher.models import FactCategory
from src.signal_strategy.models import (
    CandidateSignal,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    SignalScore,
    StrategyResult,
)

runner = CliRunner()


def test_cli_db_init_and_status():
    with tempfile.NamedTemporaryFile() as tf:
        # Run db init
        res_init = runner.invoke(app, ["db", "init", "--db-path", tf.name])
        assert res_init.exit_code == 0
        assert "All 36 domain tables verified" in res_init.output

        # Run db status
        res_status = runner.invoke(app, ["db", "status", "--db-path", tf.name])
        assert res_status.exit_code == 0
        assert "Database Status" in res_status.output
        assert "36 / 36" in res_status.output


def test_cli_run_with_input_file(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "test_cli.db")
        input_path = Path(td) / "input.json"

        # Initialize db
        runner.invoke(app, ["db", "init", "--db-path", db_path])

        input_data = {
            "prospect": {
                "full_name": "Elena Rostova",
                "company_name": "Vanguard AI",
                "designation": "Head of Research",
                "company_domain": "vanguardai.io",
            },
            "product": {
                "product_name": "Zamp Intelligence",
                "description": "Automated research engine",
                "capabilities": ["Fact extraction", "Signal detection"],
                "value_proposition": "Turn research into verified opportunities",
            },
            "outreach": {
                "objective": "Introductory briefing",
                "sender_identity": "Alex Rivers, Architect",
                "sender_name": "Alex Rivers",
                "sender_company": "Zamp AI",
            },
        }
        input_path.write_text(json.dumps(input_data), encoding="utf-8")

        cand = CandidateSignal(
            signal_id="sig_test_1",
            label="Infrastructure Expansion",
            description="Vanguard AI expanded infrastructure in Q3.",
            supporting_fact_ids=["fact_1"],
            fact_category=FactCategory.PRODUCT_LAUNCH,
        )
        score = SignalScore(
            evidence_strength=0.9,
            recency_score=0.9,
            prospect_relevance=0.8,
            product_alignment=0.85,
            business_significance=0.8,
            composite_score=0.85,
        )
        ranked = RankedSignal(signal=cand, score=score, rationale="High recency")
        strat = OutreachStrategy(
            strategy_id="strat_test_1",
            primary_signal=ranked,
            alignment=ProspectProductAlignment(
                prospect_role_relevance="Relevant to research scale",
                business_implications="Compute cost control",
                capability_mapping=["Fact extraction"],
            ),
            messaging_approach="Expansion Congratulation",
            central_message="Scale model training reliably",
            suggested_subject_line="Scaling compute at Vanguard AI",
            cta="15 minute call",
            tone_guidance="Executive and direct",
        )
        mock_strat_result = StrategyResult(is_viable=True, strategy=strat, ranked_signals=[ranked])

        dummy_result = RunResult(
            run_id="run-cli-test-12345",
            status=RunStatus.COMPLETED,
            prospect=ProspectContext(**input_data["prospect"]),
            product=ProductContext(**input_data["product"]),
            outreach=OutreachContext(**input_data["outreach"]),
            research_session_id="session-cli-test",
            strategy_result=mock_strat_result,
            stage_records=[
                StageExecutionRecord(
                    stage=StageIdentifier.VALIDATION,
                    status=StageStatus.COMPLETED,
                    started_at="2026-11-07T20:00:00Z",
                    completed_at="2026-11-07T20:00:01Z",
                    duration_ms=10.0,
                )
            ],
            started_at="2026-11-07T20:00:00Z",
            completed_at="2026-11-07T20:00:05Z",
        )

        with patch.object(ApplicationOrchestrator, "run", return_value=dummy_result):
            res = runner.invoke(app, [
                "run",
                "--input-file", str(input_path),
                "--db-path", db_path,
                "--non-interactive",
            ])
            assert res.exit_code == 0
            assert "COMPLETED" in res.output
            assert "Formulated Outreach Strategy" in res.output
            assert "Infrastructure Expansion" in res.output


def test_cli_run_missing_inputs_non_interactive():
    res = runner.invoke(app, [
        "run",
        "--non-interactive",
    ])
    assert res.exit_code != 0
    assert "Error: Missing required prospect" in res.output


def test_cli_run_list_and_inspect():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "test_cli2.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        prospect = ProspectContext(full_name="Elena Rostova", company_name="Vanguard AI")
        pid = store.save_prospect(prospect)

        cand = CandidateSignal(
            signal_id="sig_test_1",
            label="Infrastructure Expansion",
            description="Vanguard AI expanded infrastructure in Q3.",
            supporting_fact_ids=["fact_1"],
            fact_category=FactCategory.PRODUCT_LAUNCH,
        )
        score = SignalScore(
            evidence_strength=0.9,
            recency_score=0.9,
            prospect_relevance=0.8,
            product_alignment=0.85,
            business_significance=0.8,
            composite_score=0.85,
        )
        strat = OutreachStrategy(
            strategy_id="strat_1",
            primary_signal=RankedSignal(signal=cand, score=score, rationale="High"),
            alignment=ProspectProductAlignment(
                prospect_role_relevance="Relevant",
                business_implications="Impact",
                capability_mapping=["Fact extraction"],
            ),
            messaging_approach="Expansion Congratulation",
            central_message="Scale training",
            suggested_subject_line="Subject",
            cta="CTA",
            tone_guidance="Tone",
        )
        strat_res = StrategyResult(is_viable=True, strategy=strat)

        stages = [
            {
                "stage": "validation",
                "status": "completed",
                "started_at": "2026-11-07T20:00:00Z",
                "completed_at": "2026-11-07T20:00:01Z",
                "duration_ms": 10.0,
                "details": {},
            }
        ]

        run_id = "test-run-inspect-999"
        store.create_run(
            run_id=run_id,
            prospect_id=pid,
            prospect_snapshot_json=prospect.model_dump_json(),
            product_snapshot_json=ProductContext(
                product_name="Zamp",
                description="Engine",
                capabilities=["Fact extraction"],
                value_proposition="VP",
            ).model_dump_json(),
            outreach_snapshot_json=OutreachContext(
                objective="Intro",
                sender_identity="Alex",
            ).model_dump_json(),
            stage_statuses_json=json.dumps(stages),
            strategy_result_json=strat_res.model_dump_json(),
            status="completed",
        )

        # CLI run list
        res_list = runner.invoke(app, ["run", "list", "--db-path", db_path])
        assert res_list.exit_code == 0
        assert "Elena Rostova" in res_list.output
        assert "Vanguard AI" in res_list.output

        # CLI run inspect
        res_insp = runner.invoke(app, ["run", "inspect", run_id, "--db-path", db_path])
        assert res_insp.exit_code == 0
        assert "COMPLETED" in res_insp.output
        assert "Infrastructure Expansion" in res_insp.output

        # CLI run stages
        res_stages = runner.invoke(app, ["run", "stages", run_id, "--db-path", db_path])
        assert res_stages.exit_code == 0
        assert "validation" in res_stages.output

        # CLI strategy inspect
        res_strat = runner.invoke(app, ["strategy", "inspect", run_id, "--db-path", db_path])
        assert res_strat.exit_code == 0
        assert "Strategy Overview" in res_strat.output
        assert "Infrastructure Expansion" in res_strat.output



def test_cli_research_inspect():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "test_cli3.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        session_id = "session-test-cli-777"
        prospect = ProspectContext(full_name="Elena Rostova", company_name="Vanguard AI")
        pid = store.save_prospect(prospect)

        with store.uow() as uow:
            uow.research.create_session(
                session_id=session_id,
                prospect_id=pid,
                context_id="ctx-test",
                status="completed",
                is_sufficient=True,
                is_complete=True,
                sufficiency_notes="Coverage sufficient across all critical goals.",
            )
            uow.research.create_objective(
                objective_id="obj_1",
                session_id=session_id,
                type="neutral_discovery",
                target_entity="company",
                description="Understand core business model",
                priority=1,
                is_critical=1,
            )
            uow.assertions.create_canonical_fact(
                fact_id="fact_1",
                session_id=session_id,
                claim_text="Vanguard AI is an enterprise AI company.",
                fact_category="strategy",
                composite_confidence="HIGH",
            )

        res = runner.invoke(app, ["research", "inspect", session_id, "--db-path", db_path])
        assert res.exit_code == 0
        assert "Research Session session-test-cli-777" in res.output
        assert "Coverage sufficient" in res.output
        assert "Understand core business" in res.output
        assert "enterprise AI company" in res.output


def test_cli_serve_invokes_uvicorn():
    with patch("uvicorn.run") as mock_uvicorn:
        res = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "8008"])
        assert res.exit_code == 0
        assert "Starting Zamp API Server" in res.output
        mock_uvicorn.assert_called_once_with(
            "src.api.app:create_app",
            host="127.0.0.1",
            port=8008,
            reload=False,
            factory=True,
        )

