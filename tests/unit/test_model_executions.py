from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock
import pytest

from src.config.settings import GROQ_SMALL
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.observability import (
    active_attempt_num,
    active_job_id,
    active_run_id,
    active_stage_name,
    compute_prompt_hash,
    get_model_executions,
    record_model_execution,
)
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.provider.provider_response import ProviderResponse
from src.generator.provider.usage_stats import UsageStats


def test_prompt_hash_is_sha256_and_does_not_leak_prompt():
    raw_prompt = "Confidential customer details and secret prompt content"
    p_hash = compute_prompt_hash(raw_prompt)
    assert len(p_hash) == 64
    assert raw_prompt not in p_hash


def test_model_executions_recording_in_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = StateStore(db_path)
        store.initialize()

        exec_id = record_model_execution(
            provider="groq",
            model_name="openai/gpt-oss-120b",
            task_type="sufficiency_evaluation",
            prompt_text="Hello world",
            duration_ms=150.5,
            input_tokens=25,
            output_tokens=50,
            status="completed",
            run_id="run-1234-abcd",
            job_id="job-5678-efgh",
            stage_name="research_sufficiency",
            attempt_num=1,
            db_path=db_path,
        )

        assert exec_id is not None

        # Fetch records
        records = get_model_executions(run_id="run-1234-abcd", db_path=db_path)
        assert len(records) == 1
        rec = records[0]
        assert rec["execution_id"] == exec_id
        assert rec["model_name"] == "openai/gpt-oss-120b"
        assert rec["provider"] == "groq"
        assert rec["prompt_hash"] == compute_prompt_hash("Hello world")
        assert rec["input_tokens"] == 25
        assert rec["output_tokens"] == 50
        assert (rec["input_tokens"] + rec["output_tokens"]) == 75
        assert rec["duration_ms"] == 150.5
        assert rec["status"] == "completed"
        assert rec["run_id"] == "run-1234-abcd"
        assert rec["job_id"] is None  # uncommitted job id set to None
        assert rec["stage_name"] == "research_sufficiency"
        assert rec["attempt_num"] == 1
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_generator_engine_records_observability():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = StateStore(db_path)
        store.initialize()

        # Insert a valid job row so foreign key succeeds
        with store.uow() as uow:
            uow.conn.execute(
                "INSERT INTO jobs (job_id, job_type, created_at) VALUES (?, ?, ?);",
                ("job-live-888", "pipeline", "2026-01-01T00:00:00Z"),
            )

        # Set context variables
        token_run = active_run_id.set("run-live-999")
        token_job = active_job_id.set("job-live-888")
        token_stage = active_stage_name.set("entity_resolution")
        token_attempt = active_attempt_num.set(2)

        os.environ["GROQ_API_KEY"] = "mock_key_for_test"
        engine = GeneratorEngine(default_provider=GROQ_SMALL)
        # Mock executor to set context.provider_response
        mock_executor = MagicMock()
        def fake_execute(ctx):
            ctx.provider_response = ProviderResponse(
                text='{"name": "test"}',
                usage=UsageStats(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            )
        mock_executor.execute.side_effect = fake_execute
        engine.executor = mock_executor

        prompt = Prompt()
        prompt.add_message(Message(role=Role.USER, content="Resolve entity Acme"))
        request = GenerationRequest(
            prompt=prompt,
            provider_name=GROQ_SMALL,
            metadata={"db_path": db_path},
        )

        response = engine.generate(request)
        exec_id = response.metadata.get("execution_id")
        assert exec_id is not None

        records = get_model_executions(run_id="run-live-999", db_path=db_path)
        assert len(records) == 1
        assert records[0]["execution_id"] == exec_id
        assert records[0]["status"] == "completed"
        assert records[0]["run_id"] == "run-live-999"
        assert records[0]["job_id"] == "job-live-888"
        assert records[0]["stage_name"] == "entity_resolution"
        assert records[0]["attempt_num"] == 2
        assert records[0]["input_tokens"] == 40
        assert records[0]["output_tokens"] == 20

        # Reset context variables
        active_run_id.reset(token_run)
        active_job_id.reset(token_job)
        active_stage_name.reset(token_stage)
        active_attempt_num.reset(token_attempt)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
