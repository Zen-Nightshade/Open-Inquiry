from __future__ import annotations

import os
import pytest
from pydantic import BaseModel

from src.config.settings import GROQ_LARGE, GROQ_SMALL, init_env
from src.generator import (
    DecodeConfig,
    GenerationRequest,
    GeneratorEngine,
    Message,
    Prompt,
    PydanticParser,
    Role,
)


class MathAnswer(BaseModel):
    operation: str
    result: int
    confidence: float


@pytest.mark.integration
def test_groq_live_structured_completion():
    """Verify live Groq generation with Pydantic structured output parsing."""
    init_env()
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured in environment")

    engine = GeneratorEngine(default_provider=GROQ_SMALL)
    parser = PydanticParser(MathAnswer)

    system_msg = Message(
        role=Role.SYSTEM,
        content=(
            "You are a strict JSON generator. "
            "Respond ONLY with a valid JSON object matching the requested schema: "
            '{"operation": str, "result": int, "confidence": float}. '
            "Do not output markdown fences or explanatory text."
        ),
    )
    user_msg = Message(
        role=Role.USER,
        content="Calculate 12 * 8. Return JSON object with operation='multiplication', result=96, confidence=1.0.",
    )
    prompt = Prompt(messages=[system_msg, user_msg])

    req = GenerationRequest(
        prompt=prompt,
        decoding=DecodeConfig(max_tokens=150, temperature=0.1),
        output_schema=parser,
    )

    response = engine.generate(req)

    assert response.text is not None
    assert response.parsed_output is not None
    assert isinstance(response.parsed_output, MathAnswer)
    assert response.parsed_output.result == 96
    assert response.parsed_output.operation.lower() == "multiplication"
    assert response.parsed_output.confidence == 1.0

    # Verify latency and provider tracking
    assert response.provider_name == GROQ_SMALL
    assert response.metadata.get("latency_ms") is not None
    assert response.metadata["latency_ms"] > 0
