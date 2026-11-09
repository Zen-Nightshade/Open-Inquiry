from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from src.generator.engine.context import ExecutionContext
from src.generator.engine.parser import OutputParser
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.prompt import Prompt
from src.generator.provider.provider_response import ProviderResponse
from src.generator.structured_output.pydantic_parser import PydanticParser


class SampleOutput(BaseModel):
    name: str
    count: int
    tags: list[str] = Field(default_factory=list)


def test_pydantic_parser_constructor_validation():
    # Must be a Pydantic BaseModel subclass
    with pytest.raises(TypeError):
        PydanticParser(dict)  # type: ignore

    with pytest.raises(TypeError):
        PydanticParser("NotAClass")  # type: ignore

    parser = PydanticParser(SampleOutput)
    assert parser.model_class is SampleOutput


def test_pydantic_parser_clean_json():
    parser = PydanticParser(SampleOutput)
    raw = '{"name": "test_object", "count": 42, "tags": ["a", "b"]}'
    res = parser.parse(raw)
    assert isinstance(res, SampleOutput)
    assert res.name == "test_object"
    assert res.count == 42
    assert res.tags == ["a", "b"]


def test_pydantic_parser_markdown_fences():
    parser = PydanticParser(SampleOutput)

    # Markdown with ```json
    fenced_json = """```json
    {
        "name": "fenced_item",
        "count": 10,
        "tags": ["fenced"]
    }
    ```"""
    res1 = parser.parse(fenced_json)
    assert res1.name == "fenced_item"
    assert res1.count == 10

    # Markdown without language tag
    fenced_plain = """```
    {
        "name": "plain_fenced",
        "count": 5
    }
    ```"""
    res2 = parser.parse(fenced_plain)
    assert res2.name == "plain_fenced"
    assert res2.count == 5


def test_pydantic_parser_conversational_preamble_and_postamble():
    parser = PydanticParser(SampleOutput)
    conversational = """
    Sure, here is the structured data you requested:
    
    {
        "name": "extracted_name",
        "count": 99,
        "tags": ["embedded"]
    }
    
    Hope this helps! Let me know if you need anything else.
    """
    res = parser.parse(conversational)
    assert res.name == "extracted_name"
    assert res.count == 99


def test_pydantic_parser_validation_errors():
    parser = PydanticParser(SampleOutput)

    # Empty string
    with pytest.raises(ValueError):
        parser.parse("")

    with pytest.raises(ValueError):
        parser.parse("   ")

    # Invalid JSON syntax
    with pytest.raises(ValidationError):
        parser.parse("Here is no json whatsoever")

    # Missing required field 'count'
    with pytest.raises(ValidationError):
        parser.parse('{"name": "missing_count"}')


def test_output_parser_engine_integration():
    """Verify that OutputParser in generator engine works seamlessly with PydanticParser."""
    parser = PydanticParser(SampleOutput)
    request = GenerationRequest(
        prompt=Prompt(),
        output_schema=parser,
    )
    context = ExecutionContext(request)
    context.provider_response = ProviderResponse(
        text='```json\n{"name": "integrated", "count": 7}\n```'
    )

    engine_parser = OutputParser()
    engine_parser.parse(context)

    assert context.parsed_output is not None
    assert isinstance(context.parsed_output, SampleOutput)
    assert context.parsed_output.name == "integrated"
    assert context.parsed_output.count == 7
