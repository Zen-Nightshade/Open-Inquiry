from __future__ import annotations

import json
import re
from typing import Generic, Type, TypeVar
from pydantic import BaseModel, ValidationError

from .base_parser import BaseParser

T = TypeVar("T", bound=BaseModel)


class PydanticParser(BaseParser, Generic[T]):
    """Structured output parser for Pydantic models.

    Converts LLM response text into validated Pydantic model instances.
    Implements the .parse(text) protocol expected by GeneratorEngine's OutputParser.
    """

    def __init__(self, output_schema: Type[T]):
        if not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise TypeError(
                f"output_schema must be a Pydantic BaseModel class, got {output_schema!r}"
            )
        self.model_class: Type[T] = output_schema

    def parse(self, text: str) -> T:
        """Parse raw response text into a validated Pydantic model instance."""
        if not text or not text.strip():
            raise ValueError("Cannot parse empty response text into Pydantic model")

        cleaned = self._extract_json(text)
        return self.model_class.model_validate_json(cleaned)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract valid JSON substring from markdown fences or conversational preambles."""
        stripped = text.strip()

        # 1. Strip markdown fences if present
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            if candidate:
                return candidate

        # 2. Direct JSON check
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            return stripped

        # 3. Find outermost JSON object or array
        start_obj = stripped.find("{")
        end_obj = stripped.rfind("}")
        start_arr = stripped.find("[")
        end_arr = stripped.rfind("]")

        # Determine which enclosure comes first and forms a valid boundary
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            if start_arr == -1 or start_obj < start_arr:
                return stripped[start_obj : end_obj + 1]

        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return stripped[start_arr : end_arr + 1]

        # Fallback to stripped text to let model_validate_json surface the error
        return stripped
