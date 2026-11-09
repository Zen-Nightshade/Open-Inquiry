from typing import Optional, Any, Dict
from ..prompting import Prompt
from ..decoding import DecodeConfig


class GenerationRequest:
    def __init__(
        self,
        prompt: Prompt,
        decoding: Optional[DecodeConfig] = None,
        provider_name: Optional[str] = None,
        provider_overrides: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Any] = None,
        tool_definition: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.prompt = prompt
        self.decoding = decoding
        self.provider_name = provider_name
        self.provider_overrides = provider_overrides or {}
        self.output_schema = output_schema
        self.tool_definition = tool_definition
        self.metadata = metadata or {}