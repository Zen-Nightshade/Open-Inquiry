from typing import Optional, Any
from ..provider import ProviderResponse, UsageStats

class GenerationResponse:
    def __init__(
        self,
        text: str,
        provider_name: str,
        raw_response: ProviderResponse,
        usage: Optional[UsageStats] = None,
        parsed_output: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ):
        self.text = text
        self.provider_name = provider_name
        self.raw_response = raw_response
        self.usage = usage
        self.parsed_output = parsed_output
        self.metadata = metadata or {}