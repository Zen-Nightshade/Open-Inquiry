import time
from typing import Optional
from .request import GenerationRequest
from ..provider import BaseProvider, ProviderResponse


class ExecutionContext:
    def __init__(self, request: GenerationRequest):
        self.request = request

        self.resolved_provider_name: Optional[str] = None
        self.provider_instance: Optional[BaseProvider] = None
        self.decoding = None

        self.provider_response: Optional[ProviderResponse] = None
        self.parsed_output = None

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None