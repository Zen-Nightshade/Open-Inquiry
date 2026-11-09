from abc import ABC, abstractmethod
from .provider_response import ProviderResponse
from ..prompting import Prompt
from ..decoding import DecodeConfig

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prmopt: Prompt, decoding: DecodeConfig = None) -> ProviderResponse:
        pass