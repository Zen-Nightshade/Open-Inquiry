from ..base_provider import ProviderResponse, BaseProvider
from ...prompting import Prompt
from ...decoding import DecodeConfig

class SimpleTextProvider(BaseProvider):
    def __init__(self, config: dict):
        self.config = config

    def generate(self, prompt: Prompt, decoding: DecodeConfig = None):
        messages = prompt.to_provider_format()