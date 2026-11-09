from ..base_provider import BaseProvider
from ..provider_response import ProviderResponse
from ..provider_capabilities import ProviderCapabilities
from ...errors import GeneratorNotFoundError

class LocalTextProvider(BaseProvider):

    def __init__(self):
        self.capabilities = ProviderCapabilities(
            supports_decoding=False,
            returns_usage=False,
        )

    def generate(self, prompt, decoding=None):
        text_input = prompt.to_plain_text()

        # call local model here
        raise GeneratorNotFoundError("No local model found")
        result_text = ""

        return ProviderResponse(
            text=result_text,
            usage=None,
            raw=None,
        )
