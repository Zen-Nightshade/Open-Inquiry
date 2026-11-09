from ..base_provider import BaseProvider
from ..provider_capabilities import ProviderCapabilities
from ..provider_response import ProviderResponse
from ..usage_stats import UsageStats
from ...prompting.prompt import Prompt
from ...decoding import DecodeConfig

class CerebrasProvider(BaseProvider):

    def __init__(self, model_name: str, api_key: str):
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            raise ImportError(
                "cerebras.cloud.sdk is required for CerebrasProvider. "
                "Install it with `pip install cerebras_cloud_sdk`."
            )
        self.client = Cerebras(api_key=api_key)
        self.model_name = model_name

        self.capabilities = ProviderCapabilities(
            supports_decoding=True,
            returns_usage=True,
        )

    def generate(self, prompt, decoding: DecodeConfig =None):

        messages = [
            {"role": msg.role.value if hasattr(msg.role, "value") else str(msg.role), "content": msg.content}
            for msg in prompt.get_messages()
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=decoding.temperature if decoding else None,
            max_tokens=decoding.max_tokens if decoding else None,
            top_p=decoding.top_p if decoding else None,
        )

        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return ProviderResponse(
            text=response.choices[0].message.content,
            usage=usage,
            raw=response,
        )
