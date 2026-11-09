from typing import Optional
from ..base_provider import BaseProvider
from ..provider_capabilities import ProviderCapabilities
from ..provider_response import ProviderResponse
from ..usage_stats import UsageStats
from ...prompting.prompt import Prompt
from ...decoding import DecodeConfig


class GoogleGemmaProvider(BaseProvider):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

        self.capabilities = ProviderCapabilities(
            supports_decoding=True,
            returns_usage=False,  # Google often doesn't return detailed usage
        )

    def generate(
        self,
        prompt: Prompt,
        decoding: Optional[DecodeConfig] = None,
    ) -> ProviderResponse:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for GoogleGemmaProvider. "
                "Install it with `pip install langchain-google-genai`."
            )

        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=decoding.temperature if decoding else None,
            max_output_tokens=decoding.max_tokens if decoding else None,
            top_p=decoding.top_p if decoding else None,
        )

        lc_messages = []

        for msg in prompt.get_messages():
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role_val == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif role_val == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif role_val == "assistant":
                lc_messages.append(AIMessage(content=msg.content))

        response = llm.invoke(lc_messages)

        return ProviderResponse(
            text=response.content,
            usage=None,
            raw=response,
        )
