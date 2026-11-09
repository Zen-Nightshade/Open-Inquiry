from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..base_provider import BaseProvider
from ..provider_capabilities import ProviderCapabilities
from ..provider_response import ProviderResponse
from ..usage_stats import UsageStats
from ...prompting.prompt import Prompt, Role
from ...decoding import DecodeConfig


class GroqProvider(BaseProvider):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

        self.capabilities = ProviderCapabilities(
            supports_decoding=True,
            returns_usage=True,
        )

    def generate(
        self,
        prompt: Prompt,
        decoding: Optional[DecodeConfig] = None,
    ) -> ProviderResponse:

        model_kwargs = {}

        if decoding is not None:
            if decoding.top_p is not None:
                model_kwargs["top_p"] = decoding.top_p

            if decoding.presence_penalty is not None:
                model_kwargs["presence_penalty"] = decoding.presence_penalty

            if decoding.frequency_penalty is not None:
                model_kwargs["frequency_penalty"] = decoding.frequency_penalty

        kwargs = {}

        if decoding:
            if decoding.temperature is not None:
                kwargs["temperature"] = decoding.temperature

            if decoding.max_tokens is not None:
                kwargs["max_tokens"] = decoding.max_tokens

            if decoding.stop_sequences:
                kwargs["stop"] = decoding.stop_sequences


            if decoding.seed is not None:
                kwargs["seed"] = decoding.seed

        llm = ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            model_kwargs=model_kwargs,
            **kwargs
        )

        lc_messages = []

        for msg in prompt.get_messages():
            if msg.role ==  Role.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == Role.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))

        response = llm.invoke(lc_messages)

        usage_data = response.response_metadata.get("token_usage", {})

        usage = UsageStats(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )

        return ProviderResponse(
            text=response.content,
            usage=usage,
            raw=response,
        )
