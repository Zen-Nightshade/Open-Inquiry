from langchain import ChatOpenAI # this is incorrect
from langchain_core.messages import HumanMessage, SystemMessage


from ..provider_capabilities import ProviderCapabilities
from ..base_provider import BaseProvider
from ..provider_response import ProviderResponse

from ..usage_stats import UsageStats
from ...prompting import Prompt
from ...decoding import DecodeConfig


class OpenAIProvider(BaseProvider):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.capabilities = ProviderCapabilities(
            supports_decoding=True,
            returns_usage=True,
        )

    def generate(self, prompt: Prompt, decoding:DecodeConfig =None):

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=decoding.temperature if decoding else None,
            max_tokens=decoding.max_tokens if decoding else None,
        )

        lc_messages = []

        for msg in prompt.messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))

        response = llm(lc_messages)

        usage = UsageStats(
            prompt_tokens=response.response_metadata["token_usage"]["prompt_tokens"],
            completion_tokens=response.response_metadata["token_usage"]["completion_tokens"],
            total_tokens=response.response_metadata["token_usage"]["total_tokens"],
        )

        return ProviderResponse(
            text=response.content,
            usage=usage,
            raw=response,
        )

# the version of openai that marker required is differnt from the version tht langchain-openai require