import os
from dataclasses import dataclass
from typing import Dict, Type, Any

from .base_provider import BaseProvider
from .configurable import GroqProvider
from .configurable import GoogleGemmaProvider
from .configurable import CerebrasProvider


@dataclass
class ProviderConfig:
    provider_name: str
    provider_class: Type[BaseProvider]
    tokenizer: str
    max_context_tokens: int
    env_api_key: str
    default_kwargs: Dict[str, Any]

def resolve_api_key(config: ProviderConfig) -> str:
    key = os.environ.get(config.env_api_key)

    if not key:
        raise ValueError(f"Missing environment variable: {config.env_api_key}")

    return key

PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {

    "groq_gpt_oss_120b": ProviderConfig(
        provider_name="groq",
        provider_class=GroqProvider,
        tokenizer="gpt",
        max_context_tokens=131072,
        env_api_key="GROQ_API_KEY",
        default_kwargs={
            "model_name": "openai/gpt-oss-120b",
        },
    ),

    "groq_qwen3_6_27b": ProviderConfig(
        provider_name="groq",
        provider_class=GroqProvider,
        tokenizer="qwen",
        max_context_tokens=131072,
        env_api_key="GROQ_API_KEY",
        default_kwargs={
            "model_name": "qwen/qwen3.6-27b",
        },
    ),

    "groq_qwen3_8_27b": ProviderConfig(
        provider_name="groq",
        provider_class=GroqProvider,
        tokenizer="qwen",
        max_context_tokens=131072,
        env_api_key="GROQ_API_KEY",
        default_kwargs={
            "model_name": "qwen/qwen3.8-27b",
        },
    ),

    "groq_gpt_oss_20b": ProviderConfig(
        provider_name="groq",
        provider_class=GroqProvider,
        tokenizer="gpt",
        max_context_tokens=131072,
        env_api_key="GROQ_API_KEY",
        default_kwargs={
            "model_name": "openai/gpt-oss-20b",
        },
    ),

    "google_gemma_27b": ProviderConfig(
        provider_name="google",
        provider_class=GoogleGemmaProvider,
        tokenizer="gemma",
        max_context_tokens=8192,
        env_api_key="GOOGLE_API_KEY",
        default_kwargs={
            "model_name": "gemma-3-27b-it",
        },
    ),

    "cerebras_llama_70b": ProviderConfig(
        provider_name="cerebras",
        provider_class=CerebrasProvider,
        tokenizer="llama",
        max_context_tokens=8192,
        env_api_key="CEREBRAS_API_KEY",
        default_kwargs={
            "model_name": "llama-3.3-70b",
        },
    ),
}