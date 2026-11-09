from typing import Optional
from ..provider import ProviderFactory
from .context import ExecutionContext


class ProviderResolver:

    def __init__(self, provider_factory: ProviderFactory, default_provider: str):
        self.provider_factory = provider_factory
        self.default_provider = default_provider

    def resolve_into(self, context: ExecutionContext) -> None:
        request = context.request

        provider_name = request.provider_name or self.default_provider

        provider = self.provider_factory.create(
            provider_name,
            override_kwargs=request.provider_overrides,
        )

        decoding = request.decoding
        if decoding and not provider.capabilities.supports_decoding:
            decoding = None

        context.resolved_provider_name = provider_name
        context.provider_instance = provider
        context.decoding = decoding