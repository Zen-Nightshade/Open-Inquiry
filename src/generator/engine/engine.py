from ..provider import ProviderFactory
from .request import GenerationRequest
from .response import GenerationResponse
from .context import ExecutionContext
from .resolver import ProviderResolver
from .executor import GenerationExecutor
from .parser import OutputParser


class GeneratorEngine:

    def __init__(self, default_provider: str):
        provider_factory = ProviderFactory()

        self.resolver = ProviderResolver(
            provider_factory=provider_factory,
            default_provider=default_provider,
        )

        self.executor = GenerationExecutor()
        self.parser = OutputParser()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        context = ExecutionContext(request)

        # Resolve provider and decoding
        self.resolver.resolve_into(context)

        # Execute provider call
        self.executor.execute(context)

        # Parse structured output (if any)
        self.parser.parse(context)

        # Build final response
        provider_response = context.provider_response

        metadata = {
            "latency_ms": context.latency_ms,
            **request.metadata,
        }

        return GenerationResponse(
            text=provider_response.text,
            provider_name=context.resolved_provider_name,
            raw_response=provider_response,
            usage=provider_response.usage,
            parsed_output=context.parsed_output,
            metadata=metadata,
        )