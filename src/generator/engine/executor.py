import time
from .context import ExecutionContext


class GenerationExecutor:
    
    def execute(self, context: ExecutionContext) -> None:
        provider = context.provider_instance
        request = context.request

        context.start_time = time.time()

        provider_response = provider.generate(
            request.prompt,
            context.decoding,
        )

        context.end_time = time.time()

        context.provider_response = provider_response