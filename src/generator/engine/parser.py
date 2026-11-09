from .context import ExecutionContext


class OutputParser:
    
    def parse(self, context: ExecutionContext) -> None:
        request = context.request
        provider_response = context.provider_response

        if not request.output_schema:
            return

        parsed = request.output_schema.parse(provider_response.text)
        context.parsed_output = parsed