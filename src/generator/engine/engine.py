from ..observability import record_model_execution
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

        # Extract prompt text for SHA-256 hash
        prompt_text = ""
        if hasattr(request, "prompt") and request.prompt:
            if hasattr(request.prompt, "render"):
                prompt_text = str(request.prompt.render())
            elif hasattr(request.prompt, "messages"):
                prompt_text = " ".join(str(getattr(m, "content", m)) for m in request.prompt.messages)
            else:
                prompt_text = str(request.prompt)

        task_type = request.metadata.get("task", "generation")
        run_id = request.metadata.get("run_id")
        job_id = request.metadata.get("job_id")
        stage_name = request.metadata.get("stage_name")
        attempt_num = request.metadata.get("attempt_num")
        db_path = request.metadata.get("db_path")

        # Resolve provider and decoding
        self.resolver.resolve_into(context)

        # Execute provider call with failure tracking
        try:
            self.executor.execute(context)
        except Exception as exc:
            record_model_execution(
                provider=context.resolved_provider_name or request.provider_name or "unknown",
                model_name=getattr(context.decoding, "model", None) or "unknown",
                task_type=task_type,
                prompt_text=prompt_text,
                duration_ms=context.latency_ms,
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                run_id=run_id,
                job_id=job_id,
                stage_name=stage_name,
                attempt_num=attempt_num,
                db_path=db_path,
            )
            raise

        provider_response = context.provider_response
        is_empty = not provider_response or not bool(getattr(provider_response, "text", "").strip())

        # Parse structured output (if any) with failure tracking
        if not is_empty:
            try:
                self.parser.parse(context)
            except Exception as exc:
                record_model_execution(
                    provider=context.resolved_provider_name or request.provider_name or "unknown",
                    model_name=getattr(context.decoding, "model", None) or "unknown",
                    task_type=task_type,
                    prompt_text=prompt_text,
                    duration_ms=context.latency_ms,
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=f"ParseError: {exc}",
                    run_id=run_id,
                    job_id=job_id,
                    stage_name=stage_name,
                    attempt_num=attempt_num,
                    db_path=db_path,
                )
                raise

        # Record execution for non-exception outcomes
        in_tokens = None
        out_tokens = None
        if provider_response and getattr(provider_response, "usage", None):
            in_tokens = getattr(provider_response.usage, "prompt_tokens", None)
            out_tokens = getattr(provider_response.usage, "completion_tokens", None)

        status = "empty" if is_empty else "completed"
        error_type = "EmptyResponse" if is_empty else None
        error_message = "Empty response returned from model" if is_empty else None

        exec_id = record_model_execution(
            provider=context.resolved_provider_name or request.provider_name or "unknown",
            model_name=getattr(context.decoding, "model", None) or "unknown",
            task_type=task_type,
            prompt_text=prompt_text,
            duration_ms=context.latency_ms,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            status=status,
            error_type=error_type,
            error_message=error_message,
            run_id=run_id,
            job_id=job_id,
            stage_name=stage_name,
            attempt_num=attempt_num,
            db_path=db_path,
        )

        metadata = {
            "latency_ms": context.latency_ms,
            "execution_id": exec_id,
            **request.metadata,
        }

        return GenerationResponse(
            text=provider_response.text if provider_response else "",
            provider_name=context.resolved_provider_name or request.provider_name or "unknown",
            raw_response=provider_response,
            usage=provider_response.usage if provider_response else None,
            parsed_output=context.parsed_output,
            metadata=metadata,
        )