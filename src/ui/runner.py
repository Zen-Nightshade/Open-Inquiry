from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from repo_paths import get_db_path, PROVIDERS_ENV
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.orchestration.app_orchestrator import ApplicationOrchestrator
from src.orchestration.models import RunResult, RunStatus

logger = logging.getLogger(__name__)

# Module-level thread pool and task registry
_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()
_ACTIVE_TASKS: Dict[str, concurrent.futures.Future] = {}
_TASK_ERRORS: Dict[str, str] = {}


def get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return singleton ThreadPoolExecutor for background run execution."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="open_inquiry_worker_",
            )
        return _EXECUTOR


def _worker_task(
    run_id: str,
    prospect: ProspectContext,
    product: ProductContext,
    outreach: OutreachContext,
    db_path: Optional[str] = None,
    max_research_iterations: int = 3,
    allow_insufficient_research: bool = False,
    total_research_budget: int = 20,
    project_id: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> RunResult:
    """Synchronous worker executing the full pipeline inside a background thread."""
    # Ensure environment is initialized
    if api_key_override and api_key_override.strip():
        os.environ["GROQ_API_KEY"] = api_key_override.strip()
    elif not os.environ.get("GROQ_API_KEY") and PROVIDERS_ENV.exists():
        try:
            import dotenv
            dotenv.load_dotenv(PROVIDERS_ENV)
        except Exception as e:
            logger.debug("Could not load PROVIDERS_ENV: %s", e)

    store = StateStore(db_path=get_db_path(db_path))
    orchestrator = ApplicationOrchestrator(state_store=store)

    try:
        logger.info("Starting background execution for run %s", run_id)
        result = orchestrator.run(
            prospect=prospect,
            product=product,
            outreach=outreach,
            run_id=run_id,
            max_research_iterations=max_research_iterations,
            allow_insufficient_research=allow_insufficient_research,
            total_research_budget=total_research_budget,
            project_id=project_id,
        )
        logger.info("Background run %s completed with status: %s", run_id, result.status.value)
        return result
    except Exception as exc:
        logger.error("Fatal unhandled exception in background run %s: %s", run_id, exc, exc_info=True)
        _TASK_ERRORS[run_id] = str(exc)
        # Guarantee run status is updated in SQLite so it does not remain pending/running
        try:
            store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                error_message=f"Fatal unhandled pipeline error: {exc}",
            )
        except Exception as upd_err:
            logger.critical("Failed to record error for run %s: %s", run_id, upd_err)
        raise exc


def start_background_run(
    prospect: ProspectContext,
    product: ProductContext,
    outreach: OutreachContext,
    db_path: Optional[str] = None,
    max_research_iterations: int = 3,
    allow_insufficient_research: bool = False,
    total_research_budget: int = 20,
    project_id: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> str:
    """Validate, persist pending record to SQLite, and launch background pipeline execution."""
    active_run_id = str(uuid.uuid4())
    store = StateStore(db_path=get_db_path(db_path))
    
    # Auto-initialize DB schema if needed
    db_status = store.get_database_status()
    if not db_status["file_exists"] or db_status["table_count"] == 0:
        store.initialize()

    orchestrator = ApplicationOrchestrator(state_store=store)

    # 1. Atomically create pending run and persist validation stage
    orchestrator.create_pending_run(
        prospect=prospect,
        product=product,
        outreach=outreach,
        run_id=active_run_id,
    )

    # 2. Dispatch to background thread pool
    executor = get_executor()
    future = executor.submit(
        _worker_task,
        run_id=active_run_id,
        prospect=prospect,
        product=product,
        outreach=outreach,
        db_path=db_path,
        max_research_iterations=max_research_iterations,
        allow_insufficient_research=allow_insufficient_research,
        total_research_budget=total_research_budget,
        project_id=project_id,
        api_key_override=api_key_override,
    )

    with _EXECUTOR_LOCK:
        _ACTIVE_TASKS[active_run_id] = future

    return active_run_id


def is_run_in_progress(run_id: str, db_path: Optional[str] = None) -> bool:
    """Check if a run is currently executing in background or marked running in SQLite."""
    with _EXECUTOR_LOCK:
        future = _ACTIVE_TASKS.get(run_id)
        if future is not None and not future.done():
            return True

    try:
        store = StateStore(db_path=get_db_path(db_path))
        row = store.get_run(run_id)
        if row and row["status"] in (RunStatus.PENDING.value, RunStatus.RUNNING.value):
            return True
    except Exception:
        pass

    return False


def get_task_error(run_id: str) -> Optional[str]:
    """Retrieve error message from in-memory task dictionary if fatal exception occurred."""
    return _TASK_ERRORS.get(run_id)
