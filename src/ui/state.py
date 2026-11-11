from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from repo_paths import get_db_path
from src.db.state_store import StateStore
from src.email_generation.review_models import DraftReviewHistory
from src.email_generation.review_service import ReviewService
from src.orchestration.app_orchestrator import ApplicationOrchestrator
from src.orchestration.models import ApplicationRunSnapshot, RunStatus, StageExecutionRecord
from src.ui.redaction import redact_data_structure

logger = logging.getLogger(__name__)


def get_state_store(db_path: Optional[Union[str, Path]] = None) -> StateStore:
    """Return StateStore instance for resolved database path."""
    return StateStore(db_path=get_db_path(db_path))


def get_orchestrator(store: Optional[StateStore] = None) -> ApplicationOrchestrator:
    """Return ApplicationOrchestrator instance."""
    st = store or get_state_store()
    return ApplicationOrchestrator(state_store=st)


def get_review_service(store: Optional[StateStore] = None) -> ReviewService:
    """Return ReviewService instance."""
    st = store or get_state_store()
    return ReviewService(state_store=st)


def get_database_info(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Retrieve database status, table counts, file sizes, and empty-state indicators."""
    target_path = get_db_path(db_path)
    store = StateStore(db_path=target_path)
    status = store.get_database_status()

    file_size_bytes = 0
    if target_path.exists() and str(target_path) != ":memory:":
        file_size_bytes = target_path.stat().st_size

    total_runs = 0
    pending_drafts = 0
    if status["is_initialized"] and status["is_schema_valid"]:
        try:
            with store.uow() as uow:
                # Count total runs
                cur = uow.conn.cursor()
                row = cur.execute("SELECT count(*) FROM application_runs;").fetchone()
                total_runs = row[0] if row else 0

                # Count pending review drafts
                row_drafts = cur.execute(
                    "SELECT count(*) FROM outreach_drafts WHERE human_review_status = 'pending_review';"
                ).fetchone()
                pending_drafts = row_drafts[0] if row_drafts else 0
        except Exception as e:
            logger.debug("Could not count database records: %s", e)

    return {
        "db_path": str(target_path),
        "file_exists": status["file_exists"],
        "file_size_bytes": file_size_bytes,
        "is_initialized": status["is_initialized"],
        "table_count": status["table_count"],
        "is_schema_valid": status["is_schema_valid"],
        "total_runs": total_runs,
        "pending_drafts": pending_drafts,
        "is_empty": total_runs == 0,
    }


def ensure_db_initialized(db_path: Optional[Union[str, Path]] = None) -> None:
    """Ensure database schema is created and valid."""
    store = get_state_store(db_path)
    status = store.get_database_status()
    if not status["file_exists"] or status["table_count"] == 0:
        store.initialize()


def list_runs(limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List historical application runs ordered by started_at DESC."""
    store = get_state_store()
    try:
        return store.list_runs(limit=limit, status=status)
    except Exception as e:
        logger.debug("Failed to list runs: %s", e)
        return []


def get_run_snapshot(run_id: str) -> Optional[ApplicationRunSnapshot]:
    """Retrieve full application run snapshot."""
    orch = get_orchestrator()
    try:
        return orch.get_run(run_id)
    except Exception as e:
        logger.debug("Failed to get run %s: %s", run_id, e)
        return None


def get_run_stages(run_id: str) -> List[StageExecutionRecord]:
    """Retrieve stage execution timeline for a run."""
    orch = get_orchestrator()
    try:
        return orch.get_run_stages(run_id)
    except Exception as e:
        logger.debug("Failed to get stages for run %s: %s", run_id, e)
        return []


def get_strategy_explainability(run_id: str) -> Dict[str, Any]:
    """Retrieve strategy explainability tree connecting claims back to facts and sources."""
    orch = get_orchestrator()
    try:
        return orch.get_strategy_explainability(run_id)
    except Exception as e:
        logger.debug("Failed to get strategy explainability: %s", e)
        return {"error": str(e)}


def get_session_sources(session_id: str) -> List[Dict[str, Any]]:
    """Fetch sources and snapshots explored during a research session."""
    store = get_state_store()
    sources_out: List[Dict[str, Any]] = []
    try:
        with store.uow() as uow:
            snapshots = uow.sources.get_snapshots_for_session(session_id)
            for snp in snapshots:
                src = uow.sources.get_source(snp["source_id"])
                sources_out.append({
                    "snapshot_id": snp["snapshot_id"],
                    "source_id": snp["source_id"],
                    "url": src["canonical_url"] if src else snp.get("url", "Unknown"),
                    "domain": src["domain"] if src else "Unknown",
                    "http_status": snp.get("http_status"),
                    "content_length": len(snp.get("clean_text") or snp.get("raw_content") or ""),
                    "fetched_at": snp.get("fetched_at"),
                    # Disclose credibility score semantics: Step B0 heuristic default vs measured
                    "credibility_score": snp.get("credibility_score", 0.8),
                    "is_heuristic_default": True,
                    "clean_text_preview": (snp.get("clean_text") or "")[:500],
                })
    except Exception as e:
        logger.debug("Error retrieving sources for session %s: %s", session_id, e)
    return sources_out


def get_session_facts_and_evidence(session_id: str) -> List[Dict[str, Any]]:
    """Fetch canonical facts and linked evidence passages for a research session."""
    store = get_state_store()
    facts_out: List[Dict[str, Any]] = []
    try:
        with store.uow() as uow:
            facts = uow.assertions.get_facts_for_session(session_id)
            for f in facts:
                fid = f["fact_id"]
                evidence_list = uow.assertions.get_evidence_for_fact(fid)
                entities_list = uow.assertions.get_entities_for_fact(fid)
                
                passages: List[Dict[str, Any]] = []
                for ev in evidence_list:
                    src = uow.sources.get_source(ev["source_id"])
                    passages.append({
                        "evidence_id": ev["evidence_id"],
                        "verbatim_excerpt": ev["verbatim_excerpt"],
                        "excerpt_validated": bool(ev.get("excerpt_validated")),
                        "event_date": ev.get("event_date"),
                        "credibility_score": ev.get("credibility_score"),
                        "source_url": src["canonical_url"] if src else None,
                        "domain": src["domain"] if src else None,
                    })

                facts_out.append({
                    "fact_id": fid,
                    "claim_text": f["claim_text"],
                    "category": f.get("fact_category") or f.get("category"),
                    "confidence_level": f.get("composite_confidence") or f.get("confidence_level"),
                    "observed_at": f.get("observed_at"),
                    "entities": [e.get("primary_name") for e in entities_list if e.get("primary_name")],
                    "evidence_count": len(passages),
                    "evidence_passages": passages,
                })
    except Exception as e:
        logger.debug("Error retrieving facts for session %s: %s", session_id, e)
    return facts_out


def get_session_queries(session_id: str) -> List[Dict[str, Any]]:
    """Fetch research queries executed during a session."""
    store = get_state_store()
    try:
        with store.uow() as uow:
            queries = uow.research.get_queries(session_id)
            return [dict(q) for q in queries]
    except Exception as e:
        logger.debug("Error retrieving queries for session %s: %s", session_id, e)
        return []


def get_session_entities(session_id: str) -> List[Dict[str, Any]]:
    """Fetch entity resolution candidates and status for a session."""
    store = get_state_store()
    results: List[Dict[str, Any]] = []
    try:
        with store.uow() as uow:
            cur = uow.conn.cursor()
            # Fetch identity assessments for this session or related entities
            rows = cur.execute(
                """
                SELECT ia.*, e.primary_name, e.entity_type
                FROM identity_assessments ia
                LEFT JOIN entities e ON ia.entity_id = e.entity_id
                ORDER BY ia.assessed_at DESC
                LIMIT 20;
                """
            ).fetchall()
            for r in rows:
                results.append(dict(r))
    except Exception as e:
        logger.debug("Error retrieving entities for session: %s", e)
    return results


def get_model_traces(run_id: str, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch model executions for run/job with full redaction of secrets/credentials."""
    store = get_state_store()
    try:
        raw_traces = store.get_model_executions(job_id=job_id, run_id=run_id)
        cleaned: List[Dict[str, Any]] = []
        for t in raw_traces:
            item = dict(t)
            # Redact error messages and parameters
            cleaned_item = redact_data_structure(item)
            cleaned.append(cleaned_item)
        return cleaned
    except Exception as e:
        logger.debug("Error retrieving model executions: %s", e)
        return []


def get_draft_by_id(draft_id: str) -> Optional[Dict[str, Any]]:
    """Fetch raw outreach draft metadata."""
    svc = get_review_service()
    try:
        return svc.get_draft(draft_id)
    except Exception as e:
        logger.debug("Failed to get draft %s: %s", draft_id, e)
        return None


def get_draft_history(draft_id: str) -> Optional[DraftReviewHistory]:
    """Fetch complete revision and review event history for draft."""
    svc = get_review_service()
    try:
        return svc.get_draft_history(draft_id)
    except Exception as e:
        logger.debug("Failed to get draft history %s: %s", draft_id, e)
        return None


def format_iso_timestamp(ts: Optional[str]) -> str:
    """Format ISO 8601 string to human-readable format."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M:%S UTC")
    except Exception:
        return str(ts)


def format_duration(start_iso: Optional[str], end_iso: Optional[str]) -> str:
    """Format duration between two timestamps."""
    if not start_iso or not end_iso:
        return "—"
    try:
        t0 = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        secs = max(0.0, (t1 - t0).total_seconds())
        if secs < 60:
            return f"{secs:.1f}s"
        mins = int(secs // 60)
        rem_secs = int(secs % 60)
        return f"{mins}m {rem_secs}s"
    except Exception:
        return "—"
