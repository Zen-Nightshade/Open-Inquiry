from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.db.state_store import StateStore
from src.researcher.entity.service import EntityResolutionService
from src.researcher.evidence_request.models import (
    BiasedEvidenceRequestError,
    EvidenceFeedbackBudget,
    EvidenceRequest,
    EvidenceRequestStatus,
    FeedbackRoundResult,
)
from src.researcher.evidence_request.validator import check_commercial_bias
from src.researcher.extraction.models import FetchStatus
from src.researcher.extraction.service import FactExtractionService
from src.researcher.models import CanonicalFact, EntityRecord, ResearchAccumulator, ResolutionStatus, SourceRecord
from src.researcher.orchestration.orchestrator import Researcher
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.syndication.detector import SyndicationDetector
from src.researcher.syndication.minhash import compute_minhash_signature
from src.researcher.syndication.normalizer import (
    compute_normalized_content_hash,
    compute_raw_content_hash,
    normalize_source_text,
)
from src.researcher.syndication.shingling import generate_5gram_shingles, tokenize

logger = logging.getLogger(__name__)


class EvidenceRequestExecutor:
    """Executes EvidenceRequests independently in the Researcher domain without commercial bias."""

    def __init__(
        self,
        researcher: Researcher,
        state_store: StateStore,
    ) -> None:
        self.researcher = researcher
        self.state_store = state_store

    def execute_requests(
        self,
        requests: List[EvidenceRequest],
        session_id: str,
        run_id: str,
        budget: EvidenceFeedbackBudget,
        accumulator: Optional[ResearchAccumulator] = None,
        round_number: int = 1,
    ) -> FeedbackRoundResult:
        """Execute a batch of EvidenceRequests within the remaining feedback budget.
        
        Strictly enforces:
          - Only 1 feedback round allowed.
          - Max 3 requests per round.
          - 15% budget cap (budget.can_consume).
          - No binding of facts to ambiguous or unresolved entities.
          - Neutral missing outcome on empty search results (not NO_MATCH or global absence).
        """
        active_acc = accumulator or ResearchAccumulator()
        current_budget = budget

        # Check recursion / round limit
        if round_number > 1 or not current_budget.can_execute_round():
            logger.warning("Feedback round limit reached (round %d, max %d); deferring all requests.", round_number, current_budget.max_feedback_rounds)
            deferred = [
                req.model_copy(update={
                    "status": EvidenceRequestStatus.DEFERRED,
                    "rejection_reason": "Maximum feedback rounds exceeded; recursive feedback prohibited.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                for req in requests
            ]
            return FeedbackRoundResult(
                round_number=round_number,
                run_id=run_id,
                requests=deferred,
                budget=current_budget,
                status="deferred",
                summary="Feedback round limit reached; all requests deferred.",
            )

        executed_requests: List[EvidenceRequest] = []
        newly_extracted_fact_ids: List[str] = []

        # Enforce max requests per round
        allowed_requests = requests[:current_budget.max_requests_per_round]
        overflow_requests = requests[current_budget.max_requests_per_round:]

        for req in allowed_requests:
            now_iso = datetime.now(timezone.utc).isoformat()

            # 1. Deterministic Commercial Bias Check
            is_q_biased, q_bias_reason = check_commercial_bias(req.neutral_question)
            if is_q_biased:
                logger.warning("EvidenceRequest %s rejected for commercial bias: %s", req.request_id, q_bias_reason)
                executed_requests.append(
                    req.model_copy(update={
                        "status": EvidenceRequestStatus.REJECTED_BIASED,
                        "rejection_reason": q_bias_reason,
                        "completed_at": now_iso,
                    })
                )
                continue

            # 2. Target Entity Verification (cannot bind facts to ambiguous entity)
            with self.state_store.uow() as uow:
                db_ent = uow.entities.get_entity(req.target_entity_id)

            if db_ent:
                status_str = db_ent.get("resolution_status", "unresolved").lower()
                if status_str in ("ambiguous", "unresolved"):
                    reason = f"Target entity {req.target_entity_id} has {status_str} status; cannot bind factual assertions."
                    logger.warning("EvidenceRequest %s cannot proceed: %s", req.request_id, reason)
                    executed_requests.append(
                        req.model_copy(update={
                            "status": EvidenceRequestStatus.UNRESOLVED,
                            "rejection_reason": reason,
                            "completed_at": now_iso,
                        })
                    )
                    continue

            # 3. Budget Check
            if not current_budget.can_consume(req.budget_cost_units):
                reason = (
                    f"Budget exhausted: executing request requiring {req.budget_cost_units} units would exceed "
                    f"feedback cap ({current_budget.feedback_units_consumed + req.budget_cost_units} > {current_budget.feedback_budget_cap})."
                )
                logger.warning("EvidenceRequest %s deferred: %s", req.request_id, reason)
                executed_requests.append(
                    req.model_copy(update={
                        "status": EvidenceRequestStatus.BUDGET_EXHAUSTED,
                        "rejection_reason": reason,
                        "completed_at": now_iso,
                    })
                )
                continue

            # 4. Consume Budget deterministically
            current_budget = current_budget.with_consumption(cost_units=req.budget_cost_units)

            # 5. Build Balanced Search Queries (Neutral + Disconfirming)
            queries_to_run: List[str] = []
            if req.neutral_search_terms:
                queries_to_run.append(req.neutral_search_terms[0])
            if req.disconfirming_search_terms:
                queries_to_run.append(req.disconfirming_search_terms[0])

            # Persist objective & queries for auditability
            objective_id = f"obj_feed_{req.request_id[:24]}"
            target_ent = "both"
            if req.target_entity_id in ("prospect", "company", "both"):
                target_ent = req.target_entity_id
            elif "prospect" in req.target_entity_id.lower() or "person" in req.target_entity_id.lower():
                target_ent = "prospect"
            else:
                target_ent = "company"

            temp_scope_str = None
            if req.valid_from or req.valid_to:
                temp_scope_str = f"{req.valid_from or ''}..{req.valid_to or ''}"

            if session_id:
                try:
                    with self.state_store.uow() as uow:
                        uow.research.create_objective(
                            objective_id=objective_id,
                            session_id=session_id,
                            type="neutral_discovery",
                            target_entity=target_ent,
                            description=req.neutral_question,
                            temporal_scope=temp_scope_str,
                            priority=1,
                            is_critical=False,
                        )
                        for q_text in queries_to_run:
                            uow.research.create_query(
                                query_id=f"q_feed_{uuid.uuid4().hex[:8]}",
                                session_id=session_id,
                                objective_id=objective_id,
                                query_text=q_text,
                            )
                except Exception as db_exc:
                    logger.warning("Could not persist evidence request objective/queries: %s", db_exc)

            # 6. Execute Searches and Fetches
            fetched_snapshot_ids: List[str] = []
            searched_scope: List[str] = list(queries_to_run)
            op_failed = False
            op_fail_reason: Optional[str] = None

            has_search = hasattr(self.researcher, "search_client") and self.researcher.search_client is not None
            has_fetcher = hasattr(self.researcher, "page_fetcher") and self.researcher.page_fetcher is not None

            if not has_search:
                op_failed = True
                op_fail_reason = "No search client configured on researcher."
            else:
                for q_str in queries_to_run:
                    try:
                        s_resp = self.researcher.search_client.search(query=q_str, max_results=3)
                        if s_resp.status == SearchStatus.FAILED:
                            op_failed = True
                            op_fail_reason = s_resp.error_message or "Search provider returned failure status."
                            continue
                        elif s_resp.status == SearchStatus.EMPTY:
                            continue

                        if not has_fetcher:
                            continue

                        for res in s_resp.results:
                            if not res.url:
                                continue
                            f_outcome = self.researcher.page_fetcher.fetch_page(res.url)
                            if f_outcome.status == FetchStatus.SUCCESS and f_outcome.source_record:
                                sr = f_outcome.source_record
                                snap_id = f"snap_feed_{uuid.uuid4().hex[:8]}"
                                raw_hash = compute_raw_content_hash(sr.raw_text)
                                norm_hash = compute_normalized_content_hash(sr.raw_text)
                                norm_text = normalize_source_text(sr.raw_text)
                                tokens = tokenize(norm_text)
                                shingles = generate_5gram_shingles(tokens)
                                minhash_sig = compute_minhash_signature(shingles)

                                with self.state_store.uow() as uow:
                                    uow.sources.upsert_source(
                                        source_id=sr.source_id,
                                        canonical_url=getattr(sr, "canonical_url", sr.url),
                                        domain=sr.domain,
                                    )
                                    uow.sources.create_snapshot(
                                        snapshot_id=snap_id,
                                        source_id=sr.source_id,
                                        fetch_status="success",
                                        raw_text=sr.raw_text,
                                        http_status_code=f_outcome.http_status_code or 200,
                                        duration_ms=f_outcome.duration_ms,
                                        raw_content_hash=raw_hash,
                                        normalized_content_hash=norm_hash,
                                        minhash_signature=json.dumps(list(minhash_sig)),
                                    )
                                fetched_snapshot_ids.append(snap_id)
                                active_acc.add_source(sr)

                    except Exception as exc:
                        logger.error("Error executing query '%s' for request %s: %s", q_str, req.request_id, exc)
                        op_failed = True
                        op_fail_reason = str(exc)

            # 7. Apply M2.3 Syndication Detection on all accumulated sources
            if len(active_acc.sources) >= 2 and hasattr(self.researcher, "syndication_detector") and self.researcher.syndication_detector:
                synd_res = self.researcher.syndication_detector.detect_syndication(active_acc.sources)
                if synd_res.clusters:
                    with self.state_store.uow() as uow:
                        for cluster in synd_res.clusters:
                            uow.sources.save_syndication_cluster(cluster)

            # 8. Fact Extraction across new snapshots
            resulting_fact_ids: List[str] = []
            if hasattr(self.researcher, "fact_service") and self.researcher.fact_service:
                for snap_id in fetched_snapshot_ids:
                    extracted = self.researcher.fact_service.extract_and_persist_facts_for_session(
                        session_id=session_id,
                        snapshot_id=snap_id,
                        accumulator=active_acc,
                    )
                    for f in extracted:
                        if f.fact_id not in resulting_fact_ids:
                            resulting_fact_ids.append(f.fact_id)
                        if f.fact_id not in newly_extracted_fact_ids:
                            newly_extracted_fact_ids.append(f.fact_id)

            # 9. Determine Request Outcome
            if op_failed and not fetched_snapshot_ids:
                status = EvidenceRequestStatus.FAILED
                reason = f"Operational failure during execution: {op_fail_reason}"
            elif resulting_fact_ids:
                status = EvidenceRequestStatus.COMPLETED
                reason = None
            else:
                # No relevant evidence found within examined scope: Neutral UNRESOLVED outcome (not NO_MATCH)
                status = EvidenceRequestStatus.UNRESOLVED
                reason = "No verifiable evidence found within examined sources."

            completed_req = req.model_copy(update={
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "resulting_fact_ids": resulting_fact_ids,
                "resulting_source_ids": [s.source_id for s in active_acc.sources if s.source_id in fetched_snapshot_ids],
                "searched_scope": searched_scope,
                "rejection_reason": reason,
            })
            executed_requests.append(completed_req)

        # Append any requests exceeding the per-round cap as DEFERRED
        for ov_req in overflow_requests:
            executed_requests.append(
                ov_req.model_copy(update={
                    "status": EvidenceRequestStatus.DEFERRED,
                    "rejection_reason": f"Exceeded maximum of {current_budget.max_requests_per_round} requests per feedback round.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
            )

        # Mark round completed in budget
        current_budget = current_budget.with_round_completed()

        # Determine round status
        has_completed = any(r.status == EvidenceRequestStatus.COMPLETED for r in executed_requests)
        has_exhausted = any(r.status == EvidenceRequestStatus.BUDGET_EXHAUSTED for r in executed_requests)
        has_failed = any(r.status == EvidenceRequestStatus.FAILED for r in executed_requests)

        round_status = "completed"
        if has_exhausted:
            round_status = "exhausted" if not has_completed else "partial"
        elif has_failed and not has_completed:
            round_status = "failed"
        elif any(r.status == EvidenceRequestStatus.DEFERRED for r in executed_requests) and has_completed:
            round_status = "partial"

        summary = (
            f"Executed feedback round 1 with {len(executed_requests)} requests. "
            f"Budget expended: {current_budget.feedback_units_consumed}/{current_budget.feedback_budget_cap} units. "
            f"Produced {len(newly_extracted_fact_ids)} new facts."
        )

        return FeedbackRoundResult(
            round_number=round_number,
            run_id=run_id,
            requests=executed_requests,
            budget=current_budget,
            status=round_status,
            summary=summary,
        )
