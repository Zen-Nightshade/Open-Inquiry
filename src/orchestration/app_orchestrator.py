from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.email_generation.generator import EmailGenerationError
from src.email_generation.models import VerifiedOutreachDraft
from src.email_generation.service import DraftPersistenceError, EmailGenerationService
from src.researcher.epistemic_dag import DAGAdapter, GlobalResearchBudget
from src.researcher.evidence_request.models import EvidenceFeedbackBudget, FeedbackRoundResult
from src.researcher.evidence_request.executor import EvidenceRequestExecutor
from src.researcher.models import ResearchAccumulator, ResearchPackage, ResolutionStatus
from src.researcher.orchestration.orchestrator import Researcher
from src.researcher.retrieval import HistoricalRetrievalService
from src.signal_strategy.evidence_requests import EvidenceGapDetector
from src.signal_strategy.models import StrategyResult
from src.signal_strategy.service import SignalStrategyService
from .models import (
    ApplicationRunSnapshot,
    RunResult,
    RunStatus,
    StageExecutionRecord,
    StageIdentifier,
    StageStatus,
)

from src.generator.observability import (
    active_job_id,
    active_run_id as ctx_run_id,
    active_stage_name,
)

logger = logging.getLogger(__name__)


class ApplicationOrchestrator:
    """Top-level application orchestrator governing the end-to-end Open Inquiry outreach pipeline.
    
    Responsibilities:
      - Validates domain contexts and inputs.
      - Manages persistent run lifecycle in SQLite (application_runs, jobs, workflow_events).
      - Executes multi-turn research via the Researcher orchestrator without holding DB locks.
      - Enforces Phase F epistemic sufficiency gating (distinguishing facts from sufficient research).
      - Executes commercial signal identification and strategy formulation via SignalStrategyService.
      - Executes operational email draft generation and readability linting via EmailGenerationService.
      - Persists final strategy blueprints, email drafts in PENDING_REVIEW, and stage execution records for audit.
      - Supports run retrieval, stage inspection, and deep evidence provenance after process restart.
    """

    def __init__(
        self,
        state_store: StateStore,
        researcher: Optional[Researcher] = None,
        strategy_service: Optional[SignalStrategyService] = None,
        email_service: Optional[EmailGenerationService] = None,
        retrieval_service: Optional[HistoricalRetrievalService] = None,
    ) -> None:
        self.state_store = state_store
        self.retrieval_service = retrieval_service or HistoricalRetrievalService(state_store=self.state_store)
        self.researcher = researcher or Researcher(
            state_store=self.state_store,
            retrieval_service=self.retrieval_service,
        )
        if getattr(self.researcher, "retrieval_service", None) is None:
            try:
                self.researcher.retrieval_service = self.retrieval_service
            except Exception:
                pass
        self.strategy_service = strategy_service or SignalStrategyService()
        self.email_service = email_service or EmailGenerationService(state_store=self.state_store)

    def run(
        self,
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        run_id: Optional[str] = None,
        max_research_iterations: int = 3,
        allow_insufficient_research: bool = False,
        total_research_budget: int = 20,
        project_id: Optional[str] = None,
    ) -> RunResult:
        """Execute the full end-to-end outreach pipeline with stage-level persistence."""
        # Ensure database is initialized and schema is valid before executing pipeline
        db_status = self.state_store.get_database_status()
        if not db_status["file_exists"] or db_status["table_count"] == 0:
            logger.info("Database not initialized. Auto-initializing schema at %s...", db_status["db_path"])
            self.state_store.initialize()
        elif not db_status["is_initialized"] or not db_status["is_schema_valid"]:
            raise RuntimeError(
                f"Database at {db_status['db_path']} has an incompatible schema: {db_status['incompatibility_details']}. "
                f"Per Open Inquiry manual reset policy, back up your data if needed, delete the database file, and run 'open-inquiry db init'."
            )

        active_run_id = run_id or str(uuid.uuid4())
        existing_run = self.state_store.get_run(active_run_id) if run_id else None

        if existing_run:
            job_id = existing_run.get("job_id") or f"job_{uuid.uuid4().hex}"
            session_id = existing_run.get("research_session_id") or f"session_{uuid.uuid4().hex}"
            context_id = existing_run.get("context_id") or project_id or str(uuid.uuid4())
            prospect_id = existing_run["prospect_id"]
            if existing_run.get("stage_statuses_json"):
                try:
                    loaded = json.loads(existing_run["stage_statuses_json"])
                    stage_records = [StageExecutionRecord.model_validate(s) for s in loaded]
                except Exception:
                    stage_records = []
            else:
                stage_records = []
        else:
            job_id = f"job_{uuid.uuid4().hex}"
            session_id = f"session_{uuid.uuid4().hex}"
            context_id = project_id or str(uuid.uuid4())
            stage_records = []

        # Propagate observability context variables
        ctx_run_id.set(active_run_id)
        active_job_id.set(job_id)
        active_stage_name.set(StageIdentifier.VALIDATION.value)

        warnings: List[str] = []

        start_time_iso = datetime.now(timezone.utc).isoformat()
        current_stage = StageIdentifier.VALIDATION

        # Helper to log and persist stage transitions
        def record_stage(
            stage: StageIdentifier,
            status: StageStatus,
            started_at: str,
            details: Optional[Dict[str, Any]] = None,
            error_message: Optional[str] = None,
        ) -> StageExecutionRecord:
            active_stage_name.set(stage.value)
            completed_at = datetime.now(timezone.utc).isoformat()
            t0 = datetime.fromisoformat(started_at)
            t1 = datetime.fromisoformat(completed_at)
            duration_ms = (t1 - t0).total_seconds() * 1000.0

            rec = StageExecutionRecord(
                stage=stage,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=round(duration_ms, 2),
                details=details or {},
                error_message=error_message,
            )
            stage_records.append(rec)

            # Persist stage event into workflow_events if job/session available
            try:
                with self.state_store.uow() as uow:
                    uow.jobs.record_workflow_event(
                        event_id=str(uuid.uuid4()),
                        session_id=session_id,
                        job_id=job_id,
                        event_type=f"stage_{stage.value}_{status.value}",
                        payload=json.dumps({
                            "stage": stage.value,
                            "status": status.value,
                            "duration_ms": rec.duration_ms,
                            "details": details or {},
                            "error": error_message,
                        }),
                    )
            except Exception as ev_err:
                logger.debug("Could not record workflow event: %s", ev_err)

            return rec

        try:
            # =========================================================================
            # Stage 1: Validation
            # =========================================================================
            current_stage = StageIdentifier.VALIDATION
            s_start = datetime.now(timezone.utc).isoformat()

            if not prospect.full_name or not prospect.company_name:
                raise ValueError("Prospect full_name and company_name are required.")
            if not product.product_name or not product.value_proposition:
                raise ValueError("Product product_name and value_proposition are required.")

            if not existing_run:
                record_stage(
                    stage=StageIdentifier.VALIDATION,
                    status=StageStatus.COMPLETED,
                    started_at=s_start,
                    details={
                        "prospect_name": prospect.full_name,
                        "company_name": prospect.company_name,
                        "product_name": product.product_name,
                    },
                )

            # =========================================================================
            # Stage 2: Initialization & Run Record Creation
            # =========================================================================
            current_stage = StageIdentifier.INITIALIZATION
            s_start = datetime.now(timezone.utc).isoformat()

            if existing_run:
                # Update existing pending run and job to running status
                self.state_store.update_run(
                    run_id=active_run_id,
                    status=RunStatus.RUNNING.value,
                )
                if job_id:
                    try:
                        with self.state_store.uow() as uow:
                            uow.jobs.update_job_status(job_id=job_id, status="running")
                    except Exception as j_err:
                        logger.debug("Could not update job status: %s", j_err)
            else:
                prospect_id = self.state_store.save_prospect(prospect)
                version_id = self.state_store.save_product_context(product)
                self.state_store.save_outreach_context(
                    outreach=outreach,
                    prospect_id=prospect_id,
                    version_id=version_id,
                    context_id=context_id,
                )

                # Create research session and job records in database
                with self.state_store.uow() as uow:
                    uow.research.create_session(
                        session_id=session_id,
                        prospect_id=prospect_id,
                        context_id=context_id,
                        status="pending",
                    )
                    uow.jobs.create_job(
                        job_id=job_id,
                        job_type="outreach_pipeline",
                        status="running",
                        idempotency_key=active_run_id,
                    )

                # Create initial application_runs record
                initial_stages_json = json.dumps([rec.model_dump() for rec in stage_records])
                self.state_store.create_run(
                    run_id=active_run_id,
                    job_id=job_id,
                    prospect_id=prospect_id,
                    context_id=context_id,
                    research_session_id=session_id,
                    status=RunStatus.RUNNING.value,
                    prospect_snapshot_json=prospect.model_dump_json(),
                    product_snapshot_json=product.model_dump_json(),
                    outreach_snapshot_json=outreach.model_dump_json(),
                    stage_statuses_json=initial_stages_json,
                    started_at=start_time_iso,
                )

            record_stage(
                stage=StageIdentifier.INITIALIZATION,
                status=StageStatus.COMPLETED,
                started_at=s_start,
                details={
                    "run_id": active_run_id,
                    "job_id": job_id,
                    "prospect_id": prospect_id,
                    "session_id": session_id,
                },
            )

            # =========================================================================
            # Stage 3: Research Execution (NO DB TRANSACTION HELD OPEN)
            # =========================================================================
            current_stage = StageIdentifier.RESEARCH
            s_start = datetime.now(timezone.utc).isoformat()
            logger.info("Executing research for run %s (session %s)", active_run_id, session_id)

            global_budget = GlobalResearchBudget(total_budget_units=total_research_budget)

            research_package: ResearchPackage = self.researcher.conduct_research(
                prospect=prospect,
                outreach=outreach,
                product=product,
                session_id=session_id,
                context_id=context_id,
                max_iterations=max_research_iterations,
                budget=global_budget,
                project_id=project_id or context_id,
                run_id=active_run_id,
            )

            if research_package.budget_state is not None and isinstance(research_package.budget_state, GlobalResearchBudget):
                global_budget = research_package.budget_state

            record_stage(
                stage=StageIdentifier.RESEARCH,
                status=StageStatus.COMPLETED,
                started_at=s_start,
                details={
                    "facts_count": len(research_package.facts),
                    "sources_count": len(research_package.sources),
                    "is_sufficient": research_package.is_sufficient,
                    "is_complete": research_package.is_complete,
                    "budget_initial": global_budget.total_budget_units,
                    "budget_consumed": global_budget.total_units_consumed,
                    "budget_remaining": global_budget.remaining_total_units,
                    "baseline_consumed": global_budget.baseline_units_consumed,
                    "feedback_consumed": global_budget.feedback_units_consumed,
                },
            )

            # =========================================================================
            # Stage 4: Epistemic Sufficiency Gating
            # =========================================================================
            current_stage = StageIdentifier.SUFFICIENCY_GATE
            s_start = datetime.now(timezone.utc).isoformat()

            # Rule A: Zero facts is an automatic failure
            if len(research_package.facts) == 0:
                reason = "Research yielded zero corroborated facts for prospect or company."
                warnings.append(reason)
                record_stage(
                    stage=StageIdentifier.SUFFICIENCY_GATE,
                    status=StageStatus.FAILED,
                    started_at=s_start,
                    details={"gate_decision": "halt", "reason": reason},
                )
                record_stage(
                    stage=StageIdentifier.EMAIL_GENERATION,
                    status=StageStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    details={"reason": f"Skipped due to insufficient research: {reason}"},
                )
                return self._finalize_run(
                    run_id=active_run_id,
                    job_id=job_id,
                    status=RunStatus.INSUFFICIENT_RESEARCH,
                    prospect=prospect,
                    product=product,
                    outreach=outreach,
                    research_session_id=session_id,
                    research_package=research_package,
                    strategy_result=None,
                    stage_records=stage_records,
                    warnings=warnings,
                    error_message=reason,
                    started_at=start_time_iso,
                )

            # Rule B: Unresolved or ambiguous identity is fatal unless allow_insufficient_research is requested
            had_sufficiency_warning = False
            prospect_unresolved = bool(
                research_package.prospect_entity and research_package.prospect_entity.resolution_status in (
                    ResolutionStatus.UNRESOLVED,
                    ResolutionStatus.AMBIGUOUS,
                )
            )
            company_unresolved = bool(
                research_package.company_entity and research_package.company_entity.resolution_status in (
                    ResolutionStatus.UNRESOLVED,
                    ResolutionStatus.AMBIGUOUS,
                )
            )

            if prospect_unresolved or company_unresolved:
                unresolved_reasons = []
                if company_unresolved:
                    unresolved_reasons.append(
                        f"Company identity resolution failed with status: {research_package.company_entity.resolution_status.value}."
                    )
                if prospect_unresolved:
                    unresolved_reasons.append(
                        f"Prospect identity resolution failed with status: {research_package.prospect_entity.resolution_status.value}."
                    )
                reason = " ".join(unresolved_reasons)
                warnings.append(reason)
                if not allow_insufficient_research:
                    record_stage(
                        stage=StageIdentifier.SUFFICIENCY_GATE,
                        status=StageStatus.FAILED,
                        started_at=s_start,
                        details={"gate_decision": "halt", "reason": reason},
                    )
                    record_stage(
                        stage=StageIdentifier.EMAIL_GENERATION,
                        status=StageStatus.SKIPPED,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        details={"reason": f"Skipped due to unresolved identity: {reason}"},
                    )
                    return self._finalize_run(
                        run_id=active_run_id,
                        job_id=job_id,
                        status=RunStatus.INSUFFICIENT_RESEARCH,
                        prospect=prospect,
                        product=product,
                        outreach=outreach,
                        research_session_id=session_id,
                        research_package=research_package,
                        strategy_result=None,
                        stage_records=stage_records,
                        warnings=warnings,
                        error_message=reason,
                        started_at=start_time_iso,
                    )
                else:
                    had_sufficiency_warning = True
                    warnings.append(f"[Warning: Unresolved Identity Overridden] {reason}")

            # Rule C: Epistemic insufficiency evaluation from Phase F
            if not research_package.is_sufficient:
                missing_str = ", ".join(research_package.missing_information) if research_package.missing_information else "Unmet objectives"
                reason = f"Research declared insufficient by sufficiency evaluator: {missing_str}"
                
                if not allow_insufficient_research:
                    warnings.append(reason)
                    record_stage(
                        stage=StageIdentifier.SUFFICIENCY_GATE,
                        status=StageStatus.FAILED,
                        started_at=s_start,
                        details={"gate_decision": "halt", "reason": reason, "missing": research_package.missing_information},
                    )
                    record_stage(
                        stage=StageIdentifier.EMAIL_GENERATION,
                        status=StageStatus.SKIPPED,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        details={"reason": f"Skipped due to insufficient research: {reason}"},
                    )
                    return self._finalize_run(
                        run_id=active_run_id,
                        job_id=job_id,
                        status=RunStatus.INSUFFICIENT_RESEARCH,
                        prospect=prospect,
                        product=product,
                        outreach=outreach,
                        research_session_id=session_id,
                        research_package=research_package,
                        strategy_result=None,
                        stage_records=stage_records,
                        warnings=warnings,
                        error_message=reason,
                        started_at=start_time_iso,
                    )
                else:
                    had_sufficiency_warning = True
                    warnings.append(f"[Warning: Insufficient Research Overridden] {reason}")

            record_stage(
                stage=StageIdentifier.SUFFICIENCY_GATE,
                status=StageStatus.COMPLETED,
                started_at=s_start,
                details={
                    "gate_decision": "proceed",
                    "is_sufficient": research_package.is_sufficient,
                    "warnings": warnings,
                },
            )

            # =========================================================================
            # Stage 5: Signal Strategy Synthesis (NO DB TRANSACTION HELD OPEN)
            # =========================================================================
            current_stage = StageIdentifier.SIGNAL_STRATEGY
            s_start = datetime.now(timezone.utc).isoformat()
            logger.info("Developing signal strategy for run %s", active_run_id)

            strategy_result: StrategyResult = self.strategy_service.develop_strategy(
                research_package=research_package,
                prospect=prospect,
                product=product,
                outreach=outreach,
            )

            record_stage(
                stage=StageIdentifier.SIGNAL_STRATEGY,
                status=StageStatus.COMPLETED if strategy_result.is_viable else StageStatus.FAILED,
                started_at=s_start,
                details={
                    "is_viable": strategy_result.is_viable,
                    "rejection_reason": strategy_result.rejection_reason,
                    "messaging_approach": strategy_result.strategy.messaging_approach if strategy_result.strategy else None,
                },
            )

            # =========================================================================
            # Stage 5b: Bounded EvidenceRequest Feedback Protocol (M3.1)
            # =========================================================================
            feedback_round_result: Optional[FeedbackRoundResult] = None
            if (not strategy_result.is_viable or (
                strategy_result.strategy and getattr(strategy_result.strategy, "is_conditional", False)
            )):
                s_feedback_start = datetime.now(timezone.utc).isoformat()
                if global_budget.is_exhausted or global_budget.is_feedback_exhausted:
                    logger.info("Skipping feedback round: global budget or feedback limit exhausted")
                    record_stage(
                        stage=StageIdentifier.RESEARCH_FEEDBACK,
                        status=StageStatus.SKIPPED,
                        started_at=s_feedback_start,
                        details={
                            "reason": "Global budget or feedback sub-limit exhausted",
                            "budget_consumed": global_budget.total_units_consumed,
                            "remaining": global_budget.remaining_total_units,
                        },
                    )
                else:
                    gap_detector = EvidenceGapDetector(max_requests=3)
                    evidence_requests = gap_detector.detect_gaps_and_create_requests(
                        strategy_result=strategy_result,
                        research_package=research_package,
                        prospect=prospect,
                        product=product,
                        run_id=active_run_id,
                    )

                    if not evidence_requests:
                        logger.info("No actionable evidence gaps detected; skipping feedback research")
                        record_stage(
                            stage=StageIdentifier.RESEARCH_FEEDBACK,
                            status=StageStatus.SKIPPED,
                            started_at=s_feedback_start,
                            details={
                                "reason": "No evidence gaps detected",
                                "global_budget_consumed": global_budget.total_units_consumed,
                                "global_budget_remaining": global_budget.remaining_total_units,
                            },
                        )
                    else:
                        logger.info(
                            "Executing bounded evidence feedback round for run %s with %d requests",
                            active_run_id,
                            len(evidence_requests),
                        )
                        feedback_budget = EvidenceFeedbackBudget(
                            baseline_budget_units=global_budget.total_budget_units,
                            max_feedback_budget_ratio=global_budget.max_feedback_budget_ratio,
                            feedback_budget_cap=global_budget.feedback_budget_cap,
                            feedback_units_consumed=global_budget.feedback_units_consumed,
                            rounds_executed=global_budget.feedback_rounds_executed,
                            requests_executed_count=global_budget.feedback_requests_executed,
                        )
                        executor = EvidenceRequestExecutor(researcher=self.researcher, state_store=self.state_store)

                        acc = (
                            self.researcher._hydrate_accumulator(session_id)
                            if hasattr(self.researcher, "_hydrate_accumulator")
                            else ResearchAccumulator(facts=list(research_package.facts), sources=list(research_package.sources))
                        )

                        feedback_round_result = executor.execute_requests(
                            requests=evidence_requests,
                            session_id=session_id,
                            run_id=active_run_id,
                            budget=feedback_budget,
                            accumulator=acc,
                            round_number=1,
                        )

                        delta_feedback = feedback_round_result.budget.feedback_units_consumed - global_budget.feedback_units_consumed
                        if delta_feedback > 0:
                            global_budget = global_budget.consume_feedback(
                                cost_units=delta_feedback,
                                increment_request=False,
                            )
                        global_budget = global_budget.complete_feedback_round()

                        # Update research package facts from rehydrated accumulator
                        if hasattr(self.researcher, "_hydrate_accumulator"):
                            updated_acc = self.researcher._hydrate_accumulator(session_id)
                        else:
                            updated_acc = acc

                        if research_package.question_dag_json:
                            try:
                                summary = json.loads(research_package.question_dag_json)
                                summary.setdefault("feedback_requests", []).extend([
                                    {"request_id": r.request_id, "predicate": r.requested_predicate, "status": r.status.value}
                                    for r in feedback_round_result.requests
                                ])
                                summary["global_budget_consumed"] = global_budget.total_units_consumed
                                summary["global_budget_remaining"] = global_budget.remaining_total_units
                                research_package = research_package.model_copy(
                                    update={
                                        "facts": list(updated_acc.facts),
                                        "sources": list(updated_acc.sources),
                                        "question_dag_json": json.dumps(summary),
                                    }
                                )
                            except Exception:
                                research_package = research_package.model_copy(
                                    update={
                                        "facts": list(updated_acc.facts),
                                        "sources": list(updated_acc.sources),
                                    }
                                )
                        else:
                            research_package = research_package.model_copy(
                                update={
                                    "facts": list(updated_acc.facts),
                                    "sources": list(updated_acc.sources),
                                }
                            )

                        # Re-evaluate strategy with updated research package
                        re_evaluated_strategy = self.strategy_service.develop_strategy(
                            research_package=research_package,
                            prospect=prospect,
                            product=product,
                            outreach=outreach,
                        )

                        if re_evaluated_strategy.is_viable and not (
                            re_evaluated_strategy.strategy and getattr(re_evaluated_strategy.strategy, "is_conditional", False)
                        ):
                            logger.info("Feedback research successfully resolved evidence gaps; strategy is now VIABLE.")
                            strategy_result = re_evaluated_strategy
                            feedback_round_result = feedback_round_result.model_copy(update={"strategy_revised": True})
                        elif re_evaluated_strategy.is_viable:
                            strategy_result = re_evaluated_strategy
                            feedback_round_result = feedback_round_result.model_copy(update={"strategy_revised": True})
                        else:
                            logger.info("Strategy remains unviable after feedback research.")
                            feedback_round_result = feedback_round_result.model_copy(update={"strategy_invalidated": True})

                        record_stage(
                            stage=StageIdentifier.RESEARCH_FEEDBACK,
                            status=StageStatus.COMPLETED,
                            started_at=s_feedback_start,
                            details={
                                "requests_count": len(evidence_requests),
                                "budget_consumed": feedback_round_result.budget.feedback_units_consumed,
                                "budget_cap": feedback_round_result.budget.feedback_budget_cap,
                                "global_budget_consumed": global_budget.total_units_consumed,
                                "global_budget_remaining": global_budget.remaining_total_units,
                                "strategy_became_viable": re_evaluated_strategy.is_viable,
                                "status": feedback_round_result.status,
                            },
                        )

            # =========================================================================
            # Stage 6: Strategy Viability Check & Email Generation
            # =========================================================================
            draft: Optional[VerifiedOutreachDraft] = None
            email_stage_err: Optional[str] = None
            email_stage_status = StageStatus.COMPLETED

            if not strategy_result.is_viable:
                rejection = strategy_result.rejection_reason or "No viable commercial signal identified."
                warnings.extend(strategy_result.unviable_reasons)
                final_status = RunStatus.NO_VIABLE_SIGNAL
                record_stage(
                    stage=StageIdentifier.EMAIL_GENERATION,
                    status=StageStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    details={"reason": f"Skipped: {rejection}"},
                )
            elif not strategy_result.strategy:
                rejection = "StrategyResult contains no strategy."
                final_status = RunStatus.NO_VIABLE_SIGNAL
                record_stage(
                    stage=StageIdentifier.EMAIL_GENERATION,
                    status=StageStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    details={"reason": f"Skipped: {rejection}"},
                )
            elif not getattr(strategy_result.strategy, "toulmin_argument", None):
                # Missing Toulmin argument: SKIPPED with explicit reason per compatibility policy
                skip_msg = "OutreachStrategy lacks validated ToulminArgument per M1.3 contract."
                record_stage(
                    stage=StageIdentifier.EMAIL_GENERATION,
                    status=StageStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    details={"reason": skip_msg},
                )
                final_status = RunStatus.COMPLETED_WITH_WARNINGS if had_sufficiency_warning else RunStatus.COMPLETED
            else:
                # =========================================================================
                # Stage 7: Email Generation & Readability Linting
                # =========================================================================
                current_stage = StageIdentifier.EMAIL_GENERATION
                s_start = datetime.now(timezone.utc).isoformat()
                email_details: Dict[str, Any] = {}

                try:
                    draft = self.email_service.generate_draft(
                        prospect=prospect,
                        product=product,
                        strategy_result=strategy_result,
                        outreach=outreach,
                        research_package=research_package,
                        context_id=context_id,
                        research_session_id=session_id,
                    )
                    email_details = {
                        "draft_id": draft.draft_id,
                        "word_count": draft.draft.word_count,
                        "linter_passed": draft.linter_result.passed if draft.linter_result else None,
                        "verifier_passed": getattr(draft.verifier_result, "passed", None) if getattr(draft, "verifier_result", None) else None,
                        "has_verifier_errors": getattr(draft.verifier_result, "has_errors", False) if getattr(draft, "verifier_result", None) else False,
                        "verifier_errors_count": len([f for f in draft.verifier_result.findings if f.severity.value == "error"]) if getattr(draft, "verifier_result", None) else 0,
                        "unverified_claim_ids": getattr(draft.verifier_result, "unverified_claim_ids", []) if getattr(draft, "verifier_result", None) else [],
                        "claim_verifications_count": len(draft.claim_verifications),
                        "human_review_status": draft.human_review_status.value,
                    }
                    has_verifier_warnings = bool(getattr(draft, "verifier_result", None) and getattr(draft.verifier_result, "has_warnings", False))
                    has_verifier_errors = bool(getattr(draft, "verifier_result", None) and getattr(draft.verifier_result, "has_errors", False))
                    has_unverified_claims = bool(getattr(draft, "verifier_result", None) and getattr(draft.verifier_result, "unverified_claim_ids", []))
                    verifier_passed = bool(getattr(draft.verifier_result, "passed", False)) if getattr(draft, "verifier_result", None) else True

                    if has_verifier_errors or not verifier_passed:
                        err_count = email_details['verifier_errors_count']
                        email_stage_status = StageStatus.FAILED
                        email_stage_err = f"Draft verification failed with {err_count} error finding(s). Draft quarantined."
                        final_status = RunStatus.FAILED
                        warnings.append(email_stage_err)
                    elif had_sufficiency_warning or (draft.linter_result and draft.linter_result.has_warnings) or has_verifier_warnings or has_unverified_claims:
                        if has_unverified_claims:
                            warnings.append(f"Draft contains {len(email_details['unverified_claim_ids'])} unverified claim(s). Requires human review.")
                        if has_verifier_warnings:
                            warnings.append("Draft verification produced warnings requiring human review.")
                        final_status = RunStatus.COMPLETED_WITH_WARNINGS
                    else:
                        final_status = RunStatus.COMPLETED
                except DraftPersistenceError as p_err:
                    logger.error("Draft persistence failed: %s", p_err)
                    email_stage_status = StageStatus.FAILED
                    email_stage_err = str(p_err)
                    final_status = RunStatus.FAILED
                    warnings.append(f"Draft persistence failed: {p_err}")
                except EmailGenerationError as g_err:
                    logger.error("Email generation failed: %s", g_err)
                    email_stage_status = StageStatus.FAILED
                    email_stage_err = str(g_err)
                    final_status = RunStatus.FAILED
                    warnings.append(f"Email generator failed: {g_err}")
                except Exception as e_err:
                    logger.error("Email generation unexpected failure: %s", e_err)
                    email_stage_status = StageStatus.FAILED
                    email_stage_err = str(e_err)
                    final_status = RunStatus.FAILED
                    warnings.append(f"Email generation failed: {e_err}")

                record_stage(
                    stage=StageIdentifier.EMAIL_GENERATION,
                    status=email_stage_status,
                    started_at=s_start,
                    details=email_details,
                    error_message=email_stage_err,
                )

            return self._finalize_run(
                run_id=active_run_id,
                job_id=job_id,
                status=final_status,
                prospect=prospect,
                product=product,
                outreach=outreach,
                research_session_id=session_id,
                research_package=research_package,
                strategy_result=strategy_result,
                draft=draft,
                feedback_round=feedback_round_result,
                stage_records=stage_records,
                warnings=warnings,
                error_message=email_stage_err if (email_stage_status == StageStatus.FAILED) else (strategy_result.rejection_reason if not strategy_result.is_viable else None),
                started_at=start_time_iso,
            )

        except Exception as exc:
            logger.error("Unhandled error in application run %s at stage %s: %s", active_run_id, current_stage, exc, exc_info=True)
            record_stage(
                stage=current_stage,
                status=StageStatus.FAILED,
                started_at=datetime.now(timezone.utc).isoformat(),
                error_message=str(exc),
            )
            return self._finalize_run(
                run_id=active_run_id,
                job_id=job_id,
                status=RunStatus.FAILED,
                prospect=prospect,
                product=product,
                outreach=outreach,
                research_session_id=session_id,
                research_package=None,
                strategy_result=None,
                draft=None,
                feedback_round=None,
                stage_records=stage_records,
                warnings=warnings,
                error_message=str(exc),
                started_at=start_time_iso,
            )

    def _finalize_run(
        self,
        run_id: str,
        job_id: str,
        status: RunStatus,
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        research_session_id: Optional[str],
        research_package: Optional[ResearchPackage],
        strategy_result: Optional[StrategyResult],
        stage_records: List[StageExecutionRecord],
        warnings: List[str],
        error_message: Optional[str],
        started_at: str,
        draft: Optional[VerifiedOutreachDraft] = None,
        feedback_round: Optional[FeedbackRoundResult] = None,
    ) -> RunResult:
        """Atomically persist run completion state, update job record, and return RunResult."""
        completed_at_iso = datetime.now(timezone.utc).isoformat()
        stages_json = json.dumps([rec.model_dump() for rec in stage_records])
        warnings_json = json.dumps(warnings) if warnings else None
        strat_json = strategy_result.model_dump_json() if strategy_result else None
        draft_id = draft.draft_id if draft else None
        feedback_json = feedback_round.model_dump_json() if feedback_round else None

        # 1. Update application_runs row
        self.state_store.update_run(
            run_id=run_id,
            status=status.value,
            research_session_id=research_session_id,
            strategy_result_json=strat_json,
            stage_statuses_json=stages_json,
            warnings_json=warnings_json,
            feedback_round_json=feedback_json,
            error_message=error_message,
            completed_at=completed_at_iso,
            draft_id=draft_id,
        )

        # 2. Update jobs table
        job_outcome = "completed" if status in (RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS) else "failed"
        try:
            with self.state_store.uow() as uow:
                uow.jobs.update_job_status(
                    job_id=job_id,
                    status=job_outcome,
                    completed_at=completed_at_iso,
                    error_message=error_message,
                )
        except Exception as j_err:
            logger.debug("Could not update job status: %s", j_err)

        return RunResult(
            run_id=run_id,
            status=status,
            prospect=prospect,
            product=product,
            outreach=outreach,
            research_session_id=research_session_id,
            research_package=research_package,
            strategy_result=strategy_result,
            feedback_round=feedback_round,
            draft=draft,
            draft_id=draft_id,
            stage_records=stage_records,
            warnings=warnings,
            error_message=error_message,
            started_at=started_at,
            completed_at=completed_at_iso,
        )

    def get_run(self, run_id: str) -> Optional[ApplicationRunSnapshot]:
        """Retrieve and reconstruct a previous application run from SQLite."""
        row = self.state_store.get_run(run_id)
        if not row:
            return None

        prospect = ProspectContext.model_validate_json(row["prospect_snapshot_json"])
        product = ProductContext.model_validate_json(row["product_snapshot_json"])
        outreach = OutreachContext.model_validate_json(row["outreach_snapshot_json"])

        strategy_result = None
        if row.get("strategy_result_json"):
            strategy_result = StrategyResult.model_validate_json(row["strategy_result_json"])

        feedback_round = None
        if row.get("feedback_round_json"):
            try:
                feedback_round = FeedbackRoundResult.model_validate_json(row["feedback_round_json"])
            except Exception:
                feedback_round = None

        stage_statuses = json.loads(row["stage_statuses_json"]) if row.get("stage_statuses_json") else []
        warnings = json.loads(row["warnings_json"]) if row.get("warnings_json") else []

        return ApplicationRunSnapshot(
            run_id=row["run_id"],
            job_id=row.get("job_id"),
            prospect_id=row["prospect_id"],
            context_id=row.get("context_id"),
            research_session_id=row.get("research_session_id"),
            draft_id=row.get("draft_id"),
            status=RunStatus(row["status"]),
            prospect=prospect,
            product=product,
            outreach=outreach,
            strategy_result=strategy_result,
            feedback_round=feedback_round,
            stage_statuses=stage_statuses,
            warnings=warnings,
            error_message=row.get("error_message"),
            started_at=row["started_at"],
            completed_at=row.get("completed_at"),
        )

    def list_runs(self, limit: int = 50, status: Optional[RunStatus] = None) -> List[Dict[str, Any]]:
        """List historical application runs."""
        status_str = status.value if status is not None else None
        return self.state_store.list_runs(limit=limit, status=status_str)

    def get_run_stages(self, run_id: str) -> List[StageExecutionRecord]:
        """Retrieve stage execution timeline for a run."""
        row = self.state_store.get_run(run_id)
        if not row or not row.get("stage_statuses_json"):
            return []
        raw_stages = json.loads(row["stage_statuses_json"])
        return [StageExecutionRecord.model_validate(s) for s in raw_stages]

    def get_strategy_explainability(self, run_id: str) -> Dict[str, Any]:
        """Retrieve complete domain explainability tree connecting strategy back to evidence and sources."""
        snapshot = self.get_run(run_id)
        if not snapshot or not snapshot.strategy_result or not snapshot.strategy_result.strategy:
            return {"error": f"No viable strategy available for run {run_id}"}

        strategy = snapshot.strategy_result.strategy
        primary_ranked = strategy.primary_signal
        candidate_signal = primary_ranked.signal
        session_id = snapshot.research_session_id

        # Query supporting facts and evidence from database
        supporting_facts: List[Dict[str, Any]] = []
        evidence_chain: List[Dict[str, Any]] = []

        if session_id and candidate_signal.supporting_fact_ids:
            with self.state_store.uow() as uow:
                for fid in candidate_signal.supporting_fact_ids:
                    fact_row = uow.assertions.get_canonical_fact(fid)
                    if fact_row:
                        supporting_facts.append(fact_row)
                        ev_passages = uow.assertions.get_evidence_for_fact(fid)
                        for ev_row in ev_passages:
                            src_row = uow.sources.get_source(ev_row["source_id"])
                            evidence_chain.append({
                                "fact_id": fid,
                                "claim_text": fact_row["claim_text"],
                                "evidence_id": ev_row["evidence_id"],
                                "verbatim_excerpt": ev_row["verbatim_excerpt"],
                                "excerpt_validated": bool(ev_row["excerpt_validated"]),
                                "event_date": ev_row["event_date"],
                                "credibility_score": ev_row["credibility_score"],
                                "source_url": src_row["canonical_url"] if src_row else None,
                                "domain": src_row["domain"] if src_row else None,
                            })

        return {
            "run_id": run_id,
            "session_id": session_id,
            "prospect_name": snapshot.prospect.full_name,
            "company_name": snapshot.prospect.company_name,
            "messaging_approach": strategy.messaging_approach,
            "central_message": strategy.central_message,
            "suggested_subject_line": strategy.suggested_subject_line,
            "cta": strategy.cta,
            "tone_guidance": strategy.tone_guidance,
            "primary_signal": primary_ranked.model_dump(),
            "supporting_signals": [s.model_dump() for s in strategy.supporting_signals],
            "alignment": strategy.alignment.model_dump(),
            "supporting_facts": supporting_facts,
            "evidence_chain": evidence_chain,
            "rejection_reasons": snapshot.strategy_result.unviable_reasons,
        }

    def create_pending_run(
        self,
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        run_id: Optional[str] = None,
    ) -> ApplicationRunSnapshot:
        """Validate input context and persist an initial run in PENDING status before asynchronous execution."""
        db_status = self.state_store.get_database_status()
        if not db_status["file_exists"] or db_status["table_count"] == 0:
            self.state_store.initialize()
        elif not db_status["is_initialized"] or not db_status["is_schema_valid"]:
            raise RuntimeError(
                f"Database at {db_status['db_path']} has an incompatible schema: {db_status['incompatibility_details']}."
            )

        if not prospect.full_name or not prospect.company_name:
            raise ValueError("Prospect full_name and company_name are required.")
        if not product.product_name or not product.value_proposition:
            raise ValueError("Product product_name and value_proposition are required.")

        active_run_id = run_id or str(uuid.uuid4())
        job_id = f"job_{uuid.uuid4().hex}"
        session_id = f"session_{uuid.uuid4().hex}"
        context_id = str(uuid.uuid4())
        start_time_iso = datetime.now(timezone.utc).isoformat()

        ctx_run_id.set(active_run_id)
        active_job_id.set(job_id)
        active_stage_name.set(StageIdentifier.VALIDATION.value)

        prospect_id = self.state_store.save_prospect(prospect)
        version_id = self.state_store.save_product_context(product)
        self.state_store.save_outreach_context(
            outreach=outreach,
            prospect_id=prospect_id,
            version_id=version_id,
            context_id=context_id,
        )

        with self.state_store.uow() as uow:
            uow.research.create_session(
                session_id=session_id,
                prospect_id=prospect_id,
                context_id=context_id,
                status="pending",
            )
            uow.jobs.create_job(
                job_id=job_id,
                job_type="outreach_pipeline",
                status="pending",
                idempotency_key=active_run_id,
            )

        initial_stages = [
            StageExecutionRecord(
                stage=StageIdentifier.VALIDATION,
                status=StageStatus.COMPLETED,
                started_at=start_time_iso,
                completed_at=start_time_iso,
                duration_ms=0.0,
                details={
                    "prospect_name": prospect.full_name,
                    "company_name": prospect.company_name,
                    "product_name": product.product_name,
                },
            )
        ]

        self.state_store.create_run(
            run_id=active_run_id,
            job_id=job_id,
            prospect_id=prospect_id,
            context_id=context_id,
            research_session_id=session_id,
            status=RunStatus.PENDING.value,
            prospect_snapshot_json=prospect.model_dump_json(),
            product_snapshot_json=product.model_dump_json(),
            outreach_snapshot_json=outreach.model_dump_json(),
            stage_statuses_json=json.dumps([rec.model_dump() for rec in initial_stages]),
            started_at=start_time_iso,
        )

        snapshot = self.get_run(active_run_id)
        if not snapshot:
            raise RuntimeError(f"Failed to retrieve newly created pending run {active_run_id}")
        return snapshot

    def get_research_inspection(self, session_or_run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete structured research inspection for a session or run ID."""
        target_session_id = session_or_run_id
        target_run_id = None

        row = self.state_store.get_run(session_or_run_id)
        if not row:
            matched_runs = [r for r in self.state_store.list_runs(limit=100) if r["run_id"].startswith(session_or_run_id)]
            if matched_runs:
                row = matched_runs[0]
        if row:
            target_run_id = row["run_id"]
            if row.get("research_session_id"):
                target_session_id = row["research_session_id"]

        with self.state_store.uow() as uow:
            session = uow.research.get_session(target_session_id)
            if not session:
                cursor = uow.research.conn.execute("SELECT session_id FROM research_sessions;")
                matched_sess = [r[0] for r in cursor.fetchall() if r[0].startswith(session_or_run_id)]
                if matched_sess:
                    target_session_id = matched_sess[0]
                    session = uow.research.get_session(target_session_id)

            if not session:
                return None

            objectives = uow.research.get_objectives(target_session_id)
            raw_facts = uow.assertions.get_facts_for_session(target_session_id)
            conflicts = uow.assertions.get_conflicts_for_session(target_session_id)
            assessments = uow.entities.get_assessments(session_id=target_session_id)
            iterations = uow.orchestration.list_iterations(target_session_id)
            latest_assessment = uow.orchestration.get_latest_assessment(target_session_id)

            facts: List[Dict[str, Any]] = []
            for f in raw_facts:
                fid = f["fact_id"]
                ev_passages = uow.assertions.get_evidence_for_fact(fid)
                evidence_list: List[Dict[str, Any]] = []
                for ev in ev_passages:
                    src = uow.sources.get_source(ev["source_id"]) if ev.get("source_id") else None
                    evidence_list.append({
                        "evidence_id": ev["evidence_id"],
                        "verbatim_excerpt": ev["verbatim_excerpt"],
                        "excerpt_validated": bool(ev.get("excerpt_validated")),
                        "credibility_score": float(ev["credibility_score"]) if ev.get("credibility_score") is not None else 0.5,
                        "event_date": ev.get("event_date"),
                        "source": {
                            "source_id": src["source_id"],
                            "canonical_url": src["canonical_url"],
                            "domain": src.get("domain"),
                            "source_type": src.get("source_type"),
                        } if src else None,
                    })

                facts.append({
                    "fact_id": fid,
                    "fact_category": f.get("fact_category", "other"),
                    "claim_text": f["claim_text"],
                    "composite_confidence": f.get("composite_confidence", "medium"),
                    "primary_event_date": f.get("primary_event_date"),
                    "created_at": f.get("created_at"),
                    "evidence": evidence_list,
                })

            identities: List[Dict[str, Any]] = []
            seen_entity_ids = set()
            for ass in assessments:
                eid = ass["entity_id"]
                ent = uow.entities.get_entity(eid)
                if ent and eid not in seen_entity_ids:
                    seen_entity_ids.add(eid)
                    identities.append({
                        "entity_id": eid,
                        "entity_type": ent["entity_type"],
                        "raw_name": ent["raw_name"],
                        "canonical_name": ent.get("canonical_name"),
                        "resolution_status": ass["resolution_status"],
                        "confidence": ass["confidence"],
                        "rationale": ass["rationale"],
                        "assessed_by": ass.get("assessed_by", "entity_resolver"),
                        "assessed_at": ass.get("assessed_at"),
                    })

            return {
                "session_id": target_session_id,
                "run_id": target_run_id,
                "status": session["status"],
                "is_sufficient": bool(session.get("is_sufficient")),
                "is_complete": bool(session.get("is_complete")),
                "sufficiency_notes": session.get("sufficiency_notes"),
                "started_at": session.get("started_at"),
                "completed_at": session.get("completed_at"),
                "objectives": objectives,
                "facts": facts,
                "conflicts": conflicts,
                "identities": identities,
                "iterations": iterations,
                "latest_assessment": latest_assessment,
            }

    def reconcile_interrupted_runs(self) -> int:
        """Find runs left in RUNNING status (e.g., from process termination) and mark them FAILED."""
        active_runs = self.state_store.list_runs(limit=1000, status=RunStatus.RUNNING.value)
        reconciled = 0
        now = datetime.now(timezone.utc).isoformat()
        for r in active_runs:
            run_id = r["run_id"]
            stages = self.get_run_stages(run_id)
            stages.append(
                StageExecutionRecord(
                    stage=StageIdentifier.FINALIZATION,
                    status=StageStatus.FAILED,
                    started_at=now,
                    completed_at=now,
                    duration_ms=0.0,
                    error_message="Execution interrupted by server shutdown or restart",
                )
            )
            stages_json = json.dumps([s.model_dump() for s in stages])
            self.state_store.update_run(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                stage_statuses_json=stages_json,
                error_message="Execution interrupted by server shutdown or restart",
                completed_at=now,
            )
            reconciled += 1
            logger.info("Reconciled interrupted run %s to FAILED status", run_id)
        return reconciled

