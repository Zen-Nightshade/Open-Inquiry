from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.db.unit_of_work import UnitOfWork
from src.researcher.entity.service import EntityResolutionService
from src.researcher.execution import ResearchExecutionCoordinator
from src.researcher.extraction.fetcher import PageFetcher
from src.researcher.extraction.models import FetchOutcome, FetchStatus
from src.researcher.extraction.service import FactExtractionService
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    ResearchPackage,
    ResolutionStatus,
    SourceRecord,
    SourceType,
)
from src.researcher.orchestration.evaluator import ResearchSufficiencyEvaluator
from src.researcher.orchestration.models import (
    ResearchSufficiencyAssessment,
    SufficiencyDecision,
)
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    TargetEntity,
)
from src.researcher.planning.planner import ResearchPlanner
from src.researcher.search.followup_generator import FollowUpQueryGenerator
from src.researcher.search.models import SearchResponse, SearchStatus
from src.researcher.search.query_generator import GeneratedQuery, QueryGenerator
from src.researcher.search.search_client import SearchClient

logger = logging.getLogger(__name__)


class Researcher:
    """Primary researcher orchestrator coordinating multi-turn iterative investigation."""

    def __init__(
        self,
        state_store: StateStore,
        planner: Optional[ResearchPlanner] = None,
        query_generator: Optional[QueryGenerator] = None,
        followup_generator: Optional[FollowUpQueryGenerator] = None,
        search_client: Optional[SearchClient] = None,
        page_fetcher: Optional[PageFetcher] = None,
        entity_service: Optional[EntityResolutionService] = None,
        fact_service: Optional[FactExtractionService] = None,
        evaluator: Optional[ResearchSufficiencyEvaluator] = None,
    ) -> None:
        self.state_store = state_store
        self.planner = planner or ResearchPlanner()
        self.query_generator = query_generator or QueryGenerator()
        self.followup_generator = followup_generator or FollowUpQueryGenerator()
        self.search_client = search_client or SearchClient()
        self.page_fetcher = page_fetcher or PageFetcher()
        self.entity_service = entity_service or EntityResolutionService(state_store=self.state_store)
        self.fact_service = fact_service or FactExtractionService(state_store=self.state_store)
        self.evaluator = evaluator or ResearchSufficiencyEvaluator()

    def conduct_research(
        self,
        prospect: ProspectContext,
        outreach: OutreachContext,
        product: Optional[ProductContext] = None,
        session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        max_iterations: int = 3,
        max_results_per_query: int = 5,
        max_pages_to_fetch_per_iter: int = 6,
    ) -> ResearchPackage:
        """Execute full multi-turn research lifecycle with sufficiency evaluation and checkpointed recovery."""
        active_session_id = session_id or str(uuid.uuid4())
        active_context_id = context_id or str(uuid.uuid4())

        # Ensure prospect is persisted
        prospect_record = self._ensure_prospect_persisted(prospect)
        prospect_id = prospect_record["prospect_id"]

        # Ensure session row exists
        with self.state_store.uow() as uow:
            existing_session = uow.research.get_session(active_session_id)
            if not existing_session:
                uow.research.create_session(
                    session_id=active_session_id,
                    prospect_id=prospect_id,
                    context_id=active_context_id,
                    status="in_progress",
                )

        # Stage 1: Planning Checkpoint
        objectives = self._get_or_create_objectives(
            session_id=active_session_id,
            prospect=prospect,
            outreach=outreach,
            product=product,
        )

        # Initialize accumulator with any previously saved state
        accumulator = self._hydrate_accumulator(active_session_id)

        # Determine starting iteration (checkpoint recovery)
        current_iter_num, latest_assessment = self._determine_resumption_point(
            session_id=active_session_id,
            max_iterations=max_iterations,
        )

        is_sufficient = False
        is_complete = False
        final_assessment: Optional[ResearchSufficiencyAssessment] = latest_assessment

        # Multi-turn Iteration Loop
        while current_iter_num <= max_iterations:
            iteration_id = f"{active_session_id}_it_{current_iter_num}"

            # Check if this iteration was already completed in a prior run
            with self.state_store.uow() as uow:
                existing_iter = uow.orchestration.get_iteration(iteration_id)
                if not existing_iter:
                    uow.orchestration.create_iteration(
                        iteration_id=iteration_id,
                        session_id=active_session_id,
                        iteration_number=current_iter_num,
                        status="in_progress",
                    )
                elif existing_iter["status"] == "completed":
                    # Check its assessment
                    cached_ass_row = uow.orchestration.get_assessment_for_iteration(iteration_id)
                    if cached_ass_row:
                        if cached_ass_row["decision"] in (SufficiencyDecision.SUFFICIENT.value, SufficiencyDecision.INSUFFICIENT.value):
                            is_sufficient = (cached_ass_row["decision"] == SufficiencyDecision.SUFFICIENT.value)
                            is_complete = True
                            break
                        else:
                            current_iter_num += 1
                            continue

            logger.info("Executing research turn %d of %d (Session: %s)", current_iter_num, max_iterations, active_session_id)

            try:
                # Step A: Query Generation
                queries: List[GeneratedQuery] = []
                if current_iter_num == 1:
                    for obj in objectives:
                        q_list = self.query_generator.generate_queries(
                            objective=obj,
                            prospect=prospect,
                            existing_queries=accumulator.search_queries_used,
                        )
                        queries.extend(q_list)
                else:
                    if latest_assessment:
                        queries = self.followup_generator.generate_followup_queries(
                            prospect=prospect,
                            assessment=latest_assessment,
                            existing_queries=accumulator.search_queries_used,
                            max_queries=4,
                        )

                # Persist queries
                valid_obj_ids = {obj.objective_id for obj in objectives}
                default_obj_id = objectives[0].objective_id if objectives else None
                with self.state_store.uow() as uow:
                    for q in queries:
                        target_obj_id = q.objective_id if q.objective_id in valid_obj_ids else default_obj_id
                        uow.research.create_query(
                            query_id=q.query_id,
                            session_id=active_session_id,
                            objective_id=target_obj_id,
                            query_text=q.query_text,
                        )
                for q in queries:
                    if q.query_text not in accumulator.search_queries_used:
                        accumulator.search_queries_used.append(q.query_text)

                # Step B: Search Execution & Page Fetching
                new_snapshot_ids = self._execute_searches_and_fetches(
                    session_id=active_session_id,
                    queries=queries,
                    accumulator=accumulator,
                    max_results=max_results_per_query,
                    max_pages=max_pages_to_fetch_per_iter,
                )

                # Step C: Entity Resolution
                known_entities = self._run_entity_resolution(
                    session_id=active_session_id,
                    prospect=prospect,
                    accumulator=accumulator,
                )

                # Step D: Fact Extraction & Consolidation for new snapshots
                new_facts_count = 0
                for snap_id in new_snapshot_ids:
                    extracted = self.fact_service.extract_and_persist_facts_for_session(
                        session_id=active_session_id,
                        snapshot_id=snap_id,
                        known_entities=known_entities,
                        accumulator=accumulator,
                    )
                    new_facts_count += len(extracted)

                # Step E: Research Sufficiency Evaluation
                with self.state_store.uow() as uow:
                    conflicts = uow.assertions.get_conflicts_for_session(active_session_id)

                assessment = self.evaluator.evaluate(
                    session_id=active_session_id,
                    iteration_id=iteration_id,
                    iteration_number=current_iter_num,
                    max_iterations=max_iterations,
                    objectives=objectives,
                    facts=accumulator.facts,
                    sources=accumulator.sources,
                    prospect_entity=accumulator.prospect_entity,
                    company_entity=accumulator.company_entity,
                    conflicts=conflicts,
                )
                final_assessment = assessment

                # Persist Sufficiency Assessment & complete iteration
                with self.state_store.uow() as uow:
                    uow.orchestration.create_assessment(
                        assessment_id=assessment.assessment_id,
                        session_id=active_session_id,
                        iteration_id=iteration_id,
                        decision=assessment.decision.value,
                        coverage_status=assessment.dimensions.objective_coverage.status,
                        identity_status=assessment.dimensions.identity_certainty.status,
                        evidence_status=assessment.dimensions.evidence_adequacy.status,
                        marginal_value_status=assessment.dimensions.marginal_research_value.status,
                        explanation=assessment.explanation,
                        objective_details_json=[o.model_dump() for o in assessment.objective_assessments],
                        remaining_gaps_json=list(assessment.remaining_gaps),
                        recommended_actions_json=[a.model_dump() for a in assessment.recommended_actions],
                        dimension_details_json=assessment.dimensions.model_dump(),
                    )
                    uow.orchestration.update_iteration(
                        iteration_id=iteration_id,
                        status="completed",
                        queries_executed=len(queries),
                        snapshots_fetched=len(new_snapshot_ids),
                        facts_extracted=new_facts_count,
                        conflicts_detected=len(conflicts),
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )

                # Check Termination
                if assessment.decision == SufficiencyDecision.SUFFICIENT:
                    is_sufficient = True
                    is_complete = True
                    break
                elif current_iter_num >= max_iterations or assessment.decision == SufficiencyDecision.INSUFFICIENT:
                    is_sufficient = False
                    is_complete = True
                    break
                else:
                    latest_assessment = assessment
                    current_iter_num += 1

            except Exception as exc:
                logger.error("Error executing iteration %s: %s", iteration_id, exc, exc_info=True)
                with self.state_store.uow() as uow:
                    uow.orchestration.update_iteration(
                        iteration_id=iteration_id,
                        status="failed",
                        error_message=str(exc),
                    )
                raise

        # Package Assembly
        remaining_gaps = list(final_assessment.remaining_gaps) if final_assessment else []
        explanation = final_assessment.explanation if final_assessment else "Research completed"

        package = accumulator.to_package(
            is_sufficient=is_sufficient,
            is_complete=is_complete,
            missing_information=remaining_gaps,
            sufficiency_notes=explanation,
        )

        # Update Session
        with self.state_store.uow() as uow:
            uow.research.update_session(
                session_id=active_session_id,
                status="completed" if is_complete else "failed",
                is_sufficient=is_sufficient,
                is_complete=is_complete,
                completed_at=datetime.now(timezone.utc).isoformat(),
                sufficiency_notes=explanation,
                missing_information=json.dumps(remaining_gaps),
            )

        return package

    def _ensure_prospect_persisted(self, prospect: ProspectContext) -> Dict[str, Any]:
        with self.state_store.uow() as uow:
            prospects = uow.outreach.list_prospects()
            for p in prospects:
                if (
                    p["full_name"].lower() == prospect.full_name.lower()
                    and p["company_name"].lower() == prospect.company_name.lower()
                ):
                    return p

            prospect_id = str(uuid.uuid4())
            return uow.outreach.create_prospect(
                prospect_id=prospect_id,
                full_name=prospect.full_name,
                company_name=prospect.company_name,
                company_domain=prospect.company_domain,
                designation=prospect.designation,
                linkedin_url=prospect.linkedin_url,
                location=prospect.location,
            )

    def _get_or_create_objectives(
        self,
        session_id: str,
        prospect: ProspectContext,
        outreach: OutreachContext,
        product: Optional[ProductContext],
    ) -> List[ResearchObjective]:
        with self.state_store.uow() as uow:
            existing_rows = uow.research.get_objectives(session_id)
            if existing_rows:
                return [
                    ResearchObjective(
                        objective_id=r["objective_id"],
                        type=ResearchObjectiveType(r["type"]),
                        target_entity=TargetEntity(r["target_entity"]),
                        description=r["description"],
                        temporal_scope=r["temporal_scope"],
                        priority=r["priority"],
                        is_critical=bool(r.get("is_critical", 1)),
                    )
                    for r in existing_rows
                ]

        # Generate new plan
        plan = self.planner.plan(
            prospect=prospect,
            outreach=outreach,
            product=product,
            session_id=session_id,
        )

        with self.state_store.uow() as uow:
            for obj in plan.objectives:
                uow.research.create_objective(
                    objective_id=obj.objective_id,
                    session_id=session_id,
                    type=obj.type.value,
                    target_entity=obj.target_entity.value,
                    description=obj.description,
                    temporal_scope=obj.temporal_scope,
                    priority=obj.priority,
                    is_critical=obj.is_critical,
                )

        return list(plan.objectives)

    def _hydrate_accumulator(self, session_id: str) -> ResearchAccumulator:
        accumulator = ResearchAccumulator()
        with self.state_store.uow() as uow:
            # Hydrate queries
            queries = uow.research.get_queries(session_id=session_id)
            for q in queries:
                if q["query_text"] not in accumulator.search_queries_used:
                    accumulator.search_queries_used.append(q["query_text"])

            # Hydrate snapshots & sources
            snapshots = uow.sources.get_snapshots_for_session(session_id)
            for snap in snapshots:
                src_row = uow.sources.get_source(snap["source_id"])
                if src_row:
                    source_record = SourceRecord(
                        source_id=src_row["source_id"],
                        url=src_row["canonical_url"],
                        original_url=snap.get("original_url"),
                        final_url=snap.get("final_url"),
                        html_canonical_url=snap.get("html_canonical_url"),
                        domain=src_row["domain"],
                        title=snap.get("title"),
                        raw_text=snap.get("cleaned_text") or "",
                        source_type=SourceType(src_row.get("source_type", "other")),
                        credibility_score=src_row.get("credibility_score", 0.7),
                        fetch_date=datetime.now(timezone.utc).date(),
                    )
                    accumulator.add_source(source_record)

            # Hydrate facts
            fact_rows = uow.assertions.get_facts_for_session(session_id)
            for f_row in fact_rows:
                ev_rows = uow.assertions.get_evidence_for_fact(f_row["fact_id"])
                ev_records = [
                    EvidenceRecord(
                        evidence_id=ev["evidence_id"],
                        source_id=ev["source_id"],
                        verbatim_excerpt=ev["verbatim_text"],
                        confidence=ConfidenceLevel.MEDIUM,
                    )
                    for ev in ev_rows
                ]
                if ev_records:
                    accumulator.facts.append(
                        CanonicalFact(
                            fact_id=f_row["fact_id"],
                            claim_text=f_row["claim_text"],
                            fact_category=FactCategory(f_row.get("fact_category", "other")),
                            evidence_records=ev_records,
                            composite_confidence=ConfidenceLevel(f_row.get("composite_confidence", "medium")),
                        )
                    )

            # Hydrate resolved entities
            assessments = uow.entities.get_assessments(session_id=session_id)
            for ass in assessments:
                ent_row = uow.entities.get_entity(ass["entity_id"])
                if ent_row:
                    ent_rec = EntityRecord(
                        entity_id=ent_row["entity_id"],
                        entity_type=EntityType(ent_row["entity_type"]),
                        raw_name=ent_row["raw_name"],
                        canonical_name=ent_row["canonical_name"],
                        domain=ent_row.get("domain"),
                        linkedin_url=ent_row.get("linkedin_url"),
                        resolution_status=ResolutionStatus(ent_row.get("resolution_status", "resolved")),
                        confidence=ConfidenceLevel(ent_row.get("confidence", "high")),
                    )
                    if ent_row["entity_type"] == "person" and not accumulator.prospect_entity:
                        accumulator.prospect_entity = ent_rec
                    elif ent_row["entity_type"] == "organization" and not accumulator.company_entity:
                        accumulator.company_entity = ent_rec

        return accumulator

    def _determine_resumption_point(
        self,
        session_id: str,
        max_iterations: int,
    ) -> tuple[int, Optional[ResearchSufficiencyAssessment]]:
        with self.state_store.uow() as uow:
            iterations = uow.orchestration.list_iterations(session_id)
            if not iterations:
                return 1, None

            latest = uow.orchestration.get_latest_iteration(session_id)
            if not latest:
                return 1, None

            latest_ass = uow.orchestration.get_latest_assessment(session_id)

            # If latest iteration failed or was in progress, resume on the same number
            if latest["status"] in ("in_progress", "failed", "interrupted"):
                return latest["iteration_number"], None

            # If completed, check assessment
            if latest_ass and latest_ass.get("decision") == SufficiencyDecision.CONTINUE.value:
                return latest["iteration_number"] + 1, None

            return latest["iteration_number"], None

    def _execute_searches_and_fetches(
        self,
        session_id: str,
        queries: Sequence[GeneratedQuery],
        accumulator: ResearchAccumulator,
        max_results: int,
        max_pages: int,
    ) -> List[str]:
        raw_occurrences: List[Dict[str, Any]] = []

        for q in queries:
            start_t = time.time()
            resp: SearchResponse = self.search_client.search(query=q.query_text, max_results=max_results)
            duration_ms = (time.time() - start_t) * 1000

            execution_id = str(uuid.uuid4())
            provider = getattr(self.search_client, "provider", None)
            provider_name = getattr(provider, "name", "duckduckgo") if provider else "duckduckgo"
            idempotency_key = f"{q.query_id}_{provider_name}"

            with self.state_store.uow() as uow:
                uow.research.record_search_execution(
                    execution_id=execution_id,
                    query_id=q.query_id,
                    provider=provider_name,
                    status=resp.status.value,
                    error_message=resp.error_message,
                    duration_ms=duration_ms,
                    idempotency_key=idempotency_key,
                )

                if resp.status == SearchStatus.SUCCESS and resp.results:
                    for pos, res in enumerate(resp.results, start=1):
                        source_id = str(uuid.uuid4())
                        saved_source = uow.sources.upsert_source(
                            source_id=source_id,
                            canonical_url=res.url,
                            domain=res.domain,
                            source_type="other",
                            credibility_score=0.7,
                            credibility_rationale="Discovered via search",
                        )
                        actual_source_id = saved_source["source_id"]

                        occ_id = str(uuid.uuid4())
                        uow.research.record_search_result_occurrence(
                            occurrence_id=occ_id,
                            execution_id=execution_id,
                            source_id=actual_source_id,
                            position=pos,
                            title=res.title,
                            snippet=res.snippet,
                            original_url=res.original_url or res.url,
                        )

                        raw_occurrences.append(
                            {
                                "source_id": actual_source_id,
                                "canonical_url": res.url,
                                "original_url": res.original_url or res.url,
                                "domain": res.domain,
                                "title": res.title,
                            }
                        )

        # Unique URLs to fetch not yet fetched
        new_urls: List[str] = []
        for occ in raw_occurrences:
            url = occ["canonical_url"]
            if url not in accumulator.seen_urls and url not in new_urls:
                new_urls.append(url)

        urls_to_fetch = new_urls[:max_pages]
        created_snapshot_ids: List[str] = []

        for url in urls_to_fetch:
            outcome: FetchOutcome = self.page_fetcher.fetch(url)

            with self.state_store.uow() as uow:
                source_record = uow.sources.get_source_by_url(url)
                if not source_record:
                    source_record = uow.sources.upsert_source(
                        source_id=str(uuid.uuid4()),
                        canonical_url=url,
                        domain=outcome.url,
                    )
                source_id = source_record["source_id"]
                snapshot_id = str(uuid.uuid4())

                if outcome.status == FetchStatus.SUCCESS and outcome.source_record:
                    sr = outcome.source_record.model_copy(update={"source_id": source_id})
                    uow.sources.create_snapshot(
                        snapshot_id=snapshot_id,
                        source_id=source_id,
                        session_id=session_id,
                        fetch_status=outcome.status.value,
                        original_url=sr.original_url,
                        final_url=sr.final_url,
                        html_canonical_url=sr.html_canonical_url,
                        title=sr.title,
                        raw_text=sr.raw_text,
                        http_status_code=outcome.http_status_code or 200,
                        duration_ms=outcome.duration_ms,
                    )
                    created_snapshot_ids.append(snapshot_id)
                    accumulator.add_source(sr)

        return created_snapshot_ids

    def _run_entity_resolution(
        self,
        session_id: str,
        prospect: ProspectContext,
        accumulator: ResearchAccumulator,
    ) -> List[EntityRecord]:
        resolution_result = self.entity_service.resolve_entities_for_session(
            session_id=session_id,
            prospect=prospect,
            sources=accumulator.sources,
        )

        known_entities: List[EntityRecord] = []
        if resolution_result.prospect_entity:
            accumulator.prospect_entity = resolution_result.prospect_entity
            known_entities.append(resolution_result.prospect_entity)
        if resolution_result.company_entity:
            accumulator.company_entity = resolution_result.company_entity
            known_entities.append(resolution_result.company_entity)

        return known_entities
