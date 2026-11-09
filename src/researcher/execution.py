from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.db.unit_of_work import UnitOfWork
from src.researcher.extraction.fetcher import PageFetcher
from src.researcher.extraction.models import FetchOutcome, FetchStatus
from src.researcher.planning.models import ResearchPlan
from src.researcher.planning.planner import ResearchPlanner
from src.researcher.search.models import SearchResponse, SearchResult, SearchStatus
from src.researcher.search.query_generator import GeneratedQuery, QueryGenerator
from src.researcher.search.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearchExecutionCoordinator:
    """Coordinates research planning, query generation, search, fetching, and persistence.
    
    Guarantees:
      - Every database mutation runs in a short, bounded UnitOfWork transaction.
      - Transactions are NEVER held open across external I/O (LLM, search, or HTTP fetching).
      - Intent is persisted before execution; outcomes are persisted immediately after.
      - Occurrence-level provenance is preserved when multiple searches discover the same canonical source.
      - Failures and empty results are tracked as distinct execution outcomes.
    """

    def __init__(
        self,
        state_store: StateStore,
        planner: Optional[ResearchPlanner] = None,
        query_generator: Optional[QueryGenerator] = None,
        search_client: Optional[SearchClient] = None,
        page_fetcher: Optional[PageFetcher] = None,
    ) -> None:
        self.state_store = state_store
        self.planner = planner or ResearchPlanner()
        self.query_generator = query_generator or QueryGenerator()
        self.search_client = search_client or SearchClient()
        self.page_fetcher = page_fetcher or PageFetcher()

    def execute_discovery(
        self,
        prospect: ProspectContext,
        outreach: OutreachContext,
        product: Optional[ProductContext] = None,
        session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        max_results_per_query: int = 5,
        max_pages_to_fetch: int = 8,
    ) -> Dict[str, Any]:
        """Execute Phase C research discovery: plan -> queries -> searches -> page fetches -> persistence."""
        active_session_id = session_id or str(uuid.uuid4())
        active_context_id = context_id or str(uuid.uuid4())

        # Ensure prospect is persisted in database
        prospect_record = self._ensure_prospect_persisted(prospect)
        prospect_id = prospect_record["prospect_id"]

        # =========================================================================
        # Stage 1: Research Planning (LLM call - NO DB transaction)
        # =========================================================================
        logger.info("Generating research plan for prospect %s", prospect.full_name)
        plan: ResearchPlan = self.planner.plan(
            prospect=prospect,
            outreach=outreach,
            product=product,
            session_id=active_session_id,
        )

        # Persist Plan & Objectives Intent (Short Transaction 1)
        with self.state_store.uow() as uow:
            uow.research.create_session(
                session_id=active_session_id,
                prospect_id=prospect_id,
                context_id=active_context_id,
                status="in_progress",
            )
            for obj in plan.objectives:
                uow.research.create_objective(
                    objective_id=obj.objective_id,
                    session_id=active_session_id,
                    type=obj.type.value,
                    target_entity=obj.target_entity.value,
                    description=obj.description,
                    temporal_scope=obj.temporal_scope,
                    priority=obj.priority,
                )

        # =========================================================================
        # Stage 2: Query Generation (LLM calls - NO DB transaction)
        # =========================================================================
        all_generated_queries: List[GeneratedQuery] = []
        seen_query_texts: List[str] = []

        for obj in plan.objectives:
            queries = self.query_generator.generate_queries(
                objective=obj,
                prospect=prospect,
                existing_queries=seen_query_texts,
            )
            all_generated_queries.extend(queries)
            seen_query_texts.extend(q.query_text for q in queries)

        # Persist Query Intents (Short Transaction 2)
        with self.state_store.uow() as uow:
            for q in all_generated_queries:
                uow.research.create_query(
                    query_id=q.query_id,
                    session_id=active_session_id,
                    objective_id=q.objective_id,
                    query_text=q.query_text,
                )

        # =========================================================================
        # Stage 3: Search Execution & Occurrence Recording
        # =========================================================================
        raw_occurrences: List[Dict[str, Any]] = []

        for q in all_generated_queries:
            # Network call outside DB transaction
            start_t = time.time()
            resp: SearchResponse = self.search_client.search(
                query=q.query_text,
                max_results=max_results_per_query,
            )
            duration_ms = (time.time() - start_t) * 1000

            execution_id = str(uuid.uuid4())
            provider_name = getattr(self.search_client.provider, "name", "duckduckgo")
            idempotency_key = f"{q.query_id}_{provider_name}"

            # Persist search execution & result occurrences (Short Transaction per query)
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
                        # Upsert source to obtain canonical source_id
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

                        # Record distinct occurrence
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

        # =========================================================================
        # Stage 4: Fetch Web Pages & Record Snapshots
        # =========================================================================
        # Deduplicate candidates across queries, preserving URL order
        unique_urls_to_fetch: List[str] = []
        seen_fetch_urls: Set[str] = set()

        for item in raw_occurrences:
            c_url = item["canonical_url"]
            if c_url not in seen_fetch_urls:
                seen_fetch_urls.add(c_url)
                unique_urls_to_fetch.append(c_url)

        urls_to_fetch = unique_urls_to_fetch[:max_pages_to_fetch]
        fetched_count = 0

        for url in urls_to_fetch:
            # Network fetch outside DB transaction
            outcome: FetchOutcome = self.page_fetcher.fetch(url)

            # Persist snapshot (Short Transaction per fetched page)
            with self.state_store.uow() as uow:
                # Find matching canonical source
                source_record = uow.sources.get_source_by_url(url)
                if not source_record:
                    # Fallback upsert if not found
                    source_record = uow.sources.upsert_source(
                        source_id=str(uuid.uuid4()),
                        canonical_url=url,
                        domain=outcome.url,
                    )
                source_id = source_record["source_id"]

                snapshot_id = str(uuid.uuid4())

                if outcome.status == FetchStatus.SUCCESS and outcome.source_record:
                    sr = outcome.source_record
                    uow.sources.create_snapshot(
                        snapshot_id=snapshot_id,
                        source_id=source_id,
                        session_id=active_session_id,
                        original_url=sr.original_url,
                        final_url=sr.final_url,
                        html_canonical_url=sr.html_canonical_url,
                        title=sr.title,
                        raw_text=sr.raw_text,
                        fetch_status=outcome.status.value,
                        http_status_code=outcome.http_status_code,
                        publication_date=(
                            sr.publication_date.isoformat() if sr.publication_date else None
                        ),
                        publication_date_src=sr.publication_date_source,
                        fetch_date=sr.fetch_date.isoformat(),
                        duration_ms=outcome.duration_ms,
                    )
                    # Update source with refined metadata
                    uow.sources.upsert_source(
                        source_id=source_id,
                        canonical_url=sr.url,
                        domain=sr.domain,
                        source_type=sr.source_type.value,
                        credibility_score=sr.credibility_score,
                        credibility_rationale=sr.credibility_rationale,
                    )
                    fetched_count += 1
                else:
                    # Record unsuccessful fetch snapshot
                    uow.sources.create_snapshot(
                        snapshot_id=snapshot_id,
                        source_id=source_id,
                        session_id=active_session_id,
                        original_url=outcome.original_url,
                        final_url=outcome.final_url,
                        fetch_status=outcome.status.value,
                        http_status_code=outcome.http_status_code,
                        duration_ms=outcome.duration_ms,
                    )

        # =========================================================================
        # Stage 5: Finalize Session (Short Transaction 5)
        # =========================================================================
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.state_store.uow() as uow:
            uow.research.update_session(
                session_id=active_session_id,
                status="completed",
                completed_at=now_iso,
            )

        return {
            "session_id": active_session_id,
            "prospect_id": prospect_id,
            "plan": plan,
            "objectives_count": len(plan.objectives),
            "queries_count": len(all_generated_queries),
            "occurrences_count": len(raw_occurrences),
            "fetched_pages_count": fetched_count,
        }

    def _ensure_prospect_persisted(self, prospect: ProspectContext) -> Dict[str, Any]:
        """Verify if prospect is already stored or persist a new prospect record."""
        with self.state_store.uow() as uow:
            prospects = uow.outreach.list_prospects()
            for p in prospects:
                if (
                    p["full_name"].lower() == prospect.full_name.lower()
                    and p["company_name"].lower() == prospect.company_name.lower()
                ):
                    return p

            # Not found: create new prospect
            pid = str(uuid.uuid4())
            return uow.outreach.create_prospect(
                prospect_id=pid,
                full_name=prospect.full_name,
                company_name=prospect.company_name,
                company_domain=prospect.company_domain,
                designation=prospect.designation,
                linkedin_url=prospect.linkedin_url,
                location=prospect.location,
            )
