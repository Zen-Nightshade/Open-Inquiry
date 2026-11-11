from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from datetime import date
from src.db.state_store import StateStore
from src.researcher.extraction.consolidator import (
    FactConsolidationResult,
    FactConsolidator,
)
from src.researcher.extraction.extractor import FactExtractor
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
)
from src.researcher.molecular_models import (
    deserialize_molecular_facts,
    serialize_molecular_facts,
)

logger = logging.getLogger(__name__)


class FactExtractionService:
    """Orchestrates LLM fact extraction, excerpt validation, consolidation, and database persistence.
    
    Guarantees:
      - Deterministic excerpt verification before any fact enters the database or accumulator.
      - External LLM calls execute strictly outside database transactions (zero locks held).
      - Persists canonical facts, evidence passages, entity links, evidence links, conflicts, and assessment history.
      - Tracks syndication groups to prevent repeated copies from falsely inflating confidence.
    """

    def __init__(
        self,
        state_store: StateStore,
        extractor: Optional[FactExtractor] = None,
        consolidator: Optional[FactConsolidator] = None,
    ) -> None:
        self.state_store = state_store
        self.extractor = extractor or FactExtractor()
        self.consolidator = consolidator or FactConsolidator()

    def extract_and_persist_facts_for_session(
        self,
        session_id: str,
        snapshot_id: str,
        entity_context: Optional[Dict[str, Any]] = None,
        known_entities: Optional[List[EntityRecord]] = None,
        accumulator: Optional[ResearchAccumulator] = None,
    ) -> List[CanonicalFact]:
        """Extract validated facts from a source snapshot, consolidate into accumulator, and persist to database."""
        # Step 1: Read snapshot outside long transaction
        with self.state_store.uow() as uow:
            snapshot = uow.sources.get_snapshot(snapshot_id)
            if not snapshot:
                logger.error("Snapshot %s not found in database", snapshot_id)
                return []

            source_id = snapshot["source_id"]
            source_row = uow.sources.get_source(source_id)
            source_domain = source_row["domain"] if source_row else ""

        # Step 2: Run LLM extraction and deterministic excerpt validation (NO DB TRANSACTION)
        extracted_facts: List[CanonicalFact] = self.extractor.extract(
            source=snapshot,
            entity_context=entity_context,
            known_entities=known_entities,
        )

        if not extracted_facts:
            logger.info("No validated facts extracted for snapshot %s", snapshot_id)
            return []

        # Step 3: Read syndication origin groups and all source domains for independence checking
        with self.state_store.uow() as uow:
            syndication_groups = uow.sources.get_syndication_groups()
            all_sources = uow.sources.list_sources()
            source_domains: Dict[str, str] = {
                s["source_id"]: s["domain"]
                for s in all_sources
                if s.get("source_id") and s.get("domain")
            }
            if source_id and source_domain:
                source_domains[source_id] = source_domain

        # Step 4: Consolidate with accumulator, preloading existing session facts if needed
        active_acc = accumulator or ResearchAccumulator()
        if not active_acc.facts:
            with self.state_store.uow() as uow:
                db_facts = uow.assertions.get_facts_for_session(session_id)
                for df in db_facts:
                    db_ev = uow.assertions.get_evidence_for_fact(df["fact_id"])
                    ev_records = [
                        EvidenceRecord(
                            evidence_id=e["evidence_id"],
                            source_id=e["source_id"],
                            verbatim_excerpt=e["verbatim_excerpt"],
                            event_date=date.fromisoformat(e["event_date"]) if e["event_date"] else None,
                            confidence=ConfidenceLevel(e["confidence"]),
                            credibility_score=e["credibility_score"],
                        )
                        for e in db_ev
                    ]
                    linked_ents = uow.assertions.get_entities_for_fact(df["fact_id"])
                    ambig_cids = (
                        json.loads(df["ambiguous_candidates_json"])
                        if df.get("ambiguous_candidates_json")
                        else []
                    )
                    c_fact = CanonicalFact(
                        fact_id=df["fact_id"],
                        claim_text=df["claim_text"],
                        fact_category=FactCategory(df["fact_category"]),
                        entity_ids=[ent["entity_id"] for ent in linked_ents],
                        evidence_records=ev_records if ev_records else [
                            EvidenceRecord(source_id=source_id, verbatim_excerpt=df["claim_text"])
                        ],
                        primary_event_date=(
                            date.fromisoformat(df["primary_event_date"])
                            if df["primary_event_date"]
                            else None
                        ),
                        composite_confidence=ConfidenceLevel(df["composite_confidence"]),
                        molecular_facts=deserialize_molecular_facts(df.get("molecular_data_json")),
                        valid_from=df.get("valid_from"),
                        valid_to=df.get("valid_to"),
                        observed_at=df.get("observed_at"),
                        recorded_at=df.get("recorded_at"),
                        independent_source_count=df.get("independent_source_count"),
                        ambiguous_candidate_ids=ambig_cids,
                        entity_resolution_status=df.get("entity_resolution_status"),
                    )
                    active_acc.facts.append(c_fact)

        consolidation_result: FactConsolidationResult = self.consolidator.consolidate(
            new_facts=extracted_facts,
            accumulator=active_acc,
            syndication_groups=syndication_groups,
            source_domains=source_domains,
        )

        # Step 5: Persist facts, passages, links, and conflicts (Short Bounded Transaction)
        persisted_facts: List[CanonicalFact] = []
        with self.state_store.uow() as uow:
            # Build entity status lookup from known_entities or database
            entity_status_lookup: Dict[str, str] = {}
            if known_entities:
                for ke in known_entities:
                    stat = ke.resolution_status.value if hasattr(ke.resolution_status, "value") else str(ke.resolution_status)
                    entity_status_lookup[ke.entity_id] = stat

            for fact in consolidation_result.consolidated_facts:
                fact_res_status = fact.entity_resolution_status
                ambiguous_cids = list(fact.ambiguous_candidate_ids)
                confirmed_entity_ids: List[str] = []

                for eid in fact.entity_ids:
                    status = entity_status_lookup.get(eid)
                    if not status:
                        db_ent = uow.entities.get_entity(eid)
                        if db_ent:
                            status = db_ent.get("resolution_status", "unresolved")
                            entity_status_lookup[eid] = status

                    if status == "ambiguous":
                        if eid not in ambiguous_cids:
                            ambiguous_cids.append(eid)
                        fact_res_status = "ambiguous"
                    elif status == "unresolved":
                        if not fact_res_status or fact_res_status == "match":
                            fact_res_status = "unresolved"
                    elif status == "resolved":
                        confirmed_entity_ids.append(eid)
                        if not fact_res_status:
                            fact_res_status = "match"
                    else:
                        confirmed_entity_ids.append(eid)

                if ambiguous_cids:
                    fact_res_status = "ambiguous"

                if fact_res_status != fact.entity_resolution_status or ambiguous_cids != fact.ambiguous_candidate_ids:
                    fact = fact.model_copy(
                        update={
                            "entity_resolution_status": fact_res_status,
                            "ambiguous_candidate_ids": ambiguous_cids,
                        }
                    )
                persisted_facts.append(fact)

                # 1. Create or update canonical fact
                ambiguous_json = (
                    json.dumps(fact.ambiguous_candidate_ids)
                    if fact.ambiguous_candidate_ids
                    else None
                )
                uow.assertions.create_canonical_fact(
                    fact_id=fact.fact_id,
                    session_id=session_id,
                    claim_text=fact.claim_text,
                    fact_category=fact.fact_category.value,
                    primary_event_date=(
                        fact.primary_event_date.isoformat() if fact.primary_event_date else None
                    ),
                    composite_confidence=fact.composite_confidence.value,
                    molecular_data_json=(
                        serialize_molecular_facts(fact.molecular_facts)
                        if fact.molecular_facts
                        else None
                    ),
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    observed_at=fact.observed_at,
                    recorded_at=fact.recorded_at,
                    independent_source_count=fact.independent_source_count,
                    ambiguous_candidates_json=ambiguous_json,
                    entity_resolution_status=fact.entity_resolution_status,
                )

                # 2. Link fact to entities
                for eid in fact.entity_ids:
                    if uow.entities.get_entity(eid):
                        uow.assertions.link_fact_entity(
                            fact_id=fact.fact_id,
                            entity_id=eid,
                        )

                # 3. Create evidence passages and link to fact
                for ev in fact.evidence_records:
                    uow.sources.create_evidence_passage(
                        evidence_id=ev.evidence_id,
                        snapshot_id=snapshot_id,
                        source_id=ev.source_id,
                        verbatim_excerpt=ev.verbatim_excerpt,
                        excerpt_validated=1,
                        event_date=ev.event_date.isoformat() if ev.event_date else None,
                        extracted_at=ev.extracted_date.isoformat() if ev.extracted_date else None,
                        confidence=ev.confidence.value,
                        credibility_score=ev.credibility_score,
                    )
                    uow.assertions.link_fact_evidence(
                        fact_id=fact.fact_id,
                        evidence_id=ev.evidence_id,
                        support_type="supporting",
                    )

            # 4. Persist detected conflicts
            for conflict in consolidation_result.conflicts:
                uow.assertions.create_conflict(
                    conflict_id=conflict.conflict_id,
                    session_id=session_id,
                    description=conflict.description,
                    status="open",
                )
                uow.assertions.link_conflict_fact(
                    conflict_id=conflict.conflict_id,
                    fact_id=conflict.fact_id_a,
                )
                uow.assertions.link_conflict_fact(
                    conflict_id=conflict.conflict_id,
                    fact_id=conflict.fact_id_b,
                )

            # 5. Record assessment history entry with incremental iteration
            existing_hist = uow.assertions.get_assessment_history(session_id)
            current_iteration = (max((h["iteration"] for h in existing_hist), default=0) + 1)

            uow.assertions.record_assessment_history(
                assessment_id=str(uuid.uuid4()),
                session_id=session_id,
                iteration=current_iteration,
                assessor="FactExtractionService",
                finding_type="facts_extracted",
                details=json.dumps(
                    {
                        "snapshot_id": snapshot_id,
                        "facts_extracted": len(consolidation_result.consolidated_facts),
                        "conflicts_detected": len(consolidation_result.conflicts),
                    }
                ),
            )

        logger.info(
            "Persisted %d facts and %d conflicts for session %s (Snapshot: %s)",
            len(consolidation_result.consolidated_facts),
            len(consolidation_result.conflicts),
            session_id,
            snapshot_id,
        )
        return persisted_facts
