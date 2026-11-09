from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from src.context.models import ProspectContext
from src.db.state_store import StateStore
from src.researcher.entity.candidates import (
    DiscoveredEntityCandidate,
    EntityCandidateGenerator,
    normalize_org_name,
)
from src.researcher.entity.resolver import (
    IdentityResolutionResult,
    IdentityResolver,
)

logger = logging.getLogger(__name__)


class EntityResolutionService:
    """Orchestrates candidate generation, LLM identity resolution, and database persistence.
    
    Guarantees:
      - Short, strictly bounded UnitOfWork database transactions.
      - Zero database locks or transactions held open during candidate extraction or LLM calls.
      - Discovered candidates are persisted before resolution evaluation.
      - Assessments, entities, affiliations (with temporal dates), and identifiers are atomically saved.
    """

    def __init__(
        self,
        state_store: StateStore,
        candidate_generator: Optional[EntityCandidateGenerator] = None,
        resolver: Optional[IdentityResolver] = None,
    ) -> None:
        self.state_store = state_store
        self.candidate_generator = candidate_generator or EntityCandidateGenerator()
        self.resolver = resolver or IdentityResolver()

    def resolve_entities_for_session(
        self,
        session_id: str,
        prospect: ProspectContext,
        sources: List[Any],
        search_results: Optional[List[Any]] = None,
    ) -> IdentityResolutionResult:
        """Execute candidate discovery, identity resolution, and persistence for a research session."""
        logger.info(
            "Starting entity resolution for prospect %s at %s (Session: %s)",
            prospect.full_name,
            prospect.company_name,
            session_id,
        )

        # Stage 1: Deterministic candidate generation (in-memory, NO DB transaction)
        candidates: List[DiscoveredEntityCandidate] = self.candidate_generator.generate_candidates(
            prospect=prospect,
            sources=sources,
            search_results=search_results,
        )

        # Stage 2: Persist Candidates (Transaction 1 - bounded)
        with self.state_store.uow() as uow:
            # Ensure referenced sources exist in sources table so FK constraint is satisfied
            source_id_map: Dict[str, str] = {}
            for src in sources:
                sid = getattr(src, "source_id", None)
                surl = getattr(src, "url", None) or getattr(src, "canonical_url", None)
                sdom = getattr(src, "domain", None) or "unknown"
                if sid:
                    existing_id = uow.sources.get_source(sid)
                    if existing_id:
                        source_id_map[sid] = existing_id["source_id"]
                        continue
                if surl:
                    existing_url = uow.sources.get_source_by_url(surl)
                    if existing_url:
                        source_id_map[sid] = existing_url["source_id"]
                        continue
                if sid and surl:
                    created = uow.sources.upsert_source(
                        source_id=sid,
                        canonical_url=surl,
                        domain=sdom,
                    )
                    source_id_map[sid] = created.get("source_id", sid)

            for c in candidates:
                raw_sid = c.primary_source_id
                resolved_sid = source_id_map.get(raw_sid, raw_sid) if raw_sid else None
                # Verify that resolved_sid actually exists in sources table
                if resolved_sid:
                    src_check = uow.sources.get_source(resolved_sid)
                    if not src_check:
                        resolved_sid = None

                uow.entities.record_candidate(
                    candidate_id=c.candidate_id,
                    session_id=session_id,
                    entity_type=c.entity_type.value,
                    raw_name=c.raw_name,
                    attributes=c.to_attributes_json(),
                    source_id=resolved_sid,
                )

        # Stage 3: LLM Identity Resolution & Guardrail Enforcement (NO DB transaction held!)
        result: IdentityResolutionResult = self.resolver.resolve(
            prospect=prospect,
            candidates=candidates,
            sources=sources,
            session_id=session_id,
        )

        # Stage 4: Persist Entities, Identifiers, Affiliations, and Assessment (Transaction 2 - bounded)
        with self.state_store.uow() as uow:
            # 1. Upsert or create prospect entity
            existing_p = uow.entities.get_entity(result.prospect_entity.entity_id)
            if not existing_p:
                uow.entities.create_entity(
                    entity_id=result.prospect_entity.entity_id,
                    entity_type=result.prospect_entity.entity_type.value,
                    raw_name=result.prospect_entity.raw_name,
                    canonical_name=result.prospect_entity.canonical_name,
                    resolution_status=result.prospect_entity.resolution_status.value,
                    confidence=result.prospect_entity.confidence.value,
                    disambiguation_notes=result.prospect_entity.disambiguation_notes,
                )
            else:
                uow.entities.update_entity(
                    entity_id=result.prospect_entity.entity_id,
                    canonical_name=result.prospect_entity.canonical_name,
                    resolution_status=result.prospect_entity.resolution_status.value,
                    confidence=result.prospect_entity.confidence.value,
                    disambiguation_notes=result.prospect_entity.disambiguation_notes,
                )

            # 2. Upsert or create company entity
            existing_c = uow.entities.get_entity(result.company_entity.entity_id)
            if not existing_c:
                uow.entities.create_entity(
                    entity_id=result.company_entity.entity_id,
                    entity_type=result.company_entity.entity_type.value,
                    raw_name=result.company_entity.raw_name,
                    canonical_name=result.company_entity.canonical_name,
                    resolution_status=result.company_entity.resolution_status.value,
                    confidence=result.company_entity.confidence.value,
                    disambiguation_notes=result.company_entity.disambiguation_notes,
                )
            else:
                uow.entities.update_entity(
                    entity_id=result.company_entity.entity_id,
                    canonical_name=result.company_entity.canonical_name,
                    resolution_status=result.company_entity.resolution_status.value,
                    confidence=result.company_entity.confidence.value,
                    disambiguation_notes=result.company_entity.disambiguation_notes,
                )

            # 3. Add identifiers
            if result.prospect_entity.linkedin_url:
                uow.entities.add_identifier(
                    identifier_id=str(uuid.uuid4()),
                    entity_id=result.prospect_entity.entity_id,
                    id_type="linkedin_url",
                    id_value=result.prospect_entity.linkedin_url,
                )
            if result.company_entity.domain:
                uow.entities.add_identifier(
                    identifier_id=str(uuid.uuid4()),
                    entity_id=result.company_entity.entity_id,
                    id_type="company_domain",
                    id_value=result.company_entity.domain,
                )
            for ident in result.identifiers:
                target_eid = (
                    result.prospect_entity.entity_id
                    if ident.id_type == "linkedin_url"
                    else result.company_entity.entity_id
                )
                uow.entities.add_identifier(
                    identifier_id=str(uuid.uuid4()),
                    entity_id=target_eid,
                    id_type=ident.id_type,
                    id_value=ident.id_value,
                    source_id=ident.source_id,
                )

            # 4. Add aliases
            for alias in result.prospect_entity.aliases:
                uow.entities.add_alias(
                    alias_id=str(uuid.uuid4()),
                    entity_id=result.prospect_entity.entity_id,
                    alias=alias,
                )
            for alias in result.company_entity.aliases:
                uow.entities.add_alias(
                    alias_id=str(uuid.uuid4()),
                    entity_id=result.company_entity.entity_id,
                    alias=alias,
                )

            # 5. Add affiliations (temporal with start_date and end_date, never single current_company)
            for aff in result.affiliations:
                org_id = result.company_entity.entity_id
                if normalize_org_name(aff.organization_name) != normalize_org_name(result.company_entity.raw_name):
                    other_org_id = str(uuid.uuid4())
                    uow.entities.create_entity(
                        entity_id=other_org_id,
                        entity_type="organization",
                        raw_name=aff.organization_name,
                        canonical_name=aff.organization_name,
                    )
                    org_id = other_org_id

                uow.entities.add_affiliation(
                    affiliation_id=str(uuid.uuid4()),
                    entity_id=result.prospect_entity.entity_id,
                    organization_id=org_id,
                    role=aff.role,
                    start_date=aff.start_date,
                    end_date=aff.end_date,
                    source_id=aff.source_id,
                )

            # 6. Add relationships (e.g. acquired_by, subsidiary_of)
            for rel in result.relationships:
                other_org_id = str(uuid.uuid4())
                uow.entities.create_entity(
                    entity_id=other_org_id,
                    entity_type="organization",
                    raw_name=rel.related_entity_name,
                    canonical_name=rel.related_entity_name,
                )
                uow.entities.add_relationship(
                    relationship_id=str(uuid.uuid4()),
                    from_entity_id=result.company_entity.entity_id,
                    to_entity_id=other_org_id,
                    relationship_type=rel.relationship_type,
                    valid_from=rel.valid_from,
                    valid_until=rel.valid_until,
                    source_id=rel.source_id,
                )

            # 7. Record Identity Assessment
            uow.entities.record_identity_assessment(
                assessment_id=result.assessment_id,
                session_id=session_id,
                entity_id=result.prospect_entity.entity_id,
                candidate_ids=json.dumps(result.matched_candidate_ids),
                resolution_status=result.resolution_status.value,
                confidence=result.confidence.value,
                rationale=result.rationale,
                assessed_by="llm",
            )

        logger.info(
            "Identity resolution completed and persisted for %s. Status: %s, Confidence: %s",
            prospect.full_name,
            result.resolution_status.value,
            result.confidence.value,
        )
        return result
