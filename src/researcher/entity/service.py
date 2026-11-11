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
from src.researcher.entity.probabilistic_models import (
    RelationshipPredicate,
    ResolutionOutcome,
    TypedEntityRelationship,
)
from src.researcher.entity.probabilistic_resolver import (
    ProbabilisticEntityResolver,
    ProbabilisticResolutionResult,
)
from src.researcher.entity.resolver import (
    IdentityResolutionResult,
    IdentityResolver,
)
from src.researcher.models import ConfidenceLevel, EntityType, ResolutionStatus

logger = logging.getLogger(__name__)


class EntityResolutionService:
    """Orchestrates candidate generation, probabilistic identity resolution, and database persistence.
    
    Guarantees:
      - Short, strictly bounded UnitOfWork database transactions.
      - Zero database locks or transactions held open during candidate extraction or resolution.
      - Discovered candidates are persisted before resolution evaluation.
      - Assessments, entities, affiliations (with temporal dates), and identifiers are atomically saved.
      - Enforces explainable margin separation and uncertainty preservation.
    """

    def __init__(
        self,
        state_store: StateStore,
        candidate_generator: Optional[EntityCandidateGenerator] = None,
        resolver: Optional[IdentityResolver] = None,
        probabilistic_resolver: Optional[ProbabilisticEntityResolver] = None,
    ) -> None:
        self.state_store = state_store
        self.candidate_generator = candidate_generator or EntityCandidateGenerator()
        self.resolver = resolver or IdentityResolver()
        self.probabilistic_resolver = probabilistic_resolver or ProbabilisticEntityResolver()

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

        # Stage 3: Probabilistic Evaluation & Resolution
        prob_prospect = self.probabilistic_resolver.resolve_candidates(
            candidates=candidates,
            target_prospect=prospect,
            entity_type=EntityType.PERSON,
        )
        prob_company = self.probabilistic_resolver.resolve_candidates(
            candidates=candidates,
            target_prospect=prospect,
            entity_type=EntityType.ORGANIZATION,
        )

        result: IdentityResolutionResult = self.resolver.resolve(
            prospect=prospect,
            candidates=candidates,
            sources=sources,
            session_id=session_id,
        )

        p_entity = result.prospect_entity
        c_entity = result.company_entity
        status = result.resolution_status
        conf = result.confidence
        matched_cids = list(result.matched_candidate_ids)
        conflicts = list(result.conflicts_or_ambiguities)

        # Enforce probabilistic resolution policy over prospect entity
        if prob_prospect.outcome == ResolutionOutcome.AMBIGUOUS:
            p_entity = p_entity.model_copy(
                update={
                    "resolution_status": ResolutionStatus.AMBIGUOUS,
                    "confidence": ConfidenceLevel.LOW,
                    "disambiguation_notes": prob_prospect.rationale,
                }
            )
            status = ResolutionStatus.AMBIGUOUS
            conf = ConfidenceLevel.LOW
            matched_cids = prob_prospect.competing_candidate_ids
            if prob_prospect.rationale not in conflicts:
                conflicts.append(prob_prospect.rationale)
        elif prob_prospect.outcome == ResolutionOutcome.UNRESOLVED:
            p_entity = p_entity.model_copy(
                update={
                    "resolution_status": ResolutionStatus.UNRESOLVED,
                    "confidence": ConfidenceLevel.UNRESOLVED,
                    "disambiguation_notes": prob_prospect.rationale,
                }
            )
            status = ResolutionStatus.UNRESOLVED
            conf = ConfidenceLevel.UNRESOLVED
            matched_cids = []
        elif prob_prospect.outcome == ResolutionOutcome.NO_MATCH:
            if p_entity.resolution_status != ResolutionStatus.PARTIALLY_RESOLVED:
                p_entity = p_entity.model_copy(
                    update={
                        "resolution_status": ResolutionStatus.UNRESOLVED,
                        "confidence": ConfidenceLevel.LOW,
                        "disambiguation_notes": prob_prospect.rationale,
                    }
                )
                status = ResolutionStatus.UNRESOLVED
                conf = ConfidenceLevel.LOW
                matched_cids = []
        elif prob_prospect.outcome == ResolutionOutcome.MATCH:
            if prob_prospect.selected_candidate:
                cand = prob_prospect.selected_candidate
                p_entity = p_entity.model_copy(
                    update={
                        "resolution_status": ResolutionStatus.RESOLVED,
                        "confidence": prob_prospect.confidence,
                        "canonical_name": cand.raw_name,
                        "disambiguation_notes": prob_prospect.rationale,
                    }
                )
                matched_cids = [prob_prospect.selected_candidate_id]
                status = ResolutionStatus.RESOLVED
                conf = prob_prospect.confidence

        # Enforce probabilistic resolution policy over company entity
        if prob_company.outcome == ResolutionOutcome.AMBIGUOUS:
            c_entity = c_entity.model_copy(
                update={
                    "resolution_status": ResolutionStatus.AMBIGUOUS,
                    "confidence": ConfidenceLevel.LOW,
                    "disambiguation_notes": prob_company.rationale,
                }
            )
            if status == ResolutionStatus.RESOLVED:
                status = ResolutionStatus.AMBIGUOUS
                conf = ConfidenceLevel.LOW
            if prob_company.rationale not in conflicts:
                conflicts.append(prob_company.rationale)
        elif prob_company.outcome == ResolutionOutcome.UNRESOLVED:
            c_entity = c_entity.model_copy(
                update={
                    "resolution_status": ResolutionStatus.UNRESOLVED,
                    "confidence": ConfidenceLevel.UNRESOLVED,
                    "disambiguation_notes": prob_company.rationale,
                }
            )
            if status == ResolutionStatus.RESOLVED:
                status = ResolutionStatus.UNRESOLVED
                conf = ConfidenceLevel.UNRESOLVED
            if prob_company.rationale not in conflicts:
                conflicts.append(prob_company.rationale)
        elif prob_company.outcome == ResolutionOutcome.MATCH and prob_company.selected_candidate:
            cand = prob_company.selected_candidate
            c_entity = c_entity.model_copy(
                update={
                    "resolution_status": ResolutionStatus.RESOLVED,
                    "confidence": prob_company.confidence,
                    "canonical_name": cand.raw_name,
                    "domain": cand.domains[0] if cand.domains else c_entity.domain,
                    "disambiguation_notes": prob_company.rationale,
                }
            )

        result = result.model_copy(
            update={
                "prospect_entity": p_entity,
                "company_entity": c_entity,
                "resolution_status": status,
                "confidence": conf,
                "matched_candidate_ids": matched_cids,
                "conflicts_or_ambiguities": conflicts,
            }
        )

        # Stage 4: Persist Entities, Identifiers, Affiliations, Relationships, and Assessment
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

            # 5. Add affiliations and typed temporal relationships (works_for)
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

            # 6. Add inter-entity relationships (e.g. acquired_by, subsidiary_of)
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

            # 7. Record Identity Assessment with feature breakdown and thresholds
            decision_meta = {
                "prospect_decision": {
                    "outcome": prob_prospect.outcome.value,
                    "best_score": prob_prospect.best_score,
                    "runner_up_score": prob_prospect.runner_up_score,
                    "score_margin": prob_prospect.score_margin,
                    "competing_ids": prob_prospect.competing_candidate_ids,
                    "thresholds": prob_prospect.thresholds,
                },
                "company_decision": {
                    "outcome": prob_company.outcome.value,
                    "best_score": prob_company.best_score,
                    "thresholds": prob_company.thresholds,
                },
            }
            uow.entities.record_identity_assessment(
                assessment_id=result.assessment_id,
                session_id=session_id,
                entity_id=result.prospect_entity.entity_id,
                candidate_ids=json.dumps(result.matched_candidate_ids),
                resolution_status=result.resolution_status.value,
                confidence=result.confidence.value,
                rationale=result.rationale,
                assessed_by="probabilistic_resolver",
                decision_metadata_json=json.dumps(decision_meta),
            )

        logger.info(
            "Identity resolution completed and persisted for %s. Status: %s, Confidence: %s",
            prospect.full_name,
            result.resolution_status.value,
            result.confidence.value,
        )
        return result
