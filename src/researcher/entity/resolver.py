from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.config.settings import GROQ_SMALL
from src.context.models import ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.entity.candidates import (
    DiscoveredEntityCandidate,
    clean_domain,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.models import (
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    ResolutionStatus,
)

logger = logging.getLogger(__name__)


from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class AffiliationData(BaseModel):
    """Employment or advisory affiliation linking an individual to an organization."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    role: Optional[str] = None
    organization_name: str = Field(
        default="",
        validation_alias=AliasChoices("organization_name", "organization", "company", "company_name"),
    )
    is_current: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None  # None = ongoing
    source_id: Optional[str] = None


class IdentifierData(BaseModel):
    """External verifiable identifier (e.g. LinkedIn URL, company domain)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id_type: str  # 'linkedin_url' | 'company_domain' | 'twitter_handle' | etc.
    id_value: str
    source_id: Optional[str] = None


class RelationshipData(BaseModel):
    """Typed organizational relationship (e.g. parent_of, subsidiary_of, acquired_by)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    relationship_type: str
    related_entity_name: str = Field(
        default="",
        validation_alias=AliasChoices("related_entity_name", "entity_name", "target_company", "organization"),
    )
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    source_id: Optional[str] = None


class IdentityAssessmentOutput(BaseModel):
    """Structured output produced by the LLM identity resolution reasoner."""

    model_config = ConfigDict(populate_by_name=True)

    resolution_status: ResolutionStatus
    confidence: ConfidenceLevel
    matched_candidate_ids: List[str] = Field(default_factory=list)
    prospect_canonical_name: Optional[str] = None
    company_canonical_name: Optional[str] = None
    prospect_aliases: List[str] = Field(default_factory=list)
    company_aliases: List[str] = Field(default_factory=list)
    affiliations: List[AffiliationData] = Field(default_factory=list)
    identifiers: List[IdentifierData] = Field(default_factory=list)
    relationships: List[RelationshipData] = Field(default_factory=list)
    rationale: str = Field(description="Detailed explanation of matching or non-matching evidence")
    disambiguation_notes: Optional[str] = Field(
        default=None,
        description="Notes detailing why candidates were selected or rejected, and any ambiguity",
    )
    conflicts_or_ambiguities: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_root_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Unnest if wrapped under sub-keys
        for key in ("assessment", "identity_assessment", "result", "identity_resolution"):
            if key in data and isinstance(data[key], dict):
                merged = dict(data[key])
                for k, v in data.items():
                    if k != key and k not in merged:
                        merged[k] = v
                data = merged
                break

        if "resolution_status" not in data:
            status_val = data.get("status") or (data.get("target_identity", {}).get("status") if isinstance(data.get("target_identity"), dict) else None)
            data["resolution_status"] = status_val or ResolutionStatus.UNRESOLVED

        if "confidence" not in data:
            data["confidence"] = data.get("confidence_level") or ConfidenceLevel.LOW

        if "rationale" not in data:
            data["rationale"] = (
                data.get("reasoning")
                or data.get("explanation")
                or data.get("notes")
                or "Evaluated candidates against target context."
            )

        return data

    @field_validator("rationale", mode="before")
    @classmethod
    def normalize_rationale(cls, v: Any) -> Any:
        if isinstance(v, list):
            return " ".join(str(item) for item in v)
        return str(v) if v is not None else ""

    @field_validator("disambiguation_notes", mode="before")
    @classmethod
    def normalize_disambiguation_notes(cls, v: Any) -> Any:
        if isinstance(v, list):
            return " ".join(str(item) for item in v)
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            if v >= 0.75:
                return ConfidenceLevel.HIGH
            elif v >= 0.45:
                return ConfidenceLevel.MEDIUM
            elif v > 0.0:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.UNRESOLVED
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for level in ConfidenceLevel:
                if level.value == v_clean:
                    return level
        return v

    @field_validator("resolution_status", mode="before")
    @classmethod
    def normalize_resolution_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for status in ResolutionStatus:
                if status.value == v_clean:
                    return status
        return v


class IdentityResolutionResult(BaseModel):
    """Final, validated result of identity assessment including entity records."""

    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_prospect: ProspectContext
    resolution_status: ResolutionStatus
    confidence: ConfidenceLevel
    matched_candidate_ids: List[str] = Field(default_factory=list)
    prospect_entity: EntityRecord
    company_entity: EntityRecord
    affiliations: List[AffiliationData] = Field(default_factory=list)
    identifiers: List[IdentifierData] = Field(default_factory=list)
    relationships: List[RelationshipData] = Field(default_factory=list)
    rationale: str
    disambiguation_notes: Optional[str] = None
    conflicts_or_ambiguities: List[str] = Field(default_factory=list)


RESOLVER_SYSTEM_PROMPT = """You are an expert identity resolution and entity deduplication engine for B2B research.
Your task is to evaluate discovered entity candidates against the target prospect and target company across multiple attributes.

CRITICAL RESOLUTION RULES & GUARDRAILS:
1. Multi-Attribute Corroboration:
   - To assign status 'resolved', there MUST be matching name PLUS at least ONE independent corroborating attribute:
     * Matching web domain
     * Matching organization affiliation
     * Consistent job role/designation
     * Confirmed LinkedIn profile URL
   - Name similarity alone must NEVER result in 'resolved'. If only the name matches without corroborating attributes, status must be 'unresolved' or 'partially_resolved'.

2. Ambiguity & Conflicting Candidates:
   - If two or more candidates match the prospect's name but have conflicting organizations, conflicting domains, or conflicting locations, status MUST be 'ambiguous'.
   - NEVER force a match between competing plausible candidates.

3. Affiliation Changes & Transitions:
   - If evidence shows the person moved to a different company than the target context, record both affiliations (current with end_date=null, former with end_date specified) and set status to 'partially_resolved', rather than silently overwriting the target company.

4. Grounding:
   - Only reference candidate IDs that exist in the input candidate list. Do not invent candidate IDs.

Output format: You must respond ONLY with a JSON object adhering to the schema."""


class IdentityResolver:
    """Evaluates candidate entities using LLM structured reasoning and deterministic guardrails."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_SMALL
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(IdentityAssessmentOutput)

    def resolve(
        self,
        prospect: ProspectContext,
        candidates: List[DiscoveredEntityCandidate],
        sources: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
    ) -> IdentityResolutionResult:
        """Resolve target prospect and company identity against discovered candidates."""
        # Fast path: No candidates found
        if not candidates:
            return self._build_empty_resolution_result(prospect)

        # Build prompt and execute LLM reasoning
        prompt_content = self._build_prompt_content(prospect, candidates)
        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=RESOLVER_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=prompt_content))

        decoding = DecodeConfig(
            temperature=0.1,
            max_tokens=2048,
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=decoding,
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "identity_resolution", "prospect": prospect.full_name},
        )

        response = self.engine.generate(request)

        if not response.parsed_output or not isinstance(response.parsed_output, IdentityAssessmentOutput):
            raise ValueError(
                f"Failed to parse identity assessment output. Raw text: {response.text[:200]}"
            )

        raw_output: IdentityAssessmentOutput = response.parsed_output

        # Enforce deterministic post-LLM guardrails
        execution_id = response.metadata.get("execution_id") if hasattr(response, "metadata") and response.metadata else None
        return self._apply_guardrails(
            prospect,
            candidates,
            raw_output,
            model_execution_id=execution_id,
            model_name=getattr(response, "model_name", None),
        )

    def _build_empty_resolution_result(self, prospect: ProspectContext) -> IdentityResolutionResult:
        """Construct fallback result when no candidates were discovered."""
        empty_prov = OutputProvenance(
            origin=ProvenanceOrigin.DETERMINISTIC_RULE,
            component="identity_resolver",
            rule_id="empty_candidates_fallback",
            is_fallback=True,
        )
        prospect_entity = EntityRecord(
            entity_type=EntityType.PERSON,
            raw_name=prospect.full_name,
            resolution_status=ResolutionStatus.UNRESOLVED,
            confidence=ConfidenceLevel.UNRESOLVED,
            disambiguation_notes="No candidate entities discovered in research sources.",
            provenance=empty_prov,
        )
        company_entity = EntityRecord(
            entity_type=EntityType.ORGANIZATION,
            raw_name=prospect.company_name,
            resolution_status=ResolutionStatus.UNRESOLVED,
            confidence=ConfidenceLevel.UNRESOLVED,
            disambiguation_notes="No candidate entities discovered in research sources.",
            provenance=empty_prov,
        )
        return IdentityResolutionResult(
            target_prospect=prospect,
            resolution_status=ResolutionStatus.UNRESOLVED,
            confidence=ConfidenceLevel.UNRESOLVED,
            matched_candidate_ids=[],
            prospect_entity=prospect_entity,
            company_entity=company_entity,
            rationale="No candidate entities were discovered from research sources.",
            disambiguation_notes="Zero candidates available for resolution.",
        )

    def _build_prompt_content(
        self,
        prospect: ProspectContext,
        candidates: List[DiscoveredEntityCandidate],
    ) -> str:
        """Format target identity and candidates for LLM prompt."""
        lines = [
            "### Target Identity to Resolve",
            f"- Full Name: {prospect.full_name}",
            f"- Company Name: {prospect.company_name}",
            f"- Company Web Domain: {prospect.company_domain or 'Unknown'}",
            f"- Designation / Title: {prospect.designation or 'Unknown'}",
            f"- LinkedIn Profile: {prospect.linkedin_url or 'Unknown'}",
            f"- Location: {prospect.location or 'Unknown'}",
            "",
            f"### Discovered Candidates ({len(candidates)} total)",
        ]

        for idx, c in enumerate(candidates, start=1):
            lines.append(f"Candidate #{idx} [ID: {c.candidate_id}]:")
            lines.append(f"  - Entity Type: {c.entity_type.value}")
            lines.append(f"  - Raw Name: {c.raw_name}")
            lines.append(f"  - Normalized Name: {c.normalized_name}")
            if c.domains:
                lines.append(f"  - Domains: {', '.join(c.domains)}")
            if c.roles:
                lines.append(f"  - Roles: {', '.join(c.roles)}")
            if c.organizations:
                lines.append(f"  - Organizations: {', '.join(c.organizations)}")
            if c.linkedin_urls:
                lines.append(f"  - LinkedIn URLs: {', '.join(c.linkedin_urls)}")
            if c.locations:
                lines.append(f"  - Locations: {', '.join(c.locations)}")
            if c.observations:
                lines.append(f"  - Observations Count: {len(c.observations)}")
                for obs in c.observations[:2]:
                    if obs.snippet:
                        lines.append(f"    * Snippet: {obs.snippet[:150]}")
            lines.append("")

        lines.append(
            "Evaluate these candidates against the target. Determine resolution status, confidence, "
            "matched candidate IDs, canonical names, aliases, affiliations, and detailed rationale.\n\n"
            "Output MUST be a JSON object with this exact structure:\n"
            "{\n"
            '  "resolution_status": "resolved" | "partially_resolved" | "ambiguous" | "unresolved",\n'
            '  "confidence": "high" | "medium" | "low" | "unresolved",\n'
            '  "matched_candidate_ids": ["string"],\n'
            '  "prospect_canonical_name": "string",\n'
            '  "company_canonical_name": "string",\n'
            '  "rationale": "string",\n'
            '  "disambiguation_notes": "string"\n'
            "}"
        )
        return "\n".join(lines)

    def _apply_guardrails(
        self,
        prospect: ProspectContext,
        candidates: List[DiscoveredEntityCandidate],
        output: IdentityAssessmentOutput,
        model_execution_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> IdentityResolutionResult:
        """Enforce strict resolution guardrails deterministically over LLM output."""
        valid_candidates_map = {c.candidate_id: c for c in candidates}

        # 1. Reject hallucinated candidate IDs
        valid_matched_ids = [cid for cid in output.matched_candidate_ids if cid in valid_candidates_map]

        target_norm_name = normalize_person_name(prospect.full_name)
        target_norm_org = normalize_org_name(prospect.company_name)
        target_domain = clean_domain(prospect.company_domain or "")

        # Find all person candidates that closely match the target name
        name_matching_persons = [
            c for c in candidates
            if c.entity_type == EntityType.PERSON and c.normalized_name == target_norm_name
        ]

        status = output.resolution_status
        confidence = output.confidence
        disambiguation_notes = output.disambiguation_notes or ""
        conflicts = list(output.conflicts_or_ambiguities)

        # 2. Guardrail: Multiple conflicting candidates with matching names across different companies
        if len(name_matching_persons) >= 2:
            org_sets = [
                {normalize_org_name(o) for o in c.organizations if o} | {clean_domain(d) for d in c.domains if d}
                for c in name_matching_persons
            ]
            # If any two have non-empty but disjoint org sets, they conflict
            has_conflict = False
            for i in range(len(org_sets)):
                for j in range(i + 1, len(org_sets)):
                    if org_sets[i] and org_sets[j] and not (org_sets[i] & org_sets[j]):
                        has_conflict = True
                        break
                if has_conflict:
                    break

            if has_conflict:
                status = ResolutionStatus.AMBIGUOUS
                confidence = ConfidenceLevel.LOW
                conf_msg = (
                    f"Multiple conflicting candidates found with name '{prospect.full_name}' "
                    "at different organizations/domains."
                )
                conflicts.append(conf_msg)
                disambiguation_notes = (
                    f"{disambiguation_notes} [Guardrail Enforced: Multiple conflicting candidates discovered "
                    f"matching '{prospect.full_name}'. Preserving AMBIGUOUS status.]"
                ).strip()

        # 3. Guardrail: Name similarity alone must NEVER result in RESOLVED
        if status == ResolutionStatus.RESOLVED:
            matched_person_objs = [
                valid_candidates_map[cid]
                for cid in valid_matched_ids
                if valid_candidates_map[cid].entity_type == EntityType.PERSON
            ]

            corroborated = False
            for p_cand in matched_person_objs:
                # Check domain
                if target_domain and any(clean_domain(d) == target_domain for d in p_cand.domains):
                    corroborated = True
                    break
                # Check org
                if target_norm_org and any(normalize_org_name(o) == target_norm_org for o in p_cand.organizations):
                    corroborated = True
                    break
                # Check linkedin
                if prospect.linkedin_url and any(
                    prospect.linkedin_url.lower() in lu.lower() for lu in p_cand.linkedin_urls
                ):
                    corroborated = True
                    break
                # Check title if provided
                if prospect.designation and any(
                    prospect.designation.lower() in r.lower() for r in p_cand.roles
                ):
                    corroborated = True
                    break

            if not corroborated:
                status = ResolutionStatus.UNRESOLVED
                confidence = ConfidenceLevel.LOW
                disambiguation_notes = (
                    f"{disambiguation_notes} [Guardrail Enforced: Name similarity alone without corroborating "
                    "domain, affiliation, role, or verified profile cannot resolve identity.]"
                ).strip()

        # 4. Guardrail: Person found at different company
        if status == ResolutionStatus.RESOLVED:
            for cid in valid_matched_ids:
                cand = valid_candidates_map[cid]
                if cand.entity_type == EntityType.PERSON:
                    cand_orgs = {normalize_org_name(o) for o in cand.organizations if o}
                    # If candidate has organizations specified and none match target org
                    if cand_orgs and target_norm_org and target_norm_org not in cand_orgs:
                        status = ResolutionStatus.PARTIALLY_RESOLVED
                        confidence = ConfidenceLevel.MEDIUM
                        disambiguation_notes = (
                            f"{disambiguation_notes} [Guardrail Enforced: Candidate confirmed but affiliated with "
                            f"a different organization ({', '.join(cand.organizations)}). Marked PARTIALLY_RESOLVED.]"
                        ).strip()
                        break

        guardrail_triggered = (status != output.resolution_status or confidence != output.confidence)
        prov = OutputProvenance(
            origin=ProvenanceOrigin.MODEL if not guardrail_triggered else ProvenanceOrigin.DETERMINISTIC_RULE,
            component="identity_resolver",
            model_name=model_name,
            provider=self.provider_name,
            model_execution_id=model_execution_id,
            rule_id="identity_resolution_guardrails" if guardrail_triggered else None,
            is_fallback=guardrail_triggered,
        )

        # Construct final EntityRecord instances
        prospect_entity = EntityRecord(
            entity_type=EntityType.PERSON,
            raw_name=prospect.full_name,
            canonical_name=output.prospect_canonical_name or prospect.full_name,
            aliases=output.prospect_aliases,
            linkedin_url=prospect.linkedin_url,
            resolution_status=status,
            confidence=confidence,
            disambiguation_notes=disambiguation_notes or None,
            provenance=prov,
        )

        # Company entity resolution status matches or reflects org candidate
        company_status = status if status == ResolutionStatus.RESOLVED else ResolutionStatus.PARTIALLY_RESOLVED
        if not output.company_canonical_name and not target_domain:
            company_status = ResolutionStatus.UNRESOLVED

        company_entity = EntityRecord(
            entity_type=EntityType.ORGANIZATION,
            raw_name=prospect.company_name,
            canonical_name=output.company_canonical_name or prospect.company_name,
            aliases=output.company_aliases,
            domain=prospect.company_domain,
            resolution_status=company_status,
            confidence=confidence,
            disambiguation_notes=disambiguation_notes or None,
            provenance=prov,
        )

        return IdentityResolutionResult(
            target_prospect=prospect,
            resolution_status=status,
            confidence=confidence,
            matched_candidate_ids=valid_matched_ids,
            prospect_entity=prospect_entity,
            company_entity=company_entity,
            affiliations=output.affiliations,
            identifiers=output.identifiers,
            relationships=output.relationships,
            rationale=output.rationale,
            disambiguation_notes=disambiguation_notes or None,
            conflicts_or_ambiguities=conflicts,
        )
