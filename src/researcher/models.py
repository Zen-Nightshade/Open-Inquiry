from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"       # Evidence directly entails claim
    INFERRED = "inferred"         # Reasonable inference from factual evidence
    UNSUPPORTED = "unsupported"   # No supporting evidence found in research
    CONTRADICTED = "contradicted" # Contradicted by research evidence



class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


class SourceType(str, Enum):
    NEWS = "news"
    COMPANY_WEBSITE = "company_website"
    PRESS_RELEASE = "press_release"
    LINKEDIN = "linkedin"
    JOB_BOARD = "job_board"
    SEC_FILING = "sec_filing"
    BLOG = "blog"
    OTHER = "other"


class FactCategory(str, Enum):
    FUNDING = "funding"
    HIRING = "hiring"
    PRODUCT_LAUNCH = "product_launch"
    STRATEGY = "strategy"
    CHALLENGE = "challenge"
    ACHIEVEMENT = "achievement"
    ORGANIZATION_CHANGE = "organization_change"
    OTHER = "other"


class EntityRecord(BaseModel):
    """Identity record for a person or organization.
    
    Supports confirmed resolution as well as partial or unresolved states.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    raw_name: str = Field(description="Name as originally provided or found")
    canonical_name: Optional[str] = Field(default=None, description="Disambiguated official name if resolved")
    aliases: List[str] = Field(default_factory=list, description="Known alternative names")
    domain: Optional[str] = Field(default=None, description="Company web domain if resolved")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL if resolved")
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.UNRESOLVED,
        description="Whether this entity is resolved, ambiguous, or unresolved",
    )
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.UNRESOLVED)
    disambiguation_notes: Optional[str] = Field(default=None, description="Notes on resolution confidence or ambiguity")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")


class SourceRecord(BaseModel):
    """Provenance and credibility metadata for a fetched web page or document."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str = Field(description="Canonical normalized URL used for deduplication")
    original_url: Optional[str] = Field(default=None, description="Original requested URL from search provider")
    final_url: Optional[str] = Field(default=None, description="Actual network URL reached after all HTTP redirects")
    html_canonical_url: Optional[str] = Field(default=None, description="Canonical URL declared in HTML link rel=canonical if present")
    domain: str = Field(description="Domain name of the canonical resource")
    title: Optional[str] = None
    raw_text: str = Field(description="Parsed and cleaned text content")
    source_type: SourceType = Field(default=SourceType.OTHER, description="Classification of the source")
    credibility_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Heuristic baseline score for channel authority (0.0 to 1.0). Does not represent claim truth or evidence strength.",
    )
    credibility_rationale: Optional[str] = Field(default=None, description="Heuristic justification for credibility score")
    fetch_date: date = Field(description="Date source was fetched")
    publication_date: Optional[date] = Field(default=None, description="Date content was published")
    publication_date_source: Optional[str] = Field(default=None, description="Origin or method used to extract publication date")
    entity_ids: List[str] = Field(default_factory=list, description="Associated resolved or candidate entities")


class EvidenceRecord(BaseModel):
    """Independent piece of source evidence supporting a canonical fact.
    
    Multiple evidence records can support a single canonical fact. Deduplication
    must consolidate evidence rather than discarding it.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(description="Foreign key to SourceRecord.source_id")
    verbatim_excerpt: str = Field(min_length=1, description="Exact quote from source supporting the claim")
    event_date: Optional[date] = Field(default=None, description="Date the event occurred according to this source")
    extracted_date: Optional[date] = Field(default=None, description="Date this evidence was extracted if known")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    credibility_score: float = Field(default=0.7, ge=0.0, le=1.0)


class CanonicalFact(BaseModel):
    """A deduplicated, normalized factual claim supported by one or more independent evidence records."""

    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = Field(description="Normalized factual statement")
    fact_category: FactCategory = Field(default=FactCategory.OTHER)
    entity_ids: List[str] = Field(default_factory=list, description="Associated entity IDs")
    evidence_records: List[EvidenceRecord] = Field(
        min_length=1,
        description="Independent evidence instances supporting this fact across sources",
    )
    primary_event_date: Optional[date] = Field(default=None, description="Consensus event date")
    composite_confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")
    molecular_facts: Tuple[Any, ...] = Field(
        default=(),
        description="Atomic molecular propositions representing this claim; empty if un-decomposed",
    )

    @property
    def source_ids(self) -> List[str]:
        """All unique source IDs that corroborate this fact."""
        return list(dict.fromkeys(ev.source_id for ev in self.evidence_records))


# Alias for backwards compatibility with earlier blueprint references
ExtractedFact = CanonicalFact


class SufficiencyResult(BaseModel):
    """Evaluation of research sufficiency and stopping decision."""

    model_config = ConfigDict(frozen=True)

    is_sufficient: bool
    identity_confirmed: bool
    has_commercial_signal: bool
    coverage_gaps: Tuple[str, ...] = Field(default=())
    stop_reason: str = Field(default="sufficient", description="Reason for stopping or continuing")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Origin and producing component metadata")


class ResearchPackage(BaseModel):
    """Complete, immutable output contract of the research stage.
    
    Supports incomplete research and unresolved prospect/company identities.
    """

    model_config = ConfigDict(frozen=True)

    prospect_entity: Optional[EntityRecord] = Field(default=None, description="Resolved target prospect")
    company_entity: Optional[EntityRecord] = Field(default=None, description="Resolved company organization")
    sources: Tuple[SourceRecord, ...] = Field(default=())
    facts: Tuple[CanonicalFact, ...] = Field(default=())
    search_queries_used: Tuple[str, ...] = Field(default=())
    is_sufficient: bool = Field(default=False, description="Whether research threshold was met for outreach")
    is_complete: bool = Field(default=False, description="Whether all search and gap iterations finished normally")
    missing_information: Tuple[str, ...] = Field(default=(), description="Identified uncollected information")
    sufficiency_notes: Optional[str] = Field(default=None)


@dataclass
class ResearchAccumulator:
    """Mutable accumulator used strictly within the researcher multi-turn loop.
    
    Collects raw extractions, merges facts, and freezes into an immutable ResearchPackage.
    """

    sources: List[SourceRecord] = field(default_factory=list)
    facts: List[CanonicalFact] = field(default_factory=list)
    search_queries_used: List[str] = field(default_factory=list)
    seen_urls: Set[str] = field(default_factory=set)
    prospect_entity: Optional[EntityRecord] = None
    company_entity: Optional[EntityRecord] = None

    def add_source(self, source: SourceRecord) -> None:
        """Add a fetched source, maintaining seen URLs."""
        self.sources.append(source)
        self.seen_urls.add(source.url)

    def merge_or_add_fact(
        self,
        new_fact: CanonicalFact,
        match_fact_id: Optional[str] = None,
        is_independent: bool = True,
    ) -> None:
        """Add a new fact or merge evidence into an existing fact without losing source evidence."""
        if match_fact_id is not None:
            for idx, existing in enumerate(self.facts):
                if existing.fact_id == match_fact_id:
                    # Deduplicate evidence records by stable evidence identity or source+excerpt
                    existing_ev_ids = {ev.evidence_id for ev in existing.evidence_records}
                    existing_ev_tuples = {(ev.source_id, ev.verbatim_excerpt) for ev in existing.evidence_records}

                    merged_evidence = list(existing.evidence_records)
                    for new_ev in new_fact.evidence_records:
                        if (
                            new_ev.evidence_id not in existing_ev_ids
                            and (new_ev.source_id, new_ev.verbatim_excerpt) not in existing_ev_tuples
                        ):
                            merged_evidence.append(new_ev)
                            existing_ev_ids.add(new_ev.evidence_id)
                            existing_ev_tuples.add((new_ev.source_id, new_ev.verbatim_excerpt))

                    merged_entity_ids = list(dict.fromkeys(existing.entity_ids + new_fact.entity_ids))
                    unique_sources = list(dict.fromkeys(ev.source_id for ev in merged_evidence))

                    # Confidence calibration:
                    # Keep source credibility and evidence independence as separate dimensions.
                    # Multiple low-quality sources cannot automatically establish HIGH confidence.
                    # Upgrading to HIGH requires independent corroboration AND at least one substantial evidence record (>= MEDIUM).
                    has_substantial_evidence = any(
                        ev.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
                        for ev in merged_evidence
                    )

                    if is_independent and len(unique_sources) > 1:
                        if has_substantial_evidence:
                            composite_conf = ConfidenceLevel.HIGH
                        else:
                            composite_conf = ConfidenceLevel.MEDIUM
                    else:
                        composite_conf = existing.composite_confidence

                    updated = CanonicalFact(
                        fact_id=existing.fact_id,
                        claim_text=existing.claim_text,
                        fact_category=existing.fact_category,
                        entity_ids=merged_entity_ids,
                        evidence_records=merged_evidence,
                        primary_event_date=existing.primary_event_date or new_fact.primary_event_date,
                        composite_confidence=composite_conf,
                        molecular_facts=existing.molecular_facts or new_fact.molecular_facts,
                    )
                    self.facts[idx] = updated
                    return

        self.facts.append(new_fact)

    def to_package(
        self,
        is_sufficient: bool,
        is_complete: bool = False,
        missing_information: Optional[Sequence[str]] = None,
        sufficiency_notes: Optional[str] = None,
    ) -> ResearchPackage:
        """Freeze mutable accumulator into immutable ResearchPackage contract."""
        return ResearchPackage(
            prospect_entity=self.prospect_entity,
            company_entity=self.company_entity,
            sources=tuple(self.sources),
            facts=tuple(self.facts),
            search_queries_used=tuple(self.search_queries_used),
            is_sufficient=is_sufficient,
            is_complete=is_complete,
            missing_information=tuple(missing_information or ()),
            sufficiency_notes=sufficiency_notes,
        )
