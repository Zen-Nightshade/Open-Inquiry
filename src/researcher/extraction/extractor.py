from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.config.settings import GROQ_SMALL
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.extraction.excerpt_validator import ExcerptValidator
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EvidenceRecord,
    FactCategory,
    SourceRecord,
)

logger = logging.getLogger(__name__)


def parse_iso_date(date_str: Optional[str]) -> Optional[datetime.date]:
    """Parse various date strings (YYYY-MM-DD, YYYY-MM, YYYY) into datetime.date."""
    if not date_str:
        return None
    cleaned = date_str.strip()
    if not cleaned:
        return None
    # YYYY-MM-DD
    try:
        return datetime.date.fromisoformat(cleaned[:10])
    except ValueError:
        pass
    # YYYY
    if len(cleaned) == 4 and cleaned.isdigit():
        try:
            return datetime.date(int(cleaned), 1, 1)
        except ValueError:
            pass
    return None


class RawExtractedEvidence(BaseModel):
    """Evidence excerpt extracted by LLM before deterministic validation."""

    model_config = ConfigDict(populate_by_name=True)

    verbatim_excerpt: str = Field(description="Exact word-for-word quote from the source text")
    event_date: Optional[str] = Field(default=None, description="Date of event mentioned in the excerpt")
    support_type: str = Field(default="supporting", description="'supporting' or 'contradicting'")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)

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
            return ConfidenceLevel.UNRESOLVED
        if isinstance(v, str):
            clean = v.strip().lower()
            for level in ConfidenceLevel:
                if level.value == clean:
                    return level
        return v


class RawExtractedFact(BaseModel):
    """Raw fact extracted by LLM before deterministic validation."""

    model_config = ConfigDict(populate_by_name=True)

    claim_text: str = Field(
        default="",
        validation_alias=AliasChoices("claim_text", "claim", "fact", "statement", "text", "description"),
        description="Atomic, unambiguous factual statement",
    )
    fact_category: FactCategory = Field(default=FactCategory.OTHER)
    subject_entity_name: str = Field(
        default="Target Entity",
        validation_alias=AliasChoices("subject_entity_name", "subject", "entity_name", "entity", "subject_entity"),
        description="The primary person or organization this fact describes",
    )
    predicate: Optional[str] = Field(default=None, description="Relation or action (e.g. 'raised', 'launched', 'hired')")
    object_value: Optional[str] = Field(default=None, description="Object or complement (e.g. '$50M Series B', 'Jane Doe')")
    temporal_scope: Optional[str] = Field(default=None, description="Temporal validity or timeframe")
    primary_event_date: Optional[str] = Field(default=None, description="Consensus event date if known")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    evidence: List[RawExtractedEvidence] = Field(
        default_factory=list,
        description="Verbatim excerpts directly supporting this assertion",
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_fact_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. If verbatim_excerpt provided at top-level, lift into evidence list
            if "verbatim_excerpt" in data and not data.get("evidence"):
                data["evidence"] = [
                    {
                        "verbatim_excerpt": data["verbatim_excerpt"],
                        "event_date": data.get("event_date") or data.get("primary_event_date"),
                        "confidence": data.get("confidence", "medium"),
                    }
                ]
            # 2. If claim_text is empty, use verbatim_excerpt or description as claim_text
            if not data.get("claim_text") and not data.get("claim") and not data.get("fact"):
                fallback = data.get("verbatim_excerpt") or data.get("description") or "Factual assertion"
                data["claim_text"] = fallback
        return data

    @field_validator("fact_category", mode="before")
    @classmethod
    def normalize_category(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean = v.strip().lower()
            for cat in FactCategory:
                if cat.value == clean:
                    return cat
        return FactCategory.OTHER

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
            return ConfidenceLevel.UNRESOLVED
        if isinstance(v, str):
            clean = v.strip().lower()
            for level in ConfidenceLevel:
                if level.value == clean:
                    return level
        return v


class FactExtractionOutput(BaseModel):
    """Container schema for structured LLM fact extraction."""

    model_config = ConfigDict(populate_by_name=True)

    facts: List[RawExtractedFact] = Field(
        default_factory=list,
        description="Extracted atomic factual claims grounded in the text",
    )


EXTRACTOR_SYSTEM_PROMPT = """You are an expert factual assertion extraction engine for B2B intelligence research.
Your task is to extract atomic, attributable, temporally meaningful facts about organizations and individuals from the provided document.

CRITICAL RULES:
1. ATOMIC FACTS ONLY:
   - Extract single, indivisible factual claims.
   - Do NOT combine multiple events into one compound fact.
   - For example: Split "Company raised $50M and opened an office in London" into TWO separate facts: one funding fact, one expansion fact.
2. VERBATIM EXCERPT MANDATE:
   - Every fact MUST include at least one 'verbatim_excerpt'.
   - The excerpt MUST be an exact, word-for-word quote from the text.
   - Do NOT paraphrase, summarize, or alter excerpts. Every excerpt will be verified by a deterministic text matcher; facts with non-verbatim excerpts are discarded immediately.
3. GROUNDING & NO SPECULATION:
   - Do NOT extract unstated assumptions, marketing hype, or unsubstantiated implications.
   - Only extract claims explicitly affirmed in the text.
4. EXISTING CATEGORIES:
   - Assign 'fact_category' from: 'funding', 'hiring', 'product_launch', 'strategy', 'challenge', 'achievement', 'organization_change', 'other'.
5. TEMPORAL ACCURACY:
   - Extract the event date or temporal scope if stated (e.g. '2026-05', 'Q3 2025', '2024'). Do not guess.

Output format: You must respond ONLY with a JSON object adhering to the schema:
{
  "facts": [
    {
      "claim_text": "Company raised $50M Series B funding led by Vertex Ventures.",
      "fact_category": "funding",
      "subject_entity_name": "Company Name",
      "evidence": [
        {
          "verbatim_excerpt": "Company raised $50M Series B funding led by Vertex Ventures",
          "event_date": "2026-01"
        }
      ]
    }
  ]
}"""


class FactExtractor:
    """Extracts atomic, verifiable factual claims and validates their verbatim excerpts."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        validator: Optional[ExcerptValidator] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_SMALL
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.validator = validator or ExcerptValidator()
        self.parser = PydanticParser(FactExtractionOutput)

    def extract(
        self,
        source: Any,
        entity_context: Optional[Dict[str, Any]] = None,
        known_entities: Optional[List[EntityRecord]] = None,
    ) -> List[CanonicalFact]:
        """Extract atomic facts from source record or snapshot, discarding any with unverified excerpts."""
        if isinstance(source, dict):
            source_id = str(source.get("source_id", "") or "")
            raw_text = str(source.get("raw_text", "") or "")
            title = str(source.get("title", "") or "")
            credibility_score = float(source.get("credibility_score", 0.7) or 0.7)
            raw_fetch_date = source.get("fetch_date")
        else:
            source_id = str(getattr(source, "source_id", "") or "")
            raw_text = str(getattr(source, "raw_text", "") or "")
            title = str(getattr(source, "title", "") or "")
            credibility_score = float(getattr(source, "credibility_score", 0.7) or 0.7)
            raw_fetch_date = getattr(source, "fetch_date", None)

        fetch_date = parse_iso_date(raw_fetch_date) if isinstance(raw_fetch_date, str) else raw_fetch_date


        if not raw_text or len(raw_text.strip()) < 20:
            logger.info("Source %s raw_text is empty or too short. Skipping extraction.", source_id)
            return []

        # 1. Build prompt
        prompt_content = self._build_prompt_content(title=title, text=raw_text, entity_context=entity_context)
        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=EXTRACTOR_SYSTEM_PROMPT))
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
            metadata={"task": "fact_extraction", "source_id": source_id},
        )

        # 2. Call GeneratorEngine outside any database transaction
        try:
            response = self.engine.generate(request)
        except Exception as exc:
            logger.warning("Fact extraction LLM generation/parsing failed for source %s: %s", source_id, exc)
            return []

        if not response.parsed_output or not isinstance(response.parsed_output, FactExtractionOutput):
            logger.warning("Failed to parse structured output from fact extraction for source %s", source_id)
            return []

        raw_facts: List[RawExtractedFact] = response.parsed_output.facts
        logger.info("LLM generated %d raw facts for source %s. Validating excerpts...", len(raw_facts), source_id)

        # 3. Deterministic Excerpt Validation & CanonicalFact Construction
        validated_facts: List[CanonicalFact] = []
        today = datetime.date.today()

        for raw_fact in raw_facts:
            valid_evidence_records: List[EvidenceRecord] = []

            for raw_ev in raw_fact.evidence:
                excerpt = raw_ev.verbatim_excerpt
                # Deterministic whitespace-normalized substring check
                if self.validator.is_valid_excerpt(excerpt, raw_text):
                    ev_date = parse_iso_date(raw_ev.event_date)
                    valid_evidence_records.append(
                        EvidenceRecord(
                            source_id=source_id,
                            verbatim_excerpt=excerpt,  # Preserve exact original excerpt
                            event_date=ev_date,
                            extracted_date=fetch_date,
                            confidence=raw_ev.confidence,
                            credibility_score=credibility_score,
                        )
                    )
                else:
                    logger.info(
                        "Discarding invalid excerpt for claim '%s': %r not found verbatim in source %s",
                        raw_fact.claim_text,
                        excerpt[:60],
                        source_id,
                    )

            # Rule: Discard facts with zero valid supporting evidence records
            if not valid_evidence_records:
                logger.info(
                    "Discarding fact '%s' because no valid supporting verbatim excerpts were confirmed.",
                    raw_fact.claim_text,
                )
                continue

            # 4. Resolve entity association
            entity_ids = self._resolve_entity_ids(
                subject_name=raw_fact.subject_entity_name,
                entity_context=entity_context,
                known_entities=known_entities,
            )

            primary_date = parse_iso_date(raw_fact.primary_event_date) or valid_evidence_records[0].event_date

            canonical_fact = CanonicalFact(
                fact_id=str(uuid.uuid4()),
                claim_text=raw_fact.claim_text.strip(),
                fact_category=raw_fact.fact_category,
                entity_ids=entity_ids,
                evidence_records=valid_evidence_records,
                primary_event_date=primary_date,
                composite_confidence=raw_fact.confidence,
            )
            validated_facts.append(canonical_fact)

        logger.info(
            "Extraction complete for source %s: %d / %d facts passed excerpt validation.",
            source_id,
            len(validated_facts),
            len(raw_facts),
        )
        return validated_facts

    def _build_prompt_content(
        self,
        title: str,
        text: str,
        entity_context: Optional[Dict[str, Any]],
    ) -> str:
        lines = []
        if entity_context:
            lines.append("### Target Entities in Context")
            for k, v in entity_context.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("### Document Content to Extract From")
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Content:\n{text[:6000]}")
        lines.append("")
        lines.append(
            "Extract atomic, attributable facts about target entities from the document above. "
            "Ensure EVERY fact contains a verbatim_excerpt quoted exactly from the content."
        )
        return "\n".join(lines)

    def _resolve_entity_ids(
        self,
        subject_name: str,
        entity_context: Optional[Dict[str, Any]],
        known_entities: Optional[List[EntityRecord]],
    ) -> List[str]:
        """Map subject name to entity IDs, preserving uncertainty if ambiguous."""
        if not subject_name:
            return []

        norm_subj = subject_name.strip().lower()

        if known_entities:
            # 1. Exact match pass: exact name or canonical name
            exact_matches = [
                ent.entity_id
                for ent in known_entities
                if norm_subj == ent.raw_name.lower().strip()
                or (ent.canonical_name and norm_subj == ent.canonical_name.lower().strip())
            ]
            if exact_matches:
                return list(dict.fromkeys(exact_matches))

            # 2. Word boundary containment pass (only if no exact match found)
            matched_ids = []
            for ent in known_entities:
                raw_l = ent.raw_name.lower().strip()
                can_l = ent.canonical_name.lower().strip() if ent.canonical_name else ""
                if (f" {raw_l} " in f" {norm_subj} ") or (can_l and f" {can_l} " in f" {norm_subj} "):
                    matched_ids.append(ent.entity_id)

            if matched_ids:
                return list(dict.fromkeys(matched_ids))

        if entity_context:
            # 1. Exact match in entity_context
            exact_ctx = [
                eid
                for name, eid in entity_context.items()
                if norm_subj == name.lower().strip() and isinstance(eid, str)
            ]
            if exact_ctx:
                return list(dict.fromkeys(exact_ctx))

            # 2. Word boundary containment in entity_context
            matched_ctx = []
            for name, eid in entity_context.items():
                name_l = name.lower().strip()
                if (f" {name_l} " in f" {norm_subj} ") and isinstance(eid, str):
                    matched_ctx.append(eid)
            if matched_ctx:
                return list(dict.fromkeys(matched_ctx))

        return []
