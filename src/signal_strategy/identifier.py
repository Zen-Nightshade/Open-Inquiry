from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.config.settings import GROQ_LARGE
from src.context.models import ProductContext, ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import CanonicalFact, FactCategory
from .models import CandidateSignal

logger = logging.getLogger(__name__)


class RawCandidateSignal(BaseModel):
    """Raw structured candidate signal output produced by LLM."""

    model_config = ConfigDict(extra="ignore")

    label: str = Field(description="Short identifier, e.g. 'Series B Expansion'")
    description: str = Field(description="Summary of the commercial event")
    supporting_fact_ids: List[str] = Field(description="UUIDs of facts directly supporting this signal")
    fact_category: str = Field(default="other", description="Category of event")
    observed_event: Optional[str] = Field(default=None, description="The concrete factual event that took place")
    business_implication: Optional[str] = Field(default=None, description="Downstream operational/commercial impact for the company")
    commercial_opportunity: Optional[str] = Field(default=None, description="Why this creates an opening for the product")
    is_inferred: bool = Field(default=False, description="Whether the signal is an inference rather than direct statement")


class RawSignalIdentificationOutput(BaseModel):
    """LLM generation container for identified candidate signals."""

    model_config = ConfigDict(extra="ignore")

    candidate_signals: List[RawCandidateSignal] = Field(default_factory=list)
    reasoning: Optional[str] = Field(default=None, description="Overview reasoning for selected signals")


SIGNAL_IDENTIFIER_SYSTEM_PROMPT = """You are a senior B2B commercial intelligence analyst.
Your task is to analyze verified research facts about a target company and prospect, and synthesize commercially meaningful candidate signals.

CRITICAL RULES FOR SIGNAL IDENTIFICATION:
1. STRICT FACTUAL GROUNDING:
   - Every candidate signal MUST cite one or more valid fact IDs from the provided facts list.
   - Do NOT invent facts, numbers, dates, or events not present in the research facts.
   - If a claim is an interpretation or inference, set is_inferred = true.

2. SEPARATION OF CONCEPTS:
   - Observed Event: What concrete activity happened (e.g. raised Series B, launched cloud platform, hired VP).
   - Business Implication: The downstream operational/business consequence for their organization (e.g. scaling compute overhead, onboarding pressure, compliance burden).
   - Commercial Opportunity: Why this specific consequence creates an opening for our product's documented capabilities.

3. RELEVANCE TO PRODUCT:
   - Focus on signals that create a logical business problem our product solves.
   - Do NOT invent product capabilities outside what is provided.
"""


class SignalIdentifier:
    """Identifies and synthesizes commercial signals from verified research facts."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(RawSignalIdentificationOutput)

    def identify_signals(
        self,
        facts: Sequence[CanonicalFact],
        prospect: ProspectContext,
        product: ProductContext,
    ) -> List[CandidateSignal]:
        """Extract and validate candidate signals from research facts."""
        if not facts:
            logger.info("No research facts provided; returning zero candidate signals.")
            return []

        facts_by_id: Dict[str, CanonicalFact] = {f.fact_id: f for f in facts}

        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=SIGNAL_IDENTIFIER_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=self._build_user_prompt(facts, prospect, product)))

        decoding = DecodeConfig(
            temperature=0.2,
            max_tokens=2048,
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=decoding,
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "signal_identification"},
        )

        try:
            response = self.engine.generate(request)
            raw_output = response.parsed_output
            exec_id = response.metadata.get("execution_id") if hasattr(response, "metadata") and response.metadata else None
            model_name = getattr(response, "model_name", None)
            if isinstance(raw_output, RawSignalIdentificationOutput):
                validated_signals = self._validate_and_ground_signals(
                    raw_output.candidate_signals,
                    facts_by_id,
                    model_execution_id=exec_id,
                    model_name=model_name,
                )
                if validated_signals:
                    return validated_signals
                logger.warning("LLM produced candidate signals but none passed grounding validation. Returning 0 signals.")
                return []
        except Exception as e:
            logger.warning(f"Error during LLM signal identification: {e}. Returning 0 signals.")
            return []

        return []

    def _build_user_prompt(
        self,
        facts: Sequence[CanonicalFact],
        prospect: ProspectContext,
        product: ProductContext,
    ) -> str:
        lines = [
            "### Target Prospect & Company",
            f"- Prospect Name: {prospect.full_name}",
            f"- Company: {prospect.company_name}",
            f"- Title/Role: {prospect.designation or 'Unknown'}",
            f"- Domain: {prospect.company_domain or 'Unknown'}",
            "",
            "### Product Offering",
            f"- Product Name: {product.product_name}",
            f"- Description: {product.description}",
            "- Core Capabilities:",
        ]
        for cap in product.capabilities:
            lines.append(f"  * {cap}")
        lines.extend([
            f"- Value Proposition: {product.value_proposition}",
            "",
            "### Verified Research Facts",
        ])

        for f in facts:
            event_date_str = f.primary_event_date.isoformat() if f.primary_event_date else "Unknown date"
            lines.append(
                f"- [Fact ID: {f.fact_id}] Category: {f.fact_category.value} | "
                f"Confidence: {f.composite_confidence.value} | Date: {event_date_str}\n"
                f"  Claim: \"{f.claim_text}\""
            )

        lines.extend([
            "",
            "Synthesize candidate commercial signals that connect these verified facts to the product capabilities.",
        ])
        return "\n".join(lines)

    def _validate_and_ground_signals(
        self,
        raw_signals: List[RawCandidateSignal],
        facts_by_id: Dict[str, CanonicalFact],
        model_execution_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[CandidateSignal]:
        """Validate fact citations, discard hallucinated fact IDs, and construct grounded CandidateSignals."""
        grounded_signals: List[CandidateSignal] = []

        for raw in raw_signals:
            # Filter fact IDs to only those that genuinely exist in facts_by_id
            valid_fact_ids = [fid for fid in raw.supporting_fact_ids if fid in facts_by_id]
            if not valid_fact_ids:
                logger.warning(
                    f"Discarding ungrounded signal '{raw.label}': cited fact IDs {raw.supporting_fact_ids} "
                    f"do not exist in research package facts."
                )
                continue

            # Resolve category
            category = FactCategory.OTHER
            for cat in FactCategory:
                if cat.value == raw.fact_category.lower().strip():
                    category = cat
                    break
            if category == FactCategory.OTHER and valid_fact_ids:
                category = facts_by_id[valid_fact_ids[0]].fact_category

            # Gather associated entity IDs and dates from supporting facts
            associated_entity_ids: List[str] = []
            event_dates = []
            for fid in valid_fact_ids:
                fact = facts_by_id[fid]
                for eid in fact.entity_ids:
                    if eid not in associated_entity_ids:
                        associated_entity_ids.append(eid)
                if fact.primary_event_date:
                    event_dates.append(fact.primary_event_date)

            latest_event_date = max(event_dates) if event_dates else None

            prov = OutputProvenance(
                origin=ProvenanceOrigin.MODEL,
                component="signal_identifier",
                model_name=model_name,
                provider=self.provider_name,
                model_execution_id=model_execution_id,
            )

            signal = CandidateSignal(
                label=raw.label.strip(),
                description=raw.description.strip(),
                supporting_fact_ids=valid_fact_ids,
                fact_category=category,
                associated_entity_ids=associated_entity_ids,
                observed_event=raw.observed_event.strip() if raw.observed_event else raw.description.strip(),
                business_implication=raw.business_implication.strip() if raw.business_implication else None,
                commercial_opportunity=raw.commercial_opportunity.strip() if raw.commercial_opportunity else None,
                event_date=latest_event_date,
                is_inferred=raw.is_inferred,
                provenance=prov,
            )
            grounded_signals.append(signal)

        return grounded_signals
