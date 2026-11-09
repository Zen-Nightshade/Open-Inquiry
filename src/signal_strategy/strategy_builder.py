from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import GROQ_LARGE
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import ResearchPackage
from .models import (
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    StrategyResult,
)

logger = logging.getLogger(__name__)


class RawStrategyOutput(BaseModel):
    """Raw LLM strategy blueprint output."""

    model_config = ConfigDict(extra="ignore")

    prospect_role_relevance: str = Field(description="Why this specific individual cares about this event given their title/role")
    business_implications: str = Field(description="Downstream operational or commercial challenges/pressures for their company")
    capability_mapping: List[str] = Field(description="Exact product capabilities solving this problem (must come from product capabilities)")
    messaging_approach: str = Field(description="Strategic framing approach (e.g. 'challenge-led', 'expansion-scaling', 'peer-proof')")
    central_message: str = Field(description="One-sentence core proposition connecting the signal to product value")
    suggested_subject_line: str = Field(description="Recommended concise, conversational subject line")
    cta: str = Field(description="Specific, low-friction next action")
    tone_guidance: str = Field(description="Tone guidance matching recipient and relationship")
    constraint_checklist: List[str] = Field(default_factory=list, description="Explicit constraints to enforce during email drafting")


STRATEGY_BUILDER_SYSTEM_PROMPT = """You are a master B2B outreach strategist formulating an outreach blueprint.
Your task is to take the primary commercial signal and translate it into a compelling, relevant, and evidence-grounded strategy blueprint for email generation.

CRITICAL GUARDRAILS:
1. STRICT CAPABILITY BOUNDING:
   - Every capability in capability_mapping MUST be selected from the provided Product Capabilities list.
   - Do NOT invent features, SLAs, integrations, or capabilities not explicitly listed in the product context.

2. COMMERCIAL OPPORTUNITY LOGIC:
   - Central message must explain WHY this specific event creates an operational need that our product solves.
   - Do NOT send a generic template or a lazy "saw you raised money, want to chat?" email.
   - Ground the hook in the observed event, transition through the business implication, and introduce the solution.

3. CONVERSATIONAL & LOW-FRICTION CTA:
   - Propose an easy, conversational next step (e.g. 15-minute introductory call, sharing benchmark data, or brief response).
   - Follow any desired_cta in the outreach context.
"""


class StrategyBuilder:
    """Builds a verified, structured OutreachStrategy from ranked commercial signals."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
        min_composite_score: float = 0.35,
        min_product_alignment: float = 0.25,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.min_composite_score = min_composite_score
        self.min_product_alignment = min_product_alignment
        self.parser = PydanticParser(RawStrategyOutput)

    def build_strategy(
        self,
        ranked_signals: Sequence[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        research_package: Optional[ResearchPackage] = None,
    ) -> StrategyResult:
        """Evaluate viability and construct an immutable OutreachStrategy blueprint."""
        # Check 1: Any signals exist
        if not ranked_signals:
            logger.info("Strategy formulation aborted: zero ranked signals provided.")
            return StrategyResult(
                is_viable=False,
                rejection_reason="No candidate signals available from research.",
                unviable_reasons=["The research package contains no facts that could be synthesized into candidate signals."],
            )

        top_signal = ranked_signals[0]

        # Check 2: Minimum commercial viability thresholds
        unviable_reasons = []
        if top_signal.score.composite_score < self.min_composite_score:
            unviable_reasons.append(
                f"Top signal '{top_signal.signal.label}' composite score ({top_signal.score.composite_score:.2f}) "
                f"is below minimum viability threshold ({self.min_composite_score:.2f})."
            )
        if top_signal.score.product_alignment < self.min_product_alignment:
            unviable_reasons.append(
                f"Top signal '{top_signal.signal.label}' product alignment ({top_signal.score.product_alignment:.2f}) "
                f"is below minimum alignment threshold ({self.min_product_alignment:.2f})."
            )

        if unviable_reasons:
            logger.info(f"Strategy unviable: {unviable_reasons[0]}")
            return StrategyResult(
                is_viable=False,
                ranked_signals=list(ranked_signals),
                rejection_reason=unviable_reasons[0],
                unviable_reasons=unviable_reasons,
            )

        # Select secondary supporting signals (max 2)
        supporting_signals = [
            s for s in ranked_signals[1:3]
            if s.score.composite_score >= 0.30 and s.score.product_alignment >= 0.20
        ]

        # Formulate strategy via LLM
        strategy = self._synthesize_strategy_with_llm(
            primary_signal=top_signal,
            supporting_signals=supporting_signals,
            prospect=prospect,
            product=product,
            outreach=outreach,
        )

        if not strategy:
            # Deterministic fallback strategy
            strategy = self._deterministic_fallback_strategy(
                primary_signal=top_signal,
                supporting_signals=supporting_signals,
                prospect=prospect,
                product=product,
                outreach=outreach,
            )

        return StrategyResult(
            is_viable=True,
            strategy=strategy,
            ranked_signals=list(ranked_signals),
        )

    def _synthesize_strategy_with_llm(
        self,
        primary_signal: RankedSignal,
        supporting_signals: List[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
    ) -> Optional[OutreachStrategy]:
        """Invoke LLM to produce structured strategy blueprint with capability bounding."""
        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=STRATEGY_BUILDER_SYSTEM_PROMPT))
        prompt.add_message(
            Message(
                role=Role.USER,
                content=self._build_user_prompt(
                    primary_signal=primary_signal,
                    supporting_signals=supporting_signals,
                    prospect=prospect,
                    product=product,
                    outreach=outreach,
                ),
            )
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=DecodeConfig(temperature=0.2, max_tokens=2048),
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "strategy_building"},
        )

        try:
            response = self.engine.generate(request)
            raw = response.parsed_output
            if isinstance(raw, RawStrategyOutput):
                # Enforce capability bounding against product.capabilities
                bounded_capabilities = self._enforce_capability_bounding(raw.capability_mapping, product.capabilities)

                alignment = ProspectProductAlignment(
                    prospect_role_relevance=raw.prospect_role_relevance.strip(),
                    business_implications=raw.business_implications.strip(),
                    capability_mapping=bounded_capabilities,
                )

                # Assemble constraint checklist incorporating outreach constraints
                constraints = list(outreach.constraints)
                for c in raw.constraint_checklist:
                    if c not in constraints:
                        constraints.append(c)

                return OutreachStrategy(
                    primary_signal=primary_signal,
                    supporting_signals=supporting_signals,
                    alignment=alignment,
                    messaging_approach=raw.messaging_approach.strip(),
                    central_message=raw.central_message.strip(),
                    suggested_subject_line=raw.suggested_subject_line.strip(),
                    cta=raw.cta.strip() if raw.cta else (outreach.desired_cta or "Would you be open to a 15-minute call next week?"),
                    tone_guidance=raw.tone_guidance.strip() if raw.tone_guidance else f"Professional, direct {outreach.tone} tone",
                    constraint_checklist=constraints,
                )
        except Exception as e:
            logger.warning(f"Error during LLM strategy formulation: {e}. Executing deterministic fallback.")

        return None

    def _build_user_prompt(
        self,
        primary_signal: RankedSignal,
        supporting_signals: List[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
    ) -> str:
        lines = [
            "### Prospect Information",
            f"- Full Name: {prospect.full_name}",
            f"- Title / Role: {prospect.designation or 'Executive'}",
            f"- Company: {prospect.company_name}",
            f"- Domain: {prospect.company_domain or 'Unknown'}",
            "",
            "### Product Offering",
            f"- Product Name: {product.product_name}",
            f"- Description: {product.description}",
            "- Product Capabilities (MUST SELECT FROM THESE):",
        ]
        for cap in product.capabilities:
            lines.append(f"  * {cap}")

        lines.extend([
            f"- Value Proposition: {product.value_proposition}",
            f"- Differentiators: {', '.join(product.differentiators) if product.differentiators else 'None'}",
            f"- Boundaries / Limitations: {', '.join(product.limitations) if product.limitations else 'None'}",
            "",
            "### Outreach Guidelines",
            f"- Objective: {outreach.objective}",
            f"- Sender Identity: {outreach.sender_identity}",
            f"- Desired CTA: {outreach.desired_cta or 'Flexible conversational next step'}",
            f"- Tone: {outreach.tone}",
            f"- Relationship: {outreach.relationship}",
            f"- Existing Constraints: {', '.join(outreach.constraints) if outreach.constraints else 'Standard B2B professional limits'}",
            "",
            "### Primary Lead Signal",
            f"- Label: {primary_signal.signal.label}",
            f"- Category: {primary_signal.signal.fact_category.value}",
            f"- Description: {primary_signal.signal.description}",
            f"- Observed Event: {primary_signal.signal.observed_event or primary_signal.signal.description}",
            f"- Business Implication: {primary_signal.signal.business_implication or 'Unspecified'}",
            f"- Commercial Opportunity: {primary_signal.signal.commercial_opportunity or 'Unspecified'}",
            f"- Score: Composite={primary_signal.score.composite_score}, Product Alignment={primary_signal.score.product_alignment}",
            f"- Rationale: {primary_signal.rationale}",
            "",
        ])

        if supporting_signals:
            lines.append("### Secondary Context Signals:")
            for s in supporting_signals:
                lines.append(f"- {s.signal.label}: {s.signal.description}")
            lines.append("")

        lines.append("Formulate the outreach strategy blueprint connecting this primary signal to our product capabilities.")
        return "\n".join(lines)

    def _enforce_capability_bounding(
        self,
        candidate_capabilities: List[str],
        valid_capabilities: List[str],
    ) -> List[str]:
        """Strictly bound capabilities to the documented product capabilities."""
        if not valid_capabilities:
            return ["Core product solution"]

        bounded: List[str] = []
        valid_lower = {v.lower().strip(): v for v in valid_capabilities}

        for cand in candidate_capabilities:
            cand_clean = cand.lower().strip()
            # Exact match
            if cand_clean in valid_lower:
                matched = valid_lower[cand_clean]
                if matched not in bounded:
                    bounded.append(matched)
                continue

            # Substring match
            for v_low, original in valid_lower.items():
                if v_low in cand_clean or cand_clean in v_low:
                    if original not in bounded:
                        bounded.append(original)
                    break

        if not bounded:
            logger.info("No LLM-generated capabilities matched product capabilities. Falling back to primary documented capability.")
            bounded.append(valid_capabilities[0])

        return bounded

    def _deterministic_fallback_strategy(
        self,
        primary_signal: RankedSignal,
        supporting_signals: List[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
    ) -> OutreachStrategy:
        """Deterministic strategy construction when LLM synthesis is unparseable or fails."""
        sig = primary_signal.signal
        cap = product.capabilities[0] if product.capabilities else "Enterprise solution"

        alignment = ProspectProductAlignment(
            prospect_role_relevance=f"As {prospect.designation or 'leader'} at {prospect.company_name}, this event directly impacts operational priorities.",
            business_implications=sig.business_implication or f"Expansion and operational requirements driven by {sig.label}.",
            capability_mapping=[cap],
        )

        subject = f"{prospect.company_name} — {sig.label}"
        central_msg = f"{product.product_name} helps teams at {prospect.company_name} address {sig.label.lower()} through {cap}."
        cta = outreach.desired_cta or "Are you open to a brief 15-minute conversation next week?"

        constraints = list(outreach.constraints)
        if "Keep email concise (under 100 words)" not in constraints:
            constraints.append("Keep email concise (under 100 words)")
        if "Only reference verified research facts" not in constraints:
            constraints.append("Only reference verified research facts")

        return OutreachStrategy(
            primary_signal=primary_signal,
            supporting_signals=supporting_signals,
            alignment=alignment,
            messaging_approach="event-hook",
            central_message=central_msg,
            suggested_subject_line=subject,
            cta=cta,
            tone_guidance=f"Professional, direct {outreach.tone} tone",
            constraint_checklist=constraints,
        )
