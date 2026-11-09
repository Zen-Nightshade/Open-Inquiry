from __future__ import annotations

import logging
import uuid
from typing import List, Optional, Set

from src.config.settings import GROQ_LARGE
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from .models import ResearchPlan

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a senior B2B research intelligence planner.
Your goal is to construct a targeted, grounded Research Plan for investigating a prospective customer and their organization.

Critical Guidelines:
1. Objectivity: Formulate objective, neutral discovery goals. Never prematurely bias research toward selling or pitching a specific product.
2. Separation of Concerns: Clearly distinguish between the individual prospect (identity, career history, public statements) and the organization (business initiatives, challenges, growth, funding).
3. Required Objectives:
   - Always include an 'identity_resolution' objective to corroborate the prospect's current position and organization affiliation.
   - Always include a 'neutral_discovery' objective investigating recent company commercial milestones, strategy, or operational challenges.
   - An optional 'product_oriented' objective may only explore whether specific business problems exist in the prospect's domain, without assuming product fit.
4. Criticality: Mark core objectives (identity resolution, key organizational milestones) as is_critical=true. Mark purely secondary or exploratory objectives as is_critical=false.
5. Budgeting: Produce between 2 and 4 high-value objectives. Total query budget must be between 6 and 15 queries.

Output format: You must respond ONLY with a valid JSON object matching the ResearchPlan schema:
{
  "total_query_budget": 10,
  "max_iterations": 3,
  "objectives": [
    {
      "type": "identity_resolution",
      "target_entity": "prospect",
      "description": "Corroborate the prospect's current role, title, and organization affiliation.",
      "priority": 1,
      "is_critical": true,
      "max_queries": 3
    },
    {
      "type": "neutral_discovery",
      "target_entity": "company",
      "description": "Investigate company core offerings, strategic initiatives, and commercial milestones.",
      "priority": 2,
      "is_critical": true,
      "max_queries": 3
    }
  ]
}"""


class ResearchPlanner:
    """Plans research objectives and allocates query budgets prior to search execution."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(ResearchPlan)

    def plan(
        self,
        prospect: ProspectContext,
        outreach: OutreachContext,
        product: Optional[ProductContext] = None,
        session_id: Optional[str] = None,
    ) -> ResearchPlan:
        """Generate a validated ResearchPlan based on prospect, outreach, and product context."""
        user_content = self._build_prompt_content(prospect, outreach, product)

        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=PLANNER_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=user_content))

        # Allow sufficient tokens for structured JSON plan
        decoding = DecodeConfig(
            temperature=0.2,
            max_tokens=2048,
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=decoding,
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "research_planning"},
        )

        response = self.engine.generate(request)

        if not response.parsed_output or not isinstance(response.parsed_output, ResearchPlan):
            raise ValueError(
                f"Failed to parse valid ResearchPlan from LLM output. Raw text: {response.text[:200]}"
            )

        plan = response.parsed_output
        if session_id:
            # Reconstruct with caller's explicit session_id if provided
            plan = ResearchPlan(
                session_id=session_id,
                objectives=plan.objectives,
                total_query_budget=plan.total_query_budget,
                max_iterations=plan.max_iterations,
            )

        return plan

    def _build_prompt_content(
        self,
        prospect: ProspectContext,
        outreach: OutreachContext,
        product: Optional[ProductContext],
    ) -> str:
        lines = [
            "### Target Prospect",
            f"- Full Name: {prospect.full_name}",
            f"- Company: {prospect.company_name}",
            f"- Domain: {prospect.company_domain or 'Unknown'}",
            f"- Title/Role: {prospect.designation or 'Unknown'}",
            f"- LinkedIn: {prospect.linkedin_url or 'None provided'}",
            f"- Location: {prospect.location or 'Unknown'}",
            "",
            "### Outreach Context",
            f"- Objective: {outreach.objective}",
            f"- Sender Identity: {outreach.sender_identity}",
            f"- Relationship: {outreach.relationship or 'cold'}",
            f"- Tone: {outreach.tone}",
        ]

        if product:
            lines.extend(
                [
                    "",
                    "### Offering Context (Background awareness only - do not bias research)",
                    f"- Product: {product.product_name}",
                    f"- Value Proposition: {product.value_proposition}",
                    f"- Relevant Problem Spaces: {', '.join(product.capabilities[:4])}",
                ]
            )

        lines.extend(
            [
                "",
                "Generate a ResearchPlan JSON with 2-4 objectives, distinct target entities, priorities, and query allocations.",
            ]
        )

        return "\n".join(lines)
