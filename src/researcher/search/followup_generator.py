from __future__ import annotations

import logging
import uuid
from typing import List, Optional, Sequence, Set
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import GROQ_SMALL
from src.context.models import ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.orchestration.models import (
    ActionType,
    RecommendedAction,
    ResearchSufficiencyAssessment,
)
from src.researcher.planning.models import TargetEntity
from src.researcher.search.query_generator import GeneratedQuery, QueryListOutput

logger = logging.getLogger(__name__)

FOLLOWUP_SYSTEM_PROMPT = """You are an expert search investigator generating targeted follow-up search queries.
An earlier round of research left specific factual gaps, identity uncertainties, or contradictions.
Your task is to generate concise, highly specific search queries to directly address these recommended actions.

Rules:
1. Natural Phrasing: Plain, human search phrases. Do NOT use boolean operators (AND, OR, NOT) or quotation marks.
2. Direct Gap Targeting: Directly query the specific missing aspect, conflicting claim, or identity attribute.
3. Grounding: Anchor all queries with the person's name or company name/domain.
4. Novelty: Do NOT repeat previously used queries. Formulate fresh search angles.

Output format: You must respond ONLY with a JSON object containing a 'queries' list of strings."""


class FollowUpQueryGenerator:
    """Generates focused follow-up search queries based on sufficiency evaluation gaps and recommended actions."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_SMALL
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(QueryListOutput)

    def generate_followup_queries(
        self,
        prospect: ProspectContext,
        assessment: ResearchSufficiencyAssessment,
        existing_queries: Sequence[str] = (),
        max_queries: int = 4,
    ) -> List[GeneratedQuery]:
        """Generate targeted follow-up queries strictly addressing recommended actions and remaining gaps."""
        if not assessment.recommended_actions and not assessment.remaining_gaps:
            return []

        # Track existing queries for deduplication
        seen_queries_lower: Set[str] = {q.strip().lower() for q in existing_queries}

        # Filter actionable recommended actions
        actionable_items = [
            a for a in assessment.recommended_actions
            if a.action_type in (
                ActionType.TARGETED_SEARCH,
                ActionType.CONFLICT_INVESTIGATION,
                ActionType.IDENTITY_VERIFICATION,
            )
        ]

        if not actionable_items and not assessment.remaining_gaps:
            return []

        user_content = self._build_prompt(prospect, actionable_items, assessment.remaining_gaps, existing_queries)

        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=FOLLOWUP_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=user_content))

        decoding = DecodeConfig(temperature=0.3, max_tokens=1024)
        request = GenerationRequest(
            prompt=prompt,
            decoding=decoding,
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "followup_query_generation"},
        )

        response = self.engine.generate(request)
        raw_queries: List[str] = []

        if response.parsed_output and isinstance(response.parsed_output, QueryListOutput):
            raw_queries = response.parsed_output.queries
        else:
            logger.warning("FollowUpQueryGenerator LLM output unparsed. Using deterministic fallback.")
            raw_queries = self._deterministic_fallback_queries(prospect, actionable_items, assessment.remaining_gaps)

        generated: List[GeneratedQuery] = []
        # Round-robin assign queries to target objectives where possible
        target_obj_ids = [a.target_objective_id for a in actionable_items if a.target_objective_id]
        fallback_obj_id = target_obj_ids[0] if target_obj_ids else None

        for idx, q_text in enumerate(raw_queries):
            cleaned = q_text.strip()
            if not cleaned or len(cleaned) < 3:
                continue
            if cleaned.lower() in seen_queries_lower:
                continue

            # Associate with corresponding action's objective if available
            obj_id = target_obj_ids[idx % len(target_obj_ids)] if target_obj_ids else fallback_obj_id

            generated.append(
                GeneratedQuery(
                    query_id=str(uuid.uuid4()),
                    query_text=cleaned,
                    objective_id=obj_id,
                    target_entity=TargetEntity.COMPANY if "company" in cleaned.lower() else TargetEntity.PROSPECT,
                )
            )
            seen_queries_lower.add(cleaned.lower())

            if len(generated) >= max_queries:
                break

        # If LLM returned only duplicates or failed to produce anything, use fallback
        if not generated and (actionable_items or assessment.remaining_gaps):
            for q_fallback in self._deterministic_fallback_queries(prospect, actionable_items, assessment.remaining_gaps):
                if q_fallback.strip().lower() not in seen_queries_lower:
                    generated.append(
                        GeneratedQuery(
                            query_id=str(uuid.uuid4()),
                            query_text=q_fallback.strip(),
                            objective_id=fallback_obj_id,
                            target_entity=TargetEntity.PROSPECT,
                        )
                    )
                    seen_queries_lower.add(q_fallback.strip().lower())
                    if len(generated) >= max_queries:
                        break

        return generated

    def _build_prompt(
        self,
        prospect: ProspectContext,
        actions: Sequence[RecommendedAction],
        gaps: Sequence[str],
        existing_queries: Sequence[str],
    ) -> str:
        lines = [
            f"### Target Entity Context",
            f"- Prospect: {prospect.full_name}",
            f"- Company: {prospect.company_name}",
            f"- Domain: {prospect.company_domain or 'None'}",
            f"- Designation: {prospect.designation or 'None'}",
            "",
            "### Recommended Actions from Previous Iteration",
        ]

        if actions:
            for a in actions:
                lines.append(f"- [{a.action_type.value}] Gap: {a.gap_description} | Suggested Focus: {a.suggested_focus}")
        else:
            for g in gaps:
                lines.append(f"- Remaining Gap: {g}")

        if existing_queries:
            lines.extend(["", "### Existing Queries (DO NOT REPEAT):"])
            for eq in existing_queries[-8:]:
                lines.append(f"- {eq}")

        lines.extend([
            "",
            "Generate 2-4 fresh, targeted follow-up queries to resolve these specific gaps.",
        ])

        return "\n".join(lines)

    def _deterministic_fallback_queries(
        self,
        prospect: ProspectContext,
        actions: Sequence[RecommendedAction],
        gaps: Sequence[str],
    ) -> List[str]:
        fallbacks: List[str] = []
        for a in actions:
            if a.suggested_focus:
                fallbacks.append(f"{prospect.company_name} {a.suggested_focus}")
            else:
                fallbacks.append(f"{prospect.full_name} {prospect.company_name} {a.gap_description}")

        for g in gaps:
            fallbacks.append(f"{prospect.company_name} {g}")

        return fallbacks
