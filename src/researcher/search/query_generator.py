from __future__ import annotations

import logging
import uuid
from typing import List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import GROQ_SMALL
from src.context.models import ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.planning.models import ResearchObjective, TargetEntity

logger = logging.getLogger(__name__)


class GeneratedQuery(BaseModel):
    """An individual search query generated for a specific research objective."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = Field(min_length=3, description="Plain human search query string")
    objective_id: Optional[str] = Field(default=None, description="Traceability link to the originating ResearchObjective")
    target_entity: TargetEntity = Field(description="Whether this query targets the prospect, company, or both")


class QueryListOutput(BaseModel):
    """Intermediate structured output schema for the query generator LLM."""

    queries: List[str] = Field(
        min_length=1,
        description="List of plain natural language search queries without boolean operators",
    )


QUERY_GENERATOR_SYSTEM_PROMPT = """You are a specialized search query formulation engine for B2B research.
Your task is to transform a research objective into effective, concise search queries.

Rules:
1. Natural Phrasing: Generate search queries as plain, natural search strings that a human researcher would type.
2. NO Boolean Operators: Do NOT use 'AND', 'OR', 'NOT', or nested parentheses. These reduce search recall on modern web engines.
3. Specificity & Grounding: Anchor queries to specific identifiers (e.g. prospect name, company name, domain, title).
4. Focus Separation:
   - For 'prospect' targets: Include the person's name and company or career context.
   - For 'company' targets: Include the company name and commercial terms (e.g., funding, product, expansion, hiring, customers).
5. No Unsupported Assumptions: Do not assume unstated events or technologies without grounding.

Output format: You must respond ONLY with a JSON object containing a 'queries' list of strings."""


class QueryGenerator:
    """Generates targeted, deduplicated web search queries for research objectives."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_SMALL
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(QueryListOutput)

    def generate_queries(
        self,
        objective: ResearchObjective,
        prospect: ProspectContext,
        existing_queries: Optional[List[str]] = None,
    ) -> List[GeneratedQuery]:
        """Generate a list of unique search queries mapped to the supplied objective."""
        known_queries: Set[str] = {q.strip().lower() for q in (existing_queries or []) if q.strip()}
        user_prompt = self._build_prompt_content(objective, prospect)

        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=QUERY_GENERATOR_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=user_prompt))

        decoding = DecodeConfig(
            temperature=0.3,
            max_tokens=1024,
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=decoding,
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "query_generation", "objective_id": objective.objective_id},
        )

        response = self.engine.generate(request)

        if not response.parsed_output or not isinstance(response.parsed_output, QueryListOutput):
            raise ValueError(
                f"Failed to generate structured queries for objective {objective.objective_id}. "
                f"Raw output: {response.text[:200]}"
            )

        raw_queries = response.parsed_output.queries
        generated: List[GeneratedQuery] = []
        batch_seen: Set[str] = set()

        for q_text in raw_queries:
            cleaned = self._clean_query(q_text)
            norm = cleaned.lower()

            if not cleaned or len(cleaned) < 3:
                continue
            if norm in known_queries or norm in batch_seen:
                continue

            batch_seen.add(norm)
            generated.append(
                GeneratedQuery(
                    query_text=cleaned,
                    objective_id=objective.objective_id,
                    target_entity=objective.target_entity,
                )
            )

            if len(generated) >= objective.max_queries:
                break

        return generated

    def _clean_query(self, query: str) -> str:
        """Strip surrounding quotes, bullets, and markdown formatting."""
        cleaned = query.strip()
        # Remove common bullet points first
        if cleaned.startswith(("- ", "* ", "• ")):
            cleaned = cleaned[2:].strip()
        elif cleaned.startswith("-") or cleaned.startswith("*"):
            cleaned = cleaned[1:].strip()

        # Then strip matching enclosing quotes/backticks
        while (
            (cleaned.startswith('"') and cleaned.endswith('"'))
            or (cleaned.startswith("'") and cleaned.endswith("'"))
            or (cleaned.startswith("`") and cleaned.endswith("`"))
        ) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()

        # Also strip single quotes/backticks on boundaries
        return cleaned.strip('"`\' ')

    def _build_prompt_content(
        self,
        objective: ResearchObjective,
        prospect: ProspectContext,
    ) -> str:
        lines = [
            "### Research Objective",
            f"- Objective ID: {objective.objective_id}",
            f"- Type: {objective.type.value}",
            f"- Target Entity: {objective.target_entity.value}",
            f"- Goal: {objective.description}",
            f"- Temporal Scope: {objective.temporal_scope or 'Any'}",
            f"- Max Queries Desired: {objective.max_queries}",
            "",
            "### Context Identifiers",
            f"- Prospect Name: {prospect.full_name}",
            f"- Company Name: {prospect.company_name}",
            f"- Company Domain: {prospect.company_domain or 'Unknown'}",
            f"- Designation: {prospect.designation or 'Unknown'}",
            f"- Location: {prospect.location or 'Unknown'}",
            "",
            f"Generate up to {objective.max_queries} distinct, high-signal plain search queries targeting this objective.",
        ]
        return "\n".join(lines)
