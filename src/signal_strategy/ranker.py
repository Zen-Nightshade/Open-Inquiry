from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import GROQ_LARGE
from src.context.models import ProductContext, ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import CanonicalFact, ConfidenceLevel, FactCategory
from .models import CandidateSignal, RankedSignal, SignalScore

logger = logging.getLogger(__name__)


class RawEvaluatedSignal(BaseModel):
    """LLM scoring output for a candidate signal."""

    model_config = ConfigDict(extra="ignore")

    signal_id: str
    prospect_relevance: float = Field(ge=0.0, le=1.0, description="Relevance to the prospect's role and decision-making scope")
    product_alignment: float = Field(ge=0.0, le=1.0, description="Direct fit with product capabilities")
    business_significance: float = Field(ge=0.0, le=1.0, description="Scale and commercial impact of the event for target company")
    rationale: str = Field(description="Explanation of commercial relevance, product fit, and scoring")


class RawSignalEvaluationOutput(BaseModel):
    """Container for scored candidate signals from LLM."""

    model_config = ConfigDict(extra="ignore")

    evaluations: List[RawEvaluatedSignal] = Field(default_factory=list)


SIGNAL_RANKER_SYSTEM_PROMPT = """You are a senior B2B outreach strategist and commercial opportunity evaluator.
Your task is to critically evaluate candidate signals for outreach relevance, product capability alignment, and commercial significance.

EVALUATION DIMENSIONS:
1. prospect_relevance (0.0 to 1.0):
   - Does this individual's job title, role, and department care deeply about this event?
   - High (0.8-1.0): Direct operational/budgetary responsibility (e.g. VP Eng on infrastructure/dev scaling).
   - Low (0.0-0.3): Completely outside the person's functional department or seniority level.

2. product_alignment (0.0 to 1.0):
   - Do our product's explicit capabilities directly solve a business problem created by this event?
   - High (0.8-1.0): Product capability directly addresses the exact operational implication.
   - Low (0.0-0.2): No logical link between our product offering and this event. Do NOT fabricate synergies!

3. business_significance (0.0 to 1.0):
   - What is the strategic scale and commercial magnitude for the company?
   - High (0.8-1.0): Major funding, executive shift, core product expansion, acquisition.
   - Low (0.0-0.3): Routine minor activity or small social announcement.
"""


class SignalRanker:
    """Evaluates, scores, and ranks candidate signals across multiple commercial dimensions."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(RawSignalEvaluationOutput)

    def rank_signals(
        self,
        signals: Sequence[CandidateSignal],
        facts: Sequence[CanonicalFact],
        prospect: ProspectContext,
        product: ProductContext,
    ) -> List[RankedSignal]:
        """Score all candidate signals and return them sorted in descending order of composite score."""
        if not signals:
            return []

        facts_by_id = {f.fact_id: f for f in facts}

        # Step 1: Precompute deterministic evidence_strength and recency_score
        deterministic_scores: Dict[str, tuple[float, float]] = {}
        for sig in signals:
            ev_strength = self._calculate_evidence_strength(sig, facts_by_id)
            recency = self._calculate_recency_score(sig)
            deterministic_scores[sig.signal_id] = (ev_strength, recency)

        # Step 2: Use LLM for semantic evaluation (prospect relevance, product alignment, business significance)
        llm_evaluations = self._evaluate_signals_with_llm(signals, prospect, product)

        # Step 3: Combine scores and compute composite score with gating
        ranked_signals: List[RankedSignal] = []
        for sig in signals:
            ev_strength, recency = deterministic_scores[sig.signal_id]
            eval_data = llm_evaluations.get(sig.signal_id)

            if eval_data:
                p_rel = eval_data.prospect_relevance
                p_align = eval_data.product_alignment
                b_sig = eval_data.business_significance
                rationale = eval_data.rationale
            else:
                p_rel, p_align, b_sig, rationale = self._heuristic_fallback_scores(sig, prospect, product)

            composite = self._calculate_composite_score(
                evidence_strength=ev_strength,
                recency_score=recency,
                prospect_relevance=p_rel,
                product_alignment=p_align,
                business_significance=b_sig,
            )

            score = SignalScore(
                evidence_strength=round(ev_strength, 2),
                recency_score=round(recency, 2),
                prospect_relevance=round(p_rel, 2),
                product_alignment=round(p_align, 2),
                business_significance=round(b_sig, 2),
                composite_score=round(composite, 2),
            )

            ranked_signals.append(
                RankedSignal(
                    signal=sig,
                    score=score,
                    rationale=rationale,
                )
            )

        # Sort descending by composite score
        ranked_signals.sort(key=lambda s: s.score.composite_score, reverse=True)
        return ranked_signals

    def _calculate_evidence_strength(
        self,
        signal: CandidateSignal,
        facts_by_id: Dict[str, CanonicalFact],
    ) -> float:
        """Deterministically calculate evidence strength from verified CanonicalFacts."""
        supporting_facts = [facts_by_id[fid] for fid in signal.supporting_fact_ids if fid in facts_by_id]
        if not supporting_facts:
            return 0.15

        fact_scores = []
        for fact in supporting_facts:
            if fact.composite_confidence == ConfidenceLevel.HIGH:
                score = 0.90
            elif fact.composite_confidence == ConfidenceLevel.MEDIUM:
                score = 0.65
            elif fact.composite_confidence == ConfidenceLevel.LOW:
                score = 0.35
            else:
                score = 0.15

            # Boost if independently corroborated across multiple sources
            if len(fact.source_ids) > 1:
                score = min(1.0, score + 0.05)
            fact_scores.append(score)

        avg_score = sum(fact_scores) / len(fact_scores)
        if signal.is_inferred:
            avg_score *= 0.85
        return min(1.0, max(0.0, avg_score))

    def _calculate_recency_score(self, signal: CandidateSignal) -> float:
        """Deterministically compute freshness from event date."""
        if not signal.event_date:
            return 0.45

        today = datetime.date.today()
        days_ago = (today - signal.event_date).days

        if days_ago < 0:
            # Future event or forward-looking plan
            return 0.90
        elif days_ago <= 30:
            return 1.0
        elif days_ago <= 90:
            return 0.85
        elif days_ago <= 180:
            return 0.70
        elif days_ago <= 365:
            return 0.50
        else:
            return 0.30

    def _calculate_composite_score(
        self,
        evidence_strength: float,
        recency_score: float,
        prospect_relevance: float,
        product_alignment: float,
        business_significance: float,
    ) -> float:
        """
        Calculate weighted composite score with commercial gating.
        
        Gating Rule: If product_alignment < 0.25, the signal cannot qualify as a viable
        lead hook for outreach regardless of how high other dimensions score.
        """
        raw_composite = (
            product_alignment * 0.35
            + prospect_relevance * 0.25
            + business_significance * 0.15
            + evidence_strength * 0.15
            + recency_score * 0.10
        )

        if product_alignment < 0.25:
            return min(raw_composite, product_alignment)

        return min(1.0, max(0.0, raw_composite))

    def _evaluate_signals_with_llm(
        self,
        signals: Sequence[CandidateSignal],
        prospect: ProspectContext,
        product: ProductContext,
    ) -> Dict[str, RawEvaluatedSignal]:
        """Invoke LLM to semantically score prospect relevance, product alignment, and business significance."""
        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=SIGNAL_RANKER_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=self._build_ranking_prompt(signals, prospect, product)))

        request = GenerationRequest(
            prompt=prompt,
            decoding=DecodeConfig(temperature=0.2, max_tokens=2048),
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "signal_ranking"},
        )

        try:
            response = self.engine.generate(request)
            raw = response.parsed_output
            if isinstance(raw, RawSignalEvaluationOutput):
                eval_by_id = {ev.signal_id: ev for ev in raw.evaluations if ev.signal_id}
                # Support positional fallback if signal_id is missing or empty
                if len(raw.evaluations) == len(signals):
                    for idx, sig in enumerate(signals):
                        if sig.signal_id not in eval_by_id:
                            eval_by_id[sig.signal_id] = raw.evaluations[idx]
                return eval_by_id
        except Exception as e:
            logger.warning(f"Error during LLM signal ranking: {e}. Using heuristic scoring.")

        return {}

    def _build_ranking_prompt(
        self,
        signals: Sequence[CandidateSignal],
        prospect: ProspectContext,
        product: ProductContext,
    ) -> str:
        lines = [
            "### Prospect Context",
            f"- Name: {prospect.full_name}",
            f"- Company: {prospect.company_name}",
            f"- Role/Title: {prospect.designation or 'Unknown'}",
            "",
            "### Product Offering",
            f"- Product Name: {product.product_name}",
            f"- Value Proposition: {product.value_proposition}",
            "- Core Capabilities:",
        ]
        for cap in product.capabilities:
            lines.append(f"  * {cap}")

        lines.extend([
            "",
            "### Candidate Signals to Evaluate:",
        ])
        for s in signals:
            lines.extend([
                f"- [Signal ID: {s.signal_id}] {s.label}",
                f"  Category: {s.fact_category.value}",
                f"  Observed Event: {s.observed_event or s.description}",
                f"  Business Implication: {s.business_implication or 'None specified'}",
                f"  Commercial Opportunity: {s.commercial_opportunity or 'None specified'}",
                "",
            ])

        lines.append("Score each signal (0.0 to 1.0) on prospect_relevance, product_alignment, and business_significance.")
        return "\n".join(lines)

    def _heuristic_fallback_scores(
        self,
        signal: CandidateSignal,
        prospect: ProspectContext,
        product: ProductContext,
    ) -> tuple[float, float, float, str]:
        """Heuristic fallback scoring when LLM is unavailable."""
        title_lower = (prospect.designation or "").lower()
        signal_text = f"{signal.label} {signal.description}".lower()

        # Prospect relevance heuristic
        p_rel = 0.5
        if any(w in title_lower for w in ["head", "vp", "director", "chief", "lead"]):
            p_rel += 0.15
        if any(w in signal_text and w in title_lower for w in ["engineer", "ai", "product", "sales", "finance"]):
            p_rel += 0.2
        p_rel = min(1.0, p_rel)

        # Product alignment heuristic
        p_align = 0.4
        cap_words = " ".join(product.capabilities).lower().split()
        matches = sum(1 for w in cap_words if len(w) > 4 and w in signal_text)
        if matches >= 2:
            p_align = 0.8
        elif matches == 1:
            p_align = 0.6

        # Business significance heuristic
        b_sig = 0.5
        if signal.fact_category in (FactCategory.FUNDING, FactCategory.PRODUCT_LAUNCH, FactCategory.ORGANIZATION_CHANGE):
            b_sig = 0.8
        elif signal.fact_category in (FactCategory.ACHIEVEMENT, FactCategory.HIRING):
            b_sig = 0.65

        rationale = f"Heuristically scored based on {signal.fact_category.value} alignment with {product.product_name}."
        return p_rel, p_align, b_sig, rationale
