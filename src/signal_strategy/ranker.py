from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.config.settings import GROQ_LARGE
from src.context.models import CompatibilityStatus, ProductContext, ProspectContext
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import CanonicalFact, ConfidenceLevel, FactCategory, ResearchPackage
from .eligibility import SignalEligibilityEvaluator
from .models import (
    CandidateSignal,
    EligibilityStatus,
    RankedSignal,
    SignalEligibility,
    SignalScore,
)

logger = logging.getLogger(__name__)


class RawEvaluatedSignal(BaseModel):
    """LLM scoring output for a candidate signal."""

    model_config = ConfigDict(extra="ignore")

    signal_id: str
    prospect_relevance: float = Field(ge=0.0, le=1.0, description="Relevance to the prospect's role and decision-making scope")
    product_alignment: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Direct fit with product capabilities")
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
        eligibility_evaluator: Optional[SignalEligibilityEvaluator] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(RawSignalEvaluationOutput)
        self.eligibility_evaluator = eligibility_evaluator or SignalEligibilityEvaluator()

    def rank_signals(
        self,
        signals: Sequence[CandidateSignal],
        facts: Sequence[CanonicalFact],
        prospect: ProspectContext,
        product: ProductContext,
        research_package: Optional[ResearchPackage] = None,
    ) -> List[RankedSignal]:
        """Score all candidate signals separating non-compensatory eligibility from comparative ranking."""
        if not signals:
            return []

        facts_by_id = {f.fact_id: f for f in facts}

        # Step 1: Precompute deterministic evidence_strength and recency_score
        deterministic_scores: Dict[str, tuple[float, float]] = {}
        for sig in signals:
            ev_strength = self._calculate_evidence_strength(sig, facts_by_id)
            recency = self._calculate_recency_score(sig)
            deterministic_scores[sig.signal_id] = (ev_strength, recency)

        # Step 2: Use LLM for semantic evaluation proposal (prospect relevance, product alignment, business significance)
        llm_evaluations: Dict[str, RawEvaluatedSignal] = {}
        exec_id: Optional[str] = None
        model_name: Optional[str] = None

        if signals:
            llm_evaluations, exec_id, model_name = self._evaluate_signals_with_llm(
                signals, prospect, product
            )

        # Step 3: Evaluate non-compensatory eligibility gates for every signal using LLM proposals
        eligibility_by_id: Dict[str, SignalEligibility] = {}
        for sig in signals:
            eval_data = llm_evaluations.get(sig.signal_id)
            eligibility_by_id[sig.signal_id] = self.eligibility_evaluator.evaluate(
                signal=sig,
                facts_by_id=facts_by_id,
                prospect=prospect,
                product=product,
                research_package=research_package,
                proposal_alignment=eval_data.product_alignment if eval_data else None,
                proposal_rationale=eval_data.rationale if eval_data else None,
            )

        verified_signals = [s for s in signals if eligibility_by_id[s.signal_id].is_verified_eligible]
        conditional_signals = [s for s in signals if eligibility_by_id[s.signal_id].requires_diagnostic_strategy]
        ineligible_signals = [s for s in signals if eligibility_by_id[s.signal_id].is_ineligible]

        # Step 4: Rank verified-compatible candidates
        verified_ranked: List[RankedSignal] = []
        for sig in verified_signals:
            ev_strength, recency = deterministic_scores[sig.signal_id]
            eval_data = llm_evaluations.get(sig.signal_id)
            elig = eligibility_by_id[sig.signal_id]

            has_contract_verified_cap = (
                elig.matched_capability_name is not None
                and elig.compatibility_result is not None
                and elig.compatibility_result.status == CompatibilityStatus.VERIFIED_COMPATIBLE
                and elig.product_gate.passed
                and not elig.product_gate.details.get("is_unevaluated", False)
            )

            if eval_data:
                p_rel = eval_data.prospect_relevance
                b_sig = eval_data.business_significance
                rationale = eval_data.rationale
                if eval_data.product_alignment is not None:
                    p_align = eval_data.product_alignment
                elif has_contract_verified_cap:
                    p_align = 1.0
                else:
                    p_align = None

                prov = OutputProvenance(
                    origin=ProvenanceOrigin.MODEL,
                    component="signal_ranker",
                    model_name=model_name,
                    provider=self.provider_name,
                    model_execution_id=exec_id,
                )
            else:
                p_rel, p_align_fb, b_sig, rationale = self._heuristic_fallback_scores(sig, prospect, product)
                if has_contract_verified_cap:
                    p_align = 1.0
                else:
                    p_align = p_align_fb

                prov = OutputProvenance(
                    origin=ProvenanceOrigin.HEURISTIC_FALLBACK,
                    component="signal_ranker",
                    rule_id="heuristic_fallback_scores",
                    is_fallback=True,
                )

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
                product_alignment=round(p_align, 2) if p_align is not None else None,
                business_significance=round(b_sig, 2),
                composite_score=round(composite, 2),
                provenance=prov,
            )

            verified_ranked.append(
                RankedSignal(
                    signal=sig,
                    score=score,
                    rationale=rationale,
                    eligibility=elig,
                    provenance=prov,
                )
            )

        verified_ranked.sort(key=lambda s: s.score.composite_score, reverse=True)

        # Step 5: Rank conditional candidates separately (epistemic/commercial score, product_alignment=None)
        conditional_ranked: List[RankedSignal] = []
        for sig in conditional_signals:
            ev_strength, recency = deterministic_scores[sig.signal_id]
            eval_data = llm_evaluations.get(sig.signal_id)
            elig = eligibility_by_id[sig.signal_id]

            if eval_data:
                p_rel = eval_data.prospect_relevance
                b_sig = eval_data.business_significance
                rationale = eval_data.rationale
                prov = OutputProvenance(
                    origin=ProvenanceOrigin.MODEL,
                    component="signal_ranker",
                    model_name=model_name,
                    provider=self.provider_name,
                    model_execution_id=exec_id,
                )
            else:
                p_rel, _, b_sig, rationale = self._heuristic_fallback_scores(sig, prospect, product)
                prov = OutputProvenance(
                    origin=ProvenanceOrigin.HEURISTIC_FALLBACK,
                    component="signal_ranker",
                    rule_id="heuristic_fallback_scores",
                    is_fallback=True,
                )

            cond_score = self._calculate_conditional_score(
                evidence_strength=ev_strength,
                recency_score=recency,
                prospect_relevance=p_rel,
                business_significance=b_sig,
            )

            score = SignalScore(
                evidence_strength=round(ev_strength, 2),
                recency_score=round(recency, 2),
                prospect_relevance=round(p_rel, 2),
                product_alignment=None,  # Preserved as None; no numerical approximation
                business_significance=round(b_sig, 2),
                composite_score=round(cond_score, 2),
                provenance=prov,
            )

            conditional_ranked.append(
                RankedSignal(
                    signal=sig,
                    score=score,
                    rationale=rationale,
                    eligibility=elig,
                    provenance=prov,
                )
            )

        conditional_ranked.sort(key=lambda s: s.score.composite_score, reverse=True)

        # Step 6: Construct RankedSignals for ineligible signals (not in either pool)
        ineligible_ranked: List[RankedSignal] = []
        for sig in ineligible_signals:
            ev_strength, recency = deterministic_scores[sig.signal_id]
            eval_data = llm_evaluations.get(sig.signal_id)
            elig = eligibility_by_id[sig.signal_id]

            prov = OutputProvenance(
                origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                component="signal_ranker",
                rule_id="non_compensatory_ineligibility_veto",
            )

            p_align = (
                0.0
                if (eval_data and eval_data.product_alignment == 0.0)
                or (elig.product_gate and not elig.product_gate.passed and elig.compatibility_result and elig.compatibility_result.status == CompatibilityStatus.INCOMPATIBLE)
                else (eval_data.product_alignment if eval_data else None)
            )

            score = SignalScore(
                evidence_strength=round(ev_strength, 2),
                recency_score=round(recency, 2),
                prospect_relevance=0.0,
                product_alignment=p_align,
                business_significance=0.0,
                composite_score=0.0,
                provenance=prov,
            )

            reasons_str = "; ".join(elig.rejection_reasons) if elig.rejection_reasons else "Failed eligibility gate"
            rationale = eval_data.rationale if (eval_data and eval_data.product_alignment == 0.0) else f"Ineligible for outreach: {reasons_str}."

            ineligible_ranked.append(
                RankedSignal(
                    signal=sig,
                    score=score,
                    rationale=rationale,
                    eligibility=elig,
                    provenance=prov,
                )
            )

        ineligible_ranked.sort(key=lambda s: (s.score.recency_score, s.score.evidence_strength), reverse=True)

        # Invariant: verified-compatible candidates ALWAYS precede conditional candidates
        return verified_ranked + conditional_ranked + ineligible_ranked

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
        product_alignment: Optional[float],
        business_significance: float,
    ) -> float:
        """
        Calculate weighted composite score with commercial gating for verified-compatible signals.
        
        Gating Rules:
        - If product_alignment is None (unevaluated): The signal cannot qualify as a viable
          lead hook for outreach regardless of other scores. Composite score is capped at 0.20
          so that other dimensions cannot compensate for missing product relevance.
        - If product_alignment < 0.25 (evaluated low/unviable): Gated to min(raw_composite, product_alignment).
        """
        effective_alignment = product_alignment if product_alignment is not None else 0.0

        raw_composite = (
            effective_alignment * 0.35
            + prospect_relevance * 0.25
            + business_significance * 0.15
            + evidence_strength * 0.15
            + recency_score * 0.10
        )

        if product_alignment is None:
            return min(raw_composite, 0.20)

        if product_alignment < 0.25:
            return min(raw_composite, product_alignment)

        return min(1.0, max(0.0, raw_composite))

    def _calculate_conditional_score(
        self,
        evidence_strength: float,
        recency_score: float,
        prospect_relevance: float,
        business_significance: float,
    ) -> float:
        """
        Score conditional candidates within their separate diagnostic pool.
        
        Evaluates purely the epistemic and commercial dimensions without fabricating a product alignment number.
        Weights normalized across the 4 non-alignment dimensions (0.25 + 0.15 + 0.15 + 0.10 = 0.65).
        """
        raw_score = (
            prospect_relevance * 0.25
            + business_significance * 0.15
            + evidence_strength * 0.15
            + recency_score * 0.10
        ) / 0.65
        return min(1.0, max(0.0, raw_score))

    def _evaluate_signals_with_llm(
        self,
        signals: Sequence[CandidateSignal],
        prospect: ProspectContext,
        product: ProductContext,
    ) -> tuple[Dict[str, RawEvaluatedSignal], Optional[str], Optional[str]]:
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
            exec_id = response.metadata.get("execution_id") if hasattr(response, "metadata") and response.metadata else None
            model_name = getattr(response, "model_name", None)
            if isinstance(raw, RawSignalEvaluationOutput):
                eval_by_id = {ev.signal_id: ev for ev in raw.evaluations if ev.signal_id}
                # Support positional fallback if signal_id is missing or empty
                if len(raw.evaluations) == len(signals):
                    for idx, sig in enumerate(signals):
                        if sig.signal_id not in eval_by_id:
                            eval_by_id[sig.signal_id] = raw.evaluations[idx]
                return eval_by_id, exec_id, model_name
        except Exception as e:
            logger.warning(f"Error during LLM signal ranking: {e}. Using heuristic scoring.")

        return {}, None, None

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
    ) -> tuple[float, Optional[float], float, str]:
        """Heuristic fallback scoring when LLM is unavailable.
        
        Product alignment is preserved as None (unevaluated) rather than fabricating
        an arbitrary fallback score.
        """
        title_lower = (prospect.designation or "").lower()
        signal_text = f"{signal.label} {signal.description}".lower()

        # Prospect relevance heuristic
        p_rel = 0.5
        if any(w in title_lower for w in ["head", "vp", "director", "chief", "lead"]):
            p_rel += 0.15
        if any(w in signal_text and w in title_lower for w in ["engineer", "ai", "product", "sales", "finance"]):
            p_rel += 0.2
        p_rel = min(1.0, p_rel)

        # Product alignment is unevaluated when model scoring is unavailable.
        # Do not fabricate a 0.40 fallback or assume relevance without model evaluation.
        p_align: Optional[float] = None

        # Business significance heuristic
        b_sig = 0.5
        if signal.fact_category in (FactCategory.FUNDING, FactCategory.PRODUCT_LAUNCH, FactCategory.ORGANIZATION_CHANGE):
            b_sig = 0.8
        elif signal.fact_category in (FactCategory.ACHIEVEMENT, FactCategory.HIRING):
            b_sig = 0.65

        rationale = (
            f"Heuristically scored for prospect relevance and {signal.fact_category.value} business significance; "
            f"product alignment is unevaluated because LLM scoring was unavailable."
        )
        return p_rel, p_align, b_sig, rationale
