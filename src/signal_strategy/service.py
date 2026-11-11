from __future__ import annotations

import logging
from typing import Optional

from src.config.settings import GROQ_LARGE
from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.generator.engine.engine import GeneratorEngine
from src.researcher.models import ConfidenceLevel, ResearchPackage, ResolutionStatus
from .identifier import SignalIdentifier
from .models import StrategyResult
from .ranker import SignalRanker
from .strategy_builder import StrategyBuilder

logger = logging.getLogger(__name__)


class SignalStrategyService:
    """Orchestrates signal identification, multi-dimensional ranking, and strategy blueprint formulation."""

    def __init__(
        self,
        identifier: Optional[SignalIdentifier] = None,
        ranker: Optional[SignalRanker] = None,
        strategy_builder: Optional[StrategyBuilder] = None,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        p_name = provider_name or GROQ_LARGE
        self.identifier = identifier or SignalIdentifier(engine=engine, provider_name=p_name)
        self.ranker = ranker or SignalRanker(engine=engine, provider_name=p_name)
        self.strategy_builder = strategy_builder or StrategyBuilder(engine=engine, provider_name=p_name)

    def develop_strategy(
        self,
        research_package: ResearchPackage,
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
    ) -> StrategyResult:
        """
        Transform a research package and contexts into an evidence-grounded outreach strategy.
        
        Preserves ambiguity and returns an explicit unviable result when evidence does not support
        a commercially meaningful outreach hook.
        """
        # Step 1: Pre-flight check on research package
        if not research_package.facts:
            logger.info("Research package contains zero facts; aborting strategy formulation.")
            return StrategyResult(
                is_viable=False,
                rejection_reason="Research package contains no factual evidence.",
                unviable_reasons=["No canonical facts collected during research stage to identify signals from."],
            )

        # Check identity resolution status
        if research_package.company_entity and research_package.company_entity.resolution_status in (
            ResolutionStatus.AMBIGUOUS,
            ResolutionStatus.UNRESOLVED,
        ):
            reason = f"Company identity resolution failed with status: {research_package.company_entity.resolution_status.value}."
            logger.warning(reason)
            return StrategyResult(
                is_viable=False,
                rejection_reason=reason,
                unviable_reasons=[reason],
            )

        if research_package.prospect_entity and research_package.prospect_entity.resolution_status in (
            ResolutionStatus.AMBIGUOUS,
            ResolutionStatus.UNRESOLVED,
        ):
            reason = f"Prospect identity resolution failed with status: {research_package.prospect_entity.resolution_status.value}."
            logger.warning(reason)
            return StrategyResult(
                is_viable=False,
                rejection_reason=reason,
                unviable_reasons=[reason],
            )

        # Step 2: Identify candidate signals grounded in facts
        candidate_signals = self.identifier.identify_signals(
            facts=research_package.facts,
            prospect=prospect,
            product=product,
        )

        if not candidate_signals:
            logger.info("Zero candidate signals produced after grounding validation.")
            return StrategyResult(
                is_viable=False,
                rejection_reason="No qualifying commercial signals identified from factual evidence.",
                unviable_reasons=["Available facts did not describe commercial events or all candidates failed grounding validation."],
            )

        # Step 3: Multi-dimensional evaluation and ranking
        ranked_signals = self.ranker.rank_signals(
            signals=candidate_signals,
            facts=research_package.facts,
            prospect=prospect,
            product=product,
            research_package=research_package,
        )

        # Step 4: Build strategy blueprint with viability gating
        result = self.strategy_builder.build_strategy(
            ranked_signals=ranked_signals,
            prospect=prospect,
            product=product,
            outreach=outreach,
            research_package=research_package,
        )

        return result
