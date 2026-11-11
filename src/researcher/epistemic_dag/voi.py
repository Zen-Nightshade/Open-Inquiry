from __future__ import annotations

import math
from typing import Dict, List, Optional
from .budget import GlobalResearchBudget
from .models import (
    DecisionRelevance,
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    ResolutionLikelihood,
    VoIAction,
    VoIAssessment,
)


class ValueOfInformationEvaluator:
    """Decision-Directed Value-of-Information (VoI) evaluator.
    
    Computes a transparent, deterministic heuristic score:
      VoI = (Relevance * ResolutionLikelihood) + DependencyBenefit - CostPenalty
      
    Components:
      - Relevance: in [0.15, 1.0] representing prospective impact on outreach viability.
      - ResolutionLikelihood: in [0.05, 1.0] conditioned on observability class.
      - DependencyBenefit: in [0.0, 1.0] based on downstream questions unlocked.
      - CostPenalty: in [0.0, 1.0] normalized against remaining and total budget.
      
    Thresholds:
      - VoI >= 0.35: RESEARCH (high expected epistemic yield per budget unit).
      - 0.15 <= VoI < 0.35: DEFER (marginal expected yield; hold until higher priorities complete).
      - VoI < 0.15: STOP (insufficient expected return to justify query expenditure).
    """

    RESEARCH_THRESHOLD: float = 0.35
    DEFER_THRESHOLD: float = 0.15

    @classmethod
    def evaluate(
        cls,
        node: EpistemicQuestionNode,
        dag: EpistemicQuestionDAG,
        budget: GlobalResearchBudget,
    ) -> VoIAssessment:
        """Evaluate Value of Information for a specific question node within the DAG."""
        # 1. Immediate Rule: Private fact without public proxy stops public investigation
        if node.observability == ObservabilityClass.PRIVATE_OR_NONPUBLIC:
            return VoIAssessment(
                action=VoIAction.STOP,
                voi_score=0.0,
                decision_relevance=node.decision_relevance,
                resolution_likelihood=ResolutionLikelihood.NEGLIGIBLE,
                dependency_benefit=0.0,
                cost_penalty=1.0,
                estimated_cost_units=node.estimated_research_cost,
                explanation=(
                    f"Question '{node.question_id}' is classified as PRIVATE_OR_NONPUBLIC. "
                    "Public search channels cannot resolve this premise legitimately; stopping to prevent budget waste."
                ),
            )

        # 2. Immediate Rule: Global budget exhaustion
        if budget.is_exhausted or budget.remaining_total_units < node.estimated_research_cost:
            return VoIAssessment(
                action=VoIAction.STOP,
                voi_score=0.0,
                decision_relevance=node.decision_relevance,
                resolution_likelihood=ResolutionLikelihood.NEGLIGIBLE,
                dependency_benefit=0.0,
                cost_penalty=1.0,
                estimated_cost_units=node.estimated_research_cost,
                explanation=(
                    f"Global research budget exhausted (remaining units: {budget.remaining_total_units}, "
                    f"required: {node.estimated_research_cost}). Halting further investigation."
                ),
            )

        # 3. Determine Resolution Likelihood based on Observability
        if node.observability == ObservabilityClass.PUBLICLY_OBSERVABLE:
            likelihood = ResolutionLikelihood.HIGH
        elif node.observability == ObservabilityClass.CONDITIONALLY_OBSERVABLE:
            likelihood = ResolutionLikelihood.MEDIUM
        elif node.observability == ObservabilityClass.UNKNOWN:
            likelihood = ResolutionLikelihood.LOW
        else:
            likelihood = ResolutionLikelihood.NEGLIGIBLE

        # 4. Determine Decision Relevance
        # If question is critical to strategy invalidation or core identity, ensure CRITICAL relevance
        relevance = node.decision_relevance
        if "identity" in node.requested_predicate.lower() or "works_for" in node.requested_predicate.lower():
            if relevance.normalized_weight < DecisionRelevance.HIGH.normalized_weight:
                relevance = DecisionRelevance.HIGH

        relevance_score = relevance.normalized_weight
        likelihood_score = likelihood.normalized_probability

        # 5. Calculate Dependency Benefit (downstream questions unlocked)
        dependency_benefit = cls._calculate_dependency_benefit(node.question_id, dag)

        # 6. Calculate Cost Penalty normalized against total & remaining budget
        cost_penalty = cls._calculate_cost_penalty(node.estimated_research_cost, budget)

        # 7. Compute VoI Score
        raw_voi = (relevance_score * likelihood_score) + dependency_benefit - cost_penalty
        voi_score = round(raw_voi, 4)

        # 8. Determine Action & Explanation
        if voi_score >= cls.RESEARCH_THRESHOLD:
            action = VoIAction.RESEARCH
            reason = (
                f"Sufficient net epistemic yield (VoI {voi_score} >= {cls.RESEARCH_THRESHOLD}): "
                f"relevance={relevance_score}, likelihood={likelihood_score}, "
                f"dependency_benefit={dependency_benefit:.2f}, cost_penalty={cost_penalty:.2f}."
            )
        elif voi_score >= cls.DEFER_THRESHOLD:
            action = VoIAction.DEFER
            reason = (
                f"Marginal epistemic yield (VoI {voi_score} in [{cls.DEFER_THRESHOLD}, {cls.RESEARCH_THRESHOLD})). "
                f"Deferring in favor of higher-priority investigations."
            )
        else:
            action = VoIAction.STOP
            reason = (
                f"Low expected epistemic value (VoI {voi_score} < {cls.DEFER_THRESHOLD}). "
                f"Investigation cost outweighs estimated decision benefit."
            )

        return VoIAssessment(
            action=action,
            voi_score=voi_score,
            decision_relevance=relevance,
            resolution_likelihood=likelihood,
            dependency_benefit=round(dependency_benefit, 4),
            cost_penalty=round(cost_penalty, 4),
            estimated_cost_units=node.estimated_research_cost,
            explanation=reason,
        )

    @classmethod
    def _calculate_dependency_benefit(cls, question_id: str, dag: EpistemicQuestionDAG) -> float:
        """Sum weighted importance of downstream questions that directly or indirectly depend on this question."""
        dependents: List[EpistemicQuestionNode] = []
        for other_node in dag.nodes.values():
            if question_id in other_node.dependencies:
                dependents.append(other_node)

        if not dependents:
            return 0.0

        benefit = 0.0
        for dep in dependents:
            if dep.decision_relevance == DecisionRelevance.CRITICAL:
                benefit += 0.40
            elif dep.decision_relevance == DecisionRelevance.HIGH:
                benefit += 0.25
            elif dep.decision_relevance == DecisionRelevance.MEDIUM:
                benefit += 0.15
            else:
                benefit += 0.05

        # Cap dependency benefit at 1.0
        return min(1.0, benefit)

    @classmethod
    def _calculate_cost_penalty(cls, estimated_cost: int, budget: GlobalResearchBudget) -> float:
        """Calculate bounded cost penalty normalized against total and remaining budget."""
        if budget.remaining_total_units <= 0:
            return 1.0

        # Cost as proportion of remaining budget
        rem_ratio = estimated_cost / max(1, budget.remaining_total_units)
        # Cost as proportion of total configured budget
        tot_ratio = estimated_cost / max(1, budget.total_budget_units)

        # Weighted combination capped at 1.0
        penalty = (0.6 * rem_ratio) + (0.4 * tot_ratio)
        return min(1.0, penalty)
