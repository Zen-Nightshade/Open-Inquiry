from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from .budget import GlobalResearchBudget
from .models import (
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    QuestionLifecycleStatus,
    StoppingReason,
    VoIAction,
)
from .voi import ValueOfInformationEvaluator


class StoppingDecision:
    """Represents a stopping decision for a question node with full audit context."""

    def __init__(
        self,
        should_stop: bool,
        reason: Optional[StoppingReason] = None,
        new_status: Optional[QuestionLifecycleStatus] = None,
        rationale: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.should_stop = should_stop
        self.reason = reason
        self.new_status = new_status
        self.rationale = rationale
        self.details = details or {}


class StoppingEvaluator:
    """Evaluates explicit stopping decisions for Epistemic Question nodes under Open-World semantics."""

    @classmethod
    def evaluate_pre_search(
        cls,
        node: EpistemicQuestionNode,
        dag: EpistemicQuestionDAG,
        budget: GlobalResearchBudget,
    ) -> StoppingDecision:
        """Check whether a question should stop or be deferred prior to search query execution."""
        # A. Unobservable
        if node.observability == ObservabilityClass.PRIVATE_OR_NONPUBLIC:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.UNOBSERVABLE,
                new_status=QuestionLifecycleStatus.UNOBSERVABLE,
                rationale=(
                    f"Fact '{node.requested_predicate}' is private/non-public with no public research proxy. "
                    "Halting public-source investigation. Note: this does not assert the fact is absent in the world."
                ),
                details={"observability": node.observability.value, "predicate": node.requested_predicate},
            )

        # B. Dependency Blocked
        if node.dependencies:
            for dep_id in node.dependencies:
                dep_node = dag.nodes.get(dep_id)
                if not dep_node:
                    return StoppingDecision(
                        should_stop=True,
                        reason=StoppingReason.DEPENDENCY_BLOCKED,
                        new_status=QuestionLifecycleStatus.BLOCKED,
                        rationale=f"Missing prerequisite question '{dep_id}'.",
                        details={"missing_prerequisite": dep_id},
                    )
                if dep_node.status in (
                    QuestionLifecycleStatus.UNOBSERVABLE,
                    QuestionLifecycleStatus.EXHAUSTED,
                    QuestionLifecycleStatus.FAILED,
                    QuestionLifecycleStatus.SKIPPED,
                ):
                    return StoppingDecision(
                        should_stop=True,
                        reason=StoppingReason.DEPENDENCY_BLOCKED,
                        new_status=QuestionLifecycleStatus.BLOCKED,
                        rationale=(
                            f"Prerequisite '{dep_id}' terminated in {dep_node.status.value}. "
                            f"Dependent question '{node.question_id}' cannot be safely resolved independently."
                        ),
                        details={"prerequisite_id": dep_id, "prerequisite_status": dep_node.status.value},
                    )

        # C. Global Budget Exhausted
        if budget.is_exhausted or budget.remaining_total_units < node.estimated_research_cost:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.BUDGET_EXHAUSTED,
                new_status=QuestionLifecycleStatus.SKIPPED,
                rationale=(
                    f"Global research budget exhausted (remaining queries: {budget.remaining_total_units}, "
                    f"required: {node.estimated_research_cost}). Preserving completed findings."
                ),
                details={
                    "remaining_total_units": budget.remaining_total_units,
                    "total_budget_units": budget.total_budget_units,
                },
            )

        # D. Low VoI
        voi_ass = ValueOfInformationEvaluator.evaluate(node, dag, budget)
        if voi_ass.action == VoIAction.STOP:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.LOW_VOI,
                new_status=QuestionLifecycleStatus.SKIPPED,
                rationale=f"Low marginal VoI ({voi_ass.voi_score}): {voi_ass.explanation}",
                details={"voi_assessment": voi_ass.model_dump()},
            )

        return StoppingDecision(should_stop=False)

    @classmethod
    def evaluate_post_search(
        cls,
        node: EpistemicQuestionNode,
        facts_found: List[Any],
        sources_found: List[Any],
        executed_queries: List[str],
        had_provider_failure: bool = False,
    ) -> StoppingDecision:
        """Evaluate terminal status and stopping reason after search and extraction."""
        # 1. Operational Failure
        if had_provider_failure and not facts_found:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.OPERATIONAL_FAILURE,
                new_status=QuestionLifecycleStatus.FAILED,
                rationale=(
                    "External search provider or network call encountered an operational failure. "
                    "Recorded as FAILED; this does NOT represent unobservability, absence, or lack of factual existence."
                ),
                details={"had_provider_failure": True, "executed_queries": executed_queries},
            )

        # 2. Resolved Criteria Satisfaction
        if facts_found:
            # Check mandatory criteria if specified
            criteria_satisfied = True
            missing_criteria: List[str] = []

            for crit in node.required_resolution_criteria:
                if crit.is_mandatory:
                    # Verify min sources
                    if len(sources_found) < crit.min_sources:
                        criteria_satisfied = False
                        missing_criteria.append(
                            f"Requires >= {crit.min_sources} sources (found {len(sources_found)})"
                        )

            if criteria_satisfied:
                return StoppingDecision(
                    should_stop=True,
                    reason=StoppingReason.RESOLVED,
                    new_status=QuestionLifecycleStatus.RESOLVED,
                    rationale=(
                        f"Resolved with {len(facts_found)} corroborated facts across {len(sources_found)} sources."
                    ),
                    details={"facts_count": len(facts_found), "sources_count": len(sources_found)},
                )
            else:
                # Some evidence exists but incomplete
                return StoppingDecision(
                    should_stop=False,
                    reason=None,
                    new_status=QuestionLifecycleStatus.PARTIALLY_RESOLVED,
                    rationale=f"Evidence found but resolution criteria incomplete: {'; '.join(missing_criteria)}",
                    details={"missing_criteria": missing_criteria},
                )

        # 3. Exhausted Scope
        # If all search directions have been executed and no facts were found
        max_allocated = max(1, node.estimated_research_cost)
        if len(executed_queries) >= max_allocated:
            return StoppingDecision(
                should_stop=True,
                reason=StoppingReason.EXHAUSTED_SCOPE,
                new_status=QuestionLifecycleStatus.EXHAUSTED,
                rationale=(
                    f"Exhausted allocated search scope ({len(executed_queries)} queries executed). "
                    "Evidence was not established within the examined scope. This does NOT imply the fact is false or absent globally."
                ),
                details={
                    "executed_queries": executed_queries,
                    "examined_sources_count": len(sources_found),
                    "temporal_scope": node.temporal_scope or "unspecified",
                    "remaining_uncertainty": "Uncorroborated within examined web scope",
                },
            )

        # Otherwise unresolved but eligible for further turns if budget permits
        return StoppingDecision(
            should_stop=False,
            reason=None,
            new_status=QuestionLifecycleStatus.UNRESOLVED,
            rationale="No facts discovered in current turn, but query scope not yet fully exhausted.",
            details={"executed_queries": executed_queries},
        )
