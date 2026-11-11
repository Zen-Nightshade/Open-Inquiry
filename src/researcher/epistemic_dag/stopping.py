from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
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


def normalize_predicate_token(predicate: str) -> str:
    """Normalize a predicate string to its canonical semantic token."""
    p = predicate.strip().lower().replace("-", "_").replace(" ", "_")

    # Headcount & Employee Count
    if p in ("headcount", "employee_count", "team_size", "workforce", "team_headcount"):
        return "employee_count"

    # Funding
    if p in ("funding", "funding_round", "funding_raised", "announced_funding", "raised_funding", "investment"):
        return "funding_raised"

    # Hiring / Leadership
    if p in ("executive_hiring", "leadership_appointed", "leadership", "executive_appointment"):
        return "leadership_appointed"
    if p in ("hiring", "hiring_executed", "hired", "recruited"):
        return "hiring_executed"
    if p in ("hiring_planned", "plans_to_hire", "planned_hiring"):
        return "hiring_planned"

    # Location / Facilities
    if p in ("headquartered_in", "headquarters", "based_in", "hq"):
        return "headquartered_in"
    if p in ("facility_opened", "office_opened", "opened_office"):
        return "facility_opened"
    if p in ("facility_closed", "office_closed", "closed_office"):
        return "facility_closed"
    if p in ("operating_in", "operates_in"):
        return "operating_in"

    # Technology
    if p in ("technology_stack", "tech_stack", "technologies", "tech_infrastructure"):
        return "technology_stack"

    # M&A / Corporate
    if p in ("acquisition_acquired", "acquisition", "acquired", "merger"):
        return "acquisition_acquired"
    if p in ("partnership_formed", "partnership"):
        return "partnership_formed"
    if p in ("product_launched", "product_launch", "launched_product"):
        return "product_launched"
    if p in ("works_for", "employment", "employed_by"):
        return "works_for"

    return p


def fact_matches_predicate(node_predicate: Optional[str], fact: Any) -> bool:
    """Check if fact has a predicate or category that semantically matches node_predicate."""
    if not node_predicate or node_predicate in ("test_predicate", "neutral_discovery", "*"):
        return True

    norm_node_p = normalize_predicate_token(node_predicate)

    # 1. If fact is a dict
    if isinstance(fact, dict):
        raw_p = (
            fact.get("predicate")
            or fact.get("requested_predicate")
            or fact.get("normalized_predicate")
        )
        if raw_p:
            norm_fact_p = normalize_predicate_token(str(raw_p))
            if norm_node_p == norm_fact_p:
                return True
            if norm_node_p in ("leadership_appointed", "hiring_executed", "hiring_planned") and norm_fact_p in ("leadership_appointed", "hiring_executed", "hiring_planned", "hiring"):
                return True
            return False

        cat = fact.get("fact_category") or fact.get("category")
        if cat:
            cat_str = str(cat.value if hasattr(cat, "value") else cat).lower()
            if "funding" in norm_node_p and "funding" in cat_str:
                return True
            if ("hiring" in norm_node_p or "leadership" in norm_node_p or "employee" in norm_node_p) and ("hiring" in cat_str or "organization" in cat_str):
                return True
            if "product" in norm_node_p and "product" in cat_str:
                return True
            # Explicit category rejections
            if "funding" in norm_node_p and "funding" not in cat_str:
                return False
            if ("employee" in norm_node_p or "hiring" in norm_node_p or "leadership" in norm_node_p) and "funding" in cat_str:
                return False
            if "technology" in norm_node_p and ("hiring" in cat_str or "organization" in cat_str or "funding" in cat_str):
                return False
            return False

        claim_text = str(fact.get("claim_text", "")).lower()
        if not claim_text:
            return True
        if any(w in claim_text for w in ("funding", "raised", "series a", "series b", "seed round")) and ("hiring" in norm_node_p or "leadership" in norm_node_p or "headcount" in norm_node_p or "employee" in norm_node_p or "technology" in norm_node_p):
            return False
        if any(w in claim_text for w in ("hired", "recruited", "appointed", "vp", "chief", "joined as")) and ("funding" in norm_node_p or "headcount" in norm_node_p or "employee" in norm_node_p or "technology" in norm_node_p):
            return False
        if any(w in claim_text for w in ("headcount", "employees", "workforce", "team size", "staff")) and ("funding" in norm_node_p or "technology" in norm_node_p):
            return False
        return True

    # 2. If fact is a CanonicalFact or object
    mol_facts = getattr(fact, "molecular_facts", ())
    if mol_facts:
        for mf in mol_facts:
            mf_norm = getattr(mf, "normalized_predicate", None)
            if mf_norm:
                norm_mf_p = normalize_predicate_token(mf_norm.value if hasattr(mf_norm, "value") else str(mf_norm))
                if norm_node_p == norm_mf_p:
                    return True
                if norm_node_p in ("leadership_appointed", "hiring_executed", "hiring_planned") and norm_mf_p in ("leadership_appointed", "hiring_executed", "hiring_planned"):
                    return True
            raw_mf_p = getattr(mf, "raw_predicate", None)
            if raw_mf_p and normalize_predicate_token(str(raw_mf_p)) == norm_node_p:
                return True
        return False

    cat = getattr(fact, "fact_category", None)
    if cat:
        cat_val = str(cat.value if hasattr(cat, "value") else cat).lower()
        if "funding" in norm_node_p and "funding" in cat_val:
            return True
        if ("hiring" in norm_node_p or "leadership" in norm_node_p or "employee" in norm_node_p) and ("hiring" in cat_val or "organization" in cat_val):
            return True
        if "product" in norm_node_p and "product" in cat_val:
            return True
        # Explicit category rejections
        if "funding" in norm_node_p and "funding" not in cat_val:
            return False
        if ("employee" in norm_node_p or "hiring" in norm_node_p or "leadership" in norm_node_p) and "funding" in cat_val:
            return False
        if "technology" in norm_node_p and ("hiring" in cat_val or "organization" in cat_val or "funding" in cat_val):
            return False

    fact_p = getattr(fact, "predicate", None) or getattr(fact, "requested_predicate", None)
    if fact_p:
        norm_fp = normalize_predicate_token(str(fact_p))
        if norm_node_p == norm_fp:
            return True
        if norm_node_p in ("leadership_appointed", "hiring_executed", "hiring_planned") and norm_fp in ("leadership_appointed", "hiring_executed", "hiring_planned", "hiring"):
            return True
        return False

    claim_text = getattr(fact, "claim_text", "")
    if claim_text:
        claim_lower = claim_text.lower()
        if any(w in claim_lower for w in ("funding", "raised", "series a", "series b", "seed round")) and ("hiring" in norm_node_p or "leadership" in norm_node_p or "headcount" in norm_node_p or "employee" in norm_node_p or "technology" in norm_node_p):
            return False
        if any(w in claim_lower for w in ("hired", "recruited", "appointed", "vp", "chief", "joined as")) and ("funding" in norm_node_p or "headcount" in norm_node_p or "employee" in norm_node_p or "technology" in norm_node_p):
            return False
        if any(w in claim_lower for w in ("headcount", "employees", "workforce", "team size", "staff")) and ("funding" in norm_node_p or "technology" in norm_node_p):
            return False

    return True


def is_fact_attributable_to_node(
    fact: Any,
    node: EpistemicQuestionNode,
    node_query_texts: Optional[Sequence[str]] = None,
) -> bool:
    """Determine whether a fact is attributable to a specific epistemic question node.

    Enforces predicate compatibility, target entity attribution, and prevents cross-question outcome contamination.
    """
    if not fact_matches_predicate(node.requested_predicate, fact):
        return False

    # Ambiguity preservation: facts marked with ambiguous entity resolution must not resolve node
    if getattr(fact, "entity_resolution_status", None) == "ambiguous" or getattr(fact, "ambiguous_candidate_ids", None):
        return False

    # Entity ID check
    if node.target_entity_id:
        fact_entity_ids = getattr(fact, "entity_ids", []) or []
        if fact_entity_ids and node.target_entity_id not in fact_entity_ids:
            mol_facts = getattr(fact, "molecular_facts", ())
            found = any(
                getattr(mf, "subject_entity_id", None) == node.target_entity_id
                or getattr(mf, "object_entity_id", None) == node.target_entity_id
                for mf in mol_facts
            )
            if not found:
                return False

    return True


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
        matching_facts = [
            f for f in facts_found
            if fact_matches_predicate(node.requested_predicate, f)
        ]

        if matching_facts:
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
                        f"Resolved with {len(matching_facts)} corroborated facts matching requested predicate '{node.requested_predicate}' across {len(sources_found)} sources."
                    ),
                    details={"facts_count": len(matching_facts), "sources_count": len(sources_found), "predicate": node.requested_predicate},
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
                    f"No facts matching predicate '{node.requested_predicate}' were established within the examined scope. "
                    "This does NOT imply the fact is false or absent globally."
                ),
                details={
                    "executed_queries": executed_queries,
                    "examined_sources_count": len(sources_found),
                    "temporal_scope": node.temporal_scope or "unspecified",
                    "remaining_uncertainty": "Uncorroborated within examined web scope",
                    "predicate": node.requested_predicate,
                },
            )

        # Otherwise unresolved but eligible for further turns if budget permits
        return StoppingDecision(
            should_stop=False,
            reason=None,
            new_status=QuestionLifecycleStatus.UNRESOLVED,
            rationale=f"No facts matching predicate '{node.requested_predicate}' discovered in current turn, but query scope not yet fully exhausted.",
            details={"executed_queries": executed_queries, "predicate": node.requested_predicate},
        )
