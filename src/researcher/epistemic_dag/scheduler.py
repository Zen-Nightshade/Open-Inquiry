from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from .budget import GlobalResearchBudget
from .models import (
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    QuestionLifecycleStatus,
    StoppingReason,
    VoIAction,
)
from .stopping import StoppingEvaluator
from .voi import ValueOfInformationEvaluator

logger = logging.getLogger(__name__)


class EpistemicDAGScheduler:
    """Coordinates question selection, dependency evaluation, VoI prioritization, and stopping decisions.
    
    Guarantees:
      - Independent questions are scheduled without waiting for unrelated branches.
      - Dependent questions remain BLOCKED until prerequisites resolve.
      - VoI prioritization determines scheduling order among eligible READY questions.
      - Pre-search stopping halts unobservable facts and low-VoI questions before consuming queries.
      - Post-search evaluation updates node statuses and unblocks or blocks downstream dependents.
      - Does not duplicate search/fetch pipelines; coordinates existing researcher infrastructure.
    """

    @classmethod
    def select_next_batch(
        cls,
        dag: EpistemicQuestionDAG,
        budget: GlobalResearchBudget,
        max_batch_size: int = 3,
    ) -> Tuple[List[EpistemicQuestionNode], EpistemicQuestionDAG]:
        """Select the highest-VoI ready questions eligible for execution under remaining budget.
        
        Returns:
            Tuple of (selected_nodes_to_execute, updated_dag).
        """
        # 1. Ensure dependency state is fresh
        current_dag = dag.reevaluate_dependencies()

        # 2. Inspect all READY nodes
        ready_nodes = current_dag.get_ready_nodes()
        if not ready_nodes:
            logger.debug("No READY nodes available in DAG %s", current_dag.dag_id)
            return ([], current_dag)

        candidates: List[Tuple[float, EpistemicQuestionNode]] = []
        deferred_candidates: List[Tuple[float, EpistemicQuestionNode]] = []

        for node in ready_nodes:
            # Check pre-search stopping rules (Unobservable, Budget, Missing Dep)
            stop_dec = StoppingEvaluator.evaluate_pre_search(node, current_dag, budget)
            if stop_dec.should_stop:
                logger.info(
                    "Question %s stopped pre-search: reason=%s, status=%s",
                    node.question_id,
                    stop_dec.reason.value if stop_dec.reason else None,
                    stop_dec.new_status.value if stop_dec.new_status else None,
                )
                current_dag = current_dag.transition_node(
                    question_id=node.question_id,
                    new_status=stop_dec.new_status or QuestionLifecycleStatus.SKIPPED,
                    stop_reason=stop_dec.reason,
                    stop_rationale=stop_dec.rationale,
                )
                continue

            # Evaluate Value of Information
            voi_ass = ValueOfInformationEvaluator.evaluate(node, current_dag, budget)
            # Update node's latest VoI assessment
            current_dag = current_dag.transition_node(
                question_id=node.question_id,
                new_status=QuestionLifecycleStatus.READY,
                voi_assessment=voi_ass,
            )
            updated_node = current_dag.get_node(node.question_id)

            if voi_ass.action == VoIAction.RESEARCH:
                candidates.append((voi_ass.voi_score, updated_node))
            elif voi_ass.action == VoIAction.DEFER:
                logger.debug("Question %s deferred by VoI evaluator: %s", node.question_id, voi_ass.explanation)
                deferred_candidates.append((voi_ass.voi_score, updated_node))
            elif voi_ass.action == VoIAction.STOP:
                logger.info("Question %s stopped by VoI evaluator: %s", node.question_id, voi_ass.explanation)
                current_dag = current_dag.transition_node(
                    question_id=node.question_id,
                    new_status=QuestionLifecycleStatus.SKIPPED,
                    stop_reason=StoppingReason.LOW_VOI,
                    stop_rationale=voi_ass.explanation,
                )

        if not candidates and deferred_candidates:
            candidates = deferred_candidates

        if not candidates:
            return ([], current_dag)

        # Sort candidates: highest VoI score first, then priority ascending (1 highest)
        candidates.sort(key=lambda item: (-item[0], item[1].priority))

        # Select batch respecting budget limits and batch size
        selected_nodes: List[EpistemicQuestionNode] = []
        running_cost = 0

        for voi_score, cand_node in candidates:
            cost = cand_node.estimated_research_cost
            if (running_cost + cost) <= budget.remaining_total_units and len(selected_nodes) < max_batch_size:
                selected_nodes.append(cand_node)
                running_cost += cost
                # Mark as IN_PROGRESS
                current_dag = current_dag.transition_node(
                    question_id=cand_node.question_id,
                    new_status=QuestionLifecycleStatus.IN_PROGRESS,
                )

        return (selected_nodes, current_dag)

    @classmethod
    def should_terminate_research(
        cls,
        dag: EpistemicQuestionDAG,
        budget: Optional[GlobalResearchBudget] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether DAG-governed research has reached an authoritative stopping condition."""
        if budget is not None and budget.is_exhausted:
            return (True, "Global research budget exhausted")

        all_nodes = list(dag.nodes.values())
        if not all_nodes:
            return (True, "DAG contains no question nodes")

        terminal_statuses = {
            QuestionLifecycleStatus.RESOLVED,
            QuestionLifecycleStatus.UNOBSERVABLE,
            QuestionLifecycleStatus.EXHAUSTED,
            QuestionLifecycleStatus.SKIPPED,
        }

        # 1. All nodes are terminal
        if all(node.status in terminal_statuses for node in all_nodes):
            if all(node.status == QuestionLifecycleStatus.UNOBSERVABLE for node in all_nodes):
                return (True, "All questions determined unobservable")
            if all(node.status == QuestionLifecycleStatus.RESOLVED for node in all_nodes):
                return (True, "All questions resolved")
            return (True, "All questions reached terminal states")

        # 2. Check for deadlock / dependency blocks
        non_terminal = [n for n in all_nodes if n.status not in terminal_statuses]
        executable_candidates = []
        for n in non_terminal:
            unmet_deps = [
                dep for dep in n.dependencies
                if dep in dag.nodes and dag.nodes[dep].status != QuestionLifecycleStatus.RESOLVED
            ]
            if not unmet_deps:
                executable_candidates.append(n)
            else:
                # If any dependency is terminal and unresolvable (e.g. unobservable/depleted/skipped),
                # this child question can never execute.
                unresolvable_parents = [
                    dep for dep in unmet_deps
                    if dag.nodes[dep].status in (
                        QuestionLifecycleStatus.UNOBSERVABLE,
                        QuestionLifecycleStatus.EXHAUSTED,
                        QuestionLifecycleStatus.SKIPPED,
                    )
                ]
                # If parent will never resolve, child cannot execute
                if len(unresolvable_parents) == len(unmet_deps):
                    pass

        in_progress = [n for n in all_nodes if n.status == QuestionLifecycleStatus.IN_PROGRESS]
        if not executable_candidates and not in_progress:
            # Check if remaining non-terminal nodes are permanently blocked
            return (True, "All remaining questions dependency blocked or unresolvable")

        return (False, None)

    @classmethod
    def record_node_outcome(
        cls,
        dag: EpistemicQuestionDAG,
        question_id: str,
        facts_found: List[Any],
        sources_found: List[Any],
        executed_queries: List[str],
        had_provider_failure: bool = False,
    ) -> EpistemicQuestionDAG:
        """Process findings for a question, apply post-search stopping, and update DAG dependencies."""
        node = dag.get_node(question_id)

        stop_decision = StoppingEvaluator.evaluate_post_search(
            node=node,
            facts_found=facts_found,
            sources_found=sources_found,
            executed_queries=executed_queries,
            had_provider_failure=had_provider_failure,
        )

        fact_ids = [getattr(f, "fact_id", str(f)) for f in facts_found]
        source_ids = [getattr(s, "source_id", str(s)) for s in sources_found]

        target_status = stop_decision.new_status or QuestionLifecycleStatus.UNRESOLVED

        updated_dag = dag.transition_node(
            question_id=question_id,
            new_status=target_status,
            stop_reason=stop_decision.reason,
            stop_rationale=stop_decision.rationale,
            resulting_fact_ids=fact_ids,
            resulting_source_ids=source_ids,
            executed_queries=executed_queries,
        )

        logger.info(
            "Question %s outcome recorded: status=%s, stop_reason=%s, facts=%d",
            question_id,
            target_status.value,
            stop_decision.reason.value if stop_decision.reason else None,
            len(fact_ids),
        )

        return updated_dag

    @classmethod
    def serialize_dag_summary(cls, dag: EpistemicQuestionDAG) -> Dict[str, Any]:
        """Produce an auditable JSON-compatible snapshot summary of the DAG."""
        nodes_summary = []
        counts: Dict[str, int] = {st.value: 0 for st in QuestionLifecycleStatus}

        for q_id in dag.get_topological_order():
            node = dag.nodes[q_id]
            counts[node.status.value] = counts.get(node.status.value, 0) + 1
            nodes_summary.append({
                "question_id": node.question_id,
                "predicate": node.requested_predicate,
                "neutral_question": node.neutral_question,
                "status": node.status.value,
                "observability": node.observability.value,
                "stop_reason": node.stop_reason.value if node.stop_reason else None,
                "stop_rationale": node.stop_rationale,
                "voi_score": node.voi_assessment.voi_score if node.voi_assessment else None,
                "voi_action": node.voi_assessment.action.value if node.voi_assessment else None,
                "dependencies": list(node.dependencies),
                "facts_count": len(node.resulting_fact_ids),
                "queries_executed": list(node.executed_queries),
                "provenance": node.provenance,
            })

        return {
            "dag_id": dag.dag_id,
            "run_id": dag.run_id,
            "node_count": len(dag.nodes),
            "status_counts": counts,
            "nodes": nodes_summary,
        }
