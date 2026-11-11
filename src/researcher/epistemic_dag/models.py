from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ObservabilityClass(str, Enum):
    """Categorization of how accessible a factual predicate is via public channels."""

    PUBLICLY_OBSERVABLE = "publicly_observable"
    CONDITIONALLY_OBSERVABLE = "conditionally_observable"
    PRIVATE_OR_NONPUBLIC = "private_or_nonpublic"
    UNKNOWN = "unknown"


class QuestionLifecycleStatus(str, Enum):
    """Lifecycle states of an Epistemic Question Node."""

    PENDING = "pending"
    BLOCKED = "blocked"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"
    EXHAUSTED = "exhausted"
    UNOBSERVABLE = "unobservable"
    SKIPPED = "skipped"
    FAILED = "failed"


class StoppingReason(str, Enum):
    """Explicit, auditable reason for discontinuing or completing investigation."""

    RESOLVED = "resolved"
    UNOBSERVABLE = "unobservable"
    EXHAUSTED_SCOPE = "exhausted_scope"
    LOW_VOI = "low_voi"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    OPERATIONAL_FAILURE = "operational_failure"


class DecisionRelevance(str, Enum):
    """Ordinal decision impact if the question is resolved."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def normalized_weight(self) -> float:
        """Normalized weight in [0.0, 1.0]."""
        weights = {
            DecisionRelevance.CRITICAL: 1.0,
            DecisionRelevance.HIGH: 0.75,
            DecisionRelevance.MEDIUM: 0.40,
            DecisionRelevance.LOW: 0.15,
        }
        return weights[self]


class ResolutionLikelihood(str, Enum):
    """Estimated probability band of resolving the question through public channels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

    @property
    def normalized_probability(self) -> float:
        """Normalized probability estimate in [0.0, 1.0]."""
        probs = {
            ResolutionLikelihood.HIGH: 1.0,
            ResolutionLikelihood.MEDIUM: 0.60,
            ResolutionLikelihood.LOW: 0.25,
            ResolutionLikelihood.NEGLIGIBLE: 0.05,
        }
        return probs[self]


class VoIAction(str, Enum):
    """Recommended action from Value of Information evaluation."""

    RESEARCH = "research"
    DEFER = "defer"
    STOP = "stop"


class ResolutionCriterion(BaseModel):
    """Structured evidence criterion for resolving a factual question."""

    model_config = ConfigDict(frozen=True)

    criterion_type: str = Field(
        description="Type of criterion: e.g. 'primary_source', 'corroboration', 'temporal_currency', 'identity_attribution'"
    )
    description: str = Field(description="Human-readable description of the required evidence")
    is_mandatory: bool = Field(default=True, description="Whether this criterion is strictly required to mark RESOLVED")
    min_credibility: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum credibility required of satisfying evidence")
    min_sources: int = Field(default=1, ge=1, description="Minimum independent sources required (default 1 permits strong primary sources)")


class VoIAssessment(BaseModel):
    """Evaluation of the Value of Information for investigating a question."""

    model_config = ConfigDict(frozen=True)

    action: VoIAction
    voi_score: float = Field(description="Bounded VoI score (higher means greater net research value)")
    decision_relevance: DecisionRelevance
    resolution_likelihood: ResolutionLikelihood
    dependency_benefit: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized benefit from unlocking downstream questions")
    cost_penalty: float = Field(default=0.0, ge=0.0, description="Normalized penalty reflecting query budget cost")
    estimated_cost_units: int = Field(default=1, ge=1, description="Estimated search queries needed")
    explanation: str = Field(default="", description="Auditable rationale for the VoI action and score")


VALID_LIFECYCLE_TRANSITIONS: Dict[QuestionLifecycleStatus, Set[QuestionLifecycleStatus]] = {
    QuestionLifecycleStatus.PENDING: {
        QuestionLifecycleStatus.READY,
        QuestionLifecycleStatus.BLOCKED,
        QuestionLifecycleStatus.SKIPPED,
        QuestionLifecycleStatus.UNOBSERVABLE,
        QuestionLifecycleStatus.FAILED,
    },
    QuestionLifecycleStatus.BLOCKED: {
        QuestionLifecycleStatus.READY,
        QuestionLifecycleStatus.SKIPPED,
        QuestionLifecycleStatus.FAILED,
    },
    QuestionLifecycleStatus.READY: {
        QuestionLifecycleStatus.IN_PROGRESS,
        QuestionLifecycleStatus.SKIPPED,
        QuestionLifecycleStatus.BLOCKED,
        QuestionLifecycleStatus.UNOBSERVABLE,
    },
    QuestionLifecycleStatus.IN_PROGRESS: {
        QuestionLifecycleStatus.RESOLVED,
        QuestionLifecycleStatus.PARTIALLY_RESOLVED,
        QuestionLifecycleStatus.UNRESOLVED,
        QuestionLifecycleStatus.EXHAUSTED,
        QuestionLifecycleStatus.UNOBSERVABLE,
        QuestionLifecycleStatus.FAILED,
    },
    QuestionLifecycleStatus.PARTIALLY_RESOLVED: {
        QuestionLifecycleStatus.IN_PROGRESS,
        QuestionLifecycleStatus.RESOLVED,
        QuestionLifecycleStatus.EXHAUSTED,
        QuestionLifecycleStatus.UNRESOLVED,
        QuestionLifecycleStatus.FAILED,
    },
    QuestionLifecycleStatus.UNRESOLVED: {
        QuestionLifecycleStatus.IN_PROGRESS,
        QuestionLifecycleStatus.EXHAUSTED,
    },
    # Terminal states (reopening permitted only via explicit reopen API)
    QuestionLifecycleStatus.RESOLVED: set(),
    QuestionLifecycleStatus.EXHAUSTED: set(),
    QuestionLifecycleStatus.UNOBSERVABLE: set(),
    QuestionLifecycleStatus.SKIPPED: set(),
    QuestionLifecycleStatus.FAILED: {
        # Allow retry of an operationally failed question
        QuestionLifecycleStatus.READY,
        QuestionLifecycleStatus.IN_PROGRESS,
    },
}


class EpistemicQuestionNode(BaseModel):
    """An independently investigable factual question within the Epistemic DAG."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(default_factory=lambda: f"qn_{uuid.uuid4().hex[:12]}")
    run_id: str = Field(description="Associated application run ID")
    target_entity_id: Optional[str] = Field(default=None, description="Target entity ID when resolved")
    target_entity_name: Optional[str] = Field(default=None, description="Human-readable entity name")
    requested_predicate: str = Field(description="Controlled predicate, e.g. 'works_for', 'has_initiative', 'uses_vendor'")
    neutral_question: str = Field(description="Neutral, hypothesis-blind factual question")
    temporal_scope: Optional[str] = Field(default=None, description="Coarse temporal restriction, e.g. 'recent', 'career_history'")
    valid_from: Optional[str] = Field(default=None, description="ISO date lower bound")
    valid_to: Optional[str] = Field(default=None, description="ISO date upper bound")
    question_category: str = Field(default="neutral_discovery", description="Category or objective type")
    observability: ObservabilityClass = Field(default=ObservabilityClass.UNKNOWN)
    observability_rationale: str = Field(default="", description="Justification for the observability classification")
    priority: int = Field(default=5, ge=1, le=10, description="Priority ranking from 1 (highest) to 10 (lowest)")
    decision_relevance: DecisionRelevance = Field(default=DecisionRelevance.MEDIUM)
    estimated_research_cost: int = Field(default=1, ge=1, description="Estimated search queries needed")
    dependencies: List[str] = Field(default_factory=list, description="IDs of prerequisite questions that must resolve first")
    supporting_search_directions: List[str] = Field(default_factory=list, description="Search queries seeking supporting evidence")
    disconfirming_search_directions: List[str] = Field(default_factory=list, description="Search queries seeking disconfirming evidence")
    required_resolution_criteria: List[ResolutionCriterion] = Field(default_factory=list)
    status: QuestionLifecycleStatus = Field(default=QuestionLifecycleStatus.PENDING)
    stop_reason: Optional[StoppingReason] = Field(default=None)
    stop_rationale: Optional[str] = Field(default=None)
    provenance: str = Field(default="", description="Audit trail explaining why this question was created")
    resulting_fact_ids: List[str] = Field(default_factory=list, description="IDs of extracted facts addressing this question")
    resulting_source_ids: List[str] = Field(default_factory=list, description="IDs of sources fetched during investigation")
    executed_queries: List[str] = Field(default_factory=list, description="Exact search queries executed")
    voi_assessment: Optional[VoIAssessment] = Field(default=None)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(default=None)
    reopen_history: List[Dict[str, Any]] = Field(default_factory=list, description="Audit log of explicit reopen events")

    @model_validator(mode="after")
    def validate_temporal_bounds(self) -> EpistemicQuestionNode:
        """Ensure temporal bounds are logically consistent."""
        if self.valid_from and self.valid_to:
            if self.valid_from > self.valid_to:
                raise ValueError(f"valid_from '{self.valid_from}' cannot be after valid_to '{self.valid_to}'.")
        return self


class EpistemicQuestionDAG(BaseModel):
    """Directed Acyclic Graph governing epistemic research questions, dependencies, and stopping."""

    model_config = ConfigDict(frozen=True)

    dag_id: str = Field(default_factory=lambda: f"dag_{uuid.uuid4().hex[:12]}")
    run_id: str = Field(description="Application run ID owning this DAG")
    nodes: Dict[str, EpistemicQuestionNode] = Field(default_factory=dict, description="Question nodes keyed by question_id")

    def add_node(self, node: EpistemicQuestionNode) -> EpistemicQuestionDAG:
        """Add a question node to the DAG after verifying ownership and non-duplication."""
        if node.question_id in self.nodes:
            raise ValueError(f"Duplicate question ID '{node.question_id}' in DAG '{self.dag_id}'.")
        if node.run_id != self.run_id:
            raise ValueError(
                f"Cross-run ownership violation: node '{node.question_id}' belongs to run '{node.run_id}', "
                f"not DAG run '{self.run_id}'."
            )
        new_nodes = dict(self.nodes)
        new_nodes[node.question_id] = node
        updated = self.model_copy(update={"nodes": new_nodes})
        updated.validate()
        return updated

    def get_node(self, question_id: str) -> EpistemicQuestionNode:
        """Retrieve node by ID or raise KeyError."""
        if question_id not in self.nodes:
            raise KeyError(f"Question node '{question_id}' not found in DAG '{self.dag_id}'.")
        return self.nodes[question_id]

    def validate(self) -> None:
        """Deterministic validation of graph structure: dependencies, self-loops, and cycles."""
        # 1. Dependency resolution and self-dependency check
        for q_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id == q_id:
                    raise ValueError(f"Self-dependency detected: question '{q_id}' depends on itself.")
                if dep_id not in self.nodes:
                    raise ValueError(
                        f"Missing dependency: question '{q_id}' depends on non-existent question '{dep_id}'."
                    )
                dep_node = self.nodes[dep_id]
                if dep_node.run_id != self.run_id:
                    raise ValueError(
                        f"Cross-run dependency: question '{q_id}' (run '{self.run_id}') depends on "
                        f"'{dep_id}' from run '{dep_node.run_id}'."
                    )

        # 2. Cycle detection via Kahn's algorithm (topological sort)
        in_degree: Dict[str, int] = {q_id: 0 for q_id in self.nodes}
        # Adjacency list: dep_id -> list of questions that depend on dep_id
        dependents: Dict[str, List[str]] = {q_id: [] for q_id in self.nodes}
        for q_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                in_degree[q_id] += 1
                dependents[dep_id].append(q_id)

        queue: List[str] = [q_id for q_id, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in dependents[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited_count != len(self.nodes):
            unresolved = [q_id for q_id, deg in in_degree.items() if deg > 0]
            raise ValueError(f"Cycle detected in EpistemicQuestionDAG: participating nodes {unresolved}")

    def get_topological_order(self) -> List[str]:
        """Return topological ordering of question IDs (prerequisites before dependents)."""
        self.validate()
        in_degree: Dict[str, int] = {q_id: 0 for q_id in self.nodes}
        dependents: Dict[str, List[str]] = {q_id: [] for q_id in self.nodes}
        for q_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                in_degree[q_id] += 1
                dependents[dep_id].append(q_id)

        # Sort queue by priority ascending (1 highest) for deterministic evaluation order
        queue: List[str] = sorted(
            [q_id for q_id, deg in in_degree.items() if deg == 0],
            key=lambda x: self.nodes[x].priority,
        )
        ordered: List[str] = []

        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for nxt in sorted(dependents[curr], key=lambda x: self.nodes[x].priority):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return ordered

    def get_ready_nodes(self) -> List[EpistemicQuestionNode]:
        """Return all nodes currently in READY status, sorted by priority (1 highest)."""
        ready = [node for node in self.nodes.values() if node.status == QuestionLifecycleStatus.READY]
        return sorted(ready, key=lambda n: n.priority)

    def get_blocked_nodes(self) -> List[EpistemicQuestionNode]:
        """Return all nodes currently in BLOCKED status."""
        return [node for node in self.nodes.values() if node.status == QuestionLifecycleStatus.BLOCKED]

    @property
    def is_fully_resolved(self) -> bool:
        """Return True if all question nodes in the DAG are RESOLVED."""
        return len(self.nodes) > 0 and all(
            node.status == QuestionLifecycleStatus.RESOLVED for node in self.nodes.values()
        )

    def transition_node(
        self,
        question_id: str,
        new_status: QuestionLifecycleStatus,
        stop_reason: Optional[StoppingReason] = None,
        stop_rationale: Optional[str] = None,
        resulting_fact_ids: Optional[List[str]] = None,
        resulting_source_ids: Optional[List[str]] = None,
        executed_queries: Optional[List[str]] = None,
        voi_assessment: Optional[VoIAssessment] = None,
    ) -> EpistemicQuestionDAG:
        """Apply a validated lifecycle transition to a node and re-evaluate dependent nodes."""
        node = self.get_node(question_id)
        current_status = node.status

        # If already at target status, return self (idempotent)
        if current_status == new_status:
            return self

        valid_targets = VALID_LIFECYCLE_TRANSITIONS.get(current_status, set())
        if new_status not in valid_targets:
            raise ValueError(
                f"Invalid lifecycle transition for question '{question_id}': "
                f"cannot transition from {current_status.value} to {new_status.value}."
            )

        updates: Dict[str, Any] = {"status": new_status}
        if stop_reason is not None:
            updates["stop_reason"] = stop_reason
        if stop_rationale is not None:
            updates["stop_rationale"] = stop_rationale
        if resulting_fact_ids is not None:
            combined_facts = sorted(list(set(node.resulting_fact_ids + resulting_fact_ids)))
            updates["resulting_fact_ids"] = combined_facts
        if resulting_source_ids is not None:
            combined_sources = sorted(list(set(node.resulting_source_ids + resulting_source_ids)))
            updates["resulting_source_ids"] = combined_sources
        if executed_queries is not None:
            combined_queries = sorted(list(set(node.executed_queries + executed_queries)))
            updates["executed_queries"] = combined_queries
        if voi_assessment is not None:
            updates["voi_assessment"] = voi_assessment

        if new_status in (
            QuestionLifecycleStatus.RESOLVED,
            QuestionLifecycleStatus.EXHAUSTED,
            QuestionLifecycleStatus.UNOBSERVABLE,
            QuestionLifecycleStatus.SKIPPED,
            QuestionLifecycleStatus.FAILED,
        ):
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()

        updated_node = node.model_copy(update=updates)
        new_nodes = dict(self.nodes)
        new_nodes[question_id] = updated_node

        dag_with_updated_node = self.model_copy(update={"nodes": new_nodes})
        # Reevaluate dependent nodes across the DAG
        return dag_with_updated_node.reevaluate_dependencies()

    def reopen_node(
        self,
        question_id: str,
        reason: str,
        provenance: str,
    ) -> EpistemicQuestionDAG:
        """Explicitly reopen a terminal node (e.g. upon discovery of material contradiction or new temporal window)."""
        node = self.get_node(question_id)
        if node.status not in (
            QuestionLifecycleStatus.RESOLVED,
            QuestionLifecycleStatus.EXHAUSTED,
            QuestionLifecycleStatus.UNRESOLVED,
            QuestionLifecycleStatus.PARTIALLY_RESOLVED,
            QuestionLifecycleStatus.SKIPPED,
            QuestionLifecycleStatus.FAILED,
        ):
            raise ValueError(f"Question '{question_id}' in status {node.status.value} does not require reopening.")

        entry = {
            "reopened_at": datetime.now(timezone.utc).isoformat(),
            "prior_status": node.status.value,
            "prior_stop_reason": node.stop_reason.value if node.stop_reason else None,
            "reason": reason,
            "provenance": provenance,
        }
        reopen_history = list(node.reopen_history) + [entry]

        # Reset to READY or PENDING depending on prerequisites
        reopened_node = node.model_copy(
            update={
                "status": QuestionLifecycleStatus.PENDING,
                "stop_reason": None,
                "stop_rationale": None,
                "completed_at": None,
                "reopen_history": reopen_history,
            }
        )
        new_nodes = dict(self.nodes)
        new_nodes[question_id] = reopened_node
        dag_updated = self.model_copy(update={"nodes": new_nodes})
        return dag_updated.reevaluate_dependencies()

    def reevaluate_dependencies(self) -> EpistemicQuestionDAG:
        """Reevaluate PENDING and BLOCKED nodes against prerequisite resolution states."""
        topo_order = self.get_topological_order()
        new_nodes = dict(self.nodes)
        mutated = False

        for q_id in topo_order:
            node = new_nodes[q_id]

            # Only re-evaluate non-terminal nodes that are waiting or blocked
            if node.status in (QuestionLifecycleStatus.PENDING, QuestionLifecycleStatus.BLOCKED):
                if not node.dependencies:
                    # Independent branch: directly READY
                    if node.status != QuestionLifecycleStatus.READY:
                        new_nodes[q_id] = node.model_copy(update={"status": QuestionLifecycleStatus.READY})
                        mutated = True
                else:
                    # Check status of prerequisites
                    all_resolved = True
                    has_permanent_failure = False
                    blocking_reasons: List[str] = []

                    for dep_id in node.dependencies:
                        dep_node = new_nodes[dep_id]
                        if dep_node.status == QuestionLifecycleStatus.RESOLVED:
                            continue
                        elif dep_node.status in (
                            QuestionLifecycleStatus.UNOBSERVABLE,
                            QuestionLifecycleStatus.EXHAUSTED,
                            QuestionLifecycleStatus.SKIPPED,
                        ):
                            all_resolved = False
                            has_permanent_failure = True
                            blocking_reasons.append(f"Prerequisite '{dep_id}' ended in {dep_node.status.value}")
                        elif dep_node.status == QuestionLifecycleStatus.FAILED:
                            all_resolved = False
                            blocking_reasons.append(f"Prerequisite '{dep_id}' operationally failed")
                        else:
                            all_resolved = False
                            blocking_reasons.append(f"Prerequisite '{dep_id}' is {dep_node.status.value}")

                    if all_resolved:
                        if node.status != QuestionLifecycleStatus.READY:
                            new_nodes[q_id] = node.model_copy(
                                update={
                                    "status": QuestionLifecycleStatus.READY,
                                    "stop_reason": None,
                                    "stop_rationale": None,
                                }
                            )
                            mutated = True
                    else:
                        rationale = "; ".join(blocking_reasons)
                        if node.status != QuestionLifecycleStatus.BLOCKED or node.stop_rationale != rationale:
                            new_nodes[q_id] = node.model_copy(
                                update={
                                    "status": QuestionLifecycleStatus.BLOCKED,
                                    "stop_reason": StoppingReason.DEPENDENCY_BLOCKED if has_permanent_failure else None,
                                    "stop_rationale": rationale,
                                }
                            )
                            mutated = True

        if mutated:
            return self.model_copy(update={"nodes": new_nodes})
        return self
