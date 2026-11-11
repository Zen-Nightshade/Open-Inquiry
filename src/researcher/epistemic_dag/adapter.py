from __future__ import annotations

import re
from typing import Dict, List, Optional
from src.researcher.planning.models import (
    ResearchObjective,
    ResearchObjectiveType,
    TargetEntity,
)
from .models import (
    DecisionRelevance,
    EpistemicQuestionDAG,
    EpistemicQuestionNode,
    ObservabilityClass,
    QuestionLifecycleStatus,
    ResolutionCriterion,
)
from .observability import ObservabilityClassifier


class DAGAdapter:
    """Backward-compatible adapter converting ResearchObjective and EvidenceRequest objects to DAG nodes.
    
    Guarantees:
      - Preserves original objective intent, metadata, and provenance.
      - Does NOT mechanically map objective types to commercial assumptions.
      - Company-level public facts are NOT blocked by unresolved prospect identity.
      - Person-specific role/employment questions depend on identity resolution when appropriate.
      - Converts M3.1 EvidenceRequest objects preserving IDs, hypothesis-blind wording, and budget units.
    """

    @classmethod
    def convert_objectives_to_dag(
        cls,
        objectives: List[ResearchObjective],
        run_id: str,
        prospect_name: Optional[str] = None,
        company_name: Optional[str] = None,
    ) -> EpistemicQuestionDAG:
        """Construct a validated EpistemicQuestionDAG from a list of ResearchObjective objects."""
        dag = EpistemicQuestionDAG(run_id=run_id)

        # First pass: identify any identity resolution objective to serve as a legitimate dependency
        # for person-specific initiative or authority questions.
        identity_node_id: Optional[str] = None
        for obj in objectives:
            if obj.type == ResearchObjectiveType.IDENTITY_RESOLUTION or (
                obj.target_entity == TargetEntity.PROSPECT and "identity" in obj.description.lower()
            ):
                identity_node_id = f"qn_{obj.objective_id.replace('-', '')[:10]}"
                break

        for obj in objectives:
            node_id = f"qn_{obj.objective_id.replace('-', '')[:10]}"

            # Determine neutral predicate and question from original intent
            predicate, neutral_q = cls._derive_predicate_and_question(
                obj=obj,
                prospect_name=prospect_name,
                company_name=company_name,
            )

            # Classify observability conservatively
            obs_class, obs_rationale, proxy_info = ObservabilityClassifier.classify(
                predicate=predicate,
                question_text=neutral_q,
            )

            # Determine decision relevance
            if obj.is_critical:
                decision_rel = DecisionRelevance.CRITICAL
            elif obj.priority == 1:
                decision_rel = DecisionRelevance.HIGH
            else:
                decision_rel = DecisionRelevance.MEDIUM

            # Determine explicit dependencies
            dependencies: List[str] = []
            # Only person-specific questions (not company questions) depend on identity resolution
            if obj.target_entity == TargetEntity.PROSPECT:
                if identity_node_id and node_id != identity_node_id:
                    dependencies.append(identity_node_id)
            # Company-level public facts never depend on prospect identity!

            # Structured resolution criteria
            resolution_criteria: List[ResolutionCriterion] = []
            if obj.type == ResearchObjectiveType.IDENTITY_RESOLUTION:
                resolution_criteria.append(
                    ResolutionCriterion(
                        criterion_type="identity_attribution",
                        description="Corroborated current employment and role affiliation with target company",
                        is_mandatory=True,
                        min_credibility=0.7,
                        min_sources=1,
                    )
                )
            else:
                resolution_criteria.append(
                    ResolutionCriterion(
                        criterion_type="factual_corroboration",
                        description=f"Evidence establishing or disconfirming {predicate}",
                        is_mandatory=True,
                        min_credibility=0.65,
                        min_sources=1,
                    )
                )

            node = EpistemicQuestionNode(
                question_id=node_id,
                run_id=run_id,
                target_entity_name=company_name if obj.target_entity == TargetEntity.COMPANY else prospect_name,
                requested_predicate=predicate,
                neutral_question=neutral_q,
                temporal_scope=obj.temporal_scope,
                question_category=obj.type.value,
                observability=obs_class,
                observability_rationale=obs_rationale,
                priority=obj.priority,
                decision_relevance=decision_rel,
                estimated_research_cost=max(1, min(obj.max_queries, 3)),
                dependencies=dependencies,
                supporting_search_directions=[neutral_q],
                disconfirming_search_directions=[],
                required_resolution_criteria=resolution_criteria,
                status=QuestionLifecycleStatus.PENDING,
                provenance=f"research_objective:{obj.objective_id}",
            )
            dag = dag.add_node(node)

        # Validate DAG and reevaluate initial readiness
        dag.validate()
        return dag.reevaluate_dependencies()

    @classmethod
    def convert_evidence_request_to_node(
        cls,
        request: Any,  # EvidenceRequest model
        run_id: str,
    ) -> EpistemicQuestionNode:
        """Convert a single M3.1 EvidenceRequest into an EpistemicQuestionNode."""
        node_id = f"qn_{request.request_id.replace('-', '')[:10]}"

        # Classify observability
        obs_class, obs_rationale, _ = ObservabilityClassifier.classify(
            predicate=request.requested_predicate,
            question_text=request.neutral_question,
        )

        resolution_criteria = [
            ResolutionCriterion(
                criterion_type="gap_resolution",
                description=f"Fact resolving {request.gap_reason.value}: {request.neutral_question}",
                is_mandatory=True,
                min_credibility=0.7,
                min_sources=1,
            )
        ]

        return EpistemicQuestionNode(
            question_id=node_id,
            run_id=run_id,
            target_entity_id=request.target_entity_id,
            target_entity_name=request.target_entity_name,
            requested_predicate=request.requested_predicate,
            neutral_question=request.neutral_question,
            valid_from=request.valid_from,
            valid_to=request.valid_to,
            question_category=f"evidence_request_{request.gap_reason.value}",
            observability=obs_class,
            observability_rationale=obs_rationale,
            priority=1,  # Critical priority because it blocked strategy
            decision_relevance=DecisionRelevance.CRITICAL,
            estimated_research_cost=request.budget_cost_units,
            dependencies=[],
            supporting_search_directions=list(request.neutral_search_terms),
            disconfirming_search_directions=list(request.disconfirming_search_terms),
            required_resolution_criteria=resolution_criteria,
            status=QuestionLifecycleStatus.PENDING,
            provenance=f"evidence_request:{request.request_id}:{request.provenance_stage}",
        )

    @classmethod
    def attach_evidence_requests_to_dag(
        cls,
        dag: EpistemicQuestionDAG,
        requests: List[Any],
        run_id: str,
    ) -> EpistemicQuestionDAG:
        """Attach M3.1 EvidenceRequests as question nodes in the existing DAG."""
        updated_dag = dag
        for req in requests:
            node = cls.convert_evidence_request_to_node(req, run_id=run_id)
            if node.question_id not in updated_dag.nodes:
                updated_dag = updated_dag.add_node(node)
        return updated_dag.reevaluate_dependencies()

    @classmethod
    def _derive_predicate_and_question(
        cls,
        obj: ResearchObjective,
        prospect_name: Optional[str],
        company_name: Optional[str],
    ) -> tuple[str, str]:
        """Derive accurate predicate and neutral question without artificial commercial assumptions."""
        clean_desc = obj.description.strip()

        if obj.type == ResearchObjectiveType.IDENTITY_RESOLUTION:
            predicate = "works_for"
            if prospect_name and company_name:
                neutral_q = f"What is {prospect_name}'s verified role and employment affiliation at {company_name}?"
            else:
                neutral_q = clean_desc
            return (predicate, neutral_q)

        # Derive predicate from description text keywords
        lower_desc = clean_desc.lower()
        if any(w in lower_desc for w in ("funding", "series", "venture", "capital", "seed")):
            predicate = "announced_funding"
        elif any(w in lower_desc for w in ("office", "expand", "headquarter", "location", "facility")):
            predicate = "office_expansion"
        elif any(w in lower_desc for w in ("hiring", "headcount", "recruiting", "open role")):
            predicate = "headcount_growth"
        elif any(w in lower_desc for w in ("vendor", "tool", "stack", "technology", "platform")):
            predicate = "uses_vendor"
        elif any(w in lower_desc for w in ("customer", "deploy", "deployment", "client", "case study", "adoption")):
            predicate = "customer_deployment"
        elif any(w in lower_desc for w in ("challenge", "problem", "bottleneck", "pain")):
            predicate = "operational_challenge"
        elif any(w in lower_desc for w in ("strategy", "initiative", "priority", "objective")):
            predicate = "company_initiatives"
        else:
            # Fallback to normalized slug of first 4 words of description
            words = re.findall(r"[a-z0-9]+", lower_desc)[:4]
            predicate = "_".join(words) if words else "factual_premise"

        return (predicate, clean_desc)
