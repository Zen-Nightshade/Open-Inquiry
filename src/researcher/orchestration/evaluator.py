from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.settings import GROQ_LARGE
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    ResolutionStatus,
    SourceRecord,
)
from src.researcher.planning.models import ResearchObjective
from .models import (
    ActionType,
    ConflictSeverity,
    CoverageStatus,
    DimensionAssessment,
    DimensionAssessments,
    MarginalValueAssessment,
    ObjectiveAssessment,
    PropositionType,
    RecommendedAction,
    ResearchSufficiencyAssessment,
    SearchViability,
    SufficiencyDecision,
    SufficiencyDimension,
)

logger = logging.getLogger(__name__)


# Raw models for LLM output parsing before deterministic policy enforcement
class RawObjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    objective_id: str
    coverage_status: str = Field(description="'adequate', 'partial', or 'unaddressed'")
    viability: str = Field(default="promising", description="'promising', 'uncertain', or 'exhausted'")
    proposition_type: str = Field(default="speech_act_announcement", description="'speech_act_announcement', 'commercial_realization', or 'evaluative_claim'")
    supporting_fact_ids: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)


class RawRecommendedAction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action_type: str = Field(description="'targeted_search', 'identity_verification', 'conflict_investigation', 'terminate_sufficiency', 'terminate_budget'")
    target_objective_id: Optional[str] = None
    gap_description: str
    suggested_focus: str
    rationale: str


class RawSufficiencyEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    objective_assessments: List[RawObjectiveAssessment] = Field(default_factory=list)
    coverage_summary: str = "Objective coverage assessment"
    identity_summary: str = "Identity resolution assessment"
    evidence_summary: str = "Evidence adequacy assessment"
    marginal_value_summary: str = "Marginal research value assessment"
    identity_uncertainties: List[str] = Field(default_factory=list)
    evidence_deficiencies: List[str] = Field(default_factory=list)
    unresolved_conflicts: List[str] = Field(default_factory=list)
    remaining_gaps: List[str] = Field(default_factory=list)
    recommended_actions: List[RawRecommendedAction] = Field(default_factory=list)
    identity_resolution_viable: bool = False
    conflict_resolution_viable: bool = False
    explanation: str = "Sufficiency evaluation completed"

    @field_validator("recommended_actions", mode="before")
    @classmethod
    def normalize_recommended_actions(cls, v: Any) -> Any:
        if isinstance(v, list):
            normalized = []
            for item in v:
                if isinstance(item, str):
                    normalized.append({
                        "action_type": "targeted_search",
                        "gap_description": item,
                        "suggested_focus": item,
                        "rationale": item,
                    })
                elif isinstance(item, dict):
                    normalized.append(item)
            return normalized
        return v


EVALUATOR_SYSTEM_PROMPT = """You are a senior epistemic evaluator assessing research sufficiency for B2B outreach.
Your task is to analyze collected factual evidence against research objectives and determine:
1. Which objectives are adequately addressed, partially addressed, or unaddressed.
2. What specific factual aspects remain missing for partially addressed objectives.
3. Whether remaining gaps are realistically discoverable via further targeted web search (promising, uncertain, exhausted).
4. The semantic nature of the claim proposition:
   - 'speech_act_announcement': public statements, company announcements, press releases, hiring intents.
   - 'commercial_realization': actual production deployment, enterprise customer adoption, verified revenue metrics.
   - 'evaluative_claim': subjective/market standing claims (e.g. 'industry leading', 'best in class').
5. Recommended follow-up actions if continuation is needed.

Important: Cite ONLY valid fact IDs provided in the context. Never fabricate facts or citations.
Output format: You must respond ONLY with a valid JSON object matching the RawSufficiencyEvaluation schema."""


class ResearchSufficiencyEvaluator:
    """Evaluates multi-turn research iterations against objectives, evidence, and identity certainty."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.parser = PydanticParser(RawSufficiencyEvaluation)

    def evaluate(
        self,
        session_id: str,
        iteration_id: str,
        iteration_number: int,
        max_iterations: int,
        objectives: Sequence[ResearchObjective],
        facts: Sequence[CanonicalFact],
        sources: Sequence[SourceRecord],
        prospect_entity: Optional[EntityRecord] = None,
        company_entity: Optional[EntityRecord] = None,
        conflicts: Sequence[Dict[str, Any]] = (),
    ) -> ResearchSufficiencyAssessment:
        """Run LLM semantic evaluation followed by strict deterministic policy enforcement."""

        # 1. Run LLM semantic evaluation
        raw_evaluation = self._run_llm_evaluation(
            objectives=objectives,
            facts=facts,
            sources=sources,
            prospect_entity=prospect_entity,
            company_entity=company_entity,
            conflicts=conflicts,
            iteration_number=iteration_number,
            max_iterations=max_iterations,
        )

        # 2. Grounding validation: filter hallucinated facts and revert ungrounded objectives
        valid_fact_ids = {f.fact_id for f in facts}
        fact_by_id = {f.fact_id: f for f in facts}
        source_by_id = {s.source_id: s for s in sources}
        target_domain = self._extract_target_domain(prospect_entity, company_entity)

        obj_map = {obj.objective_id: obj for obj in objectives}
        grounded_assessments: List[ObjectiveAssessment] = []
        all_remaining_gaps: List[str] = list(raw_evaluation.remaining_gaps)
        extra_recommended_actions: List[RecommendedAction] = []

        # Map raw assessments by objective_id
        raw_obj_by_id = {r.objective_id: r for r in raw_evaluation.objective_assessments}

        for obj in objectives:
            raw_obj = raw_obj_by_id.get(obj.objective_id)
            if not raw_obj:
                # If LLM omitted an objective, treat as unaddressed
                grounded_assessments.append(
                    ObjectiveAssessment(
                        objective_id=obj.objective_id,
                        description=obj.description,
                        coverage_status=CoverageStatus.UNADDRESSED,
                        viability=SearchViability.PROMISING,
                        proposition_type=PropositionType.SPEECH_ACT_ANNOUNCEMENT,
                        is_critical=obj.is_critical,
                        supporting_fact_ids=(),
                        missing_aspects=(obj.description,),
                    )
                )
                continue

            # Grounding check: verify cited fact IDs
            valid_supporting_ids = [fid for fid in raw_obj.supporting_fact_ids if fid in valid_fact_ids]

            # Parse status & viability safely
            status = self._parse_coverage_status(raw_obj.coverage_status)
            viability = self._parse_viability(raw_obj.viability)
            prop_type = self._parse_proposition_type(raw_obj.proposition_type)

            # If claims had no valid supporting facts in ground truth, revert to UNADDRESSED
            if status != CoverageStatus.UNADDRESSED and not valid_supporting_ids:
                if status == CoverageStatus.ADEQUATE:
                    # LLM marked adequate based on hallucinated facts; search is not exhausted
                    viability = SearchViability.PROMISING
                status = CoverageStatus.UNADDRESSED

            # 3. Proposition-relative authority validation
            # If proposition requires commercial realization or evaluative market claim,
            # verify whether sources are exclusively company-owned without third-party corroboration.
            if status == CoverageStatus.ADEQUATE and prop_type in (
                PropositionType.COMMERCIAL_REALIZATION,
                PropositionType.EVALUATIVE_CLAIM,
            ):
                is_exclusively_company_owned = self._check_exclusively_company_owned(
                    supporting_fact_ids=valid_supporting_ids,
                    fact_by_id=fact_by_id,
                    source_by_id=source_by_id,
                    target_domain=target_domain,
                )
                if is_exclusively_company_owned:
                    status = CoverageStatus.PARTIAL
                    viability = SearchViability.PROMISING
                    gap_msg = f"Independent corroboration required for {prop_type.value} on {obj.description}"
                    all_remaining_gaps.append(gap_msg)
                    extra_recommended_actions.append(
                        RecommendedAction(
                            action_type=ActionType.TARGETED_SEARCH,
                            target_objective_id=obj.objective_id,
                            gap_description=gap_msg,
                            suggested_focus=f"Third-party reviews, news, or case studies regarding {target_domain or obj.description}",
                            rationale="Company self-announcement alone cannot establish verified commercial deployment or evaluative market standing.",
                        )
                    )

            grounded_assessments.append(
                ObjectiveAssessment(
                    objective_id=obj.objective_id,
                    description=obj.description,
                    coverage_status=status,
                    viability=viability,
                    proposition_type=prop_type,
                    is_critical=obj.is_critical,
                    supporting_fact_ids=tuple(valid_supporting_ids),
                    missing_aspects=tuple(raw_obj.missing_aspects),
                )
            )

        # 4. Conflict scoping & severity handling
        evidence_deficiencies = list(raw_evaluation.evidence_deficiencies)
        blocking_conflict_descriptions = []

        for c in conflicts:
            c_desc = c.get("description", "")
            c_severity = c.get("severity", "unknown").lower()
            c_fact_ids = set(c.get("fact_ids", []))

            # Unknown severity is treated conservatively as an evidence deficiency
            if c_severity == ConflictSeverity.UNKNOWN.value:
                evidence_deficiencies.append(f"Unresolved conflict with unknown severity: {c_desc}")
                blocking_conflict_descriptions.append(f"[UNKNOWN] {c_desc}")
                # Cap any critical objective relying on these facts at PARTIAL
                for idx, a in enumerate(grounded_assessments):
                    if any(fid in c_fact_ids for fid in a.supporting_fact_ids):
                        grounded_assessments[idx] = a.model_copy(
                            update={
                                "coverage_status": CoverageStatus.PARTIAL,
                                "viability": SearchViability.PROMISING,
                            }
                        )

            elif c_severity == ConflictSeverity.CONFIRMED.value:
                blocking_conflict_descriptions.append(f"[CONFIRMED] {c_desc}")
                # Confirmed conflict directly impacts the objective
                for idx, a in enumerate(grounded_assessments):
                    if any(fid in c_fact_ids for fid in a.supporting_fact_ids):
                        grounded_assessments[idx] = a.model_copy(
                            update={
                                "coverage_status": CoverageStatus.PARTIAL,
                                "viability": SearchViability.PROMISING,
                            }
                        )

        # 5. Deterministic decision policy enforcement
        marginal_value = MarginalValueAssessment(
            identity_resolution_viable=raw_evaluation.identity_resolution_viable,
            conflict_resolution_viable=bool(blocking_conflict_descriptions) or raw_evaluation.conflict_resolution_viable,
            unaddressed_critical_viable=any(
                a.viability in (SearchViability.PROMISING, SearchViability.UNCERTAIN)
                for a in grounded_assessments
                if a.is_critical and a.coverage_status != CoverageStatus.ADEQUATE
            ),
        )

        decision = self._evaluate_deterministic_policy(
            iteration_number=iteration_number,
            max_iterations=max_iterations,
            prospect_entity=prospect_entity,
            company_entity=company_entity,
            grounded_assessments=grounded_assessments,
            blocking_conflicts=blocking_conflict_descriptions,
            marginal_value=marginal_value,
        )

        # 6. Assemble dimension assessments and final result
        recommended_actions = [
            RecommendedAction(
                action_type=self._parse_action_type(a.action_type),
                target_objective_id=a.target_objective_id,
                gap_description=a.gap_description,
                suggested_focus=a.suggested_focus,
                rationale=a.rationale,
            )
            for a in raw_evaluation.recommended_actions
        ] + extra_recommended_actions

        dimensions = DimensionAssessments(
            objective_coverage=DimensionAssessment(
                dimension_name=SufficiencyDimension.OBJECTIVE_COVERAGE,
                status="adequate" if all(a.coverage_status == CoverageStatus.ADEQUATE for a in grounded_assessments if a.is_critical) else "partial",
                summary=raw_evaluation.coverage_summary,
            ),
            identity_certainty=DimensionAssessment(
                dimension_name=SufficiencyDimension.IDENTITY_CERTAINTY,
                status="confirmed" if self._is_identity_confirmed(prospect_entity, company_entity) else "uncertain",
                summary=raw_evaluation.identity_summary,
            ),
            evidence_adequacy=DimensionAssessment(
                dimension_name=SufficiencyDimension.EVIDENCE_ADEQUACY,
                status="deficient" if evidence_deficiencies or blocking_conflict_descriptions else "adequate",
                summary=raw_evaluation.evidence_summary,
            ),
            marginal_research_value=DimensionAssessment(
                dimension_name=SufficiencyDimension.MARGINAL_RESEARCH_VALUE,
                status="viable" if decision == SufficiencyDecision.CONTINUE else "diminished",
                summary=raw_evaluation.marginal_value_summary,
            ),
        )

        return ResearchSufficiencyAssessment(
            session_id=session_id,
            iteration_id=iteration_id,
            decision=decision,
            dimensions=dimensions,
            objective_assessments=tuple(grounded_assessments),
            identity_uncertainties=tuple(raw_evaluation.identity_uncertainties),
            evidence_deficiencies=tuple(evidence_deficiencies),
            unresolved_conflicts=tuple(blocking_conflict_descriptions),
            remaining_gaps=tuple(dict.fromkeys(all_remaining_gaps)),
            recommended_actions=tuple(recommended_actions),
            explanation=raw_evaluation.explanation,
        )

    def _evaluate_deterministic_policy(
        self,
        iteration_number: int,
        max_iterations: int,
        prospect_entity: Optional[EntityRecord],
        company_entity: Optional[EntityRecord],
        grounded_assessments: Sequence[ObjectiveAssessment],
        blocking_conflicts: Sequence[str],
        marginal_value: MarginalValueAssessment,
    ) -> SufficiencyDecision:
        """Evaluate deterministic gate policy."""
        # 1. Symmetric Identity Gate
        identity_confirmed = self._is_identity_confirmed(prospect_entity, company_entity)

        # 2. Critical Objectives Evaluation
        critical_objs = [a for a in grounded_assessments if a.is_critical]
        all_critical_adequate = bool(critical_objs) and all(
            a.coverage_status == CoverageStatus.ADEQUATE for a in critical_objs
        )
        incomplete_critical = [
            a for a in critical_objs
            if a.coverage_status in (CoverageStatus.PARTIAL, CoverageStatus.UNADDRESSED)
        ]

        # 3. Blocking Conflicts
        has_blocking_conflicts = bool(blocking_conflicts)

        # --- Epistemic Gate 1: SUFFICIENT ---
        if identity_confirmed and all_critical_adequate and not has_blocking_conflicts:
            return SufficiencyDecision.SUFFICIENT

        # --- Budget Ceiling Gate ---
        if iteration_number >= max_iterations:
            return SufficiencyDecision.INSUFFICIENT

        # --- Epistemic Gate 2: CONTINUE vs INSUFFICIENT ---
        has_viable_critical_gap = any(
            a.viability in (SearchViability.PROMISING, SearchViability.UNCERTAIN)
            for a in incomplete_critical
        )
        can_resolve_identity = (not identity_confirmed) and marginal_value.identity_resolution_viable
        can_resolve_conflict = has_blocking_conflicts and marginal_value.conflict_resolution_viable

        if has_viable_critical_gap or can_resolve_identity or can_resolve_conflict:
            return SufficiencyDecision.CONTINUE

        return SufficiencyDecision.INSUFFICIENT

    def _is_identity_confirmed(
        self,
        prospect_entity: Optional[EntityRecord],
        company_entity: Optional[EntityRecord],
    ) -> bool:
        """Symmetric identity gate requiring confirmed resolution and high confidence on both prospect and company."""
        prospect_ok = (
            prospect_entity is not None
            and prospect_entity.resolution_status == ResolutionStatus.RESOLVED
            and prospect_entity.confidence == ConfidenceLevel.HIGH
        )
        company_ok = (
            company_entity is not None
            and company_entity.resolution_status == ResolutionStatus.RESOLVED
            and company_entity.confidence == ConfidenceLevel.HIGH
        )
        return prospect_ok and company_ok

    def _check_exclusively_company_owned(
        self,
        supporting_fact_ids: Sequence[str],
        fact_by_id: Dict[str, CanonicalFact],
        source_by_id: Dict[str, SourceRecord],
        target_domain: Optional[str],
    ) -> bool:
        """Returns True if all citing sources for the facts are company-owned domains."""
        if not target_domain or not supporting_fact_ids:
            return False

        target_domain_clean = target_domain.lower().replace("www.", "")
        citing_source_domains: Set[str] = set()

        for fid in supporting_fact_ids:
            fact = fact_by_id.get(fid)
            if not fact:
                continue
            for ev in fact.evidence_records:
                src = source_by_id.get(ev.source_id)
                if src:
                    parsed = urlparse(src.url if src.url.startswith("http") else f"https://{src.url}")
                    netloc = (parsed.netloc or src.domain).lower().replace("www.", "")
                    citing_source_domains.add(netloc)

        if not citing_source_domains:
            return False

        # If every domain matches target_domain, it is exclusively company owned
        return all(
            target_domain_clean in domain or domain in target_domain_clean
            for domain in citing_source_domains
        )

    def _extract_target_domain(
        self,
        prospect_entity: Optional[EntityRecord],
        company_entity: Optional[EntityRecord],
    ) -> Optional[str]:
        if company_entity and company_entity.domain:
            return company_entity.domain
        if prospect_entity and prospect_entity.domain:
            return prospect_entity.domain
        return None

    def _parse_coverage_status(self, val: str) -> CoverageStatus:
        v = (val or "").strip().lower()
        if v == "adequate":
            return CoverageStatus.ADEQUATE
        if v == "partial":
            return CoverageStatus.PARTIAL
        return CoverageStatus.UNADDRESSED

    def _parse_viability(self, val: str) -> SearchViability:
        v = (val or "").strip().lower()
        if v == "exhausted":
            return SearchViability.EXHAUSTED
        if v == "uncertain":
            return SearchViability.UNCERTAIN
        return SearchViability.PROMISING

    def _parse_proposition_type(self, val: str) -> PropositionType:
        v = (val or "").strip().lower()
        if v == "commercial_realization":
            return PropositionType.COMMERCIAL_REALIZATION
        if v == "evaluative_claim":
            return PropositionType.EVALUATIVE_CLAIM
        return PropositionType.SPEECH_ACT_ANNOUNCEMENT

    def _parse_action_type(self, val: str) -> ActionType:
        v = (val or "").strip().lower()
        if v == "identity_verification":
            return ActionType.IDENTITY_VERIFICATION
        if v == "conflict_investigation":
            return ActionType.CONFLICT_INVESTIGATION
        if v == "terminate_sufficiency":
            return ActionType.TERMINATE_SUFFICIENCY
        if v == "terminate_budget":
            return ActionType.TERMINATE_BUDGET
        return ActionType.TARGETED_SEARCH

    def _run_llm_evaluation(
        self,
        objectives: Sequence[ResearchObjective],
        facts: Sequence[CanonicalFact],
        sources: Sequence[SourceRecord],
        prospect_entity: Optional[EntityRecord],
        company_entity: Optional[EntityRecord],
        conflicts: Sequence[Dict[str, Any]],
        iteration_number: int,
        max_iterations: int,
    ) -> RawSufficiencyEvaluation:
        user_content = self._build_prompt_content(
            objectives=objectives,
            facts=facts,
            sources=sources,
            prospect_entity=prospect_entity,
            company_entity=company_entity,
            conflicts=conflicts,
            iteration_number=iteration_number,
            max_iterations=max_iterations,
        )

        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=EVALUATOR_SYSTEM_PROMPT))
        prompt.add_message(Message(role=Role.USER, content=user_content))

        decoding = DecodeConfig(
            temperature=0.2,
            max_tokens=3000,
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=decoding,
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "research_sufficiency_evaluation"},
        )

        try:
            response = self.engine.generate(request)
        except Exception as exc:
            logger.warning(
                "LLM evaluation failed or unparsed: %s. Falling back to default raw evaluation.", exc
            )
            return RawSufficiencyEvaluation(
                explanation=f"LLM evaluation failed or unparsed: {exc}",
                identity_resolution_viable=True,
            )

        if not response.parsed_output or not isinstance(response.parsed_output, RawSufficiencyEvaluation):
            logger.warning(
                "LLM failed to return structured RawSufficiencyEvaluation. Falling back to default raw evaluation."
            )
            return RawSufficiencyEvaluation(
                explanation=f"LLM evaluation failed or unparsed: {response.text[:100]}",
                identity_resolution_viable=True,
            )

        return response.parsed_output

    def _build_prompt_content(
        self,
        objectives: Sequence[ResearchObjective],
        facts: Sequence[CanonicalFact],
        sources: Sequence[SourceRecord],
        prospect_entity: Optional[EntityRecord],
        company_entity: Optional[EntityRecord],
        conflicts: Sequence[Dict[str, Any]],
        iteration_number: int,
        max_iterations: int,
    ) -> str:
        lines = [
            f"### Research Iteration {iteration_number} of {max_iterations}",
            "",
            "### Target Identity Status",
            f"- Prospect: {prospect_entity.canonical_name if prospect_entity else 'Unknown'} "
            f"(Status: {prospect_entity.resolution_status.value if prospect_entity else 'UNRESOLVED'}, "
            f"Confidence: {prospect_entity.confidence.value if prospect_entity else 'UNRESOLVED'})",
            f"- Company: {company_entity.canonical_name if company_entity else 'Unknown'} "
            f"(Domain: {company_entity.domain if company_entity else 'Unknown'}, "
            f"Status: {company_entity.resolution_status.value if company_entity else 'UNRESOLVED'}, "
            f"Confidence: {company_entity.confidence.value if company_entity else 'UNRESOLVED'})",
            "",
            "### Research Objectives",
        ]

        for obj in objectives:
            lines.append(
                f"- [ID: {obj.objective_id}] ({'CRITICAL' if obj.is_critical else 'SECONDARY'}) "
                f"[{obj.type.value}] {obj.description}"
            )

        lines.extend(["", "### Canonical Facts Collected"])
        if facts:
            for f in facts:
                lines.append(
                    f"- [Fact ID: {f.fact_id}] ({f.composite_confidence.value}): {f.claim_text}"
                )
        else:
            lines.append("No facts extracted yet.")

        lines.extend(["", "### Conflicts Detected"])
        if conflicts:
            for c in conflicts:
                lines.append(f"- [{c.get('severity', 'unknown').upper()}] {c.get('description', '')}")
        else:
            lines.append("No conflicts detected.")

        lines.extend([
            "",
            "Analyze coverage for each objective ID, assign valid supporting fact IDs, assess search viability, and provide recommended actions.",
        ])

        return "\n".join(lines)
