from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import ProspectContext
from src.researcher.entity.candidates import DiscoveredEntityCandidate
from src.researcher.entity.probabilistic_models import (
    CandidateMatchAssessment,
    ResolutionOutcome,
)
from src.researcher.entity.probabilistic_scorer import ProbabilisticCandidateScorer
from src.researcher.models import (
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    ResolutionStatus,
)

logger = logging.getLogger(__name__)


class ProbabilisticResolutionResult(BaseModel):
    """Auditable resolution decision preserving candidate assessments, margin separation, and ambiguity."""

    model_config = ConfigDict(frozen=True)

    target_id: Optional[str] = None
    target_name: Optional[str] = None
    entity_type: EntityType
    outcome: ResolutionOutcome
    resolution_status: ResolutionStatus
    confidence: ConfidenceLevel
    selected_candidate_id: Optional[str] = None
    selected_candidate: Optional[DiscoveredEntityCandidate] = None
    competing_candidate_ids: List[str] = Field(default_factory=list)
    all_assessments: List[CandidateMatchAssessment] = Field(default_factory=list)
    best_score: float = 0.0
    runner_up_score: Optional[float] = None
    score_margin: Optional[float] = None
    thresholds: Dict[str, float] = Field(default_factory=dict)
    rationale: str
    provenance: Optional[OutputProvenance] = None


class ProbabilisticEntityResolver:
    """Margin-based, explainable entity resolution policy engine.

    Enforces:
      - 4-state resolution decisions: MATCH, NO_MATCH, AMBIGUOUS, UNRESOLVED.
      - Margin separation rule: Prevents binding when close competing viable candidates exist.
      - Disqualified candidate filtering: Disqualified candidates (e.g. from incompatible entity
        types or conflicting employers/domains) cannot create artificial ambiguity against a valid match.
      - NO_MATCH vs UNRESOLVED distinction: A low score alone does not prove non-identity;
        NO_MATCH requires affirmative contradictory evidence.
    """

    def __init__(
        self,
        scorer: Optional[ProbabilisticCandidateScorer] = None,
        match_threshold: float = 0.75,
        no_match_threshold: float = 0.30,
        margin_threshold: float = 0.15,
    ) -> None:
        self.scorer = scorer or ProbabilisticCandidateScorer()
        self.match_threshold = match_threshold
        self.no_match_threshold = no_match_threshold
        self.margin_threshold = margin_threshold

    def resolve_candidates(
        self,
        candidates: Sequence[DiscoveredEntityCandidate],
        target_prospect: Optional[ProspectContext] = None,
        target_entity: Optional[EntityRecord] = None,
        entity_type: Optional[EntityType] = None,
    ) -> ProbabilisticResolutionResult:
        """Resolve a collection of candidate entities against a target prospect or entity record."""
        target_name = (
            target_prospect.full_name
            if target_prospect
            else (target_entity.raw_name if target_entity else None)
        )
        target_id = (
            target_entity.entity_id
            if target_entity
            else (getattr(target_prospect, "context_id", getattr(target_prospect, "prospect_id", None)) if target_prospect else None)
        )
        effective_type = (
            entity_type
            or (target_entity.entity_type if target_entity else None)
            or (EntityType.PERSON if target_prospect else EntityType.ORGANIZATION)
        )

        thresholds = {
            "match_threshold": self.match_threshold,
            "no_match_threshold": self.no_match_threshold,
            "margin_threshold": self.margin_threshold,
        }

        # Case 0: Empty candidates discovered
        if not candidates:
            return ProbabilisticResolutionResult(
                target_id=target_id,
                target_name=target_name,
                entity_type=effective_type,
                outcome=ResolutionOutcome.UNRESOLVED,
                resolution_status=ResolutionStatus.UNRESOLVED,
                confidence=ConfidenceLevel.UNRESOLVED,
                selected_candidate_id=None,
                selected_candidate=None,
                competing_candidate_ids=[],
                all_assessments=[],
                best_score=0.0,
                runner_up_score=None,
                score_margin=None,
                thresholds=thresholds,
                rationale="No candidate entities were discovered in research sources (insufficient evidence).",
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="probabilistic_entity_resolver",
                    rule_id="empty_candidates_unresolved",
                ),
            )

        # 1. Filter candidates to relevant entity type for resolution scope
        relevant_candidates = [c for c in candidates if c.entity_type == effective_type]
        other_candidates = [c for c in candidates if c.entity_type != effective_type]

        # Score all relevant candidates
        assessments: List[CandidateMatchAssessment] = []
        cand_map: Dict[str, DiscoveredEntityCandidate] = {}

        for c in relevant_candidates:
            cand_map[c.candidate_id] = c
            ass = self.scorer.score_candidate(
                candidate=c,
                target_prospect=target_prospect,
                target_entity=target_entity,
            )
            assessments.append(ass)

        # Also score other candidates to register type-incompatibility assessments for auditability
        for c in other_candidates:
            cand_map[c.candidate_id] = c
            ass = self.scorer.score_candidate(
                candidate=c,
                target_prospect=target_prospect,
                target_entity=target_entity,
            )
            assessments.append(ass)

        # Sort assessments by match_score descending
        relevant_assessments = [a for a in assessments if a.entity_type == effective_type]
        relevant_assessments.sort(key=lambda a: a.match_score, reverse=True)

        # Case 1: No relevant candidates of matching entity type
        if not relevant_assessments:
            return ProbabilisticResolutionResult(
                target_id=target_id,
                target_name=target_name,
                entity_type=effective_type,
                outcome=ResolutionOutcome.NO_MATCH,
                resolution_status=ResolutionStatus.UNRESOLVED,
                confidence=ConfidenceLevel.LOW,
                selected_candidate_id=None,
                selected_candidate=None,
                competing_candidate_ids=[],
                all_assessments=assessments,
                best_score=0.0,
                runner_up_score=None,
                score_margin=None,
                thresholds=thresholds,
                rationale=f"All discovered candidates had incompatible entity types (target is {effective_type.value}).",
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="probabilistic_entity_resolver",
                    rule_id="incompatible_types_no_match",
                ),
            )

        # Filter viable candidates: candidates that are not strongly disqualified
        # Disqualified candidates MUST NOT create artificial ambiguity!
        viable_assessments = [
            a for a in relevant_assessments
            if not a.is_disqualified and a.match_score >= self.no_match_threshold
        ]

        best_ass = relevant_assessments[0]
        best_score = best_ass.match_score
        runner_up_score: Optional[float] = None
        score_margin: Optional[float] = None

        if len(viable_assessments) > 1:
            runner_up_score = viable_assessments[1].match_score
            score_margin = round(best_score - runner_up_score, 3)

        # Decision Logic:
        # Case 2: Strong match candidate exists
        if best_score >= self.match_threshold:
            # Check margin against viable runner-up
            if (
                len(viable_assessments) >= 2
                and runner_up_score is not None
                and runner_up_score >= (self.match_threshold - self.margin_threshold)
                and (best_score - runner_up_score) < self.margin_threshold
            ):
                # Multiple close viable candidates without sufficient margin -> AMBIGUOUS!
                competing_ids = [
                    a.candidate_id for a in viable_assessments
                    if (best_score - a.match_score) < self.margin_threshold
                ]
                competing_names = [cand_map[cid].raw_name for cid in competing_ids if cid in cand_map]
                rationale = (
                    f"Ambiguous match: Discovered {len(competing_ids)} close viable candidates "
                    f"({', '.join(competing_names)}) with top scores ({best_score:.2f} vs {runner_up_score:.2f}) "
                    f"differing by {score_margin:.2f} < margin threshold {self.margin_threshold:.2f}."
                )
                return ProbabilisticResolutionResult(
                    target_id=target_id,
                    target_name=target_name,
                    entity_type=effective_type,
                    outcome=ResolutionOutcome.AMBIGUOUS,
                    resolution_status=ResolutionStatus.AMBIGUOUS,
                    confidence=ConfidenceLevel.LOW,
                    selected_candidate_id=None,
                    selected_candidate=None,
                    competing_candidate_ids=competing_ids,
                    all_assessments=assessments,
                    best_score=best_score,
                    runner_up_score=runner_up_score,
                    score_margin=score_margin,
                    thresholds=thresholds,
                    rationale=rationale,
                    provenance=OutputProvenance(
                        origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                        component="probabilistic_entity_resolver",
                        rule_id="margin_separation_ambiguous",
                    ),
                )
            else:
                # Clear winner with sufficient margin or distant runner-up -> MATCH!
                winner_cand = cand_map.get(best_ass.candidate_id)
                margin_desc = (
                    f"Margin over runner-up: {score_margin:.2f} >= {self.margin_threshold:.2f}."
                    if score_margin is not None
                    else "No competing viable candidates."
                )
                rationale = (
                    f"Confirmed match for candidate '{best_ass.candidate_name}' (Score: {best_score:.2f} >= "
                    f"threshold {self.match_threshold:.2f}). {margin_desc}"
                )
                confidence = ConfidenceLevel.HIGH if best_score >= 0.85 else ConfidenceLevel.MEDIUM
                return ProbabilisticResolutionResult(
                    target_id=target_id,
                    target_name=target_name,
                    entity_type=effective_type,
                    outcome=ResolutionOutcome.MATCH,
                    resolution_status=ResolutionStatus.RESOLVED,
                    confidence=confidence,
                    selected_candidate_id=best_ass.candidate_id,
                    selected_candidate=winner_cand,
                    competing_candidate_ids=[],
                    all_assessments=assessments,
                    best_score=best_score,
                    runner_up_score=runner_up_score,
                    score_margin=score_margin,
                    thresholds=thresholds,
                    rationale=rationale,
                    provenance=OutputProvenance(
                        origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                        component="probabilistic_entity_resolver",
                        rule_id="confident_match_resolved",
                    ),
                )

        # Case 3: Best score is low or candidate is disqualified
        if best_ass.is_disqualified or (best_score <= self.no_match_threshold and best_ass.negative_evidence):
            # Affirmative evidence of non-identity -> NO_MATCH
            rationale = (
                f"Confirmed non-match: Candidate '{best_ass.candidate_name}' has affirmative contradictory "
                f"evidence: {'; '.join(best_ass.negative_evidence[:2])} (Score: {best_score:.2f} <= {self.no_match_threshold:.2f})."
            )
            return ProbabilisticResolutionResult(
                target_id=target_id,
                target_name=target_name,
                entity_type=effective_type,
                outcome=ResolutionOutcome.NO_MATCH,
                resolution_status=ResolutionStatus.UNRESOLVED,
                confidence=ConfidenceLevel.LOW,
                selected_candidate_id=None,
                selected_candidate=None,
                competing_candidate_ids=[],
                all_assessments=assessments,
                best_score=best_score,
                runner_up_score=runner_up_score,
                score_margin=score_margin,
                thresholds=thresholds,
                rationale=rationale,
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="probabilistic_entity_resolver",
                    rule_id="contradictory_evidence_no_match",
                ),
            )

        # Case 4: Insufficient evidence (score between no_match and match threshold, without disqualification)
        missing_desc = (
            f"Missing attributes: {', '.join(best_ass.missing_features)}."
            if best_ass.missing_features
            else ""
        )
        rationale = (
            f"Insufficient evidence to decide: Best candidate '{best_ass.candidate_name}' scored {best_score:.2f} "
            f"(below match threshold {self.match_threshold:.2f}, above no-match threshold {self.no_match_threshold:.2f}). "
            f"{missing_desc} Treated as unobserved without concluding non-match."
        )
        return ProbabilisticResolutionResult(
            target_id=target_id,
            target_name=target_name,
            entity_type=effective_type,
            outcome=ResolutionOutcome.UNRESOLVED,
            resolution_status=ResolutionStatus.UNRESOLVED,
            confidence=ConfidenceLevel.UNRESOLVED,
            selected_candidate_id=None,
            selected_candidate=None,
            competing_candidate_ids=[],
            all_assessments=assessments,
            best_score=best_score,
            runner_up_score=runner_up_score,
            score_margin=score_margin,
            thresholds=thresholds,
            rationale=rationale,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                component="probabilistic_entity_resolver",
                rule_id="insufficient_evidence_unresolved",
            ),
        )
