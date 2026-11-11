from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Set
import uuid

from src.context.models import ProductContext, ProspectContext
from src.researcher.evidence_request.models import (
    EvidenceGapReason,
    EvidenceRequest,
    EvidenceRequestStatus,
)
from src.researcher.models import CanonicalFact, EntityRecord, ResearchPackage, VerificationStatus
from .models import CandidateSignal, EligibilityStatus, RankedSignal, StrategyResult

logger = logging.getLogger(__name__)


class EvidenceGapDetector:
    """Analyzes strategy construction results to detect genuine, material factual uncertainties and construct neutral EvidenceRequests."""

    def __init__(self, max_requests: int = 3) -> None:
        self.max_requests = max_requests

    def detect_gaps_and_create_requests(
        self,
        strategy_result: StrategyResult,
        research_package: ResearchPackage,
        prospect: ProspectContext,
        product: ProductContext,
        run_id: str,
    ) -> List[EvidenceRequest]:
        """Inspect strategy viability, ranked signals, and facts to generate up to 3 neutral EvidenceRequests.
        
        Returns an empty list if:
          - A fully verified, non-conditional strategy is already viable.
          - No genuine factual gaps exist.
          - Target entities are insufficiently resolved.
        """
        # If strategy is fully viable and NOT conditional, no feedback research is warranted
        if strategy_result.is_viable and strategy_result.strategy:
            is_cond = getattr(strategy_result.strategy, "is_conditional", False)
            unresolved = getattr(strategy_result.strategy, "unresolved_conditions", [])
            if not is_cond and not unresolved:
                logger.debug("Strategy is fully viable and unconditional; no evidence requests generated.")
                return []

        # If no ranked signals and no strategy exist, verify whether unviable reasons indicate factual gaps
        if not strategy_result.ranked_signals and not strategy_result.strategy:
            has_factual_gap_reason = any(
                any(k in r.lower() for k in ["prerequisite", "employment", "unconfirmed", "contradict", "missing evidence"])
                for r in (strategy_result.unviable_reasons or [])
            )
            if not has_factual_gap_reason:
                logger.debug("No ranked signals, strategy blueprint, or factual gap reasons; no evidence requests generated.")
                return []

        company_id = (
            research_package.company_entity.entity_id
            if research_package.company_entity
            else f"org_{re.sub(r'[^a-zA-Z0-9]', '_', prospect.company_name).lower()}"
        )
        prospect_id = (
            research_package.prospect_entity.entity_id
            if research_package.prospect_entity
            else prospect.prospect_id
        )

        requests: List[EvidenceRequest] = []
        seen_keys: Set[str] = set()

        def add_request(req: EvidenceRequest) -> None:
            dedup_key = f"{req.target_entity_id}::{req.requested_predicate}::{req.neutral_question.strip().lower()}"
            if dedup_key in seen_keys:
                return
            seen_keys.add(dedup_key)
            if len(requests) < self.max_requests:
                requests.append(req)

        # 1. Inspect Missing Prerequisites / Unknown Conditions in Ranked Signals
        for ranked in strategy_result.ranked_signals:
            if len(requests) >= self.max_requests:
                break

            elig = ranked.eligibility
            if not elig:
                continue

            # Check unknown conditions or missing prerequisites from product compatibility gate
            unresolved_conds = list(elig.unresolved_conditions)
            if elig.product_gate and elig.product_gate.details:
                for prereq in elig.product_gate.details.get("missing_prerequisites", []):
                    if prereq not in unresolved_conds:
                        unresolved_conds.append(prereq)

            for cond in unresolved_conds:
                if len(requests) >= self.max_requests:
                    break

                clean_cond = cond.replace("_", " ").strip()
                req = EvidenceRequest(
                    run_id=run_id,
                    target_entity_id=company_id,
                    target_entity_name=prospect.company_name,
                    requested_predicate=f"infrastructure_requirement::{cond}",
                    neutral_question=f"Does {prospect.company_name} operate or deploy {clean_cond}?",
                    neutral_search_terms=[
                        f"{prospect.company_name} {clean_cond}",
                        f"{prospect.company_name} infrastructure {clean_cond}",
                    ],
                    disconfirming_search_terms=[
                        f"{prospect.company_name} on-premise datacenter",
                        f"{prospect.company_name} alternative infrastructure",
                        f"{prospect.company_name} no {clean_cond}",
                    ],
                    gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
                    provenance_stage="signal_eligibility_product_gate",
                    motivated_by_signal_id=ranked.signal.signal_id,
                )
                add_request(req)

        # 2. Inspect Unresolved Employment or Leadership Role
        if len(requests) < self.max_requests and prospect.full_name and prospect.company_name:
            role_verified = False
            for fact in research_package.facts:
                claim_norm = fact.claim_text.lower()
                name_norm = prospect.full_name.lower()
                comp_norm = prospect.company_name.lower()
                if name_norm in claim_norm and comp_norm in claim_norm:
                    role_verified = True
                    break

            if not role_verified:
                designation = prospect.designation or "leadership role"
                req = EvidenceRequest(
                    run_id=run_id,
                    target_entity_id=prospect_id,
                    target_entity_name=prospect.full_name,
                    requested_predicate="works_for",
                    neutral_question=f"Does {prospect.full_name} currently hold an active position as {designation} at {prospect.company_name}?",
                    neutral_search_terms=[
                        f"{prospect.full_name} {prospect.company_name} {designation}",
                        f"{prospect.full_name} {prospect.company_name} executive",
                    ],
                    disconfirming_search_terms=[
                        f"{prospect.full_name} departed {prospect.company_name}",
                        f"{prospect.full_name} left {prospect.company_name}",
                        f"{prospect.full_name} former {prospect.company_name}",
                    ],
                    gap_reason=EvidenceGapReason.UNRESOLVED_EMPLOYMENT,
                    provenance_stage="signal_eligibility_identity_gate",
                )
                add_request(req)

        # 3. Inspect Contradicted Evidence on Supporting Facts
        if len(requests) < self.max_requests:
            for fact in research_package.facts:
                if len(requests) >= self.max_requests:
                    break
                has_contradiction = any(
                    getattr(ev, "verification_status", None) == VerificationStatus.CONTRADICTED
                    for ev in fact.evidence_records
                )
                if has_contradiction:
                    req = EvidenceRequest(
                        run_id=run_id,
                        target_entity_id=company_id,
                        target_entity_name=prospect.company_name,
                        requested_predicate="claim_contradiction_resolution",
                        neutral_question=f"What is the verified public status regarding whether {fact.claim_text}?",
                        neutral_search_terms=[
                            f"{prospect.company_name} {fact.claim_text[:60]} confirmed",
                            f"{prospect.company_name} {fact.claim_text[:60]} announcement",
                        ],
                        disconfirming_search_terms=[
                            f"{prospect.company_name} {fact.claim_text[:60]} denied",
                            f"{prospect.company_name} {fact.claim_text[:60]} cancelled",
                            f"{prospect.company_name} {fact.claim_text[:60]} reversed",
                        ],
                        gap_reason=EvidenceGapReason.MATERIAL_CONTRADICTION,
                        provenance_stage="claim_verification_contradiction_audit",
                        motivated_by_fact_ids=[fact.fact_id],
                    )
                    add_request(req)

        logger.info(f"EvidenceGapDetector generated {len(requests)} neutral EvidenceRequests.")
        return requests
