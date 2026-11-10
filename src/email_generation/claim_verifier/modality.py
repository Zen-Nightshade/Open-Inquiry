from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.context.models import CompatibilityStatus
from src.email_generation.claim_verifier.models import (
    ClaimCategory,
    DecomposedClaim,
    ModalityCheckResult,
    ModalityLevel,
    VerificationFinding,
    VerifierFindingSeverity,
)
from src.signal_strategy.models import CTAType, OutreachStrategy, ToulminArgument


class ModalityPreservationGate:
    """Gate 3: Evaluates whether epistemic modality, hedging, and diagnostic framing
    established in the approved Toulmin argument survive generation without escalation.
    """

    PRESUMPTIVE_MEETING_PATTERNS = [
        re.compile(r"\b(?:book\s+(?:a\s+)?(?:call|meeting|time|demo)|schedule\s+(?:a\s+)?(?:call|meeting|time|demo)|grab\s+(?:some\s+)?time|jump\s+on\s+a\s+call)\b", re.IGNORECASE),
        re.compile(r"\b(?:send\s+over\s+your\s+calendar|what's\s+your\s+calendar|check\s+your\s+calendar|available\s+for\s+a\s+call)\b", re.IGNORECASE),
        re.compile(r"\b(?:15|20|30|45|60)\s*[- ]?minute(?:s)?\s+(?:call|chat|demo|meeting)\b", re.IGNORECASE),
        re.compile(r"\b(?:tomorrow|next\s+week|this\s+thursday|this\s+friday)\s+(?:at\s+\d+|for\s+a\s+chat)\b", re.IGNORECASE),
    ]

    DEFINITIVE_ESCALATION_PATTERNS = [
        re.compile(r"\b(?:will\s+(?:eliminate|solve|guarantee|fix|remove)|guarantees|ensures\s+complete|eliminates\s+all)\b", re.IGNORECASE),
        re.compile(r"\b(?:we\s+solve\s+your|our\s+platform\s+fixes\s+your|we\s+will\s+optimize\s+your)\b", re.IGNORECASE),
    ]

    CONDITIONAL_HEDGE_PATTERNS = [
        re.compile(r"\b(?:if|assuming|when\s+configured|depending\s+on|might|could|may|potentially|perhaps)\b", re.IGNORECASE),
    ]

    def verify(
        self,
        claims: List[DecomposedClaim],
        strategy: OutreachStrategy,
        toulmin_argument: Optional[ToulminArgument],
    ) -> Tuple[List[ModalityCheckResult], List[VerificationFinding]]:
        """Verify epistemic modality preservation across all decomposed claims."""
        results: List[ModalityCheckResult] = []
        findings: List[VerificationFinding] = []

        if not toulmin_argument:
            # Missing Toulmin argument is an invalid state per M1.3
            findings.append(
                VerificationFinding(
                    gate="ModalityPreservation",
                    severity=VerifierFindingSeverity.ERROR,
                    code="MISSING_TOULMIN_ARGUMENT",
                    message="Strategy lacks a validated ToulminArgument to verify modality against.",
                )
            )
            return results, findings

        qualifier = toulmin_argument.qualifier
        warrant = toulmin_argument.warrant
        is_conditional_strategy = (
            qualifier.compatibility_status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
            or strategy.is_conditional
            or bool(qualifier.unresolved_prerequisites)
        )

        approved_modality = (
            ModalityLevel.CONDITIONAL if is_conditional_strategy else ModalityLevel.DEFINITIVE
        )

        for claim in claims:
            # 1. Check CTA Modality Escalation
            if claim.category == ClaimCategory.RELATIONAL_CTA:
                cta_escalated, cta_reason = self._check_cta_escalation(claim, strategy)
                if cta_escalated:
                    results.append(
                        ModalityCheckResult(
                            claim_id=claim.claim_id,
                            approved_modality=ModalityLevel.DIAGNOSTIC,
                            draft_modality=ModalityLevel.DEFINITIVE,
                            is_escalated=True,
                            rationale=cta_reason,
                        )
                    )
                    findings.append(
                        VerificationFinding(
                            gate="ModalityPreservation",
                            severity=VerifierFindingSeverity.ERROR,
                            code="CTA_MODALITY_ESCALATION",
                            message=cta_reason,
                            claim_id=claim.claim_id,
                            section=claim.section,
                            text_span=(claim.span_start, claim.span_end),
                        )
                    )
                    continue

            # 2. Check Product & Implication Claims under Conditional Strategies
            if is_conditional_strategy and claim.category in (
                ClaimCategory.PRODUCT_CAPABILITY,
                ClaimCategory.METRIC_IMPACT,
                ClaimCategory.PROSPECT_OBSERVATION,
            ):
                is_escalated, reason = self._check_conditional_escalation(claim, qualifier.unresolved_prerequisites)
                results.append(
                    ModalityCheckResult(
                        claim_id=claim.claim_id,
                        approved_modality=ModalityLevel.CONDITIONAL,
                        draft_modality=claim.modality,
                        is_escalated=is_escalated,
                        rationale=reason,
                    )
                )
                if is_escalated:
                    findings.append(
                        VerificationFinding(
                            gate="ModalityPreservation",
                            severity=VerifierFindingSeverity.ERROR,
                            code="MODALITY_ESCALATION",
                            message=reason,
                            claim_id=claim.claim_id,
                            section=claim.section,
                            text_span=(claim.span_start, claim.span_end),
                            details={"unresolved_conditions": qualifier.unresolved_prerequisites},
                        )
                    )
                continue

            # 3. Check Inferred Warrant Opportunity
            if warrant.is_inferred and claim.category == ClaimCategory.PROSPECT_OBSERVATION:
                is_inferred_escalated, reason = self._check_inferred_warrant(claim)
                if is_inferred_escalated:
                    results.append(
                        ModalityCheckResult(
                            claim_id=claim.claim_id,
                            approved_modality=ModalityLevel.HYPOTHETICAL,
                            draft_modality=ModalityLevel.DEFINITIVE,
                            is_escalated=True,
                            rationale=reason,
                        )
                    )
                    findings.append(
                        VerificationFinding(
                            gate="ModalityPreservation",
                            severity=VerifierFindingSeverity.ERROR,
                            code="INFERRED_OPPORTUNITY_AS_FACT",
                            message=reason,
                            claim_id=claim.claim_id,
                            section=claim.section,
                            text_span=(claim.span_start, claim.span_end),
                        )
                    )
                    continue

            # Default safe result
            results.append(
                ModalityCheckResult(
                    claim_id=claim.claim_id,
                    approved_modality=approved_modality,
                    draft_modality=claim.modality,
                    is_escalated=False,
                    rationale="Modality conforms to approved Toulmin argument.",
                )
            )

        return results, findings

    def _check_cta_escalation(self, claim: DecomposedClaim, strategy: OutreachStrategy) -> Tuple[bool, str]:
        """Check if an approved diagnostic CTA was converted into a high-friction meeting request."""
        # Check if strategy requires low friction / diagnostic question
        cta_type = getattr(strategy, "cta_type", None)
        if cta_type == CTAType.DIAGNOSTIC_QUESTION:
            # Check for presumptive calendar/meeting pitch
            for pat in self.PRESUMPTIVE_MEETING_PATTERNS:
                if pat.search(claim.text):
                    return True, (
                        f"Approved strategy requires DIAGNOSTIC_QUESTION CTA, but draft converted it to "
                        f"a high-friction/presumptive meeting pitch: '{claim.text}'"
                    )
        return False, ""

    def _check_conditional_escalation(
        self,
        claim: DecomposedClaim,
        unresolved_prerequisites: List[str],
    ) -> Tuple[bool, str]:
        """Check if a claim in a conditional strategy introduces definitive assertives without qualification."""
        # If claim already contains conditional hedges, it is safe
        if any(pat.search(claim.text) for pat in self.CONDITIONAL_HEDGE_PATTERNS):
            return False, "Claim includes appropriate conditional qualification."

        # Check for aggressive definitive escalation
        for pat in self.DEFINITIVE_ESCALATION_PATTERNS:
            if pat.search(claim.text):
                prereq_str = f" (unresolved: {', '.join(unresolved_prerequisites)})" if unresolved_prerequisites else ""
                return True, (
                    f"Conditional strategy requires epistemic qualification{prereq_str}, but claim makes "
                    f"definitive assertion: '{claim.text}'"
                )

        return False, "Claim phrasing is acceptable."

    def _check_inferred_warrant(self, claim: DecomposedClaim) -> Tuple[bool, str]:
        """Check if an inferred problem connection is asserted as a certified fact."""
        # If prospect claim asserts definitive problem without conditional framing
        if claim.modality == ModalityLevel.DEFINITIVE:
            # e.g., "Because your cluster suffers from lag" vs "clusters often suffer from lag"
            if re.search(r"\b(?:you\s+suffer\s+from|your\s+team\s+is\s+struggling\s+with|your\s+bottleneck\s+is)\b", claim.text, re.IGNORECASE):
                return True, (
                    f"Warrant reasoning is inferred, but claim asserts prospect problem as an established fact: '{claim.text}'"
                )
        return False, ""
