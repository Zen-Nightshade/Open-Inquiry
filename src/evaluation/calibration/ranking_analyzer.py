from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.signal_strategy.models import (
    EligibilityStatus,
    GateIdentifier,
    RankedSignal,
    SignalEligibility,
)

logger = logging.getLogger(__name__)


# Standard mapping of reviewer reason codes to the authoritative eligibility gates
REASON_CODE_TO_GATE_MAP: Dict[str, GateIdentifier] = {
    "ENTITY_MISMATCH": GateIdentifier.IDENTITY,
    "WRONG_COMPANY": GateIdentifier.IDENTITY,
    "WEAK_EVIDENCE": GateIdentifier.EVIDENCE,
    "HALLUCINATION": GateIdentifier.EVIDENCE,
    "UNSUBSTANTIATED_CLAIM": GateIdentifier.EVIDENCE,
    "MISALIGNED_PRODUCT": GateIdentifier.PRODUCT_COMPATIBILITY,
    "PRODUCT_MISMATCH": GateIdentifier.PRODUCT_COMPATIBILITY,
    "WRONG_CAPABILITY": GateIdentifier.PRODUCT_COMPATIBILITY,
    "WRONG_PERSONA": GateIdentifier.PERSONA,
    "INSUFFICIENT_PERSONALIZATION": GateIdentifier.PERSONA,
    "IRRELEVANT_BUYER": GateIdentifier.PERSONA,
}


class RankingQualityAnalyzer:
    """Evaluates ranking decisions against reviewer feedback and validates gate invariants."""

    @classmethod
    def verify_eligibility_invariants(
        cls,
        ranked_signals: Sequence[RankedSignal],
    ) -> Dict[str, Any]:
        """Verify the non-compensatory eligibility invariants across ranked signals.

        Invariants:
        1. An INELIGIBLE signal must have composite_score == 0.0.
        2. High recency or business significance must never compensate for a failed gate.
        3. No calibrated probability or score tuning can override an ineligible determination.
        """
        violations: List[str] = []
        ineligible_count = 0
        eligible_count = 0
        conditional_count = 0

        for sig in ranked_signals:
            elig = sig.eligibility
            if elig is None:
                continue

            if elig.status == EligibilityStatus.INELIGIBLE:
                ineligible_count += 1
                if sig.score.composite_score > 0.0:
                    violations.append(
                        f"Signal '{sig.signal.signal_id}' is INELIGIBLE but has non-zero composite_score "
                        f"{sig.score.composite_score} (violates non-compensatory gate invariant)."
                    )
            elif elig.status == EligibilityStatus.ELIGIBLE:
                eligible_count += 1
            elif elig.status == EligibilityStatus.CONDITIONAL:
                conditional_count += 1

        is_valid = len(violations) == 0
        return {
            "is_valid": is_valid,
            "violations": violations,
            "total_signals": len(ranked_signals),
            "eligible_count": eligible_count,
            "conditional_count": conditional_count,
            "ineligible_count": ineligible_count,
            "authoritative_gates_preserved": is_valid,
        }

    @classmethod
    def analyze_rejection_reasons(
        cls,
        rejection_events: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze empirical distribution of human review rejection reasons.

        Examines which non-compensatory gates or score components are most frequently
        associated with human reviewer rejections.
        """
        reason_counts: Dict[str, int] = {}
        gate_counts: Dict[str, int] = {g.value: 0 for g in GateIdentifier}
        ungated_counts: Dict[str, int] = {}

        total_rejections = len(rejection_events)

        for event in rejection_events:
            reason = (event.get("reason_code") or "UNSPECIFIED").strip().upper()
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            matched_gate = REASON_CODE_TO_GATE_MAP.get(reason)
            if matched_gate:
                gate_counts[matched_gate.value] += 1
            else:
                ungated_counts[reason] = ungated_counts.get(reason, 0) + 1

        # Calculate proportions
        proportions = {
            r: round(c / total_rejections, 4) if total_rejections > 0 else 0.0
            for r, c in reason_counts.items()
        }

        return {
            "total_rejections": total_rejections,
            "reason_distribution": reason_counts,
            "reason_proportions": proportions,
            "gate_attribution": gate_counts,
            "ungated_reasons": ungated_counts,
        }
