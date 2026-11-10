from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from src.context.models import CapabilityNode, CompatibilityStatus, ProductContract
from src.email_generation.claim_verifier.models import (
    CapabilityCheckResult,
    ClaimCategory,
    DecomposedClaim,
    VerificationFinding,
    VerifierFindingSeverity,
)
from src.signal_strategy.models import ToulminArgument


class ProductCapabilityBoundsGate:
    """Gate 4: Verifies product capability assertions against the closed-world
    ProductContract and its constituent CapabilityNodes and whitelisted proof points.
    """

    NUMERICAL_REGEX = re.compile(r"\b(?:\d+[\d,\.]*|\$\d+[\d,\.]*|\d+%\s*|\d+x)\b")

    def verify(
        self,
        claims: List[DecomposedClaim],
        contract: ProductContract,
        toulmin_argument: Optional[ToulminArgument] = None,
    ) -> Tuple[List[CapabilityCheckResult], List[VerificationFinding]]:
        """Verify product-facing claims against ProductContract capabilities and proof points."""
        results: List[CapabilityCheckResult] = []
        findings: List[VerificationFinding] = []

        # Extract whitelisted proof points from all capabilities and Toulmin backing
        all_whitelisted_proofs: Set[str] = set()
        for cap in contract.capabilities:
            for pp in cap.verified_proof_points:
                all_whitelisted_proofs.add(pp.strip().lower())

        if toulmin_argument and toulmin_argument.backing:
            for pp in toulmin_argument.backing.whitelisted_proof_points:
                all_whitelisted_proofs.add(pp.strip().lower())

        target_cap_id = toulmin_argument.warrant.capability_id if toulmin_argument else None
        target_cap: Optional[CapabilityNode] = None
        if target_cap_id:
            target_cap = next((c for c in contract.capabilities if c.capability_id == target_cap_id), None)

        if not target_cap and toulmin_argument and toulmin_argument.warrant and toulmin_argument.warrant.capability_name:
            target_cap = CapabilityNode(
                capability_id=toulmin_argument.warrant.capability_id or "cap-toulmin",
                name=toulmin_argument.warrant.capability_name,
                description=toulmin_argument.warrant.reasoning,
                verified_proof_points=list(toulmin_argument.backing.whitelisted_proof_points) if toulmin_argument.backing else [],
            )

        product_claims = [
            c for c in claims
            if (c.is_product_claim or c.category == ClaimCategory.PRODUCT_CAPABILITY)
            and c.category in (ClaimCategory.PRODUCT_CAPABILITY, ClaimCategory.METRIC_IMPACT)
            and c.is_material
        ]

        for claim in product_claims:
            claim_text_lower = claim.text.lower()

            # 1. Check Hard Exclusions (global and capability-specific)
            exclusion_violation = self._check_hard_exclusions(claim, contract, target_cap)
            if exclusion_violation:
                results.append(
                    CapabilityCheckResult(
                        claim_id=claim.claim_id,
                        capability_id=target_cap.capability_id if target_cap else None,
                        capability_name=target_cap.name if target_cap else None,
                        is_whitelisted=False,
                        is_precondition_satisfied=False,
                        violates_exclusion=True,
                        rationale=exclusion_violation,
                    )
                )
                findings.append(
                    VerificationFinding(
                        gate="ProductCapabilityBounds",
                        severity=VerifierFindingSeverity.ERROR,
                        code="HARD_EXCLUSION_VIOLATION",
                        message=exclusion_violation,
                        claim_id=claim.claim_id,
                        section=claim.section,
                        text_span=(claim.span_start, claim.span_end),
                    )
                )
                continue

            # 2. Check Numerical Proof Points
            metrics_in_claim = self.NUMERICAL_REGEX.findall(claim.text)
            unsupported_metrics: List[str] = []

            for metric in metrics_in_claim:
                metric_clean = metric.strip().lower()
                # Check if this metric appears in any whitelisted proof point
                is_metric_whitelisted = any(
                    metric_clean in pp
                    for pp in all_whitelisted_proofs
                )
                if not is_metric_whitelisted:
                    unsupported_metrics.append(metric)

            if unsupported_metrics:
                results.append(
                    CapabilityCheckResult(
                        claim_id=claim.claim_id,
                        capability_id=target_cap.capability_id if target_cap else None,
                        capability_name=target_cap.name if target_cap else None,
                        is_whitelisted=False,
                        is_precondition_satisfied=True,
                        violates_exclusion=False,
                        unsupported_proof_points=unsupported_metrics,
                        rationale=f"Claim asserts numerical metrics not in whitelisted proof points: {unsupported_metrics}",
                    )
                )
                findings.append(
                    VerificationFinding(
                        gate="ProductCapabilityBounds",
                        severity=VerifierFindingSeverity.ERROR,
                        code="UNSUPPORTED_PROOF_POINT",
                        message=f"Product claim asserts unverified numerical metrics: {unsupported_metrics}",
                        claim_id=claim.claim_id,
                        section=claim.section,
                        text_span=(claim.span_start, claim.span_end),
                        details={"unsupported_metrics": unsupported_metrics},
                    )
                )
                continue

            # 3. Check Capability Match
            matched_cap = self._match_capability(claim, contract, target_cap)
            if not matched_cap:
                if contract.is_exhaustive:
                    reason = f"Product assertion '{claim.text}' does not match any authorized capability under exhaustive contract."
                    results.append(
                        CapabilityCheckResult(
                            claim_id=claim.claim_id,
                            is_whitelisted=False,
                            is_precondition_satisfied=False,
                            violates_exclusion=True,
                            rationale=reason,
                        )
                    )
                    findings.append(
                        VerificationFinding(
                            gate="ProductCapabilityBounds",
                            severity=VerifierFindingSeverity.ERROR,
                            code="INCOMPATIBLE_CAPABILITY",
                            message=reason,
                            claim_id=claim.claim_id,
                            section=claim.section,
                            text_span=(claim.span_start, claim.span_end),
                        )
                    )
                else:
                    reason = f"Product assertion '{claim.text}' is unverified under non-exhaustive contract."
                    results.append(
                        CapabilityCheckResult(
                            claim_id=claim.claim_id,
                            is_whitelisted=False,
                            is_precondition_satisfied=False,
                            violates_exclusion=False,
                            rationale=reason,
                        )
                    )
                    findings.append(
                        VerificationFinding(
                            gate="ProductCapabilityBounds",
                            severity=VerifierFindingSeverity.WARNING,
                            code="CONDITIONAL_CAPABILITY_UNKNOWN",
                            message=reason,
                            claim_id=claim.claim_id,
                            section=claim.section,
                            text_span=(claim.span_start, claim.span_end),
                        )
                    )
                continue

            # Capability matched and all checks passed
            results.append(
                CapabilityCheckResult(
                    claim_id=claim.claim_id,
                    capability_id=matched_cap.capability_id,
                    capability_name=matched_cap.name,
                    is_whitelisted=True,
                    is_precondition_satisfied=True,
                    violates_exclusion=False,
                    rationale=f"Claim verified against capability '{matched_cap.name}'.",
                )
            )

        return results, findings

    def _check_hard_exclusions(
        self,
        claim: DecomposedClaim,
        contract: ProductContract,
        target_cap: Optional[CapabilityNode],
    ) -> Optional[str]:
        """Check if claim asserts support for an excluded stack or technology."""
        claim_lower = claim.text.lower()
        all_exclusions = list(contract.hard_exclusions)
        if target_cap:
            all_exclusions.extend(target_cap.hard_exclusions)

        for excl in all_exclusions:
            clean_excl = excl.strip().lower()
            if not clean_excl:
                continue
            # If the exclusion keyword is mentioned in the claim as supported
            if clean_excl in claim_lower:
                return f"Claim '{claim.text}' references explicitly excluded technology/environment: '{excl}'"
        return None

    def _match_capability(
        self,
        claim: DecomposedClaim,
        contract: ProductContract,
        target_cap: Optional[CapabilityNode] = None,
    ) -> Optional[CapabilityNode]:
        """Match claim to a CapabilityNode based on name, problems, or use cases."""
        claim_lower = claim.text.lower()

        # If target_cap from Toulmin argument is provided, check if claim relates to it
        if target_cap:
            target_words = [w for w in re.findall(r"\w+", target_cap.name.lower()) if len(w) > 3]
            if target_words and any(w in claim_lower for w in target_words):
                return target_cap
            if any(pp.lower() in claim_lower for pp in target_cap.verified_proof_points):
                return target_cap
            if any(any(w in claim_lower for w in prob.lower().split() if len(w) > 3) for prob in target_cap.target_operational_problems):
                return target_cap

        # Direct name match or word overlap
        for cap in contract.capabilities:
            if cap.name.lower() in claim_lower or claim_lower in cap.name.lower():
                return cap
            cap_words = [w for w in re.findall(r"\w+", cap.name.lower()) if len(w) > 3]
            if cap_words and sum(1 for w in cap_words if w in claim_lower) >= 2:
                return cap

        # Operational problems, use cases, or verified proof points overlap
        for cap in contract.capabilities:
            for prob in cap.target_operational_problems:
                if any(word in claim_lower for word in prob.lower().split() if len(word) > 4):
                    return cap
            for uc in cap.supported_use_cases:
                if any(word in claim_lower for word in uc.lower().split() if len(word) > 4):
                    return cap
            for pp in cap.verified_proof_points:
                if any(word in claim_lower for word in pp.lower().split() if len(word) > 4):
                    return cap

        return None
