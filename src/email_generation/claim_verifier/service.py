from __future__ import annotations

import logging
from typing import List, Optional

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import ProductContext, ProductContract, ProspectContext
from src.email_generation.claim_verifier.capability import ProductCapabilityBoundsGate
from src.email_generation.claim_verifier.decomposer import ClaimDecomposer
from src.email_generation.claim_verifier.entailment import FactEntailmentGate
from src.email_generation.claim_verifier.models import (
    CapabilityCheckResult,
    DecomposedClaim,
    EntailmentCheckResult,
    EntailmentVerdict,
    ModalityCheckResult,
    VerificationFinding,
    VerifierFindingSeverity,
    VerifierResult,
)
from src.email_generation.claim_verifier.modality import ModalityPreservationGate
from src.generator.engine.engine import GeneratorEngine
from src.researcher.models import ResearchPackage
from src.signal_strategy.models import OutreachStrategy, StrategyResult, ToulminArgument

logger = logging.getLogger(__name__)


class ClaimVerifierService:
    """Four-Gate Post-Draft Claim and Capability Verifier Service.

    Orchestrates:
    - Gate 1: Atomic Claim Decomposition
    - Gate 2: Semantic Fact Entailment
    - Gate 3: Modality Preservation
    - Gate 4: Product Capability Bounds
    """

    def __init__(
        self,
        decomposer: Optional[ClaimDecomposer] = None,
        entailment_gate: Optional[FactEntailmentGate] = None,
        modality_gate: Optional[ModalityPreservationGate] = None,
        capability_gate: Optional[ProductCapabilityBoundsGate] = None,
        engine: Optional[GeneratorEngine] = None,
    ) -> None:
        self.decomposer = decomposer or ClaimDecomposer()
        self.entailment_gate = entailment_gate or FactEntailmentGate(engine=engine)
        self.modality_gate = modality_gate or ModalityPreservationGate()
        self.capability_gate = capability_gate or ProductCapabilityBoundsGate()

    def verify(
        self,
        subject: str,
        body: str,
        prospect: ProspectContext,
        product: ProductContext,
        strategy_result: StrategyResult,
        research_package: Optional[ResearchPackage] = None,
    ) -> VerifierResult:
        """Execute all four verification gates against the generated draft."""
        all_findings: List[VerificationFinding] = []
        all_claims: List[DecomposedClaim] = []
        entailment_results: List[EntailmentCheckResult] = []
        modality_results: List[ModalityCheckResult] = []
        capability_results: List[CapabilityCheckResult] = []
        unverified_claim_ids: List[str] = []

        try:
            # 1. Gate 1: Decomposition
            decomposed_claims, decomp_findings = self.decomposer.decompose(
                subject=subject,
                body=body,
                prospect_name=prospect.full_name,
                company_name=prospect.company_name,
                product_name=product.product_name,
            )
            all_claims.extend(decomposed_claims)
            all_findings.extend(decomp_findings)

            strategy = strategy_result.strategy
            toulmin: Optional[ToulminArgument] = getattr(strategy, "toulmin_argument", None) if strategy else None
            contract: ProductContract = product.get_contract()

            # 2. Gate 2: Semantic Fact Entailment
            ent_res, ent_findings = self.entailment_gate.verify(
                claims=all_claims,
                research_package=research_package,
                toulmin_argument=toulmin,
                strategy=strategy,
            )
            entailment_results.extend(ent_res)
            all_findings.extend(ent_findings)

            # 3. Gate 3: Modality Preservation
            if strategy:
                mod_res, mod_findings = self.modality_gate.verify(
                    claims=all_claims,
                    strategy=strategy,
                    toulmin_argument=toulmin,
                )
                modality_results.extend(mod_res)
                all_findings.extend(mod_findings)
            else:
                all_findings.append(
                    VerificationFinding(
                        gate="ModalityPreservation",
                        severity=VerifierFindingSeverity.ERROR,
                        code="MISSING_STRATEGY",
                        message="StrategyResult contains no strategy to verify modality against.",
                    )
                )

            # 4. Gate 4: Product Capability Bounds
            cap_res, cap_findings = self.capability_gate.verify(
                claims=all_claims,
                contract=contract,
                toulmin_argument=toulmin,
            )
            capability_results.extend(cap_res)
            all_findings.extend(cap_findings)

            # Track unverified claims (claims associated with ERROR findings, NEUTRAL entailment, or unwhitelisted capabilities)
            error_claim_ids = {
                f.claim_id for f in all_findings
                if f.claim_id and f.severity == VerifierFindingSeverity.ERROR
            }
            neutral_claim_ids = {
                res.claim_id for res in entailment_results
                if res.verdict == EntailmentVerdict.NEUTRAL
            }
            unwhitelisted_claim_ids = {
                res.claim_id for res in capability_results
                if not res.is_whitelisted or res.violates_exclusion
            }
            unverified_set = error_claim_ids | neutral_claim_ids | unwhitelisted_claim_ids
            unverified_claim_ids.extend(sorted(list(unverified_set)))

        except Exception as exc:
            logger.error("Unexpected failure in ClaimVerifierService: %s", exc, exc_info=True)
            all_findings.append(
                VerificationFinding(
                    gate="ClaimVerifierService",
                    severity=VerifierFindingSeverity.ERROR,
                    code="VERIFIER_EXECUTION_FAILURE",
                    message=f"Critical failure during post-draft verification: {exc}",
                )
            )

        has_errors = any(f.severity == VerifierFindingSeverity.ERROR for f in all_findings)
        passed = not has_errors

        return VerifierResult(
            passed=passed,
            findings=all_findings,
            decomposed_claims=all_claims,
            entailment_results=entailment_results,
            modality_results=modality_results,
            capability_results=capability_results,
            unverified_claim_ids=unverified_claim_ids,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                component="claim_verifier_service",
            ),

        )
