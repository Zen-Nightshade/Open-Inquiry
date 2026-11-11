from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.config.settings import GROQ_LARGE
from src.context.models import (
    CapabilityNode,
    CompatibilityStatus,
    OutreachContext,
    ProductContract,
    ProductContext,
    ProspectContext,
)
from src.generator.decoding import DecodeConfig
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.prompting.message import Message, Role
from src.generator.prompting.prompt import Prompt
from src.generator.structured_output.pydantic_parser import PydanticParser
from src.researcher.models import CanonicalFact, ConfidenceLevel, ResearchPackage, VerificationStatus
from .models import (
    CTAType,
    EligibilityStatus,
    OutreachStrategy,
    ProspectProductAlignment,
    RankedSignal,
    StrategyResult,
    ToulminArgument,
    ToulminBacking,
    ToulminClaim,
    ToulminGrounds,
    ToulminQualifier,
    ToulminRebuttal,
    ToulminWarrant,
)

logger = logging.getLogger(__name__)


class RawStrategyOutput(BaseModel):
    """Raw LLM strategy blueprint output."""

    model_config = ConfigDict(extra="ignore")

    prospect_role_relevance: str = Field(description="Why this specific individual cares about this event given their title/role")
    business_implications: str = Field(description="Downstream operational or commercial challenges/pressures for their company")
    capability_mapping: List[str] = Field(description="Exact product capabilities solving this problem (must come from product capabilities)")
    messaging_approach: str = Field(description="Strategic framing approach (e.g. 'challenge-led', 'expansion-scaling', 'peer-proof')")
    central_message: str = Field(description="One-sentence core proposition connecting the signal to product value")
    suggested_subject_line: str = Field(description="Recommended concise, conversational subject line")
    cta: str = Field(description="Specific, low-friction next action")
    tone_guidance: str = Field(description="Tone guidance matching recipient and relationship")
    constraint_checklist: List[str] = Field(default_factory=list, description="Explicit constraints to enforce during email drafting")
    warrant_reasoning: Optional[str] = Field(default=None, description="Logical argument connecting observed event to product relevance")
    proposed_proof_points: List[str] = Field(default_factory=list, description="Whitelisted proof points cited from the product contract")


STRATEGY_BUILDER_SYSTEM_PROMPT = """You are a master B2B outreach strategist formulating an outreach blueprint.
Your task is to take the primary commercial signal and translate it into a compelling, relevant, and evidence-grounded strategy blueprint for email generation.

CRITICAL GUARDRAILS:
1. STRICT CAPABILITY BOUNDING:
   - Every capability in capability_mapping MUST be selected from the provided Product Capabilities list.
   - Do NOT invent features, SLAs, integrations, or capabilities not explicitly listed in the product context.

2. COMMERCIAL OPPORTUNITY LOGIC:
   - Central message must explain WHY this specific event creates an operational need that our product solves.
   - Do NOT send a generic template or a lazy "saw you raised money, want to chat?" email.
   - Ground the hook in the observed event, transition through the business implication, and introduce the solution.

3. CONVERSATIONAL & LOW-FRICTION CTA:
   - Propose an easy, conversational next step (e.g. 15-minute introductory call, sharing benchmark data, or brief response).
   - Follow any desired_cta in the outreach context.
"""


class StrategyBuilder:
    """Builds a verified, structured OutreachStrategy from ranked commercial signals."""

    def __init__(
        self,
        engine: Optional[GeneratorEngine] = None,
        provider_name: Optional[str] = None,
        min_composite_score: float = 0.35,
        min_product_alignment: float = 0.25,
    ) -> None:
        self.provider_name = provider_name or GROQ_LARGE
        self.engine = engine or GeneratorEngine(default_provider=self.provider_name)
        self.min_composite_score = min_composite_score
        self.min_product_alignment = min_product_alignment
        self.parser = PydanticParser(RawStrategyOutput)
        self._last_validation_error: Optional[str] = None

    def build_strategy(
        self,
        ranked_signals: Sequence[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        research_package: Optional[ResearchPackage] = None,
    ) -> StrategyResult:
        """Evaluate viability and construct an immutable OutreachStrategy blueprint."""
        self._last_validation_error = None
        # Check 1: Any signals exist
        if not ranked_signals:
            logger.info("Strategy formulation aborted: zero ranked signals provided.")
            return StrategyResult(
                is_viable=False,
                rejection_reason="No candidate signals available from research.",
                unviable_reasons=["The research package contains no facts that could be synthesized into candidate signals."],
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="strategy_builder",
                    rule_id="empty_ranked_signals_check",
                ),
            )

        # Check 2: Separate candidates into verified-compatible and conditional pools
        verified_candidates = [
            s for s in ranked_signals
            if s.eligibility is None or s.eligibility.is_verified_eligible
        ]
        conditional_candidates = [
            s for s in ranked_signals
            if s.eligibility is not None and s.eligibility.requires_diagnostic_strategy
        ]

        if not verified_candidates and not conditional_candidates:
            logger.info("Strategy formulation aborted: zero eligible or conditional signals available.")
            rejection_reasons = []
            for s in ranked_signals:
                if s.eligibility and s.eligibility.rejection_reasons:
                    rejection_reasons.extend(s.eligibility.rejection_reasons)
                elif s.rationale:
                    rejection_reasons.append(s.rationale)

            primary_reason = rejection_reasons[0] if rejection_reasons else "All candidate signals failed deterministic eligibility gates."
            return StrategyResult(
                is_viable=False,
                ranked_signals=list(ranked_signals),
                rejection_reason=primary_reason,
                unviable_reasons=rejection_reasons or [primary_reason],
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="strategy_builder",
                    rule_id="no_eligible_signals_abstention",
                ),
            )

        # Path 1: Normal Strategy Path (verified-compatible signals)
        normal_candidate: Optional[RankedSignal] = None
        normal_unviable_reasons: List[str] = []

        if verified_candidates:
            top_verified = verified_candidates[0]
            if top_verified.score.product_alignment is None:
                normal_unviable_reasons.append(
                    f"Top verified signal '{top_verified.signal.label}' product alignment is unevaluated; "
                    "cannot confirm product relevance."
                )
            elif top_verified.score.product_alignment < self.min_product_alignment:
                normal_unviable_reasons.append(
                    f"Top verified signal '{top_verified.signal.label}' product alignment ({top_verified.score.product_alignment:.2f}) "
                    f"is below minimum alignment threshold ({self.min_product_alignment:.2f})."
                )

            if top_verified.score.composite_score < self.min_composite_score:
                normal_unviable_reasons.append(
                    f"Top verified signal '{top_verified.signal.label}' composite score ({top_verified.score.composite_score:.2f}) "
                    f"is below minimum viability threshold ({self.min_composite_score:.2f})."
                )

            if not normal_unviable_reasons:
                normal_candidate = top_verified

        if normal_candidate is not None:
            top_signal = normal_candidate
            supporting_signals = [
                s for s in verified_candidates[1:3]
                if s.score.composite_score >= 0.30
                and s.score.product_alignment is not None
                and s.score.product_alignment >= 0.20
            ]
            strategy = self._synthesize_strategy_with_llm(
                primary_signal=top_signal,
                supporting_signals=supporting_signals,
                prospect=prospect,
                product=product,
                outreach=outreach,
                research_package=research_package,
                is_conditional=False,
                unresolved_conditions=[],
            )
            if not strategy:
                strategy = self._deterministic_fallback_strategy(
                    primary_signal=top_signal,
                    supporting_signals=supporting_signals,
                    prospect=prospect,
                    product=product,
                    outreach=outreach,
                    research_package=research_package,
                    is_conditional=False,
                    unresolved_conditions=[],
                )
            if not strategy:
                err_detail = self._last_validation_error or "invalid essential references or unresolvable capabilities."
                return StrategyResult(
                    is_viable=False,
                    ranked_signals=list(ranked_signals),
                    rejection_reason=f"Toulmin argument reference validation failed: {err_detail}",
                    unviable_reasons=[f"Toulmin argument reference validation failed: {err_detail}"],
                    provenance=OutputProvenance(
                        origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                        component="strategy_builder",
                        rule_id="toulmin_reference_validation_failure",
                    ),
                )
            return StrategyResult(
                is_viable=True,
                strategy=strategy,
                ranked_signals=list(ranked_signals),
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="strategy_builder",
                    rule_id="strategy_viability_confirmed",
                ),
            )

        # Path 2: Diagnostic Strategy Path (conditional signals when no verified-compatible strategy exists)
        diagnostic_candidate: Optional[RankedSignal] = None
        diagnostic_unviable_reasons: List[str] = []

        if conditional_candidates:
            top_conditional = conditional_candidates[0]
            if top_conditional.score.composite_score < self.min_composite_score:
                diagnostic_unviable_reasons.append(
                    f"Top conditional signal '{top_conditional.signal.label}' score ({top_conditional.score.composite_score:.2f}) "
                    f"is below minimum viability threshold ({self.min_composite_score:.2f})."
                )
            else:
                diagnostic_candidate = top_conditional

        if diagnostic_candidate is not None:
            top_signal = diagnostic_candidate
            unresolved_conditions = list(top_signal.eligibility.unresolved_conditions) if top_signal.eligibility else []
            supporting_signals = [
                s for s in conditional_candidates[1:3]
                if s.score.composite_score >= 0.30
            ]
            strategy = self._synthesize_strategy_with_llm(
                primary_signal=top_signal,
                supporting_signals=supporting_signals,
                prospect=prospect,
                product=product,
                outreach=outreach,
                research_package=research_package,
                is_conditional=True,
                unresolved_conditions=unresolved_conditions,
            )
            if not strategy:
                strategy = self._deterministic_fallback_strategy(
                    primary_signal=top_signal,
                    supporting_signals=supporting_signals,
                    prospect=prospect,
                    product=product,
                    outreach=outreach,
                    research_package=research_package,
                    is_conditional=True,
                    unresolved_conditions=unresolved_conditions,
                )
            if not strategy:
                err_detail = self._last_validation_error or "invalid essential references or unresolvable capabilities."
                return StrategyResult(
                    is_viable=False,
                    ranked_signals=list(ranked_signals),
                    rejection_reason=f"Toulmin argument reference validation failed: {err_detail}",
                    unviable_reasons=[f"Toulmin argument reference validation failed: {err_detail}"],
                    provenance=OutputProvenance(
                        origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                        component="strategy_builder",
                        rule_id="toulmin_reference_validation_failure",
                    ),
                )
            return StrategyResult(
                is_viable=True,
                strategy=strategy,
                ranked_signals=list(ranked_signals),
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                    component="strategy_builder",
                    rule_id="strategy_viability_confirmed",
                ),
            )

        # Abstention: neither verified-compatible nor diagnostic strategy is viable
        all_unviable = normal_unviable_reasons + diagnostic_unviable_reasons
        if not all_unviable:
            all_unviable = ["Neither verified-compatible nor diagnostic candidate signals meet commercial viability thresholds."]
        logger.info(f"Strategy unviable: {all_unviable[0]}")
        return StrategyResult(
            is_viable=False,
            ranked_signals=list(ranked_signals),
            rejection_reason=all_unviable[0],
            unviable_reasons=all_unviable,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                component="strategy_builder",
                rule_id="strategy_viability_check",
            ),
        )

    def _synthesize_strategy_with_llm(
        self,
        primary_signal: RankedSignal,
        supporting_signals: List[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        research_package: Optional[ResearchPackage] = None,
        is_conditional: bool = False,
        unresolved_conditions: Optional[List[str]] = None,
    ) -> Optional[OutreachStrategy]:
        """Invoke LLM to produce structured strategy blueprint with capability bounding."""
        prompt = Prompt()
        prompt.add_message(Message(role=Role.SYSTEM, content=STRATEGY_BUILDER_SYSTEM_PROMPT))
        prompt.add_message(
            Message(
                role=Role.USER,
                content=self._build_user_prompt(
                    primary_signal=primary_signal,
                    supporting_signals=supporting_signals,
                    prospect=prospect,
                    product=product,
                    outreach=outreach,
                    is_conditional=is_conditional,
                    unresolved_conditions=unresolved_conditions,
                ),
            )
        )

        request = GenerationRequest(
            prompt=prompt,
            decoding=DecodeConfig(temperature=0.2, max_tokens=2048),
            provider_name=self.provider_name,
            output_schema=self.parser,
            metadata={"task": "strategy_building"},
        )

        try:
            response = self.engine.generate(request)
            raw = response.parsed_output
            if isinstance(raw, RawStrategyOutput):
                # Enforce capability bounding against product.capabilities
                bounded_capabilities = self._enforce_capability_bounding(raw.capability_mapping, product.capabilities)

                argument, err = self._bind_and_validate_toulmin_argument(
                    primary_signal=primary_signal,
                    product=product,
                    prospect=prospect,
                    research_package=research_package,
                    candidate_capabilities=bounded_capabilities if bounded_capabilities else raw.capability_mapping,
                    raw_warrant_reasoning=raw.warrant_reasoning,
                    raw_proof_points=raw.proposed_proof_points,
                    is_conditional=is_conditional,
                    unresolved_conditions=unresolved_conditions,
                    raw_claim_text=raw.central_message,
                )
                if argument is None:
                    logger.warning(f"Toulmin argument validation failed during LLM synthesis: {err}")
                    return None

                cta_type, cta_text, cta_rationale = self.evaluate_cta_policy(
                    primary_signal=primary_signal,
                    is_conditional=is_conditional,
                    unresolved_conditions=unresolved_conditions or [],
                    outreach=outreach,
                    target_capability_name=argument.warrant.capability_name,
                )

                alignment = ProspectProductAlignment(
                    prospect_role_relevance=raw.prospect_role_relevance.strip(),
                    business_implications=raw.business_implications.strip(),
                    capability_mapping=bounded_capabilities if bounded_capabilities else [argument.warrant.capability_name],
                    is_conditional=is_conditional,
                    unresolved_conditions=unresolved_conditions or [],
                )

                # Assemble constraint checklist incorporating outreach constraints
                constraints = list(outreach.constraints)
                for c in raw.constraint_checklist:
                    if c not in constraints:
                        constraints.append(c)

                if is_conditional and unresolved_conditions:
                    for cond in unresolved_conditions:
                        non_assert = f"Do not assert verified support for '{argument.warrant.capability_name}'; prerequisite '{cond}' remains unverified."
                        if non_assert not in constraints:
                            constraints.append(non_assert)
                    inquiry_rule = "Frame outreach as an exploratory inquiry rather than confirmed capability alignment."
                    if inquiry_rule not in constraints:
                        constraints.append(inquiry_rule)

                if argument.rebuttal.missing_backing:
                    unbacked_warn = "Do not assert unverified performance metrics or quantitative ROI; backing proof is unavailable without whitelisted proof."
                    if unbacked_warn not in constraints:
                        constraints.append(unbacked_warn)

                for excl in argument.rebuttal.hard_exclusions:
                    excl_rule = f"Strict product boundary: do not claim support for '{excl}'."
                    if excl_rule not in constraints:
                        constraints.append(excl_rule)

                exec_id = response.metadata.get("execution_id") if hasattr(response, "metadata") and response.metadata else None
                model_name = getattr(response, "model_name", None)
                prov = OutputProvenance(
                    origin=ProvenanceOrigin.MODEL,
                    component="strategy_builder",
                    model_name=model_name,
                    provider=self.provider_name,
                    model_execution_id=exec_id,
                )

                return OutreachStrategy(
                    primary_signal=primary_signal,
                    supporting_signals=supporting_signals,
                    alignment=alignment,
                    messaging_approach="diagnostic" if is_conditional else raw.messaging_approach.strip(),
                    central_message=argument.claim.text,
                    suggested_subject_line=raw.suggested_subject_line.strip(),
                    cta=cta_text,
                    cta_type=cta_type,
                    cta_rationale=cta_rationale,
                    tone_guidance=raw.tone_guidance.strip() if raw.tone_guidance else f"Professional, direct {outreach.tone} tone",
                    constraint_checklist=constraints,
                    is_conditional=is_conditional,
                    unresolved_conditions=unresolved_conditions or [],
                    toulmin_argument=argument,
                    provenance=prov,
                )
        except Exception as e:
            logger.warning(f"Error during LLM strategy formulation: {e}. Executing deterministic fallback.")

        return None

    def _build_user_prompt(
        self,
        primary_signal: RankedSignal,
        supporting_signals: List[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        is_conditional: bool = False,
        unresolved_conditions: Optional[List[str]] = None,
    ) -> str:
        lines = [
            "### Prospect Information",
            f"- Full Name: {prospect.full_name}",
            f"- Title / Role: {prospect.designation or 'Executive'}",
            f"- Company: {prospect.company_name}",
            f"- Domain: {prospect.company_domain or 'Unknown'}",
            "",
            "### Product Offering",
            f"- Product Name: {product.product_name}",
            f"- Description: {product.description}",
            "- Product Capabilities (MUST SELECT FROM THESE):",
        ]
        for cap in product.capabilities:
            lines.append(f"  * {cap}")

        lines.extend([
            f"- Value Proposition: {product.value_proposition}",
            f"- Differentiators: {', '.join(product.differentiators) if product.differentiators else 'None'}",
            f"- Boundaries / Limitations: {', '.join(product.limitations) if product.limitations else 'None'}",
            "",
            "### Outreach Guidelines",
            f"- Objective: {outreach.objective}",
            f"- Sender Identity: {outreach.sender_identity}",
            f"- Desired CTA: {outreach.desired_cta or 'Flexible conversational next step'}",
            f"- Tone: {outreach.tone}",
            f"- Relationship: {outreach.relationship}",
            f"- Existing Constraints: {', '.join(outreach.constraints) if outreach.constraints else 'Standard B2B professional limits'}",
            "",
            "### Primary Lead Signal",
            f"- Label: {primary_signal.signal.label}",
            f"- Category: {primary_signal.signal.fact_category.value}",
            f"- Description: {primary_signal.signal.description}",
            f"- Observed Event: {primary_signal.signal.observed_event or primary_signal.signal.description}",
            f"- Business Implication: {primary_signal.signal.business_implication or 'Unspecified'}",
            f"- Commercial Opportunity: {primary_signal.signal.commercial_opportunity or 'Unspecified'}",
            f"- Score: Composite={primary_signal.score.composite_score}, Product Alignment={primary_signal.score.product_alignment}",
            f"- Rationale: {primary_signal.rationale}",
            "",
        ])

        if supporting_signals:
            lines.append("### Secondary Context Signals:")
            for s in supporting_signals:
                lines.append(f"- {s.signal.label}: {s.signal.description}")
            lines.append("")

        if is_conditional and unresolved_conditions:
            lines.append("### CRITICAL CONDITIONAL ALIGNMENT GUIDANCE:")
            lines.append("- Product compatibility is CONDITIONAL on unverified technical prerequisites:")
            for cond in unresolved_conditions:
                lines.append(f"  * {cond}")
            lines.append("- Do NOT assert or imply that our product definitely works with their setup.")
            lines.append("- The outreach MUST be formulated as an exploratory or diagnostic inquiry.")
            lines.append("")

        lines.append("Formulate the outreach strategy blueprint connecting this primary signal to our product capabilities.")
        return "\n".join(lines)

    def _enforce_capability_bounding(
        self,
        candidate_capabilities: List[str],
        valid_capabilities: List[str],
    ) -> List[str]:
        """Strictly bound capabilities to the documented product capabilities.
        
        Ambiguous or undeclared capabilities are stripped rather than silently substituted.
        """
        if not valid_capabilities:
            return []

        bounded: List[str] = []
        valid_lower = {v.lower().strip(): v for v in valid_capabilities}

        for cand in candidate_capabilities:
            cand_clean = cand.lower().strip()
            # Exact match
            if cand_clean in valid_lower:
                matched = valid_lower[cand_clean]
                if matched not in bounded:
                    bounded.append(matched)
                continue

            # Substring match
            for v_low, original in valid_lower.items():
                if v_low in cand_clean or cand_clean in v_low:
                    if original not in bounded:
                        bounded.append(original)
                    break

        if not bounded:
            logger.info("No LLM-generated capabilities matched product capabilities.")

        return bounded

    def _resolve_capability(
        self,
        candidate_names: Sequence[str],
        contract: ProductContract,
    ) -> Optional[CapabilityNode]:
        """Resolve candidate capability names or IDs to an authoritative CapabilityNode.
        
        Strictly avoids silent fallback to contract.capabilities[0].
        """
        for cand in candidate_names:
            if not cand or not cand.strip():
                continue
            cand_clean = cand.strip().lower()

            # 1. Match by capability_id
            for cap in contract.capabilities:
                if cap.capability_id.lower() == cand_clean:
                    return cap

            # 2. Match by exact capability name
            for cap in contract.capabilities:
                if cap.name.strip().lower() == cand_clean:
                    return cap

            # 3. Match by substring in name
            for cap in contract.capabilities:
                c_name = cap.name.strip().lower()
                if c_name in cand_clean or cand_clean in c_name:
                    return cap

            # 4. Match by supported use cases
            for cap in contract.capabilities:
                for uc in cap.supported_use_cases:
                    uc_clean = uc.strip().lower()
                    if uc_clean == cand_clean or uc_clean in cand_clean or cand_clean in uc_clean:
                        return cap

        return None

    def _filter_proof_points(
        self,
        proposed_proof_points: Optional[List[str]],
        matched_cap: CapabilityNode,
    ) -> List[str]:
        """Strictly bound proof points to the capability's verified whitelist."""
        whitelist = matched_cap.verified_proof_points
        if not whitelist:
            return []

        if proposed_proof_points is None:
            # If not explicitly specified by caller/LLM, retain all verified proof points
            return list(whitelist)

        whitelist_map = {vp.strip().lower(): vp for vp in whitelist if vp.strip()}
        bounded: List[str] = []

        for p in proposed_proof_points:
            p_clean = p.strip().lower()
            if not p_clean:
                continue
            if p_clean in whitelist_map:
                canonical = whitelist_map[p_clean]
                if canonical not in bounded:
                    bounded.append(canonical)
                continue
            for vp_clean, canonical in whitelist_map.items():
                if vp_clean in p_clean or p_clean in vp_clean:
                    if canonical not in bounded:
                        bounded.append(canonical)
                    break

        return bounded

    def _qualify_claim_text(
        self,
        claim_text: str,
        has_whitelisted_proof: bool,
    ) -> str:
        """Ensure outcome claims are properly qualified if backing proof is missing."""
        if has_whitelisted_proof:
            return claim_text

        unbacked_patterns = [
            r"\b\d+%\b",
            r"\bguarantee(?:s|d)?\b",
            r"\bproven\s+roi\b",
            r"\bcost\s+reduction\s+of\b",
            r"\bsaves?\s+\$\d+",
        ]
        has_unbacked = any(re.search(pat, claim_text, re.IGNORECASE) for pat in unbacked_patterns)
        if has_unbacked:
            clean = re.sub(r"\bguarantees?\b", "helps address", claim_text, flags=re.IGNORECASE)
            clean = re.sub(r"\bby\s+\d+%\s*", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\b\d+%\s*", "", clean)
            clean = re.sub(r"\s{2,}", " ", clean).strip()
            return clean

        return claim_text

    def evaluate_cta_policy(
        self,
        primary_signal: RankedSignal,
        is_conditional: bool,
        unresolved_conditions: List[str],
        outreach: OutreachContext,
        target_capability_name: Optional[str] = None,
    ) -> Tuple[CTAType, str, str]:
        """Deterministic CTA policy selecting between VALUE_EXCHANGE, BINARY_CHECK, and DIAGNOSTIC_QUESTION."""
        if is_conditional:
            # High-commitment or unverified CTAs must be rejected/replaced
            if len(unresolved_conditions) == 1:
                cond = unresolved_conditions[0]
                cta_type = CTAType.BINARY_CHECK
                cta_text = f"Quick question: are you currently utilizing {cond} in your environment?"
                cta_rationale = (
                    f"Selected BINARY_CHECK because compatibility is conditional on a single unverified prerequisite: '{cond}'."
                )
            else:
                cta_type = CTAType.DIAGNOSTIC_QUESTION
                cta_text = "How is your team currently approaching this area given your current architecture?"
                cta_rationale = (
                    f"Selected DIAGNOSTIC_QUESTION because multiple or broad unknown technical conditions exist "
                    f"({len(unresolved_conditions)} unverified prerequisites)."
                )
            return cta_type, cta_text, cta_rationale

        # Verified-compatible path
        cta_type = CTAType.VALUE_EXCHANGE
        if outreach.desired_cta:
            cta_text = outreach.desired_cta
        else:
            cap_str = target_capability_name or "our solution"
            cta_text = f"Would you be open to a brief 15-minute conversation next week to compare notes on {cap_str}?"
        cta_rationale = "Selected VALUE_EXCHANGE for verified-compatible signal with direct evidence support."

        return cta_type, cta_text, cta_rationale

    def _bind_and_validate_toulmin_argument(
        self,
        primary_signal: RankedSignal,
        product: ProductContext,
        prospect: ProspectContext,
        research_package: Optional[ResearchPackage] = None,
        candidate_capabilities: Optional[List[str]] = None,
        raw_warrant_reasoning: Optional[str] = None,
        raw_proof_points: Optional[List[str]] = None,
        is_conditional: bool = False,
        unresolved_conditions: Optional[List[str]] = None,
        raw_claim_text: Optional[str] = None,
        explicit_evidence_ids: Optional[List[str]] = None,
    ) -> Tuple[Optional[ToulminArgument], Optional[str]]:
        """Construct and strictly validate a 6-part Toulmin commercial argument against domain models."""
        sig = primary_signal.signal
        contract = product.get_contract()
        unresolved_list = list(unresolved_conditions) if unresolved_conditions else []

        def fail(err_msg: str) -> Tuple[None, str]:
            self._last_validation_error = err_msg
            return None, err_msg

        # 1. Grounds Validation
        fact_ids = list(sig.supporting_fact_ids)
        if not fact_ids:
            return fail("Candidate signal contains no supporting fact references.")

        evidence_ids: List[str] = []
        source_domains: List[str] = []
        verification_status = VerificationStatus.INFERRED if sig.is_inferred else VerificationStatus.SUPPORTED
        composite_confidence = ConfidenceLevel.MEDIUM

        if research_package is not None:
            pkg_facts = {f.fact_id: f for f in research_package.facts}
            for fid in fact_ids:
                if fid not in pkg_facts:
                    return fail(f"Referenced fact '{fid}' does not exist in research package facts.")

            referenced_facts = [pkg_facts[fid] for fid in fact_ids]

            # Entity boundary check
            if research_package.company_entity and research_package.company_entity.entity_id:
                company_id = research_package.company_entity.entity_id
                prospect_id = research_package.prospect_entity.entity_id if research_package.prospect_entity else None
                for f in referenced_facts:
                    if f.entity_ids:
                        if company_id not in f.entity_ids and (prospect_id is None or prospect_id not in f.entity_ids):
                            return fail(
                                f"Referenced fact '{f.fact_id}' entity IDs {f.entity_ids} do not match "
                                f"target company entity '{company_id}'."
                            )

                if sig.associated_entity_ids:
                    for eid in sig.associated_entity_ids:
                        if eid != company_id and (prospect_id is None or eid != prospect_id):
                            return fail(
                                f"Signal associated entity ID '{eid}' does not match target company entity '{company_id}'."
                            )

            # Evidence references validation
            valid_ev_ids_for_facts = {ev.evidence_id for f in referenced_facts for ev in f.evidence_records}
            all_pkg_ev_ids = {ev.evidence_id for f in research_package.facts for ev in f.evidence_records}

            if explicit_evidence_ids is not None:
                for evid in explicit_evidence_ids:
                    if evid not in all_pkg_ev_ids:
                        return fail(f"Referenced evidence ID '{evid}' does not exist in research package.")
                    if evid not in valid_ev_ids_for_facts:
                        return fail(f"Referenced evidence ID '{evid}' does not support the fact it is attached to.")
                evidence_ids = list(explicit_evidence_ids)
            else:
                evidence_ids = list(valid_ev_ids_for_facts)

            # Source domains
            source_map = {s.source_id: s.domain for s in research_package.sources}
            for f in referenced_facts:
                for ev in f.evidence_records:
                    if ev.source_id in source_map:
                        domain = source_map[ev.source_id]
                        if domain not in source_domains:
                            source_domains.append(domain)

            # Confidence & verification status
            confidences = [f.composite_confidence for f in referenced_facts]
            if ConfidenceLevel.LOW in confidences or ConfidenceLevel.UNRESOLVED in confidences:
                composite_confidence = ConfidenceLevel.LOW
            elif ConfidenceLevel.MEDIUM in confidences:
                composite_confidence = ConfidenceLevel.MEDIUM
            else:
                composite_confidence = ConfidenceLevel.HIGH

            if sig.is_inferred:
                verification_status = VerificationStatus.INFERRED
            else:
                verification_status = VerificationStatus.SUPPORTED

        observed_event_text = sig.observed_event or sig.description
        grounds = ToulminGrounds(
            fact_ids=fact_ids,
            evidence_ids=evidence_ids,
            observed_event=observed_event_text,
            verification_status=verification_status,
            composite_confidence=composite_confidence,
            source_domains=source_domains,
        )

        # 2. Warrant Validation (Authoritative Capability Resolution)
        candidate_resolution_targets: List[str] = []
        if primary_signal.eligibility and primary_signal.eligibility.matched_capability_id:
            candidate_resolution_targets.append(primary_signal.eligibility.matched_capability_id)
        if primary_signal.eligibility and primary_signal.eligibility.matched_capability_name:
            candidate_resolution_targets.append(primary_signal.eligibility.matched_capability_name)
        if sig.target_capability:
            candidate_resolution_targets.append(sig.target_capability)
        if candidate_capabilities:
            candidate_resolution_targets.extend(candidate_capabilities)

        matched_cap = self._resolve_capability(candidate_resolution_targets, contract)
        if matched_cap is None:
            return fail(
                f"Candidate capabilities {candidate_resolution_targets} could not be resolved to any declared "
                f"CapabilityNode in ProductContract. Ambiguous or undeclared capabilities cannot be silently substituted."
            )

        reasoning = (
            raw_warrant_reasoning
            or sig.business_implication
            or primary_signal.rationale
            or f"Connects {sig.label} to {matched_cap.name}."
        )

        # Constrain ToulminWarrant.reasoning to qualitative functional connections rather than ungrounded empirical metrics
        empirical_pattern = re.compile(r"(?:\b\d+%\s*|\$\s*\d+[\d,\.]*[kmbtKMBT]?|\b\d+x\b)")
        whitelisted_pp_text = " ".join(matched_cap.verified_proof_points).lower()
        metrics_in_reasoning = empirical_pattern.findall(reasoning)
        ungrounded_in_reasoning = [m for m in metrics_in_reasoning if m.lower() not in whitelisted_pp_text]
        if ungrounded_in_reasoning:
            logger.warning(
                "Toulmin warrant reasoning contains ungrounded empirical metrics %s not in verified proof points; constraining to qualitative functional connection.",
                ungrounded_in_reasoning,
            )
            for ugm in ungrounded_in_reasoning:
                reasoning = reasoning.replace(ugm, "significantly ")
            reasoning = re.sub(r"\s+", " ", reasoning)
            if not reasoning.strip():
                reasoning = f"Connects {sig.label} functionally to {matched_cap.name}."

        warrant = ToulminWarrant(
            capability_id=matched_cap.capability_id,
            capability_name=matched_cap.name,
            reasoning=reasoning.strip(),
            is_inferred=True,
            limitations=list(matched_cap.hard_exclusions),
        )

        # 3. Backing Validation (Whitelisted Proof Points)
        whitelisted_proofs = self._filter_proof_points(raw_proof_points, matched_cap)
        has_whitelisted_proof = bool(whitelisted_proofs)

        backing = ToulminBacking(
            capability_id=matched_cap.capability_id,
            whitelisted_proof_points=whitelisted_proofs,
            has_whitelisted_proof=has_whitelisted_proof,
            provenance=matched_cap.provenance,
        )

        # 4. Qualifier
        elig_status = primary_signal.eligibility.status if primary_signal.eligibility else EligibilityStatus.ELIGIBLE
        compat_status = CompatibilityStatus.CONDITIONAL_ON_UNKNOWN if is_conditional else CompatibilityStatus.VERIFIED_COMPATIBLE

        if is_conditional:
            qual_lang = (
                f"Conditional inquiry: support for '{matched_cap.name}' is subject to verifying {', '.join(unresolved_list)}."
            )
        elif not has_whitelisted_proof:
            qual_lang = f"Verified capability alignment for '{matched_cap.name}', without independent outcome proof."
        else:
            qual_lang = f"Verified capability alignment for '{matched_cap.name}' with documented proof points."

        qualifier = ToulminQualifier(
            eligibility_status=elig_status,
            compatibility_status=compat_status,
            verification_status=verification_status,
            composite_confidence=composite_confidence,
            unresolved_prerequisites=unresolved_list,
            qualification_language=qual_lang,
        )

        # 5. Claim
        claim_type = "diagnostic_inquiry" if is_conditional else "value_assertion"
        if is_conditional:
            cond_str = ", ".join(unresolved_list) if unresolved_list else "technical prerequisites"
            claim_text = (
                f"Exploring whether {prospect.company_name} could benefit from {matched_cap.name} "
                f"to address {sig.label.lower()}, subject to {cond_str}."
            )
        else:
            if raw_claim_text:
                claim_text = self._qualify_claim_text(raw_claim_text, has_whitelisted_proof)
            else:
                base_claim = f"{product.product_name} helps teams at {prospect.company_name} address {sig.label.lower()} through {matched_cap.name}."
                claim_text = self._qualify_claim_text(base_claim, has_whitelisted_proof)

        claim = ToulminClaim(
            claim_type=claim_type,
            text=claim_text.strip(),
            target_capability=matched_cap.name,
            is_qualified=True,
        )

        # 6. Rebuttal
        contradictory_facts: List[str] = []
        if research_package is not None:
            for f in research_package.facts:
                if f.composite_confidence == ConfidenceLevel.UNRESOLVED:
                    contradictory_facts.append(f.fact_id)

        all_exclusions = list(matched_cap.hard_exclusions)
        for excl in contract.hard_exclusions:
            if excl not in all_exclusions:
                all_exclusions.append(excl)

        limitations = list(all_exclusions)
        for lim in product.limitations:
            if lim not in limitations:
                limitations.append(lim)

        rebuttal = ToulminRebuttal(
            contradictory_fact_ids=contradictory_facts,
            unresolved_prerequisites=unresolved_list,
            hard_exclusions=all_exclusions,
            missing_backing=not has_whitelisted_proof,
            limitations=limitations,
        )

        toulmin_argument = ToulminArgument(
            grounds=grounds,
            warrant=warrant,
            backing=backing,
            qualifier=qualifier,
            claim=claim,
            rebuttal=rebuttal,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.DETERMINISTIC_RULE,
                component="strategy_builder",
                rule_id="toulmin_argument_binding",
            ),
        )

        return toulmin_argument, None

    def _deterministic_fallback_strategy(
        self,
        primary_signal: RankedSignal,
        supporting_signals: List[RankedSignal],
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        research_package: Optional[ResearchPackage] = None,
        is_conditional: bool = False,
        unresolved_conditions: Optional[List[str]] = None,
    ) -> Optional[OutreachStrategy]:
        """Deterministic strategy construction when LLM synthesis is unparseable or fails."""
        sig = primary_signal.signal
        contract = product.get_contract()

        candidate_caps: List[str] = []
        if primary_signal.eligibility and primary_signal.eligibility.matched_capability_name:
            candidate_caps.append(primary_signal.eligibility.matched_capability_name)
        elif sig.target_capability:
            candidate_caps.append(sig.target_capability)
        elif contract.capabilities:
            candidate_caps.append(contract.capabilities[0].name)

        unresolved_list = list(unresolved_conditions) if unresolved_conditions else []

        argument, err = self._bind_and_validate_toulmin_argument(
            primary_signal=primary_signal,
            product=product,
            prospect=prospect,
            research_package=research_package,
            candidate_capabilities=candidate_caps,
            is_conditional=is_conditional,
            unresolved_conditions=unresolved_list,
        )
        if argument is None:
            logger.warning(f"Toulmin argument validation failed during deterministic fallback: {err}")
            return None

        cta_type, cta_text, cta_rationale = self.evaluate_cta_policy(
            primary_signal=primary_signal,
            is_conditional=is_conditional,
            unresolved_conditions=unresolved_list,
            outreach=outreach,
            target_capability_name=argument.warrant.capability_name,
        )

        alignment = ProspectProductAlignment(
            prospect_role_relevance=f"As {prospect.designation or 'leader'} at {prospect.company_name}, this event directly impacts operational priorities.",
            business_implications=sig.business_implication or f"Expansion and operational requirements driven by {sig.label}.",
            capability_mapping=candidate_caps if candidate_caps else [argument.warrant.capability_name],
            is_conditional=is_conditional,
            unresolved_conditions=unresolved_list,
        )

        if is_conditional and unresolved_list:
            subject = f"{prospect.company_name} — exploring {sig.label}"
            messaging_approach = "diagnostic"
        else:
            subject = f"{prospect.company_name} — {sig.label}"
            messaging_approach = "event-hook"

        constraints = list(outreach.constraints)
        if "Keep email concise (under 100 words)" not in constraints:
            constraints.append("Keep email concise (under 100 words)")
        if "Only reference verified research facts" not in constraints:
            constraints.append("Only reference verified research facts")

        if is_conditional and unresolved_list:
            for cond in unresolved_list:
                constraints.append(f"Do not assert verified support for '{argument.warrant.capability_name}'; prerequisite '{cond}' remains unverified.")
            constraints.append("Frame outreach as an exploratory inquiry rather than confirmed capability alignment.")

        if argument.rebuttal.missing_backing:
            unbacked_warn = "Do not assert unverified performance metrics or quantitative ROI; backing proof is unavailable without whitelisted proof."
            if unbacked_warn not in constraints:
                constraints.append(unbacked_warn)

        for excl in argument.rebuttal.hard_exclusions:
            excl_rule = f"Strict product boundary: do not claim support for '{excl}'."
            if excl_rule not in constraints:
                constraints.append(excl_rule)

        prov = OutputProvenance(
            origin=ProvenanceOrigin.HEURISTIC_FALLBACK,
            component="strategy_builder",
            rule_id="deterministic_strategy_fallback",
            is_fallback=True,
        )

        return OutreachStrategy(
            primary_signal=primary_signal,
            supporting_signals=supporting_signals,
            alignment=alignment,
            messaging_approach=messaging_approach,
            central_message=argument.claim.text,
            suggested_subject_line=subject,
            cta=cta_text,
            cta_type=cta_type,
            cta_rationale=cta_rationale,
            tone_guidance=f"Professional, direct {outreach.tone} tone",
            constraint_checklist=constraints,
            is_conditional=is_conditional,
            unresolved_conditions=unresolved_list,
            toulmin_argument=argument,
            provenance=prov,
        )
