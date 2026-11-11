from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.provenance import OutputProvenance, ProvenanceOrigin


class ProspectContext(BaseModel):
    """Context identifying the target prospect and their organization."""

    model_config = ConfigDict(frozen=True)

    full_name: str = Field(description="Full name of the target individual")
    company_name: str = Field(description="Name of the company or organization")
    company_domain: Optional[str] = Field(default=None, description="Company web domain, e.g. acme.com")
    designation: Optional[str] = Field(default=None, description="Job title or role within the organization")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    location: Optional[str] = Field(default=None, description="Geographic location (city/state/country)")


# =============================================================================
# M1.1: Structured Product Capability Contract
# =============================================================================

class CompatibilityStatus(str, Enum):
    """Semantic outcome of evaluating product-signal/requirement compatibility."""

    VERIFIED_COMPATIBLE = "verified_compatible"
    CONDITIONAL_ON_UNKNOWN = "conditional_on_unknown"
    INCOMPATIBLE = "incompatible"


class ConstraintType(str, Enum):
    """Classification of product constraints and boundaries."""

    PREREQUISITE = "prerequisite"           # Required condition or dependency for support
    HARD_EXCLUSION = "hard_exclusion"       # Explicitly unsupported technology, platform, or use case
    SCOPE_LIMITATION = "scope_limitation"   # Supported feature with restricted boundary or scale


class CompatibilityResult(BaseModel):
    """Structured result of evaluating product capability compatibility."""

    model_config = ConfigDict(frozen=True)

    status: CompatibilityStatus = Field(description="Deterministic compatibility status")
    capability_name: str = Field(description="Name or identifier of the evaluated capability")
    reason: str = Field(description="Detailed explanation of the compatibility determination")
    matched_capability_id: Optional[str] = Field(default=None, description="Matched CapabilityNode ID if found")
    matched_scope: List[str] = Field(default_factory=list, description="Supported technical scope or environments")
    missing_prerequisites: List[str] = Field(default_factory=list, description="Prerequisites not satisfied or missing")
    unknown_conditions: List[str] = Field(default_factory=list, description="Conditions that must be resolved to confirm compatibility")
    violated_exclusions: List[str] = Field(default_factory=list, description="Hard exclusions or constraints violated")
    provenance: Optional[OutputProvenance] = Field(default=None, description="Provenance of the evaluation determination")


class ProductConstraint(BaseModel):
    """Structured limitation, technical prerequisite, or explicit exclusion."""

    model_config = ConfigDict(frozen=True)

    constraint_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique constraint identifier")
    name: str = Field(..., min_length=1, description="Constraint name or short identifier")
    constraint_type: ConstraintType = Field(description="Classification of constraint")
    description: Optional[str] = Field(default=None, description="Detailed explanation of constraint")
    applies_to_capabilities: List[str] = Field(
        default_factory=list,
        description="Capability IDs or names this constraint applies to. If empty, applies globally.",
    )
    details: List[str] = Field(default_factory=list, description="Specific terms, technical boundaries, or versions")
    provenance: OutputProvenance = Field(
        default_factory=lambda: OutputProvenance(
            origin=ProvenanceOrigin.MANUAL_INPUT,
            component="product_contract",
        ),
        description="Source provenance of the constraint definition",
    )

    @model_validator(mode="after")
    def validate_constraint(self) -> ProductConstraint:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("Constraint name cannot be empty or whitespace.")
        return self


class CapabilityNode(BaseModel):
    """A discrete, structured product capability node."""

    model_config = ConfigDict(frozen=True)

    capability_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique capability identifier")
    name: str = Field(..., min_length=1, description="Canonical name of the capability")
    description: Optional[str] = Field(default=None, description="Detailed description of what the capability does")
    target_operational_problems: List[str] = Field(default_factory=list, description="Specific business or operational problems solved")
    target_buyer_personas: List[str] = Field(default_factory=list, description="Buyer personas or roles relevant to this capability")
    supported_use_cases: List[str] = Field(default_factory=list, description="Specific supported business or technical use cases")
    scope: List[str] = Field(default_factory=list, description="Supported technical scope, platforms, or scales")
    required_prerequisites: List[str] = Field(default_factory=list, description="System or technical prerequisites that must be satisfied")
    hard_exclusions: List[str] = Field(default_factory=list, description="Explicitly unsupported features, stacks, or environments")
    verified_proof_points: List[str] = Field(default_factory=list, description="Whitelisted proof points, case studies, or metrics")
    provenance: OutputProvenance = Field(
        default_factory=lambda: OutputProvenance(
            origin=ProvenanceOrigin.MANUAL_INPUT,
            component="product_contract",
        ),
        description="Source provenance of the capability assertion",
    )

    @model_validator(mode="after")
    def validate_capability_node(self) -> CapabilityNode:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("Capability name cannot be empty or whitespace.")

        # Deterministic contradiction check: prerequisite cannot also be a hard exclusion
        prereqs_norm = {p.strip().lower() for p in self.required_prerequisites if p.strip()}
        excls_norm = {e.strip().lower() for e in self.hard_exclusions if e.strip()}
        contradictions = prereqs_norm.intersection(excls_norm)
        if contradictions:
            item = next(iter(contradictions))
            raise ValueError(f"Contradiction in capability '{self.name}': '{item}' is listed as both a prerequisite and a hard exclusion.")

        return self


class ProductContract(BaseModel):
    """Authoritative structured product capability contract.
    
    Enforces the Closed-World Product Model when is_exhaustive=True, while preventing
    undeclared capabilities from being blindly assumed compatible or incompatible when
    is_exhaustive=False.
    """

    model_config = ConfigDict(frozen=True)

    product_name: str = Field(..., min_length=1, description="Name of the product or offering")
    contract_version: str = Field(default="1.0.0", description="Contract revision or schema version")
    description: str = Field(..., min_length=1, description="High-level description of what the product does")
    value_proposition: str = Field(..., min_length=1, description="Core customer value proposition")
    capabilities: List[CapabilityNode] = Field(default_factory=list, description="Structured capability definitions")
    global_constraints: List[ProductConstraint] = Field(default_factory=list, description="Global constraints and limitations")
    hard_exclusions: List[str] = Field(default_factory=list, description="Global list of explicitly unsupported technologies/use-cases")
    is_exhaustive: bool = Field(
        default=False,
        description="If True, any capability not explicitly declared is strictly INCOMPATIBLE. If False, undeclared capabilities are CONDITIONAL_ON_UNKNOWN.",
    )
    provenance: OutputProvenance = Field(
        default_factory=lambda: OutputProvenance(
            origin=ProvenanceOrigin.MANUAL_INPUT,
            component="product_contract",
        ),
        description="Source provenance of the contract",
    )

    @model_validator(mode="after")
    def validate_contract_integrity(self) -> ProductContract:
        clean_pname = self.product_name.strip()
        clean_desc = self.description.strip()
        clean_vp = self.value_proposition.strip()

        if not clean_pname:
            raise ValueError("Product name cannot be empty or whitespace.")
        if not clean_desc:
            raise ValueError("Product description cannot be empty or whitespace.")
        if not clean_vp:
            raise ValueError("Value proposition cannot be empty or whitespace.")

        # Check unique capability IDs and names
        seen_ids: Set[str] = set()
        seen_names: Set[str] = set()
        for cap in self.capabilities:
            if cap.capability_id in seen_ids:
                raise ValueError(f"Duplicate capability_id '{cap.capability_id}' in contract.")
            seen_ids.add(cap.capability_id)

            cap_norm = cap.name.strip().lower()
            if cap_norm in seen_names:
                raise ValueError(f"Duplicate capability name '{cap.name}' declared in contract.")
            seen_names.add(cap_norm)

        # Collect global hard exclusions
        global_excls_norm = {e.strip().lower() for e in self.hard_exclusions if e.strip()}
        for gc in self.global_constraints:
            if gc.constraint_type == ConstraintType.HARD_EXCLUSION:
                global_excls_norm.add(gc.name.strip().lower())
                for d in gc.details:
                    if d.strip():
                        global_excls_norm.add(d.strip().lower())

        # Contradiction check: No capability name, supported use case, or prerequisite may conflict with global exclusions
        for cap in self.capabilities:
            cap_name_norm = cap.name.strip().lower()
            if cap_name_norm in global_excls_norm:
                raise ValueError(
                    f"Contradiction: Capability '{cap.name}' directly conflicts with declared hard exclusion '{cap_name_norm}'."
                )

            for uc in cap.supported_use_cases:
                uc_norm = uc.strip().lower()
                if uc_norm in global_excls_norm:
                    raise ValueError(
                        f"Contradiction: Capability '{cap.name}' use case '{uc}' conflicts with hard exclusion '{uc_norm}'."
                    )

            for pr in cap.required_prerequisites:
                pr_norm = pr.strip().lower()
                if pr_norm in global_excls_norm:
                    raise ValueError(
                        f"Contradiction: Capability '{cap.name}' prerequisite '{pr}' conflicts with hard exclusion '{pr_norm}'."
                    )

        return self

    def evaluate_compatibility(
        self,
        required_capability: str,
        known_conditions: Optional[Dict[str, bool]] = None,
        context_attributes: Optional[List[str]] = None,
    ) -> CompatibilityResult:
        """Evaluate compatibility of a required capability against the contract.
        
        Outcomes:
          - VERIFIED_COMPATIBLE: Explicitly supported within stated scope, all prerequisites verified.
          - CONDITIONAL_ON_UNKNOWN: Support depends on unestablished conditions (e.g. unknown prerequisites,
            or undeclared capability in a non-exhaustive contract).
          - INCOMPATIBLE: Explicit hard exclusion, confirmed limitation, unsatisfied prerequisite,
            or undeclared capability in an exhaustive contract.
        """
        req_clean = required_capability.strip()
        if not req_clean:
            raise ValueError("required_capability cannot be empty or whitespace.")

        req_norm = req_clean.lower()
        known = {k.strip().lower(): v for k, v in (known_conditions or {}).items()}
        ctx = [c.strip().lower() for c in (context_attributes or []) if c.strip()]

        # 1. Check Global Hard Exclusions
        all_global_exclusions: Set[str] = {e.strip().lower() for e in self.hard_exclusions if e.strip()}
        for gc in self.global_constraints:
            if gc.constraint_type == ConstraintType.HARD_EXCLUSION:
                all_global_exclusions.add(gc.name.strip().lower())
                all_global_exclusions.update(d.strip().lower() for d in gc.details if d.strip())

        for excl in all_global_exclusions:
            if excl == req_norm or excl in req_norm or req_norm in excl:
                return CompatibilityResult(
                    status=CompatibilityStatus.INCOMPATIBLE,
                    capability_name=req_clean,
                    reason=f"Violates product hard exclusion '{excl}'.",
                    violated_exclusions=[excl],
                    provenance=self.provenance,
                )
            # Check context attributes against global exclusions
            for attr in ctx:
                if excl == attr or excl in attr or attr in excl:
                    return CompatibilityResult(
                        status=CompatibilityStatus.INCOMPATIBLE,
                        capability_name=req_clean,
                        reason=f"Prospect context attribute '{attr}' violates hard exclusion '{excl}'.",
                        violated_exclusions=[excl],
                        provenance=self.provenance,
                    )

        # 2. Match against declared CapabilityNodes
        matched_cap: Optional[CapabilityNode] = None
        for cap in self.capabilities:
            cap_name_norm = cap.name.strip().lower()
            if cap_name_norm == req_norm or req_norm in cap_name_norm or cap_name_norm in req_norm:
                matched_cap = cap
                break
            for uc in cap.supported_use_cases:
                uc_norm = uc.strip().lower()
                if uc_norm == req_norm or req_norm in uc_norm or uc_norm in req_norm:
                    matched_cap = cap
                    break

        if matched_cap:
            # Check capability-specific hard exclusions
            for c_excl in matched_cap.hard_exclusions:
                c_excl_norm = c_excl.strip().lower()
                if c_excl_norm == req_norm or c_excl_norm in req_norm or req_norm in c_excl_norm:
                    return CompatibilityResult(
                        status=CompatibilityStatus.INCOMPATIBLE,
                        capability_name=req_clean,
                        matched_capability_id=matched_cap.capability_id,
                        reason=f"Capability '{matched_cap.name}' explicitly excludes '{c_excl}'.",
                        violated_exclusions=[c_excl],
                        provenance=matched_cap.provenance,
                    )
                for attr in ctx:
                    if c_excl_norm == attr or c_excl_norm in attr or attr in c_excl_norm:
                        return CompatibilityResult(
                            status=CompatibilityStatus.INCOMPATIBLE,
                            capability_name=req_clean,
                            matched_capability_id=matched_cap.capability_id,
                            reason=f"Prospect context attribute '{attr}' violates capability hard exclusion '{c_excl}'.",
                            violated_exclusions=[c_excl],
                            provenance=matched_cap.provenance,
                        )

            # Check prerequisites
            missing_prereqs: List[str] = []
            unknown_prereqs: List[str] = []

            for prereq in matched_cap.required_prerequisites:
                prereq_norm = prereq.strip().lower()
                # Check if known explicitly
                if prereq_norm in known:
                    if not known[prereq_norm]:
                        # Confirmed unsatisfied
                        missing_prereqs.append(prereq)
                elif any(prereq_norm in attr or attr in prereq_norm for attr in ctx):
                    # Satisfied by context attribute
                    pass
                else:
                    # Unknown status
                    unknown_prereqs.append(prereq)

            if missing_prereqs:
                return CompatibilityResult(
                    status=CompatibilityStatus.INCOMPATIBLE,
                    capability_name=req_clean,
                    matched_capability_id=matched_cap.capability_id,
                    reason=f"Capability '{matched_cap.name}' has unsatisfied prerequisites: {', '.join(missing_prereqs)}.",
                    missing_prerequisites=missing_prereqs,
                    provenance=matched_cap.provenance,
                )

            if unknown_prereqs:
                return CompatibilityResult(
                    status=CompatibilityStatus.CONDITIONAL_ON_UNKNOWN,
                    capability_name=req_clean,
                    matched_capability_id=matched_cap.capability_id,
                    reason=f"Capability '{matched_cap.name}' is conditional on unknown prerequisites: {', '.join(unknown_prereqs)}.",
                    missing_prerequisites=unknown_prereqs,
                    unknown_conditions=unknown_prereqs,
                    provenance=matched_cap.provenance,
                )

            return CompatibilityResult(
                status=CompatibilityStatus.VERIFIED_COMPATIBLE,
                capability_name=req_clean,
                matched_capability_id=matched_cap.capability_id,
                matched_scope=matched_cap.scope,
                reason=f"Capability '{matched_cap.name}' is verified compatible with all conditions satisfied.",
                provenance=matched_cap.provenance,
            )

        # 3. Undeclared Capability Handling
        if self.is_exhaustive:
            return CompatibilityResult(
                status=CompatibilityStatus.INCOMPATIBLE,
                capability_name=req_clean,
                reason=f"Capability '{req_clean}' is not supported under exhaustive product contract for '{self.product_name}'.",
                provenance=self.provenance,
            )

        # In non-exhaustive contracts, undeclared capability is strictly CONDITIONAL_ON_UNKNOWN
        return CompatibilityResult(
            status=CompatibilityStatus.CONDITIONAL_ON_UNKNOWN,
            capability_name=req_clean,
            reason=f"Capability '{req_clean}' is not declared in product contract; support is unverified under non-exhaustive contract.",
            unknown_conditions=[f"product_support_for_{req_clean}"],
            provenance=self.provenance,
        )

    def to_context(self) -> ProductContext:
        """Create a backwards-compatible ProductContext from this contract."""
        cap_strings = [c.name for c in self.capabilities]
        lim_strings = list(self.hard_exclusions)
        for gc in self.global_constraints:
            if gc.name not in lim_strings:
                lim_strings.append(gc.name)

        return ProductContext(
            product_name=self.product_name,
            description=self.description,
            capabilities=cap_strings,
            value_proposition=self.value_proposition,
            limitations=lim_strings,
            contract=self,
        )


class ProductContext(BaseModel):
    """Context defining the product offering, capabilities, and target profile.
    
    Provides backwards compatibility for legacy string-based callers while supporting
    the authoritative structured ProductContract.
    """

    model_config = ConfigDict(frozen=True)

    product_name: str = Field(description="Name of the product or offering")
    description: str = Field(description="High-level description of what the product does")
    capabilities: List[str] = Field(default_factory=list, description="List of specific problems the product solves")
    value_proposition: str = Field(description="Core customer value proposition")
    icp: Optional[str] = Field(default=None, description="Ideal Customer Profile summary")
    differentiators: List[str] = Field(default_factory=list, description="Unique differentiators vs alternatives")
    limitations: List[str] = Field(default_factory=list, description="Explicit boundaries and product limitations")
    contract: Optional[ProductContract] = Field(default=None, description="Structured product capability contract")

    @model_validator(mode="after")
    def reconcile_capabilities_and_contract(self) -> ProductContext:
        # If contract provided and capabilities empty, populate capabilities from contract
        if self.contract is not None and not self.capabilities:
            object.__setattr__(self, "capabilities", [c.name for c in self.contract.capabilities])
        return self

    def get_contract(self) -> ProductContract:
        """Return the structured ProductContract, auto-wrapping legacy fields if necessary."""
        if self.contract is not None:
            return self.contract

        # Wrap legacy plain-text fields into a default non-exhaustive ProductContract
        cap_nodes = [
            CapabilityNode(
                name=c.strip(),
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.MANUAL_INPUT,
                    component="product_context_adapter",
                ),
            )
            for c in self.capabilities
            if c.strip()
        ]
        constraints = [
            ProductConstraint(
                name=lim.strip(),
                constraint_type=ConstraintType.SCOPE_LIMITATION,
                description=lim.strip(),
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.MANUAL_INPUT,
                    component="product_context_adapter",
                ),
            )
            for lim in self.limitations
            if lim.strip()
        ]
        return ProductContract(
            product_name=self.product_name,
            contract_version="legacy-adapter-1.0",
            description=self.description,
            value_proposition=self.value_proposition,
            capabilities=cap_nodes,
            global_constraints=constraints,
            hard_exclusions=[lim.strip() for lim in self.limitations if lim.strip()],
            is_exhaustive=False,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.MANUAL_INPUT,
                component="product_context_adapter",
            ),
        )


class OutreachContext(BaseModel):
    """Context governing outreach objectives, sender voice, and message constraints."""

    model_config = ConfigDict(frozen=True)

    objective: str = Field(description="Primary commercial or relational goal of the outreach")
    sender_identity: str = Field(description="Sender identification (name, role, organization)")
    sender_name: Optional[str] = Field(default=None, description="Sender given name")
    sender_company: Optional[str] = Field(default=None, description="Sender company name")
    desired_cta: Optional[str] = Field(default=None, description="Requested call-to-action")
    tone: str = Field(default="professional", description="Desired communication tone")
    relationship: Optional[str] = Field(default="cold", description="Relationship stage, e.g. cold, prior contact")
    constraints: List[str] = Field(default_factory=list, description="Additional constraints, e.g. word limit, banned terms")
