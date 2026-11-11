from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CapabilityNode as DomainCapabilityNode,
    ConstraintType as DomainConstraintType,
    OutreachContext as DomainOutreachContext,
    ProductConstraint as DomainProductConstraint,
    ProductContract as DomainProductContract,
    ProductContext as DomainProductContext,
    ProspectContext as DomainProspectContext,
)


class ProspectInput(BaseModel):
    full_name: str = Field(..., min_length=1, description="Full name of target prospect")
    company_name: str = Field(..., min_length=1, description="Company or organization name")
    company_domain: Optional[str] = Field(None, description="Company web domain, e.g. 'stripe.com'")
    designation: Optional[str] = Field(None, description="Current role or job title")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    location: Optional[str] = Field(None, description="Geographic location")

    def to_domain(self) -> DomainProspectContext:
        return DomainProspectContext(
            full_name=self.full_name,
            company_name=self.company_name,
            company_domain=self.company_domain,
            designation=self.designation,
            linkedin_url=self.linkedin_url,
            location=self.location,
        )


class CapabilityInput(BaseModel):
    name: str = Field(..., min_length=1, description="Capability name")
    description: Optional[str] = Field(None, description="Detailed capability description")
    target_operational_problems: List[str] = Field(default_factory=list, description="Target operational problems")
    target_buyer_personas: List[str] = Field(default_factory=list, description="Target buyer personas")
    supported_use_cases: List[str] = Field(default_factory=list, description="Supported use cases")
    scope: List[str] = Field(default_factory=list, description="Supported scope or platforms")
    required_prerequisites: List[str] = Field(default_factory=list, description="Required prerequisites")
    hard_exclusions: List[str] = Field(default_factory=list, description="Explicit hard exclusions")
    verified_proof_points: List[str] = Field(default_factory=list, description="Proof points or metrics")

    def to_domain(self) -> DomainCapabilityNode:
        return DomainCapabilityNode(
            name=self.name,
            description=self.description,
            target_operational_problems=self.target_operational_problems,
            target_buyer_personas=self.target_buyer_personas,
            supported_use_cases=self.supported_use_cases,
            scope=self.scope,
            required_prerequisites=self.required_prerequisites,
            hard_exclusions=self.hard_exclusions,
            verified_proof_points=self.verified_proof_points,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.MANUAL_INPUT,
                component="api_input",
            ),
        )


class ConstraintInput(BaseModel):
    name: str = Field(..., min_length=1, description="Constraint name or label")
    constraint_type: str = Field(default="scope_limitation", description="prerequisite, hard_exclusion, or scope_limitation")
    description: Optional[str] = Field(None, description="Detailed explanation")
    applies_to_capabilities: List[str] = Field(default_factory=list, description="Specific capability names")
    details: List[str] = Field(default_factory=list, description="Specific boundary terms")

    def to_domain(self) -> DomainProductConstraint:
        try:
            c_type = DomainConstraintType(self.constraint_type.lower())
        except ValueError:
            c_type = DomainConstraintType.SCOPE_LIMITATION

        return DomainProductConstraint(
            name=self.name,
            constraint_type=c_type,
            description=self.description,
            applies_to_capabilities=self.applies_to_capabilities,
            details=self.details,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.MANUAL_INPUT,
                component="api_input",
            ),
        )


class ProductContractInput(BaseModel):
    product_name: Optional[str] = Field(None, description="Product name (defaults to outer product_name if omitted)")
    contract_version: str = Field(default="1.0.0", description="Contract revision or schema version")
    description: Optional[str] = Field(None, description="Description (defaults to outer description if omitted)")
    value_proposition: Optional[str] = Field(None, description="Value proposition (defaults to outer if omitted)")
    capabilities: List[CapabilityInput] = Field(default_factory=list, description="Structured capabilities")
    global_constraints: List[ConstraintInput] = Field(default_factory=list, description="Global constraints")
    hard_exclusions: List[str] = Field(default_factory=list, description="Global hard exclusions")
    is_exhaustive: bool = Field(default=False, description="Whether declarations are exhaustive")

    def to_domain(self, fallback_name: str, fallback_desc: str, fallback_vp: str) -> DomainProductContract:
        return DomainProductContract(
            product_name=self.product_name or fallback_name,
            contract_version=self.contract_version,
            description=self.description or fallback_desc,
            value_proposition=self.value_proposition or fallback_vp,
            capabilities=[c.to_domain() for c in self.capabilities],
            global_constraints=[gc.to_domain() for gc in self.global_constraints],
            hard_exclusions=self.hard_exclusions,
            is_exhaustive=self.is_exhaustive,
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.MANUAL_INPUT,
                component="api_input",
            ),
        )


class ProductInput(BaseModel):
    product_name: str = Field(..., min_length=1, description="Product or offering name")
    description: str = Field(..., min_length=1, description="Product description")
    capabilities: List[str] = Field(default_factory=list, description="List of product capabilities/features")
    value_proposition: str = Field(..., min_length=1, description="Core customer value proposition")
    icp: Optional[str] = Field(None, description="Ideal customer profile summary")
    differentiators: List[str] = Field(default_factory=list, description="Unique differentiators")
    limitations: List[str] = Field(default_factory=list, description="Explicit boundaries and product limitations")
    contract: Optional[ProductContractInput] = Field(None, description="Optional structured product contract")
    structured_capabilities: List[CapabilityInput] = Field(default_factory=list, description="Optional structured capabilities")

    def to_domain(self) -> DomainProductContext:
        domain_contract: Optional[DomainProductContract] = None

        if self.contract:
            domain_contract = self.contract.to_domain(
                fallback_name=self.product_name,
                fallback_desc=self.description,
                fallback_vp=self.value_proposition,
            )
        elif self.structured_capabilities:
            cap_nodes = [sc.to_domain() for sc in self.structured_capabilities]
            constraints = [
                DomainProductConstraint(
                    name=lim.strip(),
                    constraint_type=DomainConstraintType.SCOPE_LIMITATION,
                    description=lim.strip(),
                    provenance=OutputProvenance(
                        origin=ProvenanceOrigin.MANUAL_INPUT,
                        component="api_input",
                    ),
                )
                for lim in self.limitations
                if lim.strip()
            ]
            domain_contract = DomainProductContract(
                product_name=self.product_name,
                contract_version="1.0.0",
                description=self.description,
                value_proposition=self.value_proposition,
                capabilities=cap_nodes,
                global_constraints=constraints,
                hard_exclusions=[lim.strip() for lim in self.limitations if lim.strip()],
                is_exhaustive=False,
                provenance=OutputProvenance(
                    origin=ProvenanceOrigin.MANUAL_INPUT,
                    component="api_input",
                ),
            )

        caps = list(self.capabilities)
        if domain_contract and not caps:
            caps = [c.name for c in domain_contract.capabilities]

        return DomainProductContext(
            product_name=self.product_name,
            description=self.description,
            capabilities=caps,
            value_proposition=self.value_proposition,
            icp=self.icp,
            differentiators=self.differentiators,
            limitations=self.limitations,
            contract=domain_contract,
        )


class OutreachInput(BaseModel):
    objective: str = Field(..., min_length=1, description="Outreach objective")
    sender_identity: str = Field(..., min_length=1, description="Sender name, role, organization")
    sender_name: Optional[str] = Field(None, description="Sender given name")
    sender_company: Optional[str] = Field(None, description="Sender company name")
    desired_cta: Optional[str] = Field(None, description="Call to action")
    tone: str = Field("professional", description="Desired tone")
    relationship: Optional[str] = Field("cold", description="Relationship stage")
    constraints: List[str] = Field(default_factory=list, description="Outreach constraints")

    def to_domain(self) -> DomainOutreachContext:
        return DomainOutreachContext(
            objective=self.objective,
            sender_identity=self.sender_identity,
            sender_name=self.sender_name,
            sender_company=self.sender_company,
            desired_cta=self.desired_cta,
            tone=self.tone,
            relationship=self.relationship,
            constraints=self.constraints,
        )
