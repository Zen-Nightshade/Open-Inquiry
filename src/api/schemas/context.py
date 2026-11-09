from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from src.context.models import (
    OutreachContext as DomainOutreachContext,
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


class ProductInput(BaseModel):
    product_name: str = Field(..., min_length=1, description="Product or offering name")
    description: str = Field(..., min_length=1, description="Product description")
    capabilities: List[str] = Field(default_factory=list, description="List of product capabilities/features")
    value_proposition: str = Field(..., min_length=1, description="Core customer value proposition")
    icp: Optional[str] = Field(None, description="Ideal customer profile summary")
    differentiators: List[str] = Field(default_factory=list, description="Unique differentiators")
    limitations: List[str] = Field(default_factory=list, description="Explicit boundaries and product limitations")

    def to_domain(self) -> DomainProductContext:
        return DomainProductContext(
            product_name=self.product_name,
            description=self.description,
            capabilities=self.capabilities,
            value_proposition=self.value_proposition,
            icp=self.icp,
            differentiators=self.differentiators,
            limitations=self.limitations,
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
