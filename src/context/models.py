from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ProspectContext(BaseModel):
    """Context identifying the target prospect and their organization."""

    model_config = ConfigDict(frozen=True)

    full_name: str = Field(description="Full name of the target individual")
    company_name: str = Field(description="Name of the company or organization")
    company_domain: Optional[str] = Field(default=None, description="Company web domain, e.g. acme.com")
    designation: Optional[str] = Field(default=None, description="Job title or role within the organization")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    location: Optional[str] = Field(default=None, description="Geographic location (city/state/country)")


class ProductContext(BaseModel):
    """Context defining the product offering, capabilities, and target profile."""

    model_config = ConfigDict(frozen=True)

    product_name: str = Field(description="Name of the product or offering")
    description: str = Field(description="High-level description of what the product does")
    capabilities: List[str] = Field(description="List of specific problems the product solves")
    value_proposition: str = Field(description="Core customer value proposition")
    icp: Optional[str] = Field(default=None, description="Ideal Customer Profile summary")
    differentiators: List[str] = Field(default_factory=list, description="Unique differentiators vs alternatives")
    limitations: List[str] = Field(default_factory=list, description="Explicit boundaries and product limitations")


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
