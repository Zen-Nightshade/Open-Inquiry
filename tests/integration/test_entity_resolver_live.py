from __future__ import annotations

import os
import pytest

from src.config.settings import GROQ_SMALL, init_env
from src.context.models import ProspectContext
from src.researcher.entity.candidates import (
    CandidateObservation,
    DiscoveredEntityCandidate,
)
from src.researcher.entity.resolver import IdentityResolver
from src.researcher.models import EntityType, ResolutionStatus


@pytest.mark.integration
def test_identity_resolver_live_groq():
    """Verify live Groq structured identity resolution with real GROQ_SMALL model."""
    init_env()
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured in environment")

    prospect = ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
        linkedin_url="https://linkedin.com/in/erostova",
    )

    candidate = DiscoveredEntityCandidate(
        candidate_id="cand-real-1",
        entity_type=EntityType.PERSON,
        raw_name="Elena Rostova",
        normalized_name="elena rostova",
        domains=["vanguardai.io"],
        roles=["Head of Research"],
        organizations=["Vanguard AI"],
        linkedin_urls=["https://linkedin.com/in/erostova"],
        observations=[
            CandidateObservation(
                source_id="src_1",
                url="https://vanguardai.io/team",
                snippet="Dr. Elena Rostova is Head of Research at Vanguard AI.",
            )
        ],
    )

    resolver = IdentityResolver()
    result = resolver.resolve(prospect=prospect, candidates=[candidate])

    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.prospect_entity.canonical_name is not None
    assert "cand-real-1" in result.matched_candidate_ids
    assert len(result.rationale) > 10
