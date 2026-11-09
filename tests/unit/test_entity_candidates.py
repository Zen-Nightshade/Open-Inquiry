from __future__ import annotations

from datetime import date
import pytest

from src.context.models import ProspectContext
from src.researcher.entity.candidates import (
    EntityCandidateGenerator,
    clean_domain,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.models import EntityType, SourceRecord


def test_name_and_org_normalization():
    # Person name normalization
    assert normalize_person_name("Dr. Jane E. Doe, PhD") == "jane e doe"
    assert normalize_person_name("Prof. John Smith Jr.") == "john smith"
    assert normalize_person_name("Ms. Alice Bob-Jones") == "alice bob-jones"
    assert normalize_person_name("Robert 'Bob' Vance") == "robert bob vance"

    # Organization name normalization
    assert normalize_org_name("Acme Technologies, Inc.") == "acme"
    assert normalize_org_name("Acme Corp.") == "acme"
    assert normalize_org_name("Acme Corporation") == "acme"
    assert normalize_org_name("Widget Labs LLC") == "widget"
    assert normalize_org_name("Global Holdings Ltd.") == "global"

    # Domain normalization
    assert clean_domain("https://www.acme.com/about/") == "acme.com"
    assert clean_domain("http://blog.acme.com:8080/post") == "blog.acme.com"
    assert clean_domain("acme.com") == "acme.com"


def test_extract_person_candidates_from_text_formats():
    generator = EntityCandidateGenerator()
    prospect = ProspectContext(
        full_name="Sarah Connor",
        company_name="Cyberdyne Systems",
        company_domain="cyberdyne.com",
        designation="VP of Engineering",
    )

    source1 = SourceRecord(
        source_id="src_1",
        url="https://tech-news.com/cyberdyne-expansion",
        domain="tech-news.com",
        title="Cyberdyne Systems expands robotics engineering division",
        raw_text=(
            "Cyberdyne Systems announced today that Sarah Connor, VP of Engineering, "
            "will lead the new robotics initiative. Connor joined Cyberdyne from TechCorp."
        ),
        fetch_date=date(2026, 11, 7),
    )

    candidates = generator.generate_candidates(prospect=prospect, sources=[source1])

    # Should discover Sarah Connor and Cyberdyne Systems
    person_candidates = [c for c in candidates if c.entity_type == EntityType.PERSON]
    org_candidates = [c for c in candidates if c.entity_type == EntityType.ORGANIZATION]

    assert len(person_candidates) >= 1
    sarah = next(c for c in person_candidates if c.normalized_name == "sarah connor")
    assert "sarah connor" in sarah.normalized_name.lower()
    assert any("VP of Engineering" in r for r in sarah.roles)
    assert any("cyberdyne" in o.lower() for o in sarah.organizations)
    assert "src_1" in sarah.source_ids

    assert len(org_candidates) >= 1
    cyberdyne = next(c for c in org_candidates if "cyberdyne" in c.normalized_name)
    assert cyberdyne.raw_name == "Cyberdyne Systems"


def test_extract_linkedin_structured_candidate():
    generator = EntityCandidateGenerator()
    prospect = ProspectContext(
        full_name="Marcus Vance",
        company_name="Vance Refrigeration",
        company_domain="vancerefrig.com",
        designation="Chief Executive Officer",
        linkedin_url="https://linkedin.com/in/marcusvance",
    )

    linkedin_source = SourceRecord(
        source_id="src_li",
        url="https://www.linkedin.com/in/marcusvance",
        domain="linkedin.com",
        title="Marcus Vance - Chief Executive Officer - Vance Refrigeration | LinkedIn",
        raw_text="Marcus Vance is the Chief Executive Officer at Vance Refrigeration in Austin, Texas.",
        fetch_date=date(2026, 11, 7),
    )

    candidates = generator.generate_candidates(prospect=prospect, sources=[linkedin_source])

    marcus = next(c for c in candidates if c.entity_type == EntityType.PERSON and c.normalized_name == "marcus vance")
    assert marcus.raw_name == "Marcus Vance"
    assert any("Chief Executive Officer" in r for r in marcus.roles)
    assert any("Vance Refrigeration" in o for o in marcus.organizations)
    assert any("linkedin.com/in/marcusvance" in lu for lu in marcus.linkedin_urls)
    assert "src_li" in marcus.source_ids


def test_provenance_retention_and_observation_consolidation():
    generator = EntityCandidateGenerator()
    prospect = ProspectContext(
        full_name="Elena Rostova",
        company_name="Apex Global",
        company_domain="apexglobal.com",
        designation="CTO",
    )

    source1 = SourceRecord(
        source_id="src_1",
        url="https://apexglobal.com/leadership",
        domain="apexglobal.com",
        title="Leadership Team - Apex Global",
        raw_text="Elena Rostova is Chief Technology Officer at Apex Global.",
        fetch_date=date(2026, 11, 7),
    )

    source2 = SourceRecord(
        source_id="src_2",
        url="https://press.wire.com/apex-funding",
        domain="press.wire.com",
        title="Apex Global Raises $50M Series B",
        raw_text="Apex Global announced its Series B. CTO Elena Rostova noted the team will double in size.",
        fetch_date=date(2026, 11, 7),
    )

    candidates = generator.generate_candidates(prospect=prospect, sources=[source1, source2])

    elena = next(c for c in candidates if c.entity_type == EntityType.PERSON and c.normalized_name == "elena rostova")

    # Observations from BOTH sources must be consolidated into ONE candidate record
    assert "src_1" in elena.source_ids
    assert "src_2" in elena.source_ids
    assert len(elena.observations) >= 2
    obs_source_ids = {obs.source_id for obs in elena.observations}
    assert "src_1" in obs_source_ids
    assert "src_2" in obs_source_ids


def test_keeping_similar_names_with_conflicting_organizations_distinct():
    """Two people with the same name at different companies must remain distinct candidates."""
    generator = EntityCandidateGenerator()
    prospect = ProspectContext(
        full_name="David Miller",
        company_name="Acme Corp",
        company_domain="acme.com",
    )

    source_acme = SourceRecord(
        source_id="src_acme",
        url="https://acme.com/team",
        domain="acme.com",
        title="Acme Team",
        raw_text="David Miller is VP of Sales at Acme Corp.",
        fetch_date=date(2026, 11, 7),
    )

    source_other = SourceRecord(
        source_id="src_other",
        url="https://widgets.org/execs",
        domain="widgets.org",
        title="Widget Execs",
        raw_text="David Miller is Director of Marketing at Widgets International.",
        fetch_date=date(2026, 11, 7),
    )

    candidates = generator.generate_candidates(prospect=prospect, sources=[source_acme, source_other])

    # Find all candidates matching "David Miller"
    david_candidates = [
        c for c in candidates
        if c.entity_type == EntityType.PERSON and c.normalized_name == "david miller"
    ]

    # Crucial requirement: Must NOT consolidate into a single candidate when organizations conflict!
    assert len(david_candidates) == 2
    assert david_candidates[0].candidate_id != david_candidates[1].candidate_id

    # One candidate belongs to Acme Corp, the other to Widgets International
    org_lists = [c.organizations for c in david_candidates]
    assert any(any("Acme" in o for o in orgs) for orgs in org_lists)
    assert any(any("Widgets" in o for o in orgs) for orgs in org_lists)


def test_reject_trivial_and_noise_entities():
    generator = EntityCandidateGenerator()
    prospect = ProspectContext(
        full_name="Jane Doe",
        company_name="Acme",
    )

    noisy_source = SourceRecord(
        source_id="src_noisy",
        url="https://example.com/blog",
        domain="example.com",
        title="Blog - Privacy Policy and Terms of Service",
        raw_text="About Us: Click Sign In to read our Privacy Policy and Terms of Service. By Admin.",
        fetch_date=date(2026, 11, 7),
    )

    candidates = generator.generate_candidates(prospect=prospect, sources=[noisy_source])

    candidate_names = [c.normalized_name for c in candidates]
    # Generic navigation and noise tokens must not be emitted as candidates
    assert "privacy policy" not in candidate_names
    assert "terms of service" not in candidate_names
    assert "sign in" not in candidate_names
    assert "about us" not in candidate_names
    assert "admin" not in candidate_names
