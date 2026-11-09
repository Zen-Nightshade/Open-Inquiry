from __future__ import annotations

import sqlite3
import tempfile
import pytest

from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.unit_of_work import UnitOfWork


@pytest.fixture
def uow():
    with tempfile.NamedTemporaryFile() as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()

        with UnitOfWork(db_path=tf.name) as u:
            yield u


def test_outreach_repository_crud(uow: UnitOfWork):
    # 1. Create prospect
    p = uow.outreach.create_prospect(
        prospect_id="p-1",
        full_name="Alex Mercer",
        company_name="Apex Corp",
        company_domain="apex.com",
        designation="VP Engineering",
        linkedin_url="https://linkedin.com/in/alexmercer",
        location="San Francisco, CA",
    )
    assert p["prospect_id"] == "p-1"
    assert p["full_name"] == "Alex Mercer"

    # Get prospect
    fetched = uow.outreach.get_prospect("p-1")
    assert fetched is not None
    assert fetched["company_domain"] == "apex.com"

    # Soft delete prospect
    assert uow.outreach.soft_delete_prospect("p-1") is True
    active_prospects = uow.outreach.list_prospects(include_deleted=False)
    assert len(active_prospects) == 0
    all_prospects = uow.outreach.list_prospects(include_deleted=True)
    assert len(all_prospects) == 1

    # 2. Outreach context & draft
    v = uow.products.create_product_version(
        version_id="v-1",
        product_name="Zamp Engine",
        snapshot_json='{"name": "Zamp"}',
    )
    ctx = uow.outreach.create_outreach_context(
        context_id="ctx-1",
        prospect_id="p-1",
        version_id="v-1",
        context_json='{"objective": "demo"}',
    )
    assert ctx["context_id"] == "ctx-1"

    draft = uow.outreach.create_outreach_draft(
        draft_id="d-1",
        context_id="ctx-1",
        research_session_id="s-dummy",
        subject="Quick question",
        body="Hi Alex...",
        word_count=50,
        product_snapshot_json='{"name": "Zamp"}',
        claim_verifications='[]',
        quality_evaluation='{"score": 5}',
    )
    assert draft["human_review_status"] == "pending_review"

    # Update review
    updated = uow.outreach.update_outreach_draft_review(
        draft_id="d-1",
        human_review_status="approved",
        reviewer_notes="Looks great!",
    )
    assert updated is True
    d_after = uow.outreach.get_outreach_draft("d-1")
    assert d_after["human_review_status"] == "approved"
    assert d_after["reviewer_notes"] == "Looks great!"

    # Record review event
    event = uow.outreach.record_human_review_event(
        event_id="e-1",
        draft_id="d-1",
        decision="approved",
        reviewer_id="rev-user",
        notes="Verified against prospect LinkedIn",
    )
    assert event["event_id"] == "e-1"
    events = uow.outreach.get_human_review_events("d-1")
    assert len(events) == 1


def test_product_repository_crud(uow: UnitOfWork):
    pv1 = uow.products.create_product_version(
        version_id="pv-1",
        product_name="Product X",
        snapshot_json='{"v": 1}',
        created_at="2026-01-01T00:00:00Z",
    )
    pv2 = uow.products.create_product_version(
        version_id="pv-2",
        product_name="Product X",
        snapshot_json='{"v": 2}',
        created_at="2026-01-02T00:00:00Z",
    )
    latest = uow.products.get_latest_version_for_product("Product X")
    assert latest is not None
    assert latest["version_id"] == "pv-2"

    all_versions = uow.products.list_product_versions("Product X")
    assert len(all_versions) == 2


def test_research_repository_crud(uow: UnitOfWork):
    p = uow.outreach.create_prospect("p-2", "Jane Doe", "Acme")
    v = uow.products.create_product_version("v-2", "Tool", "{}")
    c = uow.outreach.create_outreach_context("ctx-2", "p-2", "v-2", "{}")

    # Session
    sess = uow.research.create_session(
        session_id="sess-1",
        prospect_id="p-2",
        context_id="ctx-2",
    )
    assert sess["session_id"] == "sess-1"

    # Objective
    obj = uow.research.create_objective(
        objective_id="obj-1",
        session_id="sess-1",
        type="neutral_discovery",
        target_entity="company",
        description="Find company news",
        priority=1,
    )
    assert obj["objective_id"] == "obj-1"

    # Query
    q = uow.research.create_query(
        query_id="q-1",
        session_id="sess-1",
        query_text="Acme recent product announcements",
        objective_id="obj-1",
    )
    assert q["query_id"] == "q-1"

    # Search Execution
    exec_row = uow.research.record_search_execution(
        execution_id="exec-1",
        query_id="q-1",
        provider="duckduckgo",
        status="success",
        duration_ms=250.0,
        idempotency_key="idemp-1",
    )
    assert exec_row["execution_id"] == "exec-1"

    # Duplicate idempotency key must be retrievable
    found_exec = uow.research.get_search_execution_by_idempotency_key("idemp-1")
    assert found_exec is not None
    assert found_exec["execution_id"] == "exec-1"

    # Update session status
    updated = uow.research.update_session(
        session_id="sess-1",
        status="completed",
        is_sufficient=True,
        is_complete=True,
    )
    assert updated is True
    sess_after = uow.research.get_session("sess-1")
    assert sess_after["status"] == "completed"
    assert sess_after["is_sufficient"] == 1


def test_source_repository_crud(uow: UnitOfWork):
    # Upsert source
    s = uow.sources.upsert_source(
        source_id="src-1",
        canonical_url="https://techcrunch.com/2026/01/acme-series-b",
        domain="techcrunch.com",
        source_type="news",
        credibility_score=0.9,
    )
    assert s["source_id"] == "src-1"

    # Upserting same canonical_url updates credibility
    s2 = uow.sources.upsert_source(
        source_id="src-ignored",
        canonical_url="https://techcrunch.com/2026/01/acme-series-b",
        domain="techcrunch.com",
        source_type="news",
        credibility_score=0.95,
    )
    assert s2["credibility_score"] == 0.95
    assert s2["source_id"] == "src-1"  # Preserves canonical primary key

    # Snapshot
    snap = uow.sources.create_snapshot(
        snapshot_id="snap-1",
        source_id="src-1",
        fetch_status="success",
        http_status_code=200,
        raw_text="Acme announced Series B funding of $30M.",
    )
    assert snap["snapshot_id"] == "snap-1"

    # Evidence passage
    ev = uow.sources.create_evidence_passage(
        evidence_id="ev-1",
        snapshot_id="snap-1",
        source_id="src-1",
        verbatim_excerpt="Acme announced Series B funding of $30M.",
        excerpt_validated=1,
        confidence="high",
        credibility_score=0.95,
    )
    assert ev["evidence_id"] == "ev-1"
    assert ev["excerpt_validated"] == 1

    # Syndication origin group
    grp = uow.sources.create_origin_group(
        group_id="grp-1",
        canonical_source_id="src-1",
        description="Wire press release",
    )
    assert grp["group_id"] == "grp-1"

    # Add second syndicated source
    uow.sources.upsert_source(
        source_id="src-2",
        canonical_url="https://finance.yahoo.com/news/acme-series-b",
        domain="finance.yahoo.com",
    )
    added = uow.sources.add_origin_member("grp-1", "src-2")
    assert added is True
    members = uow.sources.get_origin_members("grp-1")
    assert len(members) == 1
    assert members[0]["source_id"] == "src-2"


def test_entity_repository_crud(uow: UnitOfWork):
    # Entity
    ent = uow.entities.create_entity(
        entity_id="ent-1",
        entity_type="person",
        raw_name="Dr. Robert Smith",
        canonical_name="Robert Smith",
        resolution_status="resolved",
        confidence="high",
    )
    assert ent["canonical_name"] == "Robert Smith"

    # Identifier
    uow.entities.add_identifier(
        identifier_id="id-1",
        entity_id="ent-1",
        id_type="linkedin_url",
        id_value="https://linkedin.com/in/rsmith",
    )
    found = uow.entities.find_entity_by_identifier("linkedin_url", "https://linkedin.com/in/rsmith")
    assert found is not None
    assert found["entity_id"] == "ent-1"

    # Affiliation (temporal start/end date, never single current_company)
    uow.entities.create_entity("ent-org-1", "organization", "Tech Inc")
    aff = uow.entities.add_affiliation(
        affiliation_id="aff-1",
        entity_id="ent-1",
        organization_id="ent-org-1",
        role="Chief Architect",
        start_date="2022-01-01",
        end_date=None,
    )
    assert aff["role"] == "Chief Architect"
    assert aff["end_date"] is None

    # Relationship
    uow.entities.create_entity("ent-org-2", "organization", "Parent Corp")
    rel = uow.entities.add_relationship(
        relationship_id="rel-1",
        from_entity_id="ent-org-2",
        to_entity_id="ent-org-1",
        relationship_type="parent_of",
    )
    assert rel["relationship_type"] == "parent_of"


def test_assertion_repository_crud(uow: UnitOfWork):
    # Set up session & entity & source & evidence
    uow.outreach.create_prospect("p-3", "Bob", "Corp")
    uow.research.create_session("sess-3", "p-3", "ctx-3")
    uow.entities.create_entity("ent-3", "organization", "Corp")
    uow.sources.upsert_source("src-3", "https://example.com/3", "example.com")
    uow.sources.create_snapshot("snap-3", "src-3", "success")
    uow.sources.create_evidence_passage("ev-3", "snap-3", "src-3", "Corp raised $5M.")

    # Canonical Fact
    fact = uow.assertions.create_canonical_fact(
        fact_id="fact-1",
        session_id="sess-3",
        claim_text="Corp raised $5M in Seed funding.",
        fact_category="funding",
        composite_confidence="high",
    )
    assert fact["fact_id"] == "fact-1"

    # Many-to-many entity link
    uow.assertions.link_fact_entity("fact-1", "ent-3")
    linked_entities = uow.assertions.get_entities_for_fact("fact-1")
    assert len(linked_entities) == 1
    assert linked_entities[0]["entity_id"] == "ent-3"

    # Many-to-many evidence link
    uow.assertions.link_fact_evidence("fact-1", "ev-3", support_type="supporting")
    linked_evidence = uow.assertions.get_evidence_for_fact("fact-1")
    assert len(linked_evidence) == 1
    assert linked_evidence[0]["evidence_id"] == "ev-3"

    # Conflict and resolution decision
    conf = uow.assertions.create_conflict(
        conflict_id="conf-1",
        session_id="sess-3",
        description="Source A says $5M, Source B says $10M",
    )
    assert conf["status"] == "open"
    uow.assertions.link_conflict_fact("conf-1", "fact-1")

    decision = uow.assertions.record_resolution_decision(
        decision_id="dec-1",
        conflict_id="conf-1",
        rationale="SEC filing confirmed $5M.",
        decided_by="human",
    )
    assert decision["decided_by"] == "human"
    conf_after = uow.assertions.get_conflict("conf-1")
    assert conf_after["status"] == "resolved"


def test_job_repository_crud(uow: UnitOfWork):
    job = uow.jobs.create_job(
        job_id="job-1",
        job_type="research_session",
        idempotency_key="job-key-1",
    )
    assert job["status"] == "pending"

    # Attempt
    att = uow.jobs.record_attempt(
        attempt_id="att-1",
        job_id="job-1",
        attempt_num=1,
        started_at="2026-01-01T00:00:00Z",
    )
    assert att["attempt_num"] == 1

    # Model execution metric
    model_exec = uow.jobs.record_model_execution(
        execution_id="mexec-1",
        job_id="job-1",
        provider="groq",
        model_name="openai/gpt-oss-120b",
        task_type="query_generation",
        prompt_hash="hash-1234",
        input_tokens=150,
        output_tokens=60,
        duration_ms=320.0,
    )
    assert model_exec["input_tokens"] == 150

    # Workflow event
    wf = uow.jobs.record_workflow_event(
        event_id="wf-1",
        event_type="iteration_started",
        job_id="job-1",
        payload='{"iteration": 1}',
    )
    assert wf["event_type"] == "iteration_started"


def test_foreign_key_enforcement(uow: UnitOfWork):
    """Verify that foreign key constraints raise sqlite3.IntegrityError."""
    # Attempt to insert a research session with a non-existent prospect_id
    with pytest.raises(sqlite3.IntegrityError):
        uow.research.create_session(
            session_id="invalid-sess",
            prospect_id="non-existent-prospect",
            context_id="ctx",
        )

    # Attempt to insert an evidence passage with non-existent source
    with pytest.raises(sqlite3.IntegrityError):
        uow.sources.create_evidence_passage(
            evidence_id="invalid-ev",
            snapshot_id="non-existent-snapshot",
            source_id="non-existent-source",
            verbatim_excerpt="quote",
        )
