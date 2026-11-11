-- Open Inquiry Single Authoritative Database Schema
-- Defines all 36 core domain tables in topological order respecting referential integrity.
-- PRAGMA foreign_keys = ON must be active when executing or using this database.

-- ============================================================================
-- 1. Operations & Job Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    job_type        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    idempotency_key TEXT UNIQUE,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    error_message   TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key);

CREATE TABLE IF NOT EXISTS job_attempts (
    attempt_id   TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    attempt_num  INTEGER NOT NULL,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    outcome      TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_attempts_job ON job_attempts(job_id);

CREATE TABLE IF NOT EXISTS model_executions (
    execution_id  TEXT PRIMARY KEY,
    run_id        TEXT,
    job_id        TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    stage_name    TEXT,
    task_type     TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    duration_ms   REAL,
    status        TEXT NOT NULL DEFAULT 'completed',
    error_type    TEXT,
    error_message TEXT,
    attempt_num   INTEGER NOT NULL DEFAULT 1,
    executed_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_executions_job ON model_executions(job_id);
CREATE INDEX IF NOT EXISTS idx_model_executions_run ON model_executions(run_id);

-- ============================================================================
-- 2. Prospects, Product Contexts & Outreach
-- ============================================================================

CREATE TABLE IF NOT EXISTS prospects (
    prospect_id    TEXT PRIMARY KEY,
    full_name      TEXT NOT NULL,
    company_name   TEXT NOT NULL,
    company_domain TEXT,
    designation    TEXT,
    linkedin_url   TEXT,
    location       TEXT,
    created_at     TEXT NOT NULL,
    deleted_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_prospects_company_name ON prospects(company_name);

CREATE TABLE IF NOT EXISTS product_versions (
    version_id    TEXT PRIMARY KEY,
    product_name  TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_product_versions_name ON product_versions(product_name);

CREATE TABLE IF NOT EXISTS outreach_contexts (
    context_id   TEXT PRIMARY KEY,
    prospect_id  TEXT NOT NULL REFERENCES prospects(prospect_id) ON DELETE RESTRICT,
    version_id   TEXT NOT NULL REFERENCES product_versions(version_id) ON DELETE RESTRICT,
    context_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outreach_contexts_prospect ON outreach_contexts(prospect_id);

CREATE TABLE IF NOT EXISTS outreach_drafts (
    draft_id              TEXT PRIMARY KEY,
    context_id            TEXT NOT NULL REFERENCES outreach_contexts(context_id) ON DELETE RESTRICT,
    research_session_id   TEXT NOT NULL,
    subject               TEXT NOT NULL,
    body                  TEXT NOT NULL,
    word_count            INTEGER NOT NULL,
    generation_attempt    INTEGER NOT NULL DEFAULT 1,
    product_snapshot_json TEXT NOT NULL,
    claim_verifications   TEXT NOT NULL,
    quality_evaluation    TEXT NOT NULL,
    human_review_status   TEXT NOT NULL DEFAULT 'pending_review'
        CHECK (human_review_status IN ('pending_review', 'approved', 'rejected', 'needs_revision')),
    reviewer_notes        TEXT,
    current_revision      INTEGER NOT NULL DEFAULT 1,
    created_at            TEXT NOT NULL,
    deleted_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_outreach_drafts_context ON outreach_drafts(context_id);
CREATE INDEX IF NOT EXISTS idx_outreach_drafts_session ON outreach_drafts(research_session_id);

CREATE TABLE IF NOT EXISTS human_review_events (
    event_id              TEXT PRIMARY KEY,
    draft_id              TEXT NOT NULL REFERENCES outreach_drafts(draft_id) ON DELETE CASCADE,
    action                TEXT NOT NULL DEFAULT 'APPROVED'
        CHECK (action IN ('APPROVED', 'REJECTED', 'EDITED')),
    decision              TEXT,
    reviewed_revision     INTEGER NOT NULL DEFAULT 1,
    resulting_revision    INTEGER NOT NULL DEFAULT 1,
    reviewer_id           TEXT,
    notes                 TEXT,
    reason_code           TEXT
        CHECK (reason_code IS NULL OR reason_code IN ('HALLUCINATION', 'TONE_MISMATCH', 'WRONG_CAPABILITY', 'TIMING_IRRELEVANCE', 'INSUFFICIENT_PERSONALIZATION', 'OTHER')),
    original_subject      TEXT,
    original_body         TEXT,
    resulting_subject     TEXT,
    resulting_body        TEXT,
    char_edit_distance    INTEGER,
    word_edit_distance    INTEGER,
    normalized_char_ratio REAL,
    normalized_word_ratio REAL,
    decided_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_human_review_events_draft ON human_review_events(draft_id);
CREATE INDEX IF NOT EXISTS idx_human_review_events_action ON human_review_events(action);
CREATE INDEX IF NOT EXISTS idx_human_review_events_decided ON human_review_events(decided_at);

-- Triggers enforcing append-only immutability at the persistence layer
CREATE TRIGGER IF NOT EXISTS prevent_human_review_events_update
BEFORE UPDATE ON human_review_events
BEGIN
    SELECT RAISE(ABORT, 'human_review_events is append-only; updates are not permitted');
END;

CREATE TRIGGER IF NOT EXISTS prevent_human_review_events_delete
BEFORE DELETE ON human_review_events
BEGIN
    SELECT RAISE(ABORT, 'human_review_events is append-only; deletions are not permitted');
END;

-- Read-only view for review events query interface
CREATE VIEW IF NOT EXISTS review_events AS
SELECT
    event_id,
    draft_id,
    action,
    reviewed_revision,
    resulting_revision,
    reviewer_id,
    notes,
    reason_code,
    original_subject,
    original_body,
    resulting_subject,
    resulting_body,
    char_edit_distance,
    word_edit_distance,
    normalized_char_ratio,
    normalized_word_ratio,
    decided_at AS timestamp
FROM human_review_events;

-- ============================================================================
-- 3. Research Sessions & Execution Plan
-- ============================================================================

CREATE TABLE IF NOT EXISTS research_sessions (
    session_id          TEXT PRIMARY KEY,
    prospect_id         TEXT NOT NULL REFERENCES prospects(prospect_id) ON DELETE RESTRICT,
    context_id          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'in_progress',
    started_at          TEXT NOT NULL,
    completed_at        TEXT,
    is_sufficient       INTEGER NOT NULL DEFAULT 0,
    is_complete         INTEGER NOT NULL DEFAULT 0,
    sufficiency_notes   TEXT,
    missing_information TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_sessions_prospect ON research_sessions(prospect_id);

CREATE TABLE IF NOT EXISTS research_objectives (
    objective_id   TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    type           TEXT NOT NULL
        CHECK (type IN ('neutral_discovery', 'product_oriented', 'identity_resolution', 'commercial_realization')),
    target_entity  TEXT NOT NULL
        CHECK (target_entity IN ('prospect', 'company', 'both')),
    description    TEXT NOT NULL,
    temporal_scope TEXT,
    priority       INTEGER NOT NULL DEFAULT 5,
    is_critical    INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_research_objectives_session ON research_objectives(session_id);

CREATE TABLE IF NOT EXISTS research_queries (
    query_id     TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    objective_id TEXT REFERENCES research_objectives(objective_id) ON DELETE SET NULL,
    query_text   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_queries_session ON research_queries(session_id);
CREATE INDEX IF NOT EXISTS idx_research_queries_objective ON research_queries(objective_id);

CREATE TABLE IF NOT EXISTS workflow_events (
    event_id    TEXT PRIMARY KEY,
    session_id  TEXT REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    job_id      TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT,
    occurred_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflow_events_session ON workflow_events(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_events_job ON workflow_events(job_id);

-- ============================================================================
-- 4. Sources, Search Executions & Evidence Passages
-- ============================================================================

CREATE TABLE IF NOT EXISTS sources (
    source_id             TEXT PRIMARY KEY,
    canonical_url         TEXT NOT NULL UNIQUE,
    domain                TEXT NOT NULL,
    source_type           TEXT NOT NULL DEFAULT 'other',
    credibility_score     REAL NOT NULL DEFAULT 0.7,
    credibility_rationale TEXT,
    first_seen_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(canonical_url);
CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain);

CREATE TABLE IF NOT EXISTS search_executions (
    execution_id    TEXT PRIMARY KEY,
    query_id        TEXT NOT NULL REFERENCES research_queries(query_id) ON DELETE CASCADE,
    provider        TEXT NOT NULL,
    status          TEXT NOT NULL
        CHECK (status IN ('success', 'empty', 'failed')),
    error_message   TEXT,
    duration_ms     REAL,
    executed_at     TEXT NOT NULL,
    idempotency_key TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_search_executions_query ON search_executions(query_id);
CREATE INDEX IF NOT EXISTS idx_search_executions_idempotency ON search_executions(idempotency_key);

CREATE TABLE IF NOT EXISTS search_result_occurrences (
    occurrence_id TEXT PRIMARY KEY,
    execution_id  TEXT NOT NULL REFERENCES search_executions(execution_id) ON DELETE CASCADE,
    source_id     TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    title         TEXT,
    snippet       TEXT,
    original_url  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_occurrences_execution ON search_result_occurrences(execution_id);
CREATE INDEX IF NOT EXISTS idx_occurrences_source ON search_result_occurrences(source_id);

CREATE TABLE IF NOT EXISTS source_snapshots (
    snapshot_id          TEXT PRIMARY KEY,
    source_id            TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    session_id           TEXT REFERENCES research_sessions(session_id) ON DELETE SET NULL,
    original_url         TEXT,
    final_url            TEXT,
    html_canonical_url   TEXT,
    title                TEXT,
    raw_text             TEXT,
    raw_content_ref      TEXT,
    fetch_status         TEXT NOT NULL,
    http_status_code     INTEGER,
    publication_date     TEXT,
    publication_date_src TEXT,
    fetch_date           TEXT NOT NULL,
    duration_ms          REAL,
    raw_content_hash     TEXT,
    normalized_content_hash TEXT,
    minhash_signature    TEXT
);
CREATE INDEX IF NOT EXISTS idx_source_snapshots_source ON source_snapshots(source_id);
CREATE INDEX IF NOT EXISTS idx_source_snapshots_session ON source_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_source_snapshots_raw_hash ON source_snapshots(raw_content_hash);
CREATE INDEX IF NOT EXISTS idx_source_snapshots_norm_hash ON source_snapshots(normalized_content_hash);

CREATE TABLE IF NOT EXISTS evidence_passages (
    evidence_id       TEXT PRIMARY KEY,
    snapshot_id       TEXT NOT NULL REFERENCES source_snapshots(snapshot_id) ON DELETE CASCADE,
    source_id         TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    verbatim_excerpt  TEXT NOT NULL,
    excerpt_validated INTEGER NOT NULL DEFAULT 0,
    event_date        TEXT,
    extracted_at      TEXT NOT NULL,
    confidence        TEXT NOT NULL DEFAULT 'medium',
    credibility_score REAL NOT NULL DEFAULT 0.7
);
CREATE INDEX IF NOT EXISTS idx_evidence_passages_snapshot ON evidence_passages(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_evidence_passages_source ON evidence_passages(source_id);

CREATE TABLE IF NOT EXISTS evidence_origin_groups (
    group_id                 TEXT PRIMARY KEY,
    canonical_source_id      TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    description              TEXT,
    created_at               TEXT NOT NULL,
    dependence_type          TEXT,
    primary_detection_method TEXT,
    average_similarity       REAL
);

CREATE TABLE IF NOT EXISTS evidence_origin_members (
    group_id  TEXT NOT NULL REFERENCES evidence_origin_groups(group_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    PRIMARY KEY (group_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_origin_members_source ON evidence_origin_members(source_id);

-- ============================================================================
-- 5. Entity Registry & Identity Resolution
-- ============================================================================

CREATE TABLE IF NOT EXISTS entities (
    entity_id            TEXT PRIMARY KEY,
    entity_type          TEXT NOT NULL
        CHECK (entity_type IN ('person', 'organization')),
    raw_name             TEXT NOT NULL,
    canonical_name       TEXT,
    resolution_status    TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (resolution_status IN ('resolved', 'partially_resolved', 'unresolved', 'ambiguous')),
    confidence           TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (confidence IN ('high', 'medium', 'low', 'unresolved')),
    disambiguation_notes TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_name ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS entity_identifiers (
    identifier_id TEXT PRIMARY KEY,
    entity_id     TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    id_type       TEXT NOT NULL,
    id_value      TEXT NOT NULL,
    source_id     TEXT REFERENCES sources(source_id) ON DELETE SET NULL,
    observed_at   TEXT NOT NULL,
    UNIQUE (entity_id, id_type, id_value)
);
CREATE INDEX IF NOT EXISTS idx_entity_identifiers_lookup ON entity_identifiers(id_type, id_value);
CREATE INDEX IF NOT EXISTS idx_entity_identifiers_entity ON entity_identifiers(entity_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id    TEXT PRIMARY KEY,
    entity_id   TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    source_id   TEXT REFERENCES sources(source_id) ON DELETE SET NULL,
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(alias);

CREATE TABLE IF NOT EXISTS entity_affiliations (
    affiliation_id  TEXT PRIMARY KEY,
    entity_id       TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    organization_id TEXT REFERENCES entities(entity_id) ON DELETE SET NULL,
    role            TEXT,
    start_date      TEXT,
    end_date        TEXT,
    source_id       TEXT REFERENCES sources(source_id) ON DELETE SET NULL,
    observed_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_affiliations_entity ON entity_affiliations(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_affiliations_org ON entity_affiliations(organization_id);

CREATE TABLE IF NOT EXISTS entity_relationships (
    relationship_id   TEXT PRIMARY KEY,
    from_entity_id    TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    to_entity_id      TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    valid_from        TEXT,
    valid_until       TEXT,
    source_id         TEXT REFERENCES sources(source_id) ON DELETE SET NULL,
    observed_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_from ON entity_relationships(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_to ON entity_relationships(to_entity_id);

CREATE TABLE IF NOT EXISTS entity_candidates (
    candidate_id TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    entity_type  TEXT NOT NULL,
    raw_name     TEXT NOT NULL,
    attributes   TEXT NOT NULL,
    source_id    TEXT REFERENCES sources(source_id) ON DELETE SET NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_candidates_session ON entity_candidates(session_id);

CREATE TABLE IF NOT EXISTS identity_assessments (
    assessment_id     TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    entity_id         TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    candidate_ids     TEXT NOT NULL,
    resolution_status TEXT NOT NULL,
    confidence        TEXT NOT NULL,
    rationale         TEXT NOT NULL,
    assessed_by       TEXT NOT NULL,
    decision_metadata_json TEXT,
    assessed_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_identity_assessments_session ON identity_assessments(session_id);
CREATE INDEX IF NOT EXISTS idx_identity_assessments_entity ON identity_assessments(entity_id);

-- ============================================================================
-- 6. Canonical Facts, Evidence Links & Conflicts
-- ============================================================================

CREATE TABLE IF NOT EXISTS canonical_facts (
    fact_id              TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    claim_text           TEXT NOT NULL,
    fact_category        TEXT NOT NULL DEFAULT 'other',
    primary_event_date   TEXT,
    composite_confidence TEXT NOT NULL DEFAULT 'medium',
    molecular_data_json  TEXT,
    valid_from           TEXT,
    valid_to             TEXT,
    observed_at          TEXT,
    recorded_at          TEXT,
    independent_source_count INTEGER,
    ambiguous_candidates_json TEXT,
    entity_resolution_status TEXT,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_canonical_facts_session ON canonical_facts(session_id);
CREATE INDEX IF NOT EXISTS idx_canonical_facts_temporal ON canonical_facts(valid_from, valid_to, recorded_at);

CREATE TABLE IF NOT EXISTS fact_entity_links (
    fact_id   TEXT NOT NULL REFERENCES canonical_facts(fact_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_entity_links_entity ON fact_entity_links(entity_id);

CREATE TABLE IF NOT EXISTS fact_evidence_links (
    fact_id      TEXT NOT NULL REFERENCES canonical_facts(fact_id) ON DELETE CASCADE,
    evidence_id  TEXT NOT NULL REFERENCES evidence_passages(evidence_id) ON DELETE CASCADE,
    support_type TEXT NOT NULL DEFAULT 'supporting'
        CHECK (support_type IN ('supporting', 'contradicting', 'contextual')),
    PRIMARY KEY (fact_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_evidence_links_evidence ON fact_evidence_links(evidence_id);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id  TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    description  TEXT NOT NULL,
    severity     TEXT NOT NULL DEFAULT 'unknown'
        CHECK (severity IN ('confirmed', 'potential', 'unknown')),
    status       TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved', 'unresolvable')),
    resolved_at  TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conflicts_session ON conflicts(session_id);

CREATE TABLE IF NOT EXISTS conflict_facts (
    conflict_id TEXT NOT NULL REFERENCES conflicts(conflict_id) ON DELETE CASCADE,
    fact_id     TEXT NOT NULL REFERENCES canonical_facts(fact_id) ON DELETE CASCADE,
    PRIMARY KEY (conflict_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_conflict_facts_fact ON conflict_facts(fact_id);

CREATE TABLE IF NOT EXISTS assessment_history (
    assessment_id TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    iteration     INTEGER NOT NULL,
    assessor      TEXT NOT NULL,
    finding_type  TEXT NOT NULL,
    details       TEXT NOT NULL,
    assessed_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assessment_history_session ON assessment_history(session_id);

CREATE TABLE IF NOT EXISTS resolution_decisions (
    decision_id TEXT PRIMARY KEY,
    conflict_id TEXT NOT NULL REFERENCES conflicts(conflict_id) ON DELETE CASCADE,
    rationale   TEXT NOT NULL,
    decided_by  TEXT NOT NULL,
    decided_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resolution_decisions_conflict ON resolution_decisions(conflict_id);

-- ============================================================================
-- 7. Research Orchestration & Iteration Sufficiency
-- ============================================================================

CREATE TABLE IF NOT EXISTS research_iterations (
    iteration_id       TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL REFERENCES research_sessions(session_id) ON DELETE CASCADE,
    iteration_number   INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'completed', 'failed', 'interrupted')),
    queries_executed   INTEGER NOT NULL DEFAULT 0,
    snapshots_fetched  INTEGER NOT NULL DEFAULT 0,
    facts_extracted    INTEGER NOT NULL DEFAULT 0,
    conflicts_detected INTEGER NOT NULL DEFAULT 0,
    started_at         TEXT NOT NULL,
    completed_at       TEXT,
    error_message      TEXT,
    UNIQUE (session_id, iteration_number),
    UNIQUE (session_id, iteration_id)
);
CREATE INDEX IF NOT EXISTS idx_research_iterations_session ON research_iterations(session_id);

CREATE TABLE IF NOT EXISTS sufficiency_assessments (
    assessment_id            TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL,
    iteration_id             TEXT NOT NULL UNIQUE,
    decision                 TEXT NOT NULL
        CHECK (decision IN ('sufficient', 'continue', 'insufficient')),
    coverage_status          TEXT NOT NULL,
    identity_status          TEXT NOT NULL,
    evidence_status          TEXT NOT NULL,
    marginal_value_status    TEXT NOT NULL,
    explanation              TEXT NOT NULL,
    objective_details_json   TEXT NOT NULL,
    remaining_gaps_json      TEXT NOT NULL,
    recommended_actions_json TEXT NOT NULL,
    dimension_details_json   TEXT NOT NULL,
    assessed_at              TEXT NOT NULL,
    FOREIGN KEY (session_id, iteration_id) REFERENCES research_iterations(session_id, iteration_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sufficiency_assessments_session ON sufficiency_assessments(session_id);

-- ============================================================================
-- 8. Application Runs & Stage Lifecycle
-- ============================================================================

CREATE TABLE IF NOT EXISTS application_runs (
    run_id                  TEXT PRIMARY KEY,
    job_id                  TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
    prospect_id             TEXT REFERENCES prospects(prospect_id) ON DELETE CASCADE,
    context_id              TEXT REFERENCES outreach_contexts(context_id) ON DELETE SET NULL,
    research_session_id     TEXT REFERENCES research_sessions(session_id) ON DELETE SET NULL,
    draft_id                TEXT REFERENCES outreach_drafts(draft_id) ON DELETE SET NULL,
    status                  TEXT NOT NULL
        CHECK (status IN (
            'pending',
            'running',
            'completed',
            'completed_with_warnings',
            'insufficient_research',
            'no_viable_signal',
            'failed'
        )),
    prospect_snapshot_json  TEXT NOT NULL,
    product_snapshot_json   TEXT NOT NULL,
    outreach_snapshot_json  TEXT NOT NULL,
    strategy_result_json    TEXT,
    stage_statuses_json     TEXT NOT NULL,
    warnings_json           TEXT,
    error_message           TEXT,
    started_at              TEXT NOT NULL,
    completed_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_application_runs_status ON application_runs(status);
CREATE INDEX IF NOT EXISTS idx_application_runs_prospect ON application_runs(prospect_id);
CREATE INDEX IF NOT EXISTS idx_application_runs_session ON application_runs(research_session_id);
CREATE INDEX IF NOT EXISTS idx_application_runs_created ON application_runs(started_at);
