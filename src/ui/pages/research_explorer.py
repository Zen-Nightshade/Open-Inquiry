from __future__ import annotations

import json
import streamlit as st

from src.ui.state import (
    format_iso_timestamp,
    get_model_traces,
    get_run_snapshot,
    get_session_entities,
    get_session_facts_and_evidence,
    get_session_queries,
    get_session_sources,
    list_runs,
)


def render_research_explorer() -> None:
    st.title("Research Explorer & Epistemic Audit")
    st.caption("Inspect explored web sources, canonical facts, evidence passages, epistemic DAG, and redacted model traces.")

    all_runs = list_runs(limit=50)
    if not all_runs:
        st.info("No research runs found in database. Start a new run to populate research artifacts.")
        return

    run_ids = [r["run_id"] for r in all_runs]
    current_selected = st.session_state.get("selected_run_id")
    selected_idx = 0
    if current_selected in run_ids:
        selected_idx = run_ids.index(current_selected)

    selected_run_id = st.selectbox(
        "Select Research Run to Inspect",
        options=run_ids,
        index=selected_idx,
        format_func=lambda x: f"Run {x[:8]}...{x[-4:]} ({next((r['status'] for r in all_runs if r['run_id'] == x), '')})",
    )
    st.session_state["selected_run_id"] = selected_run_id

    snapshot = get_run_snapshot(selected_run_id)
    if not snapshot:
        st.error("Could not load snapshot for selected run.")
        return

    session_id = snapshot.research_session_id
    if not session_id:
        st.warning("No research session associated with this run.")
        return

    # Tabs for distinct investigation areas
    tab_sources, tab_facts, tab_dag, tab_queries, tab_entities, tab_traces = st.tabs([
        "🌐 Sources Explored",
        "📌 Canonical Facts & Evidence",
        "🕸️ Epistemic Question DAG",
        "🔍 Research Queries",
        "👤 Entity Resolution",
        "🤖 LLM Observability Traces",
    ])

    # 1. Sources Explored Tab
    with tab_sources:
        st.subheader("Discovered Web Sources & Snapshots")
        st.info(
            "ℹ️ **Credibility Score Disclosure**: In the Phase 2 pipeline architecture (Step B0), "
            "historical snapshot records are hydrated with an empirical heuristic credibility score of `0.8` by default "
            "rather than a measured Bayesian evaluation. True source authority varies by domain and claim category."
        )

        sources = get_session_sources(session_id)
        if not sources:
            st.write("No sources captured for this session.")
        else:
            st.metric("Total Sources Captured", len(sources))
            for idx, s in enumerate(sources):
                with st.container(border=True):
                    c1, c2 = st.columns([7, 3])
                    with c1:
                        st.markdown(f"**[{s['domain']}]({s['url']})**")
                        st.caption(f"Canonical URL: `{s['url']}`")
                    with c2:
                        st.markdown(f"Credibility: `{s['credibility_score']}` *(default heuristic)*")
                        st.caption(f"Fetched: {format_iso_timestamp(s.get('fetched_at'))} | Bytes: {s.get('content_length', 0)}")
                    
                    if s.get("clean_text_preview"):
                        with st.expander("Clean Content Preview", expanded=False):
                            st.text(s["clean_text_preview"])

    # 2. Canonical Facts & Evidence Tab
    with tab_facts:
        st.subheader("Corroborated Canonical Facts")
        facts = get_session_facts_and_evidence(session_id)
        if not facts:
            st.write("No canonical facts extracted for this session.")
        else:
            st.metric("Corroborated Facts Count", len(facts))
            for f in facts:
                with st.container(border=True):
                    st.markdown(f"**Claim:** *\"{f['claim_text']}\"*")
                    cf1, cf2, cf3 = st.columns(3)
                    cf1.caption(f"Category: `{f.get('category') or 'general'}`")
                    cf2.caption(f"Confidence: `{f.get('confidence_level') or 'medium'}`")
                    cf3.caption(f"Evidence Passages: `{f['evidence_count']}`")

                    if f.get("entities"):
                        st.caption(f"Linked Entities: {', '.join(f['entities'])}")

                    if f["evidence_passages"]:
                        with st.expander(f"View Supporting Evidence Passages ({len(f['evidence_passages'])})", expanded=False):
                            for p in f["evidence_passages"]:
                                st.markdown(f"> \"{p['verbatim_excerpt']}\"")
                                st.caption(
                                    f"Source: {p.get('domain')} | Validated: {p.get('excerpt_validated')} | Date: {p.get('event_date') or 'Unknown'}"
                                )
                                st.divider()

    # 3. Epistemic Question DAG Tab
    with tab_dag:
        st.subheader("Epistemic Question Dependency Graph")
        dag_summary: dict | None = None
        if snapshot.strategy_result and hasattr(snapshot, "feedback_round"):
            pass

        # Try to extract question DAG from research package snapshot if available in raw state
        store_dag_data = None
        try:
            from src.ui.state import get_state_store
            with get_state_store().uow() as uow:
                cur = uow.conn.cursor()
                row = cur.execute(
                    "SELECT question_dag_json FROM research_sessions WHERE session_id = ?;",
                    (session_id,),
                ).fetchone()
                if row and row[0]:
                    store_dag_data = json.loads(row[0])
        except Exception:
            pass

        if store_dag_data:
            st.json(store_dag_data)
        else:
            st.info("Epistemic Question DAG summary:")
            # Display reconstructed multi-turn questions
            queries = get_session_queries(session_id)
            if queries:
                for q in queries:
                    st.markdown(f"- ❓ **Question / Objective:** `{q.get('query_text')}`")
                    st.caption(f"Status: `{q.get('status', 'executed')}` | Results: `{q.get('result_count', 0)}`")
            else:
                st.write("No questions recorded in session DAG.")

    # 4. Research Queries Tab
    with tab_queries:
        st.subheader("Executed Search & Discovery Queries")
        queries = get_session_queries(session_id)
        if not queries:
            st.write("No queries recorded for this session.")
        else:
            st.metric("Total Search Queries Executed", len(queries))
            for q in queries:
                with st.container(border=True):
                    st.markdown(f"**Query:** `{q.get('query_text')}`")
                    q1, q2 = st.columns(2)
                    q1.caption(f"Status: `{q.get('status', 'completed')}`")
                    q2.caption(f"Occurrences Found: `{q.get('result_count', 0)}`")

    # 5. Entity Resolution Tab
    with tab_entities:
        st.subheader("Prospect & Company Identity Resolution")
        entities = get_session_entities(session_id)
        if not entities:
            if snapshot.prospect:
                st.markdown(f"**Prospect Name:** `{snapshot.prospect.full_name}`")
                st.markdown(f"**Company Name:** `{snapshot.prospect.company_name}`")
                st.markdown(f"**Domain:** `{snapshot.prospect.company_domain or 'None'}`")
                st.info("No ambiguous identity collisions detected during resolution.")
            else:
                st.write("No entity resolution records.")
        else:
            for ent in entities:
                with st.container(border=True):
                    st.markdown(f"**Entity:** {ent.get('primary_name')} ({ent.get('entity_type')})")
                    st.caption(f"Resolution Status: `{ent.get('decision', 'RESOLVED')}`")
                    st.json(ent)

    # 6. LLM Observability Traces Tab
    with tab_traces:
        st.subheader("Model Execution Traces & Latency Telemetry")
        st.caption("Zero-leakage telemetry: API keys, authorization tokens, and credentials are authenticated and redacted.")
        traces = get_model_traces(run_id=selected_run_id, job_id=snapshot.job_id)
        if not traces:
            st.info("No model execution records found for this run ID in `model_executions`.")
        else:
            st.metric("Model Calls Logged", len(traces))
            for t in traces:
                with st.container(border=True):
                    tr1, tr2, tr3, tr4 = st.columns([3, 3, 2, 2])
                    tr1.markdown(f"**Task:** `{t.get('task_type')}`")
                    tr2.markdown(f"**Model:** `{t.get('provider')}` / `{t.get('model_name')}`")
                    tr3.markdown(f"Duration: `{t.get('duration_ms', 0):.1f}ms`")
                    tr4.badge(t.get("status", "completed").upper(), color="green" if t.get("status") == "completed" else "red")

                    col_tokens, col_hash = st.columns(2)
                    col_tokens.caption(f"Tokens: In={t.get('input_tokens') or '—'}, Out={t.get('output_tokens') or '—'}")
                    col_hash.caption(f"Prompt SHA-256: `{t.get('prompt_hash') or '—'}`")

                    # Explicit disclosure on prompt privacy
                    with st.expander("Prompt & Completion Telemetry", expanded=False):
                        st.markdown(
                            "**Prompt Content:** *Unavailable (persisted as SHA-256 hash for privacy/zero-data-leakage guarantee)*"
                        )
                        if t.get("error_message"):
                            st.error(f"Error: {t['error_message']}")
