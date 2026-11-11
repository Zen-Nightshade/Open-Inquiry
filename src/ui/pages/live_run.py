from __future__ import annotations

import streamlit as st
from src.orchestration.models import RunStatus, StageExecutionRecord, StageIdentifier, StageStatus
from src.ui.runner import is_run_in_progress
from src.ui.state import (
    format_duration,
    format_iso_timestamp,
    get_run_snapshot,
    get_run_stages,
    list_runs,
)

# Standard pipeline stage ordering and display labels
STAGE_METADATA = [
    (StageIdentifier.VALIDATION, "1. Input Context Validation", "Verify required prospect and product boundaries"),
    (StageIdentifier.INITIALIZATION, "2. Session Initialization", "Persist contexts, research session, and job"),
    (StageIdentifier.RESEARCH, "3. Multi-Turn Research", "Execute multi-turn question DAG, web discovery, and fact extraction"),
    (StageIdentifier.SUFFICIENCY_GATE, "4. Epistemic Sufficiency Gate", "Enforce corroboration thresholds and identity resolution"),
    (StageIdentifier.SIGNAL_STRATEGY, "5. Signal Strategy Synthesis", "Identify viable commercial catalysts and formulate Toulmin blueprint"),
    (StageIdentifier.RESEARCH_FEEDBACK, "5b. Bounded Evidence Feedback", "Resolve evidence gaps with directed query feedback (if needed)"),
    (StageIdentifier.EMAIL_GENERATION, "6. Email Generation & Linting", "Synthesize brevity-bounded outreach and run readability linter"),
]


@st.fragment(run_every=2.5)
def render_live_pipeline_view(run_id: str) -> None:
    snapshot = get_run_snapshot(run_id)
    if not snapshot:
        st.warning(f"Run `{run_id}` not found in database.")
        return

    stages = get_run_stages(run_id)
    is_active = is_run_in_progress(run_id)

    # Header status banner
    status_val = snapshot.status.value
    if is_active:
        status_val = "RUNNING"

    badge_color = "gray"
    if status_val in ("completed", "completed_with_warnings"):
        badge_color = "green"
    elif status_val in ("running", "pending"):
        badge_color = "blue"
    elif status_val in ("insufficient_research", "no_viable_signal"):
        badge_color = "orange"
    elif status_val == "failed":
        badge_color = "red"

    st.markdown("### Pipeline Execution Status")
    c_status, c_dur, c_prog = st.columns([3, 3, 4])
    with c_status:
        st.badge(status_val.upper(), color=badge_color)
        if is_active:
            st.caption("⚡ Background worker running...")
        else:
            st.caption("Execution finished.")
    with c_dur:
        st.markdown(f"**Started:** {format_iso_timestamp(snapshot.started_at)}")
        st.markdown(f"**Duration:** {format_duration(snapshot.started_at, snapshot.completed_at)}")
    with c_prog:
        if snapshot.prospect:
            st.markdown(f"**Prospect:** {snapshot.prospect.full_name} ({snapshot.prospect.company_name})")
        if snapshot.product:
            st.markdown(f"**Product:** {snapshot.product.product_name}")

    if snapshot.error_message:
        st.error(f"⚠️ **Stopping Reason / Error:** {snapshot.error_message}")

    if snapshot.warnings:
        with st.expander(f"⚠️ Execution Warnings ({len(snapshot.warnings)})", expanded=False):
            for w in snapshot.warnings:
                st.warning(w)

    st.divider()

    # Stage cards
    stage_map = {s.stage: s for s in stages}

    for stage_id, label, description in STAGE_METADATA:
        record: StageExecutionRecord | None = stage_map.get(stage_id)

        # Determine visual state
        if record:
            st_color = "green" if record.status == StageStatus.COMPLETED else (
                "red" if record.status == StageStatus.FAILED else (
                    "gray" if record.status == StageStatus.SKIPPED else "blue"
                )
            )
            st_label = record.status.value.upper()
        else:
            # If pipeline is still running and previous stage completed, this one might be next
            st_color = "gray"
            st_label = "PENDING"

        with st.container(border=True):
            col_icon, col_txt, col_badge = st.columns([1, 7, 2])
            with col_icon:
                if st_label == "COMPLETED":
                    st.markdown("### ✅")
                elif st_label == "RUNNING":
                    st.markdown("### ⏳")
                elif st_label == "FAILED":
                    st.markdown("### ❌")
                elif st_label == "SKIPPED":
                    st.markdown("### ⏭️")
                else:
                    st.markdown("### ⚪")
            with col_txt:
                st.markdown(f"**{label}**")
                st.caption(description)
                if record and record.duration_ms:
                    st.caption(f"Elapsed: {record.duration_ms:.1f}ms")
            with col_badge:
                st.badge(st_label, color=st_color)

            # Details expander if details or error present
            if record and (record.details or record.error_message):
                with st.expander("Stage Details & Telemetry", expanded=False):
                    if record.error_message:
                        st.error(f"Error: {record.error_message}")
                    if record.details:
                        st.json(record.details)

    st.divider()

    # Post-run Navigation Buttons
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("🔄 Refresh Data", key=f"ref_{run_id}", use_container_width=True):
            st.rerun()
    with c_btn2:
        if st.button("🔍 Explore Research Data", key=f"to_exp_{run_id}", use_container_width=True):
            st.session_state["selected_run_id"] = run_id
            st.session_state["nav_page"] = "Research Explorer"
            st.rerun()
    with c_btn3:
        if snapshot.draft_id:
            if st.button("✍️ Review Generated Draft", key=f"to_rev_{run_id}", type="primary", use_container_width=True):
                st.session_state["selected_run_id"] = run_id
                st.session_state["selected_draft_id"] = snapshot.draft_id
                st.session_state["nav_page"] = "Draft Review"
                st.rerun()


def render_live_run() -> None:
    st.title("Live Run Telemetry")
    st.caption("Real-time pipeline stage transitions, epistemic sufficiency gating, and execution logs.")

    all_runs = list_runs(limit=50)
    if not all_runs:
        st.info("No runs found in SQLite. Start a new run to view live execution.")
        if st.button("Go to New Run"):
            st.session_state["nav_page"] = "New Run"
            st.rerun()
        return

    # Run selector
    run_ids = [r["run_id"] for r in all_runs]
    current_selected = st.session_state.get("selected_run_id")
    selected_idx = 0
    if current_selected in run_ids:
        selected_idx = run_ids.index(current_selected)

    selected_run_id = st.selectbox(
        "Select Inquiry Run",
        options=run_ids,
        index=selected_idx,
        format_func=lambda x: f"Run {x[:8]}...{x[-4:]} ({next((r['status'] for r in all_runs if r['run_id'] == x), '')})",
    )
    st.session_state["selected_run_id"] = selected_run_id

    # Render pipeline view with fragment auto-refresh
    render_live_pipeline_view(selected_run_id)
