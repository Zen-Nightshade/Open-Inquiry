from __future__ import annotations

import streamlit as st
from src.orchestration.models import RunStatus
from src.ui.runner import is_run_in_progress
from src.ui.state import (
    ensure_db_initialized,
    format_duration,
    format_iso_timestamp,
    get_database_info,
    list_runs,
)


def render_dashboard() -> None:
    ensure_db_initialized()
    db_info = get_database_info()

    st.title("Open Inquiry — Research & Outreach Dashboard")
    st.caption("Epistemic Autonomous Research, Signal Synthesis & Grounded Outreach Generation")

    # Top metrics
    all_runs = list_runs(limit=200)
    total_runs = len(all_runs)
    completed_runs = sum(
        1 for r in all_runs if r["status"] in (RunStatus.COMPLETED.value, RunStatus.COMPLETED_WITH_WARNINGS.value)
    )
    failed_runs = sum(
        1 for r in all_runs if r["status"] in (RunStatus.FAILED.value, RunStatus.INSUFFICIENT_RESEARCH.value, RunStatus.NO_VIABLE_SIGNAL.value)
    )
    in_progress_runs = sum(
        1 for r in all_runs if is_run_in_progress(r["run_id"]) or r["status"] in (RunStatus.PENDING.value, RunStatus.RUNNING.value)
    )
    pending_drafts = db_info.get("pending_drafts", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Runs", total_runs)
    m2.metric("Completed", completed_runs)
    m3.metric("Running", in_progress_runs)
    m4.metric("Failed / Insufficient", failed_runs)
    m5.metric("Drafts for Review", pending_drafts)

    st.divider()

    # Empty database state handling
    if total_runs == 0:
        st.info("ℹ️ **Database Initialized (Empty State)**: No research runs found in `data/open_inquiry.db`.")
        st.write(
            """
            The schema with all 36 tables has been verified and initialized. 
            No simulated or fabricated records exist. To begin your first grounded research and outreach investigation,
            launch a new run below.
            """
        )
        if st.button("🚀 Launch First Research Run", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "New Run"
            st.rerun()
        return

    # Recent runs table
    st.subheader("Recent Research Runs")
    
    col_filter, col_refresh = st.columns([4, 1])
    with col_filter:
        status_filter = st.selectbox(
            "Filter by status",
            options=["All", "Completed", "Running / Pending", "Failed / Insufficient"],
            index=0,
        )
    with col_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    filtered_runs = all_runs
    if status_filter == "Completed":
        filtered_runs = [r for r in all_runs if r["status"] in (RunStatus.COMPLETED.value, RunStatus.COMPLETED_WITH_WARNINGS.value)]
    elif status_filter == "Running / Pending":
        filtered_runs = [r for r in all_runs if r["status"] in (RunStatus.PENDING.value, RunStatus.RUNNING.value)]
    elif status_filter == "Failed / Insufficient":
        filtered_runs = [r for r in all_runs if r["status"] in (RunStatus.FAILED.value, RunStatus.INSUFFICIENT_RESEARCH.value, RunStatus.NO_VIABLE_SIGNAL.value)]

    for run in filtered_runs:
        run_id = run["run_id"]
        status = run["status"]
        running = is_run_in_progress(run_id)
        if running:
            status = "RUNNING"

        badge_color = "gray"
        if status in ("completed", "completed_with_warnings"):
            badge_color = "green"
        elif status in ("running", "pending"):
            badge_color = "blue"
        elif status in ("insufficient_research", "no_viable_signal"):
            badge_color = "orange"
        elif status == "failed":
            badge_color = "red"

        with st.container(border=True):
            c_info, c_status, c_actions = st.columns([4, 2, 3])
            with c_info:
                st.markdown(f"**Run ID:** `{run_id[:8]}...{run_id[-4:]}`")
                st.markdown(f"Started: `{format_iso_timestamp(run.get('started_at'))}` | Duration: `{format_duration(run.get('started_at'), run.get('completed_at'))}`")
            with c_status:
                st.badge(status.upper(), color=badge_color)
                if run.get("draft_id"):
                    st.caption(f"Draft: `{run['draft_id'][:8]}...`")
                elif run.get("error_message"):
                    st.caption(f"Error: {run['error_message'][:40]}...")
            with c_actions:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("📊 Live Run", key=f"btn_live_{run_id}", use_container_width=True):
                        st.session_state["selected_run_id"] = run_id
                        st.session_state["nav_page"] = "Live Run"
                        st.rerun()
                with col_btn2:
                    if run.get("draft_id"):
                        if st.button("✍️ Review Draft", key=f"btn_draft_{run_id}", use_container_width=True):
                            st.session_state["selected_run_id"] = run_id
                            st.session_state["selected_draft_id"] = run["draft_id"]
                            st.session_state["nav_page"] = "Draft Review"
                            st.rerun()
                    else:
                        if st.button("🔍 Explorer", key=f"btn_exp_{run_id}", use_container_width=True):
                            st.session_state["selected_run_id"] = run_id
                            st.session_state["nav_page"] = "Research Explorer"
                            st.rerun()
