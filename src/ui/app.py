from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repo root to sys.path so modules resolve cleanly when running directly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from repo_paths import PROVIDERS_ENV, get_db_path
from src.ui.pages.dashboard import render_dashboard
from src.ui.pages.draft_review import render_draft_review
from src.ui.pages.live_run import render_live_run
from src.ui.pages.new_run import render_new_run
from src.ui.pages.research_explorer import render_research_explorer
from src.ui.pages.settings import render_settings
from src.ui.runner import is_run_in_progress
from src.ui.state import ensure_db_initialized, get_database_info


def init_app() -> None:
    """Initialize environment, Streamlit secrets, and database schema."""
    # 1. Load Streamlit secrets into environment if present
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            if not os.environ.get("GROQ_API_KEY"):
                os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
        if hasattr(st, "secrets") and "OPEN_INQUIRY_DB_PATH" in st.secrets:
            if not os.environ.get("OPEN_INQUIRY_DB_PATH"):
                os.environ["OPEN_INQUIRY_DB_PATH"] = str(st.secrets["OPEN_INQUIRY_DB_PATH"])
    except Exception:
        pass

    # 2. Load providers.env if available and GROQ_API_KEY is missing
    if not os.environ.get("GROQ_API_KEY") and PROVIDERS_ENV.exists():
        try:
            import dotenv
            dotenv.load_dotenv(PROVIDERS_ENV)
        except Exception:
            pass

    # 3. Ensure database schema is initialized and verified
    ensure_db_initialized()


def main() -> None:
    st.set_page_config(
        page_title="Open Inquiry",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_app()

    # Sidebar Navigation & Branding
    st.sidebar.markdown("# ⚡ Open Inquiry")
    st.sidebar.caption("Epistemic Autonomous Outreach Engine")
    st.sidebar.divider()

    pages = [
        "Dashboard",
        "New Run",
        "Live Run",
        "Research Explorer",
        "Draft Review",
        "Settings",
    ]

    # Handle programmatic navigation redirects
    current_page = st.session_state.get("nav_page", "Dashboard")
    if current_page not in pages:
        current_page = "Dashboard"

    selected_page = st.sidebar.radio(
        "Navigation",
        options=pages,
        index=pages.index(current_page),
        key="nav_radio",
    )

    # Sync back if user clicked a radio option
    if selected_page != current_page:
        st.session_state["nav_page"] = selected_page
        st.rerun()

    # Sidebar Status Card
    st.sidebar.divider()
    active_run = st.session_state.get("selected_run_id")
    if active_run:
        running = is_run_in_progress(active_run)
        with st.sidebar.container(border=True):
            st.markdown(f"**Active Run:** `{active_run[:8]}...`")
            if running:
                st.badge("RUNNING", color="blue")
            else:
                st.badge("IDLE / COMPLETED", color="gray")

    # DB footprint in sidebar
    db_info = get_database_info()
    st.sidebar.caption(
        f"DB: `{Path(db_info['db_path']).name}` ({db_info['table_count']} tables | {db_info['total_runs']} runs)"
    )

    # Render selected page
    if current_page == "Dashboard":
        render_dashboard()
    elif current_page == "New Run":
        render_new_run()
    elif current_page == "Live Run":
        render_live_run()
    elif current_page == "Research Explorer":
        render_research_explorer()
    elif current_page == "Draft Review":
        render_draft_review()
    elif current_page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
