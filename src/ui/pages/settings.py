from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from repo_paths import PROVIDERS_ENV, get_db_path
from src.ui.state import get_database_info


def test_groq_connection(api_key: str) -> tuple[bool, str]:
    """Test authentication against Groq API without exposing or persisting key."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        # Call a lightweight call or list models
        models = client.models.list()
        avail = [m.id for m in models.data]
        return True, f"Authentication successful! {len(avail)} models accessible on Groq endpoint."
    except Exception as exc:
        err_msg = str(exc)
        if "401" in err_msg or "invalid_api_key" in err_msg:
            return False, "Authentication Failure (HTTP 401): The provided API key is invalid or revoked."
        elif "429" in err_msg:
            return False, "Rate Limit Exceeded (HTTP 429): Request limit reached. Check rate limit headers (this is distinct from quota exhaustion)."
        elif "quota" in err_msg.lower() or "billing" in err_msg.lower() or "402" in err_msg:
            return False, "Quota / Billing Exhaustion (HTTP 402): Account has exceeded its available balance or credits."
        elif "503" in err_msg or "500" in err_msg or "connection" in err_msg.lower():
            return False, f"Provider Outage / Network Issue: {err_msg}"
        return False, f"Provider Connection Error: {err_msg}"


def render_settings() -> None:
    st.title("Settings & Deployment Configuration")
    st.caption("Manage BYOK API keys, inspect SQLite persistence boundaries, and review environment configuration.")

    # 1. BYOK Section
    st.subheader("1. Bring Your Own Key (BYOK) — API Credentials")
    st.markdown(
        """
        Open Inquiry supports **session-scoped Bring Your Own Key (BYOK)**.
        
        **Key Resolution Precedence:**
        1. **Session User Key** *(highest priority — entered in this browser session)*
        2. **Streamlit Secrets** (`st.secrets["GROQ_API_KEY"]`)
        3. **Deployment Environment** (`os.environ["GROQ_API_KEY"]` or `providers.env`)
        """
    )

    current_session_key = st.session_state.get("user_byok_api_key", "")
    deployment_key_present = bool(os.environ.get("GROQ_API_KEY")) or (
        PROVIDERS_ENV.exists() and "GROQ_API_KEY" in PROVIDERS_ENV.read_text()
    )

    col_k1, col_k2 = st.columns([3, 1])
    with col_k1:
        new_key = st.text_input(
            "Groq API Key (Session-Scoped)",
            value=current_session_key,
            type="password",
            placeholder="gsk_...",
            help="Supplied key is preserved only in this browser session's transient memory. Never written to SQLite, logs, or disk.",
        )
    with col_k2:
        st.write("")
        st.write("")
        if st.button("Save Session Key", use_container_width=True):
            st.session_state["user_byok_api_key"] = new_key.strip()
            st.success("Session key saved!")
            st.rerun()

    # Precedence indicator
    if current_session_key:
        st.success("🟢 **Active Provider Source:** User-Provided Session Key (Precedence Level 1)")
    elif deployment_key_present:
        st.info("🔵 **Active Provider Source:** Deployment-Provided Key (Precedence Level 3)")
    else:
        st.warning("⚠️ **No Active Key Found**: Supply a key above or configure `GROQ_API_KEY` in environment.")

    # Test Connection Button
    key_to_test = current_session_key
    if not key_to_test:
        key_to_test = os.environ.get("GROQ_API_KEY", "")
        if not key_to_test and PROVIDERS_ENV.exists():
            try:
                import dotenv
                dotenv.load_dotenv(PROVIDERS_ENV)
                key_to_test = os.environ.get("GROQ_API_KEY", "")
            except Exception:
                pass

    if key_to_test:
        if st.button("🧪 Test Provider Connection", type="secondary"):
            with st.spinner("Connecting to provider endpoint..."):
                ok, msg = test_groq_connection(key_to_test)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
    
    st.divider()

    # 2. Database & Storage Boundaries
    st.subheader("2. Database & Persistence Boundaries")
    db_info = get_database_info()

    st.markdown(
        """
        > [!WARNING]
        > **Ephemeral Storage Limitation on Streamlit Cloud & Containers:**
        > Streamlit Community Cloud and ephemeral Docker containers do not provide durable storage across restarts.
        > The local SQLite database (`data/open_inquiry.db`) will be reset upon container teardown or app reboot.
        > For production deployments, configure `OPEN_INQUIRY_DB_PATH` to point to a persistent mounted volume
        > or cloud block storage.
        """
    )

    c_db1, c_db2, c_db3 = st.columns(3)
    c_db1.metric("Database Path", Path(db_info["db_path"]).name)
    c_db2.metric("Tables Initialized", f"{db_info['table_count']} / 36")
    c_db3.metric("Database Size", f"{db_info['file_size_bytes'] / 1024:.1f} KB")

    with st.expander("Database Status & Storage Details", expanded=False):
        st.json(db_info)

    st.divider()

    # 3. Deployment Configuration Instructions
    st.subheader("3. Deployment Secrets & Environment Variables")
    st.markdown(
        """
        When deploying to **Streamlit Community Cloud**, **Hugging Face Spaces**, or a Docker container:
        
        | Variable | Purpose | Default |
        |---|---|---|
        | `OPEN_INQUIRY_DB_PATH` | Path to SQLite database file | `data/open_inquiry.db` |
        | `GROQ_API_KEY` | Deployment default Groq LLM API key | None |
        | `HEADLESS` | Run Streamlit in headless server mode | `true` |
        
        **Streamlit Secrets Format (`.streamlit/secrets.toml`):**
        ```toml
        GROQ_API_KEY = "gsk_your_groq_api_key"
        OPEN_INQUIRY_DB_PATH = "data/open_inquiry.db"
        ```
        """
    )
