from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.ui.runner import start_background_run


def render_new_run() -> None:
    st.title("New Research & Outreach Inquiry")
    st.caption("Configure prospect profile, commercial product boundaries, and research constraints.")

    with st.form("new_run_form", clear_on_submit=False):
        st.subheader("1. Prospect Context")
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input(
                "Prospect Full Name *",
                value="Patrick Collison",
                help="Legal or operational full name of the decision maker.",
            )
            role_title = st.text_input(
                "Role / Designation",
                value="Chief Executive Officer",
                help="Job title or functional responsibility.",
            )
        with c2:
            company_name = st.text_input(
                "Company Name *",
                value="Stripe",
                help="Target enterprise or organization name.",
            )
            company_domain = st.text_input(
                "Company Domain",
                value="stripe.com",
                help="Primary website domain used for web discovery.",
            )

        st.subheader("2. Product Context")
        c3, c4 = st.columns(2)
        with c3:
            product_name = st.text_input(
                "Product Name *",
                value="Open Inquiry Intelligence",
                help="The commercial offering being proposed.",
            )
            product_desc = st.text_input(
                "Product Short Description *",
                value="Autonomous epistemic research & outreach intelligence platform",
                help="High-level description of what the product does.",
            )
            value_prop = st.text_area(
                "Value Proposition *",
                value="Autonomous epistemic research platform that synthesizes corroborated buying signals to drive grounded executive engagement.",
                height=90,
                help="Core commercial differentiation and problem-solution fit.",
            )
        with c4:
            capabilities = st.text_area(
                "Capabilities (one per line)",
                value="Epistemic DAG scheduling\nMulti-hop evidence corroboration\nDeterministic claim verification\nBrevity-enforced outreach generation",
                height=90,
                help="Explicit technical and operational capabilities of the product.",
            )
            constraints = st.text_area(
                "Constraints & Exclusions (one per line)",
                value="Requires publicly corroborable web footprint\nZero tolerance for hallucinated metrics or quotes",
                height=90,
                help="Operational boundaries, compliance limits, or negative claims.",
            )

        st.subheader("3. Outreach Context")
        c5, c6 = st.columns(2)
        with c5:
            sender_name_val = st.text_input(
                "Sender Name *",
                value="Alex Mercer",
                help="Name of the person or executive sending the outreach.",
            )
            objective = st.text_input(
                "Outreach Objective *",
                value="Schedule a 15-minute diagnostic consultation",
                help="Direct call to action purpose.",
            )
        with c6:
            sender_company_val = st.text_input(
                "Sender Organization",
                value="Open Inquiry Intelligence",
                help="Company or platform representing the sender.",
            )
            max_words = st.number_input(
                "Maximum Word Count",
                min_value=40,
                max_value=200,
                value=100,
                step=10,
                help="Strict brevity limit enforced during generation and linting.",
            )

        with st.expander("⚙️ Advanced Research & Budget Parameters", expanded=False):
            st.markdown("**Epistemic Budget and Scoping Controls**")
            col_iter, col_budget = st.columns(2)
            with col_iter:
                max_iterations = st.slider(
                    "Max Multi-Turn Research Iterations",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="Upper bound on iterative question DAG refinement loops.",
                )
            with col_budget:
                research_budget = st.slider(
                    "Global Research Budget Units",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Authoritative budget units allocated across search and page fetching.",
                )
            
            allow_insufficient = st.checkbox(
                "Allow Insufficient Research (Bypass Sufficiency Gate Halt)",
                value=False,
                help="If unchecked (default), research lacking corroborated facts or clear entity resolution halts safely at Gate 4.",
            )
            
            project_id = st.text_input(
                "Project ID (Scope)",
                value="",
                placeholder="Optional: leave blank for prospect-level isolation",
                help="Isolation boundary for historical evidence retrieval.",
            )
            st.caption(
                "ℹ️ **Project Isolation Disclosure**: When Project ID is left blank, the system falls back to "
                "prospect-level isolation (`effective_project_id = prospect_id`). Historical retrieval across runs "
                "will only reuse verified evidence matching this prospect."
            )

        submit_button = st.form_submit_button(
            "🚀 Launch Research Pipeline",
            type="primary",
            use_container_width=True,
        )

    if submit_button:
        # Validate required inputs
        clean_name = (full_name or "").strip()
        clean_company = (company_name or "").strip()
        clean_product = (product_name or "").strip()
        clean_desc = (product_desc or "").strip()
        clean_vp = (value_prop or "").strip()

        if not clean_name or not clean_company:
            st.error("❌ Validation Error: Prospect Full Name and Company Name are required.")
            return

        if not clean_product or not clean_vp or not clean_desc:
            st.error("❌ Validation Error: Product Name, Description, and Value Proposition are required.")
            return

        caps_list = [c.strip() for c in capabilities.split("\n") if c.strip()]
        constraints_list = [c.strip() for c in constraints.split("\n") if c.strip()]

        try:
            prospect_ctx = ProspectContext(
                full_name=clean_name,
                role_title=role_title.strip() if role_title else None,
                company_name=clean_company,
                company_domain=company_domain.strip() if company_domain else None,
            )
            product_ctx = ProductContext(
                product_name=clean_product,
                description=clean_desc,
                value_proposition=clean_vp,
                capabilities=caps_list,
                limitations=constraints_list,
            )
            clean_sname = (sender_name_val or "").strip() or "Alex Mercer"
            clean_scomp = (sender_company_val or "").strip() or "Open Inquiry Intelligence"
            outreach_ctx = OutreachContext(
                objective=objective.strip() if objective else "Schedule discovery conversation",
                sender_identity=f"{clean_sname}, Commercial Lead, {clean_scomp}",
                sender_name=clean_sname,
                sender_company=clean_scomp,
                desired_cta=objective.strip(),
                tone="professional",
                constraints=[f"Max {int(max_words)} words"],
            )
        except ValidationError as val_err:
            st.error(f"❌ Input validation failed: {val_err}")
            return

        # Check for user-provided BYOK key in session state
        byok_key = st.session_state.get("user_byok_api_key")

        with st.spinner("Initializing run and launching background pipeline..."):
            try:
                run_id = start_background_run(
                    prospect=prospect_ctx,
                    product=product_ctx,
                    outreach=outreach_ctx,
                    max_research_iterations=max_iterations,
                    allow_insufficient_research=allow_insufficient,
                    total_research_budget=research_budget,
                    project_id=project_id.strip() if project_id else None,
                    api_key_override=byok_key,
                )
                st.session_state["selected_run_id"] = run_id
                st.session_state["nav_page"] = "Live Run"
                st.success(f"Run `{run_id[:8]}` launched successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to launch run: {e}")
