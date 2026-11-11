from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple
from .models import ObservabilityClass


class ObservabilityClassifier:
    """Predicate-aware classifier evaluating whether a factual premise is observable via public channels.
    
    Guarantees:
      - Conservative decision-making bounded by available public research channels.
      - An entity being an SMB or privately held does NOT make all its facts unobservable.
      - Empty search results do NOT establish unobservability (they reflect scope exhaustion or lack of evidence).
      - Provider failures do NOT establish unobservability (they reflect operational errors).
      - UNKNOWN never silently becomes PRIVATE_OR_NONPUBLIC.
      - Distinguishes private facts from observable public proxies (without conflating them).
    """

    # Publicly observable predicate tokens and patterns
    PUBLIC_PREDICATES = {
        "office_expansion",
        "announced_funding",
        "funding_round",
        "press_release",
        "executive_appointment",
        "leadership_change",
        "public_announcement",
        "headcount_growth",
        "job_posting",
        "founded",
        "registered_address",
        "sec_filing",
        "acquisition_announcement",
        "product_launch",
    }

    # Conditionally observable predicate tokens (observable via indirect sources, job boards, case studies)
    CONDITIONAL_PREDICATES = {
        "works_for",
        "current_role",
        "current_title",
        "employment_status",
        "uses_vendor",
        "tech_stack",
        "software_stack",
        "partnership",
        "integration",
        "career_history",
        "reporting_structure",
        "customer_deployment",
        "deployment",
        "case_study",
        "customer_adoption",
    }

    # Inherently private / non-public predicates
    PRIVATE_PREDICATES = {
        "internal_budget",
        "infrastructure_budget",
        "internal_dissatisfaction",
        "vendor_dissatisfaction",
        "private_cash_flow",
        "unannounced_roadmap",
        "board_deliberation",
        "internal_sentiment",
        "proprietary_margin",
    }

    # Known public proxy mappings for private facts
    PROXY_MAPPINGS = {
        "internal_budget": {
            "proxy_predicate": "hiring_volume_for_infrastructure",
            "proxy_question": "Has the company publicly posted open roles for infrastructure or platform engineering?",
            "rationale": "Public job postings indicate infrastructure investment without exposing internal financial ledgers.",
        },
        "infrastructure_budget": {
            "proxy_predicate": "infrastructure_job_posting",
            "proxy_question": "Are there public hiring requisitions for infrastructure engineers or cloud architects?",
            "rationale": "Job requisitions reflect technology investment proxying internal infrastructure budgets.",
        },
        "internal_dissatisfaction": {
            "proxy_predicate": "public_rfp_issued",
            "proxy_question": "Has the organization issued a public Request for Proposal (RFP) or migration tender?",
            "rationale": "Public RFP filings or migration tenders indicate system evaluation without proving private internal dissatisfaction.",
        },
        "vendor_dissatisfaction": {
            "proxy_predicate": "evaluates_alternative_rfp",
            "proxy_question": "Has the organization publicly solicited proposals for replacing their software vendor?",
            "rationale": "Public solicitation proxies replacement evaluation while keeping private dissatisfaction distinct.",
        },
        "unannounced_roadmap": {
            "proxy_predicate": "trademark_or_patent_filing",
            "proxy_question": "Has the company filed public patents or trademarks indicating future capability areas?",
            "rationale": "Patent and trademark filings proxy forward-looking R&D without disclosing proprietary internal roadmaps.",
        },
    }

    @classmethod
    def classify(
        cls,
        predicate: str,
        question_text: str = "",
        is_smb: bool = False,
        is_private: bool = False,
        search_status: Optional[str] = None,
    ) -> Tuple[ObservabilityClass, str, Optional[Dict[str, str]]]:
        """Classify observability of a factual premise.
        
        Returns:
            Tuple of (ObservabilityClass, rationale, optional proxy specification dict).
        """
        # Strict Invariant: Empty search or provider failure must NEVER infer unobservability
        if search_status in ("empty", "failed"):
            return (
                ObservabilityClass.UNKNOWN,
                f"Search outcome '{search_status}' indicates search execution status, not metaphysical or channel unobservability.",
                None,
            )

        norm_pred = predicate.strip().lower()
        norm_q = question_text.strip().lower()

        # Check explicit private predicates
        is_private_predicate = any(p in norm_pred for p in cls.PRIVATE_PREDICATES)
        private_pattern_match = bool(
            re.search(
                r"\b(internal budget|dissatisfied|dissatisfaction|private cash|secret roadmap|unannounced roadmap|unannounced product|board deliberation|internal sentiment|private ledger)\b",
                norm_q,
            )
        )

        if is_private_predicate or private_pattern_match:
            matched_key = next((p for p in cls.PRIVATE_PREDICATES if p in norm_pred), None)
            if not matched_key:
                for k in cls.PROXY_MAPPINGS:
                    if k.replace("_", " ") in norm_q:
                        matched_key = k
                        break

            proxy_info = cls.PROXY_MAPPINGS.get(matched_key) if matched_key else None
            rationale = (
                "The requested fact concerns inherently internal, proprietary, or private organizational data "
                "not reasonably accessible through legitimate public research channels."
            )
            return (ObservabilityClass.PRIVATE_OR_NONPUBLIC, rationale, proxy_info)

        # Check explicit publicly observable predicates
        is_public_predicate = any(p in norm_pred for p in cls.PUBLIC_PREDICATES)
        public_pattern_match = bool(
            re.search(
                r"\b(publicly announced|press release|opened new office|expansion|funding round|sec filing|registered address|hiring|job posting|founded in)\b",
                norm_q,
            )
        )

        if is_public_predicate or public_pattern_match:
            rationale = (
                "The requested fact is ordinarily available via official announcements, public filings, "
                "company websites, press releases, or published hiring activity."
            )
            if is_smb or is_private:
                rationale += " Note: SMB or private company status does not restrict publicly published milestones."
            return (ObservabilityClass.PUBLICLY_OBSERVABLE, rationale, None)

        # Check conditionally observable predicates
        is_conditional_predicate = any(p in norm_pred for p in cls.CONDITIONAL_PREDICATES)
        conditional_pattern_match = bool(
            re.search(
                r"\b(works for|employed|current title|current role|uses vendor|tech stack|software stack|reporting structure|partnership|integration|customer deployment|deployment|deployments|case study|customer case study|adoption)\b",
                norm_q,
            )
        )

        if is_conditional_predicate or conditional_pattern_match:
            rationale = (
                "The requested fact is conditionally observable through specialized sources, technical footprints, "
                "case studies, job descriptions, or verified professional profiles."
            )
            return (ObservabilityClass.CONDITIONALLY_OBSERVABLE, rationale, None)

        # Conservative fallback: UNKNOWN (never assume private without specific evidence)
        return (
            ObservabilityClass.UNKNOWN,
            "Insufficient context to determine public observability with high confidence. Proceeding with caution.",
            None,
        )
