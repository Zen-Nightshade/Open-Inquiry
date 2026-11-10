from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore
from src.email_generation.generator import EmailGenerator, EmailGenerationError
from src.email_generation.linter import ReadabilityLinter
from src.email_generation.models import (
    EmailDraft,
    HumanReviewStatus,
    LinterResult,
    LinterSeverity,
    VerifiedOutreachDraft,
)
from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.researcher.models import ResearchPackage
from src.signal_strategy.models import (
    OutreachStrategy,
    StrategyResult,
)

logger = logging.getLogger(__name__)


class DraftPersistenceError(RuntimeError):
    """Raised when persistence of the generated draft to SQLite fails."""
    pass


class EmailGenerationService:
    """Operational service coordinating email draft generation, readability linting,
    targeted rewrites for critical errors, and draft persistence for human review.
    """

    def __init__(
        self,
        generator: Optional[EmailGenerator] = None,
        linter: Optional[ReadabilityLinter] = None,
        state_store: Optional[StateStore] = None,
    ) -> None:
        self.generator = generator or EmailGenerator()
        self.linter = linter or ReadabilityLinter()
        self.state_store = state_store

    def generate_draft(
        self,
        prospect: ProspectContext,
        product: ProductContext,
        strategy_result: StrategyResult,
        outreach: OutreachContext,
        research_package: Optional[ResearchPackage] = None,
        context_id: Optional[str] = None,
        research_session_id: Optional[str] = None,
        max_rewrite_attempts: int = 1,
    ) -> VerifiedOutreachDraft:
        """Generate, lint, and persist a personalized outreach draft.

        Validates the incoming strategy, enforces deterministic readability checks,
        allows at most one rewrite attempt for ERROR-level lint findings, and stores
        the resulting draft in PENDING_REVIEW state.
        """
        # 1. Validate strategy eligibility and presence
        if not strategy_result.is_viable:
            rejection = strategy_result.rejection_reason or "Strategy is not viable for outreach."
            raise ValueError(f"Cannot generate email draft: StrategyResult is not viable ({rejection}).")

        if not strategy_result.strategy:
            raise ValueError("Cannot generate email draft: StrategyResult contains no OutreachStrategy.")

        strategy: OutreachStrategy = strategy_result.strategy
        toulmin = getattr(strategy, "toulmin_argument", None)
        if not toulmin:
            raise ValueError(
                "Cannot generate email draft: OutreachStrategy lacks a validated ToulminArgument per M1.3 contract."
            )

        # 2. Initial draft generation
        attempt = 1
        raw_draft = self.generator.generate(
            prospect=prospect,
            product=product,
            strategy=strategy,
            outreach=outreach,
            research_package=research_package,
            attempt_num=attempt,
        )

        # 3. Readability & Quality Linting
        lint_res: LinterResult = self.linter.lint(
            subject=raw_draft.subject,
            body=raw_draft.body,
            strategy=strategy,
            outreach=outreach,
            toulmin_argument=toulmin,
        )

        # 4. At most one targeted rewrite attempt if ERROR findings exist (not for warnings alone)
        if lint_res.has_errors and max_rewrite_attempts > 0:
            error_explanations = [f.explanation for f in lint_res.findings if f.severity == LinterSeverity.ERROR]
            logger.info("Draft attempt 1 generated %d errors: %s. Attempting 1 targeted rewrite.", len(error_explanations), error_explanations)
            attempt += 1
            try:
                rewritten_draft = self.generator.generate(
                    prospect=prospect,
                    product=product,
                    strategy=strategy,
                    outreach=outreach,
                    research_package=research_package,
                    rewrite_feedback=error_explanations,
                    attempt_num=attempt,
                )
                # Re-lint rewritten draft
                rewritten_lint = self.linter.lint(
                    subject=rewritten_draft.subject,
                    body=rewritten_draft.body,
                    strategy=strategy,
                    outreach=outreach,
                    toulmin_argument=toulmin,
                )
                raw_draft = rewritten_draft
                lint_res = rewritten_lint
            except Exception as rewrite_err:
                logger.warning("Rewrite attempt failed (%s). Retaining original draft with lint errors for review.", rewrite_err)

        # 5. Assemble draft contract
        review_reasons: List[str] = [
            f"{f.severity.value.upper()}: [{f.rule_id}] {f.explanation}"
            for f in lint_res.findings
            if f.severity in (LinterSeverity.ERROR, LinterSeverity.WARNING)
        ]

        email_draft = EmailDraft(
            subject=raw_draft.subject,
            body=raw_draft.body,
            word_count=lint_res.word_count,
            generation_attempt=attempt,
        )

        verified_draft = VerifiedOutreachDraft(
            draft_id=str(uuid.uuid4()),
            draft=email_draft,
            claim_verifications=[],
            quality_evaluation=None,
            research_package=research_package,
            outreach_strategy=strategy,
            linter_result=lint_res,
            selected_cta=(
                getattr(strategy, "cta_type", None)
                or getattr(toulmin, "allowed_cta_friction", None)
            ),
            cta_text=getattr(strategy, "cta", None) or getattr(strategy, "call_to_action", ""),
            provenance=OutputProvenance(
                origin=ProvenanceOrigin.MODEL,
                component="email_generation_service",
                model_name=getattr(self.generator, "provider_name", "groq"),
                rule_id=f"toulmin_strategy_{strategy.strategy_id}",
            ),
            generation_attempts=attempt,
            human_review_status=HumanReviewStatus.PENDING_REVIEW,
            human_review_reasons=review_reasons,
        )

        # 6. Persist if state store is available
        if self.state_store:
            self._persist_draft(
                draft=verified_draft,
                prospect=prospect,
                product=product,
                outreach=outreach,
                context_id=context_id,
                session_id=research_session_id,
            )

        return verified_draft

    def _persist_draft(
        self,
        draft: VerifiedOutreachDraft,
        prospect: ProspectContext,
        product: ProductContext,
        outreach: OutreachContext,
        context_id: Optional[str],
        session_id: Optional[str],
    ) -> None:
        """Persist generated draft to SQLite outreach_drafts table within UnitOfWork."""
        cid = context_id
        sess_id = session_id or f"session_{uuid.uuid4().hex[:12]}"

        try:
            # If context_id is not provided or does not exist, ensure a valid context exists
            if not cid:
                pid = self.state_store.save_prospect(prospect)
                vid = self.state_store.save_product_context(product)
                cid = self.state_store.save_outreach_context(
                    outreach=outreach,
                    prospect_id=pid,
                    version_id=vid,
                )

            with self.state_store.uow() as uow:
                uow.outreach.create_outreach_draft(
                    draft_id=draft.draft_id,
                    context_id=cid,
                    research_session_id=sess_id,
                    subject=draft.draft.subject,
                    body=draft.draft.body,
                    word_count=draft.draft.word_count,
                    generation_attempt=draft.generation_attempts,
                    product_snapshot_json=product.model_dump_json(),
                    claim_verifications=json.dumps([]),
                    quality_evaluation=draft.linter_result.model_dump_json() if draft.linter_result else "{}",
                    human_review_status=HumanReviewStatus.PENDING_REVIEW.value,
                    reviewer_notes=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            logger.info("Persisted outreach draft %s for context %s (PENDING_REVIEW)", draft.draft_id, cid)
        except Exception as exc:
            logger.error("Failed to persist outreach draft %s to SQLite: %s", draft.draft_id, exc)
            raise DraftPersistenceError(f"Database persistence of outreach draft failed: {exc}") from exc
