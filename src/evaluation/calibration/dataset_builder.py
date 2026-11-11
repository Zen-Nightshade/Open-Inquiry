from __future__ import annotations

import json
import logging
import random
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.db.state_store import StateStore
from src.evaluation.calibration.models import (
    CalibrationDatasetRecord,
    CalibrationTarget,
    ScoreFamily,
    SplitStrategy,
)

logger = logging.getLogger(__name__)

MINIMUM_ANNOTATED_DRAFTS_GATE = 200


class CalibrationDatasetBuilder:
    """Builds and partitions empirical calibration datasets from persisted review history."""

    def __init__(self, state_store: Optional[StateStore] = None) -> None:
        self.state_store = state_store

    def build_dataset_from_db(
        self,
        conn: Optional[sqlite3.Connection] = None,
        target: CalibrationTarget = CalibrationTarget.DRAFT_ACCEPTANCE,
        score_family: ScoreFamily = ScoreFamily.SIGNAL_COMPOSITE,
    ) -> List[CalibrationDatasetRecord]:
        """Extract draft-level records with final review outcomes and associated raw scores.

        Enforces:
        - Revisions of the same draft are grouped together.
        - Final reviewed revision determines outcome (APPROVED -> 1, REJECTED -> 0).
        - Drafts with pending or intermediate EDITED status are excluded from binary labels.
        """
        if conn is None:
            if self.state_store is None:
                raise ValueError("Must provide either a sqlite3.Connection or a StateStore.")
            with self.state_store.uow() as uow:
                return self._extract_from_conn(uow.conn, target, score_family)
        return self._extract_from_conn(conn, target, score_family)

    def _extract_from_conn(
        self,
        conn: sqlite3.Connection,
        target: CalibrationTarget,
        score_family: ScoreFamily,
    ) -> List[CalibrationDatasetRecord]:
        cursor = conn.cursor()

        # Query all review events joined with outreach_drafts and prospects
        query = """
        SELECT
            e.draft_id,
            e.action,
            e.reviewed_revision,
            e.resulting_revision,
            e.reason_code,
            e.char_edit_distance,
            e.decided_at AS event_timestamp,
            ctx.prospect_id,
            p.company_name,
            ar.strategy_result_json
        FROM human_review_events e
        LEFT JOIN outreach_drafts d ON d.draft_id = e.draft_id
        LEFT JOIN outreach_contexts ctx ON ctx.context_id = d.context_id
        LEFT JOIN prospects p ON p.prospect_id = ctx.prospect_id
        LEFT JOIN application_runs ar ON ar.draft_id = e.draft_id
        ORDER BY e.draft_id, e.resulting_revision ASC, e.decided_at ASC;
        """
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Could not query human_review_events: %s", exc)
            return []

        # Group review events by draft_id
        events_by_draft: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if isinstance(row, sqlite3.Row):
                d_id = row["draft_id"]
                action = row["action"]
                rev_num = row["resulting_revision"] or row["reviewed_revision"] or 1
                reason = row["reason_code"]
                edit_dist = row["char_edit_distance"]
                ts = row["event_timestamp"]
                p_id = row["prospect_id"]
                c_name = row["company_name"]
                strat_json = row["strategy_result_json"]
            else:
                d_id = row[0]
                action = row[1]
                rev_num = row[3] or row[2] or 1
                reason = row[4]
                edit_dist = row[5]
                ts = row[6]
                p_id = row[7]
                c_name = row[8]
                strat_json = row[9]

            item = {
                "action": (action or "").strip().upper(),
                "revision_number": rev_num,
                "reason_code": reason,
                "char_edit_distance": edit_dist,
                "timestamp": ts,
                "prospect_id": p_id,
                "company_name": c_name,
                "strategy_result_json": strat_json,
            }
            events_by_draft.setdefault(d_id, []).append(item)

        records: List[CalibrationDatasetRecord] = []

        for draft_id, draft_events in events_by_draft.items():
            if not draft_events:
                continue

            # The final reviewed action determines the outcome
            last_event = draft_events[-1]
            final_action = last_event["action"]

            # Map final action to binary label
            if final_action == "APPROVED":
                label = 1
                outcome = "APPROVED"
            elif final_action == "REJECTED":
                label = 0
                outcome = "REJECTED"
            else:
                # Intermediate/pending revision (e.g. EDITED without subsequent approval or rejection)
                continue

            # Extract raw score from application run strategy JSON or fallback
            raw_score = self._extract_raw_score(last_event.get("strategy_result_json"), score_family)

            record = CalibrationDatasetRecord(
                draft_id=draft_id,
                revision_number=last_event["revision_number"],
                prospect_id=last_event["prospect_id"],
                company_name=last_event["company_name"],
                raw_score=raw_score,
                label=label,
                outcome=outcome,
                rejection_reason=last_event["reason_code"] if outcome == "REJECTED" else None,
                char_edit_distance=last_event["char_edit_distance"],
                timestamp=last_event["timestamp"],
                metadata={
                    "event_count": len(draft_events),
                    "initial_action": draft_events[0]["action"],
                },
            )
            records.append(record)

        return records

    @staticmethod
    def _extract_raw_score(strategy_json: Optional[str], score_family: ScoreFamily) -> float:
        """Extract continuous raw score from strategy execution snapshot JSON."""
        if not strategy_json:
            return 0.5  # Neutral default prior

        try:
            strat_data = json.loads(strategy_json)
            # 1. Top selected signal composite score
            if score_family == ScoreFamily.SIGNAL_COMPOSITE:
                selected_sig = strat_data.get("selected_signal") or {}
                score_obj = selected_sig.get("score") or {}
                if "composite_score" in score_obj:
                    return float(score_obj["composite_score"])

            # 2. Toulmin composite confidence
            toulmin = strat_data.get("toulmin_argument") or {}
            qual = toulmin.get("qualifier") or {}
            if "composite_confidence" in qual:
                return float(qual["composite_confidence"])

            # 3. Readability / draft acceptance composite
            if "overall_confidence" in strat_data:
                return float(strat_data["overall_confidence"])
        except Exception:
            pass

        return 0.5

    @staticmethod
    def check_dataset_readiness(
        records: Sequence[CalibrationDatasetRecord],
        gate_threshold: int = MINIMUM_ANNOTATED_DRAFTS_GATE,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Audit dataset against the mandatory 200-annotated-draft gate.

        Returns:
            Tuple of (is_ready, rationale, stats_dict).
        """
        n_total = len(records)
        n_pos = sum(1 for r in records if r.label == 1)
        n_neg = sum(1 for r in records if r.label == 0)

        prospects = {r.prospect_id for r in records if r.prospect_id}
        companies = {r.company_name for r in records if r.company_name}

        stats = {
            "total_annotated_drafts": n_total,
            "approved_drafts": n_pos,
            "rejected_drafts": n_neg,
            "distinct_prospects": len(prospects),
            "distinct_companies": len(companies),
            "gate_threshold": gate_threshold,
        }

        if n_total < gate_threshold:
            rationale = (
                f"Dataset contains {n_total} annotated drafts, which is below the mandatory "
                f"threshold of {gate_threshold}. As mandated by M4.1 policy, production calibration "
                f"models will NOT be fitted. Production scores remain explicitly uncalibrated."
            )
            return False, rationale, stats

        if n_pos == 0 or n_neg == 0:
            rationale = (
                f"Dataset contains {n_total} drafts but is single-class (approved={n_pos}, rejected={n_neg}). "
                "Calibration cannot proceed without examples of both classes."
            )
            return False, rationale, stats

        rationale = (
            f"Dataset satisfies readiness gate with {n_total} drafts across {len(companies)} companies "
            f"(approved={n_pos}, rejected={n_neg}). Calibration fitting is authorized."
        )
        return True, rationale, stats

    @staticmethod
    def split_grouped(
        records: Sequence[CalibrationDatasetRecord],
        eval_ratio: float = 0.20,
        group_by: SplitStrategy = SplitStrategy.PROSPECT_GROUPED,
        seed: int = 42,
    ) -> Tuple[List[CalibrationDatasetRecord], List[CalibrationDatasetRecord]]:
        """Partition records into train and evaluation sets without group leakage.

        Guarantees:
        - All revisions and drafts for a given prospect/company are assigned strictly
          to the train split OR the eval split, never split between the two.
        - Deterministic grouping using seed.
        """
        if not records:
            return [], []

        # 1. Group records by the chosen key
        grouped: Dict[str, List[CalibrationDatasetRecord]] = {}
        for rec in records:
            if group_by == SplitStrategy.PROSPECT_GROUPED:
                key = rec.prospect_id or rec.company_name or rec.draft_id
            elif group_by == SplitStrategy.COMPANY_GROUPED:
                key = rec.company_name or rec.prospect_id or rec.draft_id
            else:
                key = rec.draft_id
            grouped.setdefault(key, []).append(rec)

        # 2. Sort keys deterministically
        group_keys = sorted(list(grouped.keys()))
        rng = random.Random(seed)
        rng.shuffle(group_keys)

        total_records = len(records)
        target_eval_count = max(1, int(round(total_records * eval_ratio)))

        train_records: List[CalibrationDatasetRecord] = []
        eval_records: List[CalibrationDatasetRecord] = []
        current_eval_count = 0

        # Greedy allocation to achieve target eval sample count
        for key in group_keys:
            group_items = grouped[key]
            if current_eval_count + len(group_items) <= target_eval_count or (not eval_records and len(group_keys) > 1):
                eval_records.extend(group_items)
                current_eval_count += len(group_items)
            else:
                train_records.extend(group_items)

        # Edge case: if all items ended up in eval, swap one group back to train
        if not train_records and len(eval_records) > 1:
            first_group_key = group_keys[0]
            items_to_move = grouped[first_group_key]
            eval_records = [r for r in eval_records if r not in items_to_move]
            train_records.extend(items_to_move)

        return train_records, eval_records
