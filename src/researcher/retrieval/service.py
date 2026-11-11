from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from src.db.schema import _ensure_fts_indexes, check_fts_integrity, rebuild_fts_indexes
from src.db.state_store import StateStore
from src.researcher.retrieval.models import (
    RetrievalResponse,
    RetrievalScope,
    RetrievedItem,
    RetrievedRecordType,
)
from src.researcher.retrieval.query_parser import FTSQueryParser

logger = logging.getLogger(__name__)


class HistoricalRetrievalService:
    """In-process lexical retrieval layer powered by SQLite FTS5."""

    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store

    def _ensure_initialized(self, conn: sqlite3.Connection) -> None:
        """Ensure SQLite FTS5 virtual tables and synchronization triggers exist, auto-backfilling if missing."""
        cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_snapshots_fts';")
        if not cur.fetchone():
            _ensure_fts_indexes(conn)

    def search(
        self,
        query: str,
        scope: Optional[RetrievalScope] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> RetrievalResponse:
        """Execute a lexical search against historical source snapshots and evidence records.

        Safe against SQL injection and FTS5 syntax errors via FTSQueryParser.
        """
        start_time = time.perf_counter()
        sanitized = FTSQueryParser.sanitize(query)

        if not sanitized:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResponse(
                query=query,
                sanitized_query="",
                total_matches=0,
                items=[],
                duration_ms=round(duration_ms, 3),
            )

        # Build subqueries according to requested record_type
        scope = scope or RetrievalScope()
        target_type = scope.record_type

        include_snapshots = target_type is None or target_type == RetrievedRecordType.SOURCE_SNAPSHOT
        include_passages = target_type is None or target_type == RetrievedRecordType.EVIDENCE_RECORD

        subqueries: List[str] = []
        count_subqueries: List[str] = []
        params: List[Any] = []
        count_params: List[Any] = []

        if include_snapshots:
            snap_sql, snap_count_sql, snap_params, snap_count_params = self._build_snapshot_query(sanitized, scope)
            subqueries.append(snap_sql)
            count_subqueries.append(snap_count_sql)
            params.extend(snap_params)
            count_params.extend(snap_count_params)

        if include_passages:
            pass_sql, pass_count_sql, pass_params, pass_count_params = self._build_passage_query(sanitized, scope)
            subqueries.append(pass_sql)
            count_subqueries.append(pass_count_sql)
            params.extend(pass_params)
            count_params.extend(pass_count_params)

        if not subqueries:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResponse(
                query=query,
                sanitized_query=sanitized,
                total_matches=0,
                items=[],
                duration_ms=round(duration_ms, 3),
            )

        # Unified query
        inner_sql = " UNION ALL ".join(subqueries)
        combined_sql = f"SELECT * FROM ({inner_sql}) ORDER BY bm25_rank ASC, rowid DESC LIMIT ? OFFSET ?"
        count_sql = f"SELECT COUNT(*) FROM ({' UNION ALL '.join(count_subqueries)})"

        params.append(max(1, limit))
        params.append(max(0, offset))

        items: List[RetrievedItem] = []
        total_matches = 0

        with self.state_store.uow() as uow:
            self._ensure_initialized(uow.conn)
            # 1. Total matches
            count_cur = uow.conn.execute(count_sql, count_params)
            count_row = count_cur.fetchone()
            total_matches = count_row[0] if count_row else 0

            # 2. Page of results
            cur = uow.conn.execute(combined_sql, params)
            rows = cur.fetchall()

            for row in rows:
                # row is a sqlite3.Row or tuple
                rec_type_str = row["record_type"] if isinstance(row, sqlite3.Row) else row[0]
                rec_id = row["record_id"] if isinstance(row, sqlite3.Row) else row[1]
                src_id = row["source_id"] if isinstance(row, sqlite3.Row) else row[2]
                sess_id = row["session_id"] if isinstance(row, sqlite3.Row) else row[3]
                run_id = row["run_id"] if isinstance(row, sqlite3.Row) else row[4]
                proj_id = row["project_id"] if isinstance(row, sqlite3.Row) else row[5]
                title = row["title"] if isinstance(row, sqlite3.Row) else row[6]
                snippet = row["snippet"] if isinstance(row, sqlite3.Row) else row[7]
                raw_text = row["raw_text"] if isinstance(row, sqlite3.Row) else row[8]
                bm25_val = float(row["bm25_rank"] if isinstance(row, sqlite3.Row) else row[9])
                captured_at = row["captured_at"] if isinstance(row, sqlite3.Row) else row[10]
                url = row["url"] if isinstance(row, sqlite3.Row) else row[11]
                domain = row["domain"] if isinstance(row, sqlite3.Row) else row[12]

                rec_type = (
                    RetrievedRecordType.SOURCE_SNAPSHOT
                    if rec_type_str == "source_snapshot"
                    else RetrievedRecordType.EVIDENCE_RECORD
                )
                # Preserve raw BM25 score without pretending abs() is a normalized probability
                score = bm25_val

                provenance = {
                    "source_id": src_id,
                    "session_id": sess_id,
                    "run_id": run_id,
                    "project_id": proj_id,
                    "captured_at": captured_at,
                    "url": url,
                    "domain": domain,
                }

                items.append(
                    RetrievedItem(
                        record_type=rec_type,
                        record_id=rec_id,
                        source_id=src_id,
                        session_id=sess_id,
                        run_id=run_id,
                        project_id=proj_id,
                        title=title,
                        snippet=snippet or "",
                        raw_text=raw_text or "",
                        rank=bm25_val,
                        score=score,
                        captured_at=captured_at,
                        url=url,
                        domain=domain,
                        provenance=provenance,
                    )
                )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalResponse(
            query=query,
            sanitized_query=sanitized,
            total_matches=total_matches,
            items=items,
            duration_ms=round(duration_ms, 3),
        )

    def _build_snapshot_query(
        self,
        sanitized_query: str,
        scope: RetrievalScope,
    ) -> Tuple[str, str, List[Any], List[Any]]:
        """Construct the SQL subquery for source snapshots with explicit run attribution."""
        select_params: List[Any] = []
        if scope.run_id:
            run_col = "? AS run_id"
            select_params.append(scope.run_id)
        else:
            run_col = "NULL AS run_id"

        sql = f"""
        SELECT
            'source_snapshot' AS record_type,
            s.snapshot_id AS record_id,
            s.source_id,
            s.session_id,
            {run_col},
            rs.prospect_id AS project_id,
            s.title,
            snippet(source_snapshots_fts, -1, '[[MATCH]]', '[[/MATCH]]', '...', 15) AS snippet,
            s.raw_text,
            bm25(source_snapshots_fts) AS bm25_rank,
            s.fetch_date AS captured_at,
            src.canonical_url AS url,
            src.domain,
            s.rowid
        FROM source_snapshots_fts fts
        JOIN source_snapshots s ON s.rowid = fts.rowid
        LEFT JOIN sources src ON src.source_id = s.source_id
        LEFT JOIN research_sessions rs ON rs.session_id = s.session_id
        WHERE source_snapshots_fts MATCH ?
        """
        count_sql = """
        SELECT s.rowid
        FROM source_snapshots_fts fts
        JOIN source_snapshots s ON s.rowid = fts.rowid
        LEFT JOIN sources src ON src.source_id = s.source_id
        LEFT JOIN research_sessions rs ON rs.session_id = s.session_id
        WHERE source_snapshots_fts MATCH ?
        """
        params: List[Any] = select_params + [sanitized_query]
        count_params: List[Any] = [sanitized_query]

        filter_clauses, filter_params = self._build_scope_filters(scope)
        if filter_clauses:
            clause_str = " AND " + " AND ".join(filter_clauses)
            sql += clause_str
            count_sql += clause_str
            params.extend(filter_params)
            count_params.extend(filter_params)

        return sql, count_sql, params, count_params

    def _build_passage_query(
        self,
        sanitized_query: str,
        scope: RetrievalScope,
    ) -> Tuple[str, str, List[Any], List[Any]]:
        """Construct the SQL subquery for evidence passages with explicit run attribution."""
        select_params: List[Any] = []
        if scope.run_id:
            run_col = "? AS run_id"
            select_params.append(scope.run_id)
        else:
            run_col = "NULL AS run_id"

        sql = f"""
        SELECT
            'evidence_record' AS record_type,
            ep.evidence_id AS record_id,
            ep.source_id,
            s.session_id,
            {run_col},
            rs.prospect_id AS project_id,
            s.title,
            snippet(evidence_passages_fts, 0, '[[MATCH]]', '[[/MATCH]]', '...', 15) AS snippet,
            ep.verbatim_excerpt AS raw_text,
            bm25(evidence_passages_fts) AS bm25_rank,
            ep.extracted_at AS captured_at,
            src.canonical_url AS url,
            src.domain,
            ep.rowid
        FROM evidence_passages_fts fts
        JOIN evidence_passages ep ON ep.rowid = fts.rowid
        LEFT JOIN source_snapshots s ON s.snapshot_id = ep.snapshot_id
        LEFT JOIN sources src ON src.source_id = ep.source_id
        LEFT JOIN research_sessions rs ON rs.session_id = s.session_id
        WHERE evidence_passages_fts MATCH ?
        """
        count_sql = """
        SELECT ep.rowid
        FROM evidence_passages_fts fts
        JOIN evidence_passages ep ON ep.rowid = fts.rowid
        LEFT JOIN source_snapshots s ON s.snapshot_id = ep.snapshot_id
        LEFT JOIN sources src ON src.source_id = ep.source_id
        LEFT JOIN research_sessions rs ON rs.session_id = s.session_id
        WHERE evidence_passages_fts MATCH ?
        """
        params: List[Any] = select_params + [sanitized_query]
        count_params: List[Any] = [sanitized_query]

        filter_clauses, filter_params = self._build_scope_filters(scope)
        if filter_clauses:
            clause_str = " AND " + " AND ".join(filter_clauses)
            sql += clause_str
            count_sql += clause_str
            params.extend(filter_params)
            count_params.extend(filter_params)

        return sql, count_sql, params, count_params

    def _build_scope_filters(self, scope: RetrievalScope) -> Tuple[List[str], List[Any]]:
        """Build WHERE clauses and parameters from a RetrievalScope."""
        clauses: List[str] = []
        params: List[Any] = []

        if scope.session_id:
            clauses.append("s.session_id = ?")
            params.append(scope.session_id)

        target_project = scope.project_id or scope.prospect_id
        if target_project:
            clauses.append("(rs.prospect_id = ? OR rs.context_id = ?)")
            params.append(target_project)
            params.append(target_project)

        if scope.run_id:
            clauses.append(
                "s.session_id IN (SELECT research_session_id FROM application_runs WHERE run_id = ?)"
            )
            params.append(scope.run_id)

        if scope.domain:
            clauses.append("src.domain = ?")
            params.append(scope.domain)

        return clauses, params

    def rebuild_indexes(self) -> None:
        """Administrative rebuild of all FTS5 full-text indexes."""
        with self.state_store.uow() as uow:
            self._ensure_initialized(uow.conn)
            rebuild_fts_indexes(uow.conn)

    def verify_integrity(self) -> Dict[str, bool]:
        """Validate integrity of all FTS5 virtual tables."""
        with self.state_store.uow() as uow:
            self._ensure_initialized(uow.conn)
            return check_fts_integrity(uow.conn)
