from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


class SourceRepository:
    """Repository managing canonical web sources, fetch snapshots, evidence passages, and syndication groups."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    # Sources
    def upsert_source(
        self,
        source_id: str,
        canonical_url: str,
        domain: str,
        source_type: str = "other",
        credibility_score: float = 0.7,
        credibility_rationale: Optional[str] = None,
        first_seen_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = first_seen_at or datetime.now(timezone.utc).isoformat()

        # 1. Check if source already exists by source_id
        existing_by_id = self.get_source(source_id)
        if existing_by_id:
            self.conn.execute(
                """
                UPDATE sources SET
                    domain = ?,
                    source_type = ?,
                    credibility_score = ?,
                    credibility_rationale = COALESCE(?, credibility_rationale)
                WHERE source_id = ?;
                """,
                (domain, source_type, credibility_score, credibility_rationale, source_id),
            )
            return self.get_source(source_id) or {}

        # 2. Check if source already exists by canonical_url
        existing_by_url = self.get_source_by_url(canonical_url)
        if existing_by_url:
            self.conn.execute(
                """
                UPDATE sources SET
                    domain = ?,
                    source_type = ?,
                    credibility_score = ?,
                    credibility_rationale = COALESCE(?, credibility_rationale)
                WHERE canonical_url = ?;
                """,
                (domain, source_type, credibility_score, credibility_rationale, canonical_url),
            )
            return self.get_source_by_url(canonical_url) or {}

        # 3. New source insertion
        sql = """
        INSERT INTO sources (
            source_id, canonical_url, domain, source_type,
            credibility_score, credibility_rationale, first_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                source_id,
                canonical_url,
                domain,
                source_type,
                credibility_score,
                credibility_rationale,
                now,
            ),
        )
        return self.get_source_by_url(canonical_url) or {}

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM sources WHERE source_id = ?;",
            (source_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_source_by_url(self, canonical_url: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM sources WHERE canonical_url = ?;",
            (canonical_url,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_sources(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        if domain:
            cursor = self.conn.execute(
                "SELECT * FROM sources WHERE domain = ? ORDER BY first_seen_at DESC;",
                (domain,),
            )
        else:
            cursor = self.conn.execute("SELECT * FROM sources ORDER BY first_seen_at DESC;")
        return [dict(r) for r in cursor.fetchall()]

    # Snapshots
    def create_snapshot(
        self,
        snapshot_id: str,
        source_id: str,
        fetch_status: str,
        fetch_date: Optional[str] = None,
        session_id: Optional[str] = None,
        original_url: Optional[str] = None,
        final_url: Optional[str] = None,
        html_canonical_url: Optional[str] = None,
        title: Optional[str] = None,
        raw_text: Optional[str] = None,
        raw_content_ref: Optional[str] = None,
        http_status_code: Optional[int] = None,
        publication_date: Optional[str] = None,
        publication_date_src: Optional[str] = None,
        duration_ms: Optional[float] = None,
        raw_content_hash: Optional[str] = None,
        normalized_content_hash: Optional[str] = None,
        minhash_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = fetch_date or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO source_snapshots (
            snapshot_id, source_id, session_id, original_url, final_url,
            html_canonical_url, title, raw_text, raw_content_ref, fetch_status,
            http_status_code, publication_date, publication_date_src, fetch_date, duration_ms,
            raw_content_hash, normalized_content_hash, minhash_signature
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(
            sql,
            (
                snapshot_id,
                source_id,
                session_id,
                original_url,
                final_url,
                html_canonical_url,
                title,
                raw_text,
                raw_content_ref,
                fetch_status,
                http_status_code,
                publication_date,
                publication_date_src,
                now,
                duration_ms,
                raw_content_hash,
                normalized_content_hash,
                minhash_signature,
            ),
        )
        return self.get_snapshot(snapshot_id) or {}

    def record_snapshot_fingerprint(
        self,
        snapshot_id: str,
        raw_content_hash: str,
        normalized_content_hash: str,
        minhash_signature: Optional[str] = None,
    ) -> bool:
        sql = """
        UPDATE source_snapshots
        SET raw_content_hash = ?,
            normalized_content_hash = ?,
            minhash_signature = ?
        WHERE snapshot_id = ?;
        """
        cursor = self.conn.execute(
            sql, (raw_content_hash, normalized_content_hash, minhash_signature, snapshot_id)
        )
        return cursor.rowcount > 0

    def get_snapshot_by_raw_hash(self, raw_hash: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM source_snapshots WHERE raw_content_hash = ? ORDER BY fetch_date DESC LIMIT 1;",
            (raw_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_snapshot_by_normalized_hash(self, normalized_hash: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM source_snapshots WHERE normalized_content_hash = ? ORDER BY fetch_date DESC LIMIT 1;",
            (normalized_hash,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM source_snapshots WHERE snapshot_id = ?;",
            (snapshot_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_snapshots_for_source(self, source_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM source_snapshots WHERE source_id = ? ORDER BY fetch_date DESC;",
            (source_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_snapshots_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM source_snapshots WHERE session_id = ? ORDER BY fetch_date DESC;",
            (session_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Evidence passages
    def create_evidence_passage(
        self,
        evidence_id: str,
        snapshot_id: str,
        source_id: str,
        verbatim_excerpt: str,
        extracted_at: Optional[str] = None,
        excerpt_validated: int = 0,
        event_date: Optional[str] = None,
        confidence: str = "medium",
        credibility_score: float = 0.7,
    ) -> Dict[str, Any]:
        now = extracted_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO evidence_passages (
            evidence_id, snapshot_id, source_id, verbatim_excerpt,
            excerpt_validated, event_date, extracted_at, confidence, credibility_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evidence_id) DO UPDATE SET
            excerpt_validated = excluded.excerpt_validated,
            confidence = excluded.confidence,
            credibility_score = excluded.credibility_score;
        """
        self.conn.execute(
            sql,
            (
                evidence_id,
                snapshot_id,
                source_id,
                verbatim_excerpt,
                excerpt_validated,
                event_date,
                now,
                confidence,
                credibility_score,
            ),
        )
        cursor = self.conn.execute(
            "SELECT * FROM evidence_passages WHERE evidence_id = ?;",
            (evidence_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_evidence_passage(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM evidence_passages WHERE evidence_id = ?;",
            (evidence_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_evidence_for_source(self, source_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM evidence_passages WHERE source_id = ? ORDER BY extracted_at DESC;",
            (source_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_evidence_for_snapshot(self, snapshot_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM evidence_passages WHERE snapshot_id = ? ORDER BY extracted_at DESC;",
            (snapshot_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

    # Syndication origin groups
    def create_origin_group(
        self,
        group_id: str,
        canonical_source_id: str,
        description: Optional[str] = None,
        created_at: Optional[str] = None,
        dependence_type: Optional[str] = None,
        primary_detection_method: Optional[str] = None,
        average_similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO evidence_origin_groups (
            group_id, canonical_source_id, description, created_at,
            dependence_type, primary_detection_method, average_similarity
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(group_id) DO UPDATE SET
            canonical_source_id = excluded.canonical_source_id,
            description = coalesce(excluded.description, evidence_origin_groups.description),
            dependence_type = coalesce(excluded.dependence_type, evidence_origin_groups.dependence_type),
            primary_detection_method = coalesce(excluded.primary_detection_method, evidence_origin_groups.primary_detection_method),
            average_similarity = coalesce(excluded.average_similarity, evidence_origin_groups.average_similarity);
        """
        self.conn.execute(
            sql,
            (
                group_id,
                canonical_source_id,
                description,
                now,
                dependence_type,
                primary_detection_method,
                average_similarity,
            ),
        )
        return self.get_origin_group(group_id) or {}

    def get_origin_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM evidence_origin_groups WHERE group_id = ?;",
            (group_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def add_origin_member(self, group_id: str, source_id: str) -> bool:
        sql = """
        INSERT INTO evidence_origin_members (group_id, source_id)
        VALUES (?, ?)
        ON CONFLICT(group_id, source_id) DO NOTHING;
        """
        cursor = self.conn.execute(sql, (group_id, source_id))
        return cursor.rowcount > 0

    def get_origin_members(self, group_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT s.* FROM sources s
        JOIN evidence_origin_members m ON s.source_id = m.source_id
        WHERE m.group_id = ?;
        """
        cursor = self.conn.execute(sql, (group_id,))
        return [dict(r) for r in cursor.fetchall()]

    def save_syndication_cluster(self, cluster: Any) -> Dict[str, Any]:
        """Persist a SyndicationCluster object or dict to evidence_origin_groups and evidence_origin_members."""
        if hasattr(cluster, "cluster_id"):
            c_id = cluster.cluster_id
            can_id = cluster.canonical_source_id
            desc = cluster.description
            c_at = cluster.created_at
            dep_type = (
                cluster.dependence_type.value
                if hasattr(cluster.dependence_type, "value")
                else str(cluster.dependence_type)
            )
            det_method = (
                cluster.primary_detection_method.value
                if hasattr(cluster.primary_detection_method, "value")
                else str(cluster.primary_detection_method)
            )
            avg_sim = cluster.average_similarity
            members = cluster.member_source_ids
        else:
            c_id = cluster["cluster_id"]
            can_id = cluster["canonical_source_id"]
            desc = cluster.get("description")
            c_at = cluster.get("created_at")
            dep_type = cluster.get("dependence_type")
            det_method = cluster.get("primary_detection_method")
            avg_sim = cluster.get("average_similarity", 1.0)
            members = cluster.get("member_source_ids", set())

        self.create_origin_group(
            group_id=c_id,
            canonical_source_id=can_id,
            description=desc,
            created_at=c_at,
            dependence_type=dep_type,
            primary_detection_method=det_method,
            average_similarity=avg_sim,
        )

        for s_id in members:
            self.add_origin_member(c_id, s_id)

        return self.get_origin_group(c_id) or {}

    def get_syndication_groups(self) -> Dict[str, Set[str]]:
        """Return all syndication origin groups as a mapping from group_id to a set of member source_ids."""
        cursor = self.conn.execute("SELECT group_id, source_id FROM evidence_origin_members ORDER BY group_id;")
        groups: Dict[str, Set[str]] = {}
        for row in cursor.fetchall():
            gid = row["group_id"]
            sid = row["source_id"]
            groups.setdefault(gid, set()).add(sid)
        return groups
