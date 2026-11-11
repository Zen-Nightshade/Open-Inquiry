from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.schema import check_fts_integrity, rebuild_fts_indexes
from src.db.state_store import StateStore
from src.researcher.retrieval import HistoricalRetrievalService, RetrievalScope, RetrievedRecordType

random.seed(42)

TOPICS = [
    ("robotics", "autonomous navigation", "warehouse robots with computer vision and lidar"),
    ("machine learning", "deep neural networks", "distributed cluster training across GPUs"),
    ("cloud infrastructure", "kubernetes clusters", "zero-downtime microservice orchestration"),
    ("cybersecurity", "zero-trust architecture", "identity federation and credential rotation"),
    ("biotech", "genomic sequencing", "crispr targeted gene editing in clinical trials"),
    ("fintech", "algorithmic trading", "low-latency high-frequency quantitative market making"),
    ("energy", "renewable grid storage", "lithium iron phosphate battery energy storage systems"),
]

WORDS = [
    "scalability", "latency", "throughput", "reliability", "infrastructure",
    "deployment", "optimization", "pipeline", "architecture", "telemetry",
    "automation", "resilience", "observability", "distributed", "verification",
]


def generate_synthetic_data(num_records: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    snapshots = []
    passages = []
    for i in range(num_records):
        topic, phrase, desc = random.choice(TOPICS)
        extra_words = " ".join(random.sample(WORDS, 5))
        title = f"{topic.title()} Innovation Report #{i}: {phrase.title()}"
        raw_text = f"Company #{i % 20} announced that {desc}. Key architectural benefits include {extra_words}."
        snap_id = f"snap_{i}"
        src_id = f"src_{i}"
        sess_id = f"sess_{i % 10}"
        proj_id = f"proj_{i % 5}"

        snapshots.append({
            "snapshot_id": snap_id,
            "source_id": src_id,
            "session_id": sess_id,
            "project_id": proj_id,
            "url": f"https://example{i % 20}.com/page_{i}",
            "domain": f"example{i % 20}.com",
            "title": title,
            "raw_text": raw_text,
            "fetch_date": "2026-11-10",
        })

        passages.append({
            "evidence_id": f"ev_{i}",
            "snapshot_id": snap_id,
            "source_id": src_id,
            "verbatim_excerpt": f"{phrase} achieved {extra_words}",
            "extracted_at": "2026-11-10",
        })
    return snapshots, passages


def run_benchmark_for_size(num_records: int) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = StateStore(db_path)
        store.initialize()

        snapshots, passages = generate_synthetic_data(num_records)

        # 1. Measure insert & index build time
        t0 = time.perf_counter()
        with store.uow() as uow:
            for p_id in {s["project_id"] for s in snapshots}:
                uow.conn.execute(
                    "INSERT OR IGNORE INTO prospects (prospect_id, full_name, company_name, created_at) VALUES (?, ?, ?, '2026-11-10');",
                    (p_id, f"User {p_id}", f"Company {p_id}"),
                )
            for s in snapshots:
                uow.conn.execute(
                    "INSERT OR IGNORE INTO research_sessions (session_id, prospect_id, context_id, started_at) VALUES (?, ?, 'ctx', '2026-11-10');",
                    (s["session_id"], s["project_id"]),
                )
                uow.conn.execute(
                    "INSERT OR IGNORE INTO sources (source_id, canonical_url, domain, first_seen_at) VALUES (?, ?, ?, '2026-11-10');",
                    (s["source_id"], s["url"], s["domain"]),
                )
                uow.conn.execute(
                    "INSERT INTO source_snapshots (snapshot_id, source_id, session_id, title, raw_text, fetch_status, fetch_date) "
                    "VALUES (?, ?, ?, ?, ?, 'success', ?);",
                    (s["snapshot_id"], s["source_id"], s["session_id"], s["title"], s["raw_text"], s["fetch_date"]),
                )
            for p in passages:
                uow.conn.execute(
                    "INSERT INTO evidence_passages (evidence_id, snapshot_id, source_id, verbatim_excerpt, extracted_at) "
                    "VALUES (?, ?, ?, ?, ?);",
                    (p["evidence_id"], p["snapshot_id"], p["source_id"], p["verbatim_excerpt"], p["extracted_at"]),
                )
        build_time_s = time.perf_counter() - t0

        # 2. Database size
        db_size_bytes = os.path.getsize(db_path)
        db_size_kb = db_size_bytes / 1024.0

        # 3. Administrative Rebuild time
        t0_rebuild = time.perf_counter()
        with store.uow() as uow:
            rebuild_fts_indexes(uow.conn)
        rebuild_time_s = time.perf_counter() - t0_rebuild

        # 4. Integrity check
        with store.uow() as uow:
            integrity = check_fts_integrity(uow.conn)
        assert integrity["source_snapshots_fts"] is True
        assert integrity["evidence_passages_fts"] is True

        service = HistoricalRetrievalService(store)

        # 5. Cold-start query latency
        t0_cold = time.perf_counter()
        service.search("autonomous navigation")
        cold_latency_ms = (time.perf_counter() - t0_cold) * 1000.0

        # 6. Warm query latency across multiple queries
        queries = [
            "robotics",
            '"deep neural networks"',
            "kubernetes clusters optimization",
            "crispr targeted gene editing",
            "algorithmic trading market making",
            "battery energy storage",
            "zero-trust architecture",
            "observability pipeline",
        ]

        warm_latencies_ms = []
        for _ in range(5):
            for q in queries:
                t0_q = time.perf_counter()
                service.search(q, limit=10)
                dur = (time.perf_counter() - t0_q) * 1000.0
                warm_latencies_ms.append(dur)

        p50 = float(np.percentile(warm_latencies_ms, 50))
        p90 = float(np.percentile(warm_latencies_ms, 90))
        p95 = float(np.percentile(warm_latencies_ms, 95))

        # 7. Scoped vs Unscoped query latency
        unscoped_times = []
        scoped_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            service.search("robotics", limit=10)
            unscoped_times.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            service.search("robotics", scope=RetrievalScope(project_id="proj_0"), limit=10)
            scoped_times.append((time.perf_counter() - t0) * 1000.0)

        unscoped_p50 = float(np.percentile(unscoped_times, 50))
        scoped_p50 = float(np.percentile(scoped_times, 50))

        return {
            "num_records": num_records,
            "build_time_s": round(build_time_s, 3),
            "rebuild_time_s": round(rebuild_time_s, 3),
            "db_size_kb": round(db_size_kb, 1),
            "cold_latency_ms": round(cold_latency_ms, 2),
            "warm_p50_ms": round(p50, 2),
            "warm_p90_ms": round(p90, 2),
            "warm_p95_ms": round(p95, 2),
            "unscoped_p50_ms": round(unscoped_p50, 2),
            "scoped_p50_ms": round(scoped_p50, 2),
        }
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def main() -> None:
    print("=" * 80)
    print("Open Inquiry — SQLite FTS5 Full-Text Search Performance Benchmark (M3.3)")
    print("=" * 80)
    print()

    sizes = [100, 1000, 5000]
    results = []

    for size in sizes:
        print(f"Running benchmark with corpus size = {size:,} records...")
        res = run_benchmark_for_size(size)
        results.append(res)
        print(f"  ✓ Build: {res['build_time_s']}s | Rebuild: {res['rebuild_time_s']}s | DB: {res['db_size_kb']} KB")
        print(f"  ✓ Cold: {res['cold_latency_ms']} ms | Warm p50: {res['warm_p50_ms']} ms | p95: {res['warm_p95_ms']} ms")
        print(f"  ✓ Scoped p50: {res['scoped_p50_ms']} ms vs Unscoped p50: {res['unscoped_p50_ms']} ms")
        print()

    print("=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    header = f"{'Corpus Size':<12} | {'Build (s)':<10} | {'Rebuild (s)':<12} | {'DB Size (KB)':<12} | {'Cold (ms)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Scoped (ms)':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['num_records']:<12} | {r['build_time_s']:<10.3f} | {r['rebuild_time_s']:<12.3f} | {r['db_size_kb']:<12.1f} | {r['cold_latency_ms']:<10.2f} | {r['warm_p50_ms']:<10.2f} | {r['warm_p95_ms']:<10.2f} | {r['scoped_p50_ms']:<12.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
