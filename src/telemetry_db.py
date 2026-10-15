"""SQLite Telemetry & Semantic Cache Store for RouteOps.

Provides thread-safe asynchronous and synchronous logging for inference telemetry,
and high-performance cosine similarity lookups for the semantic vector cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("routeops.telemetry_db")

DEFAULT_DB_PATH = Path("data/routeops_telemetry.db")


class TelemetryDB:
    """Thread-safe SQLite storage for RouteOps telemetry logs and semantic caching."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Initialize the database connection and schema.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_key = (
            os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("SUPABASE_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
        )
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with optimized PRAGMA settings."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_schema(self) -> None:
        """Create inference_logs and semantic_cache tables if they do not exist."""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    request_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    query_text TEXT NOT NULL,
                    detected_lang TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entropy REAL NOT NULL,
                    routed_tier TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    naive_frontier_cost_usd REAL NOT NULL,
                    cost_saved_usd REAL NOT NULL
                );
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp
                ON inference_logs(timestamp);
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_routed_tier
                ON inference_logs(routed_tier);
                """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 1,
                    created_at REAL NOT NULL
                );
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_class
                ON semantic_cache(predicted_class);
                """)
            conn.commit()
            logger.info("Initialized telemetry database schema at %s", self.db_path)

    # -------------------------------------------------------------------------
    # Inference Logging
    # -------------------------------------------------------------------------

    def log_inference(
        self,
        request_id: str,
        query_text: str,
        detected_lang: str,
        predicted_class: str,
        confidence: float,
        entropy: float,
        routed_tier: str,
        latency_ms: float,
        tokens_used: int,
        estimated_cost_usd: float,
        naive_frontier_cost_usd: float,
        cost_saved_usd: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Log a routed query and its economic impact to the database."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO inference_logs (
                    request_id, timestamp, query_text, detected_lang,
                    predicted_class, confidence, entropy, routed_tier,
                    latency_ms, tokens_used, estimated_cost_usd,
                    naive_frontier_cost_usd, cost_saved_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    request_id,
                    ts,
                    query_text,
                    detected_lang,
                    predicted_class,
                    confidence,
                    entropy,
                    routed_tier,
                    latency_ms,
                    tokens_used,
                    estimated_cost_usd,
                    naive_frontier_cost_usd,
                    cost_saved_usd,
                ),
            )
            conn.commit()

        # Asynchronous Supabase REST push (if configured)
        if self.supabase_url and self.supabase_key:
            self._async_log_supabase(
                {
                    "request_id": request_id,
                    "created_at": ts,
                    "query_text": query_text,
                    "detected_lang": detected_lang,
                    "predicted_class": predicted_class,
                    "routed_class": predicted_class,
                    "routed_tier": routed_tier,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "cost_saved_usd": cost_saved_usd,
                }
            )

    def _async_log_supabase(self, payload: Dict[str, Any]) -> None:
        """Non-blocking background push to Supabase REST endpoint."""

        def _post() -> None:
            try:
                import urllib.request

                url = f"{self.supabase_url}/rest/v1/routeops_telemetry_logs"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Prefer": "return=minimal",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status not in (200, 201):
                        logger.debug(
                            "Supabase telemetry logged status: %s", resp.status
                        )
            except Exception as e:
                logger.debug("Supabase background telemetry skipped (%s)", e)

        t = threading.Thread(target=_post, daemon=True)
        t.start()

    # -------------------------------------------------------------------------
    # Semantic Vector Cache Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_query_hash(query_text: str) -> str:
        """Compute normalized SHA-256 hash for exact-match fast path."""
        norm_text = " ".join(query_text.strip().lower().split())
        return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

    def lookup_cache(
        self,
        query_embedding: np.ndarray,
        predicted_class: str,
        query_text: Optional[str] = None,
        similarity_threshold: float = 0.92,
    ) -> Optional[Tuple[str, str, float]]:
        """Search the cache for semantic hits within the same predicted class.

        Args:
            query_embedding: L2-normalized dense vector for the incoming query.
            predicted_class: Routing class constraint.
            query_text: Optional raw text for exact-hash acceleration.
            similarity_threshold: Minimum cosine similarity required (default: 0.92).

        Returns:
            Tuple of (response_text, cached_query_text, similarity_score) if hit, else None.
        """
        # Fast-Path: Check exact hash match
        if query_text:
            q_hash = self.compute_query_hash(query_text)
            with self._lock, self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT query_hash, query_text, response_text
                    FROM semantic_cache
                    WHERE query_hash = ? AND predicted_class = ?;
                    """,
                    (q_hash, predicted_class),
                )
                row = cur.fetchone()
                if row is not None:
                    conn.execute(
                        """
                        UPDATE semantic_cache
                        SET hit_count = hit_count + 1
                        WHERE query_hash = ?;
                        """,
                        (q_hash,),
                    )
                    conn.commit()
                    return row["response_text"], row["query_text"], 1.0

        # Vector Scan Path: Compare against candidates in the same class
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT query_hash, query_text, embedding_json, response_text
                FROM semantic_cache
                WHERE predicted_class = ?;
                """,
                (predicted_class,),
            )
            rows = cur.fetchall()

        if not rows:
            return None

        best_sim = -1.0
        best_row: Optional[sqlite3.Row] = None

        # Ensure query vector is 1D float32
        q_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        norm_q = np.linalg.norm(q_vec)
        if norm_q > 1e-8:
            q_vec = q_vec / norm_q

        for row in rows:
            try:
                c_vec = np.asarray(
                    json.loads(row["embedding_json"]), dtype=np.float32
                ).flatten()
                norm_c = np.linalg.norm(c_vec)
                if norm_c > 1e-8:
                    c_vec = c_vec / norm_c
                sim = float(np.dot(q_vec, c_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_row = row
            except Exception as err:
                logger.debug("Failed to decode embedding: %s", err)
                continue

        if best_row is not None and best_sim >= similarity_threshold:
            # Increment hit counter
            with self._lock, self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE semantic_cache
                    SET hit_count = hit_count + 1
                    WHERE query_hash = ?;
                    """,
                    (best_row["query_hash"],),
                )
                conn.commit()
            return best_row["response_text"], best_row["query_text"], float(best_sim)

        return None

    def insert_cache(
        self,
        query_text: str,
        predicted_class: str,
        embedding: np.ndarray,
        response_text: str,
    ) -> None:
        """Store a verified query-response pair in the semantic cache.

        Args:
            query_text: User query string.
            predicted_class: Canonical intent category.
            embedding: Dense vector representation (will be normalized).
            response_text: Generated/retrieved response content.
        """
        q_hash = self.compute_query_hash(query_text)
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        embedding_json = json.dumps(vec.tolist())
        created_at = time.time()

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic_cache (
                    query_hash, query_text, predicted_class,
                    embedding_json, response_text, hit_count, created_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    response_text = excluded.response_text,
                    hit_count = hit_count + 1;
                """,
                (
                    q_hash,
                    query_text,
                    predicted_class,
                    embedding_json,
                    response_text,
                    created_at,
                ),
            )
            conn.commit()

    # -------------------------------------------------------------------------
    # Telemetry Analytics & Reporting
    # -------------------------------------------------------------------------

    def get_recent_logs(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve recent inference log records."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM inference_logs
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate system metrics from inference logs and cache."""
        with self._lock, self._get_connection() as conn:
            cur_logs = conn.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    TOTAL(estimated_cost_usd) as total_cost_usd,
                    TOTAL(naive_frontier_cost_usd) as total_naive_cost_usd,
                    TOTAL(cost_saved_usd) as total_saved_usd,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(confidence) as avg_confidence,
                    AVG(entropy) as avg_entropy,
                    TOTAL(tokens_used) as total_tokens
                FROM inference_logs;
                """)
            log_stats = dict(cur_logs.fetchone())

            cur_tiers = conn.execute("""
                SELECT routed_tier, COUNT(*) as count
                FROM inference_logs
                GROUP BY routed_tier;
                """)
            tier_counts = {
                row["routed_tier"]: row["count"] for row in cur_tiers.fetchall()
            }

            cur_cache = conn.execute("""
                SELECT COUNT(*) as cached_items, TOTAL(hit_count) as total_cache_hits
                FROM semantic_cache;
                """)
            cache_stats = dict(cur_cache.fetchone())

        total_reqs = log_stats["total_requests"] or 0
        total_naive = log_stats["total_naive_cost_usd"] or 0.0
        total_saved = log_stats["total_saved_usd"] or 0.0
        cost_reduction_pct = (
            (total_saved / total_naive * 100.0) if total_naive > 0 else 0.0
        )

        return {
            "total_requests": total_reqs,
            "total_cost_usd": round(float(log_stats["total_cost_usd"] or 0.0), 6),
            "total_naive_cost_usd": round(float(total_naive), 6),
            "total_saved_usd": round(float(total_saved), 6),
            "cost_reduction_pct": round(float(cost_reduction_pct), 2),
            "avg_latency_ms": round(float(log_stats["avg_latency_ms"] or 0.0), 2),
            "avg_confidence": round(float(log_stats["avg_confidence"] or 0.0), 4),
            "avg_entropy": round(float(log_stats["avg_entropy"] or 0.0), 4),
            "total_tokens": int(log_stats["total_tokens"] or 0),
            "tier_counts": tier_counts,
            "cache_entries": int(cache_stats["cached_items"] or 0),
            "total_cache_hits": int(cache_stats["total_cache_hits"] or 0),
        }

    def clear_all(self) -> None:
        """Clear all tables (primarily for test fixtures)."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM inference_logs;")
            conn.execute("DELETE FROM semantic_cache;")
            conn.commit()
