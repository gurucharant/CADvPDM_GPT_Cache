import json
import os
from datetime import timedelta

import numpy as np
from psycopg2.extras import Json

from database.connection import get_connection


class SemanticResponseCache:
    """Semantic response cache for stable organizational-knowledge answers.

    This cache is intentionally enabled only for SEMANTIC and KNOWLEDGE
    routes. ANALYTICS and MIXED are excluded because their numerical results
    may change whenever Iceberg data is refreshed.
    """

    CACHEABLE_ROUTES = {"SEMANTIC", "KNOWLEDGE"}

    def __init__(self, embedding_service, similarity_threshold=None, ttl_hours=None):
        self.embedding_service = embedding_service
        self.similarity_threshold = float(
            similarity_threshold
            if similarity_threshold is not None
            else os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.96")
        )
        self.ttl_hours = int(
            ttl_hours
            if ttl_hours is not None
            else os.getenv("SEMANTIC_CACHE_TTL_HOURS", "24")
        )
        self.knowledge_version = os.getenv("SEMANTIC_CACHE_KNOWLEDGE_VERSION", "v1")
        self.enabled = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in {
            "1", "true", "yes", "on"
        }
        if self.enabled:
            self._ensure_table()

    def _ensure_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS cadvpdm_semantic_cache (
            id BIGSERIAL PRIMARY KEY,
            route VARCHAR(20) NOT NULL,
            question TEXT NOT NULL,
            question_embedding vector(384) NOT NULL,
            answer TEXT NOT NULL,
            results JSONB NOT NULL DEFAULT '[]'::jsonb,
            hybrid_results JSONB NOT NULL DEFAULT '[]'::jsonb,
            semantic_query TEXT,
            knowledge_version VARCHAR(100) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_hit_at TIMESTAMPTZ,
            hit_count BIGINT NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_cadvpdm_semantic_cache_route_version_created
        ON cadvpdm_semantic_cache(route, knowledge_version, created_at DESC);
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _json_safe(value):
        # Round-trip through JSON so tuples/numpy scalars/etc. become JSON-safe.
        return json.loads(json.dumps(value, default=str))

    def _embedding(self, question):
        return np.asarray(
            self.embedding_service.generate_embedding(question),
            dtype=np.float32,
        )

    def lookup(self, question, route):
        if not self.enabled or route not in self.CACHEABLE_ROUTES:
            return None

        vector = self._embedding(question)
        sql = """
        SELECT
            id,
            question,
            answer,
            results,
            hybrid_results,
            semantic_query,
            1 - (question_embedding <=> %s) AS similarity
        FROM cadvpdm_semantic_cache
        WHERE route = %s
          AND knowledge_version = %s
          AND created_at >= NOW() - (%s * INTERVAL '1 hour')
        ORDER BY question_embedding <=> %s
        LIMIT 1;
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (vector, route, self.knowledge_version, self.ttl_hours, vector),
                )
                row = cur.fetchone()
                if not row:
                    return None
                similarity = float(row[6] or 0.0)
                if similarity < self.similarity_threshold:
                    return None
                cur.execute(
                    """
                    UPDATE cadvpdm_semantic_cache
                    SET hit_count = hit_count + 1, last_hit_at = NOW()
                    WHERE id = %s;
                    """,
                    (row[0],),
                )
            conn.commit()
            return {
                "answer": row[2],
                "results": row[3] or [],
                "hybrid_results": row[4] or [],
                "mode": route,
                "data": None,
                "chart": None,
                "sections": {},
                "semantic_query": row[5],
                "cache_hit": True,
                "cache_similarity": similarity,
                "cached_question": row[1],
            }
        finally:
            conn.close()

    def store(self, question, route, payload):
        if not self.enabled or route not in self.CACHEABLE_ROUTES:
            return

        vector = self._embedding(question)
        results = self._json_safe(payload.get("results", []))
        hybrid_results = self._json_safe(payload.get("hybrid_results", []))
        sql = """
        INSERT INTO cadvpdm_semantic_cache (
            route, question, question_embedding, answer,
            results, hybrid_results, semantic_query, knowledge_version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        route,
                        question,
                        vector,
                        payload.get("answer", ""),
                        Json(results),
                        Json(hybrid_results),
                        payload.get("semantic_query"),
                        self.knowledge_version,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
