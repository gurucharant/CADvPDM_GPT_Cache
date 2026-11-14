import numpy as np
from typing import Optional

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService


class DenseSearch:
    """
    Dense semantic retrieval for CADvPDM GPT.

    Uses:
        Sentence Transformer embeddings
                    +
              PostgreSQL pgvector

    This retrieval method is good at understanding meaning,
    paraphrases and semantically similar historical incidents.

    Example:

        Query:
        "Batch login is failing because authentication is rejected"

        Dense retrieval may find:
        "CAD Batch job failed with unauthorized authentication error"

    even when the wording is different.
    """

    DEFAULT_SOURCES = [
        "SUPPORT",
        "JIRA",
        "CONFLUENCE",
        "LOG",
        "TEAMS",
        "GIT",
    ]

    def __init__(self):

        print("=" * 90)
        print("INITIALIZING CADvPDM DENSE SEARCH")
        print("=" * 90)

        self.embedding_service = EmbeddingService()

        print("Dense search ready.")

    # ==================================================================
    # GENERATE QUERY VECTOR
    # ==================================================================

    def _generate_query_vector(self, question: str):

        query_embedding = (
            self.embedding_service
            .generate_embedding(question)
        )

        query_vector = np.array(
            query_embedding,
            dtype=np.float32
        )

        return query_vector

    # ==================================================================
    # SEARCH ONE SOURCE
    # ==================================================================

    @staticmethod
    def _search_source(
        cursor,
        query_vector,
        source_type: str,
        top_k: int
    ):

        """
        Search one organizational-memory source independently.

        We first restrict PostgreSQL to one source using a
        MATERIALIZED CTE.

        pgvector then performs cosine-distance ranking only
        within that source.
        """

        sql = """
            WITH source_docs AS MATERIALIZED
            (
                SELECT
                    source_type,
                    source_id,
                    chunk_index,
                    title,
                    site,
                    application,
                    error_code,
                    content,
                    metadata,
                    embedding

                FROM cadvpdm_knowledge

                WHERE embedding IS NOT NULL
                  AND source_type = %s
            )

            SELECT
                source_type,
                source_id,
                chunk_index,
                title,
                site,
                application,
                error_code,
                content,
                metadata,

                1 - (
                    embedding <=> %s
                ) AS similarity

            FROM source_docs

            ORDER BY
                embedding <=> %s

            LIMIT %s;
        """

        cursor.execute(
            sql,
            (
                source_type,
                query_vector,
                query_vector,
                top_k
            )
        )

        return cursor.fetchall()

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        question: str,
        top_k: int = 20,
        source_types: Optional[list[str]] = None
    ) -> list[dict]:

        print("\n" + "=" * 90)
        print("DENSE VECTOR SEARCH")
        print("=" * 90)

        print(f"\nQuestion: {question}")

        # --------------------------------------------------------------
        # 1. Generate embedding for user question
        # --------------------------------------------------------------

        query_vector = self._generate_query_vector(
            question
        )

        print(
            f"Query embedding dimensions: "
            f"{len(query_vector)}"
        )

        # --------------------------------------------------------------
        # 2. Decide which organizational-memory sources to search
        # --------------------------------------------------------------

        active_sources = (
            source_types
            if source_types
            else self.DEFAULT_SOURCES
        )

        # --------------------------------------------------------------
        # 3. Connect to PostgreSQL
        # --------------------------------------------------------------

        conn = get_connection()
        cursor = conn.cursor()

        all_results = []

        try:

            # ----------------------------------------------------------
            # 4. Search every source independently
            # ----------------------------------------------------------

            for source in active_sources:

                print(
                    f"\nSearching source: {source}"
                )

                rows = self._search_source(
                    cursor=cursor,
                    query_vector=query_vector,
                    source_type=source,
                    top_k=top_k
                )

                print(
                    f"Retrieved {len(rows)} "
                    f"candidate(s) from {source}"
                )

                # ------------------------------------------------------
                # Convert tuple results into dictionaries
                # ------------------------------------------------------

                for row in rows:

                    (
                        source_type,
                        source_id,
                        chunk_index,
                        title,
                        site,
                        application,
                        error_code,
                        content,
                        metadata,
                        similarity
                    ) = row

                    result = {
                        "source_type": source_type,
                        "source_id": source_id,
                        "chunk_index": chunk_index,
                        "title": title,
                        "site": site,
                        "application": application,
                        "error_code": error_code,
                        "content": content,
                        "metadata": metadata,

                        # Dense-specific score
                        "similarity_score": float(
                            similarity
                        ),

                        "retrieval_method": "DENSE"
                    }

                    all_results.append(
                        result
                    )

        finally:

            cursor.close()
            conn.close()

        # --------------------------------------------------------------
        # 5. Globally sort candidates from ALL sources
        # --------------------------------------------------------------

        all_results.sort(
            key=lambda item:
                item["similarity_score"],
            reverse=True
        )

        # --------------------------------------------------------------
        # Important:
        #
        # _search_source() retrieved top_k PER SOURCE.
        #
        # Now take top_k globally.
        #
        # If top_k=20:
        #     up to 120 candidates may initially be retrieved
        #     from six sources, then globally reduced to 20.
        # --------------------------------------------------------------

        final_results = all_results[:top_k]

        self._display_results(
            final_results
        )

        return final_results

    # ==================================================================
    # DISPLAY
    # ==================================================================

    @staticmethod
    def _display_results(results):

        print("\n" + "=" * 90)
        print("DENSE SEARCH RESULTS")
        print("=" * 90)

        if not results:

            print("No dense search matches found.")
            return

        for rank, result in enumerate(
            results,
            start=1
        ):

            print("\n" + "#" * 90)

            print(
                f"RESULT #{rank} | "
                f"{result['source_type']}"
            )

            print("#" * 90)

            print(
                f"Source ID   : "
                f"{result['source_id']}"
            )

            print(
                f"Chunk       : "
                f"{result['chunk_index']}"
            )

            print(
                f"Title       : "
                f"{result['title']}"
            )

            print(
                f"Site        : "
                f"{result['site']}"
            )

            print(
                f"Application : "
                f"{result['application']}"
            )

            print(
                f"Error Code  : "
                f"{result['error_code']}"
            )

            print(
                f"Similarity  : "
                f"{result['similarity_score']:.4f}"
            )

            print(
                f"Method      : "
                f"{result['retrieval_method']}"
            )

            print("\nCONTENT")
            print("-" * 90)

            content = (
                result["content"]
                or ""
            )

            print(
                content[:1000]
            )


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    search_engine = DenseSearch()

    question = (
        "How was the CAD Batch AUTH-401 "
        "issue resolved previously?"
    )

    search_engine.search(
        question=question,
        top_k=10
    )