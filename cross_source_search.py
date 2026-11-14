import numpy as np

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService


class CrossSourceSearch:

    def __init__(self):

        print("Initializing CADvPDM Cross-Source Search...")

        self.embedding_service = EmbeddingService()

        self.sources = [
            "SUPPORT",
            "JIRA",
            "CONFLUENCE",
            "LOG",
            "TEAMS",
            "GIT"
        ]

    def search_source(
            self,
            cursor,
            query_vector,
            source_type,
            top_k=3
    ):
        """
        Search one source independently.

        The MATERIALIZED CTE first filters the database
        to one source such as JIRA, SUPPORT, CONFLUENCE
        or LOG.

        Vector similarity is then calculated only within
        that source.
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

    def search(
            self,
            question,
            results_per_source=3,
            source_types=None
    ):

        print("\n" + "=" * 90)
        print("CADvPDM QUESTION")
        print("=" * 90)

        print(question)

        # --------------------------------------------------
        # 1. Convert question into embedding
        # --------------------------------------------------

        query_embedding = (
            self.embedding_service
            .generate_embedding(question)
        )

        query_vector = np.array(
            query_embedding,
            dtype=np.float32
        )

        # --------------------------------------------------
        # 2. Connect to PostgreSQL
        # --------------------------------------------------

        conn = get_connection()
        cursor = conn.cursor()

        all_results = []

        try:

            # --------------------------------------------------
            # 3. Search each knowledge source independently
            # --------------------------------------------------

            active_sources = source_types or self.sources

            for source in active_sources:

                print(
                    f"\nSearching source: {source}"
                )

                results = self.search_source(
                    cursor=cursor,
                    query_vector=query_vector,
                    source_type=source,
                    top_k=results_per_source
                )

                print(
                    f"Retrieved {len(results)} "
                    f"results from {source}"
                )

                all_results.extend(
                    results
                )

        finally:

            cursor.close()
            conn.close()

        # --------------------------------------------------
        # 4. Sort combined evidence by similarity
        # --------------------------------------------------

        all_results.sort(
            key=lambda row: float(row[9]),
            reverse=True
        )

        # --------------------------------------------------
        # 5. Display results
        # --------------------------------------------------

        print("\n")
        print("=" * 90)
        print("CROSS-SOURCE ORGANIZATIONAL MEMORY")
        print("=" * 90)

        for index, row in enumerate(
                all_results,
                start=1
        ):

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

            print("\n")
            print("#" * 90)

            print(
                f"RESULT #{index} | "
                f"{source_type}"
            )

            print("#" * 90)

            print(
                f"Source ID   : {source_id}"
            )

            print(
                f"Chunk       : {chunk_index}"
            )

            print(
                f"Title       : {title}"
            )

            print(
                f"Site        : {site}"
            )

            print(
                f"Application : {application}"
            )

            print(
                f"Error Code  : {error_code}"
            )

            print(
                f"Similarity  : "
                f"{float(similarity):.4f}"
            )

            print("\nCONTENT")
            print("-" * 90)

            print(
                content[:900]
            )

        return all_results


if __name__ == "__main__":

    search_engine = CrossSourceSearch()

    question = """
    A CAD Batch job failed during login with an unauthorized authentication error.
    Have we experienced this before, what caused it, and how was it resolved?
    """

    search_engine.search(
        question=question,
        results_per_source=3
    )

