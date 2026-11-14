import numpy as np

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService


class SemanticSearch:

    def __init__(self):
        print("Initializing CADvPDM Semantic Search...")

        self.embedding_service = EmbeddingService()

    def search(self, question, top_k=5):

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        # --------------------------------------------------
        # 1. Convert user question into embedding
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

        # --------------------------------------------------
        # 3. Vector similarity search
        # --------------------------------------------------

        sql = """
            SELECT
                source_type,
                source_id,
                title,
                site,
                application,
                error_code,
                content,

                1 - (
                    embedding <=> %s
                ) AS similarity

            FROM cadvpdm_knowledge

            WHERE embedding IS NOT NULL AND source_type = 'CONFLUENCE'

            ORDER BY
                embedding <=> %s

            LIMIT %s;
        """

        cursor.execute(
            sql,
            (
                query_vector,
                query_vector,
                top_k
            )
        )

        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # --------------------------------------------------
        # 4. Display results
        # --------------------------------------------------

        print("\n")
        print("=" * 80)
        print("MOST RELEVANT HISTORICAL RECORDS")
        print("=" * 80)

        for index, row in enumerate(results, start=1):

            (
                source_type,
                source_id,
                title,
                site,
                application,
                error_code,
                content,
                similarity
            ) = row

            print(f"\nRESULT #{index}")
            print("-" * 80)

            print(
                f"Source       : {source_type}"
            )

            print(
                f"Source ID    : {source_id}"
            )

            print(
                f"Title        : {title}"
            )

            print(
                f"Site         : {site}"
            )

            print(
                f"Application  : {application}"
            )

            print(
                f"Error Code   : {error_code}"
            )

            print(
                f"Similarity   : "
                f"{float(similarity):.4f}"
            )

            print("\nContent:")
            print(content[:1000])

        return results


if __name__ == "__main__":

    search_engine = SemanticSearch()

    question = """
    A CAD Batch job failed during login 
    with an unauthorized authentication error.
    Have we experienced this before, what 
    caused it, and how was it resolved?
    """

    search_engine.search(
        question=question,
        top_k=5
    )