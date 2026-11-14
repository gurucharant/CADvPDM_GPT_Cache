import sys
from pathlib import Path

import numpy as np
from psycopg2.extras import execute_values, Json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService
from ingestion.jira_loader import load_jira_documents
from ingestion.support_loader import load_support_documents


EMBEDDING_BATCH_SIZE = 64
DATABASE_BATCH_SIZE = 500


def insert_batch(
        conn,
        documents,
        embeddings
):

    rows = []

    for doc, embedding in zip(
            documents,
            embeddings
    ):

        rows.append(
            (
                doc["source_type"],
                doc["source_id"],
                doc["chunk_index"],
                doc["title"],
                doc["content"],
                doc["site"],
                doc["application"],
                doc["error_code"],
                Json(doc["metadata"]),
                np.array(
                    embedding,
                    dtype=np.float32
                )
            )
        )

    sql = """
        INSERT INTO cadvpdm_knowledge
        (
            source_type,
            source_id,
            chunk_index,
            title,
            content,
            site,
            application,
            error_code,
            metadata,
            embedding
        )
        VALUES %s

        ON CONFLICT
        (
            source_type,
            source_id,
            chunk_index
        )

        DO UPDATE SET

            title = EXCLUDED.title,
            content = EXCLUDED.content,
            site = EXCLUDED.site,
            application = EXCLUDED.application,
            error_code = EXCLUDED.error_code,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
    """

    execute_values(
        conn.cursor(),
        sql,
        rows,
        page_size=DATABASE_BATCH_SIZE
    )


def ingest_documents(
        documents,
        embedding_service,
        conn
):

    total = len(documents)

    print(
        f"\nStarting ingestion of "
        f"{total:,} documents..."
    )

    for start in range(
            0,
            total,
            DATABASE_BATCH_SIZE
    ):

        end = min(
            start + DATABASE_BATCH_SIZE,
            total
        )

        batch = documents[start:end]

        texts = [
            doc["content"]
            for doc in batch
        ]

        embeddings = (
            embedding_service
            .generate_embeddings(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE
            )
        )

        insert_batch(
            conn,
            batch,
            embeddings
        )

        conn.commit()

        print(
            f"Inserted "
            f"{end:,} / {total:,}"
        )


def main():

    print("\nCADvPDM GPT")
    print("Phase 1 Knowledge Ingestion")
    print("=" * 40)

    jira_documents = (
        load_jira_documents()
    )

    support_documents = (
        load_support_documents()
    )

    documents = (
        jira_documents
        + support_documents
    )

    print(
        f"\nTotal documents: "
        f"{len(documents):,}"
    )

    embedding_service = (
        EmbeddingService()
    )

    conn = get_connection()

    try:

        ingest_documents(
            documents,
            embedding_service,
            conn
        )

    finally:

        conn.close()

    print("\nIngestion completed!")


if __name__ == "__main__":
    main()