import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService

from ingestion.confluence_loader import (
    load_confluence_documents
)

from ingestion.ingest_phase1 import (
    ingest_documents
)


def main():

    print("\nCADvPDM GPT")
    print("Confluence Knowledge Ingestion")
    print("=" * 50)

    # -----------------------------------------
    # TEST FIRST WITH ONLY 10 CONFLUENCE PAGES
    # -----------------------------------------

    documents = load_confluence_documents()

    print(
        f"\nTotal Confluence chunks: "
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

    print(
        "\nConfluence ingestion completed!"
    )


if __name__ == "__main__":
    main()