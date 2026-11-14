import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from database.connection import get_connection

from embeddings.embedding_service import (
    EmbeddingService
)

from ingestion.xml_log_loader import (
    load_xml_log_documents
)

from ingestion.ingest_phase1 import (
    ingest_documents
)


def main():

    print("\nCADvPDM GPT")
    print("XML / Control-M Log Ingestion")
    print("=" * 50)

    # First test only 20 files
    documents = load_xml_log_documents()

    print(
        f"\nTotal log documents: "
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
        "\nLog ingestion completed!"
    )


if __name__ == "__main__":
    main()