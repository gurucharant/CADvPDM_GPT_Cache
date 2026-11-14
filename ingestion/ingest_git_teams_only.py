"""
Ingest only GIT and TEAMS documents into pgvector semantic memory.

Use this when Jira, Support, Confluence and Logs are already present
in cadvpdm_knowledge and we only want to add the new semantic sources.
"""

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService
from ingestion.ingest_phase1 import ingest_documents
from ingestion.git_loader import load_git_documents
from ingestion.teams_loader import load_teams_documents


def main():
    docs = []

    print("Loading GIT documents...")
    git_docs = load_git_documents()
    print(f"GIT documents/chunks: {len(git_docs):,}")
    docs += git_docs

    print("Loading TEAMS documents...")
    teams_docs = load_teams_documents()
    print(f"TEAMS documents/chunks: {len(teams_docs):,}")
    docs += teams_docs

    print(f"\nTotal new semantic documents/chunks: {len(docs):,}")

    if not docs:
        print("No GIT or TEAMS documents found. Nothing to ingest.")
        return

    service = EmbeddingService()
    conn = get_connection()

    try:
        ingest_documents(docs, service, conn)
        print("\nGIT + TEAMS semantic ingestion completed successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()