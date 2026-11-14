"""Populate pgvector semantic memory with curated/searchable content.
Run after the cadvpdm_knowledge table exists.
"""
from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService
from ingestion.ingest_phase1 import ingest_documents
from ingestion.jira_loader import load_jira_documents
from ingestion.support_loader import load_support_documents
from ingestion.confluence_loader import load_confluence_documents
from ingestion.xml_log_loader import load_xml_log_documents
from ingestion.git_loader import load_git_documents
from ingestion.teams_loader import load_teams_documents

def main():
    docs=[]
    docs += load_jira_documents()
    docs += load_support_documents()
    docs += load_confluence_documents()
    # Logs/Git are selectively embedded in a production design; included here to demonstrate cross-source POC.
    docs += load_xml_log_documents()
    docs += load_git_documents()
    docs += load_teams_documents()
    print(f"Total semantic documents/chunks: {len(docs):,}")
    service=EmbeddingService(); conn=get_connection()
    try: ingest_documents(docs,service,conn)
    finally: conn.close()

if __name__=='__main__': main()
