import numpy as np

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService


embedding_service = EmbeddingService()

text = """
CATIA batch failed because the service account
credential expired. Support renewed the credential
and restarted the failed batch.
"""

embedding = embedding_service.generate_embedding(text)

print("Embedding dimensions:", len(embedding))

conn = get_connection()
cursor = conn.cursor()

cursor.execute(
    """
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
    VALUES
    (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    """,
    (
        "SUPPORT",
        "SUP-TEST-001",
        0,
        "CATIA service credential failure",
        text,
        "SE-CENTRAL",
        "CAD Batch",
        "AUTH-401",
        '{"priority":"P2","resolver":"CAD Support"}',
        np.array(embedding, dtype=np.float32)
    )
)

conn.commit()

print("Record inserted successfully!")

cursor.close()
conn.close()