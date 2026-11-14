import numpy as np

from database.connection import get_connection
from embeddings.embedding_service import EmbeddingService


embedding_service = EmbeddingService()

question = """
Have we previously experienced CATIA login
or authentication problems?
"""

query_embedding = embedding_service.generate_embedding(question)

conn = get_connection()
cursor = conn.cursor()

cursor.execute(
    """
    SELECT
        source_type,
        source_id,
        title,
        content,
        error_code,
        1 - (embedding <=> %s) AS similarity
    FROM cadvpdm_knowledge
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s
    LIMIT 5;
    """,
    (
        np.array(query_embedding, dtype=np.float32),
        np.array(query_embedding, dtype=np.float32)
    )
)

results = cursor.fetchall()

for result in results:
    print("\n------------------------------")
    print("Source:", result[0])
    print("ID:", result[1])
    print("Title:", result[2])
    print("Error Code:", result[4])
    print("Similarity:", round(result[5], 4))
    print("Content:", result[3])

cursor.close()
conn.close()