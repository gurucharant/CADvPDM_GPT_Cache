from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")

    def generate_embedding(self, text):
        """
        Generate embedding for a single text.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_embeddings(self, texts, batch_size=64):
        """
        Generate embeddings for multiple texts.
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings


if __name__ == "__main__":

    service = EmbeddingService()

    # Test single embedding
    text = """
    CATIA batch failed because the
    service account credential expired.
    """

    vector = service.generate_embedding(text)

    print("\nSingle embedding")
    print("Number of dimensions:", len(vector))
    print(vector[:10])

    # Test batch embeddings
    texts = [
        "CATIA login problem",
        "Database connection timeout",
        "License server unavailable"
    ]

    vectors = service.generate_embeddings(texts)

    print("\nBatch embeddings")
    print("Number of documents:", len(vectors))
    print("Dimensions per document:", len(vectors[0]))