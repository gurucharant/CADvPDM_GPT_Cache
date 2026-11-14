import re
from typing import Optional

from rank_bm25 import BM25Okapi

from database.connection import get_connection


class BM25Search:
    """
    BM25 sparse/lexical search for CADvPDM GPT.

    Particularly useful for exact technical terms such as:
        AUTH-401
        AUTH-403
        SUP-528672
        SE-CENTRAL
        CAD Batch
        ENOVIA
    """

    def __init__(self):
        print("=" * 90)
        print("INITIALIZING CADvPDM BM25 SEARCH")
        print("=" * 90)

        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        self._load_documents()
        self._build_index()

    # ------------------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:

        if not text:
            return []

        text = text.lower()

        # Keeps technical identifiers intact:
        #
        # AUTH-401
        # SUP-528672
        # SE-CENTRAL
        # user_name
        #
        tokens = re.findall(
            r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
            text
        )

        return tokens

    # ------------------------------------------------------------------
    # BUILD SEARCHABLE DOCUMENT
    # ------------------------------------------------------------------

    @staticmethod
    def build_search_text(document: dict) -> str:

        fields = [
            document.get("source_type"),
            document.get("source_id"),
            document.get("title"),
            document.get("site"),
            document.get("application"),
            document.get("error_code"),
            document.get("content"),
        ]

        return " ".join(
            str(value)
            for value in fields
            if value is not None
        )

    # ------------------------------------------------------------------
    # LOAD ORGANIZATIONAL MEMORY
    # ------------------------------------------------------------------

    def _load_documents(self):

        print("\nLoading documents from PostgreSQL...")

        conn = get_connection()
        cursor = conn.cursor()

        sql = """
            SELECT
                source_type,
                source_id,
                chunk_index,
                title,
                site,
                application,
                error_code,
                content,
                metadata
            FROM cadvpdm_knowledge
            WHERE content IS NOT NULL;
        """

        try:

            cursor.execute(sql)

            rows = cursor.fetchall()

            for row in rows:

                (
                    source_type,
                    source_id,
                    chunk_index,
                    title,
                    site,
                    application,
                    error_code,
                    content,
                    metadata
                ) = row

                document = {
                    "source_type": source_type,
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "site": site,
                    "application": application,
                    "error_code": error_code,
                    "content": content,
                    "metadata": metadata,
                }

                self.documents.append(document)

        finally:

            cursor.close()
            conn.close()

        print(
            f"Loaded {len(self.documents):,} "
            f"knowledge chunks."
        )

    # ------------------------------------------------------------------
    # BUILD BM25 INDEX
    # ------------------------------------------------------------------

    def _build_index(self):

        print("Building BM25 index...")

        if not self.documents:

            raise RuntimeError(
                "No documents found in cadvpdm_knowledge."
            )

        for document in self.documents:

            searchable_text = self.build_search_text(
                document
            )

            tokens = self.tokenize(
                searchable_text
            )

            self.tokenized_corpus.append(
                tokens
            )

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print("BM25 index ready.")

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        question: str,
        top_k: int = 10,
        source_types: Optional[list[str]] = None
    ) -> list[dict]:

        print("\n" + "=" * 90)
        print("BM25 SPARSE SEARCH")
        print("=" * 90)

        print(f"\nQuestion: {question}")

        query_tokens = self.tokenize(
            question
        )

        print(
            f"Query tokens: {query_tokens}"
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        results = []

        for index, score in enumerate(scores):

            document = self.documents[index]

            # Optional source filtering
            if source_types:

                if (
                    document["source_type"]
                    not in source_types
                ):
                    continue

            # Ignore documents with zero lexical relevance
            if score <= 0:
                continue

            result = {
                **document,
                "bm25_score": float(score),
                "retrieval_method": "BM25"
            }

            results.append(result)

        # Highest BM25 score first
        results.sort(
            key=lambda item: item["bm25_score"],
            reverse=True
        )

        results = results[:top_k]

        self._display_results(
            results
        )

        return results

    # ------------------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------------------

    @staticmethod
    def _display_results(results):

        print("\n" + "=" * 90)
        print("BM25 RESULTS")
        print("=" * 90)

        if not results:

            print("No BM25 matches found.")
            return

        for rank, result in enumerate(
            results,
            start=1
        ):

            print("\n" + "#" * 90)

            print(
                f"RESULT #{rank} | "
                f"{result['source_type']}"
            )

            print("#" * 90)

            print(
                f"Source ID   : "
                f"{result['source_id']}"
            )

            print(
                f"Chunk       : "
                f"{result['chunk_index']}"
            )

            print(
                f"Title       : "
                f"{result['title']}"
            )

            print(
                f"Site        : "
                f"{result['site']}"
            )

            print(
                f"Application : "
                f"{result['application']}"
            )

            print(
                f"Error Code  : "
                f"{result['error_code']}"
            )

            print(
                f"BM25 Score  : "
                f"{result['bm25_score']:.4f}"
            )

            print("\nCONTENT")
            print("-" * 90)

            content = (
                result["content"]
                or ""
            )

            print(
                content[:1000]
            )


# ======================================================================
# STANDALONE TEST
# ======================================================================

if __name__ == "__main__":

    search_engine = BM25Search()

    question = (
        "How was the CAD Batch AUTH-401 "
        "issue resolved previously?"
    )

    search_engine.search(
        question=question,
        top_k=10
    )