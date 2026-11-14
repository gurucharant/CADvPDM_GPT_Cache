from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    CrossEncoder reranker for CADvPDM GPT.

    Pipeline position
    -----------------
    Dense Search
        +
    BM25 Search
        ↓
    RRF Fusion
        ↓
    Source Diversity
        ↓
    Semantic Deduplication
        ↓
    CrossEncoder Reranking   <-- THIS CLASS
        ↓
    Best Evidence
        ↓
    Context Builder
        ↓
    LLM

    Why use a CrossEncoder?
    -----------------------
    Dense retrieval and BM25 are excellent for finding candidate
    documents.

    But retrieval score does not necessarily mean that a document
    directly answers the user's question.

    A CrossEncoder reads:

        [user question, candidate document]

    together and produces a relevance score.

    This makes it especially useful for questions such as:

        "How was the CAD Batch AUTH-401 issue resolved previously?"

    because documents containing an actual resolution should normally
    rank above documents that merely mention AUTH-401.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL
    ):

        print("\n")
        print("=" * 90)
        print("INITIALIZING CROSS-ENCODER RERANKER")
        print("=" * 90)

        print(
            f"\nLoading reranker model:\n"
            f"{model_name}"
        )

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name
        )

        print(
            "\nCrossEncoder reranker ready."
        )

    # ==================================================================
    # BUILD DOCUMENT TEXT
    # ==================================================================

    @staticmethod
    def _build_document_text(
        document: dict
    ) -> str:

        """
        Build the text that will be evaluated against the user's
        question.

        We include useful enterprise metadata because fields such as
        application and error code can be important relevance signals.
        """

        parts = []

        source_type = document.get(
            "source_type"
        )

        source_id = document.get(
            "source_id"
        )

        title = document.get(
            "title"
        )

        site = document.get(
            "site"
        )

        application = document.get(
            "application"
        )

        error_code = document.get(
            "error_code"
        )

        content = document.get(
            "content"
        )

        if source_type:
            parts.append(
                f"Source Type: {source_type}"
            )

        if source_id:
            parts.append(
                f"Source ID: {source_id}"
            )

        if title:
            parts.append(
                f"Title: {title}"
            )

        if site:
            parts.append(
                f"Site: {site}"
            )

        if application:
            parts.append(
                f"Application: {application}"
            )

        if error_code:
            parts.append(
                f"Error Code: {error_code}"
            )

        if content:
            parts.append(
                f"Content:\n{content}"
            )

        return "\n".join(parts)

    # ==================================================================
    # RERANK
    # ==================================================================

    def rerank(
        self,
        question: str,
        results: list[dict],
        top_k: int | None = None
    ) -> list[dict]:

        print("\n")
        print("=" * 90)
        print("CROSS-ENCODER RERANKING")
        print("=" * 90)

        print(
            f"\nQuestion: {question}"
        )

        print(
            f"Input candidates: {len(results)}"
        )

        if not results:
            print(
                "\nNo candidates supplied "
                "to reranker."
            )

            return []

        # --------------------------------------------------------------
        # Build question-document pairs
        # --------------------------------------------------------------

        pairs = []

        for result in results:

            document_text = (
                self._build_document_text(
                    result
                )
            )

            pairs.append(
                [
                    question,
                    document_text
                ]
            )

        # --------------------------------------------------------------
        # CrossEncoder prediction
        # --------------------------------------------------------------

        print(
            "\nCalculating CrossEncoder "
            "relevance scores..."
        )

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        # --------------------------------------------------------------
        # Preserve original ranking before reranking
        # --------------------------------------------------------------

        reranked_results = []

        for original_rank, (
            result,
            score
        ) in enumerate(
            zip(results, scores),
            start=1
        ):

            updated_result = (
                result.copy()
            )

            updated_result[
                "pre_rerank_rank"
            ] = original_rank

            updated_result[
                "reranker_score"
            ] = float(score)

            reranked_results.append(
                updated_result
            )

        # --------------------------------------------------------------
        # Sort by CrossEncoder relevance
        # --------------------------------------------------------------

        reranked_results.sort(
            key=lambda item:
            item.get(
                "reranker_score",
                float("-inf")
            ),
            reverse=True
        )

        # --------------------------------------------------------------
        # Add final rank
        # --------------------------------------------------------------

        for final_rank, result in enumerate(
            reranked_results,
            start=1
        ):

            result[
                "reranker_rank"
            ] = final_rank

        # --------------------------------------------------------------
        # Optional final top-k
        # --------------------------------------------------------------

        if top_k is not None:

            reranked_results = (
                reranked_results[:top_k]
            )

        # --------------------------------------------------------------
        # Display ranking
        # --------------------------------------------------------------

        self._display_results(
            reranked_results
        )

        return reranked_results

    # ==================================================================
    # DISPLAY RESULTS
    # ==================================================================

    @staticmethod
    def _display_results(
        results: list[dict]
    ):

        print("\n")
        print("=" * 90)
        print("RERANKED RESULTS")
        print("=" * 90)

        for index, result in enumerate(
            results,
            start=1
        ):

            print("\n")
            print("#" * 90)

            print(
                f"RERANKED RESULT #{index}"
            )

            print("#" * 90)

            print(
                f"Source       : "
                f"{result.get('source_type')}"
            )

            print(
                f"Source ID    : "
                f"{result.get('source_id')}"
            )

            print(
                f"Title        : "
                f"{result.get('title')}"
            )

            print(
                f"Site         : "
                f"{result.get('site')}"
            )

            print(
                f"Application  : "
                f"{result.get('application')}"
            )

            print(
                f"Error Code   : "
                f"{result.get('error_code')}"
            )

            print("-" * 90)

            print(
                f"Previous Rank: "
                f"{result.get('pre_rerank_rank')}"
            )

            print(
                f"Reranker Rank: "
                f"{result.get('reranker_rank')}"
            )

            reranker_score = result.get(
                "reranker_score"
            )

            if reranker_score is not None:

                print(
                    f"Reranker Score: "
                    f"{reranker_score:.6f}"
                )

            similarity = result.get(
                "similarity_score"
            )

            if similarity is not None:

                print(
                    f"Dense Score  : "
                    f"{similarity:.4f}"
                )

            bm25_score = result.get(
                "bm25_score"
            )

            if bm25_score is not None:

                print(
                    f"BM25 Score   : "
                    f"{bm25_score:.4f}"
                )

            rrf_score = result.get(
                "rrf_score"
            )

            if rrf_score is not None:

                print(
                    f"RRF Score    : "
                    f"{rrf_score:.6f}"
                )

            print("\nCONTENT")
            print("-" * 90)

            content = (
                result.get("content")
                or ""
            )

            # Avoid flooding console
            print(
                content[:1500]
            )

            if len(content) > 1500:
                print(
                    "\n...[content truncated]"
                )