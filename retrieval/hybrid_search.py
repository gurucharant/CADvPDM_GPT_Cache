"""
CADvPDM GPT
Hybrid Search V3

Retrieval pipeline:

    Dense pgvector retrieval
              +
         BM25 retrieval
              |
              v
    Reciprocal Rank Fusion
              |
              v
       Source Diversity
              |
              v
    Semantic Deduplication
              |
              v
    CrossEncoder Reranking
              |
              v
       Final Evidence
"""

from collections import defaultdict

from retrieval.dense_search import DenseSearch
from retrieval.bm25_search import BM25Search
from retrieval.deduplicator import SemanticDeduplicator
from retrieval.reranker import CrossEncoderReranker


class HybridSearch:
    """
    Enterprise hybrid retrieval engine for CADvPDM GPT.

    Combines:

        1. Dense semantic search
        2. BM25 lexical search
        3. Reciprocal Rank Fusion
        4. Source diversity
        5. Semantic deduplication
        6. CrossEncoder reranking
    """

    DEFAULT_SOURCES = [
        "SUPPORT",
        "JIRA",
        "CONFLUENCE",
        "LOG",
        "TEAMS",
        "GIT",
    ]

    def __init__(
        self,
        rrf_k=60,
        dedup_threshold=0.92,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print()
        print("=" * 90)
        print("INITIALIZING CADvPDM HYBRID SEARCH V3")
        print("=" * 90)

        self.rrf_k = rrf_k

        # ----------------------------------------------------------
        # Dense retrieval
        # ----------------------------------------------------------

        print()
        print("Initializing Dense Search...")

        self.dense_search = DenseSearch()

        # ----------------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------------

        print()
        print("Initializing BM25 Search...")

        self.bm25_search = BM25Search()

        # ----------------------------------------------------------
        # Semantic deduplication
        # ----------------------------------------------------------

        print()
        print("Initializing Semantic Deduplicator...")

        self.deduplicator = SemanticDeduplicator(
            embedding_service=self.dense_search.embedding_service,
            similarity_threshold=dedup_threshold,
        )

        # ----------------------------------------------------------
        # CrossEncoder reranking
        # ----------------------------------------------------------

        print()
        print("Initializing CrossEncoder Reranker...")

        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
        )

        print()
        print("=" * 90)
        print("CADvPDM HYBRID SEARCH V3 READY")
        print("=" * 90)

    # ==============================================================
    # Helpers
    # ==============================================================

    @staticmethod
    def _document_key(result):
        """
        Stable identity for one organizational-memory chunk.
        """

        return (
            str(result.get("source_type", "")),
            str(result.get("source_id", "")),
            int(result.get("chunk_index", 0) or 0),
        )

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ==============================================================
    # Dense retrieval
    # ==============================================================

    def _dense_retrieval(
        self,
        question,
        source_types,
        dense_per_source_k,
    ):
        """
        Retrieve dense candidates independently for every source.

        This is important.

        We do NOT globally truncate the dense candidates before RRF,
        because doing so can allow one source such as SUPPORT to
        dominate the candidate pool.
        """

        all_results = []

        for source_type in source_types:

            print(
                f"  Dense search: {source_type}"
            )

            results = self.dense_search.search(
                question=question,
                top_k=dense_per_source_k,
                source_types=[source_type],
            )

            for rank, result in enumerate(
                results,
                start=1,
            ):
                item = dict(result)

                item["dense_rank"] = rank
                item["retrieval_method"] = "DENSE"

                all_results.append(item)

        return all_results

    # ==============================================================
    # BM25 retrieval
    # ==============================================================

    def _bm25_retrieval(
        self,
        question,
        source_types,
        bm25_candidate_k,
    ):
        """
        Retrieve lexical candidates using BM25.
        """

        results = self.bm25_search.search(
            question=question,
            top_k=bm25_candidate_k,
            source_types=source_types,
        )

        output = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            item = dict(result)

            item["bm25_rank"] = rank
            item["retrieval_method"] = "BM25"

            output.append(item)

        return output

    # ==============================================================
    # Reciprocal Rank Fusion
    # ==============================================================

    def _rrf_fusion(
        self,
        dense_results,
        bm25_results,
    ):
        """
        Fuse Dense and BM25 rankings using Reciprocal Rank Fusion.

        RRF contribution:

            1 / (k + rank)

        Dense similarity and BM25 score are deliberately NOT averaged.
        They are different scoring systems and should not be treated
        as directly comparable.
        """

        fused = {}

        # ----------------------------------------------------------
        # Dense ranking contribution
        # ----------------------------------------------------------

        for result in dense_results:

            key = self._document_key(result)

            if key not in fused:
                fused[key] = dict(result)

                fused[key]["rrf_score"] = 0.0
                fused[key]["dense_rank"] = None
                fused[key]["bm25_rank"] = None

                fused[key]["retrieval_sources"] = []

            rank = result.get("dense_rank")

            if rank is not None:
                fused[key]["dense_rank"] = rank

                fused[key]["rrf_score"] += (
                    1.0 /
                    (
                        self.rrf_k
                        + rank
                    )
                )

            if "DENSE" not in fused[key]["retrieval_sources"]:
                fused[key]["retrieval_sources"].append(
                    "DENSE"
                )

            # Preserve dense similarity
            if result.get("similarity_score") is not None:
                fused[key]["similarity_score"] = (
                    result.get("similarity_score")
                )

        # ----------------------------------------------------------
        # BM25 ranking contribution
        # ----------------------------------------------------------

        for result in bm25_results:

            key = self._document_key(result)

            if key not in fused:
                fused[key] = dict(result)

                fused[key]["rrf_score"] = 0.0
                fused[key]["dense_rank"] = None
                fused[key]["bm25_rank"] = None

                fused[key]["retrieval_sources"] = []

            rank = result.get("bm25_rank")

            if rank is not None:
                fused[key]["bm25_rank"] = rank

                fused[key]["rrf_score"] += (
                    1.0 /
                    (
                        self.rrf_k
                        + rank
                    )
                )

            if "BM25" not in fused[key]["retrieval_sources"]:
                fused[key]["retrieval_sources"].append(
                    "BM25"
                )

            if result.get("bm25_score") is not None:
                fused[key]["bm25_score"] = (
                    result.get("bm25_score")
                )

        # ----------------------------------------------------------
        # Final retrieval-source label
        # ----------------------------------------------------------

        for result in fused.values():

            methods = result.get(
                "retrieval_sources",
                [],
            )

            if (
                "DENSE" in methods
                and "BM25" in methods
            ):
                result["retrieval_method"] = "HYBRID"

            elif "DENSE" in methods:
                result["retrieval_method"] = "DENSE"

            elif "BM25" in methods:
                result["retrieval_method"] = "BM25"

            else:
                result["retrieval_method"] = "UNKNOWN"

        results = list(
            fused.values()
        )

        results.sort(
            key=lambda item: self._safe_float(
                item.get("rrf_score")
            ),
            reverse=True,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result["rrf_rank"] = rank

        return results

    # ==============================================================
    # Source diversity
    # ==============================================================

    def _apply_source_diversity(
        self,
        results,
        max_per_source=5,
    ):
        """
        Prevent one enterprise source from dominating the final
        candidate set.

        Example:
            SUPPORT should not occupy all final evidence slots simply
            because many synthetic SUPPORT records are highly similar.
        """

        source_counts = defaultdict(int)

        selected = []

        for result in results:

            source_type = str(
                result.get(
                    "source_type",
                    "UNKNOWN",
                )
            ).upper()

            if (
                source_counts[source_type]
                >= max_per_source
            ):
                continue

            selected.append(result)

            source_counts[source_type] += 1

        return selected

    # ==============================================================
    # Semantic deduplication
    # ==============================================================

    def _semantic_deduplicate(
        self,
        results,
    ):
        """
        Remove semantically near-identical evidence.

        The dedicated SemanticDeduplicator owns the embedding-based
        duplicate detection logic.
        """

        if not results:
            return []

        # Support the interface used by the existing deduplicator.
        if hasattr(
            self.deduplicator,
            "deduplicate",
        ):
            return self.deduplicator.deduplicate(
                results
            )

        if hasattr(
            self.deduplicator,
            "remove_duplicates",
        ):
            return self.deduplicator.remove_duplicates(
                results
            )

        raise AttributeError(
            "SemanticDeduplicator must provide either "
            "deduplicate() or remove_duplicates()."
        )

    # ==============================================================
    # CrossEncoder reranking
    # ==============================================================

    def _rerank(
        self,
        question,
        results,
    ):
        """
        Apply final question-aware CrossEncoder reranking.
        """

        if not results:
            return []

        if hasattr(
            self.reranker,
            "rerank",
        ):
            reranked = self.reranker.rerank(
                question=question,
                results=results,
            )

        elif hasattr(
            self.reranker,
            "rank",
        ):
            reranked = self.reranker.rank(
                question=question,
                results=results,
            )

        else:
            raise AttributeError(
                "CrossEncoderReranker must provide "
                "rerank() or rank()."
            )

        # ----------------------------------------------------------
        # Ensure rank exists even if reranker only supplies score.
        # ----------------------------------------------------------

        for rank, result in enumerate(
            reranked,
            start=1,
        ):
            result["reranker_rank"] = rank

        return reranked

    # ==============================================================
    # Main Search
    # ==============================================================

    def search(
        self,
        question,
        top_k=10,
        dense_per_source_k=10,
        bm25_candidate_k=60,
        max_per_source=5,
        source_types=None,
    ):
        """
        Execute the complete Hybrid Search V3 pipeline.
        """

        if source_types is None:
            source_types = (
                self.DEFAULT_SOURCES.copy()
            )

        source_types = [
            str(source).upper()
            for source in source_types
        ]

        print()
        print("=" * 90)
        print("CADvPDM HYBRID SEARCH V3")
        print("=" * 90)

        print()
        print("Question:")
        print(question)

        print()
        print(
            "Sources:",
            ", ".join(source_types),
        )

        # ==========================================================
        # STEP 1
        # Dense retrieval
        # ==========================================================

        print()
        print(
            "STEP 1 - DENSE MULTI-SOURCE RETRIEVAL"
        )
        print("-" * 90)

        dense_results = (
            self._dense_retrieval(
                question=question,
                source_types=source_types,
                dense_per_source_k=(
                    dense_per_source_k
                ),
            )
        )

        print()
        print(
            "Dense candidates:",
            len(dense_results),
        )

        # ==========================================================
        # STEP 2
        # BM25 retrieval
        # ==========================================================

        print()
        print(
            "STEP 2 - BM25 RETRIEVAL"
        )
        print("-" * 90)

        bm25_results = (
            self._bm25_retrieval(
                question=question,
                source_types=source_types,
                bm25_candidate_k=(
                    bm25_candidate_k
                ),
            )
        )

        print()
        print(
            "BM25 candidates:",
            len(bm25_results),
        )

        # ==========================================================
        # STEP 3
        # RRF
        # ==========================================================

        print()
        print(
            "STEP 3 - RECIPROCAL RANK FUSION"
        )
        print("-" * 90)

        fused_results = (
            self._rrf_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
            )
        )

        print()
        print(
            "Unique fused candidates:",
            len(fused_results),
        )

        # ==========================================================
        # STEP 4
        # Source diversity
        # ==========================================================

        print()
        print(
            "STEP 4 - SOURCE DIVERSITY"
        )
        print("-" * 90)

        diverse_results = (
            self._apply_source_diversity(
                results=fused_results,
                max_per_source=max_per_source,
            )
        )

        print()
        print(
            "Candidates after source diversity:",
            len(diverse_results),
        )

        # ==========================================================
        # STEP 5
        # Semantic deduplication
        # ==========================================================

        print()
        print(
            "STEP 5 - SEMANTIC DEDUPLICATION"
        )
        print("-" * 90)

        before_dedup = len(
            diverse_results
        )

        deduplicated_results = (
            self._semantic_deduplicate(
                diverse_results
            )
        )

        after_dedup = len(
            deduplicated_results
        )

        print()
        print(
            "Candidates before dedup:",
            before_dedup,
        )

        print(
            "Candidates after dedup:",
            after_dedup,
        )

        print(
            "Near-duplicates removed:",
            before_dedup - after_dedup,
        )

        # ==========================================================
        # STEP 6
        # CrossEncoder
        # ==========================================================

        print()
        print(
            "STEP 6 - CROSS-ENCODER RERANKING"
        )
        print("-" * 90)

        reranked_results = (
            self._rerank(
                question=question,
                results=deduplicated_results,
            )
        )

        # ==========================================================
        # STEP 7
        # Final results
        # ==========================================================

        print()
        print(
            "STEP 7 - FINAL HYBRID RESULTS"
        )
        print("-" * 90)

        final_results = (
            reranked_results[:top_k]
        )

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            result["final_rank"] = rank

            reranker_score = (
                result.get(
                    "reranker_score"
                )
            )

            if reranker_score is None:
                score_text = "N/A"
            else:
                score_text = (
                    f"{self._safe_float(reranker_score):.6f}"
                )

            print()
            print(
                f"{rank}. "
                f"{result.get('source_type')} | "
                f"{result.get('source_id')}"
            )

            print(
                "   Application:",
                result.get(
                    "application"
                ),
            )

            print(
                "   Error Code:",
                result.get(
                    "error_code"
                ),
            )

            print(
                "   Retrieval:",
                result.get(
                    "retrieval_method"
                ),
            )

            print(
                "   Reranker Score:",
                score_text,
            )

        print()
        print(
            "Final result count:",
            len(final_results),
        )

        return final_results


# ==================================================================
# Standalone Test
# ==================================================================

if __name__ == "__main__":

    print()
    print("=" * 90)
    print(
        "CADvPDM HYBRID SEARCH V3 - STANDALONE TEST"
    )
    print("=" * 90)

    engine = HybridSearch(
        rrf_k=60,
        dedup_threshold=0.92,
        reranker_model=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
    )

    question = (
        "How was the CAD Batch AUTH-401 "
        "issue resolved previously?"
    )

    results = engine.search(
        question=question,
        top_k=10,
        dense_per_source_k=10,
        bm25_candidate_k=60,
        max_per_source=5,
        source_types=[
            "SUPPORT",
            "JIRA",
            "CONFLUENCE",
            "LOG",
            "TEAMS",
            "GIT",
        ],
    )

    print()
    print("=" * 90)
    print("TEST COMPLETE")
    print("=" * 90)

    print(
        "Returned hybrid results:",
        len(results),
    )