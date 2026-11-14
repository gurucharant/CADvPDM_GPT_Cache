import numpy as np
from sentence_transformers.util import cos_sim


class SemanticDeduplicator:
    """
    Semantic near-duplicate removal for CADvPDM GPT.

    Purpose
    -------
    Hybrid retrieval may return different records that contain
    essentially the same knowledge.

    Example:

        SUP-500109
        SUP-500133
        SUP-510289

    may all describe:

        Root Cause:
        Expired CAD Batch service credential

        Resolution:
        Renew credential
        Update secure credential store
        Validate login
        Re-run batch

    These are different incidents, but giving all of them to the
    LLM wastes context-window space.

    This class keeps the highest-ranked representative and removes
    semantically redundant candidates.
    """

    def __init__(
        self,
        embedding_service,
        similarity_threshold: float = 0.92
    ):

        self.embedding_service = embedding_service
        self.similarity_threshold = (
            similarity_threshold
        )

    # ==================================================================
    # BUILD TEXT USED FOR DUPLICATE DETECTION
    # ==================================================================

    @staticmethod
    def _build_comparison_text(
        document: dict
    ) -> str:

        """
        Build semantic representation of the document.

        IMPORTANT:
        We intentionally do not emphasize source_id, site, etc.

        Two incidents from Sweden and Germany can still represent
        the same troubleshooting knowledge.
        """

        parts = [
            document.get("title") or "",
            document.get("application") or "",
            document.get("error_code") or "",
            document.get("content") or "",
        ]

        text = "\n".join(
            str(part).strip()
            for part in parts
            if part
        )

        return text

    # ==================================================================
    # EMBEDDINGS
    # ==================================================================

    def _generate_embeddings(
        self,
        documents: list[dict]
    ) -> np.ndarray:

        texts = [
            self._build_comparison_text(document)
            for document in documents
        ]

        # --------------------------------------------------------------
        # Reuse the SAME embedding model already loaded by DenseSearch.
        #
        # We deliberately avoid creating another SentenceTransformer
        # instance.
        # --------------------------------------------------------------

        model = (
            self.embedding_service.model
        )

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings

    # ==================================================================
    # DEDUPLICATION
    # ==================================================================

    def deduplicate(
        self,
        results: list[dict],
        top_k: int | None = None
    ) -> list[dict]:

        print("\n")
        print("=" * 90)
        print("SEMANTIC NEAR-DUPLICATE REMOVAL")
        print("=" * 90)

        if not results:

            print(
                "No candidates supplied."
            )

            return []

        print(
            f"\nInput candidates: "
            f"{len(results)}"
        )

        print(
            f"Similarity threshold: "
            f"{self.similarity_threshold}"
        )

        # --------------------------------------------------------------
        # Generate embeddings once for all candidates
        # --------------------------------------------------------------

        embeddings = (
            self._generate_embeddings(
                results
            )
        )

        selected_results = []
        selected_indices = []

        removed_count = 0

        # --------------------------------------------------------------
        # Results arrive in RRF ranking order.
        #
        # Therefore the first representative of a semantic cluster is
        # retained and lower-ranked near-duplicates are removed.
        # --------------------------------------------------------------

        for candidate_index, candidate in enumerate(
            results
        ):

            candidate_embedding = (
                embeddings[candidate_index]
            )

            duplicate = False
            duplicate_of = None
            duplicate_similarity = None

            # ----------------------------------------------------------
            # Compare against candidates already accepted
            # ----------------------------------------------------------

            for selected_index in selected_indices:

                selected_embedding = (
                    embeddings[selected_index]
                )

                similarity = float(
                    cos_sim(
                        candidate_embedding,
                        selected_embedding
                    ).item()
                )

                if (
                    similarity
                    >= self.similarity_threshold
                ):

                    duplicate = True

                    duplicate_of = (
                        results[selected_index]
                    )

                    duplicate_similarity = (
                        similarity
                    )

                    break

            # ----------------------------------------------------------
            # Skip near duplicate
            # ----------------------------------------------------------

            if duplicate:

                removed_count += 1

                print(
                    "\nRemoved near duplicate:"
                )

                print(
                    f"  {candidate.get('source_type')} "
                    f"{candidate.get('source_id')}"
                )

                print(
                    "  Similar to:"
                )

                print(
                    f"  {duplicate_of.get('source_type')} "
                    f"{duplicate_of.get('source_id')}"
                )

                print(
                    f"  Semantic similarity: "
                    f"{duplicate_similarity:.4f}"
                )

                continue

            # ----------------------------------------------------------
            # Keep candidate
            # ----------------------------------------------------------

            selected_results.append(
                candidate
            )

            selected_indices.append(
                candidate_index
            )

            # ----------------------------------------------------------
            # Optional early stop
            # ----------------------------------------------------------

            if (
                top_k is not None
                and len(selected_results) >= top_k
            ):

                break

        print("\n" + "-" * 90)

        print(
            f"Near duplicates removed: "
            f"{removed_count}"
        )

        print(
            f"Candidates retained: "
            f"{len(selected_results)}"
        )

        return selected_results