import re

from retrieval.hybrid_search import HybridSearch
from llm.ollama_client import OllamaClient

from rag.context_builder import build_context
from rag.prompt_builder import build_rag_prompt
from rag.answer_composer import compose_mixed_answer

from routing.query_router_v2 import route_query
from analytics.analytics_engine import AnalyticsEngine
from rag.semantic_cache import SemanticResponseCache
from security.guardrails import SecurityGuardrails


class CADvPDMRAG:

    ALL_SEMANTIC_SOURCES = [
        "SUPPORT",
        "JIRA",
        "CONFLUENCE",
        "LOG",
        "TEAMS",
        "GIT",
    ]

    # ==============================================================
    # Initialization
    # ==============================================================

    def __init__(self):

        print("=" * 90)
        print(
            "Initializing CADvPDM GPT - "
            "Hybrid Intelligence V5"
        )
        print("=" * 90)

        # ----------------------------------------------------------
        # NEW HYBRID RETRIEVAL ENGINE
        #
        # Dense pgvector
        #       +
        # BM25
        #       ↓
        # RRF
        #       ↓
        # Source Diversity
        #       ↓
        # Semantic Deduplication
        #       ↓
        # CrossEncoder Reranking
        # ----------------------------------------------------------

        self.search_engine = HybridSearch(
            rrf_k=60,
            dedup_threshold=0.92,
            reranker_model=(
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
        )

        # Structured analytics / Iceberg
        self.analytics = AnalyticsEngine()

        # Local LLM
        self.llm = OllamaClient(
            model="llama3.2:3b"
        )

        # Semantic response cache. Reuse the SAME MiniLM embedding model
        # already loaded by HybridSearch; do not load a second model.
        self.semantic_cache = SemanticResponseCache(
            embedding_service=(
                self.search_engine.dense_search.embedding_service
            )
        )

        # Security/governance guardrails. This sits in front of routing so
        # blocked requests never reach Router V4, Iceberg, retrieval, cache,
        # or the LLM.
        self.security = SecurityGuardrails()

        print()
        print("=" * 90)
        print(
            "CADvPDM GPT - Hybrid Intelligence V5 READY"
        )
        print("=" * 90)

    # ==============================================================
    # Hybrid Search
    # ==============================================================

    def _search(
        self,
        question,
        results_per_source=3,
        preferred_sources=None,
    ):
        """
        Run the complete hybrid retrieval pipeline.

        HybridSearch performs:

            Dense pgvector retrieval
                    +
            BM25 lexical retrieval
                    ↓
            Reciprocal Rank Fusion
                    ↓
            Source Diversity
                    ↓
            Semantic Deduplication
                    ↓
            CrossEncoder Reranking
                    ↓
            Final Evidence

        NOTE:
        results_per_source is retained for compatibility with the
        existing RAG API. Dense retrieval always preserves at least
        10 candidates per active source.
        """

        active_sources = (
            preferred_sources
            if preferred_sources
            else self.ALL_SEMANTIC_SOURCES
        )

        return self.search_engine.search(
            question=question,
            top_k=10,
            dense_per_source_k=max(
                10,
                results_per_source,
            ),
            bm25_candidate_k=60,
            max_per_source=5,
            source_types=active_sources,
        )

    # ==============================================================
    # Hybrid result -> legacy tuple adapter
    # ==============================================================

    def _hybrid_result_to_legacy_tuple(
        self,
        result,
    ):
        """
        HybridSearch returns dictionaries.

        Existing components such as:

            context_builder.py
            answer_composer.py
            app.py

        currently expect the old tuple structure.

        This adapter allows the new retrieval engine to be introduced
        without changing those components yet.

        Legacy tuple layout:

            0  source_type
            1  source_id
            2  chunk_index
            3  title
            4  site
            5  application
            6  error_code
            7  content
            8  metadata
            9  relevance score
        """

        relevance_score = result.get(
            "reranker_score"
        )

        if relevance_score is None:
            relevance_score = result.get(
                "rrf_score"
            )

        if relevance_score is None:
            relevance_score = result.get(
                "similarity_score",
                0.0,
            )

        metadata = result.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        return (
            result.get(
                "source_type",
                "",
            ),
            result.get(
                "source_id",
                "",
            ),
            result.get(
                "chunk_index",
                0,
            ),
            result.get(
                "title",
                "",
            ),
            result.get(
                "site",
                "",
            ),
            result.get(
                "application",
                "",
            ),
            result.get(
                "error_code",
                "",
            ),
            result.get(
                "content",
                "",
            ),
            metadata,
            float(
                relevance_score or 0.0
            ),
        )

    def _convert_hybrid_results(
        self,
        hybrid_results,
    ):
        """
        Convert all HybridSearch dictionary results into the legacy
        tuple structure expected by the existing application.
        """

        return [
            self._hybrid_result_to_legacy_tuple(
                result
            )
            for result in hybrid_results
        ]

    # ==============================================================
    # Document identity helper
    # ==============================================================

    @staticmethod
    def _document_key(
        result,
    ):
        """
        Stable identity for one retrieved organizational record.
        """

        return (
            result.get(
                "source_type",
                "",
            ),
            result.get(
                "source_id",
                "",
            ),
            result.get(
                "chunk_index",
                0,
            ),
        )

    # ==============================================================
    # Source Selection
    # ==============================================================

    def _get_preferred_sources(
        self,
        plan,
    ):
        """
        Select sources based on query intent.

        KNOWLEDGE:
            Prefer Confluence because the user is asking for
            documented organizational knowledge.

        SEMANTIC:
            Search all organizational sources.

        MIXED:
            Search all relevant organizational sources because the
            semantic portion may require incident history, procedures,
            logs, Teams discussions or Git activity.
        """

        if plan.route == "KNOWLEDGE":

            return [
                "CONFLUENCE"
            ]

        if (
            plan.route == "MIXED"
            and getattr(
                plan,
                "needs_knowledge",
                False,
            )
        ):

            return [
                "CONFLUENCE",
                "SUPPORT",
                "JIRA",
                "LOG",
                "TEAMS",
                "GIT",
            ]

        return self.ALL_SEMANTIC_SOURCES

    # ==============================================================
    # Semantic-query decomposition
    # ==============================================================

    def _build_semantic_query(
        self,
        original_question,
        plan,
    ):
        """
        For pure SEMANTIC / KNOWLEDGE questions we retain the user's
        original question.

        For MIXED questions, analytical phrases such as:

            how many
            top sites
            which site had the most
            percentage

        are not particularly useful for semantic retrieval.

        Therefore, MIXED mode constructs a semantic sub-query focused
        on organizational knowledge.
        """

        if plan.route != "MIXED":
            return original_question

        parts = []

        # ----------------------------------------------------------
        # Preserve explicit technical identifiers
        # ----------------------------------------------------------

        error_codes = re.findall(
            r"\b[A-Z][A-Z0-9]*-\d+\b",
            original_question.upper(),
        )

        parts.extend(
            error_codes
        )

        # ----------------------------------------------------------
        # Preserve known application names
        # ----------------------------------------------------------

        application_terms = [
            "CAD Batch",
            "ENOVIA",
            "CATIA V5",
            "PDM Sync",
        ]

        q_lower = (
            original_question.lower()
        )

        for application in application_terms:

            if application.lower() in q_lower:

                parts.append(
                    application
                )

        # ----------------------------------------------------------
        # Historical root cause
        # ----------------------------------------------------------

        if getattr(
            plan,
            "needs_root_cause",
            False,
        ):

            parts.extend([
                "historical root cause",
                "cause",
                "diagnostic findings",
            ])

        # ----------------------------------------------------------
        # Historical resolution
        # ----------------------------------------------------------

        if getattr(
            plan,
            "needs_resolution",
            False,
        ):

            parts.extend([
                "historical resolution",
                "remediation",
                "how issue was resolved",
                "previous fixes",
            ])

        # ----------------------------------------------------------
        # Historical experience
        # ----------------------------------------------------------

        if getattr(
            plan,
            "needs_semantic_history",
            False,
        ):

            parts.extend([
                "previous incidents",
                "historical cases",
            ])

        # ----------------------------------------------------------
        # Documented knowledge
        # ----------------------------------------------------------

        if getattr(
            plan,
            "needs_knowledge",
            False,
        ):

            parts.extend([
                "documented procedure",
                "methodology",
                "troubleshooting guide",
                "Confluence",
            ])

        # ----------------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------------

        if len(parts) <= len(
            error_codes
        ):

            parts.extend([
                "historical incidents",
                "root causes",
                "resolutions",
                "engineering knowledge",
            ])

        # Remove duplicates while retaining order
        semantic_query = " ".join(
            dict.fromkeys(
                parts
            )
        )

        print()
        print("Semantic sub-query:")
        print(semantic_query)

        return semantic_query

    # ==============================================================
    # GENERAL prompt
    # ==============================================================

    def _build_general_prompt(
        self,
        question,
    ):

        return f"""
You are CADvPDM GPT.

The user is asking a general knowledge or explanatory question.

Answer directly using your general knowledge.

STRICT RULES:

- Do NOT claim that organizational memory was searched.
- Do NOT use company-specific information.
- Do NOT assume the question refers to incidents.
- Do NOT create an incident report unless explicitly requested.
- Do NOT create root-cause analysis unless explicitly requested.
- Do NOT provide historical organizational claims.
- Do NOT mention Jira, Support, Confluence, Git, Teams, logs,
  pgvector, BM25, CrossEncoder, RRF or Apache Iceberg unless
  directly relevant to the user's question.
- Keep the answer clear and practical.
- Use simple language where appropriate.

USER QUESTION:

{question}
"""

    # ==============================================================
    # MIXED semantic prompt
    # ==============================================================

    def _build_mixed_semantic_prompt(
        self,
        question,
        context,
    ):
        """
        The LLM handles ONLY the organizational-memory portion of a
        MIXED query.

        Structured analytics remains responsible for numerical facts.
        """

        return f"""
You are CADvPDM GPT.

You are performing ONLY the historical organizational-knowledge part
of a larger MIXED organizational-intelligence answer.

Structured Python analytics is separately responsible for:

- total counts
- rankings
- top sites
- resolver counts
- contact information
- percentages
- trends
- SLA statistics

DO NOT repeat, estimate or calculate those values here.


============================================================
YOUR TASK
============================================================

Using ONLY the retrieved organizational evidence, identify information
that is relevant to the user's question.

Depending on what the user asked, this may include:

1. Recurring historical root-cause pattern
2. Recurring historical resolution pattern
3. Important documented validation or operational guidance
4. Useful cross-source observations

Only include categories that are relevant to the user's question.

Prefer recurring patterns supported by multiple independent records.


============================================================
STRICT GROUNDING
============================================================

- Use ONLY the evidence supplied below.
- Do NOT invent organizational facts.
- Do NOT estimate incident totals from retrieved documents.
- Do NOT treat retrieved evidence as the full organizational
  population.
- Do NOT expose similarity, BM25, RRF or CrossEncoder scores.
- Do NOT add generic IT recommendations.
- Do NOT add prevention plans unless the user asked for them.
- Historical actions describe what WAS DONE.
- Do not automatically convert them into advice about what SHOULD
  BE DONE.
- If evidence is insufficient, say so.
- If sources disagree, mention the disagreement.


============================================================
SOURCE PRIORITY
============================================================

For historical incident root causes and resolutions, generally
prioritize:

1. SUPPORT
2. JIRA
3. CONFLUENCE
4. LOG
5. TEAMS
6. GIT

For documented methodology or procedure, prioritize CONFLUENCE.


============================================================
USER QUESTION
============================================================

{question}


============================================================
RETRIEVED ORGANIZATIONAL EVIDENCE
============================================================

{context}


============================================================
FINAL INSTRUCTION
============================================================

Produce a concise organizational-memory summary.

Do not include numerical population analytics.

Do not add generic recommendations.

Do not expose retrieval implementation details.
"""

    # ==============================================================
    # MIXED evidence selection
    # ==============================================================

    def _select_mixed_evidence(
        self,
        hybrid_results,
        max_results=8,
    ):
        """
        HybridSearch has already performed:

            RRF
            source diversity
            semantic deduplication
            CrossEncoder reranking

        Therefore this method does NOT apply another relevance
        threshold.

        It simply keeps a compact, high-quality evidence set for the
        MIXED LLM summarization step.
        """

        if not hybrid_results:
            return []

        selected = []
        seen = set()

        # ----------------------------------------------------------
        # First pass:
        # preserve the final CrossEncoder ordering
        # ----------------------------------------------------------

        for result in hybrid_results:

            key = self._document_key(
                result
            )

            if key in seen:
                continue

            selected.append(
                result
            )

            seen.add(
                key
            )

            if len(selected) >= max_results:
                break

        return selected

    # ==============================================================
    # Debug printer for final evidence
    # ==============================================================

    def _print_final_evidence(
        self,
        hybrid_results,
        title="Final hybrid evidence",
    ):

        print()
        print(title + ":")

        if not hybrid_results:
            print("  No evidence.")
            return

        for index, result in enumerate(
            hybrid_results,
            start=1,
        ):

            score = result.get(
                "reranker_score"
            )

            if score is None:
                score = result.get(
                    "rrf_score"
                )

            if score is None:
                score = result.get(
                    "similarity_score",
                    0.0,
                )

            try:
                score_text = (
                    f"{float(score):.4f}"
                )
            except (
                TypeError,
                ValueError,
            ):
                score_text = "N/A"

            print(
                f"  {index}. "
                f"{result.get('source_type', '')} | "
                f"{result.get('source_id', '')} | "
                f"{result.get('application', '')} | "
                f"{result.get('error_code', '')} | "
                f"{score_text}"
            )

    # ==============================================================
    # Main ask()
    # ==============================================================

    def ask(
        self,
        question,
        results_per_source=3,
    ):
        """Security-first public entry point.

        Input guardrails run BEFORE Router V4. Legitimate knowledge questions
        about passwords/credentials remain allowed; requests for actual secret
        values, prompt bypasses, destructive actions, or privilege changes are
        blocked immediately.
        """
        decision = self.security.evaluate_input(question)

        if not decision.allowed:
            print()
            print("=" * 90)
            print("SECURITY GUARDRAIL - REQUEST BLOCKED")
            print("=" * 90)
            print(f"Category: {decision.category}")
            print(f"Risk: {decision.risk_level}")
            print(f"Reason: {decision.reason}")

            self.security.audit(
                question=question,
                decision=decision,
                route="BLOCKED",
                cache_hit=False,
                sources=[],
            )

            return {
                "answer": (
                    "I can help with documented troubleshooting, authentication "
                    "procedures, credential-expiry handling, and access guidance, "
                    "but I can’t reveal passwords, API keys, tokens, private keys, "
                    "or other secret values."
                ),
                "results": [],
                "hybrid_results": [],
                "mode": "SECURITY_BLOCKED",
                "data": None,
                "chart": None,
                "sections": {},
                "routing_reasons": [decision.reason],
                "semantic_query": None,
                "cache_hit": False,
                "security_blocked": True,
                "security_category": decision.category,
                "security_risk": decision.risk_level,
            }

        # Only allowed requests reach the existing V5 pipeline.
        result = self._ask_after_guardrail(
            question=question,
            results_per_source=results_per_source,
        )

        # Context/output guardrail: redact secret VALUES while preserving the
        # structure expected by app.py. This also protects cached answers and
        # retrieved evidence shown in the UI.
        if isinstance(result, dict):
            result["answer"] = self.security.sanitize_output(
                result.get("answer", "")
            )
            result["results"] = self.security.redact_structure(
                result.get("results", [])
            )
            result["hybrid_results"] = self.security.redact_structure(
                result.get("hybrid_results", [])
            )
            result["sections"] = self.security.redact_structure(
                result.get("sections", {})
            )

        source_types = []
        for item in (result or {}).get("results", []):
            if isinstance(item, (tuple, list)) and item:
                source = item[0]
            elif isinstance(item, dict):
                source = item.get("source_type")
            else:
                source = None
            if source and source not in source_types:
                source_types.append(source)

        self.security.audit(
            question=question,
            decision=decision,
            route=(result or {}).get("mode"),
            cache_hit=(result or {}).get("cache_hit"),
            sources=source_types,
        )

        return result

    def _ask_after_guardrail(
        self,
        question,
        results_per_source=3,
    ):

        print()
        print("=" * 90)
        print("CADvPDM GPT QUESTION")
        print("=" * 90)
        print()
        print(question)

        # ==========================================================
        # Query Routing
        # ==========================================================

        plan = route_query(
            question
        )

        mode = plan.route

        print()
        print("=" * 90)
        print("QUERY ROUTER")
        print("=" * 90)
        print()
        print(
            f"Selected route: {mode}"
        )

        if getattr(
            plan,
            "reasons",
            None,
        ):

            print("Reasons:")

            for reason in plan.reasons:

                print(
                    f"  - {reason}"
                )

        # ==========================================================
        # GENERAL MODE
        #
        # IMPORTANT:
        # Return immediately.
        #
        # No Iceberg analytics.
        # No pgvector.
        # No BM25.
        # No organizational retrieval.
        # ==========================================================

        if mode == "GENERAL":

            print()
            print(
                "GENERAL question detected."
            )

            print(
                "Skipping structured analytics."
            )

            print(
                "Skipping organizational hybrid retrieval."
            )

            print(
                "Generating direct LLM answer..."
            )

            general_prompt = (
                self._build_general_prompt(
                    question
                )
            )

            answer = self.llm.generate(
                prompt=general_prompt,
                temperature=0.1,
            )

            return {
                "answer": answer,
                "results": [],
                "hybrid_results": [],
                "mode": mode,
                "data": None,
                "chart": None,
                "sections": {},
                "routing_reasons": (
                    plan.reasons
                ),
                "semantic_query": None,
            }

        # ==========================================================
        # SEMANTIC RESPONSE CACHE
        #
        # Only stable organizational-knowledge routes are cached.
        # ANALYTICS and MIXED deliberately bypass this cache.
        # ==========================================================

        if mode in ("SEMANTIC", "KNOWLEDGE"):
            cached = self.semantic_cache.lookup(
                question=question,
                route=mode,
            )

            if cached is not None:
                cached["routing_reasons"] = plan.reasons
                print()
                print(
                    "Semantic cache HIT "
                    f"(similarity={cached['cache_similarity']:.4f})"
                )
                print(
                    "Cached question:",
                    cached.get("cached_question", ""),
                )
                return cached

            print()
            print("Semantic cache MISS - continuing normal pipeline.")

        # ==========================================================
        # Structured Analytics
        # ==========================================================

        analytics_result = None

        if mode in (
            "ANALYTICS",
            "MIXED",
        ):

            print()
            print(
                "Running structured analytics..."
            )

            analytics_result = (
                self.analytics.answer(
                    question
                )
            )

        # ==========================================================
        # ANALYTICS ONLY
        # ==========================================================

        if mode == "ANALYTICS":

            if analytics_result is None:

                return {
                    "answer": (
                        "No structured analytical "
                        "result could be produced."
                    ),
                    "results": [],
                    "hybrid_results": [],
                    "mode": mode,
                    "data": None,
                    "chart": None,
                    "sections": {},
                    "routing_reasons": (
                        plan.reasons
                    ),
                    "semantic_query": None,
                }

            return {
                **analytics_result,
                "results": [],
                "hybrid_results": [],
                "mode": mode,
                "routing_reasons": (
                    plan.reasons
                ),
                "semantic_query": None,
            }

        # ==========================================================
        # SEMANTIC / KNOWLEDGE / MIXED retrieval
        # ==========================================================

        preferred_sources = (
            self._get_preferred_sources(
                plan
            )
        )

        semantic_query = (
            self._build_semantic_query(
                question,
                plan,
            )
        )

        print()
        print("=" * 90)
        print(
            "RUNNING HYBRID ORGANIZATIONAL RETRIEVAL"
        )
        print("=" * 90)

        print()
        print(
            "Preferred sources:",
            ", ".join(
                preferred_sources
            ),
        )

        print()
        print(
            "Retrieval query:",
            semantic_query,
        )

        # ----------------------------------------------------------
        # IMPORTANT:
        # This is now HybridSearch V3.
        #
        # Returned format:
        #     list[dict]
        # ----------------------------------------------------------

        hybrid_results = self._search(
            question=semantic_query,
            results_per_source=(
                results_per_source
            ),
            preferred_sources=(
                preferred_sources
            ),
        )

        self._print_final_evidence(
            hybrid_results,
            title=(
                "Evidence returned by "
                "HybridSearch V3"
            ),
        )

        # ----------------------------------------------------------
        # Compatibility conversion
        #
        # Existing context_builder.py and app.py still expect tuples.
        # ----------------------------------------------------------

        search_results = (
            self._convert_hybrid_results(
                hybrid_results
            )
        )

        # ==========================================================
        # MIXED MODE
        # ==========================================================

        if mode == "MIXED":

            # ------------------------------------------------------
            # HybridSearch already deduplicated/reranked.
            # Keep a compact evidence set for summarization.
            # ------------------------------------------------------

            selected_hybrid_results = (
                self._select_mixed_evidence(
                    hybrid_results,
                    max_results=8,
                )
            )

            selected_mixed_results = (
                self._convert_hybrid_results(
                    selected_hybrid_results
                )
            )

            mixed_context = (
                build_context(
                    selected_mixed_results
                )
            )

            self._print_final_evidence(
                selected_hybrid_results,
                title=(
                    "Selected evidence for "
                    "MIXED summarization"
                ),
            )

            print()
            print(
                "Generating historical "
                "knowledge summary..."
            )

            semantic_prompt = (
                self._build_mixed_semantic_prompt(
                    question,
                    mixed_context,
                )
            )

            semantic_summary = (
                self.llm.generate(
                    prompt=semantic_prompt,
                    temperature=0.0,
                )
            )

            print()
            print(
                "Composing deterministic "
                "MIXED answer..."
            )

            answer = compose_mixed_answer(
                analytics_result=(
                    analytics_result
                ),
                semantic_summary=(
                    semantic_summary
                ),
                search_results=(
                    selected_mixed_results
                ),
            )

            return {
                "answer": answer,

                # Legacy tuples for current frontend
                "results": (
                    selected_mixed_results
                ),

                # Rich HybridSearch dictionaries
                "hybrid_results": (
                    selected_hybrid_results
                ),

                "mode": mode,

                "data": (
                    analytics_result.get(
                        "data"
                    )
                    if analytics_result
                    else None
                ),

                "chart": (
                    analytics_result.get(
                        "chart"
                    )
                    if analytics_result
                    else None
                ),

                "sections": (
                    analytics_result.get(
                        "sections",
                        {},
                    )
                    if analytics_result
                    else {}
                ),

                "routing_reasons": (
                    plan.reasons
                ),

                "semantic_query": (
                    semantic_query
                ),
            }

        # ==========================================================
        # SEMANTIC / KNOWLEDGE MODE
        # ==========================================================

        context = build_context(
            search_results
        )

        prompt = build_rag_prompt(
            question=question,
            context=context,
            mode=mode,
            analytics_summary="",
        )

        print()
        print(
            "Generating final grounded answer..."
        )

        answer = self.llm.generate(
            prompt=prompt,
            temperature=0.0,
        )

        response = {
            "answer": answer,

            # ------------------------------------------------------
            # Existing frontend compatibility
            # ------------------------------------------------------
            "results": search_results,

            # ------------------------------------------------------
            # Rich HybridSearch results
            # ------------------------------------------------------
            "hybrid_results": hybrid_results,

            "mode": mode,
            "data": None,
            "chart": None,
            "sections": {},
            "routing_reasons": (
                plan.reasons
            ),
            "semantic_query": (
                semantic_query
            ),
            "cache_hit": False,
            "cache_similarity": None,
            "cached_question": None,
        }

        # Cache only SEMANTIC / KNOWLEDGE grounded answers.
        self.semantic_cache.store(
            question=question,
            route=mode,
            payload=response,
        )

        return response


# ==================================================================
# Standalone test
# ==================================================================

if __name__ == "__main__":

    rag = CADvPDMRAG()

    print()
    print("=" * 90)
    print("TEST 1 - GENERAL")
    print("=" * 90)

    result = rag.ask(
        "What is machine learning?"
    )

    print()
    print("MODE:")
    print(
        result.get("mode")
    )

    print()
    print("ANSWER:")
    print(
        result.get("answer")
    )

    print()
    print("=" * 90)
    print("TEST 2 - SEMANTIC")
    print("=" * 90)

    result = rag.ask(
        "How was the CAD Batch AUTH-401 "
        "issue resolved previously?"
    )

    print()
    print("MODE:")
    print(
        result.get("mode")
    )

    print()
    print("ANSWER:")
    print(
        result.get("answer")
    )

    print()
    print(
        "FINAL HYBRID EVIDENCE COUNT:"
    )

    print(
        len(
            result.get(
                "hybrid_results",
                [],
            )
        )
    )

    print()
    print("=" * 90)
    print("TEST 3 - KNOWLEDGE")
    print("=" * 90)

    result = rag.ask(
        "What is the documented procedure "
        "for investigating ENOVIA access problems?"
    )

    print()
    print("MODE:")
    print(
        result.get("mode")
    )

    print()
    print("ANSWER:")
    print(
        result.get("answer")
    )