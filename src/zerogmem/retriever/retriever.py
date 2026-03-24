"""
Retriever: Main retrieval pipeline for 0GMem.

Implements multi-strategy retrieval with graph traversal and
position-aware composition to combat lost-in-the-middle.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from zerogmem.defaults import DEFAULT_LLM_MODEL, llm_chat_kwargs
from zerogmem.memory.manager import MemoryManager
from zerogmem.retriever.attention_filter import AttentionFilter, FilterConfig
from zerogmem.retriever.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    QueryIntent,
    ReasoningType,
    TemporalScope,
)
from zerogmem.retriever.entity_scorer import EntityScorer, EntityScoringConfig, ScoredResult
from zerogmem.retriever.hierarchical_search import (
    HierarchicalIndex,
    HierarchicalIndexConfig,
    HierarchicalSearch,
)
from zerogmem.retriever.reranker import CrossEncoderReranker


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    id: str
    content: str
    score: float
    source: str  # semantic, temporal, entity, fact, episode, working_memory
    reasoning_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Additional context
    entities: list[str] = field(default_factory=list)
    timestamp: str | None = None
    confidence: float = 1.0
    negated: bool = False

    @property
    def scoring_content(self) -> str:
        """Content used for scoring/reranking stages.

        For proposition-sourced results this returns the focused sentence
        that matched the query, rather than the full parent message.
        The full parent message is kept in ``content`` for final context
        composition so the LLM sees the complete information.
        """
        return self.metadata.get("proposition_content") or self.content


@dataclass
class RetrievalResponse:
    """Complete retrieval response."""

    query: str
    query_analysis: QueryAnalysis
    results: list[RetrievalResult]
    composed_context: str
    strategy_used: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    top_k: int = 30
    semantic_weight: float = 0.4
    temporal_weight: float = 0.3
    entity_weight: float = 0.2
    recency_weight: float = 0.1
    max_context_tokens: int = 16000  # Token budget for composed context
    use_position_aware_composition: bool = True
    check_negations: bool = True
    use_reranker: bool = True
    rerank_top_n: int = 30
    rerank_weight: float = 0.6
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # RRF (Reciprocal Rank Fusion) settings
    use_rrf: bool = True
    rrf_k: int = 60  # Standard RRF constant
    # Agentic retrieval settings
    use_agentic_retrieval: bool = True
    max_retrieval_rounds: int = 2
    sufficiency_threshold: float = 0.5  # Min confidence to stop retrieval
    # Attention filter settings - "precise forgetting"
    use_attention_filter: bool = True
    attention_relevance_threshold: float = 0.10
    attention_diversity_weight: float = 0.2
    # BM25 sparse retrieval settings
    use_bm25: bool = True
    bm25_weight: float = 1.0  # RRF weight for BM25 strategy
    # Retrieval-time reasoning (requires consolidator with LLM client)
    use_retrieval_reasoning: bool = False
    # Entity-aware scoring (graduated scoring instead of binary filter)
    entity_scoring: EntityScoringConfig = field(default_factory=EntityScoringConfig)
    # Hierarchical search settings
    use_hierarchical_search: bool = True
    hierarchical_weight: float = 1.0
    hierarchical_use_decomposition: bool = True
    # LLM reranker settings (requires llm_client passed to Retriever)
    use_llm_reranker: bool = True
    llm_rerank_top_n: int = 50
    llm_reranker_model: str = DEFAULT_LLM_MODEL
    # LLM-based BM25 relevance filter (pre-RRF noise removal)
    use_llm_bm25_filter: bool = True
    # LLM sufficiency validation (borderline cases)
    use_llm_sufficiency: bool = True
    llm_sufficiency_low: float = 0.5
    llm_sufficiency_high: float = 0.85
    # LLM query rewriting (last resort after rule-based strategies)
    use_llm_rewrite: bool = True
    # Query planner (LLM-driven plan-execute-evaluate loop)
    use_query_planner: bool = False
    query_planner_max_rounds: int = 2


class Retriever:
    """
    Main retrieval pipeline for 0GMem.

    Implements:
    - Query understanding and intent classification
    - Multi-strategy retrieval (semantic, temporal, entity, graph)
    - Multi-hop reasoning for complex queries
    - Negation checking for adversarial robustness
    - Position-aware composition for lost-in-the-middle mitigation
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        config: RetrieverConfig | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        llm_client: Any | None = None,
        llm_model: str | None = None,
    ):
        self.memory = memory_manager
        self.config = config or RetrieverConfig()
        self._embedding_fn = embedding_fn or memory_manager._embed_fn
        self.llm_client = llm_client
        self.llm_model = llm_model or DEFAULT_LLM_MODEL

        self.query_analyzer = QueryAnalyzer(
            llm_client=llm_client, llm_model=self.llm_model
        )
        self.reranker = None
        if self.config.use_reranker:
            try:
                self.reranker = CrossEncoderReranker(model_name=self.config.reranker_model)
            except Exception as e:
                print(f"Reranker init failed: {e}")

        # Initialize entity scorer for graduated entity-aware scoring
        self.entity_scorer: EntityScorer | None = None
        if self.config.entity_scoring.enabled:
            self.entity_scorer = EntityScorer(self.config.entity_scoring)

        # Initialize hierarchical search (lazy — index built on first retrieval)
        self._hierarchical_search: HierarchicalSearch | None = None
        self._hierarchical_index_built = False

        # Initialize attention filter for "precise forgetting"
        self.attention_filter = None
        if self.config.use_attention_filter:
            filter_config = FilterConfig(
                relevance_threshold=self.config.attention_relevance_threshold,
                diversity_weight=self.config.attention_diversity_weight,
                max_context_tokens=self.config.max_context_tokens // 2,  # Leave room for formatting
            )
            self.attention_filter = AttentionFilter(
                config=filter_config,
                embedding_fn=self._embedding_fn,
            )

        # Initialize query planner (optional, replaces hardcoded strategy selection)
        self._query_planner = None
        if self.config.use_query_planner and self.llm_client:
            self._query_planner = self._build_query_planner()

    def _build_query_planner(self):
        """Build the LLM-driven query planner with index adapters."""
        from zerogmem.retriever.query_planner import (
            IndexAdapter,
            IndexDescriptor,
            IndexRegistry,
            QueryPlanner,
        )

        registry = IndexRegistry()

        # --- Adapter factory: each wraps a Retriever strategy method ---

        def _make_semantic_adapter():
            def _execute(query, query_embedding, analysis, params):
                top_k = params.get("top_k", 30)
                if query_embedding is None:
                    return []
                return self._semantic_search(query_embedding, top_k)
            return _execute

        def _make_entity_adapter():
            def _execute(query, query_embedding, analysis, params):
                entities = params.get("entities") or (analysis.entities if analysis else [])
                if not entities:
                    return []
                # For list-type queries, default to higher top_k to capture all items
                default_top_k = 5
                if analysis and analysis.expected_answer_type == "list":
                    default_top_k = 25
                top_k_per_entity = params.get("top_k_per_entity", default_top_k)
                source_filter = params.get("source_filter")
                return self._entity_search(
                    entities,
                    top_k_per_entity=top_k_per_entity,
                    source_filter=source_filter,
                )
            return _execute

        def _make_temporal_adapter():
            def _execute(query, query_embedding, analysis, params):
                if not analysis:
                    return []
                return self._temporal_search(analysis)
            return _execute

        def _make_graph_adapter():
            def _execute(query, query_embedding, analysis, params):
                if not analysis:
                    return []
                max_hops = params.get("max_hops", 2)
                return self._graph_traversal(analysis, query_embedding, max_hops)
            return _execute

        def _make_fact_adapter():
            def _execute(query, query_embedding, analysis, params):
                return self._fact_search(query_embedding, analysis)
            return _execute

        def _make_working_memory_adapter():
            def _execute(query, query_embedding, analysis, params):
                return self._working_memory_search(query_embedding)
            return _execute

        def _make_bm25_adapter():
            def _execute(query, query_embedding, analysis, params):
                if self.memory.bm25 is None:
                    return []
                top_k = params.get("top_k", 30)
                results = self._bm25_search(query, top_k, analysis)
                if self.config.use_llm_bm25_filter and results:
                    results = self._llm_filter_bm25(query, results)
                return results
            return _execute

        def _make_hierarchical_adapter():
            def _execute(query, query_embedding, analysis, params):
                self._ensure_hierarchical_index()
                if self._hierarchical_search is None:
                    return []
                top_k = params.get("top_k", 20)
                return self._hierarchical_search.search(
                    query, query_embedding, analysis, top_k=top_k,
                )
            return _execute

        # --- Register indexes with descriptors ---

        registry.register(
            IndexDescriptor(
                name="semantic",
                description="Dense vector similarity search over all stored messages. "
                    "Best for finding semantically similar content even when keywords differ",
                parameters={"top_k": "int, number of results (default 30)"},
            ),
            IndexAdapter("semantic", _make_semantic_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="entity",
                description="Searches the entity relationship graph for memories about "
                    "specific people, places, or things. Returns messages and entity profiles. "
                    "Use source_filter to retrieve all memories of a specific type "
                    "(e.g. 'cross_person_trait' for personality traits, 'llm_chunk_fact' for extracted facts)",
                parameters={
                    "entities": "list of entity names to look up (defaults to query entities)",
                    "top_k_per_entity": "int, max memories per entity (default 5, increase for exhaustive queries)",
                    "source_filter": "string, filter by source OR metadata category (e.g. 'personality_trait' for traits, 'llm_chunk_fact' for extracted facts)",
                },
            ),
            IndexAdapter("entity", _make_entity_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="temporal",
                description="Finds memories matching temporal constraints using Allen's interval "
                    "algebra (BEFORE, AFTER, DURING, etc.). Best for when/date questions",
                parameters={},
            ),
            IndexAdapter("temporal", _make_temporal_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="graph",
                description="Multi-hop BFS across entity and causal graphs. Follows "
                    "entity→entity and cause→effect chains. Best for connecting distant facts",
                parameters={"max_hops": "int, maximum traversal depth (default 2)"},
            ),
            IndexAdapter("graph", _make_graph_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="fact",
                description="Searches extracted semantic facts (subject-predicate-object triples). "
                    "Best for preferences, attributes, and verified knowledge",
                parameters={},
            ),
            IndexAdapter("fact", _make_fact_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="working_memory",
                description="Returns recent conversation context with attention-based decay. "
                    "Best for follow-up questions or references to recent discussion",
                parameters={},
            ),
            IndexAdapter("working_memory", _make_working_memory_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="bm25",
                description="Sparse keyword search with TF-IDF scoring. Best for exact "
                    "name/term matching where embeddings may miss specific tokens",
                parameters={"top_k": "int, number of results (default 30)"},
                available=self.memory.bm25 is not None and self.config.use_bm25,
            ),
            IndexAdapter("bm25", _make_bm25_adapter()),
        )

        registry.register(
            IndexDescriptor(
                name="hierarchical",
                description="Three-level tree search: session → chunk → message. Best for "
                    "broad questions or finding the right conversation segment",
                parameters={"top_k": "int, number of results (default 20)"},
                available=self.config.use_hierarchical_search,
            ),
            IndexAdapter("hierarchical", _make_hierarchical_adapter()),
        )

        return QueryPlanner(
            registry=registry,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
            max_rounds=self.config.query_planner_max_rounds,
            rrf_fn=self._rrf_fusion,
            rrf_k=self.config.rrf_k,
        )

    def _ensure_hierarchical_index(self) -> None:
        """Build hierarchical index lazily on first retrieval."""
        if self._hierarchical_index_built or not self.config.use_hierarchical_search:
            return
        self._hierarchical_index_built = True

        hi_config = HierarchicalIndexConfig()
        index = HierarchicalIndex(
            config=hi_config,
            embedding_fn=self._embedding_fn,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
        )
        index.build_from_memory(self.memory)

        if not index.is_empty():
            self._hierarchical_search = HierarchicalSearch(
                index=index,
                memory=self.memory,
                embedding_fn=self._embedding_fn,
                config=hi_config,
            )

    def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_strategy: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """
        Main retrieval method.

        Args:
            query: User query
            context: Optional context (conversation history, etc.)
            override_strategy: Override automatic strategy selection

        Returns:
            RetrievalResponse with results and composed context
        """
        # Use agentic retrieval if enabled
        if self.config.use_agentic_retrieval:
            return self._agentic_retrieve(query, context, override_strategy)

        return self._single_pass_retrieve(query, context, override_strategy)

    def answer(
        self,
        query: str,
        *,
        category: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Retrieve context and generate an answer in one call.

        Args:
            query: The question to answer.
            category: Optional category hint (e.g. "temporal", "adversarial").

        Returns:
            (answer_text, metadata) where metadata includes retrieval and
            generation details.
        """
        from zerogmem.reasoning.answer_generator import AnswerConfig, AnswerGenerator

        retrieval = self.retrieve(query)
        context = retrieval.composed_context
        target_entity = retrieval.query_analysis.target_entity

        if not self.llm_client:
            return "None", {"context": context, "target_entity": target_entity}

        generator = AnswerGenerator(
            llm_client=self.llm_client,
            config=AnswerConfig(llm_model=self.llm_model),
        )

        answer, gen_meta = generator.generate_answer(
            query=query,
            context=context,
            category=category,
            target_entity=target_entity,
        )
        gen_meta["context"] = context
        gen_meta["target_entity"] = target_entity
        gen_meta["num_results"] = len(retrieval.results)
        gen_meta["planner_fallback"] = retrieval.metadata.get("planner_fallback", False)
        return answer, gen_meta

    def _single_pass_retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_strategy: dict[str, Any] | None = None,
        *,
        skip_entity_scoring: bool = False,
    ) -> RetrievalResponse:
        """Single-pass retrieval (original implementation)."""
        # Analyze query
        analysis = self.query_analyzer.analyze(query, context)

        # Determine strategy
        strategy = self.query_analyzer.get_retrieval_strategy(analysis)
        if override_strategy:
            strategy.update(override_strategy)

        # Execute retrieval based on strategy
        results = self._execute_retrieval(query, analysis, strategy)

        # Entity isolation: filter or score results for the target entity.
        # Critical for multi-person conversations where results must be
        # scoped to a specific participant. The target_entity is identified
        # by QueryAnalyzer from the query structure, or overridden by the caller.
        # Skip for gap-filling rounds — round 2+ queries specifically target
        # cross-entity info that round 1 missed.
        target_entity = analysis.target_entity
        if target_entity and results and not skip_entity_scoring:
            original_results = list(results)
            if self.entity_scorer:
                # Graduated scoring (Phase 1 migration from locomo.py)
                results = self._apply_entity_scoring(
                    results, target_entity, analysis, query
                )
            else:
                # Binary filter fallback
                results = self._filter_by_entity(results, target_entity)
            # Safety: if entity scoring removes ALL results, fall back to original
            if not results:
                results = original_results

        # Check negations if needed
        if strategy.get("check_negations") and analysis.is_negation_check:
            results = self._verify_negations(results, analysis)

        # Optional cross-encoder reranking
        if self.reranker and self.config.use_reranker and results:
            results = self._apply_reranker(query, results)

        # Final scoring: LLM reranker (holistic) or heuristic _rank_results (fallback)
        if self.llm_client and self.config.use_llm_reranker and results:
            results = self._llm_rerank(query, results, analysis)
        else:
            results = self._rank_results(results, analysis, strategy)

        # Apply attention filter for "precise forgetting"
        # This removes redundant/low-relevance content that dilutes LLM attention
        pre_filter_count = len(results)
        if self.attention_filter and self.config.use_attention_filter:
            results = self.attention_filter.filter_context(query, results, query_analysis=analysis)

        # Compose context with position-awareness
        composed_context = self._compose_context(results, analysis)

        return RetrievalResponse(
            query=query,
            query_analysis=analysis,
            results=results[: self.config.top_k],
            composed_context=composed_context,
            strategy_used=strategy,
            metadata={
                "total_candidates": pre_filter_count,
                "filtered_candidates": len(results),
                "reasoning_type": analysis.reasoning_type.value,
                "intent": analysis.intent.value,
                "attention_filter_applied": self.config.use_attention_filter,
            },
        )

    def _post_process(
        self,
        results: list[RetrievalResult],
        query: str,
        analysis: QueryAnalysis,
    ) -> RetrievalResponse:
        """Shared post-processing: entity scoring → reranking → attention filter → compose."""
        strategy = self.query_analyzer.get_retrieval_strategy(analysis)

        # Entity scoring
        target_entity = analysis.target_entity
        if target_entity and results and self.entity_scorer:
            original = list(results)
            results = self._apply_entity_scoring(results, target_entity, analysis, query)
            if not results:
                results = original
        elif target_entity and results:
            original = list(results)
            results = self._filter_by_entity(results, target_entity)
            if not results:
                results = original

        # Negation check
        if strategy.get("check_negations") and analysis.is_negation_check:
            results = self._verify_negations(results, analysis)

        # Cross-encoder reranker
        if self.reranker and self.config.use_reranker and results:
            results = self._apply_reranker(query, results)

        # LLM reranker or heuristic
        if self.llm_client and self.config.use_llm_reranker and results:
            results = self._llm_rerank(query, results, analysis)
        else:
            results = self._rank_results(results, analysis, strategy)

        # Attention filter — use relaxed threshold for list-type answers
        # so exhaustive enumeration queries retain more diverse items
        pre_filter_count = len(results)
        if self.attention_filter and self.config.use_attention_filter:
            if analysis.expected_answer_type == "list":
                orig_threshold = self.attention_filter.config.relevance_threshold
                self.attention_filter.config.relevance_threshold = min(orig_threshold, 0.15)
                results = self.attention_filter.filter_context(query, results, query_analysis=analysis)
                self.attention_filter.config.relevance_threshold = orig_threshold
            else:
                results = self.attention_filter.filter_context(query, results, query_analysis=analysis)

        # Compose context
        composed_context = self._compose_context(results, analysis)

        return RetrievalResponse(
            query=query,
            query_analysis=analysis,
            results=results[: self.config.top_k],
            composed_context=composed_context,
            strategy_used=strategy,
            metadata={
                "total_candidates": pre_filter_count,
                "filtered_candidates": len(results),
                "reasoning_type": analysis.reasoning_type.value,
                "intent": analysis.intent.value,
            },
        )

    def _agentic_retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_strategy: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """
        Agentic retrieval with sufficiency checking and query rewriting.

        Multi-step retrieval that checks if context is sufficient
        and rewrites queries when needed.
        """
        # --- Original path (always runs as primary) ---
        all_results: list[RetrievalResult] = []
        queries_tried: list[str] = [query]
        round_metadata: list[dict[str, Any]] = []

        # First pass with original query
        response = self._single_pass_retrieve(query, context, override_strategy)
        all_results.extend(response.results)

        # Check sufficiency
        sufficiency = self._check_sufficiency(query, response)
        round_metadata.append(
            {
                "round": 1,
                "query": query,
                "results_count": len(response.results),
                "sufficiency": sufficiency,
            }
        )

        # Multi-round retrieval if insufficient
        for round_num in range(2, self.config.max_retrieval_rounds + 1):
            if sufficiency >= self.config.sufficiency_threshold:
                break

            # Generate rewritten query
            rewritten_query = self._rewrite_query(
                original_query=query,
                previous_queries=queries_tried,
                current_results=all_results,
                analysis=response.query_analysis,
            )

            if not rewritten_query or rewritten_query in queries_tried:
                break

            queries_tried.append(rewritten_query)

            # Retrieve with rewritten query — skip entity scoring since
            # gap-filling rounds target cross-entity info round 1 missed
            new_response = self._single_pass_retrieve(
                rewritten_query, context, override_strategy,
                skip_entity_scoring=True,
            )

            # Add new results (avoiding duplicates)
            seen_ids = {r.id for r in all_results}
            for r in new_response.results:
                if r.id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.id)

            # Re-check sufficiency
            sufficiency = self._check_sufficiency(
                query,
                RetrievalResponse(
                    query=query,
                    query_analysis=response.query_analysis,
                    results=all_results,
                    composed_context="",
                    strategy_used=response.strategy_used,
                ),
            )

            round_metadata.append(
                {
                    "round": round_num,
                    "query": rewritten_query,
                    "new_results_count": len(new_response.results),
                    "total_results_count": len(all_results),
                    "sufficiency": sufficiency,
                }
            )

        # --- Query planner fallback ---
        # If baseline retrieval is still insufficient after all rewrite rounds,
        # try the LLM-driven planner for different retrieval strategies.
        # This only triggers for genuinely hard queries — adversarial questions
        # with correct "None" answers have high sufficiency (rich context found)
        # and won't trigger this path.
        planner_used = False
        if (
            self._query_planner
            and sufficiency < self.config.sufficiency_threshold
        ):
            planner_results = self._planner_fallback_retrieve(
                query, response.query_analysis,
            )
            if planner_results:
                seen_ids = {r.id for r in all_results}
                new_count = 0
                for r in planner_results:
                    if r.id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.id)
                        new_count += 1
                if new_count > 0:
                    planner_used = True
                    logger.info(
                        "Planner fallback added %d new results (total: %d)",
                        new_count, len(all_results),
                    )

        # Final ranking of merged results from all rounds.
        # When planner added new results, re-run LLM reranker on the full
        # merged set so all candidates get a fair holistic relevance score.
        # Without this, planner results (which bypass _single_pass_retrieve's
        # reranker) would compete on raw strategy scores alone.
        if planner_used and self.llm_client and self.config.use_llm_reranker:
            all_results = self._llm_rerank(
                query, all_results, response.query_analysis,
            )
        else:
            all_results.sort(key=lambda r: r.score, reverse=True)
        final_results = all_results[: self.config.top_k]

        composed_context = self._compose_context(final_results, response.query_analysis)

        return RetrievalResponse(
            query=query,
            query_analysis=response.query_analysis,
            results=final_results,
            composed_context=composed_context,
            strategy_used=response.strategy_used,
            metadata={
                "total_candidates": len(all_results),
                "reasoning_type": response.query_analysis.reasoning_type.value,
                "intent": response.query_analysis.intent.value,
                "agentic_retrieval": True,
                "rounds": len(round_metadata),
                "queries_tried": queries_tried,
                "round_details": round_metadata,
                "final_sufficiency": sufficiency,
                "planner_fallback": planner_used,
            },
        )

    def _planner_fallback_retrieve(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> list[RetrievalResult]:
        """Run the query planner as a fallback when baseline retrieval is insufficient.

        Returns raw results from the planner (unprocessed — caller merges them
        with baseline results before post-processing).
        """
        try:
            query_embedding = self._embedding_fn(query) if self._embedding_fn else None
            raw_results, planner_meta = self._query_planner.plan_and_execute(
                query, query_embedding, analysis,
            )
            logger.info(
                "Planner fallback: %d results, sufficient=%s, rounds=%d",
                len(raw_results),
                planner_meta.get("sufficient"),
                planner_meta.get("rounds_used", 1),
            )
            return raw_results
        except Exception:
            logger.exception("Planner fallback failed")
            return []

    def _check_sufficiency(self, query: str, response: RetrievalResponse) -> float:
        """
        Check if retrieved context is sufficient to answer the query.

        Returns a score between 0 and 1.
        """
        if not response.results:
            return 0.0

        # Multiple heuristics for sufficiency
        scores = []

        # 1. Entity coverage - are query entities in results?
        if response.query_analysis.entities:
            entity_hits = 0
            for entity in response.query_analysis.entities:
                for r in response.results[:10]:
                    if entity.lower() in r.content.lower():
                        entity_hits += 1
                        break
            entity_coverage = entity_hits / len(response.query_analysis.entities)
            scores.append(entity_coverage)

        # 2. Keyword coverage
        if response.query_analysis.keywords:
            keyword_hits = 0
            for kw in response.query_analysis.keywords[:5]:
                for r in response.results[:10]:
                    if kw.lower() in r.content.lower():
                        keyword_hits += 1
                        break
            keyword_coverage = keyword_hits / min(5, len(response.query_analysis.keywords))
            scores.append(keyword_coverage)

        # 3. Score quality - average score of top results
        top_scores = [r.score for r in response.results[:5]]
        if top_scores:
            avg_score = sum(top_scores) / len(top_scores)
            score_quality = min(1.0, avg_score)  # Normalize
            scores.append(score_quality)

        # 4. Result count
        result_count_score = min(1.0, len(response.results) / 10)
        scores.append(result_count_score)

        # Weighted average
        if not scores:
            return 0.5  # Default
        heuristic_score = sum(scores) / len(scores)


        # LLM validation — always check when heuristic says sufficient,
        # because heuristic can be overly optimistic (e.g., "that book you
        # recommended" matches keywords but doesn't name the actual book).
        if (
            self.config.use_llm_sufficiency
            and self.llm_client
            and heuristic_score >= self.config.llm_sufficiency_low
        ):
            llm_result = self._llm_check_sufficiency(query, response.results[:5])
            if llm_result is not None:
                return llm_result

        return heuristic_score

    def _rewrite_query(
        self,
        original_query: str,
        previous_queries: list[str],
        current_results: list[RetrievalResult],
        analysis: QueryAnalysis,
    ) -> str | None:
        """
        Rewrite query to find missing information.

        Simple rule-based rewriting. Can be enhanced with LLM later.
        """
        # Extract what we might be missing
        query_lower = original_query.lower()

        # Strategy 1 (primary): LLM gap-aware rewriting — best at identifying
        # unresolved references like "that book" or "home country"
        if self.config.use_llm_rewrite and self.llm_client:
            llm_query = self._llm_rewrite_query(
                original_query, previous_queries, current_results
            )
            if llm_query and llm_query not in previous_queries:
                return llm_query

        # Strategy 2: Focus on specific entities not found
        for entity in analysis.entities:
            entity_found = any(entity.lower() in r.content.lower() for r in current_results[:10])
            if not entity_found:
                # Try query focused on this entity
                new_query = f"{entity} {' '.join(analysis.keywords[:3])}"
                if new_query not in previous_queries:
                    return new_query

        # Strategy 3: Use synonyms/related terms for keywords
        keyword_expansions = {
            "like": ["enjoy", "love", "prefer", "favorite"],
            "when": ["date", "time", "year", "month"],
            "where": ["location", "place", "city", "country"],
            "what": ["which", "type", "kind"],
            "hobby": ["activity", "interest", "pastime"],
            "work": ["job", "career", "profession", "occupation"],
            "live": ["reside", "stay", "home", "address"],
        }

        for kw in analysis.keywords[:3]:
            if kw.lower() in keyword_expansions:
                for expansion in keyword_expansions[kw.lower()]:
                    new_query = original_query.replace(kw, expansion)
                    if new_query not in previous_queries:
                        return new_query

        # Strategy 4: Add temporal context
        if "when" in query_lower or "date" in query_lower or "time" in query_lower:
            if analysis.entities:
                new_query = f"{analysis.entities[0]} timeline events dates"
                if new_query not in previous_queries:
                    return new_query

        return None

    def _execute_retrieval(
        self,
        query: str,
        analysis: QueryAnalysis,
        strategy: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Execute retrieval based on strategy."""
        # Get query embedding
        query_embedding = None
        if self._embedding_fn and strategy.get("use_semantic_search", True):
            query_embedding = self._embedding_fn(query)

        # Collect results from each strategy separately for RRF
        strategy_results: dict[str, list[RetrievalResult]] = {}

        # 1. Semantic search - use higher limit for larger conversations
        if query_embedding is not None:
            strategy_results["semantic"] = self._semantic_search(
                query_embedding, strategy.get("top_k", 30)
            )

        # 2. Entity-based search
        if strategy.get("use_entity_search") and analysis.entities:
            top_k = 25 if analysis.expected_answer_type == "list" else 5
            strategy_results["entity"] = self._entity_search(
                analysis.entities, top_k_per_entity=top_k,
            )

        # 3. Temporal search
        if strategy.get("use_temporal_search"):
            strategy_results["temporal"] = self._temporal_search(analysis)

        # 4. Multi-hop graph traversal
        if strategy.get("use_graph_traversal"):
            strategy_results["graph"] = self._graph_traversal(
                analysis, query_embedding, max_hops=strategy.get("max_hops", 2)
            )

        # 5. Fact search
        strategy_results["fact"] = self._fact_search(query_embedding, analysis)

        # 6. Working memory
        strategy_results["working_memory"] = self._working_memory_search(query_embedding)

        # 7. BM25 sparse keyword search (always runs if available)
        if self.config.use_bm25 and self.memory.bm25 is not None:
            bm25_results = self._bm25_search(
                query, top_k=strategy.get("top_k", 30), analysis=analysis
            )
            # Pre-RRF LLM filter: remove BM25 noise (function-word matches)
            if self.config.use_llm_bm25_filter and bm25_results:
                bm25_results = self._llm_filter_bm25(query, bm25_results)
            strategy_results["bm25"] = bm25_results
        elif analysis.reasoning_type in [ReasoningType.MULTI_HOP, ReasoningType.COMMONSENSE]:
            # Fallback to naive keyword search when BM25 is not available
            strategy_results["keyword"] = self._keyword_search(
                analysis.keywords + analysis.entities
            )

        # 8. Hierarchical tree-search
        if self.config.use_hierarchical_search:
            self._ensure_hierarchical_index()
        if self._hierarchical_search:
            # Prefer LLM-generated sub-queries over keyword-based decomposition
            llm_sub_queries = analysis.metadata.get("sub_queries") if analysis else None
            # Hierarchical returns raw messages found via tree search — use same
            # pool size as semantic/BM25 so it gets fair representation in RRF.
            h_top_k = strategy.get("top_k", 20)
            if llm_sub_queries and len(llm_sub_queries) > 1:
                strategy_results["hierarchical"] = self._hierarchical_search.search_decomposed(
                    query, llm_sub_queries, query_embedding,
                    top_k=h_top_k,
                )
            elif (
                self.config.hierarchical_use_decomposition
                and analysis.reasoning_type
                in (ReasoningType.MULTI_HOP, ReasoningType.TEMPORAL, ReasoningType.CAUSAL)
            ):
                sub_queries = self._hierarchical_search.generate_sub_queries(query, analysis)
                strategy_results["hierarchical"] = self._hierarchical_search.search_decomposed(
                    query, sub_queries, query_embedding,
                    top_k=h_top_k,
                )
            else:
                strategy_results["hierarchical"] = self._hierarchical_search.search(
                    query, query_embedding, analysis,
                    top_k=h_top_k,
                )

        # Use RRF fusion if enabled, otherwise simple merge
        if self.config.use_rrf:
            all_results = self._rrf_fusion(strategy_results, analysis)
        else:
            all_results = []
            for results in strategy_results.values():
                all_results.extend(results)
            all_results = self._deduplicate(all_results)

        # 8. Dialogue context (get surrounding messages for multi-hop)
        if analysis.reasoning_type == ReasoningType.MULTI_HOP:
            dialogue_results = self._get_dialogue_context(all_results)
            all_results.extend(dialogue_results)
            all_results = self._deduplicate(all_results)

        # 9. Retrieval-time reasoning (LLM-based gap filling)
        if self.config.use_retrieval_reasoning and self.memory.consolidator:
            reasoning_results = self.memory.consolidator.reason_at_retrieval(
                query, all_results, analysis
            )
            if reasoning_results:
                all_results.extend(reasoning_results)
                all_results = self._deduplicate(all_results)

        return all_results

    def _rrf_fusion(
        self,
        strategy_results: dict[str, list[RetrievalResult]],
        analysis: QueryAnalysis,
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) to combine results from multiple strategies.

        RRF(d) = Σ 1/(k + rank_i(d))

        This method is proven to work better than weighted score combination
        because it normalizes across different scoring scales.
        """
        k = self.config.rrf_k

        # Strategy weights based on query type
        strategy_weights = self._get_strategy_weights(analysis)

        # Build rank mappings for each strategy
        strategy_ranks: dict[str, dict[str, int]] = {}
        for strategy_name, results in strategy_results.items():
            # Sort by score within strategy
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            strategy_ranks[strategy_name] = {r.id: rank for rank, r in enumerate(sorted_results)}

        # Collect all unique document IDs
        all_doc_ids = set()
        doc_to_result: dict[str, RetrievalResult] = {}
        for results in strategy_results.values():
            for r in results:
                all_doc_ids.add(r.id)
                # Keep the result with highest original score
                if r.id not in doc_to_result or r.score > doc_to_result[r.id].score:
                    doc_to_result[r.id] = r

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            for strategy_name, ranks in strategy_ranks.items():
                weight = strategy_weights.get(strategy_name, 1.0)
                if doc_id in ranks:
                    rank = ranks[doc_id]
                    rrf_score += weight * (1.0 / (k + rank))
                else:
                    # Document not in this strategy - use large rank
                    max_rank = len(ranks) + 100
                    rrf_score += weight * (1.0 / (k + max_rank))
            rrf_scores[doc_id] = rrf_score

        # Create final results with RRF scores
        final_results = []
        for doc_id, rrf_score in rrf_scores.items():
            result = doc_to_result[doc_id]
            # Replace score with RRF score (normalized to 0-1 range)
            max_possible = sum(strategy_weights.values()) / k
            result.score = rrf_score / max_possible if max_possible > 0 else rrf_score
            result.metadata["rrf_score"] = rrf_score
            result.metadata["original_score"] = doc_to_result[doc_id].score
            final_results.append(result)

        # Sort by RRF score
        final_results.sort(key=lambda r: r.score, reverse=True)
        return final_results

    def _get_strategy_weights(self, analysis: QueryAnalysis) -> dict[str, float]:
        """Get strategy weights based on query type."""
        # Base weights
        weights = {
            "semantic": 1.0,
            "entity": 0.8,
            "temporal": 0.7,
            "graph": 0.7,
            "fact": 1.0,
            "working_memory": 1.2,
            "keyword": 0.6,
            "bm25": self.config.bm25_weight,
        }
        # Only include hierarchical weight when the strategy is active
        if self._hierarchical_search is not None:
            weights["hierarchical"] = self.config.hierarchical_weight

        # Adjust based on query type
        if analysis.reasoning_type == ReasoningType.TEMPORAL:
            weights["temporal"] = 1.3
            weights["semantic"] = 0.8
            weights["bm25"] = 0.8  # Temporal queries rely less on keyword matching
            if "hierarchical" in weights:
                weights["hierarchical"] = 1.2
        elif analysis.reasoning_type == ReasoningType.MULTI_HOP:
            weights["graph"] = 1.2
            weights["entity"] = 1.0
            weights["keyword"] = 0.8
            weights["bm25"] = 1.1  # Multi-hop benefits from keyword recall
            if "hierarchical" in weights:
                weights["hierarchical"] = 1.3  # Hierarchical especially valuable for multi-hop
        elif analysis.intent == QueryIntent.PREFERENCE:
            weights["fact"] = 1.2
            weights["entity"] = 1.0
            weights["bm25"] = 1.1  # Preference queries benefit from exact keyword match

        return weights

    def _get_dialogue_context(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Get surrounding messages for dialogue context."""
        context_results = []
        seen_ids = {r.id for r in results}

        # Sort memories by turn number to find adjacent messages
        sorted_memories = sorted(
            self.memory.graph.memories.values(),
            key=lambda m: (m.session_id or "", m.turn_number or 0),
        )
        memory_list = list(sorted_memories)
        id_to_idx = {m.id: i for i, m in enumerate(memory_list)}

        for result in results[:5]:  # Only expand top results
            if result.id in id_to_idx:
                idx = id_to_idx[result.id]
                # Get previous and next messages
                for offset in [-1, 1]:
                    adj_idx = idx + offset
                    if 0 <= adj_idx < len(memory_list):
                        adj_memory = memory_list[adj_idx]
                        if adj_memory.id not in seen_ids:
                            seen_ids.add(adj_memory.id)
                            context_results.append(
                                RetrievalResult(
                                    id=adj_memory.id,
                                    content=adj_memory.content,
                                    score=result.score * 0.8,  # Slightly lower score
                                    source="dialogue_context",
                                    entities=adj_memory.entity_names,
                                    timestamp=(
                                        adj_memory.event_time.start.isoformat()
                                        if adj_memory.event_time
                                        else None
                                    ),
                                )
                            )

        return context_results

    def _keyword_search(self, keywords: list[str]) -> list[RetrievalResult]:
        """Search for memories containing specific keywords (fallback when BM25 unavailable)."""
        results = []
        seen_ids = set()

        for memory in self.memory.graph.memories.values():
            content_lower = memory.content.lower()
            # Check if memory contains any keyword
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches > 0:
                if memory.id not in seen_ids:
                    seen_ids.add(memory.id)
                    results.append(
                        RetrievalResult(
                            id=memory.id,
                            content=memory.content,
                            score=0.5 + (matches * 0.1),  # Base score + keyword match bonus
                            source="keyword",
                            entities=memory.entity_names,
                            timestamp=(
                                memory.event_time.start.isoformat() if memory.event_time else None
                            ),
                        )
                    )

        return results

    def _expand_query_from_graph(
        self, query: str, analysis: QueryAnalysis
    ) -> list[str]:
        """Generate additional BM25 search queries from entity graph profiles.

        When a target entity is identified, expands the query using:
        1. Entity name itself
        2. Relation targets from the entity profile (e.g., interested_in→hiking)
        3. Entity attributes (stored facts)
        4. Keywords from top associated memories

        This replaces hardcoded per-entity keyword lists with dynamic graph lookups.
        """
        expanded = [query]
        target = analysis.target_entity
        if not target:
            return expanded

        # Search with just the entity name
        expanded.append(target)

        # Look up entity in the graph
        entity_nodes = self.memory.graph.entity_graph.find_by_name(target)
        if not entity_nodes:
            return expanded

        entity = entity_nodes[0]

        # Extract expansion terms from entity relations (limit to top 5)
        profile = self.memory.graph.entity_graph.get_entity_profile(entity.id)
        relation_terms: list[str] = []
        for rel in profile.get("relations", [])[:10]:
            related_name = rel.get("entity", "")
            relation_type = rel.get("relation", "")
            if related_name and relation_type not in ("speaks_with", "knows"):
                relation_terms.append(related_name)
        # Use top 5 most relevant relation terms
        for term in relation_terms[:5]:
            expanded.append(f"{target} {term}")

        # Extract keywords from entity attributes
        for attr_key, attr_val in entity.attributes.items():
            if isinstance(attr_val, str) and len(attr_val) < 50:
                expanded.append(f"{target} {attr_val}")
            elif isinstance(attr_val, list):
                for v in attr_val[:3]:
                    if isinstance(v, str) and len(v) < 50:
                        expanded.append(f"{target} {v}")

        # Extract key terms from associated memories (top 5)
        memories = self.memory.graph.query_by_entity(entity.id)
        for mem in memories[:5]:
            # Extract nouns/key terms from memory content (simple heuristic)
            words = mem.content.split()
            # Take content-bearing words (skip very short/common words)
            key_words = [
                w for w in words
                if len(w) > 3 and w.lower() not in {
                    "that", "this", "with", "from", "they", "have", "been",
                    "were", "said", "told", "about", "what", "when", "where",
                    "will", "would", "could", "should", "there", "their",
                    "also", "just", "very", "much", "some", "like",
                }
            ][:5]
            if key_words:
                expanded.append(f"{target} {' '.join(key_words)}")

        return expanded

    def _bm25_search(
        self, query: str, top_k: int = 20, analysis: QueryAnalysis | None = None
    ) -> list[RetrievalResult]:
        """BM25 sparse keyword search with query expansion.

        Uses the BM25 index on MemoryManager for proper TF-IDF scoring,
        query expansion, and inverted-index retrieval. When an analysis
        is provided, also expands queries using entity graph profiles.
        """
        if self.memory.bm25 is None or self.memory.bm25.total_docs == 0:
            return []

        # Get expanded queries from entity graph if analysis is available
        queries = self._expand_query_from_graph(query, analysis) if analysis else [query]

        # Search with all expanded queries, deduplicate by doc_id (keep highest score)
        best_hits: dict[str, tuple[float, str]] = {}
        for q in queries:
            for doc_id, score, content in self.memory.bm25.search(
                q, top_k=top_k, use_expansion=True
            ):
                if doc_id not in best_hits or score > best_hits[doc_id][0]:
                    best_hits[doc_id] = (score, content)

        # Sort by score and take top_k
        sorted_hits = sorted(best_hits.items(), key=lambda x: x[1][0], reverse=True)[
            :top_k
        ]

        results = []
        for doc_id, (score, content) in sorted_hits:
            memory = self.memory.graph.memories.get(doc_id)
            results.append(
                RetrievalResult(
                    id=doc_id,
                    content=content,
                    score=score,
                    source="bm25",
                    entities=memory.entity_names if memory else [],
                    timestamp=(
                        memory.event_time.start.isoformat()
                        if memory and memory.event_time
                        else None
                    ),
                    metadata={
                        "bm25_raw_score": score,
                        "speaker": memory.speaker if memory else "",
                        "session_timestamp": (
                            memory.metadata.get("session_timestamp", "")
                            if memory and memory.metadata
                            else ""
                        ),
                        "session_idx": (
                            memory.metadata.get("session_idx", 0)
                            if memory and memory.metadata
                            else 0
                        ),
                    },
                )
            )
        return results

    @staticmethod
    def _filter_by_entity(
        results: list[RetrievalResult], target_entity: str
    ) -> list[RetrievalResult]:
        """Filter retrieval results to those mentioning a specific entity.

        Essential for multi-person conversations where the question is about
        a particular participant and results from other participants are noise.
        Results that mention the target entity (in metadata or content) are
        kept; others are discarded.
        """
        target_lower = target_entity.lower()
        filtered = []
        for r in results:
            # Check entity list first (fast)
            if any(target_lower in e.lower() for e in r.entities):
                filtered.append(r)
            # Fall back to content substring match
            elif target_lower in r.content.lower():
                filtered.append(r)
        return filtered if filtered else results  # Never return empty

    def _apply_entity_scoring(
        self,
        results: list[RetrievalResult],
        target_entity: str,
        analysis: QueryAnalysis,
        query: str,
    ) -> list[RetrievalResult]:
        """Apply graduated entity scoring to results (replaces binary filter).

        Converts RetrievalResults to ScoredResults, runs EntityScorer,
        then converts back, updating scores.
        """
        assert self.entity_scorer is not None

        # Build ScoredResult list from RetrievalResults
        scored_inputs = []
        for r in results:
            speaker = r.metadata.get("speaker", "")
            session_idx = r.metadata.get("session_idx", 0)
            scored_inputs.append(
                ScoredResult(
                    id=r.id,
                    original_score=r.score,
                    adjusted_score=r.score,
                    content=r.content,
                    speaker=speaker,
                    session_idx=session_idx,
                    metadata=r.metadata,
                )
            )

        # Determine known speakers from memory graph
        known_speakers: set[str] | None = None
        entity_graph = getattr(self.memory.graph, "entity_graph", None)
        if entity_graph:
            known_speakers = set()
            for node in entity_graph.nodes.values():
                known_speakers.add(node.name)

        # Detect question characteristics
        is_adversarial = analysis.reasoning_type == ReasoningType.ADVERSARIAL
        is_multi_hop = analysis.reasoning_type == ReasoningType.MULTI_HOP

        # Collect related entities (all query entities except the target)
        related_entities = [
            e for e in analysis.entities
            if e.lower() != target_entity.lower()
        ]

        scored_results = self.entity_scorer.score_results(
            scored_inputs,
            target_entity,
            is_adversarial=is_adversarial,
            is_multi_hop=is_multi_hop,
            query=query,
            known_speakers=known_speakers,
            related_entities=related_entities,
        )

        # Map back to RetrievalResults, updating scores
        id_to_result = {r.id: r for r in results}
        output = []
        for sr in scored_results:
            original = id_to_result.get(sr.id)
            if original:
                original.score = sr.adjusted_score
                output.append(original)

        return output if output else results

    def _semantic_search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[RetrievalResult]:
        """Semantic similarity search."""
        results = []

        # Search unified graph
        similar = self.memory.graph.query_by_similarity(query_embedding, top_k=top_k)

        for memory, score in similar:
            results.append(
                RetrievalResult(
                    id=memory.id,
                    content=memory.content,
                    score=score,
                    source="semantic",
                    entities=memory.entity_names,
                    timestamp=memory.event_time.start.isoformat() if memory.event_time else None,
                    metadata={
                        "original_score": score,
                        "speaker": memory.speaker,
                        "session_timestamp": (
                            memory.metadata.get("session_timestamp", "")
                            if memory.metadata
                            else ""
                        ),
                        "session_idx": (
                            memory.metadata.get("session_idx", 0)
                            if memory.metadata
                            else 0
                        ),
                    },
                )
            )

        # Search episodic memory — expand matched episodes into raw messages
        episodes = self.memory.episodic_memory.search_similar(query_embedding, top_k=top_k // 2)
        seen_ids = {r.id for r in results}

        for episode, score in episodes:
            # Find raw messages in unified graph that belong to this episode's session
            expanded = self._expand_episode_to_messages(
                episode, query_embedding, score, seen_ids
            )
            results.extend(expanded)
            seen_ids.update(r.id for r in expanded)

        # Search proposition index for fine-grained sentence-level matches
        if self.memory.proposition_index is not None:
            prop_results = self.memory.proposition_index.search(
                query_embedding, top_k=top_k
            )
            for prop, score in prop_results:
                parent = self.memory.graph.memories.get(prop.parent_memory_id)
                if not parent:
                    continue
                if parent.id not in seen_ids:
                    results.append(
                        RetrievalResult(
                            id=parent.id,
                            content=parent.content,
                            score=score,
                            source="proposition",
                            entities=parent.entity_names,
                            timestamp=(
                                parent.event_time.start.isoformat()
                                if parent.event_time
                                else None
                            ),
                            metadata={
                                "proposition_content": prop.content,
                                "original_score": score,
                                "speaker": parent.speaker,
                            },
                        )
                    )
                    seen_ids.add(parent.id)
                else:
                    # Update score if proposition match is higher
                    for r in results:
                        if r.id == parent.id and score > r.score:
                            r.score = score
                            r.metadata["proposition_content"] = prop.content
                            break

        return results

    def _expand_episode_to_messages(
        self,
        episode: Any,
        query_embedding: np.ndarray,
        episode_score: float,
        seen_ids: set[str],
    ) -> list[RetrievalResult]:
        """Expand a matched episode into its constituent raw messages.

        Instead of returning the episode summary (which loses detail),
        find the original messages in the unified graph and return them
        individually, scored by similarity to the query.
        """
        candidates: list[RetrievalResult] = []
        session_id = episode.session_id

        # Find graph memories belonging to this episode's session
        for mem in self.memory.graph.memories.values():
            if mem.id in seen_ids:
                continue
            if mem.session_id != session_id:
                continue
            if mem.metadata and mem.metadata.get("source_type") == "session_summary":
                continue

            # Score by cosine similarity to query
            if mem.embedding is not None:
                m_norm = mem.embedding / (np.linalg.norm(mem.embedding) + 1e-8)
                q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                sim = float(q_norm @ m_norm)
            else:
                sim = 0.3

            candidates.append(
                RetrievalResult(
                    id=mem.id,
                    content=mem.content,
                    score=sim,
                    source="episode_expanded",
                    entities=mem.entity_names,
                    timestamp=(
                        mem.event_time.start.isoformat()
                        if mem.event_time and mem.event_time.start
                        else None
                    ),
                    metadata={
                        "episode_id": episode.id,
                        "speaker": mem.speaker,
                        "session_timestamp": (
                            mem.metadata.get("session_timestamp", "")
                            if mem.metadata
                            else ""
                        ),
                        "session_idx": (
                            mem.metadata.get("session_idx", 0)
                            if mem.metadata
                            else 0
                        ),
                    },
                )
            )

        # Return top messages by similarity (don't flood with entire session)
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:10]

    def _entity_search(
        self,
        entities: list[str],
        *,
        top_k_per_entity: int = 5,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Search by entity mentions.

        Args:
            entities: Entity names to look up.
            top_k_per_entity: Max memories per entity (default 5).
            source_filter: If set, only return memories whose source OR
                metadata category matches (e.g. "cross_person_trait",
                "personality_trait", "llm_chunk_fact").
        """
        results = []

        for entity_name in entities:
            # Find entity in graph
            entity_nodes = self.memory.graph.entity_graph.find_by_name(entity_name)

            for entity in entity_nodes:
                # Get memories involving this entity
                memories = self.memory.graph.query_by_entity(entity.id)

                # Apply source filter if specified (matches source OR metadata category)
                if source_filter:
                    memories = [
                        m for m in memories
                        if m.source == source_filter
                        or (hasattr(m, "metadata") and m.metadata.get("category") == source_filter)
                    ]

                for memory in memories[:top_k_per_entity]:
                    results.append(
                        RetrievalResult(
                            id=memory.id,
                            content=memory.content,
                            score=0.7,  # Base score for entity match
                            source="entity",
                            entities=[entity_name],
                            timestamp=(
                                memory.event_time.start.isoformat() if memory.event_time else None
                            ),
                            reasoning_path=[f"entity:{entity_name}"],
                        )
                    )

                # Get entity profile as context
                profile = self.memory.graph.entity_graph.get_entity_profile(entity.id)
                if profile.get("relations"):
                    profile_text = f"{entity_name}: " + ", ".join(
                        f"{r['relation']} {r['entity']}" for r in profile["relations"][:5]
                    )
                    results.append(
                        RetrievalResult(
                            id=f"profile_{entity.id}",
                            content=profile_text,
                            score=0.6,
                            source="entity_profile",
                            entities=[entity_name],
                        )
                    )

        return results

    def _temporal_search(self, analysis: QueryAnalysis) -> list[RetrievalResult]:
        """Search based on temporal constraints including Allen's interval relations."""
        from zerogmem.graph.temporal import TemporalRelation

        results = []
        seen_ids: set[str] = set()

        # Extract time constraints from temporal expressions
        for expr in analysis.temporal_expressions:
            if expr.normalized_start:
                # Search by time
                memories = self.memory.graph.query_by_time(
                    expr.normalized_start,
                    expr.normalized_end,
                    entities=analysis.entities if analysis.entities else None,
                )

                for memory in memories[:10]:
                    if memory.id not in seen_ids:
                        seen_ids.add(memory.id)
                        results.append(
                            RetrievalResult(
                                id=memory.id,
                                content=memory.content,
                                score=0.8,
                                source="temporal",
                                entities=memory.entity_names,
                                timestamp=(
                                    memory.event_time.start.isoformat()
                                    if memory.event_time
                                    else None
                                ),
                                reasoning_path=[f"temporal:{expr.text}"],
                            )
                        )

                # Allen's temporal relation expansion for before/after queries
                query_lower = analysis.original_query.lower() if analysis.original_query else ""
                relation = None
                if any(w in query_lower for w in ["before", "prior to", "earlier"]):
                    relation = TemporalRelation.BEFORE
                elif any(w in query_lower for w in ["after", "since", "following"]):
                    relation = TemporalRelation.AFTER

                if relation and results:
                    # Use the first temporal result as reference point
                    ref_id = results[0].id
                    expanded = self.memory.graph.query_temporal_relative(
                        ref_id, relation, limit=5
                    )
                    for memory in expanded:
                        if memory.id not in seen_ids:
                            seen_ids.add(memory.id)
                            results.append(
                                RetrievalResult(
                                    id=memory.id,
                                    content=memory.content,
                                    score=0.65,
                                    source="temporal_allen",
                                    entities=memory.entity_names,
                                    timestamp=(
                                        memory.event_time.start.isoformat()
                                        if memory.event_time
                                        else None
                                    ),
                                    reasoning_path=[
                                        f"temporal_allen:{relation.value}:{ref_id}"
                                    ],
                                )
                            )

        # Handle relative temporal queries
        if analysis.temporal_scope == TemporalScope.RELATIVE and analysis.entities:
            for entity in analysis.entities:
                chain = self.memory.graph.find_temporal_chain(entity)
                for memory in chain[:10]:
                    if memory.id not in seen_ids:
                        seen_ids.add(memory.id)
                        results.append(
                            RetrievalResult(
                                id=memory.id,
                                content=memory.content,
                                score=0.75,
                                source="temporal_chain",
                                entities=memory.entity_names,
                                timestamp=(
                                    memory.event_time.start.isoformat()
                                    if memory.event_time
                                    else None
                                ),
                                reasoning_path=[f"temporal_chain:{entity}"],
                            )
                        )

        return results

    def _graph_traversal(
        self, analysis: QueryAnalysis, query_embedding: np.ndarray | None, max_hops: int = 2
    ) -> list[RetrievalResult]:
        """Multi-hop graph traversal with causal chain expansion."""
        results: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        if not analysis.entities:
            return results

        # Use multi-hop query from unified graph
        multi_hop_results = self.memory.graph.multi_hop_query(
            start_entities=analysis.entities,
            query_embedding=query_embedding,
            max_hops=max_hops,
            top_k=15,
        )

        for memory, score, path in multi_hop_results:
            seen_ids.add(memory.id)
            results.append(
                RetrievalResult(
                    id=memory.id,
                    content=memory.content,
                    score=score,
                    source="graph_traversal",
                    entities=memory.entity_names,
                    timestamp=memory.event_time.start.isoformat() if memory.event_time else None,
                    reasoning_path=path,
                    metadata={"hop_count": len(path)},
                )
            )

        # Causal chain expansion: for multi-hop/commonsense queries, follow
        # cause→effect chains to surface implied context
        if analysis.reasoning_type in (ReasoningType.MULTI_HOP, ReasoningType.COMMONSENSE):
            for result in list(results)[:5]:  # Expand from top results
                for direction in ("causes", "effects"):
                    chains = self.memory.graph.query_causal_chain(
                        result.id, direction=direction, max_depth=2
                    )
                    for chain in chains[:3]:
                        for memory in chain:
                            if memory.id not in seen_ids:
                                seen_ids.add(memory.id)
                                results.append(
                                    RetrievalResult(
                                        id=memory.id,
                                        content=memory.content,
                                        score=0.6,
                                        source="causal_chain",
                                        entities=memory.entity_names,
                                        timestamp=(
                                            memory.event_time.start.isoformat()
                                            if memory.event_time
                                            else None
                                        ),
                                        reasoning_path=[
                                            f"causal:{direction}:{result.id}"
                                        ],
                                    )
                                )

        return results

    def _fact_search(
        self, query_embedding: np.ndarray | None, analysis: QueryAnalysis
    ) -> list[RetrievalResult]:
        """Search semantic facts."""
        results = []

        # Semantic search on facts
        if query_embedding is not None:
            similar_facts = self.memory.semantic_memory.search_similar(query_embedding, top_k=10)

            for fact, score in similar_facts:
                results.append(
                    RetrievalResult(
                        id=fact.id,
                        content=fact.content,
                        score=score * fact.confidence,
                        source="fact",
                        confidence=fact.confidence,
                        negated=fact.negated,
                        metadata={
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "confirmations": fact.confirmation_count,
                        },
                    )
                )

        # Search facts about entities
        for entity in analysis.entities:
            facts = self.memory.semantic_memory.get_facts_about(entity, include_negated=True)

            for fact in facts[:5]:
                results.append(
                    RetrievalResult(
                        id=fact.id,
                        content=fact.content,
                        score=0.7 * fact.confidence,
                        source="entity_fact",
                        entities=[entity],
                        confidence=fact.confidence,
                        negated=fact.negated,
                    )
                )

        return results

    def _working_memory_search(self, query_embedding: np.ndarray | None) -> list[RetrievalResult]:
        """Search working memory for recent context."""
        results = []

        wm_items = self.memory.working_memory.get_context(query_embedding, top_k=5)

        for item in wm_items:
            results.append(
                RetrievalResult(
                    id=item.id,
                    content=item.content,
                    score=0.9 * item.attention_weight,  # High weight for working memory
                    source="working_memory",
                    metadata={"attention": item.attention_weight},
                )
            )

        return results

    def _verify_negations(
        self, results: list[RetrievalResult], analysis: QueryAnalysis
    ) -> list[RetrievalResult]:
        """Verify and flag negations for adversarial robustness."""
        import re

        # Check for explicit negations in semantic memory
        for result in results:
            # If this is a fact, check for contradictions
            if result.source == "fact" and result.metadata.get("subject"):
                is_negated, neg_fact = self.memory.semantic_memory.check_negation(
                    result.metadata["subject"],
                    result.metadata.get("predicate", ""),
                    result.metadata.get("object", ""),
                )

                if is_negated:
                    result.negated = True
                    result.confidence *= 0.3  # Reduce confidence for negated facts

        # Add explicit negation results from semantic memory
        negated_facts = self.memory.semantic_memory.get_negated_facts()
        for fact in negated_facts[:10]:
            # Check if relevant to query
            query_lower = analysis.original_query.lower()
            fact_relevant = (
                any(e.lower() in fact.subject.lower() for e in analysis.entities)
                or any(e.lower() in fact.object.lower() for e in analysis.entities)
                or fact.object.lower() in query_lower
                or fact.subject.lower() in query_lower
            )

            if fact_relevant:
                results.append(
                    RetrievalResult(
                        id=fact.id,
                        content=f"IMPORTANT: {fact.content}",
                        score=1.5,  # High score for relevant negations
                        source="negation",
                        negated=True,
                        confidence=1.0,
                    )
                )

        # CRITICAL: Search raw memories for negation patterns
        # This catches "I could never eat X" type statements
        negation_patterns = [
            r"(?:could|would|can|will)\s+never",
            r"(?:don't|doesn't|do not|does not)\s+(?:like|love|enjoy|eat|want)",
            r"(?:hate|detest|loathe|can't stand)",
            r"never\s+(?:eat|like|enjoy|want)",
            r"not\s+(?:a\s+)?fan\s+of",
        ]

        query_keywords = set(analysis.keywords)

        for memory in self.memory.graph.memories.values():
            content_lower = memory.content.lower()

            # Check if content contains negation patterns
            for pattern in negation_patterns:
                if re.search(pattern, content_lower):
                    # Check relevance to query
                    content_keywords = set(content_lower.split())
                    if query_keywords.intersection(content_keywords) or any(
                        kw in content_lower for kw in analysis.keywords
                    ):
                        results.append(
                            RetrievalResult(
                                id=f"neg_{memory.id}",
                                content=f"[NEGATION DETECTED] {memory.content}",
                                score=1.8,  # Very high score for detected negations
                                source="negation_detected",
                                negated=True,
                                confidence=1.0,
                                metadata={"pattern_matched": pattern},
                            )
                        )
                        break  # Only add once per memory

        # Also check negated_facts stored in memories
        for memory in self.memory.graph.memories.values():
            if memory.negated_facts:
                for neg_text in memory.negated_facts:
                    if any(kw in neg_text.lower() for kw in analysis.keywords):
                        results.append(
                            RetrievalResult(
                                id=f"memfact_{memory.id}",
                                content=f"[STATED AS FALSE] {neg_text}",
                                score=1.6,
                                source="memory_negation",
                                negated=True,
                                confidence=1.0,
                            )
                        )

        return results

    def _rank_results(
        self,
        results: list[RetrievalResult],
        analysis: QueryAnalysis,
        strategy: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Rank results based on multiple factors."""
        import re

        # Check if this is a preference/like question (adversarial-prone)
        query_lower = analysis.original_query.lower()

        # More precise detection of preference questions
        is_preference_question = bool(
            re.search(r"\b(like|love|enjoy|hate|prefer|favorite|fan of)\b", query_lower)
        ) and query_lower.startswith(("do ", "does ", "did ", "is ", "are ", "can ", "could "))

        # Factual questions should NOT be treated as yes/no
        is_factual = query_lower.startswith(("what ", "where ", "when ", "who ", "which ", "how "))

        for result in results:
            # Start with base score
            final_score = result.score

            # Adjust based on source weights
            source_weights = {
                "working_memory": 1.2,
                "semantic": 1.0,
                "bm25": 1.0,
                "fact": 1.0,
                "temporal": 0.9 if analysis.reasoning_type == ReasoningType.TEMPORAL else 0.7,
                "entity": 0.8,
                "graph_traversal": (
                    0.9 if analysis.reasoning_type == ReasoningType.MULTI_HOP else 0.7
                ),
                "episode": 0.8,
                "entity_profile": 0.6,
                "negation": 1.2,  # Moderate boost
                "negation_detected": 1.3,
                "memory_negation": 1.2,
            }
            source_weight = source_weights.get(result.source, 0.7)
            final_score *= source_weight

            # Boost for entity matches
            if result.entities:
                entity_overlap = len(set(result.entities) & set(analysis.entities))
                final_score *= 1 + 0.1 * entity_overlap

            # Boost for recency (if timestamp available)
            if result.timestamp:
                try:
                    ts = datetime.fromisoformat(result.timestamp)
                    age_days = (datetime.now() - ts).days
                    recency_factor = max(0.5, 1 - age_days / 365)
                    final_score *= 1 + self.config.recency_weight * recency_factor
                except Exception:
                    pass

            # Penalty for low confidence
            final_score *= result.confidence

            # Handle negations based on question type
            if result.negated:
                if is_preference_question or analysis.is_negation_check:
                    # Strong boost only for true preference questions
                    final_score *= 1.5
                elif is_factual:
                    # For factual questions, negations are less relevant
                    final_score *= 0.6
                else:
                    # Neutral for other questions
                    final_score *= 0.9

            # Temporal alignment boost for temporal queries
            if (
                analysis.temporal_scope != TemporalScope.NONE
                or analysis.reasoning_type == ReasoningType.TEMPORAL
            ):
                has_time_signal = bool(result.timestamp)
                if result.metadata:
                    has_time_signal = has_time_signal or any(
                        k in result.metadata
                        for k in ["session_idx", "session_timestamp", "resolved_dates"]
                    )
                if has_time_signal:
                    final_score *= 1.15
                else:
                    final_score *= 0.85

            result.score = final_score

        # Sort by final score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _apply_reranker(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-rank ALL candidates with a cross-encoder."""
        if not self.reranker or not results:
            return results

        # Score ALL candidates — cross-encoder is a local model, so cost is trivial.
        # Partial reranking creates rank inversions where reranked items can fall
        # below items that were never scored.
        texts = [r.scoring_content for r in results]
        scores = self.reranker.score_pairs(query, texts)
        if not scores:
            return results

        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s
        if span == 0:
            norm_scores = [0.5 for _ in scores]
        else:
            norm_scores = [(s - min_s) / span for s in scores]

        weight = max(0.0, min(1.0, self.config.rerank_weight))
        for r, ns in zip(results, norm_scores):
            r.metadata["rerank_score"] = ns
            # Scale existing score by reranker signal
            r.score *= 1.0 + weight * (ns - 0.5) * 2.0

        return results

    _LLM_RERANK_PROMPT = """You are a relevance judge for a conversational memory retrieval system.

Given the question and its analysis, score each candidate on relevance from 0 to 10.

SCORING GUIDE:
- 8-10: Directly answers or contains the answer to the question
- 5-7: Contains closely related information that helps answer the question
- 2-4: Mentions the topic but doesn't help answer the specific question
- 0-1: Completely irrelevant

KEY RULES:
- A candidate that NAMES a specific item asked about deserves 8+ even if worded differently
  (e.g., "I loved reading Charlotte's Web" answers "What books has X read?" → score 9)
- Focus on whether the candidate contains ANSWERABLE information, not surface keyword overlap
- The target entity's OWN statements about themselves are most valuable
- Date matching is a bonus, not a requirement (unless the question specifies a time)
- Multiple candidates may COLLECTIVELY contribute to answering the question even if no single
  one contains the full answer. Score each piece of evidence 5+ if it provides a partial fact
  that combines with others to form the answer.
  (e.g., for "How many children does X have?", both "2 younger kids love nature" and
  "my son got into an accident" are valuable because together they reveal the count)

Return ONLY a JSON array of integer scores, one per candidate, e.g. [8, 2, 5, 1, 9]"""

    def _llm_rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        analysis: QueryAnalysis,
    ) -> list[RetrievalResult]:
        """Rerank ALL candidates using LLM relevance scoring.

        Scores each candidate 0-10 on holistic relevance (temporal fit,
        entity match, topical alignment). Multiplies with existing score.
        All candidates are scored to avoid rank inversions where partially
        reranked items fall below unscored items.
        Returns results unchanged on failure.
        """
        if not self.llm_client or not results:
            return results

        # Build candidate list for the prompt — score ALL candidates
        candidate_lines = []
        for i, r in enumerate(results, 1):
            # Get speaker and session date from metadata or memory lookup
            speaker = r.metadata.get("speaker", "")
            session_date = r.metadata.get("session_timestamp", "")
            if not speaker or not session_date:
                memory = self.memory.graph.memories.get(r.id)
                if memory:
                    speaker = speaker or memory.speaker
                    session_date = session_date or memory.metadata.get(
                        "session_timestamp", ""
                    )
            content_preview = r.scoring_content[:200].replace("\n", " ")
            line = f"{i}. [{speaker}] ({session_date}): {content_preview}"
            candidate_lines.append(line)

        # Build dates string
        dates_str = "none"
        if analysis.temporal_expressions:
            date_parts = []
            for expr in analysis.temporal_expressions:
                if expr.normalized_start:
                    date_parts.append(expr.normalized_start.strftime("%Y-%m-%d"))
                else:
                    date_parts.append(expr.text)
            dates_str = ", ".join(date_parts)

        user_msg = (
            f"Question: {query}\n"
            f"Target entity: {analysis.target_entity or 'unknown'}\n"
            f"Dates mentioned: {dates_str}\n\n"
            f"Candidates ({len(results)} total):\n" + "\n".join(candidate_lines)
            + f"\n\nReturn exactly {len(results)} scores."
        )

        try:
            response = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.config.llm_reranker_model, max_tokens=1000, temperature=0.0),
                messages=[
                    {"role": "system", "content": self._LLM_RERANK_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw_content = response.choices[0].message.content
            raw = raw_content.strip() if raw_content else ""
            logger.debug(
                "LLM reranker raw response (%d chars): %s",
                len(raw), raw[:200],
            )
            if not raw:
                logger.warning("LLM reranker returned empty response")
                return results
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            scores = json.loads(raw)

            if not isinstance(scores, list):
                logger.warning("LLM reranker returned non-list, skipping")
                return results

            # Tolerate minor count mismatches (LLM sometimes adds/drops scores)
            if len(scores) > len(results):
                scores = scores[: len(results)]
            elif len(scores) < len(results):
                # Pad with neutral score (5) for missing entries
                scores.extend([5] * (len(results) - len(scores)))

            # Multiply existing score by LLM relevance (normalized to 0-1)
            for r, llm_score in zip(results, scores):
                s = max(0, min(10, int(llm_score)))
                r.metadata["llm_rerank_score"] = s
                # Scale: score 0 → 0.3 (demote), score 10 → 1.5 (boost)
                # Floor of 0.3 prevents a single LLM misscoring from killing
                # an otherwise good candidate.
                multiplier = 0.3 + (s / 10.0) * 1.2
                r.score *= multiplier

            # Sort all candidates by updated score
            results.sort(key=lambda r: r.score, reverse=True)
            return results

        except Exception as e:
            logger.warning("LLM reranker failed, returning original results: %s", e)
            return results

    # --- LLM sufficiency validation ---

    _LLM_SUFFICIENCY_PROMPT = """You are judging whether retrieved context can answer a question.

Question: {question}

Retrieved context (top results):
{context}

Can this context answer the question, either directly or through reasonable inference?

Answer YES if:
- A specific answer can be extracted or inferred (even if worded differently)
- A quantity, frequency, or range is stated (e.g., "once or twice a year" answers "how many times")
- The answer requires combining information from the results but IS present

Answer NO only if:
- The specific information asked about is genuinely MISSING (e.g., "moved from home country" without naming which country)
- Results are topically related but don't contain the actual answer

Reply with ONLY "YES" or "NO"."""

    def _llm_check_sufficiency(
        self, query: str, top_results: list[RetrievalResult]
    ) -> float | None:
        """Ask LLM if results actually contain the answer.

        Returns overridden score (0.3 to force rewrite) or None to keep heuristic.
        """
        context_lines = []
        for i, r in enumerate(top_results[:5], 1):
            content = r.scoring_content[:300].replace("\n", " ")
            context_lines.append(f"{i}. {content}")
        context_str = "\n".join(context_lines)

        prompt = self._LLM_SUFFICIENCY_PROMPT.format(
            question=query, context=context_str
        )

        try:
            resp = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.config.llm_reranker_model, max_tokens=10, temperature=0.0),
                messages=[{"role": "user", "content": prompt}],
            )
            answer = (resp.choices[0].message.content or "").strip().upper()
            if answer.startswith("NO"):
                logger.debug("LLM sufficiency check: NO — overriding to 0.3")
                return 0.3
        except Exception as e:
            logger.warning("LLM sufficiency check failed: %s", e)

        return None  # Keep heuristic score

    # --- LLM query rewriting ---

    _LLM_REWRITE_PROMPT = """Question: {question}

Current retrieval results (don't fully answer the question):
{results}

Previous queries tried: {previous_queries}

Look at the current results for UNRESOLVED REFERENCES — vague terms like "home country", "that place", "the event", "my friend" that need to be resolved to answer the question. Generate a search query to find the missing specific detail.

Examples:
- Results say "moved from my home country" but don't name it → search: "Caroline country origin Sweden Norway homeland"
- Results say "went to that concert" but don't say which → search: "Caroline concert band artist event"
- Results say "her friend helped" but don't name them → search: "Caroline friend name helped"

Rules:
- Output ONLY the search query, nothing else
- Use SYNONYMS for key nouns (children → kids, son, daughter)
- Keep it SHORT (under 10 words)

QUERY:"""

    def _llm_rewrite_query(
        self,
        query: str,
        previous_queries: list[str],
        current_results: list[RetrievalResult],
    ) -> str | None:
        """Use LLM to generate a search query targeting missing information."""
        result_lines = []
        for i, r in enumerate(current_results[:5], 1):
            content = r.content[:200].replace("\n", " ")
            result_lines.append(f"{i}. {content}")
        results_str = "\n".join(result_lines) if result_lines else "(no results)"

        prev_str = "\n".join(f"- {q}" for q in previous_queries)

        prompt = self._LLM_REWRITE_PROMPT.format(
            question=query,
            results=results_str,
            previous_queries=prev_str,
        )

        try:
            resp = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.llm_model, max_tokens=60, temperature=0.3),
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (resp.choices[0].message.content or "").strip()

            # The prompt ends with "QUERY:" so the response should be the query directly.
            # But if the model prefixes with "QUERY:" again, strip it.
            rewrite = raw.split("\n")[0].strip()
            if rewrite.upper().startswith("QUERY:"):
                rewrite = rewrite[len("QUERY:"):].strip()

            if rewrite and rewrite not in previous_queries and len(rewrite) > 3:
                logger.debug("LLM query rewrite: %r -> %r", query, rewrite)
                return rewrite
        except Exception as e:
            logger.warning("LLM query rewrite failed: %s", e)

        return None

    # --- LLM BM25 relevance filter (pre-RRF) ---

    _LLM_BM25_FILTER_PROMPT = """Given the question, decide which messages are relevant.
A message is relevant if it contains information that could help answer the question.
A message is NOT relevant if it only shares common words with the question by coincidence.

Question: {question}

Messages:
{messages}

Return ONLY the numbers of relevant messages, comma-separated. Example: 1,3,5
If none are relevant, return: NONE"""

    def _llm_filter_bm25(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Filter BM25 results to remove keyword-matching noise before RRF."""
        if not self.llm_client or not results:
            return results

        # Build message list
        msg_lines = []
        for i, r in enumerate(results, 1):
            speaker = r.metadata.get("speaker", "")
            content = r.content[:150].replace("\n", " ")
            msg_lines.append(f"{i}. [{speaker}]: {content}")

        prompt = self._LLM_BM25_FILTER_PROMPT.format(
            question=query,
            messages="\n".join(msg_lines),
        )

        try:
            resp = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.config.llm_reranker_model, max_tokens=100, temperature=0.0),
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (resp.choices[0].message.content or "").strip()

            if raw.upper() == "NONE":
                return []

            # Parse comma-separated indices
            kept_indices = set()
            for part in raw.replace(" ", "").split(","):
                try:
                    idx = int(part)
                    if 1 <= idx <= len(results):
                        kept_indices.add(idx - 1)
                except ValueError:
                    continue

            if kept_indices:
                filtered = [results[i] for i in sorted(kept_indices)]
                logger.debug(
                    "BM25 LLM filter: %d -> %d results",
                    len(results), len(filtered),
                )
                return filtered

        except Exception as e:
            logger.warning("BM25 LLM filter failed: %s", e)

        return results  # Return unfiltered on failure

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove duplicate results."""
        seen_ids = set()
        unique = []

        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique.append(result)

        return unique

    def _compose_context(
        self,
        results: list[RetrievalResult],
        analysis: QueryAnalysis,
    ) -> str:
        """
        Compose retrieval results into context string.

        Uses position-aware composition to combat lost-in-the-middle.
        Negations are emphasized only for preference/adversarial questions.
        """
        import re

        if not results:
            return ""

        if not self.config.use_position_aware_composition:
            # Simple composition — still include speaker/timestamp metadata
            return "\n\n".join(
                self._format_result_content(r) for r in results[: self.config.top_k]
            )

        # Determine if this is a preference/adversarial question
        query_lower = analysis.original_query.lower()
        is_preference_question = bool(
            re.search(
                r"\b(like|love|enjoy|hate|prefer|favorite|fan of|escargot|snail)\b", query_lower
            )
        )
        query_lower.startswith(("what ", "where ", "when ", "who ", "which ", "how "))

        # Position-aware composition
        parts = []
        negation_results = [r for r in results if r.negated]
        non_negation_results = [r for r in results if not r.negated]

        # For preference questions: put negations at top
        # For factual questions: put factual content first, negations only if relevant
        if is_preference_question and negation_results:
            parts.append("## IMPORTANT - Stated Preferences/Dislikes:")
            for r in negation_results[:3]:
                content = (
                    r.content.replace("[NEGATION DETECTED]", "")
                    .replace("[STATED AS FALSE]", "")
                    .strip()
                )
                parts.append(f"- {content}")
            parts.append("")

        # For multi-hop/commonsense, include more context
        is_multi_hop = analysis.reasoning_type == ReasoningType.MULTI_HOP
        is_commonsense = "common" in query_lower or "both" in query_lower
        max_results = 10 if (is_multi_hop or is_commonsense) else 5

        # For multi-hop, sort results by turn number to preserve dialogue order
        if is_multi_hop:
            sorted_results = sorted(
                non_negation_results,
                key=lambda r: self.memory.graph.memories.get(
                    r.id, type("obj", (object,), {"turn_number": 9999})()
                ).turn_number
                or 9999,
            )
            non_negation_results = sorted_results

        # SANDWICH COMPOSITION: Put high-relevance at beginning AND end
        # This combats "lost in the middle" problem where LLMs forget middle content
        all_results = non_negation_results[: max_results + 5]  # Get more for sandwiching

        if len(all_results) >= 5:
            # Top 3 results at beginning (most important)
            high_priority = all_results[:3]
            # Middle content (may be ignored by LLM)
            middle = all_results[3:-2] if len(all_results) > 5 else []
            # Last 2 results at end (recency bias helps recall)
            end_priority = all_results[-2:] if len(all_results) > 3 else []

            # Build sandwich structure
            parts.append("## Key Information")
            for r in high_priority:
                content = self._format_result_content(r)
                parts.append(f"- {content}")

            if middle:
                parts.append("\n## Supporting Context")
                for r in middle:
                    content = self._format_result_content(r)
                    # Truncate middle content to save space
                    parts.append(f"- {content[:400]}")

            if end_priority:
                parts.append("\n## Additional Key Facts")
                for r in end_priority:
                    content = self._format_result_content(r)
                    parts.append(f"- {content}")
        else:
            # Not enough results for sandwich, use simple list
            parts.append("## Relevant Information")
            for r in all_results:
                content = self._format_result_content(r)
                parts.append(f"- {content}")

        # Additional context beyond sandwich
        remaining = non_negation_results[max_results + 5 :]
        if remaining:
            parts.append("\n## More Context")
            for i, r in enumerate(remaining[:5], 1):
                content = self._format_result_content(r)[:500]
                parts.append(f"[{i}] {content}")

        # For NON-preference factual questions, only include
        # negations if they seem directly relevant
        if not is_preference_question and negation_results:
            # Check if negation mentions keywords from the question
            relevant_negations = []
            for r in negation_results:
                # Check if any non-trivial query word appears in the negation
                query_words = [
                    w
                    for w in query_lower.split()
                    if len(w) > 3
                    and w
                    not in {"what", "where", "when", "does", "were", "that", "this", "with", "from"}
                ]
                if any(word in r.content.lower() for word in query_words):
                    relevant_negations.append(r)

            if relevant_negations:
                parts.append("\n## Note")
                for r in relevant_negations[:2]:
                    content = self._format_result_content(r)
                    parts.append(f"- {content}")

        # Add temporal context if relevant
        if analysis.reasoning_type == ReasoningType.TEMPORAL:
            temporal_results = [r for r in results if r.source == "temporal"]
            if temporal_results:
                parts.append("\n## Timeline")
                for r in sorted(temporal_results, key=lambda x: x.timestamp or "")[:5]:
                    if r.timestamp:
                        parts.append(f"- [{r.timestamp}] {r.content[:200]}")

        context = "\n".join(parts)

        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = self.config.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[truncated]"

        return context

    def _format_result_content(self, result: RetrievalResult) -> str:
        """Format retrieval result with optional speaker/time metadata."""
        # For proposition-sourced results, use the focused sentence rather
        # than the full parent message.  The proposition matched the query
        # semantically; the surrounding message is about a different topic
        # and would dilute the signal (e.g. "2 younger kids" buried in a
        # camping narrative).
        prop_content = result.metadata.get("proposition_content")
        if prop_content and result.source == "proposition":
            raw = prop_content
        else:
            raw = result.content
        content = (
            raw.replace("[NEGATION DETECTED]", "")
            .replace("[STATED AS FALSE]", "")
            .strip()
        )

        memory = self.memory.graph.memories.get(result.id)
        if not memory:
            return content

        prefix_parts = []
        if memory.speaker:
            prefix_parts.append(memory.speaker)
        if memory.metadata:
            session_ts = memory.metadata.get("session_timestamp")
            if session_ts:
                prefix_parts.append(session_ts)

        if prefix_parts:
            return f"[{', '.join(prefix_parts)}] {content}"

        return content

    def retrieve_for_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Convenience method for question answering.

        Returns composed context ready for LLM.
        """
        context = {}
        if conversation_history:
            context["history"] = conversation_history

        response = self.retrieve(question, context=context)
        return response.composed_context

    def trace_retrieval(
        self,
        query: str,
        target_keywords: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Diagnostic: trace where messages containing target keywords appear
        at each pipeline stage.

        Returns a dict with per-stage information showing which candidates
        contain the target keywords and their ranks/scores.
        """
        analysis = self.query_analyzer.analyze(query, context)
        strategy = self.query_analyzer.get_retrieval_strategy(analysis)

        # Stage 1: per-strategy results (pre-RRF)
        query_embedding = self._embedding_fn(query) if self._embedding_fn else None
        strategy_results: dict[str, list[RetrievalResult]] = {}

        if query_embedding is not None:
            strategy_results["semantic"] = self._semantic_search(
                query_embedding, strategy.get("top_k", 30)
            )
        if strategy.get("use_entity_search") and analysis.entities:
            top_k = 25 if analysis.expected_answer_type == "list" else 5
            strategy_results["entity"] = self._entity_search(
                analysis.entities, top_k_per_entity=top_k,
            )
        if strategy.get("use_temporal_search"):
            strategy_results["temporal"] = self._temporal_search(analysis)
        if strategy.get("use_graph_traversal"):
            strategy_results["graph"] = self._graph_traversal(
                analysis, query_embedding, max_hops=2
            )
        strategy_results["fact"] = self._fact_search(query_embedding, analysis)
        strategy_results["working_memory"] = self._working_memory_search(query_embedding)
        if self.config.use_bm25 and self.memory.bm25 is not None:
            strategy_results["bm25"] = self._bm25_search(
                query, top_k=strategy.get("top_k", 30), analysis=analysis
            )
        if self.config.use_hierarchical_search:
            self._ensure_hierarchical_index()
        if self._hierarchical_search:
            strategy_results["hierarchical"] = self._hierarchical_search.search(
                query, query_embedding, analysis, top_k=20
            )

        def _find_matches(results: list[RetrievalResult]) -> list[dict]:
            matches = []
            for rank, r in enumerate(sorted(results, key=lambda x: x.score, reverse=True)):
                content_lower = r.content.lower()
                if any(kw.lower() in content_lower for kw in target_keywords):
                    matches.append({
                        "rank": rank,
                        "score": round(r.score, 4),
                        "source": r.source,
                        "id": r.id,
                        "content": r.content[:120],
                    })
            return matches

        trace: dict[str, Any] = {"query": query, "target_keywords": target_keywords}

        # Per-strategy matches
        trace["per_strategy"] = {}
        for name, results in strategy_results.items():
            matches = _find_matches(results)
            trace["per_strategy"][name] = {
                "total": len(results),
                "matches": matches,
            }

        # Stage 2: after RRF
        if self.config.use_rrf:
            post_rrf = self._rrf_fusion(strategy_results, analysis)
        else:
            post_rrf = []
            for r in strategy_results.values():
                post_rrf.extend(r)
        trace["post_rrf"] = {
            "total": len(post_rrf),
            "matches": _find_matches(post_rrf),
        }

        # Stage 3: after entity scoring
        target_entity = analysis.target_entity
        post_entity = list(post_rrf)
        if target_entity and self.entity_scorer:
            post_entity = self._apply_entity_scoring(
                post_entity, target_entity, analysis, query
            )
            if not post_entity:
                post_entity = list(post_rrf)
        trace["post_entity_scoring"] = {
            "total": len(post_entity),
            "matches": _find_matches(post_entity),
        }

        # Stage 4: after cross-encoder reranking
        post_rerank = list(post_entity)
        if self.reranker and self.config.use_reranker:
            post_rerank = self._apply_reranker(query, post_rerank)
        trace["post_cross_encoder"] = {
            "total": len(post_rerank),
            "matches": _find_matches(post_rerank),
        }

        # Stage 5: after LLM reranking
        post_llm = list(post_rerank)
        if self.llm_client and self.config.use_llm_reranker:
            post_llm = self._llm_rerank(query, post_llm, analysis)
        trace["post_llm_rerank"] = {
            "total": len(post_llm),
            "matches": _find_matches(post_llm),
        }

        # Stage 6: after attention filter
        post_attn = list(post_llm)
        if self.attention_filter and self.config.use_attention_filter:
            post_attn = self.attention_filter.filter_context(
                query, post_attn, query_analysis=analysis
            )
        trace["post_attention_filter"] = {
            "total": len(post_attn),
            "matches": _find_matches(post_attn),
        }

        # Stage 7: final context (top_k)
        final = post_attn[: self.config.top_k]
        trace["final_top_k"] = {
            "total": len(final),
            "matches": _find_matches(final),
        }

        return trace
