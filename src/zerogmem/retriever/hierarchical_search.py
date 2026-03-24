"""
Hierarchical Search: Three-level tree-search retrieval strategy.

Searches through a session -> chunk -> message hierarchy:
  1. Match session summaries to find top sessions
  2. Match chunk summaries within selected sessions to find top chunks
  3. Retrieve messages from selected chunks, score by similarity

Supports multi-query decomposition with session-level intersection:
  Sub-queries independently identify sessions, then sessions matching
  multiple sub-queries are prioritized for deeper drill-down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from zerogmem.defaults import DEFAULT_LLM_MODEL, llm_chat_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable

    from zerogmem.graph.unified import UnifiedMemoryItem
    from zerogmem.memory.manager import MemoryManager
    from zerogmem.retriever.query_analyzer import QueryAnalysis
    from zerogmem.retriever.retriever import RetrievalResult

logger = logging.getLogger("0gmem.hierarchical_search")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalIndexConfig:
    """Configuration for hierarchical index and search."""

    top_sessions: int = 3
    top_chunks: int = 5
    top_messages: int = 10
    # Position-based chunking (always runs; LLM chunks complement these)
    fallback_chunk_size: int = 10
    fallback_chunk_overlap: int = 4


# ---------------------------------------------------------------------------
# Index structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkSummary:
    """A chunk summary in the hierarchical index."""

    id: str
    session_id: str
    message_ids: list[str] = field(default_factory=list)
    summary: str = ""
    embedding: np.ndarray | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class SessionIndex:
    """A session summary in the hierarchical index."""

    session_id: str
    summary: str = ""
    embedding: np.ndarray | None = None
    chunk_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    message_count: int = 0


# ---------------------------------------------------------------------------
# HierarchicalIndex
# ---------------------------------------------------------------------------

class HierarchicalIndex:
    """Builds and maintains the three-level index (session -> chunk -> message).

    Can operate in two modes:
    1. **LLM chunks**: Uses chunks created by ChunkManager during ingestion.
    2. **Fallback**: Creates simple position-based chunks from messages.
    """

    def __init__(
        self,
        config: HierarchicalIndexConfig | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        llm_client: Any | None = None,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.config = config or HierarchicalIndexConfig()
        self._embed_fn = embedding_fn
        self._llm_client = llm_client
        self._llm_model = llm_model

        # Index storage
        self.sessions: dict[str, SessionIndex] = {}
        self.chunks: dict[str, ChunkSummary] = {}

        # Precomputed embedding matrices for fast search
        self._session_embeddings: np.ndarray | None = None
        self._session_ids: list[str] = []
        self._chunk_embeddings: np.ndarray | None = None
        self._chunk_ids: list[str] = []

    def is_empty(self) -> bool:
        """Check if any sessions have been indexed."""
        return len(self.sessions) == 0

    def build_from_memory(self, memory: MemoryManager) -> None:
        """Build hierarchical index from all memories in the manager.

        If the memory has chunks from ChunkManager, uses those. Otherwise,
        creates fallback position-based chunks.
        """
        self.sessions.clear()
        self.chunks.clear()

        # Group memories by session_id (only raw messages, not summaries)
        session_messages: dict[str, list[UnifiedMemoryItem]] = {}
        for mem in memory.graph.memories.values():
            if mem.source == "chunk":
                continue  # Skip chunk summaries — they're indexed separately
            if mem.metadata.get("source_type") == "session_summary":
                continue  # Session summaries guide search at the session level, not in chunks
            sid = mem.session_id or "_default"
            if sid not in session_messages:
                session_messages[sid] = []
            session_messages[sid].append(mem)

        # Sort messages within each session by turn_number
        for sid in session_messages:
            session_messages[sid].sort(key=lambda m: (m.turn_number or 0))

        # Always build position-based chunks (with optional LLM summaries)
        for sid, messages in session_messages.items():
            self._index_session_fallback(sid, messages)

        # Additionally include LLM-generated chunks when available
        has_llm_chunks = memory.chunker is not None and len(memory.chunker._chunks) > 0
        if has_llm_chunks:
            self._add_llm_chunks(memory)

        # Build embedding matrices
        self._build_embedding_matrices()

        logger.info(
            "Built hierarchical index: %d sessions, %d chunks",
            len(self.sessions),
            len(self.chunks),
        )

    def _add_llm_chunks(self, memory: MemoryManager) -> None:
        """Add LLM-generated chunks alongside existing position-based chunks."""
        assert memory.chunker is not None

        for chunk in memory.chunker._chunks.values():
            sid = chunk.session_id
            # Add LLM chunk to the index (prefixed to avoid ID collisions)
            cid = f"llm_{chunk.id}"
            entity_names = [e["name"] for e in chunk.entities if e.get("name")]
            cs = ChunkSummary(
                id=cid,
                session_id=sid,
                message_ids=chunk.message_ids,
                summary=chunk.summary,
                embedding=chunk.embedding,
                entities=entity_names,
                keywords=chunk.topic_keywords,
            )
            self.chunks[cid] = cs

            # Add chunk ID to its session (session already exists from fallback)
            if sid in self.sessions:
                self.sessions[sid].chunk_ids.append(cid)
                self.sessions[sid].entities = list(
                    set(self.sessions[sid].entities) | set(entity_names)
                )

    def _summarize_chunk(self, messages: list[UnifiedMemoryItem]) -> str:
        """Summarize a chunk of messages using LLM or extractive fallback."""
        if self._llm_client and len(messages) > 1:
            try:
                return self._llm_summarize(messages)
            except Exception as e:
                logger.debug("LLM chunk summary failed, using extractive: %s", e)

        # Extractive fallback: first and last message content
        first_content = messages[0].content[:150]
        last_content = messages[-1].content[:150] if len(messages) > 1 else ""
        summary = first_content
        if last_content and last_content != first_content:
            summary = f"{first_content} ... {last_content}"
        return summary

    def _llm_summarize(self, messages: list[UnifiedMemoryItem]) -> str:
        """Use LLM to generate a concise topic summary for a chunk."""
        lines = []
        for m in messages:
            speaker = m.speaker or "Unknown"
            lines.append(f"[{speaker}]: {m.content[:200]}")
        conversation = "\n".join(lines)

        response = self._llm_client.chat.completions.create(
            **llm_chat_kwargs(self._llm_model, max_tokens=100, temperature=0.0),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the following conversation segment in 1-2 sentences. "
                        "Include the key topics, people, events, and any specific details "
                        "(names, dates, places, activities) mentioned. Be specific, not vague."
                    ),
                },
                {"role": "user", "content": conversation},
            ],
        )
        summary = (response.choices[0].message.content or "").strip()
        return summary if summary else self._extractive_fallback(messages)

    @staticmethod
    def _extractive_fallback(messages: list[UnifiedMemoryItem]) -> str:
        first_content = messages[0].content[:150]
        last_content = messages[-1].content[:150] if len(messages) > 1 else ""
        if last_content and last_content != first_content:
            return f"{first_content} ... {last_content}"
        return first_content

    def _index_session_fallback(
        self,
        session_id: str,
        messages: list[UnifiedMemoryItem],
    ) -> None:
        """Create simple position-based chunks as fallback."""
        chunk_size = self.config.fallback_chunk_size
        overlap = self.config.fallback_chunk_overlap
        chunk_ids: list[str] = []
        all_entities: set[str] = set()

        i = 0
        chunk_idx = 0
        while i < len(messages):
            end = min(i + chunk_size, len(messages))
            chunk_messages = messages[i:end]

            # Generate chunk summary
            summary = self._summarize_chunk(chunk_messages)

            entities: list[str] = []
            for m in chunk_messages:
                entities.extend(m.entity_names)
                all_entities.update(m.entity_names)
            entities = list(set(entities))

            chunk_embedding = None
            if self._embed_fn and summary:
                chunk_embedding = self._embed_fn(summary)

            cid = f"fallback_{session_id}_{chunk_idx}"
            cs = ChunkSummary(
                id=cid,
                session_id=session_id,
                message_ids=[m.id for m in chunk_messages],
                summary=summary,
                embedding=chunk_embedding,
                entities=entities,
            )
            self.chunks[cid] = cs
            chunk_ids.append(cid)
            chunk_idx += 1

            # Advance with overlap
            i = end - overlap if end < len(messages) else end

        # Session summary
        session_summary = " | ".join(
            self.chunks[cid].summary for cid in chunk_ids if self.chunks[cid].summary
        )
        # Limit session summary to ~7000 tokens (~28000 chars) before embedding
        # to stay within embedding model context limits. Keep the full text for
        # display but only embed a trimmed version.
        session_embedding = None
        max_embed_chars = 28000
        embed_text = session_summary[:max_embed_chars] if len(session_summary) > max_embed_chars else session_summary
        if self._embed_fn and embed_text:
            session_embedding = self._embed_fn(embed_text)

        self.sessions[session_id] = SessionIndex(
            session_id=session_id,
            summary=session_summary,
            embedding=session_embedding,
            chunk_ids=chunk_ids,
            entities=list(all_entities),
            message_count=len(messages),
        )

    def _build_embedding_matrices(self) -> None:
        """Precompute numpy matrices for fast cosine search."""
        # Sessions
        session_embs = []
        session_ids = []
        for sid, s in self.sessions.items():
            if s.embedding is not None:
                session_embs.append(s.embedding)
                session_ids.append(sid)
        if session_embs:
            self._session_embeddings = np.stack(session_embs)
            # Normalize for cosine similarity via dot product
            norms = np.linalg.norm(self._session_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._session_embeddings = self._session_embeddings / norms
        else:
            self._session_embeddings = None
        self._session_ids = session_ids

        # Chunks
        chunk_embs = []
        chunk_ids = []
        for cid, c in self.chunks.items():
            if c.embedding is not None:
                chunk_embs.append(c.embedding)
                chunk_ids.append(cid)
        if chunk_embs:
            self._chunk_embeddings = np.stack(chunk_embs)
            norms = np.linalg.norm(self._chunk_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._chunk_embeddings = self._chunk_embeddings / norms
        else:
            self._chunk_embeddings = None
        self._chunk_ids = chunk_ids

    def search_sessions(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[SessionIndex, float]]:
        """Cosine similarity search over session summaries."""
        if self._session_embeddings is None or len(self._session_ids) == 0:
            return []

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = self._session_embeddings @ q_norm
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sid = self._session_ids[idx]
            results.append((self.sessions[sid], float(scores[idx])))
        return results

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        session_ids: list[str],
        top_k: int = 5,
    ) -> list[tuple[ChunkSummary, float]]:
        """Cosine similarity search over chunk summaries within selected sessions."""
        if self._chunk_embeddings is None or len(self._chunk_ids) == 0:
            return []

        # Filter to chunks in selected sessions
        session_set = set(session_ids)
        valid_indices = [
            i for i, cid in enumerate(self._chunk_ids)
            if self.chunks[cid].session_id in session_set
        ]

        if not valid_indices:
            return []

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = self._chunk_embeddings @ q_norm

        # Sort valid indices by score
        scored = [(i, float(scores[i])) for i in valid_indices]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            cid = self._chunk_ids[idx]
            results.append((self.chunks[cid], score))
        return results

    def get_messages_from_chunks(
        self,
        chunk_ids: list[str],
        memory: MemoryManager,
    ) -> list[UnifiedMemoryItem]:
        """Retrieve actual messages from selected chunks."""
        seen: set[str] = set()
        messages: list[UnifiedMemoryItem] = []
        for cid in chunk_ids:
            cs = self.chunks.get(cid)
            if not cs:
                continue
            for mid in cs.message_ids:
                if mid in seen:
                    continue
                seen.add(mid)
                mem = memory.graph.memories.get(mid)
                if mem:
                    messages.append(mem)
        return messages


# ---------------------------------------------------------------------------
# HierarchicalSearch (strategy for the retriever)
# ---------------------------------------------------------------------------

class HierarchicalSearch:
    """Tree-search retrieval strategy that plugs into RRF fusion as strategy #8."""

    def __init__(
        self,
        index: HierarchicalIndex,
        memory: MemoryManager,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        config: HierarchicalIndexConfig | None = None,
    ) -> None:
        self.index = index
        self.memory = memory
        self._embed_fn = embedding_fn
        self.config = config or HierarchicalIndexConfig()

    def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        analysis: Any | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Execute hierarchical tree-search.

        Level 1: Match session summaries -> top N sessions
        Level 2: Match chunk summaries within selected sessions -> top M chunks
        Level 3: Retrieve messages from selected chunks, score by similarity
        """
        from zerogmem.retriever.retriever import RetrievalResult as RR

        if self.index.is_empty():
            return []

        if query_embedding is None and self._embed_fn:
            query_embedding = self._embed_fn(query)
        if query_embedding is None:
            return []

        # Level 1: Session matching
        top_sessions = self.index.search_sessions(
            query_embedding, top_k=self.config.top_sessions
        )
        if not top_sessions:
            return []

        session_ids = [s.session_id for s, _ in top_sessions]

        # Level 2: Chunk matching
        top_chunks = self.index.search_chunks(
            query_embedding, session_ids, top_k=self.config.top_chunks
        )
        if not top_chunks:
            return []

        chunk_ids = [c.id for c, _ in top_chunks]

        # Level 3: Message retrieval and scoring
        messages = self.index.get_messages_from_chunks(chunk_ids, self.memory)
        return self._score_messages(messages, query_embedding, top_k)

    def search_decomposed(
        self,
        query: str,
        sub_queries: list[str],
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Multi-query decomposition with session-level intersection.

        Each sub-query independently identifies top sessions. Sessions
        matching multiple sub-queries get an intersection bonus.
        Then drill into the intersection sessions.
        """
        from zerogmem.retriever.retriever import RetrievalResult as RR

        if self.index.is_empty() or not sub_queries:
            return self.search(query, query_embedding, top_k=top_k)

        if query_embedding is None and self._embed_fn:
            query_embedding = self._embed_fn(query)
        if query_embedding is None:
            return []

        # Step 1: Each sub-query identifies top sessions
        session_votes: dict[str, float] = {}
        session_query_hits: dict[str, set[int]] = {}

        for qi, sub_q in enumerate(sub_queries):
            sub_emb = self._embed_fn(sub_q) if self._embed_fn else None
            if sub_emb is None:
                continue

            sessions = self.index.search_sessions(
                sub_emb, top_k=self.config.top_sessions * 2
            )

            for session_idx_obj, score in sessions:
                sid = session_idx_obj.session_id
                session_votes[sid] = session_votes.get(sid, 0) + score
                if sid not in session_query_hits:
                    session_query_hits[sid] = set()
                session_query_hits[sid].add(qi)

        # Step 2: Score sessions by intersection coverage
        n_queries = len(sub_queries)
        scored_sessions: list[tuple[str, float]] = []
        for sid, base_score in session_votes.items():
            coverage = len(session_query_hits.get(sid, set())) / max(n_queries, 1)
            intersection_bonus = 1.0 + coverage  # 1.0-2.0x
            scored_sessions.append((sid, base_score * intersection_bonus))

        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        top_session_ids = [sid for sid, _ in scored_sessions[: self.config.top_sessions]]

        if not top_session_ids:
            return self.search(query, query_embedding, top_k=top_k)

        # Step 3: Drill into selected sessions using original query
        top_chunks = self.index.search_chunks(
            query_embedding, top_session_ids, top_k=self.config.top_chunks
        )

        chunk_ids = [c.id for c, _ in top_chunks]
        messages = self.index.get_messages_from_chunks(chunk_ids, self.memory)

        return self._score_messages(messages, query_embedding, top_k)

    def generate_sub_queries(
        self, query: str, analysis: Any | None = None
    ) -> list[str]:
        """Generate sub-queries for decomposed search.

        Only decomposes for MULTI_HOP/TEMPORAL/CAUSAL queries.
        Returns ``[query]`` unchanged for simple questions (zero extra cost).
        """
        from zerogmem.retriever.query_analyzer import ReasoningType

        sub_queries = [query]

        if analysis is None:
            return sub_queries

        # Only decompose for complex queries
        if hasattr(analysis, "reasoning_type") and analysis.reasoning_type not in (
            ReasoningType.MULTI_HOP,
            ReasoningType.TEMPORAL,
            ReasoningType.CAUSAL,
        ):
            return sub_queries

        # Strategy 1: Entity-facet decomposition
        target = getattr(analysis, "target_entity", None)
        keywords = getattr(analysis, "keywords", [])
        if target and len(keywords) >= 3:
            mid = len(keywords) // 2
            sub_queries.append(f"{target} {' '.join(keywords[:mid])}")
            sub_queries.append(f"{target} {' '.join(keywords[mid:])}")

        # Strategy 2: Entity + temporal vs entity + topic
        entities = getattr(analysis, "entities", [])
        if entities and keywords:
            sub_queries.append(f"{' '.join(entities)}")
            sub_queries.append(f"{' '.join(keywords)}")

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for q in sub_queries:
            q_norm = q.lower().strip()
            if q_norm not in seen:
                seen.add(q_norm)
                unique.append(q)

        return unique[:4]  # Cap at 4 sub-queries

    def _score_messages(
        self,
        messages: list[UnifiedMemoryItem],
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Score retrieved messages by cosine similarity to query.

        When a proposition index is available, checks if any sentence-level
        proposition gives a better similarity score than the whole message.
        """
        from zerogmem.retriever.retriever import RetrievalResult as RR

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Build a map of parent_memory_id -> best proposition score
        prop_best: dict[str, float] = {}
        prop_index = getattr(self.memory, "proposition_index", None)
        if prop_index:
            prop_results = prop_index.search(q_norm, top_k=top_k * 2)
            for prop, pscore in prop_results:
                cur = prop_best.get(prop.parent_memory_id, 0.0)
                if pscore > cur:
                    prop_best[prop.parent_memory_id] = pscore

        results: list[RR] = []

        for mem in messages:
            if mem.embedding is not None:
                m_norm = mem.embedding / (np.linalg.norm(mem.embedding) + 1e-8)
                sim = float(q_norm @ m_norm)
            else:
                sim = 0.3  # Fallback score

            # Use proposition score if it's higher
            sim = max(sim, prop_best.get(mem.id, 0.0))

            results.append(
                RR(
                    id=mem.id,
                    content=mem.content,
                    score=sim,
                    source="hierarchical",
                    entities=mem.entity_names,
                    timestamp=(
                        mem.event_time.start.isoformat()
                        if mem.event_time and mem.event_time.start
                        else None
                    ),
                    metadata={
                        "hierarchical_session": mem.session_id,
                        "chunk_id": mem.chunk_id,
                        "speaker": mem.speaker or "",
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
