"""Tests for HierarchicalIndex, HierarchicalSearch, and multi-query decomposition."""

import uuid
from datetime import datetime

import numpy as np
import pytest

from zerogmem import MemoryConfig, MemoryManager
from zerogmem.retriever.hierarchical_search import (
    ChunkSummary,
    HierarchicalIndex,
    HierarchicalIndexConfig,
    HierarchicalSearch,
    SessionIndex,
)


@pytest.fixture
def embed_fn():
    """Deterministic embedding function for tests."""
    def fn(text):
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(64).astype(np.float32)
    return fn


@pytest.fixture
def memory_with_sessions(embed_fn):
    """Memory manager with two sessions, each having several messages."""
    config = MemoryConfig(
        working_memory_capacity=50,
        embedding_dim=64,
        use_chunks=False,
    )
    memory = MemoryManager(config=config)
    memory.set_embedding_function(embed_fn)

    # Session 1: Travel discussion
    memory.start_session("travel_session")
    memory.add_message("Alice", "I went to Paris last summer", timestamp=datetime(2024, 6, 1))
    memory.add_message("Bob", "Nice! Did you see the Eiffel Tower?", timestamp=datetime(2024, 6, 1))
    memory.add_message("Alice", "Yes and I visited the Louvre museum", timestamp=datetime(2024, 6, 2))
    memory.add_message("Bob", "I want to go to Tokyo next year", timestamp=datetime(2024, 6, 2))
    memory.add_message("Alice", "Tokyo is amazing for food", timestamp=datetime(2024, 6, 3))
    memory.end_session()

    # Session 2: Book club
    memory.start_session("book_session")
    memory.add_message("Alice", "I just finished reading The Alchemist", timestamp=datetime(2024, 7, 1))
    memory.add_message("Bob", "That's a great book by Paulo Coelho", timestamp=datetime(2024, 7, 1))
    memory.add_message("Alice", "I'm starting Dune next", timestamp=datetime(2024, 7, 2))
    memory.add_message("Bob", "Frank Herbert is a genius", timestamp=datetime(2024, 7, 2))
    memory.end_session()

    return memory


class TestHierarchicalIndexConfig:
    def test_defaults(self):
        config = HierarchicalIndexConfig()
        assert config.top_sessions == 3
        assert config.top_chunks == 5
        assert config.top_messages == 10
        assert config.fallback_chunk_size == 10
        assert config.fallback_chunk_overlap == 4

    def test_custom(self):
        config = HierarchicalIndexConfig(top_sessions=5, top_chunks=10)
        assert config.top_sessions == 5
        assert config.top_chunks == 10


class TestHierarchicalIndexFallback:
    def test_build_creates_sessions(self, memory_with_sessions, embed_fn):
        """Build should create session entries for each session."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        assert not index.is_empty()
        assert "travel_session" in index.sessions
        assert "book_session" in index.sessions

    def test_build_creates_fallback_chunks(self, memory_with_sessions, embed_fn):
        """Without LLM chunks, should create position-based fallback chunks."""
        config = HierarchicalIndexConfig(fallback_chunk_size=3, fallback_chunk_overlap=1)
        index = HierarchicalIndex(config=config, embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        # Travel session: 5 messages with chunk_size=3, overlap=1
        # Chunk 0: msgs 0-2, chunk 1: msgs 2-4 => 2 chunks
        travel_chunks = [c for c in index.chunks.values() if c.session_id == "travel_session"]
        assert len(travel_chunks) >= 2

        # Book session: 4 messages with chunk_size=3, overlap=1
        # Chunk 0: msgs 0-2, chunk 1: msgs 2-3 => 2 chunks
        book_chunks = [c for c in index.chunks.values() if c.session_id == "book_session"]
        assert len(book_chunks) >= 1

    def test_session_embeddings_built(self, memory_with_sessions, embed_fn):
        """Session embedding matrices should be built."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        assert index._session_embeddings is not None
        assert len(index._session_ids) == 2

    def test_chunk_embeddings_built(self, memory_with_sessions, embed_fn):
        """Chunk embedding matrices should be built."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        assert index._chunk_embeddings is not None
        assert len(index._chunk_ids) > 0

    def test_is_empty_before_build(self, embed_fn):
        """Index should be empty before build."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        assert index.is_empty()


class TestHierarchicalIndexSearch:
    def test_search_sessions(self, memory_with_sessions, embed_fn):
        """Session search should return ranked results."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        query_emb = embed_fn("travel to Paris")
        results = index.search_sessions(query_emb, top_k=2)

        assert len(results) == 2
        # Results should be (SessionIndex, float_score) tuples
        for session, score in results:
            assert isinstance(session, SessionIndex)
            assert isinstance(score, float)

    def test_search_chunks_filters_by_session(self, memory_with_sessions, embed_fn):
        """Chunk search should only return chunks from specified sessions."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        query_emb = embed_fn("reading books")
        results = index.search_chunks(query_emb, ["book_session"], top_k=5)

        for chunk, score in results:
            assert chunk.session_id == "book_session"

    def test_search_chunks_empty_sessions(self, memory_with_sessions, embed_fn):
        """Searching chunks with no matching sessions should return empty."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        query_emb = embed_fn("test")
        results = index.search_chunks(query_emb, ["nonexistent_session"], top_k=5)
        assert results == []

    def test_get_messages_from_chunks(self, memory_with_sessions, embed_fn):
        """Should retrieve actual messages from chunk IDs."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        chunk_ids = list(index.chunks.keys())[:1]
        messages = index.get_messages_from_chunks(chunk_ids, memory_with_sessions)
        assert len(messages) > 0
        for msg in messages:
            assert msg.content != ""

    def test_get_messages_nonexistent_chunk(self, memory_with_sessions, embed_fn):
        """Getting messages from nonexistent chunk should return empty."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        messages = index.get_messages_from_chunks(["fake_chunk_id"], memory_with_sessions)
        assert messages == []


class TestHierarchicalSearch:
    def test_search_returns_results(self, memory_with_sessions, embed_fn):
        """Full tree-search should return scored results."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        search = HierarchicalSearch(
            index=index,
            memory=memory_with_sessions,
            embedding_fn=embed_fn,
        )

        query_emb = embed_fn("Paris travel")
        results = search.search("Paris travel", query_emb, top_k=5)

        assert len(results) > 0
        for r in results:
            assert r.source == "hierarchical"
            assert -1 <= r.score <= 1  # Cosine similarity range

    def test_search_empty_index(self, embed_fn):
        """Search on empty index should return empty."""
        config = MemoryConfig(working_memory_capacity=10, embedding_dim=64, use_chunks=False)
        memory = MemoryManager(config=config)
        memory.set_embedding_function(embed_fn)

        index = HierarchicalIndex(embedding_fn=embed_fn)
        search = HierarchicalSearch(index=index, memory=memory, embedding_fn=embed_fn)

        results = search.search("test query", embed_fn("test"), top_k=5)
        assert results == []

    def test_search_decomposed(self, memory_with_sessions, embed_fn):
        """Decomposed search should work with multiple sub-queries."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        search = HierarchicalSearch(
            index=index,
            memory=memory_with_sessions,
            embedding_fn=embed_fn,
        )

        query_emb = embed_fn("Alice book Paris")
        results = search.search_decomposed(
            query="What book did Alice read while planning Paris trip?",
            sub_queries=["Alice book reading", "Alice Paris travel"],
            query_embedding=query_emb,
            top_k=5,
        )

        assert len(results) > 0

    def test_search_decomposed_empty_sub_queries(self, memory_with_sessions, embed_fn):
        """Empty sub-queries should fall back to normal search."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        index.build_from_memory(memory_with_sessions)

        search = HierarchicalSearch(
            index=index,
            memory=memory_with_sessions,
            embedding_fn=embed_fn,
        )

        query_emb = embed_fn("test")
        results = search.search_decomposed("test", [], query_emb, top_k=5)
        assert isinstance(results, list)


class TestSubQueryGeneration:
    def test_simple_query_no_decomposition(self, memory_with_sessions, embed_fn):
        """Simple queries should not be decomposed."""
        index = HierarchicalIndex(embedding_fn=embed_fn)
        search = HierarchicalSearch(
            index=index, memory=memory_with_sessions, embedding_fn=embed_fn,
        )

        # No analysis => returns [query]
        result = search.generate_sub_queries("Where does Alice live?")
        assert result == ["Where does Alice live?"]

    def test_multi_hop_with_keywords(self, memory_with_sessions, embed_fn):
        """Multi-hop query with keywords should produce sub-queries."""
        from zerogmem.retriever.query_analyzer import ReasoningType

        index = HierarchicalIndex(embedding_fn=embed_fn)
        search = HierarchicalSearch(
            index=index, memory=memory_with_sessions, embedding_fn=embed_fn,
        )

        # Mock analysis with multi-hop type and keywords
        class MockAnalysis:
            reasoning_type = ReasoningType.MULTI_HOP
            target_entity = "Alice"
            keywords = ["book", "reading", "recovery", "injury"]
            entities = ["Alice", "John"]

        result = search.generate_sub_queries("What was Alice reading during recovery?", MockAnalysis())
        assert len(result) >= 2  # Original + at least one decomposition
        assert result[0] == "What was Alice reading during recovery?"

    def test_deduplication(self, memory_with_sessions, embed_fn):
        """Sub-queries should be deduplicated."""
        from zerogmem.retriever.query_analyzer import ReasoningType

        index = HierarchicalIndex(embedding_fn=embed_fn)
        search = HierarchicalSearch(
            index=index, memory=memory_with_sessions, embedding_fn=embed_fn,
        )

        class MockAnalysis:
            reasoning_type = ReasoningType.MULTI_HOP
            target_entity = "test"
            keywords = ["a"]  # Too few for entity-facet decomposition
            entities = ["test"]

        result = search.generate_sub_queries("test a", MockAnalysis())
        # Should not have duplicate "test"
        seen = set()
        for q in result:
            q_lower = q.lower().strip()
            assert q_lower not in seen, f"Duplicate sub-query: {q}"
            seen.add(q_lower)

    def test_max_sub_queries_capped(self, memory_with_sessions, embed_fn):
        """Sub-queries should be capped at 4."""
        from zerogmem.retriever.query_analyzer import ReasoningType

        index = HierarchicalIndex(embedding_fn=embed_fn)
        search = HierarchicalSearch(
            index=index, memory=memory_with_sessions, embedding_fn=embed_fn,
        )

        class MockAnalysis:
            reasoning_type = ReasoningType.MULTI_HOP
            target_entity = "Alice"
            keywords = ["book", "reading", "recovery", "injury", "hospital", "doctor"]
            entities = ["Alice", "Bob"]

        result = search.generate_sub_queries("complex query about many things", MockAnalysis())
        assert len(result) <= 4
