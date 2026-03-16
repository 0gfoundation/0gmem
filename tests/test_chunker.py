"""Tests for ChunkManager and hierarchical memory."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zerogmem import MemoryConfig, MemoryManager
from zerogmem.memory.chunker import Chunk, ChunkConfig, ChunkManager


@pytest.fixture
def mock_embedding_fn():
    def embed(text):
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(1536).astype(np.float32)
    return embed


@pytest.fixture
def memory_with_session(mock_embedding_fn):
    """Memory manager with a session containing messages."""
    config = MemoryConfig(
        working_memory_capacity=10,
        embedding_dim=1536,
        use_chunks=False,  # Don't auto-init chunker
    )
    memory = MemoryManager(config=config)
    memory.set_embedding_function(mock_embedding_fn)

    memory.start_session("test_session")
    for i in range(10):
        speaker = "Alice" if i % 2 == 0 else "Bob"
        memory.add_message(speaker, f"Message {i} from {speaker}", timestamp=datetime(2024, 1, 1, 10, i))
    # Don't end_session yet so we can test flush
    return memory


class TestChunkConfig:
    def test_defaults(self):
        config = ChunkConfig()
        assert config.enabled is True
        assert config.chunk_window_size == 100
        assert config.min_chunk_size == 3
        assert config.enrich_embedding_prefix is True

    def test_custom(self):
        config = ChunkConfig(chunk_window_size=50, min_chunk_size=5)
        assert config.chunk_window_size == 50
        assert config.min_chunk_size == 5


class TestEnrichContentForEmbedding:
    def test_with_all_fields(self):
        result = ChunkManager.enrich_content_for_embedding(
            "Hello world",
            speaker="Alice",
            session_id="sess_1",
            timestamp=datetime(2024, 1, 15),
        )
        assert "[Alice]" in result
        assert "Jan 15 2024" in result
        assert "Hello world" in result

    def test_speaker_only(self):
        result = ChunkManager.enrich_content_for_embedding("Hi", speaker="Bob")
        assert result == "[Bob]: Hi"

    def test_no_metadata(self):
        result = ChunkManager.enrich_content_for_embedding("Hi")
        assert result == "Hi"


class TestChunkManagerAccumulation:
    def test_on_message_added_below_threshold(self, memory_with_session, mock_embedding_fn):
        chunker = ChunkManager(
            memory=memory_with_session,
            embed_fn=mock_embedding_fn,
            config=ChunkConfig(chunk_window_size=20),
        )
        # Add 10 messages (below threshold of 20)
        for mid in list(memory_with_session.graph.memories.keys())[:10]:
            result = chunker.on_message_added(mid)
            assert result == []
        assert len(chunker._message_buffer) == 10

    def test_flush_creates_chunks(self, memory_with_session, mock_embedding_fn):
        """Flush should create a fallback chunk when LLM is unavailable."""
        chunker = ChunkManager(
            memory=memory_with_session,
            llm_client=None,  # No LLM
            embed_fn=mock_embedding_fn,
            config=ChunkConfig(chunk_window_size=100, min_chunk_size=3),
        )
        # Buffer some messages
        for mid in list(memory_with_session.graph.memories.keys())[:5]:
            chunker.on_message_added(mid)
        assert len(chunker._message_buffer) == 5

        chunks = chunker.flush()
        # Should create a fallback chunk (single chunk for the whole window)
        assert len(chunks) == 1
        assert chunks[0].summary != ""
        assert len(chunks[0].message_ids) == 5
        assert chunker._message_buffer == []

    def test_flush_skips_tiny_buffer(self, memory_with_session, mock_embedding_fn):
        """Buffer below min_chunk_size should be skipped."""
        chunker = ChunkManager(
            memory=memory_with_session,
            embed_fn=mock_embedding_fn,
            config=ChunkConfig(min_chunk_size=5),
        )
        # Only 2 messages
        mids = list(memory_with_session.graph.memories.keys())[:2]
        for mid in mids:
            chunker.on_message_added(mid)

        chunks = chunker.flush()
        assert chunks == []
        assert chunker._message_buffer == []


class TestChunkRegistration:
    def test_chunk_registered_in_graph(self, memory_with_session, mock_embedding_fn):
        """Chunks should become UnifiedMemoryItems in the graph."""
        chunker = ChunkManager(
            memory=memory_with_session,
            llm_client=None,
            embed_fn=mock_embedding_fn,
            config=ChunkConfig(chunk_window_size=100, min_chunk_size=3),
        )
        for mid in list(memory_with_session.graph.memories.keys())[:5]:
            chunker.on_message_added(mid)

        chunks = chunker.flush()
        assert len(chunks) == 1

        # The chunk should be in the graph
        chunk = chunks[0]
        assert chunk.memory_item_id is not None
        mem_item = memory_with_session.graph.memories.get(chunk.memory_item_id)
        assert mem_item is not None
        assert mem_item.source == "chunk"
        assert mem_item.metadata.get("is_chunk") is True
        assert mem_item.embedding is not None

    def test_child_messages_get_chunk_id(self, memory_with_session, mock_embedding_fn):
        """Child messages should reference their parent chunk."""
        chunker = ChunkManager(
            memory=memory_with_session,
            llm_client=None,
            embed_fn=mock_embedding_fn,
            config=ChunkConfig(chunk_window_size=100, min_chunk_size=3),
        )
        mids = list(memory_with_session.graph.memories.keys())[:5]
        for mid in mids:
            chunker.on_message_added(mid)

        chunks = chunker.flush()
        chunk = chunks[0]

        # Child messages should have chunk_id set
        for mid in chunk.message_ids:
            mem = memory_with_session.graph.memories.get(mid)
            assert mem is not None
            assert mem.chunk_id == chunk.id


class TestChunkManagerSerialization:
    def test_round_trip(self, memory_with_session, mock_embedding_fn):
        chunker = ChunkManager(
            memory=memory_with_session,
            llm_client=None,
            embed_fn=mock_embedding_fn,
            config=ChunkConfig(chunk_window_size=100),
        )
        # Create some chunks
        for mid in list(memory_with_session.graph.memories.keys())[:5]:
            chunker.on_message_added(mid)
        chunker.flush()

        # Serialize
        data = chunker.to_dict()
        assert "chunks" in data
        assert len(data["chunks"]) == 1

        # Deserialize
        restored = ChunkManager.from_dict(
            data, memory_with_session, embed_fn=mock_embedding_fn,
            config=ChunkConfig(chunk_window_size=100),
        )
        assert len(restored._chunks) == 1
        chunk = list(restored._chunks.values())[0]
        assert chunk.summary != ""
        assert len(chunk.message_ids) == 5


class TestChunkManagerLLMParsing:
    def test_parse_json_array(self):
        text = '[{"start_message": 1, "end_message": 5, "summary": "Test"}]'
        result = ChunkManager._parse_json_array(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["summary"] == "Test"

    def test_parse_json_with_markdown_fences(self):
        text = '```json\n[{"start_message": 1, "end_message": 5, "summary": "Test"}]\n```'
        result = ChunkManager._parse_json_array(text)
        assert result is not None
        assert len(result) == 1

    def test_parse_invalid_json(self):
        result = ChunkManager._parse_json_array("not json at all")
        assert result is None

    def test_parse_json_object_not_array(self):
        result = ChunkManager._parse_json_array('{"key": "value"}')
        assert result is None


class TestMemoryManagerChunkIntegration:
    def test_chunker_not_initialized_without_llm(self, mock_embedding_fn):
        """Chunker should be None when LLM client not set."""
        config = MemoryConfig(use_chunks=True)
        memory = MemoryManager(config=config)
        memory.set_embedding_function(mock_embedding_fn)
        assert memory.chunker is None

    def test_enriched_embedding_requires_chunker(self, mock_embedding_fn):
        """Without chunker, embeddings should use raw content."""
        config = MemoryConfig(use_chunks=True, enrich_embeddings=True)
        memory = MemoryManager(config=config)
        memory.set_embedding_function(mock_embedding_fn)

        memory.start_session()
        mid = memory.add_message("Alice", "I love hiking")
        memory.end_session()

        # Should still have an embedding (from raw content)
        mem = memory.graph.memories[mid]
        assert mem.embedding is not None
