"""
Memory Manager: Central orchestrator for all memory operations.

Coordinates the Unified Memory Graph, memory hierarchy,
and provides the main interface for the 0GMem system.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zerogmem.memory.chunker import ChunkManager
    from zerogmem.memory.consolidator import MemoryConsolidator
    from zerogmem.retriever.bm25_retriever import BM25Retriever
    from zerogmem.retriever.proposition_index import PropositionIndex

import re

import numpy as np

from zerogmem.defaults import DEFAULT_LLM_MODEL

from zerogmem.graph.entity import EntityNode, EntityType
from zerogmem.graph.temporal import TimeInterval
from zerogmem.graph.unified import UnifiedMemoryGraph, UnifiedMemoryItem
from zerogmem.memory.episodic import Episode, EpisodeMessage, EpisodicMemory
from zerogmem.memory.semantic import Fact, SemanticMemoryStore
from zerogmem.memory.working import WorkingMemory, WorkingMemoryItem


@dataclass
class Conversation:
    """Represents a conversation to be processed."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for memory manager."""

    working_memory_capacity: int = 20
    working_memory_decay_rate: float = 0.05
    embedding_dim: int = 3072
    auto_consolidate: bool = True
    consolidation_threshold_days: int = 7
    consolidation_min_retrievals: int = 3
    max_episodes: int = 500
    max_facts: int = 5000
    eviction_batch_size: int = 10
    use_bm25: bool = True
    # Chunk-based hierarchical memory
    use_chunks: bool = True
    chunk_window_size: int = 100
    chunk_llm_model: str = DEFAULT_LLM_MODEL
    enrich_embeddings: bool = True
    # Proposition-level indexing for fine-grained semantic search
    use_proposition_index: bool = True
    min_message_length_for_split: int = 80  # chars
    min_proposition_length: int = 20  # chars
    # Per-chunk LLM fact extraction (extracts fine-grained atomic facts with
    # pronoun resolution; subsumes regex cross-person trait extraction)
    use_llm_fact_extraction: bool = False


class MemoryManager:
    """
    Central orchestrator for the 0GMem system.

    Manages:
    - Unified Memory Graph (temporal, semantic, causal, entity)
    - Memory hierarchy (working, episodic, semantic)
    - Memory encoding (entity/relation/temporal extraction)
    - Memory consolidation and chunk-based segmentation
    - Query routing and retrieval
    """

    # Cross-person trait patterns: when speaker A compliments speaker B,
    # we extract synthesized facts like "B has personality trait: X"
    # so that retrieval for abstract queries ("personality traits") can find them.
    _CROSS_PERSON_TRAIT_PATTERNS: list[tuple[str, str | None]] = [
        # "you're so thoughtful/amazing/brave/..."
        (
            r"(?:you are|you\'re)\s+(?:so\s+|such\s+a\s+|really\s+)?"
            r"(thoughtful|amazing|incredible|inspiring|brave|strong|kind"
            r"|authentic|driven|caring|wonderful|selfless|courageous"
            r"|dedicated|creative|talented|resilient|compassionate"
            r"|generous|supportive|patient|wise|humble|honest|loyal"
            r"|passionate|determined|empathetic|adventurous)",
            None,  # trait captured in group(1)
        ),
        # "your drive/courage/strength ... is awesome"
        (
            r"your\s+(drive|courage|strength|determination|passion"
            r"|dedication|kindness|creativity|resilience|empathy"
            r"|honesty|wisdom|compassion|generosity|patience"
            r"|authenticity|thoughtfulness)\s+(?:to\s+\w+\s+)?is",
            None,
        ),
        # "you really care about being real/authentic"
        (r"you\s+(?:really\s+)?care\s+about\s+being\s+(real|authentic|honest|true)", None),
    ]

    def __init__(
        self,
        config: MemoryConfig | None = None,
        encoder_config: Any | None = None,
    ):
        self.config = config or MemoryConfig()

        # Core components
        self.graph = UnifiedMemoryGraph(embedding_dim=self.config.embedding_dim)
        self.working_memory = WorkingMemory(
            capacity=self.config.working_memory_capacity,
            decay_rate=self.config.working_memory_decay_rate,
        )
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemoryStore()

        # Internal encoder for NER, relation, temporal, negation extraction
        from zerogmem.encoder.encoder import Encoder, EncoderConfig

        self.encoder = Encoder(config=encoder_config or EncoderConfig())

        # BM25 sparse retrieval index (lazy import to avoid circular dependency)
        self.bm25: BM25Retriever | None = None
        if self.config.use_bm25:
            from zerogmem.retriever.bm25_retriever import BM25Retriever as _BM25

            self.bm25 = _BM25()

        # Proposition-level index for fine-grained semantic search
        self.proposition_index: PropositionIndex | None = None
        if self.config.use_proposition_index:
            from zerogmem.retriever.proposition_index import (
                PropositionIndex as _PropIndex,
            )

            self.proposition_index = _PropIndex(
                embedding_dim=self.config.embedding_dim
            )

        # LLM-based memory consolidation (set via set_llm_client)
        self.consolidator: MemoryConsolidator | None = None
        self._recent_message_ids: list[str] = []

        # Chunk-based hierarchical memory (set via set_llm_client)
        self.chunker: ChunkManager | None = None

        # Session tracking
        self.current_session_id: str | None = None
        self.current_episode: Episode | None = None

        # Recent entity mentions for pronoun resolution (most recent first)
        self._recent_entity_mentions: list[str] = []

        # LLM client reference (set via set_llm_client)
        self._llm_client: Any | None = None

        # Embedding function (to be set by encoder)
        self._embed_fn: Callable[[str], np.ndarray] | None = None
        # Batch embedding function for efficient bulk operations
        self._batch_embed_fn: Callable[[list[str]], list[np.ndarray]] | None = None

    def set_embedding_function(
        self,
        embed_fn: Callable[[str], np.ndarray],
        batch_embed_fn: Callable[[list[str]], list[np.ndarray]] | None = None,
    ) -> None:
        """Set the embedding function for memory encoding."""
        self._embed_fn = embed_fn
        self._batch_embed_fn = batch_embed_fn
        self.encoder._embedding_fn = embed_fn

    def set_llm_client(self, llm_client: Any) -> None:
        """Set the LLM client for all LLM-powered features.

        Which features are activated depends on ``MemoryConfig``:
        - ``auto_consolidate``: LLM-based memory consolidation
        - ``use_chunks``: chunk-based topic segmentation

        LLM knowledge extraction (entity backfill, pronoun resolution)
        is always enabled when an LLM client is available.

        Args:
            llm_client: An OpenAI-compatible client instance.
        """
        self._llm_client = llm_client

        if self.config.auto_consolidate:
            from zerogmem.memory.consolidator import ConsolidatorConfig, MemoryConsolidator

            self.consolidator = MemoryConsolidator(self, llm_client, ConsolidatorConfig())

        if self.config.use_chunks:
            from zerogmem.memory.chunker import ChunkConfig
            from zerogmem.memory.chunker import ChunkManager as _ChunkManager

            self.chunker = _ChunkManager(
                memory=self,
                llm_client=llm_client,
                embed_fn=self._embed_fn,
                batch_embed_fn=self._batch_embed_fn,
                config=ChunkConfig(
                    chunk_window_size=self.config.chunk_window_size,
                    llm_model=self.config.chunk_llm_model,
                    enrich_embedding_prefix=self.config.enrich_embeddings,
                    use_llm_fact_extraction=self.config.use_llm_fact_extraction,
                ),
            )

    def enable_bm25(self) -> None:
        """Enable BM25 index and populate it from existing memories."""
        from zerogmem.retriever.bm25_retriever import BM25Retriever as _BM25

        if self.bm25 is None:
            self.bm25 = _BM25()
        else:
            self.bm25.clear()
        for memory in self.graph.memories.values():
            self.bm25.add_document(
                doc_id=memory.id,
                content=memory.content,
                metadata={
                    "speaker": memory.speaker,
                    "session_id": memory.session_id,
                },
            )

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new conversation session."""
        self.current_session_id = session_id or str(uuid.uuid4())

        # Create new episode for this session
        self.current_episode = Episode(
            session_id=self.current_session_id,
            start_time=datetime.now(),
        )

        return self.current_session_id

    def end_session(self) -> str | None:
        """End the current session and finalize the episode."""
        if self.current_episode:
            self.current_episode.end_time = datetime.now()

            # Flush remaining chunks before consolidation
            if self.chunker:
                self.chunker.flush()

            # Reset pronoun resolution state for next session
            self._recent_entity_mentions.clear()

            # Final consolidation for remaining messages
            if self.consolidator and self._recent_message_ids:
                self.consolidator.consolidate(self._recent_message_ids)
                self._recent_message_ids.clear()

            # Decay unvalidated hypotheses
            if self.consolidator:
                self.consolidator.decay_hypotheses()

            # Generate summary if we have an embedding function
            if self._embed_fn and self.current_episode.messages:
                full_text = self.current_episode.get_full_text()
                self.current_episode.full_embedding = self._embed_fn(full_text)

            # Add to episodic memory
            self.episodic_memory.add_episode(self.current_episode)

            # Enforce capacity limits
            removed_episodes = self.episodic_memory.enforce_capacity(self.config.max_episodes)
            for _, session_id in removed_episodes:
                if session_id:
                    self._cascade_remove_by_session(session_id)

            self.semantic_memory.enforce_capacity(self.config.max_facts)

            episode_id = self.current_episode.id
            self.current_episode = None
            self.current_session_id = None
            return episode_id

        return None

    def _cascade_remove_by_session(self, session_id: str) -> None:
        """Remove unified memories linked to an evicted session.

        Finds memories whose session_id matches and removes them
        from the unified graph and BM25 index.
        """
        to_remove = [
            mid for mid, mem in self.graph.memories.items() if mem.session_id == session_id
        ]
        for mid in to_remove:
            self.graph.remove_memory(mid)
            if self.bm25 is not None:
                self.bm25.remove_document(mid)

    # ------------------------------------------------------------------
    # Entity enrichment (known-entity injection + pronoun resolution)
    # ------------------------------------------------------------------

    _THIRD_PERSON_PRONOUNS = re.compile(
        r"\b(she|he|her|him|his|hers|they|them|their|theirs)\b",
        re.IGNORECASE,
    )

    def _inject_known_entities(self, item: UnifiedMemoryItem) -> None:
        """Scan text for entity names already in the registry.

        Catches names that regex NER missed (e.g. appositive phrases
        like ``this woman, Jean, who...``).
        """
        text_lower = item.content.lower()
        for ename in list(self.graph.entity_graph._name_index):
            if len(ename) < 3:
                continue
            if ename in text_lower:
                if re.search(r"\b" + re.escape(ename) + r"\b", text_lower):
                    nodes = self.graph.entity_graph._name_index[ename]
                    for nid in nodes:
                        node = self.graph.entity_graph.nodes.get(nid)
                        if node and node.name not in item.entity_names:
                            item.entity_names.append(node.name)

    def _resolve_pronouns(self, item: UnifiedMemoryItem) -> None:
        """Tag pronoun-using messages with recently mentioned entities.

        If a message uses third-person pronouns (she/he/they) but has no
        non-speaker entity names, assign the most recently mentioned
        entities so that downstream entity scoring can find them.
        """
        speaker_lower = (item.speaker or "").lower()
        non_speaker = [
            e for e in item.entity_names
            if e.lower() != speaker_lower
            and e.lower() not in ("user", "assistant", "unknown", "summary")
        ]
        if non_speaker:
            # Update recent mentions (most recent first)
            self._recent_entity_mentions = non_speaker + [
                e for e in self._recent_entity_mentions if e not in non_speaker
            ]
            self._recent_entity_mentions = self._recent_entity_mentions[:5]
        elif self._recent_entity_mentions and self._THIRD_PERSON_PRONOUNS.search(item.content):
            for ent in self._recent_entity_mentions[:2]:
                if ent not in item.entity_names:
                    item.entity_names.append(ent)

    def add_message(
        self,
        speaker: str,
        content: str,
        timestamp: datetime | None = None,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
        reference_time: datetime | None = None,
    ) -> str:
        """Add a message to the current session.

        Runs the full pipeline: encoding (NER, relations, temporal,
        negation extraction), entity enrichment (known-entity injection,
        pronoun resolution), BM25 indexing, entity/relation/negation
        registration, working memory, consolidation, and chunk accumulation.

        Args:
            speaker: Who said this message.
            content: The message text.
            timestamp: When it was said (defaults to now).
            entities: Extra entity names to attach (merged with NER results).
            metadata: Additional metadata dict.
            embedding: Pre-computed embedding. If provided, the encoder
                       will not generate one (useful for batch embedding).
            reference_time: Reference time for temporal expression resolution.

        Returns:
            The memory ID for the stored message.
        """
        timestamp = timestamp or datetime.now()

        # Create episode message
        message = EpisodeMessage(
            speaker=speaker,
            content=content,
            timestamp=timestamp,
            entities_mentioned=entities or [],
            metadata=metadata or {},
        )
        if self.current_episode:
            self.current_episode.add_message(message)

        # Encode: NER, relations, temporal, negations, importance
        # Skip embedding generation when caller provides a pre-computed one
        # OR when the manager will generate an enriched embedding itself.
        will_enrich = (
            embedding is None
            and self.chunker
            and self.chunker.config.enrich_embedding_prefix
            and self._embed_fn is not None
        )
        encoding_result = self.encoder.encode(
            text=content,
            speaker=speaker,
            timestamp=timestamp,
            session_id=self.current_session_id,
            reference_time=reference_time or timestamp,
            metadata=metadata or {},
            skip_embedding=embedding is not None or will_enrich,
        )

        # Use pre-computed embedding if provided
        if embedding is not None:
            encoding_result.memory_item.embedding = embedding

        # Speaker-enriched embedding (if no pre-computed and chunker wants it)
        elif self.chunker and self.chunker.config.enrich_embedding_prefix and self._embed_fn:
            from zerogmem.memory.chunker import ChunkManager as _CM

            enriched = _CM.enrich_content_for_embedding(
                content, speaker, self.current_session_id, timestamp
            )
            encoding_result.memory_item.embedding = self._embed_fn(enriched)

        # Merge caller-provided entity names
        if entities:
            for e in entities:
                if e not in encoding_result.memory_item.entity_names:
                    encoding_result.memory_item.entity_names.append(e)

        # Core pipeline: graph insertion, BM25, entities, relations, negations, chunks
        item = encoding_result.memory_item
        self.add_encoding_result(encoding_result, session_id=self.current_session_id)

        # Proposition-level splitting for fine-grained semantic search
        if (
            self.proposition_index is not None
            and self._embed_fn
            and len(content) > self.config.min_message_length_for_split
        ):
            from zerogmem.retriever.proposition_index import (
                Proposition,
                split_into_propositions,
            )

            sentences = split_into_propositions(
                content, self.config.min_proposition_length
            )
            if len(sentences) >= 2:
                from zerogmem.memory.chunker import ChunkManager as _CM

                for sent in sentences:
                    enriched = _CM.enrich_content_for_embedding(
                        sent, speaker, self.current_session_id, timestamp
                    )
                    sent_emb = self._embed_fn(enriched)
                    prop = Proposition(
                        parent_memory_id=item.id,
                        content=sent,
                        embedding=sent_emb,
                    )
                    self.proposition_index.add(prop)

        # Add to working memory
        working_item = WorkingMemoryItem(
            id=item.id,
            content=content,
            embedding=item.embedding,
            source_memory_id=item.id,
        )
        self.working_memory.add(working_item)

        # Extract cross-person trait compliments and store as separate facts
        self._extract_and_store_cross_person_traits(speaker, content, timestamp)

        # Track for consolidation
        self._recent_message_ids.append(item.id)
        if (
            self.consolidator
            and len(self._recent_message_ids)
            >= self.consolidator.config.consolidation_interval
        ):
            self.consolidator.consolidate(
                self._recent_message_ids[-self.consolidator.config.max_context_messages :]
            )
            self._recent_message_ids.clear()

        return item.id

    def _extract_and_store_cross_person_traits(
        self,
        speaker: str,
        content: str,
        timestamp: datetime,
    ) -> None:
        """Extract cross-person trait compliments and store as synthesized facts.

        When speaker A says "you're so thoughtful!" to speaker B, creates a
        separate memory: "B has personality trait: thoughtful (described by A)".
        These short, focused facts are findable by semantic search for abstract
        queries like "What personality traits does A say B has?"
        """
        if not self.current_episode or not self._embed_fn:
            return

        # Determine conversation partner (the "you" in "you're so X")
        partners = [
            p for p in self.current_episode.participant_names
            if p.lower() != speaker.lower()
        ]
        if not partners:
            return
        partner = partners[0]  # Primary conversation partner

        content_lower = content.lower()
        for pattern, _ in self._CROSS_PERSON_TRAIT_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                trait = match.group(1).strip()
                if not trait or len(trait) < 3:
                    continue

                # Create a synthesized fact memory
                fact_content = (
                    f"{partner} has personality trait: {trait} "
                    f"(described by {speaker})"
                )

                # Generate embedding for the synthesized fact
                if self.chunker and self.chunker.config.enrich_embedding_prefix:
                    from zerogmem.memory.chunker import ChunkManager as _CM
                    enriched = _CM.enrich_content_for_embedding(
                        fact_content, speaker,
                        self.current_session_id or "", timestamp,
                    )
                    fact_embedding = self._embed_fn(enriched)
                else:
                    fact_embedding = self._embed_fn(fact_content)

                fact_item = UnifiedMemoryItem(
                    content=fact_content,
                    source="cross_person_trait",
                    entities=[partner, speaker],
                    entity_names=[partner, speaker],
                    speaker=speaker,
                    session_id=self.current_session_id,
                    embedding=fact_embedding,
                )
                self.graph.add_memory(fact_item)

                # Also add to BM25 for keyword search
                if self.bm25 is not None:
                    self.bm25.add_document(
                        doc_id=fact_item.id,
                        content=f"{speaker}: {fact_content}",
                        metadata={
                            "speaker": speaker,
                            "session_id": self.current_session_id or "",
                        },
                    )

    def add_memory_item(self, item: UnifiedMemoryItem) -> str:
        """Add a pre-built memory item with entity enrichment.

        Runs known-entity injection, pronoun resolution, and chunk
        accumulation.  Use this when the caller handles encoding
        externally but still wants the core entity enrichment pipeline.
        """
        self._inject_known_entities(item)
        self._resolve_pronouns(item)
        self.graph.add_memory(item)

        # Chunk accumulation (may trigger LLM segmentation + backfill)
        if self.chunker:
            self.chunker.on_message_added(item.id)

        return item.id

    def add_encoding_result(
        self,
        encoding_result: Any,
        session_id: str | None = None,
    ) -> str:
        """Add an encoded message with all extracted knowledge to the core pipeline.

        Handles: memory item insertion, entity enrichment, BM25 indexing,
        entity/relation/negation registration, and chunk accumulation.
        Use this when the caller runs encoding externally (e.g. batch
        embedding) and wants the full core pipeline applied.

        Args:
            encoding_result: An EncodingResult from Encoder.encode()
            session_id: Optional session ID for fact provenance
        """
        item = encoding_result.memory_item

        # Core entity enrichment + graph insertion + chunk accumulation
        self.add_memory_item(item)

        # BM25 indexing
        if self.bm25 is not None:
            content = f"{item.speaker}: {item.content}" if item.speaker else item.content
            self.bm25.add_document(
                doc_id=item.id,
                content=content,
                metadata={
                    "speaker": item.speaker or "",
                    "session_id": item.session_id or self.current_session_id or "",
                },
            )

        # Register entities from regex extraction
        for entity in encoding_result.entities:
            self.add_entity(name=entity.normalized, entity_type=entity.type)

        # Register relations and facts from regex extraction
        for relation in encoding_result.relations:
            self.add_relation(
                source_entity=relation.subject,
                relation=relation.predicate,
                target_entity=relation.object,
                negated=relation.negated,
                evidence_memory_id=item.id,
            )
            self.add_fact(
                subject=relation.subject,
                predicate=relation.predicate,
                obj=relation.object,
                source_episode_id=session_id or self.current_session_id,
                negated=relation.negated,
            )

        # Register negations
        for negation in encoding_result.negations:
            neg_content = negation.get("content", "")
            if neg_content:
                self.semantic_memory.add_negation(
                    subject="",
                    predicate="stated",
                    obj=neg_content,
                    source_id=item.id,
                )

        return item.id

    def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.UNKNOWN,
        attributes: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> str:
        """Add or update an entity in the entity graph."""
        # Check if entity exists
        existing = self.graph.entity_graph.find_by_name(name)
        if existing:
            # Update existing entity
            entity = existing[0]
            entity.last_seen = datetime.now()
            entity.mention_count += 1
            if attributes:
                entity.attributes.update(attributes)
            if aliases:
                entity.aliases.extend([a for a in aliases if a not in entity.aliases])
            return entity.id

        # Create new entity
        entity = EntityNode(
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
            aliases=aliases or [],
        )

        return self.graph.add_entity(entity)

    def add_relation(
        self,
        source_entity: str,
        relation: str,
        target_entity: str,
        negated: bool = False,
        evidence_memory_id: str | None = None,
    ) -> str:
        """Add a relation between entities."""
        # Ensure entities exist
        source_id = self.add_entity(source_entity)
        target_id = self.add_entity(target_entity)

        # Add relation
        return self.graph.add_entity_relation(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            negated=negated,
            evidence=[evidence_memory_id] if evidence_memory_id else [],
        )

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        category: str = "",
        source_episode_id: str | None = None,
        negated: bool = False,
    ) -> str:
        """Add a fact to semantic memory."""
        embedding = self._embed_fn(f"{subject} {predicate} {obj}") if self._embed_fn else None

        fact = Fact(
            content=f"{subject} {predicate} {obj}",
            subject=subject,
            predicate=predicate,
            object=obj,
            category=category,
            sources=[source_episode_id] if source_episode_id else [],
            embedding=embedding,
            negated=negated,
        )

        fact_id, _ = self.semantic_memory.add_fact(fact)
        return fact_id

    def add_negative_fact(
        self, subject: str, predicate: str, obj: str, source_episode_id: str | None = None
    ) -> str:
        """Add an explicit negation (what is NOT true)."""
        return self.semantic_memory.add_negation(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source_id=source_episode_id or "unknown",
        )

    # ==================== Query Methods ====================

    def query(
        self,
        query_text: str,
        query_type: str = "auto",
        top_k: int = 10,
        include_working_memory: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Main query interface for retrieving relevant memories.

        Args:
            query_text: The query string
            query_type: "semantic", "temporal", "entity", "fact", or "auto"
            top_k: Number of results to return
            include_working_memory: Whether to include working memory in results

        Returns:
            List of memory items with scores and metadata
        """
        results = []

        # Get query embedding
        query_embedding = self._embed_fn(query_text) if self._embed_fn else None

        # Include working memory context
        if include_working_memory and query_embedding is not None:
            wm_items = self.working_memory.get_context(query_embedding, top_k=5)
            for item in wm_items:
                results.append(
                    {
                        "id": item.id,
                        "content": item.content,
                        "source": "working_memory",
                        "score": item.attention_weight,
                        "type": "working",
                    }
                )

        # Semantic search in unified graph
        if query_embedding is not None:
            similar = self.graph.query_by_similarity(query_embedding, top_k=top_k)
            for memory, score in similar:
                results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "source": memory.source,
                        "score": score,
                        "type": "semantic",
                        "entities": memory.entity_names,
                        "timestamp": (
                            memory.event_time.start.isoformat() if memory.event_time else None
                        ),
                    }
                )

        # Search episodic memory
        if query_embedding is not None:
            episodes = self.episodic_memory.search_similar(query_embedding, top_k=top_k // 2)
            for episode, score in episodes:
                results.append(
                    {
                        "id": episode.id,
                        "content": episode.summary or episode.get_full_text()[:500],
                        "source": "episodic",
                        "score": score,
                        "type": "episode",
                        "participants": episode.participant_names,
                        "timestamp": episode.start_time.isoformat(),
                    }
                )

        # Search semantic facts
        if query_embedding is not None:
            facts = self.semantic_memory.search_similar(query_embedding, top_k=top_k // 2)
            for fact, score in facts:
                results.append(
                    {
                        "id": fact.id,
                        "content": fact.content,
                        "source": "semantic_fact",
                        "score": score * fact.confidence,  # Weight by confidence
                        "type": "fact",
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "negated": fact.negated,
                        "confidence": fact.confidence,
                    }
                )

        # Sort by score and deduplicate
        results.sort(
            key=lambda x: float(x["score"]),  # type: ignore[arg-type]
            reverse=True,
        )
        seen_ids = set()
        unique_results = []
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)

        return unique_results[:top_k]

    def query_temporal(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
        entities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query memories by time range."""
        memories = self.graph.query_by_time(start_time, end_time, entities)

        return [
            {
                "id": m.id,
                "content": m.content,
                "source": m.source,
                "timestamp": m.event_time.start.isoformat() if m.event_time else None,
                "entities": m.entity_names,
            }
            for m in memories
        ]

    def query_entity(self, entity_name: str) -> dict[str, Any]:
        """Get comprehensive information about an entity."""
        # Find entity
        entities = self.graph.entity_graph.find_by_name(entity_name)
        if not entities:
            return {"error": f"Entity '{entity_name}' not found"}

        entity = entities[0]

        # Get profile from entity graph
        profile = self.graph.entity_graph.get_entity_profile(entity.id)

        # Get facts from semantic memory
        facts = self.semantic_memory.get_facts_about(entity.id)
        facts.extend(self.semantic_memory.get_facts_about(entity_name))

        # Get related memories
        memories = self.graph.query_by_entity(entity.id)

        # Get temporal chain
        temporal_chain = self.graph.find_temporal_chain(entity.id)

        return {
            "entity": profile,
            "facts": [
                {
                    "content": f.content,
                    "confidence": f.confidence,
                    "negated": f.negated,
                }
                for f in facts[:20]
            ],
            "memory_count": len(memories),
            "recent_events": [
                {
                    "content": m.content[:200],
                    "timestamp": m.event_time.start.isoformat() if m.event_time else None,
                }
                for m in temporal_chain[:10]
            ],
        }

    def check_fact(self, subject: str, predicate: str, obj: str) -> dict[str, Any]:
        """Check if a fact is true, false, or unknown."""
        # Check for explicit negation
        is_negated, negation_fact = self.semantic_memory.check_negation(subject, predicate, obj)

        if is_negated:
            return {
                "status": "negated",
                "message": "Explicitly stored as NOT true",
                "confidence": 1.0,
                "evidence": negation_fact.negation_source if negation_fact else None,
            }

        # Check for positive fact
        facts = self.semantic_memory.get_facts_about(subject, predicate)
        matching = [f for f in facts if f.object == obj and not f.negated]

        if matching:
            best = max(matching, key=lambda f: f.confidence)
            return {
                "status": "confirmed",
                "message": "Fact found in memory",
                "confidence": best.confidence,
                "confirmations": best.confirmation_count,
                "evidence": best.sources,
            }

        # Check entity relations
        source_entities = self.graph.entity_graph.find_by_name(subject)
        target_entities = self.graph.entity_graph.find_by_name(obj)

        if source_entities and target_entities:
            exists, is_neg = self.graph.entity_graph.has_relation(
                source_entities[0].id, target_entities[0].id, predicate
            )
            if exists:
                return {
                    "status": "negated" if is_neg else "confirmed",
                    "message": "Relation found in entity graph",
                    "confidence": 0.8,
                }

        return {
            "status": "unknown",
            "message": "No information found about this fact",
            "confidence": 0.0,
        }

    # ==================== Consolidation ====================

    def consolidate(self) -> dict[str, int]:
        """
        Perform memory consolidation.

        - Extract facts from frequently accessed episodes
        - Update entity profiles
        - Archive old episodes
        """
        stats = {
            "facts_extracted": 0,
            "episodes_archived": 0,
            "entities_updated": 0,
        }

        # Find episodes ready for consolidation
        candidates = self.episodic_memory.get_candidates_for_consolidation(
            min_retrievals=self.config.consolidation_min_retrievals,
            min_age_days=self.config.consolidation_threshold_days,
        )

        for episode in candidates:
            # Extract facts (simplified - in production, use LLM)
            for fact_text in episode.extracted_facts:
                # This would be LLM-extracted in production
                stats["facts_extracted"] += 1

        return stats

    # ==================== Statistics ====================

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the memory system."""
        ep_count = len(self.episodic_memory.episodes)
        fact_count = len(self.semantic_memory.facts)
        return {
            "graph": self.graph.get_stats(),
            "working_memory": self.working_memory.get_stats(),
            "episodic_memory": self.episodic_memory.get_stats(),
            "semantic_memory": self.semantic_memory.get_stats(),
            "current_session": self.current_session_id,
            "capacity": {
                "max_episodes": self.config.max_episodes,
                "max_facts": self.config.max_facts,
                "episode_utilization": (
                    ep_count / self.config.max_episodes if self.config.max_episodes else 0
                ),
                "fact_utilization": (
                    fact_count / self.config.max_facts if self.config.max_facts else 0
                ),
            },
        }

    def get_context_for_response(self, query: str, max_tokens: int = 4000) -> str:
        """
        Get formatted context string for LLM response generation.

        Uses position-aware composition to combat lost-in-the-middle.
        """
        results = self.query(query, top_k=20)

        if not results:
            return ""

        # Position-aware composition
        parts = []

        # High relevance at start
        parts.append("## Most Relevant Context")
        for r in results[:3]:
            parts.append(f"- {r['content']}")

        # Medium relevance in middle with structure
        if len(results) > 6:
            parts.append("\n## Additional Context")
            for i, r in enumerate(results[3:-3], 1):
                parts.append(f"[{i}] {r['content'][:200]}")

        # Reinforce key points at end
        parts.append("\n## Key Points")
        for r in results[:2]:
            key_point = r["content"][:100]
            parts.append(f"- {key_point}")

        context = "\n".join(parts)

        # Truncate if needed (rough estimate)
        if len(context) > max_tokens * 4:
            context = context[: max_tokens * 4]

        return context

    def _episode_to_dict(self, ep: Episode) -> dict[str, Any]:
        """Serialize a single Episode to a dictionary."""
        return {
            "id": ep.id,
            "summary": ep.summary,
            "title": ep.title,
            "start_time": ep.start_time.isoformat(),
            "end_time": ep.end_time.isoformat() if ep.end_time else None,
            "session_id": ep.session_id,
            "participants": ep.participants,
            "participant_names": ep.participant_names,
            "location": ep.location,
            "topics": ep.topics,
            "emotional_valence": ep.emotional_valence,
            "importance": ep.importance,
            "retrieval_count": ep.retrieval_count,
            "last_retrieved": ep.last_retrieved.isoformat() if ep.last_retrieved else None,
            "created_at": ep.created_at.isoformat(),
            "extracted_facts": ep.extracted_facts,
            "archived": ep.archived,
            "archive_ref": ep.archive_ref,
            "metadata": ep.metadata,
            "messages": [
                {
                    "id": msg.id,
                    "speaker": msg.speaker,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "entities_mentioned": msg.entities_mentioned,
                    "sentiment": msg.sentiment,
                    "metadata": msg.metadata,
                }
                for msg in ep.messages
            ],
        }

    @staticmethod
    def _episode_from_dict(
        ep_data: dict[str, Any],
        embeddings_map: dict[str, np.ndarray] | None = None,
    ) -> Episode:
        """Deserialize a single Episode from a dictionary."""
        embeddings_map = embeddings_map or {}
        messages = []
        for msg_data in ep_data.get("messages", []):
            messages.append(
                EpisodeMessage(
                    id=msg_data.get("id", str(uuid.uuid4())),
                    speaker=msg_data.get("speaker", ""),
                    content=msg_data.get("content", ""),
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    entities_mentioned=msg_data.get("entities_mentioned", []),
                    sentiment=msg_data.get("sentiment", 0.0),
                    metadata=msg_data.get("metadata", {}),
                )
            )
        return Episode(
            id=ep_data["id"],
            summary=ep_data.get("summary", ""),
            title=ep_data.get("title", ""),
            messages=messages,
            start_time=datetime.fromisoformat(ep_data["start_time"]),
            end_time=(
                datetime.fromisoformat(ep_data["end_time"]) if ep_data.get("end_time") else None
            ),
            session_id=ep_data.get("session_id"),
            participants=ep_data.get("participants", []),
            participant_names=ep_data.get("participant_names", []),
            location=ep_data.get("location"),
            topics=ep_data.get("topics", []),
            emotional_valence=ep_data.get("emotional_valence", 0.0),
            importance=ep_data.get("importance", 0.5),
            summary_embedding=embeddings_map.get(ep_data["id"]),
            retrieval_count=ep_data.get("retrieval_count", 0),
            last_retrieved=(
                datetime.fromisoformat(ep_data["last_retrieved"])
                if ep_data.get("last_retrieved")
                else None
            ),
            created_at=(
                datetime.fromisoformat(ep_data["created_at"])
                if ep_data.get("created_at")
                else datetime.now()
            ),
            extracted_facts=ep_data.get("extracted_facts", []),
            archived=ep_data.get("archived", False),
            archive_ref=ep_data.get("archive_ref"),
            metadata=ep_data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire memory system.

        Note: Working memory is ephemeral and NOT serialized.
        Embeddings are NOT included — save separately via persistence module.
        The in-flight current_episode IS serialized so it survives restarts.
        """
        result = {
            "config": {
                "working_memory_capacity": self.config.working_memory_capacity,
                "working_memory_decay_rate": self.config.working_memory_decay_rate,
                "embedding_dim": self.config.embedding_dim,
                "auto_consolidate": self.config.auto_consolidate,
                "consolidation_threshold_days": self.config.consolidation_threshold_days,
                "consolidation_min_retrievals": self.config.consolidation_min_retrievals,
                "max_episodes": self.config.max_episodes,
                "max_facts": self.config.max_facts,
                "eviction_batch_size": self.config.eviction_batch_size,
                "use_bm25": self.config.use_bm25,
            },
            "current_session_id": self.current_session_id,
            "graph": self.graph.to_dict(),
            "episodic": self.episodic_memory.to_dict(),
            "semantic": self.semantic_memory.to_dict(),
        }
        # Proposition index
        if self.proposition_index:
            result["proposition_index"] = self.proposition_index.to_dict()
        # Serialize in-flight episode so it isn't lost on unexpected shutdown
        if self.current_episode and self.current_episode.messages:
            result["current_episode"] = self._episode_to_dict(self.current_episode)
        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        embeddings_map: dict[str, np.ndarray] | None = None,
        encoder_config: Any | None = None,
    ) -> MemoryManager:
        """Deserialize memory manager from dictionary.

        Args:
            data: Output of to_dict().
            embedding_fn: Embedding function to set on the restored manager.
            embeddings_map: Map of id -> embedding for all components.
            encoder_config: Optional EncoderConfig for the internal encoder.
        """
        embeddings_map = embeddings_map or {}
        config_data = data.get("config", {})
        config = MemoryConfig(
            working_memory_capacity=config_data.get("working_memory_capacity", 20),
            working_memory_decay_rate=config_data.get("working_memory_decay_rate", 0.05),
            embedding_dim=config_data.get("embedding_dim", 3072),
            auto_consolidate=config_data.get("auto_consolidate", True),
            consolidation_threshold_days=config_data.get("consolidation_threshold_days", 7),
            consolidation_min_retrievals=config_data.get("consolidation_min_retrievals", 3),
            max_episodes=config_data.get("max_episodes", 500),
            max_facts=config_data.get("max_facts", 5000),
            eviction_batch_size=config_data.get("eviction_batch_size", 10),
            use_bm25=config_data.get("use_bm25", True),
        )

        manager = cls(config=config, encoder_config=encoder_config)

        if embedding_fn:
            manager.set_embedding_function(embedding_fn)
        else:
            # Use the internal encoder's embedding function
            manager.set_embedding_function(manager.encoder.get_embedding)

        # Restore sub-components
        if "graph" in data:
            manager.graph = UnifiedMemoryGraph.from_dict(data["graph"], embeddings_map)
        if "episodic" in data:
            manager.episodic_memory = EpisodicMemory.from_dict(data["episodic"], embeddings_map)
        if "semantic" in data:
            manager.semantic_memory = SemanticMemoryStore.from_dict(
                data["semantic"], embeddings_map
            )

        manager.current_session_id = data.get("current_session_id")

        # Restore proposition index
        if "proposition_index" in data and manager.proposition_index is not None:
            from zerogmem.retriever.proposition_index import (
                PropositionIndex as _PropIndex,
            )

            manager.proposition_index = _PropIndex.from_dict(
                data["proposition_index"], embeddings_map
            )

        # Restore in-flight episode if it was serialized
        if "current_episode" in data:
            manager.current_episode = cls._episode_from_dict(
                data["current_episode"], embeddings_map
            )

        # Rebuild BM25 index from restored memories
        if manager.bm25 is not None and manager.graph.memories:
            manager.enable_bm25()

        return manager
