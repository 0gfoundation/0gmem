"""
Chunk Manager: LLM-based topic segmentation for hierarchical memory.

Accumulates messages during ingestion and, at configurable thresholds,
sends them to an LLM for topic-based segmentation. Each chunk becomes
a first-class UnifiedMemoryItem with its own summary embedding and
graph edges, enabling cross-message retrieval.

Hierarchy: Session -> Chunk -> Message
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from zerogmem.defaults import DEFAULT_LLM_MODEL

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from zerogmem.memory.manager import MemoryManager

logger = logging.getLogger("0gmem.chunker")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """Configuration for chunk-based memory segmentation."""

    enabled: bool = True
    chunk_window_size: int = 100        # Trigger LLM segmentation every N messages
    llm_model: str = DEFAULT_LLM_MODEL
    max_chunks_per_window: int = 10     # Safety cap on LLM output
    min_chunk_size: int = 3             # Minimum messages per chunk
    enrich_embedding_prefix: bool = True  # Speaker-enriched embeddings
    max_retries: int = 2
    temperature: float = 0.1
    max_completion_tokens: int = 16000   # Token budget for segmentation response


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A topic-based chunk of contiguous messages within a session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    summary: str = ""
    topic_keywords: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    entity_relations: list[dict[str, Any]] = field(default_factory=list)
    temporal_anchors: list[str] = field(default_factory=list)
    causal_links: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)
    start_turn: int = 0
    end_turn: int = 0
    embedding: np.ndarray | None = None
    memory_item_id: str | None = None   # ID of the UnifiedMemoryItem for this chunk
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SEGMENTATION_SYSTEM = """\
You are analyzing a conversation window to identify topic segments and \
extract structured knowledge.

Given a sequence of numbered messages, segment them into topic-based chunks \
where each chunk covers a coherent topic or activity. Chunks can vary in \
size (3-30 messages each). Start a new chunk when the conversation shifts \
to a materially different topic.

For each chunk, extract:
1. summary: A 1-3 sentence summary capturing the key information discussed
2. topic_keywords: 3-5 topic tags
3. entities: People, places, organizations, objects mentioned. \
   For each: name, type (person/location/organization/object/concept/event), aliases.
4. entity_relations: Relationships between entities. \
   Include negated relations (e.g. "does NOT like sushi" -> negated: true).
5. temporal_anchors: Any time references (dates, relative time expressions)
6. causal_links: Cause-effect relationships mentioned or implied
7. facts: Structured subject-predicate-object triples capturing key information. \
   Include negated facts. Category: identity/attribute/preference/event/status/relationship/location/activity.

Return a JSON array of chunk objects. Each object must have:
  "start_message": int (1-indexed message number where this chunk starts)
  "end_message": int (1-indexed message number where this chunk ends)
  "summary": str
  "topic_keywords": list of str
  "entities": list of {"name": str, "type": str, "aliases": list of str}
  "entity_relations": list of {"source": str, "relation": str, "target": str, "negated": bool}
  "temporal_anchors": list of str
  "causal_links": list of {"cause": str, "effect": str}
  "facts": list of {"subject": str, "predicate": str, "object": str, "negated": bool, "category": str}

Messages must be covered exactly once with no overlaps. \
Return ONLY the JSON array, no other text."""


# ---------------------------------------------------------------------------
# ChunkManager
# ---------------------------------------------------------------------------

class ChunkManager:
    """Accumulates messages and triggers LLM-based topic segmentation.

    Lifecycle:
        1. ``on_message_added()`` is called for each message during ingestion.
        2. When the buffer reaches ``chunk_window_size``, ``_segment_window()``
           sends the buffered messages to the LLM for segmentation.
        3. ``flush()`` is called at session end to handle remaining messages.

    Each resulting ``Chunk`` is registered in the unified graph as a
    first-class memory item with its own embedding.
    """

    def __init__(
        self,
        memory: MemoryManager,
        llm_client: Any | None = None,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        config: ChunkConfig | None = None,
    ) -> None:
        self.memory = memory
        self.llm_client = llm_client
        self._embed_fn = embed_fn
        self.config = config or ChunkConfig()

        # Accumulated message IDs since last segmentation
        self._message_buffer: list[str] = []
        # All chunks created, keyed by chunk ID
        self._chunks: dict[str, Chunk] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_message_added(self, message_id: str) -> list[Chunk]:
        """Called after each ``add_message()`` in MemoryManager.

        Returns newly created chunks (empty list most of the time).
        """
        self._message_buffer.append(message_id)

        if len(self._message_buffer) >= self.config.chunk_window_size:
            return self._segment_window(list(self._message_buffer))
        return []

    def flush(self) -> list[Chunk]:
        """Segment remaining buffered messages (called at session end).

        Skips segmentation if the buffer has fewer than ``min_chunk_size``
        messages — those are too short to form a meaningful chunk.
        """
        if len(self._message_buffer) < self.config.min_chunk_size:
            self._message_buffer.clear()
            return []
        return self._segment_window(list(self._message_buffer))

    @staticmethod
    def enrich_content_for_embedding(
        content: str,
        speaker: str | None = None,
        session_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Prepend speaker/session/time context to content before embedding.

        Uses bracket notation to minimise semantic dilution while still
        giving the embedding model speaker/temporal signal.

        Returns something like:
            ``[John] (Nov 21 2023): I finished rereading The Alchemist``
        """
        parts: list[str] = []
        if speaker:
            parts.append(f"[{speaker}]")
        if timestamp:
            parts.append(f"({timestamp.strftime('%b %d %Y')})")
        if parts:
            return " ".join(parts) + f": {content}"
        return content

    # ------------------------------------------------------------------
    # Internal — LLM segmentation
    # ------------------------------------------------------------------

    def _segment_window(self, message_ids: list[str]) -> list[Chunk]:
        """Send a window of messages to the LLM for topic segmentation."""
        self._message_buffer.clear()

        # Gather messages from memory
        messages = self._gather_messages(message_ids)
        if not messages:
            return []

        # Format for LLM
        formatted = self._format_messages(messages, message_ids)

        # Call LLM
        raw_chunks = self._call_llm(formatted)
        if raw_chunks is None:
            # LLM failed — create a single fallback chunk from the whole window
            logger.warning("LLM segmentation failed; creating single fallback chunk")
            raw_chunks = [self._fallback_chunk(messages, message_ids)]

        # Build Chunk objects
        session_id = self.memory.current_session_id or ""
        created: list[Chunk] = []

        for raw in raw_chunks[: self.config.max_chunks_per_window]:
            start_idx = raw.get("start_message", 1) - 1  # Convert to 0-indexed
            end_idx = raw.get("end_message", len(message_ids))  # inclusive

            # Clamp to valid range
            start_idx = max(0, min(start_idx, len(message_ids) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(message_ids)))

            chunk_msg_ids = message_ids[start_idx:end_idx]
            if len(chunk_msg_ids) < self.config.min_chunk_size and len(raw_chunks) > 1:
                # Merge tiny chunks with the next one instead of creating them
                continue

            # Normalize entities: accept both list[str] and list[dict]
            raw_entities = raw.get("entities", [])
            entities: list[dict[str, Any]] = []
            for e in raw_entities:
                if isinstance(e, str):
                    entities.append({"name": e, "type": "unknown", "aliases": []})
                elif isinstance(e, dict) and e.get("name"):
                    entities.append(e)

            chunk = Chunk(
                session_id=session_id,
                summary=raw.get("summary", ""),
                topic_keywords=raw.get("topic_keywords", []),
                entities=entities,
                entity_relations=raw.get("entity_relations", []),
                temporal_anchors=raw.get("temporal_anchors", []),
                causal_links=raw.get("causal_links", []),
                facts=raw.get("facts", []),
                message_ids=chunk_msg_ids,
                start_turn=start_idx,
                end_turn=end_idx - 1,
            )

            # Register in graph
            self._register_chunk_in_graph(chunk)
            self._chunks[chunk.id] = chunk
            created.append(chunk)

        logger.info(
            "Segmented %d messages into %d chunks (session=%s)",
            len(message_ids),
            len(created),
            session_id,
        )
        return created

    def _gather_messages(
        self, message_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Retrieve message content/speaker/timestamp from the memory graph."""
        messages: list[dict[str, Any]] = []
        for mid in message_ids:
            mem = self.memory.graph.memories.get(mid)
            if mem:
                messages.append(
                    {
                        "id": mid,
                        "speaker": mem.speaker or "",
                        "content": mem.content,
                        "timestamp": mem.event_time.start if mem.event_time else None,
                        "session_id": mem.session_id or "",
                    }
                )
        return messages

    def _format_messages(
        self, messages: list[dict[str, Any]], message_ids: list[str]
    ) -> str:
        """Format messages for the LLM prompt."""
        lines: list[str] = []
        for i, msg in enumerate(messages, 1):
            speaker = msg.get("speaker", "")
            ts = msg.get("timestamp")
            ts_str = ts.strftime("%b %d %Y") if ts else ""
            prefix = f"[{i}] [{speaker}]"
            if ts_str:
                prefix += f" ({ts_str})"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def _call_llm(self, formatted_messages: str) -> list[dict[str, Any]] | None:
        """Call LLM for segmentation with retry."""
        if self.llm_client is None:
            return None

        user_prompt = (
            f"Conversation window:\n\n{formatted_messages}\n\n"
            "Segment this window into topic-based chunks and extract knowledge for each."
        )

        for attempt in range(self.config.max_retries + 1):
            try:
                from zerogmem.defaults import llm_chat_kwargs

                response = self.llm_client.chat.completions.create(
                    **llm_chat_kwargs(
                        self.config.llm_model,
                        max_tokens=self.config.max_completion_tokens,
                        temperature=self.config.temperature,
                    ),
                    messages=[
                        {"role": "system", "content": _SEGMENTATION_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                finish = response.choices[0].finish_reason
                if not text:
                    logger.warning(
                        "LLM segmentation returned empty response (finish_reason=%s)",
                        finish,
                    )
                    continue
                if finish == "length":
                    logger.warning(
                        "LLM segmentation truncated (finish_reason=length, %d chars)",
                        len(text),
                    )
                return self._parse_json_array(text)
            except Exception:
                logger.warning("LLM segmentation attempt %d failed", attempt + 1, exc_info=True)

        return None

    @staticmethod
    def _parse_json_array(text: str) -> list[dict[str, Any]] | None:
        """Parse a JSON array from LLM output, stripping markdown fences."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            first_newline = text.index("\n") if "\n" in text else 3
            text = text[first_newline:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        # Try direct parse first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            # Handle {"chunks": [...]} wrapper
            if isinstance(parsed, dict):
                for key in ("chunks", "segments", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
        except json.JSONDecodeError:
            pass

        # Fallback: find outermost [...] in the response
        start = text.find("[")
        if start != -1:
            # Find matching closing bracket (handle nesting)
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start : i + 1])
                            if isinstance(parsed, list):
                                return parsed
                        except json.JSONDecodeError:
                            break

        logger.warning(
            "Failed to parse LLM segmentation output (first 300 chars): %s",
            text[:300],
        )
        return None

    def _fallback_chunk(
        self,
        messages: list[dict[str, Any]],
        message_ids: list[str],
    ) -> dict[str, Any]:
        """Create a single-chunk fallback when LLM fails."""
        # Extractive summary: first and last message
        first = messages[0]["content"][:200] if messages else ""
        last = messages[-1]["content"][:200] if len(messages) > 1 else ""
        summary = first
        if last and last != first:
            summary = f"{first} ... {last}"

        # Collect speakers as entities
        entities = [
            {"name": s, "type": "person", "aliases": []}
            for s in {m["speaker"] for m in messages if m.get("speaker")}
        ]

        return {
            "start_message": 1,
            "end_message": len(message_ids),
            "summary": summary,
            "topic_keywords": [],
            "entities": entities,
            "entity_relations": [],
            "temporal_anchors": [],
            "causal_links": [],
            "facts": [],
        }

    # ------------------------------------------------------------------
    # Graph integration
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_names(chunk: Chunk) -> list[str]:
        """Extract plain entity name strings from chunk.entities (list[dict])."""
        return [e["name"] for e in chunk.entities if e.get("name")]

    def _register_chunk_in_graph(self, chunk: Chunk) -> None:
        """Register a chunk as a first-class UnifiedMemoryItem in the graph."""
        import re as _re

        from zerogmem.graph.entity import EntityType
        from zerogmem.graph.temporal import TimeInterval
        from zerogmem.graph.unified import UnifiedMemoryItem

        entity_names = self._entity_names(chunk)

        # Determine time span from child messages
        start_time: datetime | None = None
        end_time: datetime | None = None
        for mid in chunk.message_ids:
            mem = self.memory.graph.memories.get(mid)
            if mem and mem.event_time and mem.event_time.start:
                if start_time is None or mem.event_time.start < start_time:
                    start_time = mem.event_time.start
                if end_time is None or mem.event_time.start > end_time:
                    end_time = mem.event_time.start

        # Embed the chunk summary
        embedding = None
        if self._embed_fn and chunk.summary:
            embedding = self._embed_fn(chunk.summary)
        chunk.embedding = embedding

        event_time = None
        if start_time:
            event_time = TimeInterval(start=start_time, end=end_time)

        memory_item = UnifiedMemoryItem(
            content=chunk.summary,
            embedding=embedding,
            event_time=event_time,
            entities=entity_names,
            entity_names=entity_names,
            concepts=chunk.topic_keywords,
            source="chunk",
            session_id=chunk.session_id,
            speaker=None,  # Chunks are multi-speaker
            metadata={
                "chunk_id": chunk.id,
                "message_ids": chunk.message_ids,
                "message_count": len(chunk.message_ids),
                "start_turn": chunk.start_turn,
                "end_turn": chunk.end_turn,
                "temporal_anchors": chunk.temporal_anchors,
                "causal_links": [c for c in chunk.causal_links],
                "is_chunk": True,
            },
        )

        # Add to unified graph
        self.memory.graph.add_memory(memory_item)
        chunk.memory_item_id = memory_item.id

        # Add to BM25 index
        if self.memory.bm25 is not None and chunk.summary:
            self.memory.bm25.add_document(
                doc_id=memory_item.id,
                content=chunk.summary,
                metadata={
                    "session_id": chunk.session_id,
                    "is_chunk": True,
                },
            )

        # Update child messages with chunk_id reference and backfill entities
        for mid in chunk.message_ids:
            mem = self.memory.graph.memories.get(mid)
            if mem:
                mem.chunk_id = chunk.id
                content_lower = mem.content.lower()
                for ename in entity_names:
                    if ename not in mem.entity_names and ename.lower() in content_lower:
                        if _re.search(
                            r"\b" + _re.escape(ename.lower()) + r"\b",
                            content_lower,
                        ):
                            mem.entity_names.append(ename)

        # Register entities with type and aliases
        _ENTITY_TYPE_MAP = {
            "person": EntityType.PERSON,
            "location": EntityType.LOCATION,
            "organization": EntityType.ORGANIZATION,
            "object": EntityType.UNKNOWN,
            "concept": EntityType.CONCEPT,
            "event": EntityType.EVENT,
        }
        for edict in chunk.entities:
            ename = edict.get("name", "")
            if not ename:
                continue
            etype = _ENTITY_TYPE_MAP.get(
                edict.get("type", "unknown"), EntityType.UNKNOWN
            )
            try:
                self.memory.add_entity(
                    name=ename,
                    entity_type=etype,
                    aliases=edict.get("aliases", []),
                )
            except Exception:
                logger.debug("Failed to add entity %s", ename, exc_info=True)

        # Pronoun resolution pass over child messages using chunk entities
        recent_ents: list[str] = []
        for mid in chunk.message_ids:
            mem = self.memory.graph.memories.get(mid)
            if not mem:
                continue
            speaker_lower = (mem.speaker or "").lower()
            non_speaker = [
                e for e in mem.entity_names
                if e.lower() != speaker_lower
                and e.lower() not in ("user", "assistant", "unknown", "summary")
            ]
            if non_speaker:
                recent_ents = non_speaker + [
                    e for e in recent_ents if e not in non_speaker
                ]
                recent_ents = recent_ents[:5]
            elif recent_ents and self.memory._THIRD_PERSON_PRONOUNS.search(mem.content):
                for ent in recent_ents[:2]:
                    if ent not in mem.entity_names:
                        mem.entity_names.append(ent)

        # Add entity relations (with negated support)
        for rel in chunk.entity_relations:
            source = rel.get("source", "")
            target = rel.get("target", "")
            relation = rel.get("relation", "")
            if source and target and relation:
                try:
                    self.memory.add_relation(
                        source_entity=source,
                        relation=relation,
                        target_entity=target,
                        negated=rel.get("negated", False),
                    )
                except Exception:
                    logger.debug("Failed to add relation %s->%s", source, target, exc_info=True)

        # Add facts as semantic triples
        for fact in chunk.facts:
            subject = fact.get("subject", "")
            predicate = fact.get("predicate", "")
            obj = fact.get("object", "")
            if subject and predicate and obj:
                try:
                    self.memory.add_fact(
                        subject=subject,
                        predicate=predicate,
                        obj=obj,
                        category=fact.get("category", ""),
                        negated=fact.get("negated", False),
                    )
                except Exception:
                    logger.debug(
                        "Failed to add fact %s %s %s", subject, predicate, obj,
                        exc_info=True,
                    )

        # Add causal edges
        for link in chunk.causal_links:
            cause = link.get("cause", "")
            effect = link.get("effect", "")
            if cause and effect:
                try:
                    from zerogmem.graph.causal import CausalEdge, CausalNode

                    cause_node = CausalNode(
                        content=cause, memory_id=memory_item.id
                    )
                    effect_node = CausalNode(
                        content=effect, memory_id=memory_item.id
                    )
                    self.memory.graph.causal_graph.add_node(cause_node)
                    self.memory.graph.causal_graph.add_node(effect_node)
                    edge = CausalEdge(
                        cause_id=cause_node.id,
                        effect_id=effect_node.id,
                        evidence=[memory_item.id],
                    )
                    self.memory.graph.causal_graph.add_edge(edge)
                except Exception:
                    logger.debug("Failed to add causal link", exc_info=True)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk manager state (excluding embeddings — those live in the graph)."""
        chunks_data: list[dict[str, Any]] = []
        for chunk in self._chunks.values():
            chunks_data.append(
                {
                    "id": chunk.id,
                    "session_id": chunk.session_id,
                    "summary": chunk.summary,
                    "topic_keywords": chunk.topic_keywords,
                    "entities": chunk.entities,
                    "entity_relations": chunk.entity_relations,
                    "temporal_anchors": chunk.temporal_anchors,
                    "causal_links": chunk.causal_links,
                    "facts": chunk.facts,
                    "message_ids": chunk.message_ids,
                    "start_turn": chunk.start_turn,
                    "end_turn": chunk.end_turn,
                    "memory_item_id": chunk.memory_item_id,
                    "created_at": chunk.created_at.isoformat(),
                    "metadata": chunk.metadata,
                }
            )
        return {
            "message_buffer": list(self._message_buffer),
            "chunks": chunks_data,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        memory: MemoryManager,
        llm_client: Any | None = None,
        embed_fn: Any | None = None,
        config: ChunkConfig | None = None,
    ) -> ChunkManager:
        """Restore chunk manager from serialized state."""
        manager = cls(memory=memory, llm_client=llm_client, embed_fn=embed_fn, config=config)
        manager._message_buffer = data.get("message_buffer", [])

        for chunk_data in data.get("chunks", []):
            created_at_str = chunk_data.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                created_at = datetime.now()

            chunk = Chunk(
                id=chunk_data["id"],
                session_id=chunk_data.get("session_id", ""),
                summary=chunk_data.get("summary", ""),
                topic_keywords=chunk_data.get("topic_keywords", []),
                entities=chunk_data.get("entities", []),
                entity_relations=chunk_data.get("entity_relations", []),
                temporal_anchors=chunk_data.get("temporal_anchors", []),
                causal_links=chunk_data.get("causal_links", []),
                facts=chunk_data.get("facts", []),
                message_ids=chunk_data.get("message_ids", []),
                start_turn=chunk_data.get("start_turn", 0),
                end_turn=chunk_data.get("end_turn", 0),
                memory_item_id=chunk_data.get("memory_item_id"),
                created_at=created_at,
                metadata=chunk_data.get("metadata", {}),
            )
            manager._chunks[chunk.id] = chunk

        return manager
