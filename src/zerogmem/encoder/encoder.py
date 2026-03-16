"""
Encoder: Main encoding pipeline for converting conversations to memories.

Orchestrates entity extraction, temporal extraction, and embedding generation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from zerogmem.defaults import DEFAULT_EMBEDDING_MODEL
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zerogmem.encoder.entity_extractor import EntityExtractor, ExtractedEntity, ExtractedRelation
from zerogmem.encoder.temporal_extractor import TemporalExpression, TemporalExtractor
from zerogmem.graph.entity import EntityType
from zerogmem.graph.temporal import TimeInterval
from zerogmem.graph.unified import UnifiedMemoryItem

logger = logging.getLogger("0gmem.encoder")


@dataclass
class EncodingResult:
    """Result of encoding a text into memory structures."""

    memory_item: UnifiedMemoryItem
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    temporal_expressions: list[TemporalExpression]
    negations: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncoderConfig:
    """Configuration for the encoder."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = 3072
    use_llm_extraction: bool = False
    extract_importance: bool = True
    importance_threshold: float = 0.3
    max_retries: int = 3


class Encoder:
    """
    Main encoding pipeline for 0GMem.

    Responsibilities:
    - Convert text to embeddings
    - Extract entities and relationships
    - Extract temporal information
    - Compute importance scores
    - Produce UnifiedMemoryItem objects
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        embedding_fn: Callable[[str], np.ndarray] | None = None,
    ):
        """
        Initialize the encoder.

        Args:
            config: Encoder configuration
            embedding_fn: Custom embedding function. If not provided,
                         will use OpenAI embeddings.
        """
        self.config = config or EncoderConfig()
        self._embedding_fn = embedding_fn
        self._client: Any | None = None

        # Initialize extractors
        self.entity_extractor = EntityExtractor()
        self.temporal_extractor = TemporalExtractor()

    def _get_embedding_fn(self) -> Callable[[str], np.ndarray]:
        """Get or create embedding function."""
        if self._embedding_fn:
            return self._embedding_fn

        # Try to create OpenAI embedding function
        try:
            import openai

            self._client = openai.OpenAI()

            retryable_exceptions = (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            )
            max_retries = self.config.max_retries

            @retry(
                retry=retry_if_exception_type(retryable_exceptions),
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=60),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            def embed(text: str) -> np.ndarray:
                assert self._client is not None
                response = self._client.embeddings.create(
                    model=self.config.embedding_model,
                    input=text,
                )
                return np.array(response.data[0].embedding)

            self._embedding_fn = embed
            return embed

        except ImportError:
            logger.warning("openai package not installed. Using random embeddings.")
        except Exception as e:
            logger.warning("Could not initialize OpenAI client: %s. Using random embeddings.", e)

        # Fallback to random embeddings ONLY for initialization failure
        def random_embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.config.embedding_dim).astype(np.float32)

        self._embedding_fn = random_embed
        return random_embed

    def encode(
        self,
        text: str,
        speaker: str | None = None,
        timestamp: datetime | None = None,
        session_id: str | None = None,
        reference_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        skip_embedding: bool = False,
    ) -> EncodingResult:
        """
        Encode a text into memory structures.

        Args:
            text: The text to encode
            speaker: Who said this (for conversation context)
            timestamp: When this was said
            session_id: Session identifier
            reference_time: Reference time for temporal expressions
            metadata: Additional metadata

        Returns:
            EncodingResult with memory item and extracted information
        """
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        # Set reference time for temporal extraction
        if reference_time:
            self.temporal_extractor.set_reference_time(reference_time)
        else:
            self.temporal_extractor.set_reference_time(timestamp)

        # Extract entities
        entities = self.entity_extractor.extract_entities(text)

        # Extract relations
        relations = self.entity_extractor.extract_relations(text)

        # Extract negations
        negations = self.entity_extractor.extract_negations(text)

        # Extract temporal information
        temporal_expressions = self.temporal_extractor.extract(text)
        temporal_context = self.temporal_extractor.get_temporal_context(temporal_expressions)

        # Generate embedding (skip when caller will supply a pre-computed one)
        if skip_embedding:
            embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
        else:
            embed_fn = self._get_embedding_fn()
            embedding = embed_fn(text)

        # Compute importance score
        importance = self._compute_importance(
            text=text,
            entities=entities,
            relations=relations,
            temporal_expressions=temporal_expressions,
            negations=negations,
        )

        # Determine event time
        event_time = self._determine_event_time(timestamp, temporal_expressions)

        # Extract entity IDs and names
        entity_ids = []
        entity_names = []
        for entity in entities:
            entity_ids.append(entity.normalized)
            if entity.text not in entity_names:
                entity_names.append(entity.text)

        # Extract concepts/topics (simplified - could use LLM)
        concepts = self._extract_concepts(text, entities, relations)

        # Identify negated facts
        negated_facts = [n["full_text"] for n in negations]

        # Identify causal relationships (simplified)
        causes, effects = self._extract_causal_hints(text)

        # Create memory item
        memory_item = UnifiedMemoryItem(
            content=text,
            summary=self._create_summary(text, entities, relations),
            embedding=embedding,
            event_time=event_time,
            ingestion_time=datetime.now(),
            entities=entity_ids,
            entity_names=entity_names,
            causes=causes,
            effects=effects,
            concepts=concepts,
            importance=importance,
            source="conversation",
            session_id=session_id,
            speaker=speaker,
            negated_facts=negated_facts,
            metadata={
                **metadata,
                "temporal_context": temporal_context,
                "has_negations": len(negations) > 0,
                "entity_count": len(entities),
                "relation_count": len(relations),
            },
        )

        return EncodingResult(
            memory_item=memory_item,
            entities=entities,
            relations=relations,
            temporal_expressions=temporal_expressions,
            negations=negations,
            metadata=metadata,
        )

    def encode_batch(self, texts: list[str], **kwargs: Any) -> list[EncodingResult]:
        """Encode multiple texts."""
        return [self.encode(text, **kwargs) for text in texts]

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        embed_fn = self._get_embedding_fn()
        return embed_fn(text)

    def _compute_importance(
        self,
        text: str,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        temporal_expressions: list[TemporalExpression],
        negations: list[dict[str, Any]],
    ) -> float:
        """
        Compute importance score for a piece of text.

        Factors:
        - Entity density (more entities = more important)
        - Relation presence (relations are important)
        - Temporal specificity (specific times are important)
        - Negation presence (negations are important for accuracy)
        - Text length and information density
        """
        score = 0.3  # Base score

        # Entity contribution
        if entities:
            entity_score = min(0.2, len(entities) * 0.05)
            score += entity_score

        # Relation contribution
        if relations:
            relation_score = min(0.2, len(relations) * 0.1)
            score += relation_score

        # Temporal contribution
        if temporal_expressions:
            # Specific times are more important
            specific_count = sum(1 for t in temporal_expressions if t.normalized_start is not None)
            temporal_score = min(0.15, specific_count * 0.05)
            score += temporal_score

        # Negation contribution (important for adversarial)
        if negations:
            score += 0.1

        # Length-based adjustment (longer = potentially more important)
        words = len(text.split())
        if words > 50:
            score += 0.05
        elif words < 10:
            score -= 0.1

        return max(0.1, min(1.0, score))

    def _determine_event_time(
        self,
        default_time: datetime,
        temporal_expressions: list[TemporalExpression],
    ) -> TimeInterval:
        """Determine the event time from temporal expressions."""
        # Look for normalized absolute/relative times
        times = []
        for expr in temporal_expressions:
            if expr.normalized_start:
                times.append(expr.normalized_start)

        if times:
            # Use the most specific time found
            start = min(times)
            end = max(times) if len(times) > 1 else None

            # Check for durations
            for expr in temporal_expressions:
                if expr.duration:
                    end = start + expr.duration
                    break

            return TimeInterval(start=start, end=end)

        return TimeInterval(start=default_time)

    def _extract_concepts(
        self,
        text: str,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[str]:
        """Extract concepts/topics from text (simplified)."""
        concepts = []

        # Add entity types as concepts
        for entity in entities:
            if entity.type != EntityType.UNKNOWN:
                concepts.append(entity.type.value)

        # Add relation types as concepts
        for relation in relations:
            concepts.append(relation.predicate)

        # Simple keyword extraction
        keywords = [
            "work",
            "family",
            "friend",
            "travel",
            "food",
            "hobby",
            "health",
            "money",
            "home",
            "school",
            "meeting",
            "project",
        ]
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                concepts.append(keyword)

        return list(set(concepts))

    def _extract_causal_hints(self, text: str) -> tuple[list[str], list[str]]:
        """Extract hints about causal relationships (simplified)."""
        causes = []
        effects = []

        # Look for causal language
        causal_patterns = [
            (r"because\s+(.+?)(?:\.|,|$)", "cause"),
            (r"due to\s+(.+?)(?:\.|,|$)", "cause"),
            (r"caused by\s+(.+?)(?:\.|,|$)", "cause"),
            (r"therefore\s+(.+?)(?:\.|,|$)", "effect"),
            (r"so\s+(.+?)(?:\.|,|$)", "effect"),
            (r"leads to\s+(.+?)(?:\.|,|$)", "effect"),
            (r"results in\s+(.+?)(?:\.|,|$)", "effect"),
        ]

        import re

        for pattern, rel_type in causal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                content = match.group(1).strip()[:100]
                if rel_type == "cause":
                    causes.append(content)
                else:
                    effects.append(content)

        return causes, effects

    def _create_summary(
        self,
        text: str,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> str:
        """Create a brief summary of the text."""
        # For now, just truncate
        # In production, use LLM for summarization
        if len(text) <= 200:
            return text

        # Try to end at sentence boundary
        truncated = text[:200]
        last_period = truncated.rfind(".")
        if last_period > 100:
            return truncated[: last_period + 1]

        return truncated + "..."
