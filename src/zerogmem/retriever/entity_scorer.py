"""
Entity Scorer: Graduated entity-aware scoring for retrieval results.

Replaces binary entity filtering with nuanced scoring that:
- Boosts messages FROM the target entity (speaker match)
- Boosts first-person statements ("I went", "my X")
- Handles secondary entities (mentioned but not speakers)
- Supports adversarial mode (strict isolation)
- Provides session diversity for multi-hop questions
- Applies recency boosting for "recently"/"latest" queries
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityScoringConfig:
    """Configuration for entity-aware scoring."""

    enabled: bool = True  # Entity-aware scoring for better retrieval
    speaker_match_boost: float = 1.5  # Messages FROM target entity
    first_person_boost: float = 1.3  # "I went", "my X" statements
    adversarial_first_person_boost: float = 1.5  # Stronger in adversarial mode
    adversarial_speaker_boost: float = 2.0  # Strong boost in adversarial mode
    secondary_entity_boost: float = 1.4  # Messages ABOUT a non-speaker entity
    secondary_multi_mention_boost: float = 1.2  # Entity name appears 2+ times
    other_speaker_about_target: float = 0.8  # Other speaker mentioning target
    adversarial_mention_factor: float = 0.5  # Mentions in adversarial mode
    irrelevant_entity_factor: float = 0.3  # Demotion for unrelated entities
    min_entity_results: int = 3  # Min results before falling back
    # Multi-hop session diversity
    session_diversity_boost: float = 1.3  # Boost uncovered sessions
    session_coverage_boost: float = 1.2  # Extra boost until min_sessions covered
    min_session_coverage: int = 3  # Minimum sessions to cover for multi-hop
    # Recency boosting
    recency_max_boost: float = 1.8  # Max boost for latest session


# First-person markers indicating the speaker is talking about themselves
FIRST_PERSON_MARKERS = [
    "i am",
    "i'm",
    "my ",
    "i have",
    "i made",
    "i went",
    "i did",
    "i like",
    "i love",
    "i play",
]

# Keywords that indicate the query asks about recent events
RECENCY_KEYWORDS = ["recently", "latest", "most recent", "last time", "newest"]


@dataclass
class ScoredResult:
    """A retrieval result with entity-aware score adjustment."""

    id: str
    original_score: float
    adjusted_score: float
    content: str
    speaker: str = ""
    session_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityScorer:
    """
    Graduated entity-aware scoring for retrieval results.

    Instead of binary keep/discard, applies multiplicative score
    adjustments based on entity relevance:
    - Speaker match: messages FROM the target entity get boosted
    - First-person: "I went", "my X" get extra boost
    - Secondary entities: messages ABOUT non-speaker entities get boosted
    - Adversarial: strict mode only includes target's own messages
    - Session diversity: ensures multi-hop coverage across sessions
    - Recency: boosts later sessions for "recently" queries
    """

    def __init__(self, config: EntityScoringConfig | None = None) -> None:
        self.config = config or EntityScoringConfig()

    def score_results(
        self,
        results: list[ScoredResult],
        target_entity: str,
        *,
        is_adversarial: bool = False,
        is_multi_hop: bool = False,
        query: str = "",
        known_speakers: set[str] | None = None,
        related_entities: list[str] | None = None,
    ) -> list[ScoredResult]:
        """
        Apply entity-aware scoring to retrieval results.

        Args:
            results: Retrieval results with scores, content, speaker, session_idx
            target_entity: Entity the question is about
            is_adversarial: If True, use strict entity isolation
            is_multi_hop: If True, apply session diversity boosting
            query: Original query text (for recency detection)
            known_speakers: Set of known speaker names in the conversation.
                Used to determine if target_entity is a secondary entity
                (mentioned but never speaks). If None, secondary entity
                detection is skipped.
            related_entities: Other entities mentioned in the query. Messages
                from these entities are not penalized as irrelevant.

        Returns:
            Rescored results sorted by adjusted_score descending
        """
        if not results or not target_entity:
            return results

        target_lower = target_entity.lower()
        related_lower = {e.lower() for e in (related_entities or [])}

        # Determine if target is a secondary entity (non-speaker)
        is_secondary = self._is_secondary_entity(target_lower, known_speakers)

        # Determine conversation partner dynamically
        other_entity = self._get_conversation_partner(target_lower, known_speakers)

        # Phase 1: Entity relevance scoring
        entity_scored = []
        cross_entity = []  # Fallback for results not mentioning target

        for r in results:
            speaker_lower = r.speaker.lower() if r.speaker else ""
            content_lower = r.content.lower()

            is_from_target = bool(speaker_lower) and target_lower in speaker_lower
            is_from_other = (
                bool(speaker_lower) and bool(other_entity) and other_entity in speaker_lower
            )
            mentions_target = target_lower in content_lower

            score = r.original_score
            scored = ScoredResult(
                id=r.id,
                original_score=r.original_score,
                adjusted_score=score,
                content=r.content,
                speaker=r.speaker,
                session_idx=r.session_idx,
                metadata=r.metadata,
            )

            if is_adversarial:
                score = self._score_adversarial(
                    score,
                    content_lower,
                    is_from_target=is_from_target,
                    is_from_other=is_from_other,
                    mentions_target=mentions_target,
                )
                if score is not None:
                    scored.adjusted_score = score
                    entity_scored.append(scored)
                # else: filtered out entirely
            else:
                is_from_related = bool(speaker_lower) and any(
                    rel in speaker_lower for rel in related_lower
                )
                score = self._score_normal(
                    score,
                    content_lower,
                    is_from_target=is_from_target,
                    mentions_target=mentions_target,
                    is_secondary=is_secondary,
                    target_lower=target_lower,
                    is_from_related=is_from_related,
                )
                if score is not None:
                    scored.adjusted_score = score
                    entity_scored.append(scored)
                else:
                    # Irrelevant — goes to fallback pool
                    scored.adjusted_score = (
                        r.original_score * self.config.irrelevant_entity_factor
                    )
                    cross_entity.append(scored)

        # Merge: use entity results if sufficient, otherwise add fallback
        if is_adversarial:
            # Strict: only entity-filtered results
            final = entity_scored if entity_scored else []
        elif len(entity_scored) >= self.config.min_entity_results:
            final = entity_scored
        else:
            # Merge with fallback, keeping entity results ranked higher
            seen = {r.id for r in entity_scored}
            for r in cross_entity:
                if r.id not in seen:
                    entity_scored.append(r)
            final = entity_scored

        # Phase 2: Session diversity for multi-hop
        if is_multi_hop and final:
            final = self._apply_session_diversity(final)

        # Phase 3: Recency boosting
        if final and query:
            query_lower = query.lower()
            if any(kw in query_lower for kw in RECENCY_KEYWORDS):
                final = self._apply_recency_boost(final)

        # Sort by adjusted score
        final.sort(key=lambda r: r.adjusted_score, reverse=True)
        return final

    def _score_adversarial(
        self,
        score: float,
        content_lower: str,
        *,
        is_from_target: bool,
        is_from_other: bool,
        mentions_target: bool,
    ) -> float | None:
        """Score a result in adversarial mode. Returns None to exclude."""
        if is_from_target:
            adjusted = score * self.config.adversarial_speaker_boost
            if self._has_first_person(content_lower):
                adjusted *= self.config.adversarial_first_person_boost
            return adjusted
        elif is_from_other:
            # Exclude entirely — prevents LLM from seeing other entity's info
            return None
        elif mentions_target:
            # System/narrator messages mentioning target — low weight
            return score * self.config.adversarial_mention_factor
        else:
            return None

    def _score_normal(
        self,
        score: float,
        content_lower: str,
        *,
        is_from_target: bool,
        mentions_target: bool,
        is_secondary: bool,
        target_lower: str,
        is_from_related: bool = False,
    ) -> float | None:
        """Score a result in normal mode. Returns None for irrelevant."""
        if is_from_target:
            adjusted = score * self.config.speaker_match_boost
            if self._has_first_person(content_lower):
                adjusted *= self.config.first_person_boost
            return adjusted
        elif mentions_target:
            if is_secondary:
                # For secondary entities (who don't speak), messages ABOUT them
                # are the primary source — boost them
                adjusted = score * self.config.secondary_entity_boost
                name_mentions = content_lower.count(target_lower)
                if name_mentions >= 2:
                    adjusted *= self.config.secondary_multi_mention_boost
                return adjusted
            else:
                # For primary entities, messages about them from others are secondary
                return score * self.config.other_speaker_about_target
        elif is_from_related:
            # Message from a related entity mentioned in the query —
            # don't penalize as irrelevant, keep original score
            return score
        else:
            return None  # Irrelevant

    def _apply_session_diversity(self, results: list[ScoredResult]) -> list[ScoredResult]:
        """Boost results from uncovered sessions for multi-hop coverage."""
        covered_sessions: set[int] = set()
        reranked = []

        for r in results:
            adjusted = r.adjusted_score
            if r.session_idx not in covered_sessions:
                adjusted *= self.config.session_diversity_boost
                if len(covered_sessions) < self.config.min_session_coverage:
                    adjusted *= self.config.session_coverage_boost
            r.adjusted_score = adjusted
            reranked.append(r)
            covered_sessions.add(r.session_idx)

        return reranked

    def _apply_recency_boost(self, results: list[ScoredResult]) -> list[ScoredResult]:
        """Boost results from later sessions for recency queries."""
        max_session = max((r.session_idx for r in results), default=0)
        if max_session <= 0:
            return results

        for r in results:
            recency_factor = 1.0 + (r.session_idx / max_session) * (
                self.config.recency_max_boost - 1.0
            )
            r.adjusted_score *= recency_factor

        return results

    @staticmethod
    def _has_first_person(content_lower: str) -> bool:
        """Check if content contains first-person markers."""
        return any(marker in content_lower for marker in FIRST_PERSON_MARKERS)

    @staticmethod
    def _is_secondary_entity(
        target_lower: str, known_speakers: set[str] | None
    ) -> bool:
        """Check if target is a secondary entity (mentioned but never speaks)."""
        if known_speakers is None:
            return False
        return all(target_lower not in s.lower() for s in known_speakers)

    @staticmethod
    def _get_conversation_partner(
        target_lower: str, known_speakers: set[str] | None
    ) -> str | None:
        """Find the conversation partner of the target entity."""
        if known_speakers is None:
            return None
        for speaker in known_speakers:
            if target_lower not in speaker.lower():
                return speaker.lower()
        return None
