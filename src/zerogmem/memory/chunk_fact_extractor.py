"""
Per-Chunk LLM Fact Extraction: Extract fine-grained atomic facts from
conversation chunks using an LLM.

For each chunk of messages, the LLM:
1. Resolves ALL pronouns to concrete entity names
2. Extracts every atomic fact at fine granularity
3. Classifies each fact as general_fact (objective) or speaker_opinion (subjective)

Extracted facts are stored as UnifiedMemoryItems with embeddings and BM25
indexing, making them findable by semantic and keyword search.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zerogmem.defaults import DEFAULT_LLM_MODEL, llm_chat_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

logger = logging.getLogger("0gmem.chunk_fact_extractor")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkFactExtractorConfig:
    """Configuration for per-chunk LLM fact extraction."""

    enabled: bool = False
    llm_model: str = DEFAULT_LLM_MODEL
    max_facts_per_chunk: int = 40
    max_completion_tokens: int = 4000
    temperature: float = 0.0
    max_retries: int = 2
    backoff_base: float = 1.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedChunkFact:
    """An atomic fact extracted from a conversation chunk."""

    subject: str
    predicate: str
    object: str
    fact_type: str  # "general_fact" or "speaker_opinion"
    speaker: str  # Who stated/implied this
    category: str = ""  # identity, preference, event, personality_trait, etc.
    negated: bool = False
    confidence: float = 1.0
    original_text: str = ""


# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You extract atomic facts from a conversation chunk. For EACH fact:

1. RESOLVE ALL PRONOUNS to concrete names using the participant list.
   - "I"/"my" → the speaker's name
   - "you"/"your" → the other participant (the person being addressed)
   - "he/she/they" → the most recently referenced person
   - "we"/"our" → list both participants

2. CLASSIFY each fact:
   - "general_fact": Objectively verifiable (events, attributes, relationships, locations, dates, possessions)
   - "speaker_opinion": Subjective — compliments, personality descriptions, judgments, feelings about someone

3. Extract EVERY distinct fact, no matter how small:
   - Personal attributes (job, hobbies, family, age, identity)
   - Events and activities (what happened, when, where)
   - Relationships (who knows whom, family ties, pets)
   - Preferences (likes, dislikes, favorites)
   - Personality traits described by others → speaker_opinion, category "personality_trait"
   - Plans and intentions
   - Possessions (things owned, gifts received)
   - Negated facts (things explicitly said to NOT be true)

Return ONLY a JSON array. Each element:
{
  "subject": "Resolved name, NEVER a pronoun",
  "predicate": "verb or relation phrase",
  "object": "resolved value, NEVER a pronoun",
  "fact_type": "general_fact" or "speaker_opinion",
  "speaker": "who said this",
  "category": "identity|attribute|preference|event|relationship|location|activity|personality_trait|possession|plan|family|negation",
  "negated": false
}

Rules:
- Keep predicates short and consistent (e.g., "is", "has", "likes", "works as", "lives in", "has personality trait")
- For personality traits from compliments like "you're so kind", extract: subject=<addressee>, predicate="has personality trait", object="kind", fact_type="speaker_opinion"
- For "you found your true self" → subject=<addressee>, predicate="has quality", object="authentic", fact_type="speaker_opinion"
- One fact per JSON object — split compound statements
- If nothing to extract, return []
- CRITICAL: Extract ONLY what is explicitly stated. NEVER infer counts, totals, or quantities that are not directly stated.
  - "The 2 younger kids love nature" → extract "Melanie's 2 younger kids love nature", NOT "Melanie has 2 kids"
  - "my son got into an accident" → extract the event, NOT "Melanie has 1 son"
  - Partial references like "the younger ones" do not tell you the total number of children
- CRITICAL: Preserve the original meaning precisely. Do not rephrase in a way that changes or adds meaning.
  - "We roasted marshmallows" → "roasted marshmallows", NOT "enjoys roasting marshmallows regularly"
  - "It was a nice way to relax after the road trip" → preserve the temporal relationship, do not drop it
- Extract FAMILY STRUCTURE facts when they can be inferred from pronoun relationships:
  - "we reassured them and explained their brother would be OK" → the "them" (kids) and "their brother" are DIFFERENT people → extract: "Melanie has at least 3 children: the brother involved in the accident, plus at least 2 other kids who were reassured about him"
  - "The 2 younger kids" implies older siblings exist → extract: "Melanie has 2 younger children (implying at least one older child)"
  - Pay special attention to sibling references that reveal family size"""

_USER_PROMPT = """\
Participants: {participants}

Conversation messages:
{messages}

Extract all atomic facts as a JSON array:"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ChunkFactExtractor:
    """Extract fine-grained facts from conversation chunks using LLM."""

    def __init__(
        self,
        llm_client: Any,
        config: ChunkFactExtractorConfig | None = None,
    ):
        self._client = llm_client
        self.config = config or ChunkFactExtractorConfig()

    def extract_facts(
        self,
        messages: list[dict[str, Any]],
        participants: list[str],
    ) -> list[ExtractedChunkFact]:
        """Extract atomic facts from a chunk of messages.

        Args:
            messages: List of dicts with 'speaker' and 'content' keys.
            participants: Names of conversation participants.

        Returns:
            List of extracted facts with resolved pronouns.
        """
        if not messages or not self._client:
            return []

        # Format messages
        lines: list[str] = []
        for i, msg in enumerate(messages, 1):
            speaker = msg.get("speaker", "Unknown")
            lines.append(f"[{i}] [{speaker}]: {msg['content']}")
        formatted = "\n".join(lines)

        user_prompt = _USER_PROMPT.format(
            participants=", ".join(participants),
            messages=formatted,
        )

        # Call LLM
        raw = self._call_llm(user_prompt)
        if raw is None:
            return []

        # Parse JSON
        facts_data = self._parse_json_array(raw)
        if not facts_data:
            return []

        # Convert to ExtractedChunkFact objects
        facts: list[ExtractedChunkFact] = []
        for item in facts_data[: self.config.max_facts_per_chunk]:
            subject = item.get("subject", "").strip()
            predicate = item.get("predicate", "").strip()
            obj = item.get("object", "").strip()
            if not subject or not predicate or not obj:
                continue

            facts.append(
                ExtractedChunkFact(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    fact_type=item.get("fact_type", "general_fact"),
                    speaker=item.get("speaker", ""),
                    category=item.get("category", ""),
                    negated=bool(item.get("negated", False)),
                )
            )

        logger.info(
            "Extracted %d facts from chunk (%d messages, %d participants)",
            len(facts),
            len(messages),
            len(participants),
        )
        return facts

    def _call_llm(self, user_prompt: str) -> str | None:
        """Call LLM with retries."""
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.chat.completions.create(
                    **llm_chat_kwargs(
                        self.config.llm_model,
                        max_tokens=self.config.max_completion_tokens,
                        temperature=self.config.temperature,
                    ),
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content: str = response.choices[0].message.content.strip()
                return content
            except Exception:
                sleep_s = min(30.0, self.config.backoff_base ** attempt)
                logger.warning(
                    "LLM fact extraction attempt %d failed, retrying in %.1fs",
                    attempt + 1,
                    sleep_s,
                    exc_info=True,
                )
                time.sleep(sleep_s)
        return None

    @staticmethod
    def _parse_json_array(text: str) -> list[dict[str, Any]]:
        """Parse a JSON array from LLM output, handling markdown fences."""
        # Strip markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Try direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try finding the array by bracket matching
        start = cleaned.find("[")
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "[":
                    depth += 1
                elif cleaned[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(cleaned[start : i + 1])
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            break

        logger.warning("Failed to parse JSON array from LLM fact extraction output")
        return []
