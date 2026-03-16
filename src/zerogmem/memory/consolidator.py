"""
Memory Consolidator: LLM-based inference and hypothesis management.

Performs two types of reasoning:
1. Consolidation-time: After ingesting messages, retrospects on history + new
   info to generate inferred knowledge (entity relations, facts, causal links).
2. Retrieval-time: When initial retrieval is insufficient, reasons about gaps
   and generates on-the-fly inferences to improve answers.

All inferred knowledge is stored as hypotheses in existing graph structures
(EntityEdge, Fact, CausalEdge) with metadata markers for tracking and feedback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from zerogmem.defaults import DEFAULT_LLM_MODEL

if TYPE_CHECKING:
    from zerogmem.memory.manager import MemoryManager
    from zerogmem.retriever.query_analyzer import QueryAnalysis
    from zerogmem.retriever.retriever import RetrievalResult

logger = logging.getLogger("0gmem.consolidator")

# Metadata keys used to mark hypotheses
_SOURCE_KEY = "source"
_HYPOTHESIS_KEY = "hypothesis"
_VALIDATIONS_KEY = "validations"

# LLM prompts
_CONSOLIDATION_SYSTEM = """\
You are analyzing a conversation to extract implicit knowledge. Given recent \
messages and existing context about the people involved, identify facts that \
are IMPLIED but not explicitly stated. Focus on:
- Relationships between people (who knows whom, how they're connected)
- Preferences and interests (what people like, dislike, are interested in)
- Life circumstances (where they live, what they do, their situation)
- Causal connections (what led to what, cause and effect)

Return ONLY facts that require inference - do NOT repeat explicitly stated info.
Output a JSON array of objects. Each object must have exactly these fields:
  subject (str), predicate (str), object (str), confidence (float 0-1), reasoning (str)
Return ONLY the JSON array, no other text."""

_CONSOLIDATION_USER = """\
Recent messages:
{recent_messages}

Known context about {entity_names}:
{entity_context}

Existing facts:
{existing_facts}

What new knowledge can you infer from these messages combined with existing context?
Return up to {max_inferences} inferences as a JSON array."""

_RETRIEVAL_REASONING_SYSTEM = """\
You are helping answer a question about a conversation. The initial search \
returned some results but they may be insufficient. Analyze what's missing \
and suggest:
1. Additional search queries that might find the missing information
2. Inferences that connect the retrieved information to the question

Output a JSON object with exactly these fields:
  additional_queries (list of strings), \
inferences (list of objects with: subject, predicate, object, confidence, reasoning)
Return ONLY the JSON object, no other text."""

_RETRIEVAL_REASONING_USER = """\
Question: {query}

Initial search results:
{initial_results}

Entity context:
{entity_context}

What additional searches or inferences would help answer this question?"""


@dataclass
class ConsolidatorConfig:
    """Configuration for memory consolidation."""

    enabled: bool = True
    consolidation_interval: int = 5
    max_context_messages: int = 20
    max_history_facts: int = 15
    retrieval_reasoning_threshold: float = 0.4
    hypothesis_initial_confidence: float = 0.6
    confidence_boost: float = 0.15
    confidence_penalty: float = 0.3
    decay_rate: float = 0.05
    min_confidence_threshold: float = 0.1
    max_inferences_per_consolidation: int = 5
    llm_model: str = DEFAULT_LLM_MODEL


class MemoryConsolidator:
    """LLM-based memory consolidation and retrieval-time reasoning.

    All methods are no-ops when ``llm_client`` is *None*, allowing the rest
    of the system to function without an LLM dependency.
    """

    def __init__(
        self,
        memory: MemoryManager,
        llm_client: Any | None = None,
        config: ConsolidatorConfig | None = None,
    ) -> None:
        self.memory = memory
        self.llm_client = llm_client
        self.config = config or ConsolidatorConfig()
        self._message_count: int = 0
        self._hypothesis_ids: list[str] = []

    @property
    def _enabled(self) -> bool:
        return self.config.enabled and self.llm_client is not None

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, recent_message_ids: list[str]) -> list[str]:
        """Run LLM consolidation over recent messages.

        Returns list of newly created hypothesis IDs.
        """
        if not self._enabled or not recent_message_ids:
            return []

        try:
            # 1. Gather recent messages
            messages_text = self._gather_messages(recent_message_ids)
            if not messages_text:
                return []

            # 2. Gather entity context
            entity_names, entity_context = self._gather_entity_context(recent_message_ids)

            # 3. Gather existing facts
            existing_facts = self._gather_existing_facts(entity_names)

            # 4. Call LLM
            user_prompt = _CONSOLIDATION_USER.format(
                recent_messages=messages_text,
                entity_names=", ".join(entity_names) if entity_names else "the participants",
                entity_context=entity_context,
                existing_facts=existing_facts,
                max_inferences=self.config.max_inferences_per_consolidation,
            )

            inferences = self._call_llm(_CONSOLIDATION_SYSTEM, user_prompt)
            if not inferences:
                return []

            # 5. Store inferences as hypotheses
            created_ids = self._store_inferences(inferences, source="consolidation")

            # 6. Validate existing hypotheses against new messages
            self._check_existing_hypotheses(recent_message_ids)

            logger.info(
                "Consolidation produced %d hypotheses from %d messages",
                len(created_ids),
                len(recent_message_ids),
            )
            return created_ids

        except Exception:
            logger.exception("Consolidation failed")
            return []

    # ------------------------------------------------------------------
    # Retrieval-time reasoning
    # ------------------------------------------------------------------

    def reason_at_retrieval(
        self,
        query: str,
        initial_results: list[RetrievalResult],
        analysis: QueryAnalysis,
    ) -> list[RetrievalResult]:
        """Generate additional results via LLM reasoning when initial retrieval is weak."""
        from zerogmem.retriever.retriever import RetrievalResult as RR

        if not self._enabled:
            return []

        # Check if reasoning is needed
        max_score = max((r.score for r in initial_results), default=0.0)
        if max_score >= self.config.retrieval_reasoning_threshold:
            return []

        try:
            # Format initial results
            results_text = "\n".join(
                f"- [{r.score:.2f}] {r.content[:200]}" for r in initial_results[:10]
            )

            # Entity context
            entity_context = ""
            if analysis.target_entity:
                nodes = self.memory.graph.entity_graph.find_by_name(
                    analysis.target_entity, fuzzy=True
                )
                if nodes:
                    profile = self.memory.graph.entity_graph.get_entity_profile(nodes[0].id)
                    entity_context = json.dumps(profile, default=str)[:1000]

            user_prompt = _RETRIEVAL_REASONING_USER.format(
                query=query,
                initial_results=results_text or "(no results found)",
                entity_context=entity_context or "(none)",
            )

            raw = self._call_llm_raw(_RETRIEVAL_REASONING_SYSTEM, user_prompt)
            if not raw:
                return []

            parsed = _parse_json(raw)
            if not isinstance(parsed, dict):
                return []

            new_results: list[RetrievalResult] = []

            # Execute additional queries
            additional_queries: list[str] = parsed.get("additional_queries", [])
            for extra_q in additional_queries[:3]:
                if not isinstance(extra_q, str):
                    continue
                # BM25 search
                if self.memory.bm25 and self.memory.bm25.total_docs > 0:
                    hits = self.memory.bm25.search(extra_q, top_k=5, use_expansion=True)
                    for doc_id, score, content in hits:
                        new_results.append(
                            RR(
                                id=doc_id,
                                content=content,
                                score=score * 0.8,  # slightly discount
                                source="retrieval_reasoning",
                                metadata={"reasoning_query": extra_q},
                            )
                        )

            # Store inferences
            inferences = parsed.get("inferences", [])
            if inferences:
                created_ids = self._store_inferences(inferences, source="retrieval_reasoning")
                for cid in created_ids:
                    # Find the stored content
                    fact = self.memory.semantic_memory.get_fact(cid)
                    if fact:
                        new_results.append(
                            RR(
                                id=cid,
                                content=fact.content,
                                score=fact.confidence * 0.5,
                                source="retrieval_reasoning",
                                metadata={"hypothesis": True},
                            )
                        )

            logger.info(
                "Retrieval reasoning produced %d additional results for: %s",
                len(new_results),
                query[:60],
            )
            return new_results

        except Exception:
            logger.exception("Retrieval-time reasoning failed")
            return []

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def validate_hypothesis(self, hypothesis_id: str, correct: bool) -> None:
        """Apply feedback to a hypothesis."""
        validation_entry = {
            "correct": correct,
            "timestamp": datetime.now().isoformat(),
        }

        # Check entity edges
        edge = self.memory.graph.entity_graph.edges.get(hypothesis_id)
        if edge and edge.metadata.get(_HYPOTHESIS_KEY):
            if correct:
                edge.confidence = min(1.0, edge.confidence + self.config.confidence_boost)
                edge.last_confirmed = datetime.now()
            else:
                edge.confidence = max(0.0, edge.confidence - self.config.confidence_penalty)
                if edge.confidence < self.config.min_confidence_threshold:
                    edge.negated = True
            edge.metadata.setdefault(_VALIDATIONS_KEY, []).append(validation_entry)
            return

        # Check semantic facts
        fact = self.memory.semantic_memory.get_fact(hypothesis_id)
        if fact and fact.metadata.get(_HYPOTHESIS_KEY):
            if correct:
                fact.confidence = min(1.0, fact.confidence + self.config.confidence_boost)
                fact.last_confirmed = datetime.now()
                fact.confirmation_count += 1
            else:
                fact.confidence = max(0.0, fact.confidence - self.config.confidence_penalty)
                if fact.confidence < self.config.min_confidence_threshold:
                    fact.negated = True
            fact.metadata.setdefault(_VALIDATIONS_KEY, []).append(validation_entry)
            return

        # Check causal edges
        causal_edge = self.memory.graph.causal_graph.edges.get(hypothesis_id)
        if causal_edge and causal_edge.metadata.get(_HYPOTHESIS_KEY):
            if correct:
                causal_edge.confidence = min(
                    1.0, causal_edge.confidence + self.config.confidence_boost
                )
            else:
                causal_edge.confidence = max(
                    0.0, causal_edge.confidence - self.config.confidence_penalty
                )
            causal_edge.metadata.setdefault(_VALIDATIONS_KEY, []).append(validation_entry)

    def decay_hypotheses(self) -> None:
        """Decay unvalidated hypotheses. Called at end of session."""
        if not self.config.enabled:
            return

        for hid in list(self._hypothesis_ids):
            # Entity edges
            edge = self.memory.graph.entity_graph.edges.get(hid)
            if edge and edge.metadata.get(_HYPOTHESIS_KEY):
                validations = edge.metadata.get(_VALIDATIONS_KEY, [])
                if not validations:
                    edge.confidence = max(0.0, edge.confidence - self.config.decay_rate)
                    if edge.confidence < self.config.min_confidence_threshold:
                        edge.negated = True
                continue

            # Semantic facts
            fact = self.memory.semantic_memory.get_fact(hid)
            if fact and fact.metadata.get(_HYPOTHESIS_KEY):
                validations = fact.metadata.get(_VALIDATIONS_KEY, [])
                if not validations:
                    fact.confidence = max(0.0, fact.confidence - self.config.decay_rate)
                    if fact.confidence < self.config.min_confidence_threshold:
                        fact.negated = True
                continue

            # Causal edges
            causal_edge = self.memory.graph.causal_graph.edges.get(hid)
            if causal_edge and causal_edge.metadata.get(_HYPOTHESIS_KEY):
                validations = causal_edge.metadata.get(_VALIDATIONS_KEY, [])
                if not validations:
                    causal_edge.confidence = max(
                        0.0, causal_edge.confidence - self.config.decay_rate
                    )

        logger.debug("Decayed %d hypotheses", len(self._hypothesis_ids))

    def get_hypotheses(
        self,
        entity_name: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Query current hypotheses for inspection."""
        results: list[dict[str, Any]] = []

        # Entity edge hypotheses
        for eid, edge in self.memory.graph.entity_graph.edges.items():
            if not edge.metadata.get(_HYPOTHESIS_KEY):
                continue
            if edge.confidence < min_confidence:
                continue
            if entity_name:
                source_node = self.memory.graph.entity_graph.nodes.get(edge.source_id)
                target_node = self.memory.graph.entity_graph.nodes.get(edge.target_id)
                names = []
                if source_node:
                    names.append(source_node.name.lower())
                if target_node:
                    names.append(target_node.name.lower())
                if entity_name.lower() not in names:
                    continue
            results.append(
                {
                    "id": eid,
                    "type": "entity_relation",
                    "content": f"{edge.source_id} {edge.relation} {edge.target_id}",
                    "confidence": edge.confidence,
                    "negated": edge.negated,
                    "source": edge.metadata.get(_SOURCE_KEY, ""),
                    "validations": edge.metadata.get(_VALIDATIONS_KEY, []),
                }
            )

        # Fact hypotheses
        for fid, fact in self.memory.semantic_memory.facts.items():
            if not fact.metadata.get(_HYPOTHESIS_KEY):
                continue
            if fact.confidence < min_confidence:
                continue
            if entity_name and entity_name.lower() not in fact.subject.lower():
                continue
            results.append(
                {
                    "id": fid,
                    "type": "fact",
                    "content": fact.content,
                    "confidence": fact.confidence,
                    "negated": fact.negated,
                    "source": fact.metadata.get(_SOURCE_KEY, ""),
                    "validations": fact.metadata.get(_VALIDATIONS_KEY, []),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_messages(self, message_ids: list[str]) -> str:
        """Format recent messages for the LLM prompt."""
        lines: list[str] = []
        for mid in message_ids:
            mem = self.memory.graph.memories.get(mid)
            if mem:
                speaker = mem.speaker or "unknown"
                lines.append(f"[{speaker}]: {mem.content}")
        return "\n".join(lines)

    def _gather_entity_context(
        self, message_ids: list[str]
    ) -> tuple[list[str], str]:
        """Extract entity names and their graph context."""
        entity_names: set[str] = set()
        for mid in message_ids:
            mem = self.memory.graph.memories.get(mid)
            if mem:
                entity_names.update(mem.entity_names)

        context_parts: list[str] = []
        for name in sorted(entity_names):
            nodes = self.memory.graph.entity_graph.find_by_name(name, fuzzy=True)
            if not nodes:
                continue
            node = nodes[0]
            profile = self.memory.graph.entity_graph.get_entity_profile(node.id)
            relations = profile.get("relations", [])[:5]
            if relations:
                rel_strs = [
                    f"  - {r['relation']}: {r['entity']}" for r in relations
                ]
                context_parts.append(f"{name}:\n" + "\n".join(rel_strs))
            elif node.attributes:
                attrs = ", ".join(
                    f"{k}={v}" for k, v in list(node.attributes.items())[:5]
                )
                context_parts.append(f"{name}: {attrs}")

        return sorted(entity_names), "\n".join(context_parts) if context_parts else "(none)"

    def _gather_existing_facts(self, entity_names: list[str]) -> str:
        """Get existing facts about mentioned entities."""
        facts: list[str] = []
        for name in entity_names:
            for fact in self.memory.semantic_memory.get_facts_about(
                name, min_confidence=0.3
            )[:3]:
                facts.append(f"- {fact.content} (confidence: {fact.confidence:.1f})")
        if not facts:
            return "(none)"
        return "\n".join(facts[: self.config.max_history_facts])

    def _call_llm(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        """Call LLM and parse JSON array response."""
        raw = self._call_llm_raw(system_prompt, user_prompt)
        if not raw:
            return []
        parsed = _parse_json(raw)
        if isinstance(parsed, list):
            return parsed
        return []

    def _call_llm_raw(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM and return raw string response."""
        if not self.llm_client:
            return ""
        try:
            from zerogmem.defaults import llm_chat_kwargs

            response = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.config.llm_model, max_tokens=1500, temperature=0.3),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception:
            logger.exception("LLM call failed")
            return ""

    def _store_inferences(
        self, inferences: list[dict[str, Any]], source: str
    ) -> list[str]:
        """Store parsed inferences as hypotheses in the graph."""
        created: list[str] = []

        for inf in inferences[: self.config.max_inferences_per_consolidation]:
            if not isinstance(inf, dict):
                continue

            subject = str(inf.get("subject", "")).strip()
            predicate = str(inf.get("predicate", "")).strip()
            obj = str(inf.get("object", "")).strip()
            confidence = float(inf.get("confidence", self.config.hypothesis_initial_confidence))

            if not subject or not predicate or not obj:
                continue

            # Clamp confidence
            confidence = min(1.0, max(0.1, confidence))
            confidence = min(confidence, self.config.hypothesis_initial_confidence)

            # Store as semantic fact
            fact_id = self.memory.add_fact(
                subject=subject,
                predicate=predicate,
                obj=obj,
                category="inferred",
            )

            # Update the fact with hypothesis metadata
            fact = self.memory.semantic_memory.get_fact(fact_id)
            if fact:
                fact.confidence = confidence
                fact.metadata[_SOURCE_KEY] = source
                fact.metadata[_HYPOTHESIS_KEY] = True
                fact.metadata["reasoning"] = str(inf.get("reasoning", ""))

            self._hypothesis_ids.append(fact_id)
            created.append(fact_id)

            logger.debug(
                "Stored hypothesis: %s %s %s (%.2f) [%s]",
                subject,
                predicate,
                obj,
                confidence,
                source,
            )

        return created

    def _check_existing_hypotheses(self, recent_message_ids: list[str]) -> None:
        """Check if new messages confirm or contradict existing hypotheses."""
        if not self._hypothesis_ids or not self._enabled:
            return

        # Only check if we have enough hypotheses to justify the LLM call
        active_hypotheses = []
        for hid in self._hypothesis_ids:
            fact = self.memory.semantic_memory.get_fact(hid)
            if fact and not fact.negated and fact.metadata.get(_HYPOTHESIS_KEY):
                active_hypotheses.append(fact)

        if not active_hypotheses:
            return

        # Gather recent messages
        messages_text = self._gather_messages(recent_message_ids)
        if not messages_text:
            return

        hypotheses_text = "\n".join(
            f"- [{h.id[:8]}] {h.content} (confidence: {h.confidence:.2f})"
            for h in active_hypotheses[:10]
        )

        system_prompt = (
            "You are checking if new conversation messages confirm or contradict "
            "existing hypotheses. For each hypothesis, determine if the new messages "
            "provide evidence for or against it.\n"
            "Output a JSON array of objects with: hypothesis_id (first 8 chars), "
            "verdict (\"confirm\" or \"contradict\" or \"neutral\").\n"
            "Return ONLY the JSON array."
        )
        user_prompt = (
            f"New messages:\n{messages_text}\n\n"
            f"Existing hypotheses:\n{hypotheses_text}\n\n"
            "Which hypotheses are confirmed or contradicted by the new messages?"
        )

        results = self._call_llm(system_prompt, user_prompt)
        for result in results:
            if not isinstance(result, dict):
                continue
            hid_prefix = str(result.get("hypothesis_id", ""))
            verdict = str(result.get("verdict", "neutral"))
            if verdict == "neutral":
                continue

            # Find matching hypothesis
            for h in active_hypotheses:
                if h.id.startswith(hid_prefix):
                    self.validate_hypothesis(h.id, correct=(verdict == "confirm"))
                    break


def _parse_json(text: str) -> Any:
    """Parse JSON from LLM output, handling markdown code fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Failed to parse LLM JSON output: %s", text[:200])
        return None
