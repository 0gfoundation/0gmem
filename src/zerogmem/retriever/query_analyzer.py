"""
Query Analyzer: Understands user queries to route to appropriate retrieval.

Classifies query intent, extracts entities, and determines reasoning type.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from zerogmem.defaults import DEFAULT_LLM_MODEL, llm_chat_kwargs

from zerogmem.encoder.temporal_extractor import TemporalExpression, TemporalExtractor

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""

    FACTUAL = "factual"  # "What is X's favorite color?"
    TEMPORAL = "temporal"  # "When did X happen?"
    CAUSAL = "causal"  # "Why did X happen?"
    RELATIONAL = "relational"  # "Who does X know?"
    PREFERENCE = "preference"  # "Does X like Y?"
    EVENT = "event"  # "What happened at/during X?"
    COMPARISON = "comparison"  # "How is X different from Y?"
    LIST = "list"  # "List all X that Y"
    VERIFICATION = "verification"  # "Is it true that X?"


class ReasoningType(Enum):
    """Types of reasoning required."""

    SINGLE_HOP = "single_hop"  # Direct fact lookup
    MULTI_HOP = "multi_hop"  # Connecting multiple facts
    TEMPORAL = "temporal"  # Time-based reasoning
    CAUSAL = "causal"  # Cause-effect reasoning
    COMMONSENSE = "commonsense"  # World knowledge integration
    ADVERSARIAL = "adversarial"  # Testing for false information


class TemporalScope(Enum):
    """Temporal scope of the query."""

    POINT = "point"  # Specific moment
    RANGE = "range"  # Time range
    RELATIVE = "relative"  # Before/after something
    NONE = "none"  # No temporal constraint


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""

    original_query: str
    intent: QueryIntent
    reasoning_type: ReasoningType
    entities: list[str]
    temporal_scope: TemporalScope
    temporal_expressions: list[TemporalExpression]
    keywords: list[str]
    target_entity: str | None = None  # The entity the question is about (subject)
    is_negation_check: bool = False  # Is this checking if something is NOT true?
    expected_answer_type: str = "text"  # text, yes_no, list, date, number
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """
    Analyzes queries to determine intent and retrieval strategy.

    Critical for routing queries to the right retrieval method,
    especially for LoCoMo's different question types.
    """

    # Intent classification patterns
    INTENT_PATTERNS = {
        QueryIntent.TEMPORAL: [
            r"\bwhen\b",
            r"\bwhat time\b",
            r"\bwhat date\b",
            r"\bhow long\b",
            r"\bafter\b.*\bhappen",
            r"\bbefore\b.*\bhappen",
            r"\bduring\b",
        ],
        QueryIntent.CAUSAL: [
            r"\bwhy\b",
            r"\breason\b",
            r"\bcause\b",
            r"\bbecause\b",
            r"\bdue to\b",
            r"\bresult\b",
            r"\blead to\b",
        ],
        QueryIntent.RELATIONAL: [
            r"\bwho\b.*\bknow",
            r"\bwho\b.*\bfriend",
            r"\brelation",
            r"\bconnect",
            r"\bwho\b.*\bwork",
            r"\bwho\b.*\blive",
        ],
        QueryIntent.PREFERENCE: [
            r"\blike\b",
            r"\blove\b",
            r"\bhate\b",
            r"\bprefer\b",
            r"\bfavorite\b",
            r"\benjoy\b",
            r"\bdislike\b",
        ],
        QueryIntent.EVENT: [
            r"\bwhat happen",
            r"\bwhat did\b.*\bdo\b",
            r"\bdescribe\b",
            r"\btell.*about\b",
            r"\bwhat.*event\b",
        ],
        QueryIntent.COMPARISON: [
            r"\bcompare\b",
            r"\bdifferent\b",
            r"\bsimilar\b",
            r"\bsame\b.*\bas\b",
            r"\bversus\b",
            r"\bvs\b",
        ],
        QueryIntent.LIST: [
            r"\blist\b",
            r"\ball\b.*\bthat\b",
            r"\bevery\b",
            r"\beach\b",
            r"\bhow many\b",
        ],
        QueryIntent.VERIFICATION: [
            r"\bis it true\b",
            r"\bdid\b.*\breally\b",
            r"\bconfirm\b",
            r"\bverify\b",
            r"\bcorrect\b.*\bthat\b",
        ],
    }

    # Reasoning type indicators
    MULTI_HOP_INDICATORS = [
        r"\band\b",
        r"\balso\b",
        r"\bboth\b",
        r"\bwho\b.*\bwho\b",  # Nested questions
        r"\bthen\b.*\bwhat\b",
        r"\bafter that\b",
        r"\bbased on\b",  # Causal connection
        r"\baccording to\b",  # Reference to source
        r"\bexperience\b",  # Personal experience connection
        r"\brecommend",  # Recommendation (often multi-hop)
        r"\bcommon\b",  # Comparing multiple things
    ]

    TEMPORAL_REASONING_INDICATORS = [
        r"\bbefore\b",
        r"\bafter\b",
        r"\bduring\b",
        r"\bwhile\b",
        r"\bsince\b",
        r"\buntil\b",
        r"\bfirst\b.*\bthen\b",
        r"\bchronolog",
        r"\bsequence\b",
        r"\border\b",
    ]

    # Negation patterns for adversarial
    NEGATION_CHECK_PATTERNS = [
        r"\bnot\b.*\btrue\b",
        r"\bnever\b",
        r"\bfalse\b",
        r"\bwrong\b",
        r"\bincorrect\b",
        r"\bdoesn't\b",
        r"\bdoes not\b",
        r"\bdidn't\b",
        r"\bdid not\b",
    ]

    def __init__(
        self,
        llm_client: Any | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.temporal_extractor = TemporalExtractor()
        self.llm_client = llm_client
        self.llm_model = llm_model or DEFAULT_LLM_MODEL

    def analyze(self, query: str, context: dict[str, Any] | None = None) -> QueryAnalysis:
        """
        Analyze a query to determine intent and retrieval strategy.

        Uses LLM-first analysis when available, with regex enhancement to catch
        anything the LLM missed. Falls back to regex-only on LLM failure.

        Args:
            query: The user's query
            context: Optional context (e.g., conversation history)

        Returns:
            QueryAnalysis with intent, entities, and routing information
        """
        # LLM-first: try LLM analysis if available
        llm_analysis: QueryAnalysis | None = None
        if self.llm_client:
            llm_analysis = self._analyze_with_llm(query)

        # Always run regex-based extraction
        regex_analysis = self._analyze_with_regex(query, context)

        # If LLM failed or unavailable, use regex-only
        if llm_analysis is None:
            return regex_analysis

        # Merge: enhance LLM results with anything regex found that LLM missed
        return self._merge_analyses(llm_analysis, regex_analysis)

    def _analyze_with_regex(
        self, query: str, context: dict[str, Any] | None = None
    ) -> QueryAnalysis:
        """Original regex-based analysis (unchanged)."""
        query_lower = query.lower()

        # Determine intent
        intent = self._classify_intent(query_lower)

        # Determine reasoning type
        reasoning_type = self._classify_reasoning(query_lower)

        # Extract entities (simple pattern-based)
        entities = self._extract_entities(query)

        # Analyze temporal aspects
        temporal_expressions = self.temporal_extractor.extract(query)
        temporal_scope = self._determine_temporal_scope(query_lower, temporal_expressions)

        # Extract keywords
        keywords = self._extract_keywords(query_lower)

        # Check for negation/adversarial
        is_negation_check = self._is_negation_check(query_lower)

        # Determine expected answer type
        expected_answer_type = self._determine_answer_type(query_lower, intent)

        # Override reasoning type based on patterns
        if reasoning_type == ReasoningType.SINGLE_HOP:
            if temporal_scope != TemporalScope.NONE:
                reasoning_type = ReasoningType.TEMPORAL
            elif is_negation_check:
                reasoning_type = ReasoningType.ADVERSARIAL

        # Identify target entity: the entity the question is about.
        # Caller can override via context["target_entity"].
        target_entity = (context or {}).get("target_entity")
        if not target_entity:
            target_entity = self._identify_target_entity(query, entities)

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            reasoning_type=reasoning_type,
            entities=entities,
            temporal_scope=temporal_scope,
            temporal_expressions=temporal_expressions,
            keywords=keywords,
            target_entity=target_entity,
            is_negation_check=is_negation_check,
            expected_answer_type=expected_answer_type,
            metadata={
                "query_length": len(query.split()),
                "has_temporal": len(temporal_expressions) > 0,
            },
        )

    _LLM_SYSTEM_PROMPT = """You are a query analyzer for a conversational memory system. Given a question about past conversations, extract structured information to help retrieve the relevant context.

Return a JSON object with these fields:
- dates: array of explicit calendar dates in ISO format (YYYY-MM-DD). Only include dates explicitly stated in the question, not inferred ones.
- target_entity: the primary person being asked about (string or null)
- entities: all people, places, and things mentioned (array of strings)
- sub_queries: 2-3 simpler search queries that would help find the answer. Each should focus on a different aspect of the question. (array of strings)
- keywords: important terms for keyword search (array of strings)
- reasoning_type: one of "single_hop", "multi_hop", "temporal", "causal"
- expected_answer_type: one of "text", "yes_no", "date", "number", "list"

Return ONLY valid JSON, no markdown fences or extra text."""

    _REASONING_TYPE_MAP = {
        "single_hop": ReasoningType.SINGLE_HOP,
        "multi_hop": ReasoningType.MULTI_HOP,
        "temporal": ReasoningType.TEMPORAL,
        "causal": ReasoningType.CAUSAL,
    }

    def _analyze_with_llm(self, query: str) -> QueryAnalysis | None:
        """Analyze query using LLM for structured extraction.

        Returns QueryAnalysis on success, None on failure.
        """
        try:
            response = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.llm_model, max_tokens=500, temperature=0.0),
                messages=[
                    {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)

            # Parse dates into TemporalExpressions
            temporal_expressions: list[TemporalExpression] = []
            for date_str in data.get("dates") or []:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    temporal_expressions.append(
                        TemporalExpression(
                            text=date_str,
                            type="absolute",
                            normalized_start=dt,
                            normalized_end=None,
                            confidence=1.0,
                        )
                    )
                except ValueError:
                    pass

            # Map reasoning type
            rt_str = data.get("reasoning_type", "single_hop")
            reasoning_type = self._REASONING_TYPE_MAP.get(rt_str, ReasoningType.SINGLE_HOP)

            # Determine temporal scope from extracted dates
            temporal_scope = TemporalScope.POINT if temporal_expressions else TemporalScope.NONE

            # Override reasoning type if dates found
            if temporal_expressions and reasoning_type == ReasoningType.SINGLE_HOP:
                reasoning_type = ReasoningType.TEMPORAL

            # Classify intent using regex (LLM doesn't extract intent)
            intent = self._classify_intent(query.lower())

            entities = data.get("entities") or []
            target_entity = data.get("target_entity")
            keywords = data.get("keywords") or []
            expected_answer_type = data.get("expected_answer_type", "text")
            sub_queries = data.get("sub_queries") or []

            return QueryAnalysis(
                original_query=query,
                intent=intent,
                reasoning_type=reasoning_type,
                entities=entities,
                temporal_scope=temporal_scope,
                temporal_expressions=temporal_expressions,
                keywords=keywords,
                target_entity=target_entity,
                is_negation_check=self._is_negation_check(query.lower()),
                expected_answer_type=expected_answer_type,
                metadata={
                    "query_length": len(query.split()),
                    "has_temporal": len(temporal_expressions) > 0,
                    "sub_queries": sub_queries,
                    "llm_analyzed": True,
                },
            )
        except Exception as e:
            logger.warning("LLM query analysis failed, falling back to regex: %s", e)
            return None

    def _merge_analyses(
        self, llm: QueryAnalysis, regex: QueryAnalysis
    ) -> QueryAnalysis:
        """Merge LLM and regex analyses — regex enhances LLM.

        Adds anything regex found that LLM missed. Keeps LLM's higher-quality
        reasoning_type, target_entity, sub_queries, expected_answer_type.
        """
        # Union dates by (year, month, day), deduplicated
        llm_dates = {
            (e.normalized_start.year, e.normalized_start.month, e.normalized_start.day)
            for e in llm.temporal_expressions
            if e.normalized_start
        }
        merged_temporal = list(llm.temporal_expressions)
        for expr in regex.temporal_expressions:
            if expr.normalized_start:
                key = (expr.normalized_start.year, expr.normalized_start.month, expr.normalized_start.day)
                if key not in llm_dates:
                    merged_temporal.append(expr)
                    llm_dates.add(key)
            else:
                # Non-date temporal (relative, duration, etc.) — always add from regex
                merged_temporal.append(expr)

        # Union entities by lowercased name
        llm_entity_lower = {e.lower() for e in llm.entities}
        merged_entities = list(llm.entities)
        for entity in regex.entities:
            if entity.lower() not in llm_entity_lower:
                merged_entities.append(entity)
                llm_entity_lower.add(entity.lower())

        # Union keywords
        llm_kw_lower = {k.lower() for k in llm.keywords}
        merged_keywords = list(llm.keywords)
        for kw in regex.keywords:
            if kw.lower() not in llm_kw_lower:
                merged_keywords.append(kw)
                llm_kw_lower.add(kw.lower())

        # Recompute temporal scope based on merged expressions
        temporal_scope = llm.temporal_scope
        if temporal_scope == TemporalScope.NONE and merged_temporal:
            temporal_scope = self._determine_temporal_scope(
                llm.original_query.lower(), merged_temporal
            )

        # Keep LLM's reasoning_type, but upgrade to TEMPORAL if regex found dates
        reasoning_type = llm.reasoning_type
        if (
            reasoning_type == ReasoningType.SINGLE_HOP
            and temporal_scope != TemporalScope.NONE
        ):
            reasoning_type = ReasoningType.TEMPORAL

        return QueryAnalysis(
            original_query=llm.original_query,
            intent=llm.intent,
            reasoning_type=reasoning_type,
            entities=merged_entities,
            temporal_scope=temporal_scope,
            temporal_expressions=merged_temporal,
            keywords=merged_keywords,
            target_entity=llm.target_entity,
            is_negation_check=llm.is_negation_check or regex.is_negation_check,
            expected_answer_type=llm.expected_answer_type,
            confidence=llm.confidence,
            metadata=llm.metadata,
        )

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify the primary intent of the query."""
        # Check patterns in priority order
        intent_scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            # Return highest scoring intent
            return max(intent_scores, key=lambda k: intent_scores[k])

        # Default to factual
        return QueryIntent.FACTUAL

    def _classify_reasoning(self, query_lower: str) -> ReasoningType:
        """Classify the type of reasoning required."""
        # Check for multi-hop indicators
        for pattern in self.MULTI_HOP_INDICATORS:
            if re.search(pattern, query_lower):
                return ReasoningType.MULTI_HOP

        # Check for temporal reasoning
        for pattern in self.TEMPORAL_REASONING_INDICATORS:
            if re.search(pattern, query_lower):
                return ReasoningType.TEMPORAL

        # Check for causal reasoning
        if re.search(r"\bwhy\b|\bcause\b|\breason\b", query_lower):
            return ReasoningType.CAUSAL

        # Check for commonsense reasoning (comparing/generalizing across contexts)
        if re.search(
            r"\bcommon\b|\btypical\b|\busual\b|\bgenerally\b|\bboth\b.*\bcit", query_lower
        ):
            return ReasoningType.COMMONSENSE

        return ReasoningType.SINGLE_HOP

    def _extract_entities(self, query: str) -> list[str]:
        """Extract entity mentions from query (simplified)."""
        entities = []

        # Common words that appear capitalized at sentence start but aren't entities
        _skip_words = {
            "what", "when", "where", "who", "why", "how", "which",
            "did", "does", "do", "has", "have", "had", "is", "are",
            "was", "were", "will", "would", "could", "should", "can",
            "the", "this", "that", "these", "those", "there",
            "tell", "describe", "explain", "not", "but", "and", "for",
        }

        # Look for capitalized words (potential names)
        name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        for match in re.finditer(name_pattern, query):
            candidate = match.group(1)
            # Skip common non-entity words; for multi-word matches,
            # check each word individually
            words = candidate.split()
            filtered_words = [w for w in words if w.lower() not in _skip_words]
            if filtered_words:
                entities.append(" ".join(filtered_words))

        # Look for quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        quoted = re.findall(r"'([^']+)'", query)
        entities.extend(quoted)

        return list(set(entities))

    @staticmethod
    def _identify_target_entity(query: str, entities: list[str]) -> str | None:
        """Identify the target entity — the entity the question is about.

        In multi-person conversations, questions like "What did Melanie read?"
        are about Melanie specifically. This method uses grammatical subject
        patterns to distinguish the target entity from other mentioned entities.

        Returns the target entity name (matching case from entities list), or
        None if no target can be identified.
        """
        if not entities:
            return None
        if len(entities) == 1:
            return entities[0]

        query_lower = query.lower()

        # Subject extraction patterns — the entity right after the auxiliary verb
        # is typically the subject (the person the question is about).
        subject_patterns = [
            r"(?:did|does|do|has|have|had|was|were|is|will|would|could|should)\s+(\w+)\s+\w+",
            r"what\s+(?:is\s+)?(\w+)['\u2019]s\s+",  # "What is Melanie's..."
            r"how\s+(?:many|much|often)\s+.*?(?:did|does|has|have)\s+(\w+)\s+",
            r"(?:tell|describe)\s+(?:me\s+)?(?:about\s+)?(\w+)",
        ]

        entities_lower = {e.lower(): e for e in entities}

        for pattern in subject_patterns:
            match = re.search(pattern, query_lower)
            if match:
                subject = match.group(1).lower()
                if subject in entities_lower:
                    return entities_lower[subject]

        # Fallback: the first mentioned entity in the query is typically the target
        first_pos = {}
        for entity in entities:
            pos = query_lower.find(entity.lower())
            if pos >= 0:
                first_pos[entity] = pos
        if first_pos:
            return min(first_pos, key=first_pos.get)

        return entities[0]

    def _extract_keywords(self, query_lower: str) -> list[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "about",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "it",
            "its",
        }

        # Tokenize and filter
        words = re.findall(r"\b\w+\b", query_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _determine_temporal_scope(
        self, query_lower: str, temporal_expressions: list[TemporalExpression]
    ) -> TemporalScope:
        """Determine the temporal scope of the query."""
        if not temporal_expressions:
            # Check for temporal keywords without specific expressions
            if re.search(r"\bwhen\b|\btime\b|\bdate\b", query_lower):
                return TemporalScope.POINT
            return TemporalScope.NONE

        # Analyze the expressions
        has_point = False
        has_range = False
        has_relative = False

        for expr in temporal_expressions:
            if expr.normalized_start and expr.normalized_end:
                has_range = True
            elif expr.normalized_start:
                has_point = True
            if expr.relation in ["before", "after", "during"]:
                has_relative = True

        if has_relative:
            return TemporalScope.RELATIVE
        if has_range:
            return TemporalScope.RANGE
        if has_point:
            return TemporalScope.POINT

        return TemporalScope.NONE

    def _is_negation_check(self, query_lower: str) -> bool:
        """Check if query is verifying a negation."""
        for pattern in self.NEGATION_CHECK_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _determine_answer_type(self, query_lower: str, intent: QueryIntent) -> str:
        """Determine expected answer type."""
        # Yes/No questions
        if re.match(
            r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b", query_lower
        ):
            return "yes_no"

        # Date/time questions
        if re.search(r"\bwhen\b|\bwhat time\b|\bwhat date\b", query_lower):
            return "date"

        # Count questions
        if re.search(r"\bhow many\b|\bhow much\b", query_lower):
            return "number"

        # List questions
        if intent == QueryIntent.LIST:
            return "list"

        return "text"

    def get_retrieval_strategy(self, analysis: QueryAnalysis) -> dict[str, Any]:
        """
        Determine the retrieval strategy based on query analysis.

        Returns configuration for the retriever.
        """
        strategy = {
            "use_semantic_search": True,
            "use_entity_search": len(analysis.entities) > 0,
            "use_temporal_search": analysis.temporal_scope != TemporalScope.NONE,
            "use_graph_traversal": analysis.reasoning_type == ReasoningType.MULTI_HOP,
            "check_negations": analysis.is_negation_check,
            "max_hops": 1,
            "top_k": 10,
        }

        # Adjust based on reasoning type
        if analysis.reasoning_type == ReasoningType.MULTI_HOP:
            strategy["max_hops"] = 3
            strategy["top_k"] = 20  # Increased for better coverage
            strategy["use_graph_traversal"] = True

        elif analysis.reasoning_type == ReasoningType.TEMPORAL:
            strategy["use_temporal_search"] = True
            strategy["temporal_priority"] = True

        elif analysis.reasoning_type == ReasoningType.CAUSAL:
            strategy["use_causal_graph"] = True
            strategy["max_hops"] = 2

        elif analysis.reasoning_type == ReasoningType.COMMONSENSE:
            strategy["top_k"] = 20  # Increased for diverse context
            strategy["use_graph_traversal"] = True
            strategy["max_hops"] = 2

        elif analysis.reasoning_type == ReasoningType.ADVERSARIAL:
            strategy["check_negations"] = True
            strategy["verify_facts"] = True

        return strategy
