"""
Counting Pipeline: Specialized answer generation for "how many" questions.

Three-tier approach:
1. Regex extraction: Find explicit count statements ("I have three X", "went twice")
2. LLM counting fallback: Ask LLM to reason over deduplicated evidence
3. Date-based instance counting: Count unique date-stamped mentions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from zerogmem.defaults import (
    DEFAULT_LLM_MODEL,
    ORDINAL_TO_NUM,
    WORD_TO_NUM,
    llm_chat_kwargs,
)


@dataclass
class CountingConfig:
    """Configuration for counting pipeline."""

    jaccard_dedup_threshold: float = 0.70
    max_evidence_items: int = 30
    use_llm_fallback: bool = True
    llm_model: str = DEFAULT_LLM_MODEL

# Words indicating a planned (not completed) event
FUTURE_INDICATORS = [
    "going to",
    "will go",
    "plan to",
    "planning to",
    "want to",
    "hoping to",
]


@dataclass
class CountingEvidence:
    """Evidence item for counting."""

    content: str
    date_key: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class CountingPipeline:
    """
    Answer "how many" questions by aggregating evidence.

    Pipeline:
    1. Extract what to count from the question
    2. Gather evidence from provided results
    3. Try regex extraction for explicit counts
    4. Fall back to LLM counting
    5. Fall back to date-based instance counting
    """

    def __init__(
        self,
        llm_client: Any = None,
        config: CountingConfig | None = None,
    ) -> None:
        self.config = config or CountingConfig()
        self._client = llm_client

    def answer_counting_question(
        self,
        query: str,
        evidence_items: list[CountingEvidence],
        *,
        target_entity: str | None = None,
    ) -> tuple[str | None, str]:
        """
        Answer a counting question using collected evidence.

        Args:
            query: The counting question
            evidence_items: Pre-gathered evidence with content and date_key
            target_entity: Entity being asked about

        Returns:
            (answer, context) or (None, "") if can't answer
        """
        q_lower = query.lower()
        target = target_entity or "the person"

        # Step 1: Extract what we're counting
        count_target, count_keywords, action_verb = self._extract_count_target(q_lower)
        if not count_keywords:
            return None, ""

        # Step 2: Filter and collect evidence
        filtered_evidence, instances_by_date = self._filter_evidence(
            evidence_items, count_keywords, target_entity, q_lower
        )

        if not filtered_evidence and not instances_by_date:
            return None, ""

        # Deduplicate
        evidence_texts = [e.content for e in filtered_evidence]
        evidence_texts = self.deduplicate_evidence(
            evidence_texts, self.config.jaccard_dedup_threshold
        )

        # Step 3: Try regex extraction for explicit counts
        combined = " ".join(evidence_texts).lower()
        kw_variants = self._build_keyword_variants(count_keywords)
        explicit_count, snippet = self._extract_explicit_count(combined, kw_variants)

        if explicit_count:
            context_parts = [f"## Counting: {count_target} for {target}"]
            context_parts.append(f"Found explicit count statement: {explicit_count}")
            if snippet:
                context_parts.append(f"Context: ...{snippet}...")
            for text in evidence_texts[:5]:
                context_parts.append(f"- {text}")
            return explicit_count, "\n".join(context_parts)

        # Step 4: LLM counting fallback
        if self.config.use_llm_fallback and evidence_texts and count_target:
            llm_count = self._llm_count(
                query, target, count_target, evidence_texts
            )
            if llm_count:
                context_parts = [f"## Counting: {count_target} for {target}"]
                context_parts.append(f"LLM-determined count: {llm_count}")
                for text in evidence_texts[:10]:
                    context_parts.append(f"- {text}")
                return llm_count, "\n".join(context_parts)

        # Step 5: Date-based instance counting
        if not instances_by_date:
            return None, ""

        count = len(instances_by_date)
        context_parts = [f"## Counting: {count_target} for {target}"]
        context_parts.append(f"Found {count} instance(s) by date:\n")
        for date_key, content in list(instances_by_date.items())[:10]:
            context_parts.append(f"- [{date_key}] {content[:100]}")

        return str(count), "\n".join(context_parts)

    def _extract_count_target(
        self, q_lower: str
    ) -> tuple[str | None, list[str], str | None]:
        """
        Extract what to count from the question.

        Returns: (count_target, keywords, action_verb)
        """
        action_patterns = [
            # "how many times has X found new hiking trails"
            (
                r"how many times (?:did|has|have) \w+ (found|discovered|received|written|won|taken|adopted) (?:new |a |an |the )?(.+?)(?:\?|$)",
                True,
            ),
            # "how many X have made it to Y"
            (r"how many (?:of )?(?:\w+'?s? )?(.+?) (?:have|has) made it", False),
            # "how many times did X go to Y"
            (
                r"how many times (?:did|has|have) \w+ (?:go|gone|been) to (?:the |a )?(.+?)(?:\?|$)",
                False,
            ),
            (r"how many times (?:did|has|have) \w+ (.+?)(?:\?|$)", False),
            (r"how many (.+?) (?:does|did|has|have) \w+", False),
            (r"how many (.+?)(?:\?|$)", False),
        ]

        for pattern, has_action in action_patterns:
            match = re.search(pattern, q_lower)
            if match:
                if has_action and match.lastindex and match.lastindex >= 2:
                    action_verb = match.group(1).strip()
                    count_target = match.group(2).strip()
                else:
                    action_verb = None
                    count_target = match.group(1).strip()

                keywords = [w for w in count_target.split() if len(w) > 2]
                if action_verb:
                    keywords.insert(0, action_verb)
                return count_target, keywords, action_verb

        return None, [], None

    def _filter_evidence(
        self,
        evidence_items: list[CountingEvidence],
        count_keywords: list[str],
        target_entity: str | None,
        q_lower: str,
    ) -> tuple[list[CountingEvidence], dict[str, str]]:
        """Filter evidence to items matching count criteria."""
        # Year filter
        year_filter = None
        year_match = re.search(r"in\s+(\d{4})", q_lower)
        if year_match:
            year_filter = year_match.group(1)

        filtered: list[CountingEvidence] = []
        instances_by_date: dict[str, str] = {}

        for item in evidence_items:
            content_lower = item.content.lower()

            # Must match keywords
            if not any(kw in content_lower for kw in count_keywords):
                continue

            # Must mention target entity if specified
            if target_entity and target_entity.lower() not in content_lower:
                continue

            # Skip planned/future events
            if any(fi in content_lower for fi in FUTURE_INDICATORS):
                continue

            # Apply year filter
            if year_filter and item.date_key != "unknown":
                if year_filter not in item.date_key:
                    continue

            filtered.append(item)

            if item.date_key not in instances_by_date:
                instances_by_date[item.date_key] = item.content[:100]

        return filtered, instances_by_date

    @staticmethod
    def deduplicate_evidence(texts: list[str], threshold: float = 0.70) -> list[str]:
        """Deduplicate evidence texts using Jaccard word-overlap similarity."""
        if len(texts) <= 1:
            return texts

        def _jaccard(a: str, b: str) -> float:
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        # Sort by length descending to keep most informative version
        indexed = sorted(enumerate(texts), key=lambda x: len(x[1]), reverse=True)
        kept: list[tuple[int, str]] = []
        for idx, text in indexed:
            if any(_jaccard(text, kept_text) >= threshold for _, kept_text in kept):
                continue
            kept.append((idx, text))
        # Restore original order
        kept.sort(key=lambda x: x[0])
        return [text for _, text in kept]

    def _extract_explicit_count(
        self, combined_evidence: str, kw_variants: list[str]
    ) -> tuple[str | None, str | None]:
        """
        Extract explicit count from evidence using regex patterns.

        Returns: (count, context_snippet) or (None, None)
        """
        if not kw_variants:
            return None, None

        kw_pattern = "|".join(re.escape(kw) for kw in kw_variants)
        num_words_pattern = "|".join(WORD_TO_NUM.keys())
        ordinal_pattern = "|".join(ORDINAL_TO_NUM.keys())

        patterns = self._build_count_patterns(kw_pattern, num_words_pattern, ordinal_pattern)

        for pattern, group_idx in patterns:
            try:
                matches = list(re.finditer(pattern, combined_evidence))
            except re.error:
                continue

            if not matches:
                continue

            # Take the last match (most likely current state)
            for match in reversed(matches):
                count_str = match.group(group_idx).lower().strip()
                count = self._resolve_count_string(count_str)
                if count is not None:
                    start = max(0, match.start() - 50)
                    end = min(len(combined_evidence), match.end() + 50)
                    snippet = combined_evidence[start:end]
                    return count, snippet

        return None, None

    def _build_count_patterns(
        self, kw_pattern: str, num_words: str, ordinals: str
    ) -> list[tuple[str, int]]:
        """Build regex patterns for explicit count extraction."""
        return [
            # "I have three turtles", "got 2 dogs"
            (
                rf"(?:i\s+)?(?:have|got|has|own|adopted)\s+(\d+|{num_words})\s+(?:{kw_pattern})",
                1,
            ),
            # "twice", "three times", "2 times"
            (rf"\b(once|twice|thrice|(?:\d+|{num_words})\s+times?)\b", 1),
            # "found 2 new hiking trails"
            (
                rf"(?:found|discovered|got)\s+(\d+|{num_words})\s+(?:new\s+)?(?:{kw_pattern})",
                1,
            ),
            # "made it to the big screen two times"
            (rf"made\s+it\s+(?:to\s+)?.*?(\d+|{num_words})\s+times?", 1),
            # "been rejected twice"
            (
                rf"(?:been\s+)?(?:rejected|accepted)\s+(once|twice|thrice|(?:\d+|{num_words})\s+times?)",
                1,
            ),
            # "went to the beach 3 times"
            (
                rf"(?:went|visited|been|gone)\s+(?:to\s+)?.*?(\d+|{num_words})\s+times?",
                1,
            ),
            # "received two letters"
            (
                rf"(?:received|got|recieved)\s+(\d+|{num_words})\s+(?:{kw_pattern})",
                1,
            ),
            # "two letters" — direct count before noun
            (rf"\b(\d+|{num_words})\s+(?:{kw_pattern})\b", 1),
            # Ordinal: "getting a third X", "my third X"
            (
                rf"(?:getting\s+)?(?:a|my|the|his|her)\s+({ordinals})\s+(?:{kw_pattern})",
                1,
            ),
            # "big enough for three"
            (rf"(?:enough|room)\s+(?:for|now\s+for)\s+({num_words}|\d+)", 1),
            # "now have three"
            (rf"now\s+(?:have|has|got)\s+({num_words}|\d+)", 1),
            # "won my fourth tournament"
            (
                rf"(?:won\s+)?my\s+({ordinals}|\d+(?:st|nd|rd|th))\s+(?:video\s+game\s+)?(?:tournament|competition)",
                1,
            ),
            # "written three screenplays"
            (
                rf"(?:written|wrote)\s+(\d+|{num_words})\s+(?:{kw_pattern})",
                1,
            ),
            # "finished my first/second/third screenplay"
            (
                rf"finished\s+(?:my|the)\s+({ordinals})\s+(?:{kw_pattern})",
                1,
            ),
        ]

    @staticmethod
    def _resolve_count_string(count_str: str) -> str | None:
        """Convert a count string (word or digit) to a number string."""
        if count_str in WORD_TO_NUM:
            return WORD_TO_NUM[count_str]
        if count_str in ORDINAL_TO_NUM:
            return ORDINAL_TO_NUM[count_str]
        if count_str.endswith((" time", " times")):
            num_part = count_str.replace(" times", "").replace(" time", "").strip()
            if num_part in WORD_TO_NUM:
                return WORD_TO_NUM[num_part]
            if num_part.isdigit():
                return num_part
        if count_str == "once":
            return "1"
        if count_str == "twice":
            return "2"
        if count_str == "thrice":
            return "3"
        if count_str.isdigit():
            return count_str
        # Handle ordinal suffixes: "4th" -> "4"
        ordinal_match = re.match(r"(\d+)(?:st|nd|rd|th)", count_str)
        if ordinal_match:
            return ordinal_match.group(1)
        return None

    @staticmethod
    def _build_keyword_variants(keywords: list[str]) -> list[str]:
        """Build singular/plural variants for keywords."""
        variants = []
        for kw in keywords:
            variants.append(kw)
            if kw.endswith("s"):
                variants.append(kw[:-1])
            else:
                variants.append(kw + "s")
        return variants

    def _llm_count(
        self,
        question: str,
        target_entity: str,
        count_target: str,
        evidence: list[str],
    ) -> str | None:
        """Use LLM to count distinct instances when regex fails."""
        if not self._client or not evidence:
            return None

        evidence_text = "\n".join(f"- {e}" for e in evidence[:15])
        prompt = (
            f"Based ONLY on the evidence below, count the number of DISTINCT instances "
            f"of {count_target} related to {target_entity}.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Instructions:\n"
            f"1. List each DISTINCT instance (skip duplicates of the same event)\n"
            f"2. Only count events that ACTUALLY HAPPENED (not planned/future)\n"
            f"3. Give the total count as a single number\n\n"
            f"Respond with ONLY the number (e.g., '3'). If you cannot determine, respond with 'unknown'."
        )

        try:
            response = self._client.chat.completions.create(
                **llm_chat_kwargs(self.config.llm_model, max_tokens=20, temperature=0.0),
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content.strip()
            digit_match = re.search(r"\b(\d+)\b", result)
            if digit_match:
                return digit_match.group(1)
        except Exception:
            pass
        return None
