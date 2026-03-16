"""
Answer Generator: LLM-based answer generation with quality improvements.

Features:
- Question-type-aware prompt construction (via PromptTemplates)
- Evasive answer detection and fallback
- Answer normalization (counting, yes/no, choice, multi-hop)
- Optional self-consistency voting
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from zerogmem.defaults import DEFAULT_LLM_MODEL, WORD_TO_NUM, llm_chat_kwargs
from zerogmem.reasoning.prompt_templates import PromptTemplates, QuestionType


@dataclass
class AnswerConfig:
    """Configuration for answer generation."""

    use_self_consistency: bool = False  # Expensive, off by default
    consistency_temperatures: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.5]
    )
    use_evasive_detection: bool = True
    use_normalization: bool = True
    max_answer_tokens: int = 200
    llm_model: str = DEFAULT_LLM_MODEL


# Patterns that indicate the LLM is evading rather than answering
EVASIVE_PATTERNS = [
    "the specific",
    "not provided",
    "not mentioned",
    "does not",
    "is not",
    "no information",
    "cannot find",
    "not found",
    "not available",
    "doesn't mention",
    "don't have",
    "no explicit",
    "cannot determine",
    "unclear",
    "not explicitly stated",
    "no direct",
    "based on the context",
    "context does not",
    "messages do not",
]


class EvasiveAnswerDetector:
    """Detect when an LLM gives an evasive non-answer."""

    @staticmethod
    def is_evasive(answer: str) -> bool:
        """Check if answer is evasive (avoids giving a direct answer)."""
        answer_lower = answer.lower().strip()
        return any(phrase in answer_lower for phrase in EVASIVE_PATTERNS)

    @staticmethod
    def is_none_like(answer: str) -> bool:
        """Check if answer is equivalent to 'no answer found'."""
        return answer.lower().strip() in ["none", "unknown", ""]


class AnswerNormalizer:
    """Normalize LLM answers based on question type."""

    # Shared number-word mapping from defaults (ordered long-first for matching)
    NUMBER_WORDS = sorted(WORD_TO_NUM.items(), key=lambda x: len(x[0]), reverse=True)

    # Yes/No indicators
    POSITIVE_INDICATORS = [
        "yes",
        "likely",
        "probably",
        "would",
        "she is",
        "he is",
        "they are",
    ]
    NEGATIVE_INDICATORS = [
        "no",
        "unlikely",
        "probably not",
        "would not",
        "wouldn't",
        "she is not",
        "he is not",
    ]

    def normalize(self, answer: str, query: str, question_type: QuestionType) -> str:
        """Normalize answer based on question type."""
        if not answer or answer.lower() in ["none", "unknown", ""]:
            return answer

        if question_type == QuestionType.COUNTING:
            return self._normalize_counting(answer)
        elif question_type == QuestionType.CHOICE:
            return self._normalize_choice(answer, query)
        elif question_type == QuestionType.YES_NO:
            return self._normalize_yes_no(answer)
        elif question_type == QuestionType.MULTI_HOP:
            return self._normalize_multi_hop(answer, query)
        return answer

    def _normalize_counting(self, answer: str) -> str:
        """Extract just the number from a counting answer."""
        # Try digits first
        digit_match = re.search(r"\b(\d+)\b", answer)
        if digit_match:
            return digit_match.group(1)

        # Try number words
        answer_lower = answer.lower()
        for word, num in self.NUMBER_WORDS:
            if re.search(rf"\b{word}\b", answer_lower):
                return num

        return answer

    def _normalize_choice(self, answer: str, query: str) -> str:
        """Extract the chosen option from a choice answer."""
        if answer.lower().startswith(("yes", "no")):
            return answer

        q_lower = query.lower()
        or_match = re.search(
            r"(?:going to|interested in|prefer)\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+or\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
            q_lower,
        )
        if or_match:
            option_a, option_b = or_match.groups()
            answer_lower = answer.lower()
            if option_a in answer_lower and option_b not in answer_lower:
                return option_a.title()
            elif option_b in answer_lower and option_a not in answer_lower:
                return option_b.title()
            elif option_a in answer_lower:
                return option_a.title()

        return answer

    def _normalize_yes_no(self, answer: str) -> str:
        """Ensure yes/no answers start with Yes or No."""
        answer_lower = answer.lower().strip()

        if answer_lower.startswith("yes") or answer_lower.startswith("no"):
            return answer

        # Try to infer from content
        if any(ind in answer_lower for ind in self.NEGATIVE_INDICATORS):
            for ind in self.NEGATIVE_INDICATORS:
                if ind in answer_lower:
                    idx = answer_lower.find(ind)
                    reason = answer[idx:].strip()
                    if reason.startswith(("no", "unlikely", "probably not")):
                        rest = answer[idx + len(ind) :].strip()
                        return f"No; {rest}" if rest else "No"
                    else:
                        return f"Likely no; {reason}"

        return answer

    def _normalize_multi_hop(self, answer: str, query: str) -> str:
        """Trim verbose multi-hop answers to concise conclusions."""
        if len(answer) <= 100:
            return answer

        q_lower = query.lower().strip()
        answer_lower = answer.lower().strip()

        # Yes/No with long explanation — trim to first sentence
        is_yes_no = q_lower.startswith((
            "would ", "does ", "did ", "is ", "are ", "was ", "were ",
            "can ", "could ", "will ", "has ", "have ", "had ", "do ", "should ",
        ))

        if is_yes_no:
            if answer_lower.startswith("yes"):
                first_period = self._find_sentence_end(answer, 0, 150)
                if first_period > 10:
                    return answer[: first_period + 1].strip()
                match = re.match(
                    r"^(Yes[,;]?\s*[^.]{10,100}\.)", answer, re.IGNORECASE
                )
                return match.group(1).strip() if match else "Yes"
            elif answer_lower.startswith("no"):
                first_period = self._find_sentence_end(answer, 0, 150)
                if first_period > 10:
                    return answer[: first_period + 1].strip()
                match = re.match(
                    r"^(No[,;]?\s*[^.]{10,100}\.)", answer, re.IGNORECASE
                )
                return match.group(1).strip() if match else "No"
            elif "likely" in answer_lower[:50]:
                match = re.search(
                    r"(likely\s+(?:yes|no)[^.]{0,80})", answer, re.IGNORECASE
                )
                if match:
                    return match.group(1).strip()

        # Trait/attribute questions — extract comma list
        if any(w in q_lower for w in ["traits", "attributes", "describe", "personality"]):
            for pattern in [
                r"(?:traits?|attributes?)[:\s]+([A-Za-z]+(?:,\s*[A-Za-z]+)+)",
                r"(?:is|are|being)[:\s]*([A-Za-z]+(?:,\s*[A-Za-z]+)+)",
            ]:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        # Conclusion patterns
        if len(answer) > 100:
            for pattern in [
                r"(?:therefore|thus|so|hence|in conclusion|overall)[,:]?\s*(.+?)(?:\.|$)",
                r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
                r"^(?:yes|no)[,.]?\s*(.+?)(?:\.|$)",
            ]:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    conclusion = match.group(1).strip()
                    if 5 < len(conclusion) < 150:
                        if answer.lower().startswith("yes"):
                            return f"Yes; {conclusion}"
                        elif answer.lower().startswith("no"):
                            return f"No; {conclusion}"
                        return conclusion

        return answer

    @staticmethod
    def _find_sentence_end(text: str, start: int = 0, max_pos: int = 200) -> int:
        """Find end of first sentence, skipping abbreviations."""
        abbrevs = [
            "dr.", "mr.", "ms.", "mrs.", "prof.", "st.", "vs.", "e.g.", "i.e.", "etc."
        ]
        pos = start
        while pos < len(text) and pos < max_pos:
            period_pos = text.find(".", pos)
            if period_pos == -1 or period_pos >= max_pos:
                return -1
            before = text[max(0, period_pos - 4) : period_pos + 1].lower()
            if not any(before.endswith(a) for a in abbrevs):
                return period_pos
            pos = period_pos + 1
        return -1


class SelfConsistencyVoter:
    """Generate multiple answers at different temperatures and vote."""

    def __init__(self, llm_client: Any, config: AnswerConfig) -> None:
        self._client = llm_client
        self._config = config

    def vote(self, prompt: str) -> str:
        """Generate answers at multiple temperatures and return majority."""
        answers: list[str] = []

        for temp in self._config.consistency_temperatures:
            try:
                response = self._client.chat.completions.create(
                    **llm_chat_kwargs(self._config.llm_model, max_tokens=self._config.max_answer_tokens, temperature=temp),
                    messages=[{"role": "user", "content": prompt}],
                )
                ans = response.choices[0].message.content.strip()
                answers.append(ans)
            except Exception:
                continue

        return self._pick_best(answers)

    @staticmethod
    def _pick_best(answers: list[str]) -> str:
        """Pick the best answer from candidates using normalized voting."""
        if not answers:
            return "None"
        if len(answers) == 1:
            return answers[0]

        def _normalize(ans: str) -> str:
            ans = ans.lower().strip()
            for prefix in [
                "yes, ", "no, ", "yes ", "no ", "based on ", "according to ",
            ]:
                if ans.startswith(prefix):
                    ans = ans[len(prefix):]
            return ans.strip()

        counts: Counter[str] = Counter()
        originals: dict[str, str] = {}
        for ans in answers:
            norm = _normalize(ans)
            counts[norm] += 1
            if norm not in originals:
                originals[norm] = ans

        # Prefer non-None answers
        non_none = {
            k: v for k, v in counts.items()
            if k not in ("none", "unknown", "")
            and "not provided" not in k
            and "not mentioned" not in k
        }

        if non_none:
            best = max(non_none.items(), key=lambda x: x[1])
            return originals[best[0]]
        return answers[0]


class AnswerGenerator:
    """
    Generate answers using LLM with question-type-aware prompts.

    Orchestrates:
    1. Prompt construction via PromptTemplates
    2. LLM call (with optional self-consistency)
    3. Evasive answer detection
    4. Answer normalization
    """

    def __init__(
        self,
        llm_client: Any = None,
        config: AnswerConfig | None = None,
    ) -> None:
        self.config = config or AnswerConfig()
        self._client = llm_client
        self._templates = PromptTemplates()
        self._normalizer = AnswerNormalizer()
        self._evasive_detector = EvasiveAnswerDetector()
        self._voter = (
            SelfConsistencyVoter(llm_client, self.config)
            if llm_client and self.config.use_self_consistency
            else None
        )

    def generate_answer(
        self,
        query: str,
        context: str,
        *,
        category: str | None = None,
        target_entity: str | None = None,
        question_type: QuestionType | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate an answer for a question given context.

        Args:
            query: The question to answer
            context: Retrieved context
            category: Optional category hint for prompt selection
            target_entity: Entity the question is about
            question_type: Pre-classified type (auto-detected if None)

        Returns:
            (answer, metadata) where metadata includes question_type, evasive, etc.
        """
        if question_type is None:
            question_type = self._templates.classifier.classify(
                query, category=category
            )

        prompt = self._templates.build_prompt(
            query,
            context,
            question_type=question_type,
            category=category,
            target_entity=target_entity,
        )

        metadata: dict[str, Any] = {
            "question_type": question_type.value,
            "used_self_consistency": False,
            "was_evasive": False,
        }

        # Generate answer
        if self._voter and self.config.use_self_consistency:
            answer = self._voter.vote(prompt)
            metadata["used_self_consistency"] = True
        elif self._client:
            answer = self._call_llm(prompt)
        else:
            return "None", metadata

        # Evasive detection
        if self.config.use_evasive_detection:
            if self._evasive_detector.is_evasive(answer) or self._evasive_detector.is_none_like(answer):
                metadata["was_evasive"] = True

        # Normalize
        if self.config.use_normalization:
            answer = self._normalizer.normalize(answer, query, question_type)

        return answer, metadata

    def _call_llm(self, prompt: str) -> str:
        """Make a single LLM call."""
        try:
            response = self._client.chat.completions.create(
                **llm_chat_kwargs(self.config.llm_model, max_tokens=self.config.max_answer_tokens, temperature=0.0),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "None"
