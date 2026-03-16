"""
Prompt Templates: Question-type-aware prompt construction for QA.

Classifies questions by type and builds specialized prompts with:
- Format instructions per type (date, count, choice, yes/no, multi-hop)
- Entity verification guidance
- Temporal guidance for relative date conversion
- Anti-evasion rules to prevent "not mentioned" responses
"""

from __future__ import annotations

import re
from enum import Enum


class QuestionType(Enum):
    """Types of questions for prompt specialization."""

    YES_NO = "yes_no"
    FACTUAL = "factual"
    TEMPORAL_DATE = "temporal_date"
    TEMPORAL_DURATION = "temporal_duration"
    COUNTING = "counting"
    CHOICE = "choice"
    MULTI_HOP = "multi_hop"
    ADVERSARIAL = "adversarial"
    DEFAULT = "default"


class QuestionTypeClassifier:
    """Classify a question into a QuestionType for prompt specialization."""

    # Yes/No question starters
    YES_NO_STARTERS = (
        "do ",
        "does ",
        "did ",
        "is ",
        "are ",
        "was ",
        "were ",
        "can ",
        "could ",
        "will ",
        "would ",
        "has ",
        "have ",
        "had ",
        "should ",
    )

    # Multi-hop indicators
    MULTI_HOP_PHRASES = ["based on", "according to", "experience", "recommend"]

    def classify(
        self,
        query: str,
        *,
        category: str | None = None,
        expected_answer_type: str | None = None,
    ) -> QuestionType:
        """
        Classify a question into a QuestionType.

        Args:
            query: The question text
            category: Optional category hint (e.g., from benchmark metadata)
            expected_answer_type: Optional hint from QueryAnalysis
        """
        q_lower = query.lower().strip()

        # Adversarial category overrides other classification
        if category == "adversarial":
            return QuestionType.ADVERSARIAL

        # Counting questions
        if "how many" in q_lower:
            return QuestionType.COUNTING

        # Choice questions: "A or B?"
        is_choice_question = (
            re.search(r"\b(or a |or the |or an |\w+ or \w+\?)", q_lower) is not None
        )
        is_preference = "more interested" in q_lower or "prefer" in q_lower or "rather" in q_lower
        if is_choice_question and (is_preference or " or " in q_lower):
            return QuestionType.CHOICE

        # Yes/No detection (excluding comparisons and choices)
        is_comparison = "before or after" in q_lower or "or after" in q_lower
        if q_lower.startswith(self.YES_NO_STARTERS) and not is_comparison:
            # Multi-hop yes/no is still multi-hop
            if category == "multi_hop" or any(
                p in q_lower for p in self.MULTI_HOP_PHRASES
            ):
                return QuestionType.MULTI_HOP
            return QuestionType.YES_NO

        # Temporal: date vs duration
        expects_date = q_lower.startswith(("when ", "what date ", "what time ")) or (
            q_lower.startswith("how long") and "how long did" not in q_lower
        )
        expects_non_date = q_lower.startswith(
            ("who ", "what ", "where ", "why ", "which ")
        ) and not q_lower.startswith(("what date", "what time"))

        if expects_date and not expects_non_date:
            is_duration = any(
                phrase in q_lower
                for phrase in [
                    "how long",
                    "how many years",
                    "years ago",
                    "since when",
                ]
            )
            return (
                QuestionType.TEMPORAL_DURATION
                if is_duration
                else QuestionType.TEMPORAL_DATE
            )

        # Multi-hop (from category or phrase detection)
        if category == "multi_hop" or any(
            p in q_lower for p in self.MULTI_HOP_PHRASES
        ):
            return QuestionType.MULTI_HOP

        # Factual questions
        if q_lower.startswith(("what ", "where ", "when ", "who ", "which ", "how ")) or is_comparison:
            return QuestionType.FACTUAL

        return QuestionType.DEFAULT

    def is_temporal(self, question_type: QuestionType) -> bool:
        """Check if a question type is temporal."""
        return question_type in (QuestionType.TEMPORAL_DATE, QuestionType.TEMPORAL_DURATION)


class PromptTemplates:
    """Build question-type-aware prompts for answer generation."""

    # Vague temporal modifiers that cause LLMs to interpret dates relative
    # to their current date rather than the conversation timeframe.  By the
    # time we reach answer generation the retriever has already used these
    # modifiers to select relevant context, so they only add confusion.
    _TEMPORAL_MODIFIERS = re.compile(
        r"\b(recently|lately)\b\s*", re.IGNORECASE
    )

    def __init__(self) -> None:
        self.classifier = QuestionTypeClassifier()

    @classmethod
    def _strip_temporal_modifiers(cls, query: str) -> str:
        """Strip vague temporal modifiers that confuse answer-generation LLMs."""
        cleaned = cls._TEMPORAL_MODIFIERS.sub("", query)
        # Clean up double/trailing spaces and trailing commas before '?'
        cleaned = re.sub(r"  +", " ", cleaned).strip()
        cleaned = re.sub(r"\s+\?", "?", cleaned)
        cleaned = re.sub(r",\s*\?", "?", cleaned)
        return cleaned

    def build_prompt(
        self,
        query: str,
        context: str,
        *,
        question_type: QuestionType | None = None,
        category: str | None = None,
        target_entity: str | None = None,
        supplementary_instructions: str = "",
    ) -> str:
        """
        Build a complete prompt for answer generation.

        Args:
            query: The question to answer
            context: Retrieved context to use
            question_type: Pre-classified type (auto-detected if None)
            category: Optional category hint for classification
            target_entity: Entity the question is about
            supplementary_instructions: Extra instructions to append
                (e.g., domain-specific inference patterns)
        """
        if question_type is None:
            question_type = self.classifier.classify(query, category=category)

        answer_format = self._format_instructions(question_type)

        # Adversarial gets its own specialized prompt
        if question_type == QuestionType.ADVERSARIAL:
            return self._adversarial_prompt(query, context, target_entity)

        # Strip vague temporal modifiers for non-temporal question types.
        # The retriever already used them; they only confuse the answer LLM.
        display_query = query
        if not self.classifier.is_temporal(question_type):
            display_query = self._strip_temporal_modifiers(query)

        supplementary = ""
        if supplementary_instructions:
            supplementary = f"\n{supplementary_instructions}\n"

        return f"""Answer the question based on the conversation context below.
If the answer can be reasonably inferred from the context, give it directly.
If the context truly contains no relevant information, say "None".
Be concise but complete — include all relevant details from the context.

Context:
{context}

Question: {display_query}

{answer_format}
{supplementary}
Answer:"""

    def _format_instructions(self, question_type: QuestionType) -> str:
        """Get format-specific instructions for a question type."""
        if question_type == QuestionType.TEMPORAL_DURATION:
            return """ANSWER FORMAT:
- This is a DURATION question asking about time periods
- Look for phrases like "since 2016", "for 4 years", "been doing X for Y years"
- Answer formats:
  * "Since 2016" (if they say "since 2016" or "started in 2016")
  * "4 years" (if they say "for 4 years" or "been 4 years")
  * "10 years ago" (if they say "10 years ago")
- For "how long has X been doing Y": look for "since [year]" or "[N] years"
- DO NOT convert to dates - keep the duration/period format"""

        if question_type == QuestionType.TEMPORAL_DATE:
            return """ANSWER FORMAT:
- ONLY return an ABSOLUTE DATE — nothing else!
- Find the message that mentions or implies the SPECIFIC EVENT asked about
- Use the [Speaker, timestamp] metadata to identify the session date

EVENT IDENTIFICATION (CRITICAL):
- Events may be referred to with pronouns: "we just did it", "that was fun", "we went there"
- Resolve "it"/"that"/"this" using surrounding context (nature, camping, hiking = outdoor activity)
- Match events by MEANING, not exact keywords — if the question asks about "a hike" and context shows an outdoor/nature activity "after the road trip", these refer to the same event
- When someone says "we just did it yesterday" in a conversation about nature/outdoors, and the question asks about a hike, this IS the hike
- DO NOT require the exact word from the question to appear — use inference

OUTPUT EXAMPLES (follow these exactly):
- "7 May 2023"
- "The week before 27 June 2023"
- "The Friday before 14 August 2023"
- "June 2023"
- "2022"

CRITICAL — ALWAYS CONVERT TO ABSOLUTE DATES:
You MUST convert ALL relative expressions to absolute dates using the session date.
NEVER return relative words like "next month", "last Friday", "last week", "tomorrow", "yesterday", "soon", "recently", "next year".

CONVERSION RULES:
1. "next month" + session date "15 May 2023" -> "June 2023"
2. "last Friday" + session date "14 August 2023" -> "The Friday before 14 August 2023"
3. "yesterday" + session date "[date]" -> the day before that date
4. "last week" + session date "[date]" -> "The week before [date]"
5. "the week before [date]" in context -> "The week before [date]"
6. Specific date in context -> That exact date
7. Year only -> Just the year (e.g., "2022")
8. "last year" in 2023 context -> "2022"
9. "next year" in 2023 context -> "2024"
10. "this summer" in May 2023 context -> "Summer 2023"
11. If question asks "together" -> ONLY count events where BOTH people participated
12. "last year at [event]" in 2023 -> The event happened in 2022

CRITICAL FOR "TOGETHER" QUESTIONS:
- "When did X and Y do Z together?" -> Find events where BOTH participated
- "we had a blast last year at the Pride fest" (Aug 2023) -> Pride fest was in 2022

NEVER return full sentences — ONLY the absolute date/time expression!"""

        if question_type == QuestionType.COUNTING:
            return """ANSWER FORMAT - COUNTING QUESTION:
- Return ONLY a NUMBER (digit or word): "1", "2", "3", "once", "twice", "three"

COUNTING METHOD:
1. First look for EXPLICIT counts: "three turtles", "twice", "my two dogs"

2. If no explicit count, INFER from distinct individuals or events:
   - "my son had an accident... they explained their brother would be OK"
     -> "they" (≥2 kids) + "their brother" (1 more) = at least 3 children
   - "my daughter's birthday" + "my son" + other siblings = count them
   - Names mentioned as children/pets/items = count distinct ones

3. For "how many times" questions:
   - Count UNIQUE events, not mentions of the same event

4. Do NOT count word occurrences — count stated or inferable quantities

RETURN JUST THE NUMBER - no explanation!"""

        if question_type == QuestionType.CHOICE:
            return """ANSWER FORMAT:
- This is a CHOICE question asking you to pick between options (A or B)
- DO NOT answer "Yes" or "No" - CHOOSE one of the options
- Give JUST the chosen option, with brief reason
- Example: "Would X prefer A or B?" -> "A; she loves outdoor activities"
- Example: "Is X close to beach or mountains?" -> "Beach"
- The answer should be ONE of the options mentioned in the question"""

        if question_type == QuestionType.YES_NO:
            return """ANSWER FORMAT:
- This is a YES/NO question. Start with "Yes" or "No"
- If the context shows negative statements like "could never", "don't like", "hate" -> answer "No"
- Add brief explanation if needed"""

        if question_type == QuestionType.MULTI_HOP:
            return """MULTI-HOP REASONING - Give a CONCISE answer.

IMPORTANT: DO NOT explain your reasoning steps. ONLY give the final answer.

For YES/NO inference questions:
- Start with "Yes" or "No" or "Likely yes" or "Likely no"
- Add ONE brief reason (max 10 words)
- Example: "Yes, since she collects classic children's books"
- Example: "Likely no; since this one went badly"

For TRAIT/ATTRIBUTE questions:
- List 2-4 key traits/attributes as comma-separated words
- Example: "Thoughtful, authentic, driven"

INFERENCE CHAIN METHOD - Follow this reasoning:
1. FIND: What facts are stated about the person?
2. CONNECT: What category/pattern do these facts fit?
3. INFER: What can we conclude from that pattern?

CHOICE QUESTIONS ("Would X prefer A or B?"):
- MUST pick ONE option from the question, not say "No" or "Yes"
- Format: "[Option]; [brief reason]"

CRITICAL RULES:
1. YES/NO answers: Start with "Yes" or "No" followed by brief reason
2. Never say "Unknown" if you can reasonably infer from context
3. KEEP ANSWERS BRIEF - just the conclusion with 1 sentence of reason"""

        if question_type == QuestionType.FACTUAL:
            return """ANSWER FORMAT:
- Give JUST the fact asked about — keep it brief
- For "How did/does X..." questions: describe the PROCESS or REACTION, not just the outcome
  * Focus on emotions, actions, and responses — not just the end result
  * Example: "He was scared at first but his family reassured him" (NOT just "He was okay")
- For "What events/activities/things..." list questions: list ALL distinct items from context
  * Prefer BREADTH over depth — mention every distinct item, keep each brief
  * Short comma-separated list: "mentoring program, school speech, talent show"
  * Do NOT elaborate on each item — just name it
- Comma-separated list if multiple items (e.g., "sunset, sunrise, horse")
- Use statements FROM or ABOUT the person asked about (look at [speaker] in brackets)
- Be SPECIFIC (item names, place names) not generic (categories, abstractions)
- SYNTHESIZE: combine details from multiple passages into one complete answer"""

        # DEFAULT
        return """ANSWER FORMAT:
- Give a direct, concise answer
- Only use "Yes/No" if the question explicitly asks for confirmation"""

    def _entity_guidance(self, target_entity: str | None, *, is_temporal: bool = False) -> str:
        """Build entity verification guidance."""
        if not target_entity:
            return ""

        if is_temporal:
            return f"""
RULES FOR TEMPORAL QUESTIONS:
- Find the date/time information asked about
- For secondary entities, look for what the speakers say ABOUT them
- "Since 2016" / "for 7 years" / specific dates are valid answers
- Only answer "None" if the date/time is truly not mentioned

CRITICAL - RELATIVE YEAR CONVERSION:
- "last year" spoken in 2023 -> answer "2022"
- "this year" spoken in 2023 -> answer "2023"
- Look for conversation DATE in context to determine the reference year"""

        return f"""
CRITICAL - ENTITY VERIFICATION:
1. Check WHO the question asks about
2. Find statements FROM or ABOUT that specific person (look at [Speaker] in brackets)
3. If the info only exists for a DIFFERENT person, answer "None"

WHEN TO ANSWER vs WHEN TO SAY "None":
- Look for statements where [{target_entity}]: says something relevant
- If the question asks about person X but only person Y has the answer -> None (misattribution)

SECONDARY ENTITIES (friends/family mentioned by main speakers):
- Questions about secondary entities may be answered by what main speakers say ABOUT them
- Look for possessive forms: "X's store", "X's studio"
- These facts are valid even if the secondary entity isn't a speaker

IMPORTANT - ANSWER FORMAT:
- NEVER respond with "The messages do not provide..." or "not mentioned" or "not provided"
- NEVER give explanatory sentences about what's not in the context
- If information is missing: just say "None" - nothing else
- If information exists: give a DIRECT answer using their actual words
- For ALL questions: extract the answer or say "None" - no explanations"""

    def _adversarial_prompt(
        self, query: str, context: str, target_entity: str | None
    ) -> str:
        """Build a specialized prompt for adversarial questions."""
        entity_name = target_entity or "target entity"
        q_lower = query.lower().strip()
        is_yes_no = q_lower.startswith(self.classifier.YES_NO_STARTERS)

        if is_yes_no:
            answer_guidance = (
                f'3. If YES and {entity_name} confirms it -> answer "Yes"\n'
                f'4. If NO or {entity_name} never mentions it -> answer "No"'
            )
            answer_line = 'Answer ("Yes", "No", or "None"):'
        else:
            answer_guidance = (
                f"3. If YES -> give the answer\n"
                f'4. If NO or UNCERTAIN -> answer "None"'
            )
            answer_line = 'Answer (just the fact or "None"):'

        return f"""Based on the following context, answer the question about {entity_name}.

WARNING: This may be a TRICK QUESTION that swaps names. The context may contain
relevant information spoken by a DIFFERENT person — you must IGNORE it.

STRICT RULES:
1. ONLY use first-person statements FROM {entity_name} (look for [{entity_name}]: prefix)
2. {entity_name} must say "I", "my", "mine" about the topic — not someone else saying it about them
3. If a DIFFERENT speaker (not {entity_name}) mentions the item/topic -> that does NOT count
4. If {entity_name} never personally claims or mentions the topic -> answer "None"

Context:
{context}

Question: {query}

VERIFICATION:
1. What specific item/attribute/action is the question asking about?
2. Scan ONLY [{entity_name}]: lines — does {entity_name} PERSONALLY mention this?
{answer_guidance}

{answer_line}"""
