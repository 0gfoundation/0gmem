"""
LLM-based Fact Extractor: Uses LLM to extract structured facts from conversations.

More accurate than regex-based extraction for complex statements.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from zerogmem.defaults import DEFAULT_LLM_MODEL, llm_chat_kwargs


@dataclass
class PersonFact:
    """A fact about a person."""

    person: str
    fact_type: (
        str  # identity, relationship, activity, location, preference, event, family, pet, object
    )
    value: str
    source_text: str
    confidence: float = 0.9
    session_date: str = ""  # Date context for temporal facts
    event_date: str = ""  # Resolved date for events (for temporal questions)


class LLMFactExtractor:
    """
    Extracts structured facts using LLM.

    Builds a profile for each person mentioned in conversations.
    """

    # Future tense indicators - events mentioned as planned, not completed
    FUTURE_INDICATORS = [
        "going to",
        "will ",
        "gonna ",
        "plan to",
        "planning to",
        "want to",
        "hoping to",
        "looking forward to",
        "excited to",
        "next week",
        "next month",
        "tomorrow",
        "soon",
        "later",
        "thinking about",
        "considering",
        "might ",
        "maybe",
    ]

    # Past tense indicators - events that actually happened
    PAST_INDICATORS = [
        "went ",
        "visited ",
        "attended ",
        "had ",
        "made ",
        "did ",
        "saw ",
        "met ",
        "joined ",
        "signed up",
        "finished ",
        "yesterday",
        "last week",
        "last month",
        "last weekend",
        "the other day",
        "recently",
        "just ",
        "finally ",
        "was great",
        "was amazing",
        "was fun",
        "loved it",
        "it was",
        "we had",
        "i had",
        "got to",
    ]

    EXTRACTION_PROMPT = """\
Extract ALL factual information about people from this conversation message.

For each person mentioned (including the speaker referring to themselves), extract:
- identity: gender identity, profession, age, nationality
- relationship_status: single, married, dating, divorced, etc.
- location: where they live, where they're from, countries
- activities: hobbies, sports, classes they attend
- events: specific things they did, attended, or participated in
- preferences: things they like/love/enjoy or dislike/hate
- family: number of kids, spouse name, relatives, children's interests
- pets: pet names, types of pets
- objects: things they own, bought, received as gifts
- art: things they painted, drew, created
- books: books they read, favorite books, recommended books
- music: instruments played, favorite musicians/songs

Message from {speaker}:
"{text}"

IMPORTANT RULES:
1. If the speaker says "I" or "my", the person is "{speaker}"
2. Extract SPECIFIC values - names, numbers, specific items
3. Include ALL mentioned facts, even small details
4. For kids' interests, use "family_kids_like" type

Return JSON:
{{
  "facts": [
    {{"person": "Name", "type": "fact_type", "value": "specific value"}},
    ...
  ]
}}

If no facts, return: {{"facts": []}}
JSON:"""

    def __init__(self, llm_client: Any | None = None, model: str | None = None):
        self._client = llm_client
        self._model = (
            model or DEFAULT_LLM_MODEL
        )
        self._current_partner: str = ""  # Set per-call for book recommendation linking
        # NOTE: Profile state (person_profiles, cross_person_traits,
        # conversation_pairs) moved to evaluation.profile_answerer.ProfileAnswerer

    def set_client(self, client: Any) -> None:
        """Set the LLM client."""
        self._client = client

    def set_model(self, model: str) -> None:
        """Set the LLM model name."""
        self._model = model

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ) -> str | None:
        """Call the chat model with basic retries; return content or None."""
        if not self._client:
            return None

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    **llm_chat_kwargs(self._model, max_tokens=max_tokens, temperature=temperature),
                    messages=messages,
                )
                content: str = response.choices[0].message.content.strip()
                return content
            except Exception:
                sleep_s = min(30.0, backoff_base**attempt)
                time.sleep(sleep_s)

        return None

    def _is_completed_event(self, text: str, event_match_start: int = 0) -> bool:
        """
        Check if an event is mentioned as COMPLETED (past tense) vs PLANNED (future tense).

        This is crucial for temporal accuracy:
        - "I'm going to the conference next week" → PLANNED (don't use this session's date)
        - "I went to the conference yesterday" → COMPLETED (use this session's date)

        Args:
            text: The full message text
            event_match_start: Position where the event was matched

        Returns:
            True if event is completed, False if planned/future
        """
        text_lower = text.lower()

        # Check for future indicators - if present, event is NOT completed
        has_future = any(fi in text_lower for fi in self.FUTURE_INDICATORS)

        # Check for past indicators - if present, event IS completed
        has_past = any(pi in text_lower for pi in self.PAST_INDICATORS)

        # If both present, check which is closer to the event mention
        if has_future and has_past:
            # Find closest indicator to event
            future_dist = float("inf")
            past_dist = float("inf")

            for fi in self.FUTURE_INDICATORS:
                pos = text_lower.find(fi)
                if pos != -1:
                    future_dist = min(future_dist, abs(pos - event_match_start))

            for pi in self.PAST_INDICATORS:
                pos = text_lower.find(pi)
                if pos != -1:
                    past_dist = min(past_dist, abs(pos - event_match_start))

            return past_dist < future_dist

        # Only past indicators present → completed
        if has_past and not has_future:
            return True

        # Only future indicators present → not completed
        if has_future and not has_past:
            return False

        # No clear indicators - default to completed (conservative)
        # This handles simple past tense like "I went camping"
        return True

    def extract_facts(
        self,
        text: str,
        speaker: str,
        session_context: str | None = None,
    ) -> list[PersonFact]:
        """
        Extract facts from text using LLM.
        """
        if not self._client:
            return self._extract_facts_regex(text, speaker)

        prompt = self.EXTRACTION_PROMPT.format(text=text, speaker=speaker)

        try:
            result = self._chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                temperature=0,
            )
            if not result:
                return self._extract_facts_regex(text, speaker)

            # Parse JSON
            # Find JSON in response
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                facts = []
                for f in data.get("facts", []):
                    fact = PersonFact(
                        person=f.get("person", speaker),
                        fact_type=f.get("type", "unknown"),
                        value=f.get("value", ""),
                        source_text=text[:200],
                    )
                    facts.append(fact)
                return facts
        except Exception:
            pass

        return self._extract_facts_regex(text, speaker)

    def _extract_facts_regex(
        self, text: str, speaker: str, session_date: str = "", partner: str = ""
    ) -> list[PersonFact]:
        """Fallback regex-based extraction with comprehensive patterns.

        Args:
            text: The message text to extract facts from
            speaker: The speaker of this message
            session_date: The session date for temporal facts
            partner: The conversation partner (for "you" statements)
        """
        facts = []
        text_lower = text.lower()
        self._current_partner = partner  # Used for book recommendation linking

        # Track session date for temporal facts
        self._current_session_date = session_date

        # Identity patterns
        if "transgender" in text_lower or "trans woman" in text_lower or "trans man" in text_lower:
            facts.append(PersonFact(speaker, "identity", "transgender woman", text[:100]))
        if "coming out" in text_lower:
            facts.append(PersonFact(speaker, "identity", "LGBTQ", text[:100]))

        # LGBTQ community participation
        if "lgbtq" in text_lower or "pride" in text_lower:
            if "activist" in text_lower or "group" in text_lower:
                facts.append(
                    PersonFact(speaker, "lgbtq_participation", "activist group", text[:100])
                )
            if "parade" in text_lower:
                facts.append(
                    PersonFact(speaker, "lgbtq_participation", "pride parades", text[:100])
                )
            if "mentor" in text_lower:
                facts.append(PersonFact(speaker, "lgbtq_participation", "mentoring", text[:100]))
            if "center" in text_lower or "centre" in text_lower:
                facts.append(PersonFact(speaker, "lgbtq_participation", "LGBTQ center", text[:100]))

        # Relationship patterns
        if (
            "single parent" in text_lower
            or "single mom" in text_lower
            or "single dad" in text_lower
        ):
            facts.append(PersonFact(speaker, "relationship_status", "single", text[:100]))
        if "breakup" in text_lower or "broke up" in text_lower:
            facts.append(PersonFact(speaker, "relationship_status", "single", text[:100]))
        married_match = re.search(
            r"(?:married|husband|wife)\s+(?:for\s+)?(\d+)\s+years?", text_lower
        )
        if married_match:
            years = married_match.group(1)
            facts.append(
                PersonFact(speaker, "relationship_status", f"married {years} years", text[:100])
            )
            facts.append(PersonFact(speaker, "married_years", years, text[:100]))
        elif "married for" in text_lower or "my husband" in text_lower or "my wife" in text_lower:
            years_match = re.search(r"(\d+)\s+years", text_lower)
            if years_match:
                facts.append(PersonFact(speaker, "married_years", years_match.group(1), text[:100]))
        # Also capture "been married" patterns
        been_married = re.search(
            r"(?:been|we've been|we have been)\s+married\s+(?:for\s+)?(\d+)\s+years?", text_lower
        )
        if been_married:
            facts.append(PersonFact(speaker, "married_years", been_married.group(1), text[:100]))

        # Duration patterns - friends
        friends_years_match = re.search(
            r"(?:friends?|known)\s+(?:for\s+)?(\d+)\s+years", text_lower
        )
        if friends_years_match:
            facts.append(
                PersonFact(speaker, "friends_years", friends_years_match.group(1), text[:100])
            )

        # Age patterns
        age_match = re.search(
            r"(?:i am|i\'m|she is|he is|turned)\s+(\d+)\s+(?:years old|years|now)", text_lower
        )
        if age_match:
            facts.append(PersonFact(speaker, "age", age_match.group(1), text[:100]))

        # "X years ago" patterns
        years_ago_match = re.search(r"(\d+)\s+years?\s+ago", text_lower)
        if years_ago_match:
            if "birthday" in text_lower or "18" in text_lower:
                facts.append(
                    PersonFact(speaker, "years_since_18", years_ago_match.group(1), text[:100])
                )

        # Enhanced year extraction with event-specific patterns
        # These patterns capture WHEN specific events happened
        event_year_patterns = [
            # Book reading: "read X in 2022", "finished reading X back in 2022"
            (
                r"(?:read|finished reading|started reading)"
                r"\s+(?:[\"'])?([^\"']+)(?:[\"'])?"
                r"\s+(?:in\s+|back in\s+)?(20\d{2}|last year)",
                "book_year",
            ),
            # Painting: "painted a sunset in 2022", "drew X last year"
            (
                r"(?:painted|drew|finished painting)"
                r"\s+(?:a\s+)?(?:the\s+)?(\w+(?:\s+\w+)?)"
                r"\s+(?:in\s+|back in\s+)?(20\d{2}|last year)",
                "painting_year",
            ),
            # Events attended: "attended the conference in 2022"
            (
                r"(?:attended|went to)"
                r"\s+(?:the\s+)?(\w+(?:\s+\w+)?)"
                r"\s+(?:in\s+|back in\s+)?(20\d{2}|last year)",
                "event_year",
            ),
            # Pride/festival: "Pride Festival 2022", "pride in 2022"
            (r"(?:pride|festival|parade)\s+(?:in\s+)?(20\d{2}|last year)", "pride_year"),
            # Activities started: "started pottery in 2016"
            (
                r"(?:started|began|took up)"
                r"\s+(\w+(?:\s+\w+)?)"
                r"\s+(?:in\s+|back in\s+)?(20\d{2}|last year)",
                "activity_start_year",
            ),
        ]

        for pattern, fact_type in event_year_patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                year_value = groups[-1]  # Year is always last group
                # Convert "last year" to actual year
                if year_value == "last year":
                    base_year = 2023
                    if session_date:
                        year_match = re.search(r"20\d{2}", session_date)
                        if year_match:
                            base_year = int(year_match.group())
                    year_value = str(base_year - 1)
                facts.append(PersonFact(speaker, fact_type, year_value, text[:100]))

        # Year-only patterns - capture "in 2022", "back in 2016", "since 2016"
        # For temporal questions like "When did Melanie read the book?"
        year_patterns = [
            (r"(?:in|back in|around)\s+(20\d{2})", "year_mentioned"),
            (r"since\s+(20\d{2})", "since_year"),
            (r"(\d+)\s+years?\s+(?:ago|now)", "years_duration"),
        ]

        for pattern, year_type in year_patterns:
            match = re.search(pattern, text_lower)
            if match:
                year_value = match.group(1)
                # Find context - what activity/topic is near this year mention
                context_start = max(0, match.start() - 50)
                context_end = min(len(text_lower), match.end() + 50)
                context = text_lower[context_start:context_end]

                # Associate with specific topics
                if "book" in context or "read" in context:
                    facts.append(PersonFact(speaker, "book_year", year_value, text[:100]))
                elif "art" in context or "paint" in context or "pottery" in context:
                    facts.append(PersonFact(speaker, "art_since_year", year_value, text[:100]))
                elif "married" in context or "wedding" in context:
                    facts.append(PersonFact(speaker, "married_year", year_value, text[:100]))
                else:
                    # Generic year fact
                    facts.append(PersonFact(speaker, year_type, year_value, text[:100]))

        # Relative date preservation - store BOTH relative and absolute formats
        # This helps answer questions about "the week before X" or "yesterday"
        relative_date_patterns = [
            (r"the week before (\d{1,2}\s+\w+\s*,?\s*\d{4})", "temporal_week_before"),
            (r"the day before (\d{1,2}\s+\w+\s*,?\s*\d{4})", "temporal_day_before"),
            (r"two weeks before (\d{1,2}\s+\w+\s*,?\s*\d{4})", "temporal_two_weeks_before"),
            (
                r"(yesterday|last night|this morning|last week|last month|last weekend)",
                "temporal_relative",
            ),
            (r"(a few days ago|couple of days ago|other day)", "temporal_recent"),
        ]

        for pattern, rel_type in relative_date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                relative_expr = match.group(1)
                facts.append(
                    PersonFact(
                        speaker, rel_type, relative_expr, text[:100], session_date=session_date
                    )
                )

        # "Last year" patterns - convert to actual year based on session date
        # Session dates are typically in 2023 for LoCoMo, so "last year" = 2022
        if "last year" in text_lower:
            # Determine year from session_date if available, else assume 2023
            base_year = 2023
            if session_date:
                year_match = re.search(r"20\d{2}", session_date)
                if year_match:
                    base_year = int(year_match.group())
            last_year = str(base_year - 1)

            # Find context - what activity is associated with "last year"
            context_start = max(0, text_lower.find("last year") - 50)
            context_end = min(len(text_lower), text_lower.find("last year") + 60)
            context = text_lower[context_start:context_end]

            if "book" in context or "read" in context:
                facts.append(PersonFact(speaker, "book_year", last_year, text[:100]))
            elif "paint" in context or "sunrise" in context or "art" in context:
                facts.append(PersonFact(speaker, "painting_year", last_year, text[:100]))
            elif "pride" in context or "festival" in context:
                facts.append(PersonFact(speaker, "pride_year", last_year, text[:100]))
            else:
                facts.append(PersonFact(speaker, "last_year_event", last_year, text[:100]))

        # Location patterns
        location_match = re.search(
            r"(?:from|moved from|home country[,\s]+)\s*"
            r"(sweden|norway|denmark|finland|germany"
            r"|france|uk|usa|canada|australia)",
            text_lower,
        )
        if location_match:
            facts.append(
                PersonFact(speaker, "location", location_match.group(1).title(), text[:100])
            )

        # Family patterns - number of kids (enhanced with more patterns)
        num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

        # Pattern 1: Direct count - "have three kids", "my 3 children"
        kids_match = re.search(
            r"(?:have|has|my|our|with)"
            r"\s+(\d+|three|two|one|four|five)"
            r"\s+(?:kids?|children|little ones)",
            text_lower,
        )
        if kids_match:
            num = kids_match.group(1)
            facts.append(PersonFact(speaker, "num_children", num_map.get(num, num), text[:100]))

        # Pattern 2: Compound patterns - "my daughter and two sons" = 3
        compound_match = re.search(
            r"(?:my|our)\s+(?:daughter|son)\s+and\s+(two|three|2|3)\s+(?:sons?|daughters?)",
            text_lower,
        )
        if compound_match:
            additional = compound_match.group(1)
            total = 1 + int(num_map.get(additional, additional))
            facts.append(PersonFact(speaker, "num_children", str(total), text[:100]))

        # Pattern 3: Multiple mentions - "my son X and daughters Y and Z"
        multi_child_match = re.search(
            r"(?:my|our)\s+(?:son|daughter)s?\s+\w+\s+and\s+(?:son|daughter)s?\s+\w+(?:\s+and\s+(?:son|daughter)s?\s+\w+)?",
            text_lower,
        )
        if multi_child_match:
            # Count 'and' to determine number of children
            child_text = multi_child_match.group(0)
            count = child_text.count(" and ") + 1
            facts.append(PersonFact(speaker, "num_children", str(count), text[:100]))

        # Pattern 4: List of kids' names - "the kids - Emma, Jake, and Lily"
        kids_list_match = re.search(
            r"(?:kids?|children)(?:\s*[-:]\s*|\s+are\s+)(\w+),\s*(\w+)(?:,?\s*and\s+(\w+))?",
            text_lower,
        )
        if kids_list_match:
            names = [g for g in kids_list_match.groups() if g]
            count = len(names)
            if count > 0:
                facts.append(PersonFact(speaker, "num_children", str(count), text[:100]))
                for name in names:
                    facts.append(PersonFact(speaker, "child_name", name.title(), text[:100]))

        # Pattern 5: "2 younger kids" implies 3 total (2 younger + at least 1 older)
        younger_kids_match = re.search(
            r"(?:the\s+)?(\d+|two|three)\s+younger\s+(?:kids?|children|ones)", text_lower
        )
        if younger_kids_match:
            younger_count = younger_kids_match.group(1)
            younger_num = int(num_map.get(younger_count, younger_count))
            # "N younger" implies at least N+1 total
            total = younger_num + 1
            facts.append(PersonFact(speaker, "num_children", str(total), text[:100]))

        # Pattern 6: "daughter's birthday" + other mentions implies multiple children
        if "daughter" in text_lower and "birthday" in text_lower:
            facts.append(PersonFact(speaker, "has_daughter", "yes", text[:100]))

        # Kids' interests - look for patterns like "kids like dinosaurs"
        kids_like_match = re.search(
            r"(?:kids?|children|son|daughter)\s+(?:like|love|enjoy|are into)\s+([^.,]+)", text_lower
        )
        if kids_like_match:
            facts.append(
                PersonFact(speaker, "kids_like", kids_like_match.group(1).strip(), text[:100])
            )

        # Pet patterns - names specifically (only match common pet names, not random words)
        common_pet_names = [
            "oliver",
            "luna",
            "bailey",
            "max",
            "bella",
            "charlie",
            "lucy",
            "buddy",
            "daisy",
            "rocky",
        ]
        for name in common_pet_names:
            if name in text_lower:
                facts.append(PersonFact(speaker, "pet_name", name.title(), text[:100]))

        # Pet types
        if "two cats" in text_lower or "2 cats" in text_lower:
            facts.append(PersonFact(speaker, "pets", "two cats", text[:100]))
        if "dog" in text_lower and ("my dog" in text_lower or "our dog" in text_lower):
            facts.append(PersonFact(speaker, "pets", "dog", text[:100]))
        if "guinea pig" in text_lower:
            facts.append(PersonFact(speaker, "pets", "guinea pig", text[:100]))

        # Activity patterns
        activity_patterns = [
            (r"signed up for (?:a )?([\w\s]+(?:class|workshop|course))", "activity"),
            (r"(?:i|we) (?:went|go) ([\w]+ing)", "activity"),
            (r"(?:i|we) (?:like|love|enjoy) ([\w]+ing)", "preference"),
            (r"(?:i|we) (?:play|plays?) (?:the )?([\w]+)", "activity"),
            (r"(?:started|began) ([\w]+ing)", "activity"),
            (r"(?:do|does) (pottery|painting|camping|hiking|swimming|running)", "activity"),
        ]
        for pattern, fact_type in activity_patterns:
            match = re.search(pattern, text_lower)
            if match:
                facts.append(PersonFact(speaker, fact_type, match.group(1), text[:100]))

        # Patriotism and military patterns
        if any(
            p in text_lower
            for p in [
                "serve my country",
                "serve our country",
                "respect for military",
                "wanted to join the military",
                "support the military",
                "serving our nation",
                "stand up for what we believe",
            ]
        ):
            facts.append(PersonFact(speaker, "patriotic", "yes", text[:100]))

        # Political aspirations
        if any(
            p in text_lower
            for p in [
                "running for office",
                "run for office",
                "local politics",
                "into politics",
                "policymaking",
            ]
        ):
            facts.append(PersonFact(speaker, "political_aspiration", "yes", text[:100]))
            facts.append(PersonFact(speaker, "career_goal", "politics", text[:100]))

        # Degree/education from self-statements
        degree_match = re.search(r"(?:because of |with )?my degree", text_lower)
        if degree_match:
            # Context around degree mention
            if "policymaking" in text_lower or "politics" in text_lower:
                facts.append(PersonFact(speaker, "degree_field", "political science", text[:100]))

        # US-specific goals (suggests not open to moving abroad)
        if any(
            p in text_lower
            for p in [
                "join the military",
                "run for office",
                "local politics",
                "my community",
                "our neighborhood",
            ]
        ):
            facts.append(PersonFact(speaker, "us_focused_goals", "yes", text[:100]))

        # Camping locations
        camp_match = re.search(
            r"camp(?:ed|ing)?\s+(?:at|in|on|near)\s+(?:the\s+)?(beach|mountains?|forest|lake)",
            text_lower,
        )
        if camp_match:
            facts.append(PersonFact(speaker, "camped_at", camp_match.group(1), text[:100]))

        # Art/painting patterns
        # Capture art types like "abstract art", "modern art"
        art_type_match = re.search(
            r"(abstract|modern|contemporary|surreal|impressionist)\s+art", text_lower
        )
        if art_type_match:
            facts.append(
                PersonFact(speaker, "art_type", art_type_match.group(1) + " art", text[:100])
            )

        # Capture specific things painted - expanded patterns
        # "painted a sunset", "painting of a horse", "finished painting the sunrise"
        _subjects = (
            r"sunset|sunrise|horse|portrait|landscape"
            r"|self[- ]?portrait|flowers?|trees?"
            r"|mountains?|beach|ocean|sky"
        )
        _subjects_short = r"sunset|sunrise|horse|portrait|landscape"
        paint_subject_patterns = [
            r"painted\s+(?:a\s+)?(" + _subjects + r")",
            (r"painting\s+(?:a\s+|of\s+(?:a\s+)?)?" r"(?:the\s+)?(" + _subjects + r")"),
            (
                r"finished\s+(?:painting|my painting of)"
                r"\s+(?:a\s+|the\s+)?(" + _subjects_short + r")"
            ),
            (
                r"(?:my|a)\s+(?:new\s+)?painting"
                r"\s+(?:of\s+)?(?:a\s+|the\s+)?(" + _subjects_short + r")"
            ),
        ]
        for pattern in paint_subject_patterns:
            match = re.search(pattern, text_lower)
            if match:
                subject = match.group(1).strip()
                facts.append(PersonFact(speaker, "painted", subject, text[:100]))
                break  # Only add once

        # More general painting pattern - "been painting" without specific subject
        if "been painting" in text_lower or "started painting" in text_lower:
            facts.append(PersonFact(speaker, "activity", "painting", text[:100]))

        # Capture pottery - only when explicit context shows it's the speaker's activity
        if "pottery" in text_lower:
            # Check context more carefully
            if any(
                p in text_lower
                for p in [
                    "i signed up for pottery",
                    "my pottery",
                    "i do pottery",
                    "pottery class",
                    "pottery workshop",
                    "made pottery",
                    "been doing pottery",
                    "love pottery",
                ]
            ):
                facts.append(PersonFact(speaker, "activity", "pottery class", text[:100]))

        if "sculpt" in text_lower:
            facts.append(PersonFact(speaker, "art", "sculpture", text[:100]))

        # Book patterns - capture quoted titles and favorite books
        book_match = re.search(r'"([^"]+)"', text)
        if book_match:
            title = book_match.group(1).strip()
            if len(title) > 3 and len(title) < 50:
                facts.append(PersonFact(speaker, "book", title, text[:100]))
                # Check if it's a favorite/recommended
                if "favorite" in text_lower or "loved" in text_lower or "childhood" in text_lower:
                    facts.append(PersonFact(speaker, "favorite_book", title, text[:100]))
                if "recommend" in text_lower or "should read" in text_lower:
                    facts.append(PersonFact(speaker, "recommended_book", title, text[:100]))

        # Pattern for reading a book recommended by someone else (cross-reference)
        # "reading that book you recommended" -> link to partner's recommended_book
        rec_book_match = re.search(
            r"(?:reading|read)\s+(?:that|the)\s+book\s+(?:you|(\w+))\s+recommend", text_lower
        )
        if rec_book_match:
            recommender = (
                rec_book_match.group(1) if rec_book_match.group(1) else self._current_partner
            )
            if recommender:
                facts.append(
                    PersonFact(speaker, "reading_recommended_from", recommender.lower(), text[:100])
                )

        # Music/instrument patterns
        instrument_match = re.search(
            r"(?:play|plays?)\s+(?:the\s+)?(violin|piano|guitar|clarinet|flute|drums|cello)",
            text_lower,
        )
        if instrument_match:
            facts.append(PersonFact(speaker, "instrument", instrument_match.group(1), text[:100]))

        # Favorite musicians
        musician_match = re.search(
            r"(?:like|love|enjoy|fan of)\s+(bach|mozart|beethoven|ed sheeran|taylor swift)",
            text_lower,
        )
        if musician_match:
            facts.append(
                PersonFact(
                    speaker, "favorite_musician", musician_match.group(1).title(), text[:100]
                )
            )

        # Concerts/bands seen - capture actual band names
        concert_patterns = [
            r"(?:saw|went to see|concert by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Proper nouns
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:concert|show|performance)",
        ]
        for pattern in concert_patterns:
            match = re.search(pattern, text)  # Case-sensitive for proper nouns
            if match:
                band_name = match.group(1).strip()
                # Filter out common non-band words
                if band_name.lower() not in [
                    "the",
                    "a",
                    "my",
                    "that",
                    "this",
                    "one",
                    "great",
                    "amazing",
                ]:
                    facts.append(PersonFact(speaker, "concerts_seen", band_name, text[:100]))

        # Running/exercise for destress
        if "run" in text_lower or "running" in text_lower:
            if any(
                p in text_lower
                for p in ["destress", "de-stress", "clear my mind", "headspace", "mental health"]
            ):
                facts.append(PersonFact(speaker, "destress", "running", text[:100]))
            elif (
                "i run" in text_lower
                or "started running" in text_lower
                or "go running" in text_lower
            ):
                facts.append(PersonFact(speaker, "activity", "running", text[:100]))

        # Research patterns - capture what was researched (limited to key phrases)
        research_match = re.search(
            r"(?:research|researching|researched)\s+(adoption|agencies|schools|universities|jobs|careers|options)",
            text_lower,
        )
        if research_match:
            facts.append(
                PersonFact(speaker, "researched", research_match.group(1).strip(), text[:100])
            )
        # Special case for "adoption agencies"
        if "adoption agenc" in text_lower:
            facts.append(PersonFact(speaker, "researched", "adoption agencies", text[:100]))

        # Career/counseling patterns
        if "counseling" in text_lower or "mental health" in text_lower:
            if "pursue" in text_lower or "career" in text_lower or "want to" in text_lower:
                facts.append(
                    PersonFact(speaker, "career", "counseling or mental health", text[:100])
                )

        # Mentoring/school/children helping events
        if "mentor" in text_lower:
            if "program" in text_lower or "mentoring" in text_lower or "join" in text_lower:
                facts.append(
                    PersonFact(speaker, "children_events", "mentoring program", text[:100])
                )

        if "school" in text_lower:
            if "speech" in text_lower or "spoke" in text_lower or "talk" in text_lower:
                facts.append(PersonFact(speaker, "children_events", "school speech", text[:100]))

        if "youth center" in text_lower or "youth" in text_lower:
            facts.append(PersonFact(speaker, "children_events", "youth center", text[:100]))

        # Symbols/important items
        if "rainbow flag" in text_lower:
            facts.append(PersonFact(speaker, "symbol", "Rainbow flag", text[:100]))
        if "transgender symbol" in text_lower or "trans symbol" in text_lower:
            facts.append(PersonFact(speaker, "symbol", "transgender symbol", text[:100]))

        # Support patterns
        if "support" in text_lower:
            supporters = []
            if "mentor" in text_lower:
                supporters.append("mentors")
            if "family" in text_lower or "families" in text_lower:
                supporters.append("family")
            if "friend" in text_lower:
                supporters.append("friends")
            if supporters:
                facts.append(PersonFact(speaker, "supporters", ", ".join(supporters), text[:100]))

        # Temporal event extraction - capture events with dates
        # "signed up for pottery class" in a session with known date
        event_patterns = [
            (
                r"(?:signed up for|joined)\s+(?:a\s+)?"
                r"(pottery|camping|hiking|mentorship"
                r"|activist|yoga)",
                "signed_up",
            ),
            (
                r"(?:went to|visited)\s+(?:the\s+)?"
                r"(museum|park|beach|conference"
                r"|parade|workshop|concert)",
                "visited",
            ),
            (r"(?:went|go)\s+(camping|hiking|biking|swimming|running)", "activity"),
            (r"(?:had|went to)\s+(?:a\s+)?(picnic|meeting|adoption|interview|birthday)", "event"),
            (r"(?:attended|went to)\s+(?:a\s+)?(pride|lgbtq|transgender)", "attended"),
            (
                r"(?:painted|drew|made)\s+(?:a\s+)?(portrait|plate|bowl|sunset|self-portrait)",
                "created",
            ),
            (r"(?:bought|purchased)\s+(?:some\s+)?(figurines|shoes|books)", "purchased"),
            (r"(?:daughter|son|kid)(?:'s)?\s+birthday", "family_event"),
            (r'(?:read)\s+(?:the book\s+)?["\']([^"\']+)["\']', "read_book"),
        ]

        for pattern, event_type in event_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Only index events that are COMPLETED (past tense), not PLANNED (future tense)
                # This prevents "I'm going to the conference next week" from being indexed
                # with the current session's date - only "I went to the conference" gets indexed
                if not self._is_completed_event(text_lower, match.start()):
                    continue  # Skip planned events - they shouldn't use this session's date

                event_value = match.group(1) if match.groups() else event_type
                fact = PersonFact(
                    speaker,
                    f"event_{event_type}",
                    event_value,
                    text[:100],
                    session_date=session_date,
                )
                facts.append(fact)

        # Secondary entity extraction - extract facts about MENTIONED people, not just speaker
        # This captures facts like "John wants to run for office" when Maria is speaking
        secondary_entity_facts = self._extract_secondary_entity_facts(text, speaker)
        facts.extend(secondary_entity_facts)

        return facts

    def _extract_all_mentioned_entities(self, text: str, speaker: str) -> set[str]:
        """
        Dynamically extract ALL entities mentioned in text.

        Uses multiple patterns to find proper nouns, possessives, and common names.
        """
        entities = set()

        # Pattern 1: Extract ALL capitalized proper nouns (not at sentence start)
        # Match words starting with capital letter that appear after punctuation/space
        proper_noun_pattern = r"(?<=[.!?\s])\s*([A-Z][a-z]{2,})\b"
        for match in re.finditer(proper_noun_pattern, text):
            name = match.group(1)
            # Filter out common non-name words
            skip_words = {
                "the",
                "this",
                "that",
                "when",
                "where",
                "what",
                "which",
                "who",
                "how",
                "why",
                "yes",
                "no",
                "but",
                "and",
                "its",
                "they",
                "them",
                "she",
                "her",
                "his",
                "him",
                "our",
                "your",
                "been",
                "have",
                "has",
                "was",
                "were",
                "are",
                "had",
                "for",
                "not",
                "you",
                "all",
                "can",
                "may",
                "june",
                "july",
                "august",
                "january",
                "february",
                "march",
                "april",
                "september",
                "october",
                "november",
                "december",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            }
            if name.lower() not in skip_words:
                entities.add(name.lower())

        # Pattern 2: Possessive forms like "John's" or "Maria's brother"
        possessive_pattern = r"([A-Z][a-z]+)'s\b"
        for match in re.finditer(possessive_pattern, text):
            entities.add(match.group(1).lower())

        # Pattern 3: Names at sentence start followed by verb patterns
        # "John wants", "Maria said", "Tom is"
        sentence_start_pattern = (
            r"^([A-Z][a-z]{2,})\s+(?:wants?|said|is"
            r"|was|has|had|loves?|likes?|went|did"
            r"|does|thinks?)\b"
        )
        for match in re.finditer(sentence_start_pattern, text, re.MULTILINE):
            name = match.group(1)
            if name.lower() not in {"the", "this", "that"}:
                entities.add(name.lower())

        # Pattern 4: "my friend/brother/etc X" patterns
        relation_pattern = (
            r"(?:my|her|his|their)\s+"
            r"(?:friend|brother|sister|mother|father"
            r"|husband|wife|son|daughter)"
            r"\s+([A-Z][a-z]+)"
        )
        for match in re.finditer(relation_pattern, text):
            entities.add(match.group(1).lower())

        # Remove the speaker from entities
        entities.discard(speaker.lower())

        return entities

    def _extract_secondary_entity_facts(self, text: str, speaker: str) -> list[PersonFact]:
        """
        Extract facts about entities MENTIONED in the text, not the speaker.

        Examples:
        - "John wants to run for office" → John: goal=political office
        - "Maria's brother is a doctor" → Maria's brother: profession=doctor
        - "Gina loves hiking" → Gina: preference=hiking
        """
        facts = []
        text_lower = text.lower()

        # Dynamically extract all mentioned entities
        mentioned_entities = self._extract_all_mentioned_entities(text, speaker)

        # Extract facts for each mentioned entity
        for entity in mentioned_entities:
            # Goal/aspiration patterns
            goal_patterns = [
                (
                    rf"{entity}\s+(?:wants?|hopes?|dreams?|plans?)\s+to\s+(\w+(?:\s+\w+)?(?:\s+\w+)?)",
                    "goal",
                ),
                (rf"{entity}\s+(?:is going|will)\s+to\s+(\w+(?:\s+\w+)?)", "future_plan"),
            ]
            for pattern, fact_type in goal_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    facts.append(
                        PersonFact(entity.title(), fact_type, match.group(1).strip(), text[:100])
                    )

            # Profession patterns
            prof_patterns = [
                (
                    rf"{entity}\s+(?:is|works as)\s+(?:a\s+)?"
                    r"(doctor|lawyer|teacher|engineer"
                    r"|nurse|developer|designer"
                    r"|artist|writer)",
                    "profession",
                ),
                (
                    rf"{entity}(?:\'s)?\s+(?:job|career|work)\s+(?:is|as)\s+(?:a\s+)?(\w+)",
                    "profession",
                ),
            ]
            for pattern, fact_type in prof_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    facts.append(
                        PersonFact(entity.title(), fact_type, match.group(1).strip(), text[:100])
                    )

            # Activity/hobby patterns for mentioned entity
            activity_patterns = [
                (rf"{entity}\s+(?:likes?|loves?|enjoys?)\s+(\w+ing)", "preference"),
                (rf"{entity}\s+(?:is|was)\s+(?:good at|into)\s+(\w+)", "skill"),
                (rf"{entity}\s+(?:plays?|does)\s+(\w+)", "activity"),
            ]
            for pattern, fact_type in activity_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    facts.append(
                        PersonFact(entity.title(), fact_type, match.group(1).strip(), text[:100])
                    )

            # Education patterns
            edu_patterns = [
                (
                    rf"{entity}\s+(?:studies?|studied|majors?|majored)\s+(?:in\s+)?(\w+(?:\s+\w+)?)",
                    "education",
                ),
                (
                    rf"{entity}\s+(?:has|got)\s+(?:a\s+)?(?:degree|phd|masters?)\s+in\s+(\w+(?:\s+\w+)?)",
                    "degree",
                ),
            ]
            for pattern, fact_type in edu_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    facts.append(
                        PersonFact(entity.title(), fact_type, match.group(1).strip(), text[:100])
                    )

            # Patriotism/country patterns
            patriot_patterns = [
                (
                    rf"{entity}\s+(?:loves?|wants? to serve"
                    r"|is proud of)\s+(?:his |her "
                    r"|the )?country",
                    "patriotic",
                ),
                (
                    rf"{entity}\s+(?:wants? to|plans? to"
                    r"|dreams? of)\s+(?:join|serve in)"
                    r"\s+(?:the )?(military|army"
                    r"|navy|marines)",
                    "patriotic",
                ),
                (
                    rf"{entity}\s+(?:wants? to|plans? to)\s+run for (?:office|congress|senate)",
                    "political_aspiration",
                ),
            ]
            for pattern, fact_type in patriot_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    facts.append(PersonFact(entity.title(), fact_type, "yes", text[:100]))

            # Location/proximity patterns
            location_patterns = [
                (
                    rf"{entity}\s+(?:lives?|grew up|is from)"
                    r"\s+(?:near|close to|by)"
                    r"\s+(?:the )?(beach|ocean"
                    r"|mountains?|lake)",
                    "lives_near",
                ),
                (
                    rf"{entity}.*(?:vacation|trip)"
                    r"\s+to\s+(?:the )?(beach"
                    r"|california|florida|hawaii)",
                    "vacation_location",
                ),
            ]
            for pattern, fact_type in location_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    facts.append(
                        PersonFact(entity.title(), fact_type, match.group(1).strip(), text[:100])
                    )

            # INNOVATION: Preference indicators for inference questions
            preference_indicators = [
                # Nature/outdoor preferences
                (
                    rf"{entity}\s+(?:loves?|enjoys?|likes?)"
                    r"\s+(?:the\s+)?(outdoors?|nature"
                    r"|camping|hiking|mountains?|beach)",
                    "prefers_outdoor",
                ),
                # Book/reading preferences
                (
                    rf"{entity}\s+(?:collects?|has|owns?)"
                    r"\s+(?:classic\s+)?(children\'s"
                    r" books?|books?|novels?)",
                    "book_collection",
                ),
                # Art preferences
                (
                    rf"{entity}\s+(?:loves?|enjoys?|likes?)\s+(art|painting|pottery|music|dance)",
                    "art_preference",
                ),
                # Experience preferences (positive/negative)
                (
                    rf"{entity}\s+(?:had\s+a\s+)?(?:bad|terrible|awful|negative)\s+(?:experience|trip|time)",
                    "negative_experience",
                ),
                (
                    rf"{entity}\s+(?:had\s+a\s+)?(?:great|wonderful|amazing|positive)\s+(?:experience|trip|time)",
                    "positive_experience",
                ),
            ]
            for pattern, fact_type in preference_indicators:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1).strip() if match.groups() else "yes"
                    facts.append(PersonFact(entity.title(), fact_type, value, text[:100]))

            # Personality/attribute patterns - ENHANCED for multi-hop questions
            attribute_patterns = [
                (
                    rf"{entity}\s+(?:is|seems?|appears?)"
                    r"\s+(?:so |very |really )?"
                    r"(selfless|kind|caring|passionate"
                    r"|rational|driven|thoughtful"
                    r"|authentic|courageous|strong"
                    r"|inspiring|brave|dedicated)",
                    "attribute",
                ),
                (
                    rf"{entity}\s+(?:always|really)"
                    r"\s+(?:puts?|thinks? of)"
                    r"\s+(?:others|family)\s+first",
                    "attribute",
                ),
                # Descriptions by others about personality
                (
                    rf"{entity}.*(?:courage|strength"
                    r"|inspiration|self-acceptance"
                    r"|determination)",
                    "personality_trait",
                ),
                (
                    rf"(?:admire|love|appreciate).*{entity}.*(?:for|because)",
                    "personality_described",
                ),
                # Specific trait mentions
                (
                    rf"{entity}\s+(?:has|shows?"
                    r"|demonstrates?)\s+(?:such\s+)?"
                    r"(courage|strength|authenticity"
                    r"|determination|compassion)",
                    "personality_trait",
                ),
                # "You are so X" patterns when directed at entity
                (
                    r"(?:you are|you\'re)"
                    r"\s+(?:so\s+|such\s+a\s+)?"
                    r"(thoughtful|amazing|incredible"
                    r"|inspiring|brave|strong"
                    r"|kind|authentic)",
                    "personality_trait",
                ),
            ]
            for pattern, fact_type in attribute_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1) if match.groups() else "caring"
                    facts.append(PersonFact(entity.title(), fact_type, value.strip(), text[:100]))

            # INNOVATION: Extract personality descriptions from "would describe" patterns
            describe_patterns = [
                (
                    rf"(?:would|might)\s+(?:describe|say)\s+{entity}\s+(?:is|as)\s+([\w\s,]+)",
                    "personality_description",
                ),
                (
                    rf"{entity}\s+(?:is\s+)?(?:described|known)\s+(?:as|for)\s+([\w\s,]+)",
                    "personality_description",
                ),
            ]
            for pattern, fact_type in describe_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    traits = match.group(1).strip()
                    # Split on commas or 'and'
                    for trait in re.split(r"[,\s]+(?:and\s+)?", traits):
                        trait = trait.strip()
                        if trait and len(trait) > 2:
                            facts.append(
                                PersonFact(entity.title(), "personality_trait", trait, text[:100])
                            )

        # NOTE: Cross-person trait extraction moved to
        # evaluation.profile_answerer.ProfileAnswerer._extract_cross_person_traits

        return facts

    # NOTE: The following methods were moved to
    # evaluation.profile_answerer.ProfileAnswerer:
    # - _add_to_profile, _apply_inference_rules
    # - get_profile, get_profile_text
    # - answer_from_profile, answer_temporal_from_profile
    # - answer_inference_question, _extract_date_from_timestamp
    # - clear, track_conversation_relationship
    # - add_cross_person_trait, get_cross_person_traits, is_lgbtq_ally
    _PROFILE_METHODS_MOVED = True
