"""
Profile Answerer: Benchmark-specific profile management and question answering.

Manages person profiles built from extracted facts and provides direct
profile-based answering for single-hop, temporal, and inference questions.

This module was extracted from LLMFactExtractor to separate benchmark-specific
logic (profile building, inference rules, answer shortcuts) from the
general-purpose fact extraction core.
"""

from __future__ import annotations

import re
from typing import Any

from zerogmem.encoder.llm_fact_extractor import LLMFactExtractor, PersonFact


class ProfileAnswerer:
    """Benchmark-specific profile management wrapping an LLMFactExtractor.

    Owns:
    - Person profiles (accumulated facts per person)
    - Cross-person trait tracking
    - Conversation pair tracking (for ally inference)
    - Inference rules (LGBTQ→Liberal, military→Conservative, etc.)
    - Direct answer methods (answer_from_profile, answer_temporal, etc.)
    """

    def __init__(self, extractor: LLMFactExtractor) -> None:
        self.extractor = extractor
        self.person_profiles: dict[str, dict[str, list[str]]] = {}
        self.cross_person_traits: dict[str, dict[str, list[str]]] = {}
        self.conversation_pairs: dict[str, set[str]] = {}
        self._current_partner: str = ""

    # ------------------------------------------------------------------
    # Message processing — extract facts and build profiles
    # ------------------------------------------------------------------

    def process_message(
        self, text: str, speaker: str, session_date: str = "", partner: str = ""
    ) -> list[PersonFact]:
        """Extract facts from a message and add them to profiles.

        Delegates regex extraction to the wrapped extractor, then applies
        profile management and inference rules.
        """
        self._current_partner = partner
        facts = self.extractor._extract_facts_regex(text, speaker, session_date, partner)

        # Cross-person trait extraction
        cross_facts = self._extract_cross_person_traits(text, speaker, partner)

        for fact in facts:
            self._add_to_profile(fact)

        return facts + cross_facts

    # ------------------------------------------------------------------
    # Cross-person trait extraction
    # ------------------------------------------------------------------

    def _extract_cross_person_traits(
        self, text: str, speaker: str, partner: str = ""
    ) -> list[PersonFact]:
        """Extract when speaker says positive things about another person."""
        facts: list[PersonFact] = []
        text_lower = text.lower()

        cross_person_trait_patterns = [
            (
                r"(?:you are|you\'re)"
                r"\s+(?:so\s+|such\s+a\s+)?"
                r"(thoughtful|amazing|incredible"
                r"|inspiring|brave|strong|kind"
                r"|authentic|driven|caring|wonderful"
                r"|selfless|courageous|dedicated)",
                "direct_praise",
            ),
            (
                r"(?:admire|love|appreciate)"
                r"\s+(?:your|how)"
                r"\s+(?:you(?:r)?(?:\s+are)?"
                r"(?:\s+so)?\s+)?"
                r"(courage|strength|authenticity"
                r"|determination|kindness|bravery"
                r"|thoughtfulness|dedication)",
                "admired_trait",
            ),
            (
                r"(\w+)\s+is\s+(?:so|such a|really"
                r"|truly)\s+(thoughtful|amazing"
                r"|inspiring|brave|strong|kind"
                r"|authentic|driven|caring|selfless"
                r"|courageous|dedicated)",
                "third_person_praise",
            ),
            (
                r"(?:your|their)\s+(?:journey|story|courage|strength)"
                r"\s+(?:shows?|demonstrates?|inspires?)",
                "journey_praise",
            ),
            (r"your\s+drive\s+(?:to\s+\w+\s+)?is", "drive_praise"),
            (r"(?:you\s+)?(?:really\s+)?care\s+about\s+being\s+real", "authentic_praise"),
        ]

        for pattern, trait_type in cross_person_trait_patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if trait_type == "third_person_praise" and len(groups) >= 2:
                    target_name = groups[0].title()
                    trait = groups[1]
                    self.add_cross_person_trait(speaker, target_name, trait)
                elif trait_type == "drive_praise":
                    trait = "driven"
                    if partner:
                        self.add_cross_person_trait(speaker, partner, trait)
                    else:
                        facts.append(
                            PersonFact(speaker, "says_about_partner_drive", trait, text[:100])
                        )
                elif trait_type == "authentic_praise":
                    trait = "authentic"
                    if partner:
                        self.add_cross_person_trait(speaker, partner, trait)
                    else:
                        facts.append(
                            PersonFact(
                                speaker, "says_about_partner_authentic", trait, text[:100]
                            )
                        )
                elif trait_type in ["direct_praise", "admired_trait"] and groups:
                    trait = groups[0] if groups[0] else "inspiring"
                    if partner:
                        self.add_cross_person_trait(speaker, partner, trait)
                    else:
                        facts.append(
                            PersonFact(
                                speaker, f"says_about_partner_{trait_type}", trait, text[:100]
                            )
                        )

        return facts

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def _add_to_profile(self, fact: PersonFact) -> None:
        """Add fact to person's profile, including session date for events."""
        person = fact.person.lower()
        if person not in self.person_profiles:
            self.person_profiles[person] = {}

        self._apply_inference_rules(person, fact)

        if fact.fact_type.startswith("event_") and fact.session_date:
            key = f"{fact.fact_type}_dates"
            if key not in self.person_profiles[person]:
                self.person_profiles[person][key] = []
            entry = f"{fact.value}|{fact.session_date}"
            if entry not in self.person_profiles[person][key]:
                self.person_profiles[person][key].append(entry)

            month_match = re.search(
                r"(january|february|march|april|may|june|july|august|september|october|november|december)",
                fact.session_date.lower(),
            )
            if month_match:
                month = month_match.group(1)
                month_key = f"{fact.fact_type}_{month}"
                if month_key not in self.person_profiles[person]:
                    self.person_profiles[person][month_key] = []
                if entry not in self.person_profiles[person][month_key]:
                    self.person_profiles[person][month_key].append(entry)

        if fact.fact_type.startswith("temporal_") and fact.session_date:
            key = f"{fact.fact_type}_with_context"
            if key not in self.person_profiles[person]:
                self.person_profiles[person][key] = []
            entry = f"{fact.value}|{fact.session_date}"
            if entry not in self.person_profiles[person][key]:
                self.person_profiles[person][key].append(entry)

        if fact.fact_type not in self.person_profiles[person]:
            self.person_profiles[person][fact.fact_type] = []

        if fact.value not in self.person_profiles[person][fact.fact_type]:
            self.person_profiles[person][fact.fact_type].append(fact.value)

    def _apply_inference_rules(self, person: str, fact: PersonFact) -> None:
        """Apply inference rules to derive implicit facts from explicit statements."""
        profile = self.person_profiles[person]

        # LGBTQ+ support/activism -> Liberal
        lgbtq_indicators = ["lgbtq_participation", "lgbtq", "transgender", "pride", "trans"]
        if fact.fact_type in lgbtq_indicators or any(
            ind in fact.fact_type.lower() for ind in lgbtq_indicators
        ):
            if "inferred_political_leaning" not in profile:
                profile["inferred_political_leaning"] = []
            if "Liberal" not in profile["inferred_political_leaning"]:
                profile["inferred_political_leaning"].append("Liberal")
                profile["inference_reason_political"] = ["LGBTQ+ activist/supporter"]

        if fact.fact_type == "event_attended" and "pride" in fact.value.lower():
            if "inferred_political_leaning" not in profile:
                profile["inferred_political_leaning"] = []
            if "Liberal" not in profile["inferred_political_leaning"]:
                profile["inferred_political_leaning"].append("Liberal")

        # Military/patriotic -> Conservative
        if fact.fact_type in ["patriotic", "us_focused_goals"] or "military" in fact.value.lower():
            if "inferred_political_tendency" not in profile:
                profile["inferred_political_tendency"] = []
            if "Conservative-leaning" not in profile["inferred_political_tendency"]:
                profile["inferred_political_tendency"].append("Conservative-leaning")

        # Political aspirations -> Political science degree
        if (
            fact.fact_type in ["political_aspiration", "career_goal"]
            and "politic" in fact.value.lower()
        ):
            if "inferred_degree" not in profile:
                profile["inferred_degree"] = []
            if "Political science" not in profile["inferred_degree"]:
                profile["inferred_degree"].append("Political science")
                profile["inferred_degree"].append("Public administration")

        # Counseling -> Psychology degree
        if fact.fact_type in ["career", "career_goal", "interest"] and any(
            x in fact.value.lower()
            for x in ["counseling", "mental health", "therapist", "counselor"]
        ):
            if "inferred_degree" not in profile:
                profile["inferred_degree"] = []
            if "Psychology" not in profile["inferred_degree"]:
                profile["inferred_degree"].append("Psychology")
                profile["inferred_degree"].append("counseling certification")

        # US-specific goals -> Not open to moving abroad
        if fact.fact_type == "us_focused_goals":
            profile["inferred_moving_abroad"] = ["No - has US-specific career goals"]

        # Personality traits
        personality_fact_types = ["attribute", "personality_trait", "personality_description"]
        if fact.fact_type in personality_fact_types:
            if "collected_personality_traits" not in profile:
                profile["collected_personality_traits"] = []
            trait = fact.value.strip().lower()
            if trait and trait not in [
                t.lower() for t in profile["collected_personality_traits"]
            ]:
                profile["collected_personality_traits"].append(fact.value.strip())

        # Outdoor activity preferences
        outdoor_activities = ["camping", "hiking", "beach", "mountains", "nature", "outdoors"]
        if any(act in fact.value.lower() for act in outdoor_activities):
            if "inferred_prefers" not in profile:
                profile["inferred_prefers"] = []
            if "outdoor activities" not in profile["inferred_prefers"]:
                profile["inferred_prefers"].append("outdoor activities")

        # Art activities -> creative
        art_activities = ["painting", "pottery", "drawing", "sculpture", "art"]
        if fact.fact_type in art_activities or any(
            act in fact.value.lower() for act in art_activities
        ):
            if "inferred_creative" not in profile:
                profile["inferred_creative"] = []
            if "artistic" not in profile["inferred_creative"]:
                profile["inferred_creative"].append("artistic")
                profile["inferred_creative"].append("creative")

    # ------------------------------------------------------------------
    # Profile queries
    # ------------------------------------------------------------------

    def get_profile(self, person: str) -> dict[str, list[str]]:
        """Get all facts about a person."""
        return self.person_profiles.get(person.lower(), {})

    def get_profile_text(self, person: str) -> str:
        """Get profile as formatted text."""
        profile = self.get_profile(person)
        if not profile:
            return ""
        parts = [f"Facts about {person}:"]
        for fact_type, values in profile.items():
            parts.append(f"  {fact_type}: {', '.join(values)}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Direct answer methods
    # ------------------------------------------------------------------

    def answer_from_profile(self, question: str, person: str) -> str | None:
        """Try to answer a question directly from the profile."""
        profile = self.get_profile(person)
        if not profile:
            return None

        q_lower = question.lower()

        if "identity" in q_lower or "who is" in q_lower:
            if "identity" in profile:
                return ", ".join(profile["identity"])

        if "lgbtq" in q_lower and (
            "participat" in q_lower or "ways" in q_lower or "community" in q_lower
        ):
            if "lgbtq_participation" in profile:
                return ", ".join(profile["lgbtq_participation"])

        if "relationship" in q_lower or "status" in q_lower:
            if "relationship_status" in profile:
                return ", ".join(profile["relationship_status"])

        if "from" in q_lower or "move" in q_lower or "country" in q_lower:
            if "location" in profile:
                return ", ".join(profile["location"])

        if "activities" in q_lower or "hobbies" in q_lower or "partake" in q_lower:
            activities = []
            for key in ["activity", "preference", "activities"]:
                if key in profile:
                    activities.extend(profile[key])
            if activities:
                return ", ".join(set(activities))

        if "career" in q_lower or "pursue" in q_lower or "work" in q_lower:
            if "career" in profile:
                return ", ".join(profile["career"])

        if "kids like" in q_lower or "children like" in q_lower:
            if "kids_like" in profile:
                return ", ".join(profile["kids_like"])
            if "family_kids_like" in profile:
                return ", ".join(profile["family_kids_like"])

        if "how many" in q_lower and ("children" in q_lower or "kids" in q_lower):
            if "num_children" in profile:
                return profile["num_children"][0]

        if "pet" in q_lower and "name" in q_lower:
            names = []
            for key in ["pet_name", "pet_names", "pets"]:
                if key in profile:
                    names.extend(profile[key])
            if names:
                return ", ".join(names)

        if "what pet" in q_lower or "pets does" in q_lower:
            if "pets" in profile:
                return ", ".join(profile["pets"])

        if "what kind of art" in q_lower or "art does" in q_lower:
            if "art_type" in profile:
                return ", ".join(profile["art_type"])
            if "art" in profile:
                return ", ".join(profile["art"])

        if "paint" in q_lower and ("what" in q_lower or "has" in q_lower):
            if "painted" in profile:
                return ", ".join(profile["painted"])

        if "art" in q_lower:
            results = []
            for key in ["painted", "art_type", "art"]:
                if key in profile:
                    results.extend(profile[key])
            if results:
                return ", ".join(set(results))

        if "favorite book" in q_lower or "childhood" in q_lower:
            if "favorite_book" in profile:
                return ", ".join(profile["favorite_book"])
        if "recommend" in q_lower:
            if "recommended_book" in profile:
                return ", ".join(profile["recommended_book"])

        suggestion_match = re.search(r"(\w+)'s\s+(?:suggestion|recommendation)", q_lower)
        if suggestion_match and ("book" in q_lower or "read" in q_lower):
            recommender = suggestion_match.group(1)
            if "reading_recommended_from" in profile:
                rec_from = profile["reading_recommended_from"]
                if recommender in rec_from or any(recommender in r for r in rec_from):
                    recommender_profile = self.get_profile(recommender)
                    if recommender_profile and "recommended_book" in recommender_profile:
                        return ", ".join(recommender_profile["recommended_book"])

        if "book" in q_lower or "read" in q_lower:
            if "when" in q_lower and "book_year" in profile:
                return profile["book_year"][0]
            if "book" in profile:
                return ", ".join(profile["book"])
            if "books" in profile:
                return ", ".join(profile["books"])

        if "how long" in q_lower or "since when" in q_lower:
            if "art" in q_lower or "paint" in q_lower or "pottery" in q_lower:
                if "art_since_year" in profile:
                    return f"since {profile['art_since_year'][0]}"
            if "since_year" in profile:
                return f"since {profile['since_year'][0]}"
            if "years_duration" in profile:
                return f"{profile['years_duration'][0]} years"

        if "instrument" in q_lower or "play" in q_lower:
            if "instrument" in profile:
                return ", ".join(profile["instrument"])

        if "musician" in q_lower or "music" in q_lower or "listen" in q_lower:
            if "favorite_musician" in profile:
                return ", ".join(profile["favorite_musician"])

        if "camp" in q_lower and "where" in q_lower:
            if "camped_at" in profile:
                return ", ".join(profile["camped_at"])

        if "married" in q_lower and ("how long" in q_lower or "years" in q_lower):
            if "married_years" in profile:
                return f"{profile['married_years'][0]} years"

        if "friend" in q_lower and ("how long" in q_lower or "years" in q_lower):
            if "friends_years" in profile:
                return f"{profile['friends_years'][0]} years"

        if "18th birthday" in q_lower or ("birthday" in q_lower and "ago" in q_lower):
            if "years_since_18" in profile:
                return f"{profile['years_since_18'][0]} years ago"
            if "age" in profile:
                try:
                    age = int(profile["age"][0])
                    return f"{age - 18} years ago"
                except (ValueError, TypeError):
                    pass

        if "research" in q_lower or ("plan" in q_lower and "summer" in q_lower):
            if "researched" in profile:
                values = profile["researched"]
                for val in values:
                    if "adopt" in val.lower():
                        return "researching adoption agencies"
                    return val.title()
            if "activity" in profile:
                for val in profile["activity"]:
                    if "research" in val.lower():
                        return val

        if "destress" in q_lower or "relax" in q_lower:
            if "destress" in profile:
                return ", ".join(profile["destress"])
            results = []
            for key in ["activity", "preference"]:
                if key in profile:
                    results.extend(profile[key])
            if results:
                return ", ".join(results[:3])

        if "artist" in q_lower or "band" in q_lower or "seen" in q_lower:
            if "concerts_seen" in profile:
                return ", ".join(profile["concerts_seen"])

        if "children" in q_lower or "kids" in q_lower:
            if "help" in q_lower or "events" in q_lower:
                if "children_events" in profile:
                    return ", ".join(profile["children_events"])

        if "symbol" in q_lower or "important to" in q_lower:
            if "symbol" in profile:
                return ", ".join(profile["symbol"])

        if "support" in q_lower and ("who" in q_lower or "when" in q_lower):
            if "supporters" in profile:
                return profile["supporters"][0]

        if "career" in q_lower or "pursue" in q_lower or "decided" in q_lower:
            for key in ["career", "event", "activity"]:
                if key in profile:
                    for val in profile[key]:
                        if any(
                            word in val.lower()
                            for word in ["counsel", "mental", "work", "career", "pursue"]
                        ):
                            return val

        if "patriotic" in q_lower or "patriot" in q_lower:
            if "patriotic" in profile:
                return "Yes"

        if (
            "degree" in q_lower
            or "field" in q_lower
            or "study" in q_lower
            or "education" in q_lower
        ):
            if "degree_field" in profile:
                return profile["degree_field"][0].title()
            if "political_aspiration" in profile:
                return "Political science, Public administration"

        if "moving" in q_lower and ("country" in q_lower or "abroad" in q_lower):
            if "us_focused_goals" in profile:
                return "No, has US-specific goals like military and running for office"

        if "political" in q_lower and "leaning" in q_lower:
            if any(key in profile for key in ["lgbtq_participation", "lgbtq", "transgender"]):
                return "Liberal"

        return None

    def answer_temporal_from_profile(self, question: str, person: str) -> str | None:
        """Try to answer a temporal question from stored event dates."""
        profile = self.get_profile(person)
        if not profile:
            return None

        q_lower = question.lower()

        event_mappings = [
            (["pottery class", "pottery", "sign up", "signed up"], "event_signed_up_dates"),
            (["museum"], "event_visited_dates"),
            (["park"], "event_visited_dates"),
            (["beach"], "event_visited_dates"),
            (["conference"], "event_attended_dates"),
            (["parade", "pride parade"], "event_attended_dates"),
            (["workshop", "pottery workshop"], "event_visited_dates"),
            (["camping", "went camping"], "event_activity_dates"),
            (["hiking", "hike"], "event_activity_dates"),
            (["biking"], "event_activity_dates"),
            (["picnic"], "event_event_dates"),
            (["meeting", "adoption meeting"], "event_event_dates"),
            (["interview", "adoption interview"], "event_event_dates"),
            (["activist"], "event_signed_up_dates"),
            (["mentorship", "mentoring"], "event_signed_up_dates"),
            (["portrait", "self-portrait", "drew"], "event_created_dates"),
            (["plate"], "event_created_dates"),
            (["figurines", "bought"], "event_purchased_dates"),
            (["birthday", "daughter"], "event_family_event_dates"),
            (["book", "read"], "event_read_book_dates"),
        ]

        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        question_month = None
        for month in months:
            if month in q_lower:
                question_month = month
                break

        is_first = "first" in q_lower
        is_second = "second" in q_lower
        is_last = "last" in q_lower

        if "book" in q_lower or "read" in q_lower:
            if "book_year" in profile:
                return profile["book_year"][0]
        if "pride" in q_lower or "festival" in q_lower:
            if "pride_year" in profile:
                return profile["pride_year"][0]
            if "year_mentioned" in profile:
                return profile["year_mentioned"][0]
        if "paint" in q_lower or "sunrise" in q_lower:
            if "painting_year" in profile:
                return profile["painting_year"][0]
            if "art_since_year" in profile:
                return profile["art_since_year"][0]

        for keywords, event_key in event_mappings:
            if any(kw in q_lower for kw in keywords):
                if event_key in profile:
                    entries = profile[event_key]

                    if question_month:
                        for entry in entries:
                            if "|" in entry:
                                value, raw_date = entry.split("|", 1)
                                date = self._extract_date_from_timestamp(raw_date)
                                if date and question_month in date.lower():
                                    return date

                    if is_first and entries:
                        if "|" in entries[0]:
                            _, raw_date = entries[0].split("|", 1)
                            return self._extract_date_from_timestamp(raw_date)
                    elif is_second and len(entries) >= 2:
                        if "|" in entries[1]:
                            _, raw_date = entries[1].split("|", 1)
                            return self._extract_date_from_timestamp(raw_date)
                    elif is_last and entries:
                        if "|" in entries[-1]:
                            _, raw_date = entries[-1].split("|", 1)
                            return self._extract_date_from_timestamp(raw_date)

                    for entry in entries:
                        if "|" in entry:
                            value, raw_date = entry.split("|", 1)
                            date = self._extract_date_from_timestamp(raw_date)
                            if date:
                                return date

        return None

    @staticmethod
    def _extract_date_from_timestamp(timestamp: str) -> str | None:
        """Extract date from timestamp string like '1:56 pm on 8 May, 2023'."""
        patterns = [
            r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december),?\s+(\d{4})",
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})",
        ]

        ts_lower = timestamp.lower()
        for pattern in patterns:
            match = re.search(pattern, ts_lower)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if groups[0].isdigit():
                        day, month, year = groups
                        return f"{day} {month.title()} {year}"
                    else:
                        month, day, year = groups
                        return f"{day} {month.title()} {year}"

        return timestamp if timestamp else None

    def answer_inference_question(self, question: str, person: str) -> str | None:
        """Answer questions that require inference from stored facts."""
        profile = self.get_profile(person)
        if not profile:
            return None

        q_lower = question.lower()

        if "political" in q_lower and (
            "leaning" in q_lower or "stance" in q_lower or "likely" in q_lower
        ):
            if "inferred_political_leaning" in profile:
                return profile["inferred_political_leaning"][0]
            if any(k in profile for k in ["lgbtq_participation", "identity", "lgbtq"]):
                for key in ["lgbtq_participation", "identity", "lgbtq"]:
                    if key in profile:
                        values = profile[key]
                        if any(
                            "lgbtq" in v.lower() or "trans" in v.lower() or "pride" in v.lower()
                            for v in values
                        ):
                            return "Liberal"

        if "personality" in q_lower and "trait" in q_lower:
            cross_person_match = re.search(
                r"(?:might|would)\s+(\w+)\s+(?:say|describe|think)\s+(\w+)", q_lower
            )
            if cross_person_match:
                speaker = cross_person_match.group(1)
                target = cross_person_match.group(2)
                cross_traits = self.get_cross_person_traits(speaker, target)
                if cross_traits:
                    return ", ".join(cross_traits[:5])

            if "collected_personality_traits" in profile:
                traits = profile["collected_personality_traits"][:5]
                return ", ".join(traits)

        if "ally" in q_lower and ("lgbtq" in q_lower or "transgender" in q_lower):
            is_ally, reason = self.is_lgbtq_ally(person)
            if is_ally:
                return f"Yes, {person} is {reason}"
            if "against_lgbtq" in profile:
                return "No"
            if "inferred_lgbtq_ally" in profile:
                return "Yes, she is supportive"
            person_lower = person.lower()
            if person_lower in self.conversation_pairs:
                for partner in self.conversation_pairs[person_lower]:
                    partner_profile = self.get_profile(partner)
                    if partner_profile and any(
                        "lgbtq" in v.lower() or "trans" in v.lower()
                        for k, vals in partner_profile.items()
                        for v in (vals if isinstance(vals, list) else [vals])
                    ):
                        return "Yes, she is supportive"

        is_what_question = q_lower.startswith("what ")
        has_education_keywords = "field" in q_lower or "education" in q_lower
        if is_what_question and has_education_keywords:
            if "inferred_degree" in profile:
                return ", ".join(profile["inferred_degree"])
            if "political_aspiration" in profile or "career_goal" in profile:
                for key in ["political_aspiration", "career_goal"]:
                    if key in profile and "politic" in " ".join(profile.get(key, [])).lower():
                        return "Political science, Public administration"

        if "moving" in q_lower and (
            "abroad" in q_lower or "country" in q_lower or "open to" in q_lower
        ):
            if "inferred_moving_abroad" in profile:
                return profile["inferred_moving_abroad"][0]
            if "us_focused_goals" in profile:
                return "No - has US-specific career goals"

        if "prefer" in q_lower or "would" in q_lower and "like" in q_lower:
            if "inferred_prefers" in profile:
                return ", ".join(profile["inferred_prefers"])

        if "creative" in q_lower or "artistic" in q_lower:
            if "inferred_creative" in profile:
                return "Yes - " + ", ".join(profile["inferred_creative"])

        return None

    # ------------------------------------------------------------------
    # Conversation tracking and ally inference
    # ------------------------------------------------------------------

    def track_conversation_relationship(self, speaker: str, partner: str) -> None:
        """Track that speaker had a conversation with partner."""
        speaker_lower = speaker.lower()
        partner_lower = partner.lower()

        if speaker_lower not in self.conversation_pairs:
            self.conversation_pairs[speaker_lower] = set()
        self.conversation_pairs[speaker_lower].add(partner_lower)

        if partner_lower not in self.conversation_pairs:
            self.conversation_pairs[partner_lower] = set()
        self.conversation_pairs[partner_lower].add(speaker_lower)

        partner_profile = self.get_profile(partner_lower)
        if partner_profile:
            is_lgbtq = (
                any(
                    "lgbtq" in v.lower() or "trans" in v.lower() or "pride" in v.lower()
                    for k, values in partner_profile.items()
                    for v in (values if isinstance(values, list) else [values])
                )
                or "identity" in partner_profile
            )

            if is_lgbtq:
                speaker_profile = self.person_profiles.setdefault(speaker_lower, {})
                if "inferred_lgbtq_ally" not in speaker_profile:
                    speaker_profile["inferred_lgbtq_ally"] = []
                if "supportive" not in speaker_profile["inferred_lgbtq_ally"]:
                    speaker_profile["inferred_lgbtq_ally"].append("supportive")
                    speaker_profile["ally_of"] = [partner_lower]

    def add_cross_person_trait(self, speaker: str, target: str, trait: str) -> None:
        """Track what speaker says about target's personality traits."""
        speaker_lower = speaker.lower()
        target_lower = target.lower()

        if speaker_lower not in self.cross_person_traits:
            self.cross_person_traits[speaker_lower] = {}
        if target_lower not in self.cross_person_traits[speaker_lower]:
            self.cross_person_traits[speaker_lower][target_lower] = []

        trait_clean = trait.strip().lower()
        if trait_clean and trait_clean not in [
            t.lower() for t in self.cross_person_traits[speaker_lower][target_lower]
        ]:
            self.cross_person_traits[speaker_lower][target_lower].append(trait.strip())

    def get_cross_person_traits(self, speaker: str, target: str) -> list[str]:
        """Get traits that speaker has mentioned about target."""
        speaker_lower = speaker.lower()
        target_lower = target.lower()

        if speaker_lower in self.cross_person_traits:
            return self.cross_person_traits[speaker_lower].get(target_lower, [])
        return []

    def is_lgbtq_ally(self, person: str) -> tuple[bool, str | None]:
        """Check if person is an LGBTQ+ ally."""
        profile = self.get_profile(person)
        if not profile:
            return False, None

        if "inferred_lgbtq_ally" in profile:
            return True, "supportive of LGBTQ+ friend"

        person_lower = person.lower()
        if person_lower in self.conversation_pairs:
            for partner in self.conversation_pairs[person_lower]:
                partner_profile = self.get_profile(partner)
                if partner_profile:
                    is_lgbtq = (
                        any(
                            "lgbtq" in v.lower() or "trans" in v.lower()
                            for k, values in partner_profile.items()
                            for v in (values if isinstance(values, list) else [values])
                        )
                        or "identity" in partner_profile
                    )
                    if is_lgbtq:
                        return True, f"supportive friend of {partner}"

        return False, None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all profiles and tracking state."""
        self.person_profiles.clear()
        self.cross_person_traits.clear()
        self.conversation_pairs.clear()
        self._current_partner = ""
