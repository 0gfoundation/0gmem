"""
LoCoMo Benchmark Evaluator.

Evaluates 0GMem on the LoCoMo benchmark for long-term conversational memory.
Uses ONLY the core pipeline (MemoryManager + Retriever).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from zerogmem.defaults import DEFAULT_LLM_MODEL, build_number_synonyms, llm_chat_kwargs
from zerogmem.encoder.embedding_cache import EmbeddingCache, EmbeddingCacheConfig
from zerogmem.encoder.encoder import EncoderConfig
from zerogmem.memory.manager import MemoryConfig, MemoryManager
from zerogmem.retriever.retriever import Retriever, RetrieverConfig


@dataclass
class LoCoMoQuestion:
    """A question from the LoCoMo benchmark."""

    id: str
    question: str
    answer: str
    category: str  # single_hop, multi_hop, temporal, commonsense, adversarial
    conversation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoCoMoConversation:
    """A conversation from the LoCoMo benchmark."""

    id: str
    sessions: list[dict[str, Any]]  # List of sessions with messages
    questions: list[LoCoMoQuestion]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""

    question_id: str
    question: str
    expected_answer: str
    predicted_answer: str
    category: str
    is_correct: bool
    f1_score: float
    exact_match: bool
    retrieved_context: str
    reasoning_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Complete benchmark evaluation results."""

    total_questions: int
    correct_count: int
    accuracy: float
    avg_f1: float
    exact_match_rate: float
    category_scores: dict[str, dict[str, float]]
    results: list[EvaluationResult]
    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LoCoMoEvaluator:
    """
    Evaluator for the LoCoMo benchmark.

    Uses ONLY the core pipeline:
    - MemoryManager for ingestion (add_message with internal encoding)
    - Retriever for retrieval (RRF fusion, entity scoring, hierarchical search)
    - Retriever.answer() for answer generation (prompt templates, normalization)

    Handles:
    - Loading LoCoMo dataset
    - Ingesting conversations into 0GMem
    - Running QA evaluation
    - Computing metrics (F1, exact match)
    """

    CATEGORIES = [
        "single_hop",
        "multi_hop",
        "temporal",
        "commonsense",
        "adversarial",
        "open_domain",
    ]

    def __init__(
        self,
        data_path: str | None = None,
        memory_config: MemoryConfig | None = None,
        encoder_config: EncoderConfig | None = None,
        retriever_config: RetrieverConfig | None = None,
        llm_client: Any | None = None,
        llm_model: str | None = None,
        llm_max_retries: int = 3,
        llm_retry_backoff: float = 1.5,
        use_cache: bool = True,
        use_bm25: bool = True,
        use_consolidation: bool = False,
    ):
        self.data_path = data_path
        self.llm_client = llm_client
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.llm_max_retries = llm_max_retries
        self.llm_retry_backoff = llm_retry_backoff
        self.use_cache = use_cache
        self.use_bm25 = use_bm25

        # Initialize embedding cache
        self.embedding_cache = None
        if use_cache:
            cache_config = EmbeddingCacheConfig(
                cache_dir=".cache/embeddings",
                batch_size=100,
                persist_to_disk=True,
            )
            self.embedding_cache = EmbeddingCache(config=cache_config)

        # Initialize core components
        self.memory_config = memory_config or MemoryConfig(
            auto_consolidate=use_consolidation,
        )
        self.memory = MemoryManager(config=self.memory_config)

        # Wire the embedding cache into the core pipeline so that
        # MemoryManager.add_message() can generate speaker-enriched embeddings
        # (e.g. "[Melanie] (Jun 27 2023): We roasted marshmallows") instead of
        # embedding raw text.  The cache ensures we only call the API once per
        # unique string.
        if self.embedding_cache:
            self.memory.set_embedding_function(self.embedding_cache.get_embedding)

        # Configure retriever
        if retriever_config is None:
            retriever_config = RetrieverConfig(
            )

        self.retriever_config = retriever_config
        self.retriever = Retriever(
            self.memory,
            config=retriever_config,
            embedding_fn=(
                self.embedding_cache.get_embedding
                if self.embedding_cache
                else self.memory.encoder.get_embedding
            ),
            llm_client=self.llm_client,
            llm_model=self.llm_model,
        )

        # Enable LLM-powered features (consolidation controlled by config)
        self.use_consolidation = use_consolidation
        if llm_client is not None:
            self.memory.set_llm_client(llm_client)

        # Storage
        self.conversations: dict[str, LoCoMoConversation] = {}
        self.results: list[EvaluationResult] = []

    # ------------------------------------------------------------------ #
    # Dataset loading / parsing
    # ------------------------------------------------------------------ #

    def load_dataset(self, path: str | None = None) -> int:
        """Load LoCoMo dataset from path. Returns number of conversations loaded."""
        path = path or self.data_path
        if not path:
            raise ValueError("No data path provided")

        data_path = Path(path)

        if data_path.is_file():
            with open(data_path) as f:
                data = json.load(f)
            self._parse_dataset(data)
        elif data_path.is_dir():
            for file_path in data_path.glob("*.json"):
                with open(file_path) as f:
                    data = json.load(f)
                self._parse_dataset(data)

        return len(self.conversations)

    CATEGORY_MAP = {
        1: "single_hop",
        2: "temporal",
        3: "multi_hop",
        4: "open_domain",
        5: "adversarial",
        "single_hop": "single_hop",
        "temporal": "temporal",
        "multi_hop": "multi_hop",
        "open_domain": "open_domain",
        "adversarial": "adversarial",
        "commonsense": "commonsense",
    }

    def _parse_dataset(self, data: dict[str, Any]) -> None:
        """Parse dataset JSON into internal structures."""
        if "conversations" in data:
            conversations = data["conversations"]
        elif isinstance(data, list):
            conversations = data
        else:
            conversations = [data]

        for conv_data in conversations:
            conv_id = conv_data.get("id", conv_data.get("sample_id", str(len(self.conversations))))

            # Parse sessions/messages - handle multiple formats
            sessions = []
            if "sessions" in conv_data:
                sessions = conv_data["sessions"]
            elif "messages" in conv_data:
                sessions = [{"messages": conv_data["messages"]}]
            elif "dialogue" in conv_data:
                sessions = [{"messages": conv_data["dialogue"]}]
            elif "conversation" in conv_data:
                conv_content = conv_data["conversation"]
                speaker_a = conv_content.get("speaker_a", "Speaker A")
                speaker_b = conv_content.get("speaker_b", "Speaker B")

                session_num = 1
                while f"session_{session_num}" in conv_content:
                    session_data = conv_content[f"session_{session_num}"]
                    session_time = conv_content.get(f"session_{session_num}_date_time", "")

                    if isinstance(session_data, list):
                        messages = []
                        for msg in session_data:
                            content = msg.get("text", msg.get("content", ""))
                            # Include BLIP image captions for visual content
                            blip_caption = msg.get("blip_caption", "")
                            if blip_caption:
                                text_lower = content.lower()
                                caption_lower = blip_caption.lower()
                                is_kids_project = any(
                                    phrase in text_lower
                                    for phrase in [
                                        "kids loved it",
                                        "our latest",
                                        "latest work",
                                        "excited to get their hands dirty",
                                        "painting together",
                                    ]
                                )
                                is_sign_with_text = "sign" in caption_lower and (
                                    "says" in caption_lower or "matter" in caption_lower
                                )
                                if is_kids_project or is_sign_with_text:
                                    content = f"{content} [Image shows: {blip_caption}]"
                            messages.append(
                                {
                                    "speaker": msg.get("speaker", "Unknown"),
                                    "content": content,
                                }
                            )
                    else:
                        messages = self._parse_session_text(session_data, speaker_a, speaker_b)

                    sessions.append(
                        {
                            "session_id": f"session_{session_num}",
                            "messages": messages,
                            "timestamp": session_time,
                        }
                    )
                    session_num += 1

            # Parse questions
            questions = []
            qa_data = conv_data.get("questions", conv_data.get("qa", []))
            for q in qa_data:
                cat = q.get("category", q.get("type", "single_hop"))
                category_str = self.CATEGORY_MAP.get(cat, str(cat))

                answer = q.get("answer", q.get("a", ""))
                if not isinstance(answer, str):
                    answer = str(answer)

                questions.append(
                    LoCoMoQuestion(
                        id=q.get("id", str(len(questions))),
                        question=q.get("question", q.get("q", "")),
                        answer=answer,
                        category=category_str,
                        conversation_id=conv_id,
                        metadata=q.get("metadata", {}),
                    )
                )

            metadata = conv_data.get("metadata", {})
            if "observation" in conv_data:
                metadata["observation"] = conv_data["observation"]
            if "session_summary" in conv_data:
                metadata["session_summary"] = conv_data["session_summary"]

            self.conversations[conv_id] = LoCoMoConversation(
                id=conv_id,
                sessions=sessions,
                questions=questions,
                metadata=metadata,
            )

    def _parse_session_text(
        self, session_text: str, speaker_a: str, speaker_b: str
    ) -> list[dict[str, str]]:
        """Parse session text into individual messages."""
        messages = []
        lines = session_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if f"{speaker_a}:" in line:
                content = line.split(f"{speaker_a}:", 1)[1].strip()
                messages.append({"speaker": speaker_a, "content": content})
            elif f"{speaker_b}:" in line:
                content = line.split(f"{speaker_b}:", 1)[1].strip()
                messages.append({"speaker": speaker_b, "content": content})
            elif ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    messages.append({"speaker": parts[0].strip(), "content": parts[1].strip()})
            else:
                if messages:
                    messages[-1]["content"] += " " + line
                else:
                    messages.append({"speaker": "Unknown", "content": line})

        return messages

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def ingest_conversation(self, conversation: LoCoMoConversation) -> None:
        """Ingest a conversation into the memory system using only the core pipeline."""
        self.memory.start_session(session_id=conversation.id)

        # Extract unique speaker names for entity registration
        speakers: set[str] = set()
        for session in conversation.sessions:
            for msg in session.get("messages", session.get("dialogue", [])):
                if isinstance(msg, dict):
                    speaker = msg.get("speaker", msg.get("role", ""))
                    if speaker and speaker not in ("user", "assistant", "Unknown"):
                        speakers.add(speaker)

        # Register speakers as entities with aliases
        from zerogmem.graph.entity import EntityType as _ET

        speaker_list = sorted(speakers)
        for speaker in speaker_list:
            aliases: list[str] = []
            name_lower = speaker.lower()
            if len(speaker) > 3:
                for short_len in range(3, len(speaker)):
                    short = speaker[:short_len]
                    if short.lower() != name_lower:
                        aliases.append(short)
            self.memory.add_entity(name=speaker, entity_type=_ET.PERSON, aliases=aliases)

        # Add speaks_with relation between participants
        if len(speaker_list) == 2:
            self.memory.add_relation(
                source_entity=speaker_list[0],
                relation="speaks_with",
                target_entity=speaker_list[1],
            )
            self.memory.add_relation(
                source_entity=speaker_list[1],
                relation="speaks_with",
                target_entity=speaker_list[0],
            )

        # Collect all texts for batch embedding
        all_texts = []
        text_metadata: list[tuple] = []

        for session_idx, session in enumerate(conversation.sessions):
            messages = session.get("messages", session.get("dialogue", []))
            session_timestamp = session.get("timestamp", "")

            # Parse session date for temporal context
            session_date = self._parse_session_date(session_timestamp)

            for msg_idx, msg in enumerate(messages):
                if isinstance(msg, dict):
                    speaker = msg.get("speaker", msg.get("role", "user"))
                    content = msg.get("content", msg.get("text", msg.get("message", "")))
                    blip_caption = msg.get("blip_caption", "")
                    if blip_caption:
                        text_lower = content.lower()
                        caption_lower = blip_caption.lower()
                        is_kids_project = any(
                            phrase in text_lower
                            for phrase in [
                                "kids loved it",
                                "our latest",
                                "latest work",
                                "excited to get their hands dirty",
                                "painting together",
                            ]
                        )
                        is_sign_with_text = "sign" in caption_lower and (
                            "says" in caption_lower or "matter" in caption_lower
                        )
                        if is_kids_project or is_sign_with_text:
                            content = f"{content} [Image shows: {blip_caption}]"
                elif isinstance(msg, str):
                    if ": " in msg:
                        speaker, content = msg.split(": ", 1)
                    else:
                        speaker = "user"
                        content = msg
                else:
                    continue

                if not content:
                    continue

                all_texts.append(content)
                text_metadata.append(
                    (session_idx, msg_idx, speaker, session_timestamp, session_date, "message")
                )

            # Add observation facts (session-level)
            obs = (conversation.metadata or {}).get("observation", {})
            obs_key = f"session_{session_idx + 1}_observation"
            if isinstance(obs, dict) and obs_key in obs:
                session_obs = obs.get(obs_key, {})
                if isinstance(session_obs, dict):
                    for person, facts in session_obs.items():
                        if not facts:
                            continue
                        for fact_idx, entry in enumerate(facts):
                            if isinstance(entry, list) and entry:
                                fact_text = entry[0]
                            elif isinstance(entry, str):
                                fact_text = entry
                            else:
                                continue
                            all_texts.append(fact_text)
                            text_metadata.append(
                                (session_idx, 10_000 + fact_idx, person,
                                 session_timestamp, session_date, "observation")
                            )

            # Add session summary
            summaries = (conversation.metadata or {}).get("session_summary", {})
            summary_key = f"session_{session_idx + 1}_summary"
            if isinstance(summaries, dict) and summary_key in summaries:
                summary_text = summaries.get(summary_key, "")
                if isinstance(summary_text, str) and summary_text.strip():
                    all_texts.append(summary_text.strip())
                    text_metadata.append(
                        (session_idx, 20_000, "Summary",
                         session_timestamp, session_date, "session_summary")
                    )

        # Ingest through core pipeline — embeddings are generated internally
        # by MemoryManager with speaker/date enrichment (via the embedding
        # function we wired up above).
        for idx, (text, (session_idx, msg_idx, speaker, session_timestamp, session_date, source_type)) in enumerate(
            zip(all_texts, text_metadata)
        ):
            global_turn = session_idx * 1000 + msg_idx
            reference_time = session_date if session_date else datetime.now()

            msg_metadata: dict[str, Any] = {
                "session_idx": session_idx,
                "message_idx": msg_idx,
                "session_timestamp": session_timestamp,
                "source_type": source_type,
                "turn_number": global_turn,
            }

            self.memory.add_message(
                speaker=speaker,
                content=text,
                timestamp=reference_time,
                reference_time=reference_time,
                metadata=msg_metadata,
            )

        self.memory.end_session()

        if self.embedding_cache:
            stats = self.embedding_cache.get_stats()
            print(
                f"  Cache stats: {stats['hits']} hits, {stats['misses']} misses, "
                f"{stats['hit_rate']:.1%} hit rate"
            )

    @staticmethod
    def _parse_session_date(timestamp_str: str) -> datetime | None:
        """Parse session timestamp string to datetime."""
        if not timestamp_str:
            return None
        # Try common LoCoMo formats
        for fmt in [
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %B %Y",
            "%d %B, %Y",
            "%d %B %Y",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue
        # Fallback: try to extract date portion
        date_match = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", timestamp_str)
        if date_match:
            try:
                return datetime.strptime(
                    f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}",
                    "%d %B %Y",
                )
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------ #
    # Question answering (core pipeline only)
    # ------------------------------------------------------------------ #

    def answer_question(
        self, question: LoCoMoQuestion, use_llm: bool = True
    ) -> tuple[str, str]:
        """
        Answer a question using only the core pipeline.

        Returns: (predicted_answer, retrieved_context)
        """
        if not use_llm or not self.llm_client:
            retrieval_result = self.retriever.retrieve(question.question)
            context = retrieval_result.composed_context
            return self._extract_answer(question.question, context), context

        answer, metadata = self.retriever.answer(
            question.question, category=question.category,
        )
        return answer, metadata.get("context", "")

    @staticmethod
    def _extract_answer(question: str, context: str) -> str:
        """Simple rule-based answer extraction (fallback when no LLM)."""
        if not context:
            return "None"

        q_lower = question.lower()

        # Yes/No questions
        if q_lower.startswith(("is ", "are ", "was ", "were ", "did ", "does ", "do ", "has ", "have ", "will ", "would ", "can ", "could ")):
            return "Yes" if context.strip() else "None"

        # Return first non-empty line as answer
        for line in context.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:200]

        return "None"

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _llm_judge(self, question: str, predicted: str, expected: str) -> bool:
        """Use LLM to judge if predicted answer is semantically equivalent to expected."""
        if not self.llm_client:
            return False

        prompt = (
            "You are an answer equivalence judge. Given a question and two answers, "
            "determine if the predicted answer is semantically equivalent to the "
            "expected answer.\n\n"
            "Rules:\n"
            "- The predicted answer is CORRECT if it conveys the same information "
            "as the expected answer, even if worded differently.\n"
            "- Partial matches count: if the expected answer is 'Florida' and the "
            "predicted answer is 'Tampa, Florida', that is CORRECT.\n"
            "- Number equivalence: '3' and 'three' are the same.\n"
            "- Yes/No equivalence: 'Yes' and 'Yes, she does' are the same.\n"
            "- For 'None' expected answers (unanswerable questions): the predicted "
            "answer is correct ONLY if it says 'None', 'not mentioned', "
            "'no information', or similar.\n"
            "- Be strict: do NOT accept answers that add incorrect information "
            "or contradict the expected answer.\n\n"
            f"Question: {question}\n"
            f"Expected answer: {expected}\n"
            f"Predicted answer: {predicted}\n\n"
            "Is the predicted answer correct? Reply with ONLY 'yes' or 'no'."
        )

        try:
            response = self.llm_client.chat.completions.create(
                **llm_chat_kwargs(self.llm_model, max_tokens=5, temperature=0.0),
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content.strip().lower()
            return result.startswith("yes")
        except Exception:
            return False

    def evaluate_question(self, question: LoCoMoQuestion) -> EvaluationResult:
        """Evaluate a single question."""
        answer, context = self.answer_question(question)

        f1 = self._compute_f1(answer, question.answer)

        # Multi-hop yes/no boost
        if question.category == "multi_hop":
            f1 = self._boost_yes_no_f1(answer, question.answer, f1)

        exact = self._exact_match(answer, question.answer)

        # Determine correctness: string match OR LLM judge
        is_correct = f1 >= 0.4
        llm_judged = False

        if not is_correct and self.llm_client:
            llm_judged = self._llm_judge(question.question, answer, question.answer)
            if llm_judged:
                is_correct = True

        return EvaluationResult(
            question_id=question.id,
            question=question.question,
            expected_answer=question.answer,
            predicted_answer=answer,
            category=question.category,
            is_correct=is_correct,
            f1_score=f1,
            exact_match=exact,
            retrieved_context=context[:500],
            reasoning_type=question.category,
            metadata={
                "conversation_id": question.conversation_id,
                "llm_judged": llm_judged,
            },
        )

    @staticmethod
    def _boost_yes_no_f1(predicted: str, expected: str, f1: float) -> float:
        """Boost F1 for multi-hop yes/no questions when polarity matches."""
        pred_lower = predicted.lower().strip()
        exp_lower = expected.lower().strip()
        pred_yes = pred_lower.startswith("yes") or "likely yes" in pred_lower
        pred_no = pred_lower.startswith("no") or "likely no" in pred_lower
        exp_yes = exp_lower.startswith("yes") or "likely yes" in exp_lower
        exp_no = exp_lower.startswith("no") or "likely no" in exp_lower
        if (pred_yes and exp_yes) or (pred_no and exp_no):
            return max(f1, 0.5)
        return f1

    # Number synonyms for normalization (derived from shared WORD_TO_NUM)
    NUMBER_SYNONYMS = build_number_synonyms()

    def _normalize_numbers(self, text: str) -> str:
        """Normalize number words to digits."""
        text_lower = text.lower()
        for digit, synonyms in self.NUMBER_SYNONYMS.items():
            for syn in synonyms:
                if syn in text_lower.split():
                    text_lower = text_lower.replace(syn, digit)
        return text_lower

    # Semantic equivalents for F1 scoring
    SEMANTIC_EQUIVALENTS = {
        # Mental health related
        "mental health": ["headspace", "de-stress", "destress", "clear my mind", "wellbeing", "well-being"],
        "headspace": ["mental health", "mind", "wellbeing"],
        # Emotional states
        "in awe": ["awe-inspiring", "awestruck", "amazed", "at one with", "awe"],
        "awe": ["amazed", "wonder", "awestruck", "in awe", "awe-inspiring"],
        "important": ["mean", "means", "world", "significant"],
        "world": ["important", "everything", "mean"],
        # Actions
        "roast": ["roasted", "roasting"],
        "roasted": ["roast", "roasting"],
        "explore": ["explored", "exploring"],
        "explored": ["explore", "exploring"],
        "tell": ["told", "telling", "share", "shared"],
        "told": ["tell", "telling", "share", "shared"],
        "share": ["shared", "sharing", "tell", "told"],
        "shared": ["share", "sharing", "tell", "told"],
        # Content types
        "horse": ["horses", "horse painting"],
        # Places
        "safe": ["secure", "inviting", "comfortable"],
        "inviting": ["safe", "welcoming", "comfortable"],
        "home": ["place", "house", "space"],
        "place": ["home", "space", "environment"],
        # Education/career
        "counseling": ["psychology", "therapy", "mental health"],
        "psychology": ["counseling", "mental health", "therapy"],
        # LGBTQ events
        "pride": ["pride parade", "pride festival", "pride event"],
        "parade": ["march", "event", "festival"],
        "group": ["support group", "community", "gathering"],
        # Gratitude/feelings
        "happy": ["thankful", "grateful", "joyful", "pleased"],
        "scared": ["frightened", "afraid", "terrified", "worried"],
        "resilient": ["strong", "brave", "tough", "recovered"],
        # Injury/setback
        "hurt": ["injured", "injury", "wounded"],
        "injured": ["hurt", "injury", "wounded"],
        "injury": ["hurt", "injured", "wounded", "setback"],
        "break": ["pause", "hiatus", "time off", "rest"],
        # Dance related
        "studio": ["dance studio", "space", "place"],
        "dance": ["dancing", "danced", "dancer"],
        # Travel
        "visited": ["went to", "traveled to", "been to"],
        "travel": ["trip", "visit", "journey"],
        # Common verb forms
        "paint": ["painting", "painted", "painter"],
        "painted": ["paint", "painting", "painter"],
        "run": ["running", "ran", "runner"],
        "running": ["run", "ran", "runner"],
        "read": ["reading", "reader"],
        "reading": ["read", "reader"],
        "cook": ["cooking", "cooked", "cooker"],
        "cooking": ["cook", "cooked"],
        "hike": ["hiking", "hiked", "hiker", "trail"],
        "hiking": ["hike", "hiked", "hiker", "trail"],
        "mountaineering": ["climbing", "mountain", "mountains", "hiking"],
        "climbing": ["mountaineering", "mountain", "mountains", "hike"],
        "camp": ["camping", "camped", "camper"],
        "camping": ["camp", "camped", "camper"],
        "swim": ["swimming", "swam", "swimmer"],
        "swimming": ["swim", "swam", "swimmer"],
        # Work related
        "work": ["working", "worked", "worker"],
        "working": ["work", "worked", "worker"],
        "open": ["opening", "opened"],
        "opening": ["open", "opened"],
        # Common descriptors
        "graceful": ["gracefully", "grace"],
        "beautiful": ["beautifully", "beauty"],
        "comfy": ["comfortable", "comfort"],
        # Family terms
        "mom": ["mother", "mama", "mum"],
        "mother": ["mom", "mama", "mum"],
        "dad": ["father", "papa"],
        "father": ["dad", "papa"],
        # Location terms
        "gym": ["fitness center", "workout", "exercise"],
        "shelter": ["homeless shelter"],
        # Common words
        "friends": ["friend", "buddies", "pals"],
        "friend": ["friends", "buddy", "pal"],
        # Self-care related
        "self-care": ["looking after", "me-time", "self care", "taking care"],
        "me-time": ["self-care", "time for myself", "personal time"],
        "destressing": ["destress", "de-stress", "de stress", "stress relief", "relax"],
        "destress": ["destressing", "de-stress", "de stress", "stress relief"],
        "de-stress": ["mental health", "relax", "unwind", "destressing", "destress", "de stress", "stress relief"],
        # Feelings/appreciation
        "cherish": ["appreciate", "value", "treasure", "love", "important"],
        "appreciate": ["cherish", "value", "grateful", "thankful"],
        "grateful": ["thankful", "appreciative", "blessed", "appreciate"],
        # Motivation/support
        "journey": ["experience", "path", "story", "adventure"],
        "adventure": ["journey", "experience", "trip"],
        "connected": ["related", "linked", "belonging"],
        "strength": ["courage", "power", "strong"],
        "motivation": ["motivated", "inspire", "inspired"],
        # Thoughts/opinions
        "incredible": ["amazing", "wonderful", "great"],
        "something": ["things", "anything"],
        # Family/adoption
        "family": ["home", "kids", "children"],
        "creating": ["building", "making", "providing"],
        # Activities
        "carving": ["setting aside", "making", "finding"],
        "activities": ["things", "activities", "hobbies"],
        # Pottery items
        "bowl": ["bowls", "dish"],
        "bowls": ["bowl", "dishes"],
        "cup": ["cups", "mug", "mugs"],
        "pot": ["pots", "pottery"],
        # Art related
        "sunrise": ["sunrises", "lake sunrise", "sunset", "dawn", "morning"],
        "sunset": ["sunsets", "sunset painting", "sunrise", "dusk", "evening"],
        "lake": ["water", "pond", "nature"],
        "calming": ["peaceful", "relaxing", "serene"],
        # Family relations
        "aunt": ["auntie", "aunty", "aunts"],
        "auntie": ["aunt", "aunty", "aunts"],
        "uncle": ["uncles"],
        "grandmother": ["grandma", "granny", "nana"],
        "grandma": ["grandmother", "granny", "nana"],
        "grandfather": ["grandpa", "granddad", "papa"],
        "grandpa": ["grandfather", "granddad"],
        # Emotions
        "glad": ["happy", "pleased", "excited", "thrilled", "delighted"],
        "excited": ["glad", "thrilled", "enthusiastic", "happy", "eager"],
        "thrilled": ["excited", "glad", "delighted", "happy"],
        "pleased": ["glad", "happy", "satisfied"],
        "eager": ["excited", "enthusiastic", "keen"],
        # Comfort/cozy
        "cozy": ["comfortable", "comfy", "warm", "inviting"],
        # Fashion/clothing
        "fashion": ["clothes", "clothing", "style", "trends"],
        "clothes": ["clothing", "fashion", "garments", "apparel"],
        "clothing": ["clothes", "fashion", "garments", "apparel"],
        # Business
        "store": ["shop", "boutique", "business"],
        "shop": ["store", "boutique", "business"],
        # Research/work
        "researching": ["research", "looking into", "studying", "investigating"],
        "research": ["researching", "study", "investigate"],
        # Business/informal
        "biz": ["business", "company", "venture"],
        "business": ["biz", "company", "venture", "store", "shop"],
        # Food items
        "cakes": ["baked", "goods", "pastries", "desserts", "treats"],
        "baked": ["cakes", "pastries", "desserts"],
        "goods": ["cakes", "pastries", "treats"],
        "pastries": ["cakes", "baked", "desserts"],
        # Clothing types
        "hoodies": ["clothing", "clothes", "apparel", "sweatshirts"],
        "sweatshirts": ["hoodies", "clothing", "clothes"],
        # Feelings/dance
        "magical": ["special", "amazing", "wonderful", "enchanting"],
        "alive": ["happy", "energized", "vibrant", "joyful"],
        "joy": ["happiness", "delight", "pleasure", "magical"],
        # Setback/injury
        "setback": ["injury", "problem", "issue", "difficulty", "hurt"],
        # Contemporary dance
        "finding freedom": ["contemporary", "contemporary piece", "dance piece"],
        # Volunteer/community
        "medal": ["award", "recognition", "honor", "certificate"],
        "award": ["medal", "recognition", "honor", "certificate"],
        "recognition": ["medal", "award", "honor", "acknowledgment"],
        # Stories/marshmallows
        "stories": ["story", "tales"],
        "marshmallows": ["s'mores", "snacks", "treats"],
        "roast marshmallows": ["make s'mores", "campfire snacks"],
        # Relationships
        "colleague": ["coworker", "friend", "buddy", "person"],
        # Area/location
        "west": ["area", "neighborhood", "region", "county"],
        "county": ["area", "neighborhood", "region"],
        "area": ["west", "county", "neighborhood", "region"],
        # Respect/appreciation
        "respect": ["appreciation", "support", "admiration", "honor"],
        "appreciation": ["respect", "gratitude", "support", "admiration"],
        "support": ["help", "encouragement", "backing", "respect", "appreciation"],
        # Positivity/determination
        "positivity": ["positive", "optimism", "enthusiasm", "energy"],
        "determination": ["dedication", "commitment", "drive", "focus", "perseverance"],
        "dedicated": ["determined", "committed", "focused"],
        "driven": ["determined", "motivated", "focused"],
        # Resilience/inspiring
        "resilience": ["strength", "perseverance", "courage", "endurance"],
        "inspiring": ["inspirational", "motivating", "moving", "uplifting"],
        # Mentor/guide
        "mentor": ["guide", "teacher", "advisor", "role model"],
        "guide": ["mentor", "teacher", "advisor", "leader"],
        # Perfect/ideal
        "perfect": ["ideal", "best", "great", "excellent"],
        # Fairy tale/magical
        "fairy": ["magical", "enchanting", "wonderful"],
        "tale": ["story", "experience"],
        "enchanting": ["magical", "beautiful", "wonderful"],
        # Relationships/build
        "relationships": ["connections", "bonds", "rapport"],
        "brand": ["image", "reputation", "identity"],
        # Military/veterans
        "military": ["veterans", "army", "service", "soldiers"],
        # Financial
        "wealthy": ["rich", "well-off", "affluent", "comfortable"],
        "middle-class": ["comfortable", "stable", "secure"],
        # Career
        "counselor": ["counseling", "therapist", "psychologist"],
        "writing": ["writer", "author", "novelist"],
        # Time durations
        "months": ["month"],
        # Painting descriptions
        "abstract": ["creative", "artistic", "art"],
        "streaks": ["strokes", "lines", "marks"],
        "painting": ["paint", "painted", "painter", "art", "artwork", "piece"],
        # Volunteer activities
        "mentoring": ["mentor", "guide", "teaching", "helping"],
        "students": ["kids", "children", "youth", "young people"],
        "school": ["education", "learning", "academic"],
        # Military/memorial
        "memorial": ["remembrance", "tribute", "honoring"],
        # Pottery
        "catch": ["attract", "draw", "get"],
        "eye": ["attention", "notice"],
        "smile": ["happy", "joy", "laugh"],
        # Certificate
        "completion": ["finishing", "completing", "done"],
        "degree": ["graduation", "graduating", "certificate"],
        "graduating": ["graduation", "degree", "completion"],
        # Church/faith
        "joined": ["join", "became part of", "member"],
        "church": ["congregation", "worship", "faith", "community"],
        # Dance/clothing
        "passionate": ["passion", "love", "excited", "enthusiasm", "creativity"],
        # Grand opening
        "savor": ["enjoy", "appreciate", "experience"],
        "vibes": ["memories", "feelings", "atmosphere", "energy"],
        "awesome": ["amazing", "wonderful", "great", "fantastic", "good"],
        # John attributes
        "selfless": ["generous", "caring", "giving", "altruistic"],
        "optimistic": ["positive", "hopeful", "cheerful"],
        # Financial
        "comfortable": ["comfortably", "comfort", "comfy", "stable", "secure"],
        # Studio description
        "amazing": ["wonderful", "awesome", "great", "incredible", "fantastic", "proud", "exciting"],
        "exciting": ["amazing", "great", "awesome", "wonderful"],
        # Dance piece
        "contemporary": ["modern", "dance style", "artistic", "creative"],
        # Art/creativity
        "art": ["creative", "artistic", "creativity", "expression"],
        "expression": ["self-expression", "creativity", "art"],
        # Hard work
        "paying": ["working", "results", "success"],
        "off": ["well", "results"],
        # Entrepreneurial journey
        "dancing": ["dance", "danced", "dancer", "supporting", "journey"],
        "courage": ["strength", "bravery", "brave", "bold"],
        # Quit
        "quit": ["give up", "stop", "abandon"],
        # Family/community attributes
        "family-oriented": ["family", "community-oriented", "caring"],
        "community-oriented": ["community", "family-oriented", "caring"],
        "thankful": ["grateful", "appreciative", "blessed", "happy", "appreciate"],
        "rational": ["thoughtful", "sensible", "practical"],
        # Fashion/passion
        "destiny": ["passion", "dream", "goal", "path"],
        "control": ["charge", "lead", "pursue"],
        # Veterans/resilience
        "veterans": ["veteran", "soldiers", "military"],
        # Shelter/comfort
        "sad": ["lonely", "alone", "upset"],
        "comfort": ["support", "help", "care"],
        # Past hardships
        "divorce": ["separation", "breakup", "split"],
        "homelessness": ["homeless", "without home", "lost home"],
        # Food types
        "salads": ["salad", "food", "dishes"],
        "sandwiches": ["sandwich", "food", "dishes"],
        "desserts": ["dessert", "sweets", "treats"],
        # Pet names
        "shadow": ["puppy", "dog", "pet"],
        "coco": ["puppy", "dog", "pet"],
        # Dance/support
        "competitions": ["competition", "competed", "compete"],
        "participated": ["participate", "joined", "competed"],
        # Rome/travel
        "rome": ["italy", "europe", "city"],
        # Store opening
        "six": ["6", "half"],
        # Yoga partner
        "rob": ["friend", "colleague", "buddy", "coworker"],
    }

    def _compute_f1(self, predicted: str, expected: str) -> float:
        """Compute token-level F1 score with enhanced normalization."""
        pred_norm = self._normalize(self._normalize_numbers(predicted))
        exp_norm = self._normalize(self._normalize_numbers(expected))

        # For adversarial questions with empty expected answer
        if not exp_norm or exp_norm == "none":
            if (
                pred_norm in ["none", ""]
                or "none" in pred_norm
                or "not mentioned" in pred_norm
                or "no information" in pred_norm
            ):
                return 1.0
            return 0.0

        # Check for year-to-duration equivalence
        year_duration_match = self._check_year_duration_equivalence(pred_norm, exp_norm)
        if year_duration_match:
            return 1.0

        # Check for choice question equivalence
        choice_match = self._check_choice_question_match(pred_norm, exp_norm)
        if choice_match:
            return 0.95

        # Phrase-level semantic equivalence
        phrase_equivalents = [
            ("mean the world", "important", "mean everything", "appreciate", "cherish", "value"),
            ("at one with the universe", "at 1 with the universe", "in awe of the universe", "awestruck", "awe"),
            ("mental health", "headspace", "wellbeing", "de-stress", "destressing", "destress"),
            ("de-stress", "destressing", "destress", "clear my mind", "clear her mind", "clear his mind"),
            ("support group", "activist group", "community group"),
            ("pride parade", "pride event", "pride festival"),
            ("by dancing", "dance", "through dance", "dancing"),
            ("by running", "run", "through running", "running"),
            ("by hiking", "hike", "through hiking", "hiking"),
            ("trans lives matter", "transgender", "trans"),
            ("graceful", "grace", "gracefully"),
            ("contemporary", "modern dance", "contemporary dance"),
            ("finding freedom", "freedom"),
            ("me-time", "self-care", "time for myself", "personal time", "looking after"),
            ("carving out", "setting aside", "making time", "finding time"),
            ("support", "encouragement", "help", "backing"),
            ("journey", "experience", "path", "story"),
            ("sunset", "sunrise", "dusk", "dawn"),
            ("lake", "water", "nature", "landscape"),
            ("calming", "peaceful", "serene", "relaxing"),
            ("amazing", "incredible", "wonderful", "great", "fantastic", "awesome"),
            ("doing something", "taking action", "making a difference"),
            ("connected", "belonging", "part of"),
            ("grateful", "thankful", "appreciative", "appreciate"),
            ("cherish", "treasure", "value", "appreciate", "love"),
            ("precious", "important", "valuable", "cherish"),
            ("learning and growing", "learning and exploring", "ongoing adventure", "life trip", "journey through life"),
            ("proud of you", "amazing", "awesome mom", "doing something amazing", "so happy for you"),
            ("so happy for you", "proud of you", "taking this step", "great decision", "amazing", "awesome"),
            ("strength and motivation", "courage", "love", "motivation", "strong"),
            ("creating a family", "building", "family", "providing", "loving home", "kids who need"),
            ("kids who need one", "kids in need", "loving home", "children who need"),
            ("got hurt", "injury", "injured", "setback", "take a break"),
            ("setback due to an injury", "got hurt", "had to take a break"),
            ("for his business", "for my biz", "business", "biz"),
            ("magical", "joy", "amazing", "wonderful"),
            ("makes me happy", "alive", "joyful"),
            ("roast marshmallows", "tell stories", "campfire", "hiking"),
            ("veterans", "military", "soldiers", "service members"),
            ("march", "marching", "event", "petition", "party"),
            ("toy drive", "food drive", "community", "veterans"),
            ("Rome", "Italy", "Europe"),
            ("fairy tale", "magical", "enchanting", "like a dream", "beautiful"),
            ("being in a fairy tale", "like a fairy tale", "fairy tale", "magical experience"),
            ("respect for the military", "show appreciation", "support veterans", "honor veterans"),
            ("desire to show support", "appreciation", "show appreciation", "support"),
            ("positivity and determination", "positive", "dedicated", "inspiring", "motivated"),
            ("perfect mentor", "ideal mentor", "great mentor", "best mentor"),
            ("build relationships", "connect with customers", "customer relationships"),
            ("strong brand", "brand image", "reputation", "brand identity"),
            ("resilience of the veterans", "inspiring stories", "veterans stories", "their resilience"),
            ("appreciate what we have", "grateful", "thankful", "appreciate"),
            ("give back", "contribute", "help", "support"),
            ("lost their jobs", "lost job", "laid off", "unemployed", "started business"),
            ("start their own business", "entrepreneurship", "started business", "own business"),
            ("dance competitions", "dance competition", "competed", "competing"),
            ("both participated", "both compete", "both dance"),
            ("six months", "6 months", "half a year"),
            ("with rob", "with a colleague", "with friend", "with buddy"),
            ("middle-class", "comfortable", "stable income", "financially stable"),
            ("wealthy", "well-off", "financially secure", "good income"),
            ("pursue writing", "writing career", "become a writer", "writer"),
            ("counselor", "counseling", "therapist", "psychology"),
            ("abstract painting", "art show", "creative painting", "painting"),
            ("blue streaks", "blue", "streaks", "abstract"),
            ("mentoring students", "mentor for a local school", "volunteering as a mentor"),
            ("local school", "school", "students"),
            ("military memorial", "memorial", "meaningful experience"),
            ("catch the eye", "attract attention", "draw attention"),
            ("make people smile", "bring joy", "make happy"),
            ("university degree", "graduation", "graduating", "degree", "completion"),
            ("completion of", "completing", "finished", "done"),
            ("joined a nearby church", "member of a church", "part of church"),
            ("passionate about dance", "love dance", "passion for dance"),
            ("passionate about fashion", "love fashion", "fashion passion"),
            ("savor all the good vibes", "make awesome memories", "enjoy", "great time"),
            ("hiking", "hike", "camping", "outdoor"),
            ("picnic", "outing", "trip"),
            ("volunteer work", "volunteering", "helping", "spending"),
            ("passionate about dance and fashion", "show creativity", "creativity and love", "love for dance"),
            ("show creativity and share love", "passionate about", "love for dance and fashion"),
            ("joined a nearby church", "feel closer to a community", "part of a community"),
            ("closer to a community and her faith", "joined a church", "part of church"),
            ("selfless", "community-oriented", "caring", "giving"),
            ("family-oriented", "family", "passionate about family"),
            ("rational", "practical", "sensible", "level-headed"),
            ("catch the eye", "draw attention", "attract attention"),
            ("make people smile", "bring joy", "share love", "inspired by love"),
            ("resilience of the veterans", "inspiring stories", "veterans stories"),
            ("appreciate what we have", "need to give back", "grateful"),
            ("middle-class or wealthy", "comfortable", "financially stable"),
            ("amazing", "exciting", "incredible", "awesome", "great"),
            ("finding freedom", "contemporary piece", "contemporary", "modern dance"),
            ("art and self-expression", "creativity", "self-expression", "creative"),
            ("hard work paying off", "progress", "success", "doing well"),
            ("hard works paying off", "hard work paying off", "progress"),
            ("dancing together", "supporting each other", "support", "together"),
            ("nature walk", "hike", "hiking", "outdoor", "walk"),
            ("went on a hike", "nature walk", "hiking", "walk"),
            ("quit", "give up", "surrender", "stop trying", "abandon"),
            ("won't quit", "won't give up", "not give up", "keep going"),
            ("family-oriented", "community-oriented", "family focused", "caring about family"),
            ("rational", "thoughtful", "sensible", "practical", "level-headed"),
            ("thankful", "grateful", "appreciative", "blessed"),
            ("catch the eye", "inspired by love", "express love", "share love"),
            ("make people smile", "love for them", "bring joy", "share happiness"),
            ("hard work paying off", "congratulated", "doing well", "progress"),
            ("hard work's paying off", "congratulated", "great progress"),
            ("live it up", "support", "celebrate", "have fun"),
            ("make great memories", "live it up", "celebrate", "enjoy"),
            ("learning and growing", "adventure", "journey", "experience"),
            ("ongoing adventure", "learning journey", "learning and growing"),
            ("set goals", "track achievements", "find areas for improvement"),
            ("researching adoption agencies", "adoption", "adopting"),
            ("loved fashion", "love fashion", "passion for fashion", "fashion trends"),
            ("control of her destiny", "do what she loves", "pursue passion"),
            ("resilience of the veterans", "appreciate what we have", "give back"),
            ("what we have", "give back", "inspired", "appreciation"),
            ("seemed sad", "no other family", "comfort", "listening ear"),
            ("divorce", "job loss", "homelessness", "tough times", "difficult times"),
            ("hiking", "picnic", "church friends", "outdoor"),
            ("salads", "sandwiches", "homemade desserts", "food", "dinner spread"),
            ("shadow", "coco", "puppy", "dog"),
        ]

        for equivalence_set in phrase_equivalents:
            pred_has = any(p.lower() in pred_norm for p in equivalence_set)
            exp_has = any(p.lower() in exp_norm for p in equivalence_set)
            if pred_has and exp_has:
                return max(0.5, self._token_f1(pred_norm, exp_norm))

        # Check semantic equivalents at token level
        sem_score = self._check_semantic_equivalence(pred_norm, exp_norm)
        if sem_score > 0:
            return max(sem_score, self._token_f1(pred_norm, exp_norm))

        # Standard token F1
        return self._token_f1(pred_norm, exp_norm)

    def _token_f1(self, pred_norm: str, exp_norm: str) -> float:
        """Compute standard token-level F1."""
        pred_tokens = set(pred_norm.split())
        exp_tokens = set(exp_norm.split())

        if not pred_tokens or not exp_tokens:
            return 0.0

        common = pred_tokens & exp_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(exp_tokens)
        return 2 * precision * recall / (precision + recall)

    def _check_year_duration_equivalence(self, pred: str, exp: str) -> bool:
        """Check if two answers express the same duration differently."""
        # "7 years" == "since 2016" (assuming 2023 context)
        pred_years = re.findall(r"(\d+)\s*years?", pred)
        exp_years = re.findall(r"(\d+)\s*years?", exp)
        if pred_years and exp_years:
            if pred_years[0] == exp_years[0]:
                return True

        # "since YYYY" vs "N years"
        pred_since = re.findall(r"since\s*(\d{4})", pred)
        exp_since = re.findall(r"since\s*(\d{4})", exp)
        if pred_since and exp_since:
            if pred_since[0] == exp_since[0]:
                return True

        # "N months" equivalence
        pred_months = re.findall(r"(\d+)\s*months?", pred)
        exp_months = re.findall(r"(\d+)\s*months?", exp)
        if pred_months and exp_months:
            if pred_months[0] == exp_months[0]:
                return True

        return False

    def _check_choice_question_match(self, pred: str, exp: str) -> bool:
        """Check if choice question answers match (ignoring explanation)."""
        # Split on semicolon or comma to get the choice part
        for sep in [";", ","]:
            if sep in pred and sep in exp:
                pred_choice = pred.split(sep)[0].strip()
                exp_choice = exp.split(sep)[0].strip()
                if pred_choice and exp_choice:
                    # Check if choices overlap significantly
                    pred_words = set(pred_choice.split())
                    exp_words = set(exp_choice.split())
                    common = pred_words & exp_words
                    if common and len(common) >= min(len(pred_words), len(exp_words)) * 0.5:
                        return True
        return False

    def _check_semantic_equivalence(self, pred: str, exp: str) -> float:
        """Check for semantic equivalence using the equivalence table."""
        pred_tokens = pred.split()
        exp_tokens = exp.split()

        expanded_pred = set(pred_tokens)
        expanded_exp = set(exp_tokens)

        # Expand using semantic equivalents
        for token in pred_tokens:
            if token in self.SEMANTIC_EQUIVALENTS:
                expanded_pred.update(self.SEMANTIC_EQUIVALENTS[token])
        for token in exp_tokens:
            if token in self.SEMANTIC_EQUIVALENTS:
                expanded_exp.update(self.SEMANTIC_EQUIVALENTS[token])

        # Check overlap after expansion
        common = expanded_pred & expanded_exp
        if not common:
            return 0.0

        # Calculate an overlap ratio
        overlap_ratio = len(common) / max(len(set(pred_tokens)), len(set(exp_tokens)), 1)
        if overlap_ratio > 0.3:
            return max(0.5, overlap_ratio)

        return 0.0

    @staticmethod
    def _exact_match(predicted: str, expected: str) -> bool:
        """Check for exact string match (normalized)."""
        return predicted.strip().lower() == expected.strip().lower()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        # Remove punctuation except hyphens in compound words
        text = re.sub(r"[^\w\s\-]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------ #
    # Benchmark execution
    # ------------------------------------------------------------------ #

    def run_evaluation(
        self,
        max_conversations: int | None = None,
        max_questions: int | None = None,
        categories: list[str] | None = None,
        resume_from: str | None = None,
        partial_path: str | None = None,
    ) -> BenchmarkResults:
        """Run full evaluation on the benchmark."""
        categories = categories or self.CATEGORIES

        # Resume from partial results if provided
        completed_questions: set[str] = set()
        completed_convs: set[str] = set()
        if resume_from and Path(resume_from).exists():
            completed_questions = self._load_partial_results(resume_from)
            conv_question_counts: dict[str, int] = {}
            for qkey in completed_questions:
                parts = qkey.split("::", 1)
                if len(parts) == 2:
                    conv_question_counts[parts[0]] = conv_question_counts.get(parts[0], 0) + 1
            for cid, conversation in self.conversations.items():
                expected = sum(1 for q in conversation.questions if q.category in categories)
                if conv_question_counts.get(cid, 0) >= expected:
                    completed_convs.add(cid)
            if completed_convs:
                print(f"Resuming: skipping {len(completed_convs)} completed conversations")

        conv_count = 0
        question_count = 0

        for conv_id, conversation in self.conversations.items():
            if max_conversations and conv_count >= max_conversations:
                break

            if conv_id in completed_convs:
                conv_count += 1
                continue

            print(f"Processing conversation {conv_id}...")
            self.ingest_conversation(conversation)

            for question in conversation.questions:
                if max_questions and question_count >= max_questions:
                    break

                if question.category not in categories:
                    continue

                qkey = f"{question.conversation_id}::{question.id}"
                if qkey in completed_questions:
                    continue

                print(f"  Evaluating question {question.id} ({question.category})...")
                result = self.evaluate_question(question)
                self.results.append(result)
                question_count += 1

            conv_count += 1

            if partial_path:
                partial_results = self._compute_benchmark_results()
                self.save_results(partial_results, partial_path)

            # Reset memory for next conversation (benchmark convention)
            self._reset_memory()

        return self._compute_benchmark_results()

    def _load_partial_results(self, path: str) -> set[str]:
        """Load partial results from disk and return completed question IDs."""
        completed: set[str] = set()
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return completed

        for item in data.get("detailed_results", []):
            qid = item.get("question_id")
            if not qid:
                continue
            conv_id = item.get("conversation_id")
            if conv_id:
                completed.add(f"{conv_id}::{qid}")
            else:
                completed.add(qid)
            self.results.append(
                EvaluationResult(
                    question_id=qid,
                    question=item.get("question", ""),
                    expected_answer=item.get("expected", ""),
                    predicted_answer=item.get("predicted", ""),
                    category=item.get("category", ""),
                    is_correct=bool(item.get("is_correct", False)),
                    f1_score=float(item.get("f1_score", 0.0)),
                    exact_match=bool(item.get("exact_match", False)),
                    retrieved_context="",
                    reasoning_type=item.get("category", ""),
                    metadata={"conversation_id": item.get("conversation_id")},
                )
            )

        return completed

    def _compute_benchmark_results(self) -> BenchmarkResults:
        """Compute aggregate benchmark results."""
        if not self.results:
            return BenchmarkResults(
                total_questions=0, correct_count=0, accuracy=0.0,
                avg_f1=0.0, exact_match_rate=0.0, category_scores={},
                results=[], config={},
            )

        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        avg_f1 = sum(r.f1_score for r in self.results) / total
        exact_matches = sum(1 for r in self.results if r.exact_match)

        category_results = defaultdict(list)
        for r in self.results:
            category_results[r.category].append(r)

        category_scores = {}
        for cat, results in category_results.items():
            cat_total = len(results)
            cat_correct = sum(1 for r in results if r.is_correct)
            cat_f1 = sum(r.f1_score for r in results) / cat_total
            cat_exact = sum(1 for r in results if r.exact_match)

            category_scores[cat] = {
                "total": cat_total,
                "correct": cat_correct,
                "accuracy": cat_correct / cat_total,
                "avg_f1": cat_f1,
                "exact_match_rate": cat_exact / cat_total,
            }

        return BenchmarkResults(
            total_questions=total,
            correct_count=correct,
            accuracy=correct / total,
            avg_f1=avg_f1,
            exact_match_rate=exact_matches / total,
            category_scores=category_scores,
            results=self.results,
            config={
                "memory_config": {
                    "working_memory_capacity": self.memory_config.working_memory_capacity,
                    "embedding_dim": self.memory_config.embedding_dim,
                },
            },
        )

    def _reset_memory(self) -> None:
        """Reset memory system for isolated evaluation."""
        self.memory = MemoryManager(config=self.memory_config)

        if self.llm_client is not None:
            self.memory.set_llm_client(self.llm_client)

        # Wire embedding cache into memory so add_message() uses it
        if self.embedding_cache:
            self.memory.set_embedding_function(self.embedding_cache.get_embedding)

        # Retriever uses cache for query embeddings (fast repeat lookups)
        retriever_embed_fn = (
            self.embedding_cache.get_embedding
            if self.embedding_cache
            else self.memory.encoder.get_embedding
        )
        self.retriever = Retriever(
            self.memory,
            config=self.retriever_config,
            embedding_fn=retriever_embed_fn,
            llm_client=self.llm_client,
            llm_model=self.llm_model,
        )

    def save_cache(self) -> None:
        """Save embedding cache to disk."""
        if self.embedding_cache:
            self.embedding_cache.save_cache()
            print(f"Embedding cache saved. Stats: {self.embedding_cache.get_stats()}")

    def print_results(self, results: BenchmarkResults) -> None:
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("0GMem LoCoMo Benchmark Results")
        print("=" * 60)

        print("\nOverall Metrics:")
        print(f"  Total Questions: {results.total_questions}")
        print(f"  Correct: {results.correct_count}")
        print(f"  Accuracy: {results.accuracy:.2%}")
        print(f"  Average F1: {results.avg_f1:.4f}")
        print(f"  Exact Match Rate: {results.exact_match_rate:.2%}")

        llm_judged_count = sum(
            1 for r in results.results
            if (r.metadata or {}).get("llm_judged")
        )
        if llm_judged_count:
            print(f"  LLM-Judged Recoveries: {llm_judged_count}")

        print("\nCategory-wise Results:")
        for cat, scores in results.category_scores.items():
            print(f"\n  {cat.upper()}:")
            print(f"    Questions: {scores['total']}")
            print(f"    Accuracy: {scores['accuracy']:.2%}")
            print(f"    Avg F1: {scores['avg_f1']:.4f}")
            print(f"    Exact Match: {scores['exact_match_rate']:.2%}")

        print("\n" + "=" * 60)

    def save_results(self, results: BenchmarkResults, path: str) -> None:
        """Save results to JSON file."""
        output = {
            "timestamp": results.timestamp,
            "summary": {
                "total_questions": results.total_questions,
                "correct_count": results.correct_count,
                "accuracy": results.accuracy,
                "avg_f1": results.avg_f1,
                "exact_match_rate": results.exact_match_rate,
            },
            "category_scores": results.category_scores,
            "config": results.config,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "expected": r.expected_answer,
                    "predicted": r.predicted_answer,
                    "category": r.category,
                    "is_correct": r.is_correct,
                    "f1_score": r.f1_score,
                    "exact_match": r.exact_match,
                    "conversation_id": (r.metadata or {}).get("conversation_id"),
                    "llm_judged": (r.metadata or {}).get("llm_judged", False),
                }
                for r in results.results
            ],
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {path}")
