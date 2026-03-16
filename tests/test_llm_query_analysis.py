"""Tests for LLM-powered query analysis and LLM reranker."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from zerogmem import MemoryConfig, MemoryManager
from zerogmem.retriever.query_analyzer import (
    QueryAnalyzer,
    QueryAnalysis,
    ReasoningType,
    TemporalScope,
)
from zerogmem.retriever.retriever import (
    Retriever,
    RetrieverConfig,
    RetrievalResult,
)


@pytest.fixture
def embed_fn():
    def fn(text):
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(64).astype(np.float32)
    return fn


def _make_mock_llm_client(response_json):
    """Create a mock OpenAI client that returns the given JSON."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(response_json)))]
    )
    return mock_client


class TestLLMQueryAnalysis:
    def test_llm_extracts_dates(self):
        """LLM analysis should extract explicit dates into TemporalExpressions."""
        mock_client = _make_mock_llm_client({
            "dates": ["2023-06-03"],
            "target_entity": "Maria",
            "entities": ["Maria"],
            "sub_queries": [
                "Maria new activity June 2023",
                "what Maria mentioned on June 3",
            ],
            "keywords": ["new activity", "start", "recently"],
            "reasoning_type": "single_hop",
            "expected_answer_type": "text",
        })

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze(
            "What new activity did Maria start recently, as mentioned on 3 June, 2023?"
        )

        assert analysis.target_entity == "Maria"
        assert len(analysis.temporal_expressions) >= 1
        # LLM should extract 2023-06-03
        llm_dates = [
            e for e in analysis.temporal_expressions
            if e.normalized_start and e.normalized_start.month == 6 and e.normalized_start.day == 3
        ]
        assert len(llm_dates) >= 1
        assert analysis.temporal_scope != TemporalScope.NONE
        assert "sub_queries" in analysis.metadata
        assert len(analysis.metadata["sub_queries"]) >= 2

    def test_llm_extracts_entities(self):
        """LLM should identify target entity and all entities."""
        mock_client = _make_mock_llm_client({
            "dates": [],
            "target_entity": "Caroline",
            "entities": ["Caroline", "Melanie"],
            "sub_queries": ["Caroline political views", "Caroline political leaning"],
            "keywords": ["political", "leaning"],
            "reasoning_type": "single_hop",
            "expected_answer_type": "text",
        })

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What would Caroline's political leaning likely be?")

        assert analysis.target_entity == "Caroline"
        assert "Caroline" in analysis.entities
        assert "Melanie" in analysis.entities

    def test_llm_reasoning_type_temporal(self):
        """LLM should set temporal reasoning when dates are present."""
        mock_client = _make_mock_llm_client({
            "dates": ["2023-07-07"],
            "target_entity": "John",
            "entities": ["John"],
            "sub_queries": ["John July 2023"],
            "keywords": ["happen"],
            "reasoning_type": "temporal",
            "expected_answer_type": "text",
        })

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What happened to John on 7 July, 2023?")

        assert analysis.reasoning_type == ReasoningType.TEMPORAL

    def test_llm_multi_hop(self):
        """LLM should detect multi-hop reasoning."""
        mock_client = _make_mock_llm_client({
            "dates": [],
            "target_entity": "Melanie",
            "entities": ["Melanie", "Caroline"],
            "sub_queries": [
                "Melanie personality traits",
                "what Melanie says about Caroline",
            ],
            "keywords": ["personality", "traits"],
            "reasoning_type": "multi_hop",
            "expected_answer_type": "text",
        })

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze(
            "What personality traits might Melanie say Caroline has?"
        )

        assert analysis.reasoning_type == ReasoningType.MULTI_HOP

    def test_llm_failure_falls_back_to_regex(self):
        """When LLM fails, regex-only analysis should be used."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What did Alice do on January 15, 2024?")

        # Should still work (regex fallback)
        assert analysis.original_query == "What did Alice do on January 15, 2024?"
        assert len(analysis.entities) > 0  # Regex extracts "Alice"
        assert len(analysis.keywords) > 0

    def test_no_llm_uses_regex_only(self):
        """Without LLM client, regex-only analysis should work."""
        analyzer = QueryAnalyzer()  # No llm_client
        analysis = analyzer.analyze("What happened on January 15, 2024?")

        assert len(analysis.temporal_expressions) >= 1
        assert analysis.temporal_scope != TemporalScope.NONE

    def test_regex_enhances_llm_dates(self):
        """Regex should add dates that LLM missed."""
        # LLM returns no dates, but regex should find "January 15, 2024"
        mock_client = _make_mock_llm_client({
            "dates": [],  # LLM missed the date
            "target_entity": "Alice",
            "entities": ["Alice"],
            "sub_queries": ["Alice January 2024"],
            "keywords": ["happened"],
            "reasoning_type": "single_hop",
            "expected_answer_type": "text",
        })

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What happened to Alice on January 15, 2024?")

        # Regex should have added the date that LLM missed
        assert len(analysis.temporal_expressions) >= 1
        jan_dates = [
            e for e in analysis.temporal_expressions
            if e.normalized_start and e.normalized_start.month == 1
        ]
        assert len(jan_dates) >= 1

    def test_regex_enhances_llm_entities(self):
        """Regex should add entities that LLM missed."""
        mock_client = _make_mock_llm_client({
            "dates": [],
            "target_entity": "Alice",
            "entities": ["Alice"],  # LLM missed "Bob"
            "sub_queries": ["Alice and Bob conversation"],
            "keywords": ["talk"],
            "reasoning_type": "single_hop",
            "expected_answer_type": "text",
        })

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What did Alice and Bob talk about?")

        # Regex should have added Bob
        entity_lower = {e.lower() for e in analysis.entities}
        assert "alice" in entity_lower
        assert "bob" in entity_lower

    def test_llm_malformed_json_falls_back(self):
        """Malformed JSON from LLM should fall back to regex."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not valid json"))]
        )

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What does Alice like?")

        # Should still produce valid analysis (regex fallback)
        assert analysis.original_query == "What does Alice like?"

    def test_llm_markdown_fences_stripped(self):
        """JSON wrapped in markdown fences should be handled."""
        response = {
            "dates": [],
            "target_entity": "Alice",
            "entities": ["Alice"],
            "sub_queries": ["Alice preferences"],
            "keywords": ["like"],
            "reasoning_type": "single_hop",
            "expected_answer_type": "text",
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=f"```json\n{json.dumps(response)}\n```"
            ))]
        )

        analyzer = QueryAnalyzer(llm_client=mock_client)
        analysis = analyzer.analyze("What does Alice like?")

        assert analysis.target_entity == "Alice"
        assert analysis.metadata.get("llm_analyzed") is True


class TestLLMReranker:
    def test_reranker_reorders_by_score(self, embed_fn):
        """LLM reranker should reorder candidates based on LLM scores."""
        mock_client = MagicMock()
        # LLM gives high score to candidate 2, low to candidate 1
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[2, 9, 5]"))]
        )

        config = RetrieverConfig(
            use_llm_reranker=True,
            llm_rerank_top_n=10,
            use_hierarchical_search=False,
            use_reranker=False,
            use_attention_filter=False,
        )
        memory = MemoryManager(
            config=MemoryConfig(embedding_dim=64, use_chunks=False)
        )
        memory.set_embedding_function(embed_fn)

        retriever = Retriever(
            memory, config=config, embedding_fn=embed_fn,
            llm_client=mock_client, llm_model="gpt-4o-mini",
        )

        results = [
            RetrievalResult(id="a", content="wrong topic", score=1.0, source="bm25",
                           metadata={"speaker": "Alice", "session_timestamp": "2023-01"}),
            RetrievalResult(id="b", content="correct answer", score=0.8, source="bm25",
                           metadata={"speaker": "Bob", "session_timestamp": "2023-06"}),
            RetrievalResult(id="c", content="partial match", score=0.5, source="semantic",
                           metadata={"speaker": "Alice", "session_timestamp": "2023-03"}),
        ]

        analysis = QueryAnalysis(
            original_query="test query",
            intent=MagicMock(),
            reasoning_type=ReasoningType.SINGLE_HOP,
            entities=["Bob"],
            temporal_scope=TemporalScope.NONE,
            temporal_expressions=[],
            keywords=["test"],
            target_entity="Bob",
        )

        reranked = retriever._llm_rerank("test query", results, analysis)

        # Candidate "b" (LLM score 9) should now rank higher than "a" (LLM score 2)
        ids = [r.id for r in reranked]
        assert ids.index("b") < ids.index("a")

    def test_reranker_skips_without_client(self, embed_fn):
        """Without LLM client, reranker should return results unchanged."""
        config = RetrieverConfig(
            use_llm_reranker=True,
            use_hierarchical_search=False,
            use_reranker=False,
            use_attention_filter=False,
        )
        memory = MemoryManager(
            config=MemoryConfig(embedding_dim=64, use_chunks=False)
        )
        memory.set_embedding_function(embed_fn)

        retriever = Retriever(memory, config=config, embedding_fn=embed_fn)
        # No llm_client passed

        results = [
            RetrievalResult(id="a", content="test", score=1.0, source="bm25"),
        ]
        analysis = QueryAnalysis(
            original_query="test",
            intent=MagicMock(),
            reasoning_type=ReasoningType.SINGLE_HOP,
            entities=[],
            temporal_scope=TemporalScope.NONE,
            temporal_expressions=[],
            keywords=[],
        )

        reranked = retriever._llm_rerank("test", results, analysis)
        assert reranked == results

    def test_reranker_graceful_on_failure(self, embed_fn):
        """On LLM failure, reranker should return original results."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        config = RetrieverConfig(
            use_llm_reranker=True,
            use_hierarchical_search=False,
            use_reranker=False,
            use_attention_filter=False,
        )
        memory = MemoryManager(
            config=MemoryConfig(embedding_dim=64, use_chunks=False)
        )
        memory.set_embedding_function(embed_fn)

        retriever = Retriever(
            memory, config=config, embedding_fn=embed_fn,
            llm_client=mock_client,
        )

        results = [
            RetrievalResult(id="a", content="test", score=1.0, source="bm25",
                           metadata={"speaker": "Alice", "session_timestamp": ""}),
        ]
        analysis = QueryAnalysis(
            original_query="test",
            intent=MagicMock(),
            reasoning_type=ReasoningType.SINGLE_HOP,
            entities=[],
            temporal_scope=TemporalScope.NONE,
            temporal_expressions=[],
            keywords=[],
        )

        reranked = retriever._llm_rerank("test", results, analysis)
        assert reranked == results

    def test_reranker_wrong_score_count_skipped(self, embed_fn):
        """If LLM returns wrong number of scores, skip reranking."""
        mock_client = MagicMock()
        # 2 scores for 3 candidates — mismatch
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[5, 3]"))]
        )

        config = RetrieverConfig(
            use_llm_reranker=True,
            use_hierarchical_search=False,
            use_reranker=False,
            use_attention_filter=False,
        )
        memory = MemoryManager(
            config=MemoryConfig(embedding_dim=64, use_chunks=False)
        )
        memory.set_embedding_function(embed_fn)

        retriever = Retriever(
            memory, config=config, embedding_fn=embed_fn,
            llm_client=mock_client,
        )

        results = [
            RetrievalResult(id="a", content="test1", score=1.0, source="bm25",
                           metadata={"speaker": "", "session_timestamp": ""}),
            RetrievalResult(id="b", content="test2", score=0.8, source="bm25",
                           metadata={"speaker": "", "session_timestamp": ""}),
            RetrievalResult(id="c", content="test3", score=0.5, source="bm25",
                           metadata={"speaker": "", "session_timestamp": ""}),
        ]
        analysis = QueryAnalysis(
            original_query="test",
            intent=MagicMock(),
            reasoning_type=ReasoningType.SINGLE_HOP,
            entities=[],
            temporal_scope=TemporalScope.NONE,
            temporal_expressions=[],
            keywords=[],
        )

        reranked = retriever._llm_rerank("test", results, analysis)
        # Should return original results (3 items), not the mismatched reranking
        assert len(reranked) == 3


class TestTemporalRegexFix:
    def test_dd_fullmonth_comma_yyyy(self):
        """'3 June, 2023' should be extracted as a date."""
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze(
            "What new activity did Maria start recently, as mentioned on 3 June, 2023?"
        )

        # Should extract a temporal expression with June 3, 2023
        june_dates = [
            e for e in analysis.temporal_expressions
            if e.normalized_start
            and e.normalized_start.month == 6
            and e.normalized_start.day == 3
            and e.normalized_start.year == 2023
        ]
        assert len(june_dates) >= 1

    def test_7_july_2023(self):
        """'7 July, 2023' should be extracted."""
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("When did John visit on 7 July, 2023?")

        july_dates = [
            e for e in analysis.temporal_expressions
            if e.normalized_start
            and e.normalized_start.month == 7
            and e.normalized_start.day == 7
        ]
        assert len(july_dates) >= 1
