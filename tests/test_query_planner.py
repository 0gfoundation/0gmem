"""Tests for the LLM-driven query planner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from zerogmem.retriever.query_planner import (
    CombineStrategy,
    IndexAdapter,
    IndexDescriptor,
    IndexRegistry,
    PlanStep,
    QueryPlan,
    QueryPlanner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    """Minimal stand-in for RetrievalResult."""

    id: str
    content: str
    score: float
    source: str = "test"


@dataclass
class FakeAnalysis:
    """Minimal stand-in for QueryAnalysis."""

    original_query: str = "test query"
    intent: MagicMock = field(default_factory=lambda: MagicMock(value="FACTUAL"))
    reasoning_type: MagicMock = field(default_factory=lambda: MagicMock(value="SINGLE_HOP"))
    entities: list = field(default_factory=lambda: ["Alice"])
    temporal_scope: MagicMock = field(default_factory=lambda: MagicMock(value="NONE"))
    temporal_expressions: list = field(default_factory=list)
    keywords: list = field(default_factory=lambda: ["test"])
    target_entity: str | None = "Alice"
    is_negation_check: bool = False
    expected_answer_type: str = "text"
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


def _make_mock_llm(plan_json: dict | None = None, eval_json: dict | None = None):
    """Create a mock OpenAI client that returns configurable JSON responses."""
    client = MagicMock()

    default_plan = {
        "rationale": "test plan",
        "combine": "rrf",
        "steps": [
            {"step_id": 1, "index_name": "semantic", "params": {"top_k": 10}, "rationale": "test"},
            {"step_id": 2, "index_name": "bm25", "params": {"top_k": 10}, "rationale": "test"},
        ],
    }
    default_eval = {"sufficient": True, "confidence": 0.9, "diagnosis": None}

    responses = iter([
        json.dumps(plan_json or default_plan),
        json.dumps(eval_json or default_eval),
        # Extra responses for multi-round
        json.dumps(plan_json or default_plan),
        json.dumps(eval_json or default_eval),
    ])

    def create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = next(responses, '{"sufficient": true}')
        return resp

    client.chat.completions.create = create
    return client


def _make_registry_with_adapters():
    """Create a registry with semantic and bm25 adapters returning fake results."""
    registry = IndexRegistry()

    def semantic_fn(query, query_embedding, analysis, params):
        return [
            FakeResult("s1", "semantic result 1", 0.9),
            FakeResult("s2", "semantic result 2", 0.7),
        ]

    def bm25_fn(query, query_embedding, analysis, params):
        return [
            FakeResult("b1", "bm25 result 1", 0.8),
            FakeResult("s1", "semantic result 1", 0.6),  # duplicate id
        ]

    def entity_fn(query, query_embedding, analysis, params):
        return [FakeResult("e1", "entity result", 0.85)]

    registry.register(
        IndexDescriptor("semantic", "Vector similarity search", {"top_k": "int"}),
        IndexAdapter("semantic", semantic_fn),
    )
    registry.register(
        IndexDescriptor("bm25", "Keyword search", {"top_k": "int"}),
        IndexAdapter("bm25", bm25_fn),
    )
    registry.register(
        IndexDescriptor("entity", "Entity graph search", {"entities": "list"}),
        IndexAdapter("entity", entity_fn),
    )
    registry.register(
        IndexDescriptor("unavailable", "Disabled index", {}, available=False),
        IndexAdapter("unavailable", lambda **kw: []),
    )
    return registry


# ---------------------------------------------------------------------------
# IndexRegistry tests
# ---------------------------------------------------------------------------


class TestIndexRegistry:
    def test_register_and_lookup(self):
        registry = IndexRegistry()
        adapter = IndexAdapter("test", lambda **kw: [])
        registry.register(IndexDescriptor("test", "desc", {}), adapter)
        assert registry.get_adapter("test") is adapter

    def test_get_adapter_missing(self):
        registry = IndexRegistry()
        assert registry.get_adapter("nonexistent") is None

    def test_available_descriptors_filters_unavailable(self):
        registry = _make_registry_with_adapters()
        names = [d.name for d in registry.available_descriptors()]
        assert "unavailable" not in names
        assert "semantic" in names
        assert "bm25" in names

    def test_available_names(self):
        registry = _make_registry_with_adapters()
        assert "unavailable" not in registry.available_names()
        assert set(registry.available_names()) == {"semantic", "bm25", "entity"}

    def test_describe_for_llm_format(self):
        registry = _make_registry_with_adapters()
        desc = registry.describe_for_llm()
        assert "**semantic**" in desc
        assert "**bm25**" in desc
        assert "**unavailable**" not in desc


# ---------------------------------------------------------------------------
# IndexAdapter tests
# ---------------------------------------------------------------------------


class TestIndexAdapter:
    def test_execute_calls_fn(self):
        called_with = {}

        def fn(query, query_embedding, analysis, params):
            called_with.update({"query": query, "params": params})
            return [FakeResult("r1", "result", 1.0)]

        adapter = IndexAdapter("test", fn)
        results = adapter.execute("q", None, FakeAnalysis(), {"top_k": 5})
        assert len(results) == 1
        assert called_with["query"] == "q"
        assert called_with["params"] == {"top_k": 5}

    def test_execute_returns_empty_on_error(self):
        def fn(**kw):
            raise ValueError("boom")

        adapter = IndexAdapter("test", fn)
        results = adapter.execute("q", None, FakeAnalysis(), {})
        assert results == []


# ---------------------------------------------------------------------------
# QueryPlan parsing tests
# ---------------------------------------------------------------------------


class TestPlanParsing:
    def _make_planner(self, registry=None):
        reg = registry or _make_registry_with_adapters()
        return QueryPlanner(
            registry=reg,
            llm_client=_make_mock_llm(),
            llm_model="gpt-test",
        )

    def test_parse_valid_plan(self):
        planner = self._make_planner()
        plan = planner._parse_plan(json.dumps({
            "rationale": "test",
            "combine": "rrf",
            "steps": [
                {"step_id": 1, "index_name": "semantic", "params": {"top_k": 10}, "rationale": "sem"},
                {"step_id": 2, "index_name": "bm25", "params": {}, "rationale": "kw"},
            ],
        }))
        assert len(plan.steps) == 2
        assert plan.combine == CombineStrategy.RRF
        assert plan.steps[0].index_name == "semantic"

    def test_parse_plan_with_markdown_fences(self):
        planner = self._make_planner()
        text = '```json\n{"rationale":"t","combine":"union","steps":[{"step_id":1,"index_name":"semantic","params":{}}]}\n```'
        plan = planner._parse_plan(text)
        assert len(plan.steps) == 1
        assert plan.combine == CombineStrategy.UNION

    def test_parse_plan_invalid_json_fallback(self):
        planner = self._make_planner()
        plan = planner._parse_plan("not json at all")
        # Should return default plan with all available indexes
        assert len(plan.steps) == 3  # semantic, bm25, entity
        assert plan.combine == CombineStrategy.RRF

    def test_parse_plan_unknown_index_skipped(self):
        planner = self._make_planner()
        plan = planner._parse_plan(json.dumps({
            "rationale": "test",
            "combine": "rrf",
            "steps": [
                {"step_id": 1, "index_name": "semantic", "params": {}},
                {"step_id": 2, "index_name": "nonexistent_index", "params": {}},
            ],
        }))
        assert len(plan.steps) == 1
        assert plan.steps[0].index_name == "semantic"

    def test_parse_plan_empty_steps_fallback(self):
        planner = self._make_planner()
        plan = planner._parse_plan(json.dumps({
            "rationale": "test",
            "combine": "rrf",
            "steps": [],
        }))
        # Should return default plan
        assert len(plan.steps) >= 1


# ---------------------------------------------------------------------------
# Plan execution tests
# ---------------------------------------------------------------------------


class TestPlanExecution:
    def _make_planner(self):
        return QueryPlanner(
            registry=_make_registry_with_adapters(),
            llm_client=_make_mock_llm(),
            llm_model="gpt-test",
        )

    def test_execute_plan_runs_all_steps(self):
        planner = self._make_planner()
        plan = QueryPlan(
            steps=[
                PlanStep(step_id=1, index_name="semantic"),
                PlanStep(step_id=2, index_name="bm25"),
            ],
            combine=CombineStrategy.UNION,
        )
        results = planner._execute_plan(plan, "test query", None, FakeAnalysis())
        assert "semantic_1" in results
        assert "bm25_2" in results
        assert len(results["semantic_1"]) == 2
        assert len(results["bm25_2"]) == 2

    def test_execute_plan_respects_dependencies(self):
        """Step 2 depends on step 1 — should run after."""
        planner = self._make_planner()
        execution_order = []

        def semantic_fn(query, query_embedding, analysis, params):
            execution_order.append("semantic")
            return [FakeResult("s1", "sem", 0.9)]

        def entity_fn(query, query_embedding, analysis, params):
            execution_order.append("entity")
            return [FakeResult("e1", "ent", 0.8)]

        planner.registry._adapters["semantic"]._execute_fn = semantic_fn
        planner.registry._adapters["entity"]._execute_fn = entity_fn

        plan = QueryPlan(
            steps=[
                PlanStep(step_id=1, index_name="semantic"),
                PlanStep(step_id=2, index_name="entity", depends_on=[1]),
            ],
        )
        planner._execute_plan(plan, "q", None, FakeAnalysis())
        assert execution_order == ["semantic", "entity"]

    def test_execute_plan_handles_adapter_error(self):
        planner = self._make_planner()

        def failing_fn(**kw):
            raise RuntimeError("adapter error")

        planner.registry._adapters["semantic"]._execute_fn = failing_fn

        plan = QueryPlan(
            steps=[
                PlanStep(step_id=1, index_name="semantic"),
                PlanStep(step_id=2, index_name="bm25"),
            ],
        )
        results = planner._execute_plan(plan, "q", None, FakeAnalysis())
        # Semantic should return empty, bm25 should succeed
        assert len(results.get("semantic_1", [])) == 0
        assert len(results["bm25_2"]) == 2


# ---------------------------------------------------------------------------
# Result combination tests
# ---------------------------------------------------------------------------


class TestResultCombination:
    def test_union_dedup_keeps_highest_score(self):
        results = {
            "step_1": [FakeResult("a", "r1", 0.9), FakeResult("b", "r2", 0.7)],
            "step_2": [FakeResult("a", "r1", 0.5), FakeResult("c", "r3", 0.8)],
        }
        combined = QueryPlanner._union(results)
        ids = [r.id for r in combined]
        assert len(combined) == 3
        # "a" should keep score 0.9 (highest)
        a_result = next(r for r in combined if r.id == "a")
        assert a_result.score == 0.9

    def test_intersect_keeps_common_only(self):
        results = {
            "step_1": [FakeResult("a", "r1", 0.9), FakeResult("b", "r2", 0.7)],
            "step_2": [FakeResult("a", "r1", 0.5), FakeResult("c", "r3", 0.8)],
        }
        intersected = QueryPlanner._intersect(results)
        assert len(intersected) == 1
        assert intersected[0].id == "a"
        assert intersected[0].score == 0.9 + 0.5  # sum


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_parse_evaluation_sufficient(self):
        sufficient, diagnosis = QueryPlanner._parse_evaluation(
            '{"sufficient": true, "confidence": 0.95, "diagnosis": null}', 5
        )
        assert sufficient is True
        assert diagnosis == ""

    def test_parse_evaluation_insufficient(self):
        sufficient, diagnosis = QueryPlanner._parse_evaluation(
            '{"sufficient": false, "confidence": 0.3, "diagnosis": "missing temporal info"}', 5
        )
        assert sufficient is False
        assert "temporal" in diagnosis

    def test_parse_evaluation_invalid_json(self):
        sufficient, _ = QueryPlanner._parse_evaluation("not json", 3)
        assert sufficient is True  # has results, default to sufficient

    def test_parse_evaluation_no_results(self):
        sufficient, _ = QueryPlanner._parse_evaluation("not json", 0)
        assert sufficient is False


# ---------------------------------------------------------------------------
# Integration: plan_and_execute
# ---------------------------------------------------------------------------


class TestPlanAndExecute:
    def test_single_round_sufficient(self):
        """Plan generates, executes, evaluator says sufficient — one round."""
        registry = _make_registry_with_adapters()
        llm = _make_mock_llm(
            eval_json={"sufficient": True, "confidence": 0.9, "diagnosis": None},
        )
        planner = QueryPlanner(registry=registry, llm_client=llm, llm_model="gpt-test")
        results, meta = planner.plan_and_execute("What are Alice's hobbies?", None, FakeAnalysis())
        assert len(results) > 0
        assert meta["sufficient"] is True

    def test_multi_round_replan(self):
        """First round insufficient, second round sufficient."""
        registry = _make_registry_with_adapters()

        # First eval: insufficient, second eval: sufficient
        responses = iter([
            json.dumps({"rationale": "r1", "combine": "rrf", "steps": [
                {"step_id": 1, "index_name": "semantic", "params": {}},
            ]}),
            json.dumps({"sufficient": False, "confidence": 0.3, "diagnosis": "missing entity info"}),
            json.dumps({"rationale": "r2", "combine": "rrf", "steps": [
                {"step_id": 1, "index_name": "entity", "params": {}},
                {"step_id": 2, "index_name": "bm25", "params": {}},
            ]}),
            json.dumps({"sufficient": True, "confidence": 0.8, "diagnosis": None}),
        ])

        call_count = 0
        llm = MagicMock()
        def create(**kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = next(responses)
            return resp
        llm.chat.completions.create = create

        planner = QueryPlanner(
            registry=registry, llm_client=llm, llm_model="gpt-test", max_rounds=2,
        )
        results, meta = planner.plan_and_execute("Who is Alice?", None, FakeAnalysis())
        assert len(results) > 0
        assert meta["sufficient"] is True
        # Should have been called 4 times (plan, eval, replan, eval)
        assert call_count == 4

    def test_max_rounds_respected(self):
        """Evaluator always says insufficient — should stop at max_rounds."""
        registry = _make_registry_with_adapters()
        llm = _make_mock_llm(
            eval_json={"sufficient": False, "confidence": 0.1, "diagnosis": "always fail"},
        )
        planner = QueryPlanner(
            registry=registry, llm_client=llm, llm_model="gpt-test", max_rounds=2,
        )
        results, meta = planner.plan_and_execute("impossible query", None, FakeAnalysis())
        # Should still return whatever results were collected
        assert isinstance(results, list)
        assert meta["sufficient"] is False
        assert meta["failure_diagnosis"] == "always fail"


# ---------------------------------------------------------------------------
# RetrieverConfig integration
# ---------------------------------------------------------------------------


class TestRetrieverConfigIntegration:
    def test_planner_disabled_by_default(self):
        from zerogmem.retriever.retriever import RetrieverConfig
        cfg = RetrieverConfig()
        assert cfg.use_query_planner is False

    def test_planner_config_fields(self):
        from zerogmem.retriever.retriever import RetrieverConfig
        cfg = RetrieverConfig(use_query_planner=True, query_planner_max_rounds=3)
        assert cfg.use_query_planner is True
        assert cfg.query_planner_max_rounds == 3
