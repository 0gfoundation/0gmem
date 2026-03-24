# Query Planner Design: Retrieval as Reasoning

## Motivation

All memory systems share one fundamental: they preserve raw messages losslessly. An agent can always brute-force scan everything. The reason we build indexes is **efficiency** — structured overlays that make retrieval faster and more precise.

Today, 0GMem hardcodes 8 retrieval strategies with rule-based weight profiles that adjust by query type. This works, but it's brittle:

- Adding a new index requires wiring new code paths into the retriever
- Strategy weights are static per query type — they can't adapt to what each index actually returns
- When retrieval fails, query rewriting changes the *query* but not the *plan* — the same strategies run again with the same weights
- There's no observability into *why* a strategy failed

The core insight: **treat retrieval like a database query optimizer.** Describe what indexes exist, let an LLM generate a query plan, execute it, evaluate the results, and replan if needed.

## Design Overview

```
                        ┌──────────────────┐
                        │  Index Registry   │
                        │ (schema of all    │
                        │  available indexes│
                        │  and their APIs)  │
                        └────────┬─────────┘
                                 │
Query ──▶ Query Analyzer ──▶ Query Planner (LLM) ──▶ Query Plan
                                 │                        │
                                 │                        ▼
                                 │                  Plan Executor
                                 │                        │
                                 │                        ▼
                                 │                  Result Set
                                 │                        │
                                 │                        ▼
                                 │               Plan Evaluator (LLM)
                                 │                   │         │
                                 │              Sufficient   Insufficient
                                 │                   │         │
                                 │                   ▼         ▼
                                 │              Answer    Replanner (LLM)
                                 │             Generator   │
                                 │                         │ failure reason
                                 │                         │ + prior results
                                 │                         ▼
                                 └──────────────── Revised Query Plan
                                   (loop until sufficient or max rounds)
```

## Component Design

### 1. Index Registry

The registry is a declarative description of every available index. Each entry tells the LLM **what the index stores, when to use it, how to query it, and what it returns**. This is the "skill description" of the memory system.

```python
@dataclass
class IndexDescriptor:
    """Describes one index to the query planner."""

    name: str
    # Natural language description for the LLM planner
    description: str
    # When this index is most useful
    best_for: list[str]
    # What parameters the index accepts
    query_params: dict[str, ParamSpec]
    # What the index returns
    returns: str
    # Example queries (few-shot for the planner)
    examples: list[IndexExample]
    # Whether this index is currently populated / available
    available: bool = True


@dataclass
class ParamSpec:
    """Describes a query parameter."""
    name: str
    type: str          # "embedding", "string", "list[str]", "date_range", "int"
    required: bool
    description: str
    default: Any = None


@dataclass
class IndexExample:
    """Few-shot example for the planner."""
    user_query: str
    plan_step: dict     # The plan step that would use this index
    rationale: str      # Why this index is appropriate
```

**Registry entries for current indexes:**

```python
INDEXES = [
    IndexDescriptor(
        name="semantic",
        description="Vector similarity search over all memory items using embedding cosine distance. "
                    "Good for finding content that is semantically related to the query, even if "
                    "exact keywords don't match.",
        best_for=["general recall", "paraphrased content", "conceptual similarity"],
        query_params={
            "query_text": ParamSpec("query_text", "string", True, "The search query"),
            "top_k": ParamSpec("top_k", "int", False, "Number of results", default=15),
        },
        returns="Ranked list of memory items with similarity scores",
        examples=[
            IndexExample(
                user_query="What are Alice's hobbies?",
                plan_step={"index": "semantic", "query_text": "Alice hobbies interests activities"},
                rationale="Broad semantic search to find hobby-related content about Alice",
            )
        ],
    ),

    IndexDescriptor(
        name="entity_graph",
        description="Structured graph of entity relationships. Stores who is connected to whom, "
                    "entity attributes, and negative relations (e.g., 'Alice does NOT like sushi'). "
                    "Supports direct entity lookup and relationship traversal.",
        best_for=["entity-specific questions", "relationship queries", "negation checks"],
        query_params={
            "entities": ParamSpec("entities", "list[str]", True, "Entity names to look up"),
            "include_relations": ParamSpec("include_relations", "bool", False, "Include related entities", default=True),
        },
        returns="Entity profiles, relationships, and associated memory items",
        examples=[
            IndexExample(
                user_query="What does Bob NOT like?",
                plan_step={"index": "entity_graph", "entities": ["Bob"], "include_relations": True},
                rationale="Entity graph stores negations as first-class relations",
            )
        ],
    ),

    IndexDescriptor(
        name="temporal",
        description="Temporal index using Allen's Interval Algebra (13 relations: BEFORE, AFTER, "
                    "DURING, OVERLAPS, etc.). Resolves event-to-date mappings and supports temporal "
                    "chain reasoning (A before B before C).",
        best_for=["when questions", "date lookups", "temporal ordering", "event timelines"],
        query_params={
            "time_expressions": ParamSpec("time_expressions", "list[str]", False, "Date/time expressions to search for"),
            "entities": ParamSpec("entities", "list[str]", False, "Entities whose timeline to search"),
            "relation": ParamSpec("relation", "string", False, "Temporal relation: BEFORE, AFTER, DURING, etc."),
        },
        returns="Memory items with resolved timestamps and temporal relations",
        examples=[
            IndexExample(
                user_query="When did Alice visit the Alps?",
                plan_step={"index": "temporal", "entities": ["Alice"], "time_expressions": ["Alps", "visit"]},
                rationale="Temporal index resolves event-to-date mappings at ingestion time",
            )
        ],
    ),

    IndexDescriptor(
        name="graph_traversal",
        description="Multi-hop BFS across entity, temporal, and causal graphs simultaneously. "
                    "Follows cause→effect chains and entity→entity paths. Essential for questions "
                    "that require connecting information across multiple messages or sessions.",
        best_for=["multi-hop reasoning", "causal chains", "connecting distant facts"],
        query_params={
            "start_entities": ParamSpec("start_entities", "list[str]", True, "Starting entity names"),
            "max_hops": ParamSpec("max_hops", "int", False, "Maximum traversal depth", default=3),
            "follow_causal": ParamSpec("follow_causal", "bool", False, "Follow causal edges", default=True),
        },
        returns="Memory items reachable within max_hops, with reasoning paths",
        examples=[
            IndexExample(
                user_query="Who did Alice meet after her trip to Japan?",
                plan_step={"index": "graph_traversal", "start_entities": ["Alice", "Japan"], "max_hops": 3, "follow_causal": False},
                rationale="Requires connecting Alice's Japan trip to subsequent meetings — multi-hop",
            )
        ],
    ),

    IndexDescriptor(
        name="fact_store",
        description="Semantic memory of extracted (subject, predicate, object) triples with confidence "
                    "scores. Stores accumulated facts, supports negation lookups, and category-based filtering.",
        best_for=["factual questions", "attribute lookup", "verification", "negation checks"],
        query_params={
            "subject": ParamSpec("subject", "string", False, "Entity subject to look up"),
            "predicate": ParamSpec("predicate", "string", False, "Relationship type"),
            "check_negation": ParamSpec("check_negation", "bool", False, "Check for negated facts", default=False),
        },
        returns="Matching facts with confidence scores and source references",
        examples=[
            IndexExample(
                user_query="Where is Bob from?",
                plan_step={"index": "fact_store", "subject": "Bob", "predicate": "origin"},
                rationale="Direct fact lookup for a specific entity attribute",
            )
        ],
    ),

    IndexDescriptor(
        name="working_memory",
        description="Attention-decayed recent context. Items lose weight over time. "
                    "Best for recent conversation context and follow-up questions.",
        best_for=["recent context", "follow-up questions", "what we just discussed"],
        query_params={
            "top_k": ParamSpec("top_k", "int", False, "Number of recent items", default=10),
        },
        returns="Recent memory items with attention weights",
        examples=[],
    ),

    IndexDescriptor(
        name="bm25",
        description="BM25 sparse keyword retrieval. Matches exact terms and handles vocabulary "
                    "that embedding models may miss (proper nouns, technical terms, numbers).",
        best_for=["exact keyword matching", "proper nouns", "technical terms", "numbers"],
        query_params={
            "query_text": ParamSpec("query_text", "string", True, "Search query"),
            "top_k": ParamSpec("top_k", "int", False, "Number of results", default=15),
        },
        returns="Keyword-matched memory items with BM25 scores",
        examples=[
            IndexExample(
                user_query="How many times did Alice mention TensorFlow?",
                plan_step={"index": "bm25", "query_text": "Alice TensorFlow", "top_k": 30},
                rationale="BM25 excels at exact term matching — 'TensorFlow' may not embed well",
            )
        ],
    ),

    IndexDescriptor(
        name="hierarchical",
        description="Three-level tree search: Session → Chunk → Message. Chunks are LLM-segmented "
                    "topic groups (~100 messages each) with extracted entities, facts, and causal links. "
                    "Good for narrowing down to the right conversation segment before diving into details.",
        best_for=["broad context questions", "cross-session reasoning", "finding the right conversation"],
        query_params={
            "query_text": ParamSpec("query_text", "string", True, "Search query"),
            "top_k_sessions": ParamSpec("top_k_sessions", "int", False, "Sessions to search", default=3),
            "top_k_chunks": ParamSpec("top_k_chunks", "int", False, "Chunks per session", default=3),
            "top_k_messages": ParamSpec("top_k_messages", "int", False, "Messages per chunk", default=5),
        },
        returns="Messages from the most relevant session/chunk combinations",
        examples=[
            IndexExample(
                user_query="What did Alice and Bob discuss about their vacation plans?",
                plan_step={"index": "hierarchical", "query_text": "Alice Bob vacation plans", "top_k_sessions": 5},
                rationale="Need to find the right conversation(s) among potentially many sessions",
            )
        ],
    ),
]
```

### 2. Query Plan

A query plan is a structured sequence of steps that the LLM generates. Each step specifies which index to query, with what parameters, and how to combine results.

```python
@dataclass
class PlanStep:
    """A single step in a query plan."""

    step_id: int
    index: str                      # Index name from registry
    params: dict[str, Any]          # Query parameters
    rationale: str                  # Why this step (for observability)
    depends_on: list[int] = field(default_factory=list)  # Step IDs this depends on
    combine: str = "union"          # "union" | "intersect" | "filter_by"


@dataclass
class QueryPlan:
    """A complete query plan generated by the LLM planner."""

    query: str                      # Original user query
    analysis: str                   # LLM's analysis of what the query needs
    steps: list[PlanStep]           # Ordered steps to execute
    combination_strategy: str       # How to merge all step results: "rrf" | "weighted_union" | "sequential_filter"
    expected_answer_type: str       # "text" | "yes_no" | "number" | "date" | "list"
    max_results: int = 15           # Final result count target
```

**Example plan for "When did Alice visit the Alps?":**

```json
{
  "query": "When did Alice visit the Alps?",
  "analysis": "Temporal question about a specific entity (Alice) and location (Alps). Need to find the event and resolve its date. Temporal index should have event-date mappings; entity graph can provide context.",
  "steps": [
    {
      "step_id": 1,
      "index": "temporal",
      "params": {"entities": ["Alice"], "time_expressions": ["Alps", "visit"]},
      "rationale": "Temporal index resolves event-to-date mappings directly",
      "depends_on": [],
      "combine": "union"
    },
    {
      "step_id": 2,
      "index": "entity_graph",
      "params": {"entities": ["Alice", "Alps"]},
      "rationale": "Entity graph may have relationship between Alice and Alps with temporal context",
      "depends_on": [],
      "combine": "union"
    },
    {
      "step_id": 3,
      "index": "bm25",
      "params": {"query_text": "Alice Alps visit", "top_k": 10},
      "rationale": "Fallback exact match for 'Alps' in case indexes miss it",
      "depends_on": [],
      "combine": "union"
    }
  ],
  "combination_strategy": "rrf",
  "expected_answer_type": "date",
  "max_results": 10
}
```

**Example plan for "How many times did Alice go hiking?":**

```json
{
  "query": "How many times did Alice go hiking?",
  "analysis": "Counting question. Need ALL instances of Alice hiking, not just top-k. BM25 and semantic search with high top_k to capture every mention. Entity graph for any hiking-related relations.",
  "steps": [
    {
      "step_id": 1,
      "index": "bm25",
      "params": {"query_text": "Alice hiking hike hiked", "top_k": 30},
      "rationale": "BM25 with high top_k to catch every mention of hiking",
      "depends_on": [],
      "combine": "union"
    },
    {
      "step_id": 2,
      "index": "semantic",
      "params": {"query_text": "Alice went hiking outdoor trail mountain walk", "top_k": 30},
      "rationale": "Semantic search catches paraphrases like 'went on a trail' or 'mountain walk'",
      "depends_on": [],
      "combine": "union"
    },
    {
      "step_id": 3,
      "index": "entity_graph",
      "params": {"entities": ["Alice"]},
      "rationale": "Get all Alice-related content to find hiking mentions in context",
      "depends_on": [],
      "combine": "union"
    }
  ],
  "combination_strategy": "weighted_union",
  "expected_answer_type": "number",
  "max_results": 30
}
```

### 3. Plan Executor

The executor runs the plan steps, respecting dependencies, and collects results.

```python
class PlanExecutor:
    """Executes a query plan against the index registry."""

    def __init__(self, indexes: dict[str, IndexAdapter]):
        self.indexes = indexes

    def execute(self, plan: QueryPlan) -> ExecutionResult:
        step_results: dict[int, list[RetrievalResult]] = {}

        # Topological sort by depends_on, then execute
        for step in self._topo_sort(plan.steps):
            # Resolve dependencies — inject prior results as context if needed
            dep_results = [step_results[d] for d in step.depends_on]

            # Execute against the appropriate index
            adapter = self.indexes[step.index]
            results = adapter.query(**step.params, prior_results=dep_results)

            step_results[step.step_id] = results

        # Combine all step results according to plan's combination strategy
        combined = self._combine(step_results, plan)

        return ExecutionResult(
            plan=plan,
            step_results=step_results,
            combined_results=combined,
            step_stats={sid: StepStats(count=len(r), ...) for sid, r in step_results.items()},
        )
```

`IndexAdapter` is a thin wrapper that translates plan parameters into actual index method calls:

```python
class IndexAdapter(Protocol):
    """Adapts a plan step to an actual index query."""

    @property
    def name(self) -> str: ...

    def query(self, **params) -> list[RetrievalResult]: ...
```

Each current retrieval strategy gets wrapped in an adapter. Adding a new index = writing an adapter + adding a registry entry. No changes to the planner or executor.

### 4. Plan Evaluator + Replanner

This is the agentic loop — but at the **plan level**, not the query level.

```python
class PlanEvaluator:
    """Evaluates execution results and decides whether to replan."""

    def evaluate(
        self,
        query: str,
        plan: QueryPlan,
        execution: ExecutionResult,
    ) -> EvaluationResult:
        """LLM evaluates whether results are sufficient."""
        ...


@dataclass
class EvaluationResult:
    sufficient: bool
    confidence: float               # 0-1
    failure_diagnosis: str | None   # Why it failed (if insufficient)
    step_assessments: dict[int, StepAssessment]  # Per-step evaluation
    suggested_changes: list[str]    # Hints for replanner


@dataclass
class StepAssessment:
    step_id: int
    useful: bool                    # Did this step contribute useful results?
    result_count: int
    quality: str                    # "high", "medium", "low", "empty"
    issue: str | None               # What went wrong (if anything)
```

The evaluator prompt gives the LLM:
- The original query
- The plan that was executed
- Per-step result counts and sample content
- The combined result set

The LLM then assesses:
1. **Sufficiency** — Do we have enough information to answer?
2. **Per-step quality** — Which steps helped? Which returned noise or nothing?
3. **Failure diagnosis** — *Why* are we missing information? (wrong entities? wrong time range? need a different index?)
4. **Suggested changes** — Concrete hints for the replanner

**Replanner:**

```python
class Replanner:
    """Generates a revised query plan based on failure diagnosis."""

    def replan(
        self,
        query: str,
        prior_plan: QueryPlan,
        evaluation: EvaluationResult,
        registry: IndexRegistry,
        round_num: int,
    ) -> QueryPlan:
        """LLM generates a new plan informed by what failed and why."""
        ...
```

The replanner prompt includes:
- Original query
- Index registry schema (what's available)
- Prior plan (what we tried)
- Evaluation results (what worked, what failed, why)
- Instruction: generate a revised plan that addresses the diagnosed failures

**Example replan cycle:**

```
Round 1:
  Plan: temporal(Alice, Alps) + entity_graph(Alice, Alps) + bm25("Alice Alps visit")
  Result: temporal returned 0 results, entity_graph found Alice but no Alps relation, bm25 found 2 messages
  Diagnosis: "The event 'Alps visit' may not have been extracted as a temporal anchor. The raw messages
              mention it but the temporal index missed it. Try semantic search for the event content,
              and hierarchical search to find the right conversation segment."

Round 2:
  Plan: semantic("Alice visited Alps mountains trip") + hierarchical("Alice Alps trip") + bm25("Alice Alps")
  Result: semantic found 5 relevant messages, hierarchical found 3 messages from session-7
  Diagnosis: "Sufficient — found messages describing Alice's Alps visit with date context."
  → Answer generated
```

### 5. Prompt Templates

#### Planner Prompt

```
You are a query planner for a memory retrieval system. Given a user's question and the available
indexes, generate a query plan that will retrieve the most relevant information.

## Available Indexes

{registry_schema}

## User Query

{query}

## Query Analysis

{analysis}

## Instructions

Generate a JSON query plan with:
1. "analysis": Your reasoning about what information is needed and which indexes are best suited
2. "steps": Array of plan steps, each with:
   - "step_id": Sequential integer
   - "index": Name of the index to query
   - "params": Parameters for the query (must match the index's query_params)
   - "rationale": Why this step is needed
   - "depends_on": List of step_ids that must complete first (empty for independent steps)
   - "combine": How to combine with other steps ("union", "intersect", "filter_by")
3. "combination_strategy": "rrf" (fuse by rank) or "weighted_union" (merge all) or "sequential_filter" (narrow progressively)
4. "expected_answer_type": "text", "yes_no", "number", "date", or "list"
5. "max_results": Target number of final results

Think about:
- Which indexes are most likely to have the answer for this type of question?
- Are there fallback strategies if the primary index misses?
- For counting questions, use high top_k to capture all instances
- For multi-hop questions, plan graph traversal from one entity to another
- Independent steps can run in parallel (no depends_on)
```

#### Evaluator Prompt

```
You are evaluating retrieval results for a memory system query.

## Original Query
{query}

## Plan Executed
{plan_json}

## Results Per Step
{step_results_summary}

## Combined Results (top 10)
{combined_preview}

## Task
Assess whether the retrieved results are sufficient to answer the query.
For each plan step, assess whether it contributed useful results.

Respond with JSON:
{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "failure_diagnosis": "..." or null,
  "step_assessments": {
    "1": {"useful": true/false, "quality": "high/medium/low/empty", "issue": "..." or null},
    ...
  },
  "suggested_changes": ["...", "..."]
}

If insufficient, be specific about WHY: which entity is missing, which time period is uncovered,
which relationship is unresolved. This diagnosis will be used to generate a better plan.
```

#### Replanner Prompt

```
You are revising a query plan that did not produce sufficient results.

## Original Query
{query}

## Available Indexes
{registry_schema}

## Prior Plan
{prior_plan_json}

## Evaluation
{evaluation_json}

## What to Fix
Based on the evaluation, the following issues need to be addressed:
{failure_diagnosis}
{suggested_changes}

## Instructions
Generate a REVISED query plan that:
1. Keeps steps that worked well (quality "high" or "medium")
2. Replaces or removes steps that failed
3. Adds new steps to address the diagnosed gaps
4. May use different indexes, different parameters, or different combination strategies

Do NOT simply repeat the same plan. Address the specific failure reasons.
```

## Integration with Current System

### Migration Path

The query planner doesn't replace the current system overnight. It wraps it:

**Phase 1: Planner as Orchestrator (replace `_agentic_retrieve`)**
- Current `QueryAnalyzer` feeds into the planner instead of hardcoded strategy selection
- Each current strategy becomes an `IndexAdapter`
- RRF fusion becomes one of the `combination_strategy` options
- Attention filter remains as post-plan processing
- The evaluator + replanner replace the current sufficiency check + query rewriting loop

**Phase 2: Index Registry as Plugin System**
- New indexes are added by writing an adapter + registry entry
- No changes to planner/executor/evaluator
- Third-party indexes can be plugged in

**Phase 3: Learn from History**
- Store successful query plans and their evaluations
- Use as few-shot examples for the planner
- Plans that consistently work for a query type become "cached plans"

### What Stays the Same

- All existing index implementations (semantic, BM25, entity graph, etc.)
- The `RetrievalResult` data format
- Attention filter (post-plan processing)
- Answer generation and verification pipelines
- The memory hierarchy and graph structures

### What Changes

| Current | Query Planner |
|---------|---------------|
| Hardcoded 8 strategies always run | LLM selects which indexes to query |
| Static weight profiles per query type | Weights emerge from plan structure |
| Query rewriting on failure | Plan rewriting on failure (can change indexes, not just query text) |
| `get_retrieval_strategy()` returns flags | Planner returns structured `QueryPlan` |
| No observability into why retrieval failed | Per-step assessment with failure diagnosis |
| Adding an index = wiring code | Adding an index = adapter + registry entry |

## Cost Analysis

Each planning round requires LLM calls:
- **Plan generation**: 1 call (~500 tokens out)
- **Plan evaluation**: 1 call (~200 tokens out)
- **Replanning**: 1 call per retry (~500 tokens out)

With max 3 rounds: 3-7 LLM calls for planning overhead. Compare with current system:
- Query analysis: 1 LLM call
- Sufficiency check: 1 LLM call per round
- Query rewrite: 1 LLM call per round
- LLM reranking: 1 LLM call

Similar total LLM cost, but the planning calls are more informative (structured plans vs. rewritten query strings).

**Optimization**: For common query patterns, cache successful plans and skip the planner entirely. The planner only runs for novel or complex queries.

## Open Questions

1. **Plan complexity**: Should plans support conditionals? ("If temporal returns <3 results, try semantic instead") Or keep it simple with fixed steps + replan loop?

2. **Parallel vs sequential**: Current design allows both (via `depends_on`). Should the executor optimize for parallelism automatically?

3. **Plan caching**: How aggressively should we cache plans? By query type? By entity pattern? By exact query template?

4. **Evaluation granularity**: Should the evaluator see full result content, or just summaries/counts? Full content = better assessment but higher token cost.

5. **Graceful degradation**: If the LLM planner is unavailable (rate limit, cost constraint), fall back to current hardcoded strategy selection?
