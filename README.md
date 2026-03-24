# 0GMem: Zero Gravity Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A next-generation AI memory system that gives LLMs structured, long-term conversational memory. Unlike flat vector stores that lose context over time, 0GMem encodes entities, temporal relationships, causality, and negations at ingestion — enabling accurate recall across hundreds of conversation sessions.

## Performance

### LoCoMo Benchmark Results

The [LoCoMo benchmark](https://snap-research.github.io/locomo/) evaluates long-term conversational memory across multi-session dialogues with 1,986 questions spanning factual recall, temporal reasoning, multi-hop inference, adversarial, and open-domain question types.

| Subset | Accuracy | Questions |
|--------|----------|-----------|
| 10-conversation | **88.67%** | 1,761/1,986 |
| 3-conversation | **96.58%** | 585/605 |

**Category breakdown (10-conversation):**

| Category | Accuracy | Questions |
|----------|----------|-----------|
| Adversarial | 95.52% | 426/446 |
| Open-domain | 92.87% | 781/841 |
| Temporal | 81.62% | 262/321 |
| Single-hop | 81.91% | 231/282 |
| Multi-hop | 63.54% | 61/96 |

### Comparison with Other Systems

| System | 10-conv Score | Notes |
|--------|---------------|-------|
| **0GMem** | **88.67%** | **Structured memory with LLM-driven retrieval planning** |
| Human Performance | 87.9 F1 | Upper bound ([LoCoMo Paper](https://arxiv.org/abs/2402.17753)) |
| Mem0 | 66.9–68.5% | Graph-enhanced variant ([Mem0 Research](https://mem0.ai/research)) |
| Zep | 58–75% | Results disputed across studies |
| OpenAI Memory | 52.9% | Built-in memory feature |
| MemGPT/Letta | 48–74% | Varies by configuration ([Letta Blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)) |
| Best RAG Baseline | 41.4 F1 | Retrieval-augmented generation |

*Note: Metrics vary across studies (F1 vs accuracy, different evaluation protocols). Direct comparisons should be interpreted with caution.*

## Why 0GMem?

Most AI memory systems treat memories as flat text chunks in a vector store — they embed, retrieve, and hope for the best. This works for simple recall but falls apart when conversations grow long and questions get harder: *"When did Alice visit the Alps?"*, *"What does Bob NOT like?"*, *"Who did Alice meet after her trip to Japan?"*

0GMem takes a fundamentally different approach: **structure at write time, intelligence at read time.**

| Challenge | Flat Vector Store | 0GMem |
|-----------|------------------|-------|
| "What does she NOT like?" | Retrieves mentions of "like" — returns both likes and dislikes | Stores negations as first-class facts; retrieves the correct polarity |
| "When did X happen?" | Finds the right event but returns the wrong session's date | Event-Date Index resolves dates at ingestion, not retrieval |
| "Who did A meet after B?" | Single-hop retrieval can't chain temporal + entity reasoning | Multi-graph BFS traverses entity, temporal, and semantic edges simultaneously |
| Long conversations (900+ messages) | Retrieves too much — LLM accuracy degrades from context noise | Attention filter performs "precise forgetting," reducing noise before LLM sees context |
| "Did she say X or Y?" | No contradiction tracking; LLM guesses | Entity graph tracks contradictions and negative relations explicitly |

## How 0GMem Works

0GMem is built around two complementary paths: a **write path** that structures information at ingestion time, and a **read path** that combines multiple retrieval strategies with LLM-driven planning at query time.

### Write Path: Structure at Ingestion

```
Message ──▶ Encoder ──▶ Memory Manager ──▶ Unified Memory Graph
              │              │
              ▼              ▼
         ┌─────────┐  ┌──────────┐
         │ Entity  │  │ Chunker  │ ◀── LLM topic segmentation
         │ Temporal│  │ (100 msg │     every 100 messages
         │Negation │  │ windows) │
         │ Facts   │  └──────────┘
         └─────────┘       │
              │            ▼
              ▼      ┌──────────────┐
         ┌─────────┐ │ Consolidator │ ◀── Cross-person trait
         │ BM25 +  │ │ (Facts,      │     synthesis, fact
         │ Vector  │ │  Profiles)   │     extraction
         │ Index   │ └──────────────┘
         └─────────┘
```

Every incoming message is decomposed into structured components:

- **Entity & relation extraction** with negation detection (e.g., "Alice does NOT like sushi")
- **Temporal anchoring** via Allen's interval algebra (13 temporal relations: BEFORE, AFTER, DURING, OVERLAPS, etc.)
- **Speaker-enriched embeddings**: `[Speaker] (date): content` gives the embedding model speaker and temporal signal
- **LLM topic segmentation**: Every 100 messages, an LLM segments the conversation into topic chunks with extracted entities, relations, causal links, and facts
- **BLIP image captions**: When conversations contain images, BLIP-generated captions are incorporated as `[Image shows: ...]` text, making visual content searchable
- **Cross-person trait synthesis**: Detects shared attributes across speakers (e.g., "both Alice and Bob are engineers")

### Read Path: Intelligence at Query Time

```
Query ──▶ Query Analyzer ──▶ Query Planner ──▶ Index Execution ──▶ Post-Processing
            │                    │                                      │
            ▼                    ▼                                      ▼
       ┌──────────┐     ┌──────────────────┐              ┌─────────────────────┐
       │ Intent   │     │ LLM generates a  │              │ Entity Scoring      │
       │ Entity   │     │ retrieval plan:  │              │ LLM Reranking       │
       │ Temporal │     │ which indexes,   │              │ Attention Filter    │
       │ Reasoning│     │ what params,     │              │ (Precise Forgetting)│
       │ Type     │     │ how to combine   │              └──────────┬──────────┘
       └──────────┘     └──────────────────┘                         ▼
                                                         Answer Generator
                                                         (Question-Type-Aware)
```

The retrieval pipeline starts with query analysis (intent classification, entity extraction, temporal scope detection), then uses the **Query Planner** to generate and execute a retrieval plan across 8 indexes. Results are post-processed through entity scoring, reranking, and an attention filter before being passed to the answer generator.

## Core Components

### 1. Unified Memory Graph

A single `UnifiedMemoryGraph` that combines four orthogonal views, traversable simultaneously:

| Graph | Purpose | Example |
|-------|---------|---------|
| **Temporal** | Allen's interval algebra for precise time relationships | "What happened BEFORE Alice's trip?" |
| **Semantic** | Embedding-based similarity with concept relationships | "What topics relate to cooking?" |
| **Causal** | Cause-effect chains | "Why did Bob change his plans?" |
| **Entity** | Entity relationships with **first-class negation** | "Alice does NOT like sushi" |

### 2. Memory Hierarchy

Inspired by cognitive science, memories are stored at multiple levels:

- **Working Memory**: Attention-decayed scratchpad that prioritizes recent context
- **Episodic Memory**: Lossless per-message storage across sessions
- **Semantic Memory**: Accumulated facts with confidence scores and contradiction tracking
- **Topic Chunks**: LLM-segmented message groups that enable cross-message inference

### 3. Query Planner (LLM-Driven Retrieval)

Instead of hardcoding which retrieval strategies to run, the **Query Planner** treats retrieval as a reasoning problem:

1. **Plan**: An LLM examines the query and available indexes, then generates a structured retrieval plan — which indexes to query, with what parameters, and how to combine results
2. **Execute**: The plan runs against 8 retrieval indexes in parallel
3. **Evaluate**: An LLM checks whether the retrieved context is sufficient to answer the query
4. **Replan**: If insufficient, the LLM diagnoses *why* and generates a revised plan — changing indexes, parameters, or combination strategy

This is enabled via `RetrieverConfig(use_query_planner=True)` and coexists with the original rule-based pipeline as a fallback.

See [docs/QUERY_PLANNER_DESIGN.md](docs/QUERY_PLANNER_DESIGN.md) for the full design.

### 4. 8-Strategy Retrieval with RRF Fusion

0GMem fuses 8 retrieval strategies via Reciprocal Rank Fusion:

| # | Strategy | What it captures |
|---|----------|-----------------|
| 1 | Semantic search | Embedding similarity |
| 2 | Entity graph lookup | Direct entity relationships |
| 3 | Temporal search | Time-based reasoning via Allen's intervals |
| 4 | Graph traversal | Multi-hop BFS across entity + causal graphs |
| 5 | Fact search | Semantic memory triple lookup |
| 6 | Working memory | Attention-weighted recent context |
| 7 | BM25 sparse search | Keyword matching for exact terms |
| 8 | Hierarchical search | Session → Chunk → Message tree traversal |

Strategy weights dynamically adjust based on query type — temporal questions boost temporal search weight, multi-hop questions boost graph traversal and hierarchical search.

### 5. Attention Filter (Precise Forgetting)

Before the LLM sees any context, the attention filter removes noise:

1. Score each result for relevance (query overlap, entity presence, source type)
2. Remove low-relevance noise (threshold-based)
3. Deduplicate semantically similar results (>85% similarity)
4. Enforce topic diversity
5. Apply token budget

Over-retrieval actively hurts accuracy — this filter ensures the LLM only sees what matters.

### 6. Question-Type-Aware Reasoning

Queries are classified into 9 types, each with specialized prompts and pipelines:

- **YES_NO, FACTUAL, CHOICE**: Direct answer extraction
- **TEMPORAL_DATE, TEMPORAL_DURATION**: Event-date resolution with temporal graph
- **COUNTING**: Evidence deduplication (Jaccard similarity) with LLM counting fallback
- **MULTI_HOP**: Query decomposition + cross-session graph traversal
- **ADVERSARIAL**: Negation verification against entity graph

### 7. Chunk Fact Extraction

An optional LLM-powered extraction pass (`MemoryConfig(use_llm_fact_extraction=True)`) that processes each conversation chunk to:

1. Resolve all pronouns to concrete entity names
2. Extract every atomic fact at fine granularity
3. Classify facts as objective or subjective (speaker opinion)

This produces ~2x more memories with richer semantic content, at the cost of additional ingestion-time LLM calls.

### 8. Centralized Model Configuration

All LLM and embedding model names are defined in a single file (`defaults.py`), making model switching a one-line change or environment variable override:

```bash
export ZEROGMEM_LLM_MODEL=gpt-4o  # or gpt-4o-mini, gpt-5.2, etc.
```

The `llm_chat_kwargs()` helper automatically handles parameter differences between model families (e.g., `max_tokens` vs `max_completion_tokens`).

## Installation

```bash
# Clone the repository
git clone https://github.com/loganionian/0gmem.git
cd 0gmem

# Install dependencies
pip install -e .

# Download spaCy model (required for entity extraction)
python -m spacy download en_core_web_sm

# For development
pip install -e ".[dev]"

# For evaluation
pip install -e ".[eval]"
```

### Environment Variables

```bash
# Required: OpenAI API key for LLM calls and embeddings
export OPENAI_API_KEY="your-key-here"

# Optional: Override default LLM model (default: gpt-5.2)
export ZEROGMEM_LLM_MODEL="gpt-4o"
```

## Quick Start

```python
from zerogmem import MemoryManager, Encoder, Retriever

# Initialize components
memory = MemoryManager()
encoder = Encoder()
memory.set_embedding_function(encoder.get_embedding)
retriever = Retriever(memory, embedding_fn=encoder.get_embedding)

# Start a conversation session
memory.start_session()

# Add messages
memory.add_message("Alice", "I love hiking in the mountains.")
memory.add_message("Bob", "Which mountains have you visited?")
memory.add_message("Alice", "I've been to the Alps last summer and Rocky Mountains in 2022.")

# End session
memory.end_session()

# Query the memory
result = retriever.retrieve("When did Alice visit the Alps?")
print(result.composed_context)
```

## MCP Integration

0GMem ships as an [MCP](https://modelcontextprotocol.io/) server, so any MCP-compatible client can use it as a persistent, structured memory backend.

### Claude Code

```bash
# Install
pip install -e .
python -m spacy download en_core_web_sm

# Add the MCP server
claude mcp add --transport stdio 0gmem -- python -m zerogmem.mcp_server

# Verify
claude mcp list
```

### OpenClaw

Add 0GMem to your `openclaw.json` (or use `openclaw config set`):

```json
{
  "mcpServers": {
    "0gmem": {
      "command": "python",
      "args": ["-m", "zerogmem.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

### Other MCP Clients

Any client that supports stdio transport can use 0GMem. The server command is:

```bash
python -m zerogmem.mcp_server
```

Pass `--data-dir /path/to/data` to customize the storage location (default: `~/.0gmem`).

### Available Tools

Once connected, the client gains access to:

| Tool | Description |
|------|-------------|
| `store_memory` | Store a conversation message or fact |
| `retrieve_memories` | Semantic search over past interactions |
| `search_memories_by_entity` | Find all memories about a person/place/thing |
| `search_memories_by_time` | Find memories from a specific time period |
| `get_memory_summary` | Get statistics about stored memories |
| `start_new_session` / `end_conversation_session` | Session lifecycle management |
| `export_memory` / `import_memory` | Portable backup and restore |
| `clear_all_memories` | Reset all stored memories |

See [docs/MCP_SERVER.md](docs/MCP_SERVER.md) for detailed configuration options and usage examples.

## Running LoCoMo Evaluation

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run full evaluation (10 conversations, ~1986 questions)
PYTHONPATH=src python scripts/run_evaluation.py \
  --data-path data/locomo/locomo10.json \
  --use-llm --use-cache --use-bm25 --use-query-planner

# Run with LLM fact extraction (more memories, slower ingestion)
PYTHONPATH=src python scripts/run_evaluation.py \
  --data-path data/locomo/locomo10.json \
  --use-llm --use-cache --use-bm25 --use-query-planner --use-llm-facts

# Limit to N conversations
PYTHONPATH=src python scripts/run_evaluation.py \
  --data-path data/locomo/locomo10.json \
  --use-llm --use-cache --use-bm25 --use-query-planner --max-conversations 3

# Trace specific questions for debugging
PYTHONPATH=src python scripts/trace_questions.py
```

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `MemoryManager` | `zerogmem.memory.manager` | Central orchestrator for all memory operations |
| `Encoder` | `zerogmem.encoder.encoder` | Converts text to structured memory representations |
| `Retriever` | `zerogmem.retriever.retriever` | Multi-strategy retrieval with RRF fusion |
| `QueryPlanner` | `zerogmem.retriever.query_planner` | LLM-driven plan-execute-evaluate retrieval loop |
| `AnswerGenerator` | `zerogmem.reasoning.answer_generator` | Question-type-aware LLM answer generation |

### Configuration

| Class | Description |
|-------|-------------|
| `MemoryConfig` | Memory capacity, decay rates, BM25, chunk fact extraction |
| `EncoderConfig` | Embedding model, extraction options |
| `RetrieverConfig` | Retrieval strategies, weights, query planner toggle |
| `AnswerConfig` | Self-consistency, evasive detection, normalization |

### Data Types

| Class | Description |
|-------|-------------|
| `RetrievalResult` | Single retrieval result with score, source, entities, timestamp |
| `RetrievalResponse` | Complete retrieval response with context and strategy metadata |
| `QueryAnalysis` | Query understanding: intent, entities, temporal scope, reasoning type |

## Project Structure

```
0gmem/
├── src/zerogmem/
│   ├── __init__.py                # Public API exports
│   ├── defaults.py                # Centralized model config & shared constants
│   ├── persistence.py             # State serialization (JSON + NPZ)
│   ├── mcp_server.py              # MCP server for Claude Code / OpenClaw
│   ├── graph/                     # Unified Memory Graph
│   │   ├── temporal.py            # Allen's interval algebra (13 relations)
│   │   ├── semantic.py            # Embedding-based similarity
│   │   ├── causal.py              # Cause-effect tracking
│   │   ├── entity.py              # Entity relationships & negations
│   │   └── unified.py             # Combined multi-graph
│   ├── memory/                    # Memory hierarchy
│   │   ├── manager.py             # Central orchestrator
│   │   ├── working.py             # Attention-decayed working memory
│   │   ├── episodic.py            # Lossless episode storage
│   │   ├── semantic.py            # Accumulated facts with confidence
│   │   ├── memcell.py             # Atomic memory units
│   │   ├── chunker.py             # LLM-based topic segmentation
│   │   ├── chunk_fact_extractor.py # Per-chunk LLM fact extraction
│   │   ├── consolidator.py        # Memory consolidation & compression
│   │   └── extractor.py           # MemCell/MemScene extraction
│   ├── encoder/                   # Memory encoding pipeline
│   │   ├── encoder.py             # Main encoder
│   │   ├── embedding_cache.py     # Embedding cache with persistence
│   │   ├── entity_extractor.py    # Named entity recognition
│   │   ├── temporal_extractor.py  # Temporal expression parsing
│   │   ├── temporal_resolver.py   # Date/time resolution
│   │   ├── fact_extractor.py      # Rule-based fact extraction
│   │   ├── llm_fact_extractor.py  # LLM-powered profile & fact extraction
│   │   ├── event_date_index.py    # Event-to-date mapping
│   │   ├── entity_timeline.py     # Per-entity temporal tracking
│   │   ├── session_summarizer.py  # Session summary generation
│   │   └── memory_types.py        # Memory type definitions
│   ├── retriever/                 # Multi-strategy retrieval
│   │   ├── retriever.py           # Main retriever with RRF fusion
│   │   ├── query_planner.py       # LLM-driven retrieval planning
│   │   ├── query_analyzer.py      # Intent classification & query rewriting
│   │   ├── hierarchical_search.py # Session → Chunk → Message tree search
│   │   ├── attention_filter.py    # Precise forgetting & noise removal
│   │   ├── entity_scorer.py       # Entity-aware scoring
│   │   ├── bm25_retriever.py      # BM25 keyword retrieval
│   │   ├── multi_query.py         # Query decomposition
│   │   ├── proposition_index.py   # Proposition-level indexing
│   │   ├── reranker.py            # LLM-based reranking
│   │   └── semantic_profile_matcher.py # Profile-based matching
│   ├── reasoning/                 # Answer generation & verification
│   │   ├── answer_generator.py    # LLM answer generation & normalization
│   │   ├── answer_verifier.py     # Answer sufficiency checking
│   │   ├── prompt_templates.py    # Question-type-aware prompts
│   │   └── question_decomposer.py # Compound question splitting
│   └── evaluation/                # Benchmarking
│       ├── locomo.py              # LoCoMo evaluator
│       └── profile_answerer.py    # Profile-based answer generation
├── scripts/                       # Utility scripts
│   ├── run_evaluation.py          # Main evaluation runner
│   ├── download_locomo.py         # Dataset downloader
│   ├── trace_questions.py         # Debug specific questions
│   ├── trace_pipeline.py          # Trace retrieval pipeline stages
│   └── analyze_errors.py          # Error analysis on results
├── tests/                         # Test suite
│   ├── conftest.py
│   ├── test_integration.py
│   └── test_query_planner.py
├── docs/                          # Documentation
│   ├── MCP_SERVER.md
│   └── QUERY_PLANNER_DESIGN.md
├── examples/                      # Usage examples
│   ├── basic_usage.py
│   └── retrieval.py
└── data/locomo/                   # Benchmark data (not in repo)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## References

- [LoCoMo Benchmark](https://snap-research.github.io/locomo/) - Long-term conversational memory evaluation
- [LoCoMo Paper (ACL 2024)](https://arxiv.org/abs/2402.17753) - "Evaluating Very Long-Term Conversational Memory of LLM Agents"

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2024 0G Labs
