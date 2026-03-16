# 0GMem: Zero Gravity Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A next-generation AI memory system that gives LLMs structured, long-term conversational memory. Unlike flat vector stores that lose context over time, 0GMem encodes entities, temporal relationships, causality, and negations at ingestion — enabling accurate recall across hundreds of conversation sessions.

## Why 0GMem?

Most AI memory systems treat memories as flat text chunks in a vector store — they embed, retrieve, and hope for the best. This works for simple recall but falls apart when conversations grow long and questions get harder: *"When did Alice visit the Alps?"*, *"What does Bob NOT like?"*, *"Who did Alice meet after her trip to Japan?"*

0GMem takes a fundamentally different approach: **structure at write time, intelligence at read time.**

### The Problem with Flat Memory

| Challenge | Flat Vector Store | 0GMem |
|-----------|------------------|-------|
| "What does she NOT like?" | Retrieves mentions of "like" — returns both likes and dislikes, often hallucinating | Stores negations as first-class facts; retrieves the correct polarity |
| "When did X happen?" | Finds the right event but returns the wrong session's date | Event-Date Index resolves dates at ingestion, not retrieval |
| "Who did A meet after B?" | Single-hop retrieval can't chain temporal + entity reasoning | Multi-graph BFS traverses entity, temporal, and semantic edges simultaneously |
| Long conversations (900+ messages) | Retrieves too much — LLM accuracy degrades from context noise | Attention filter performs "precise forgetting," the single biggest accuracy driver (+5% on 10-conv) |
| "Did she say X or Y?" | No contradiction tracking; LLM guesses | Entity graph tracks contradictions and negative relations explicitly |

### Design Principles

- **Encode structure, not just text.** Every message is decomposed into entities, temporal anchors, causal links, and negations at ingestion time — not deferred to retrieval.
- **Multiple views of the same memory.** Four orthogonal graphs (Temporal, Semantic, Causal, Entity) capture different dimensions of meaning, enabling multi-hop reasoning across all of them.
- **Cognitive-science-inspired hierarchy.** Working memory (attention-decayed scratchpad), episodic memory (lossless conversation storage), and semantic memory (accumulated facts with confidence tracking) mirror how human memory actually works.
- **Precise forgetting matters as much as precise remembering.** The attention filter removes redundant and low-relevance context before it reaches the LLM — over-retrieval actively hurts accuracy.
- **Query-aware retrieval.** Every query is classified by intent, reasoning type, and temporal scope before retrieval begins. A temporal question activates different strategies than an adversarial or multi-hop question.

### How It Compares

| | Mem0 | Zep | MemGPT/Letta | **0GMem** |
|---|---|---|---|---|
| Memory structure | Flat facts in vector store | Knowledge graph | Agent-managed paging | **Four orthogonal graphs + three-tier hierarchy** |
| Temporal reasoning | None | Basic | None | **Allen's Interval Algebra (13 relations) + bitemporal modeling** |
| Negation handling | None | None | None | **First-class negation storage and retrieval** |
| Multi-hop reasoning | Single retrieval | Entity traversal | Agent decides | **Simultaneous BFS across entity, temporal, and semantic graphs** |
| Context quality | Top-k similarity | Top-k similarity | Agent-selected | **Attention-filtered with redundancy removal and diversity enforcement** |
| LoCoMo accuracy | 66.9–68.5% | 58–75% | 48–74% | **85.6–96.6%** |

## Key Innovations

### 1. Structure at Write Time
Every message is decomposed at ingestion — not deferred to retrieval:
- **Entity & relation extraction** with negation detection
- **Temporal anchoring** via Allen's interval algebra (13 relations)
- **Speaker-enriched embeddings**: `[Speaker] (date): content` gives the embedding model speaker and temporal signal
- **LLM topic segmentation**: Every 100 messages, an LLM segments the conversation into topic chunks with extracted entities, relations, causal links, and facts
- **Cross-person trait synthesis**: Detects shared attributes across speakers (e.g., "both Alice and Bob are engineers")

### 2. Four Orthogonal Memory Graphs
A single `UnifiedMemoryGraph` combines four views that can be traversed simultaneously:
- **Temporal Graph**: Allen's interval algebra for precise time relationships (BEFORE, AFTER, DURING, OVERLAPS, etc.)
- **Semantic Graph**: Embedding-based similarity with concept relationships
- **Causal Graph**: Cause-effect chains for "why" and "what happened because of" questions
- **Entity Graph**: Entity relationships with **first-class negation support** ("Alice does NOT like sushi")

### 3. Cognitive-Science Memory Hierarchy
- **Working Memory**: Attention-decayed scratchpad that prioritizes recent context
- **Episodic Memory**: Lossless per-message storage across sessions
- **Semantic Memory**: Accumulated facts with confidence scores and contradiction tracking
- **Topic Chunks**: LLM-segmented message groups that enable cross-message inference

### 4. 8-Strategy Retrieval with RRF Fusion
Instead of single-vector similarity, 0GMem fuses 8 retrieval strategies via Reciprocal Rank Fusion:

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

### 5. Agentic Retrieval Loop
Multi-round retrieval with sufficiency checking:
1. **Round 1**: Retrieve with original query, check if context is sufficient
2. **Round 2+**: If insufficient, rewrite the query using 5 strategies (gap-filling, synonym expansion, temporal context, multi-person injection, LLM rewrite) and retrieve again
3. Results are deduplicated, re-ranked, and merged across rounds

### 6. Attention Filter (Precise Forgetting)
The single biggest accuracy driver (+5% on 10-conv). Before the LLM sees any context:
1. Score each result for relevance (query overlap, entity presence, source type)
2. Remove low-relevance noise (threshold-based)
3. Deduplicate semantically similar results (>85% similarity)
4. Enforce topic diversity
5. Apply token budget

Over-retrieval actively hurts accuracy — this filter ensures the LLM only sees what matters.

### 7. Question-Type-Aware Reasoning
Queries are classified into 9 types, each with specialized prompts and pipelines:
- **YES_NO, FACTUAL, CHOICE**: Direct answer extraction
- **TEMPORAL_DATE, TEMPORAL_DURATION**: Event-date resolution with temporal graph
- **COUNTING**: 3-tier pipeline (regex → LLM counting with Jaccard-deduplicated evidence → date-based enumeration)
- **MULTI_HOP**: Query decomposition + cross-session graph traversal
- **ADVERSARIAL**: Negation verification against entity graph

## Installation

```bash
# Clone the repository
git clone https://github.com/loganionian/0gmem.git
cd 0gmem

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For evaluation
pip install -e ".[eval]"
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

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `MemoryManager` | Central orchestrator for memory operations |
| `Encoder` | Converts text to memory representations |
| `Retriever` | Queries memories with multi-strategy retrieval |

### Configuration

| Class | Description |
|-------|-------------|
| `MemoryConfig` | Configure memory capacity, decay rates |
| `EncoderConfig` | Configure embedding model, extraction options |
| `RetrieverConfig` | Configure retrieval strategies, weights |

### Data Types

| Class | Description |
|-------|-------------|
| `RetrievalResult` | Single retrieval result with score and source |
| `RetrievalResponse` | Complete retrieval response with context |
| `QueryAnalysis` | Query understanding and intent classification |

## Running LoCoMo Evaluation

```bash
# Download/create sample data
python scripts/download_locomo.py --sample-only

# Run evaluation (without LLM)
python scripts/run_evaluation.py --data-path data/locomo/sample_locomo.json

# Run evaluation with LLM (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key-here"
python scripts/run_evaluation.py --data-path data/locomo/sample_locomo.json --use-llm
```

## Architecture

### Write Path (Ingestion)

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

### Read Path (Retrieval)

```
Query ──▶ Query Analyzer ──▶ 8-Strategy Retrieval ──▶ RRF Fusion
            │                    │
            ▼                    ▼
       ┌──────────┐     ┌──────────────────┐
       │ Intent   │     │ 1. Semantic      │
       │ Entity   │     │ 2. Entity graph  │
       │ Temporal │     │ 3. Temporal      │
       │ Reasoning│     │ 4. Graph BFS     │
       │ Type     │     │ 5. Fact search   │
       └──────────┘     │ 6. Working mem   │
                        │ 7. BM25          │
                        │ 8. Hierarchical  │
                        └──────────────────┘
                                │
                                ▼
RRF Fusion ──▶ Entity Scoring ──▶ LLM Reranking ──▶ Attention Filter
                                                         │
                                                         ▼
                                          ┌─────────────────────────┐
                                          │ Precise Forgetting:     │
                                          │ • Relevance threshold   │
                                          │ • Semantic dedup (>85%) │
                                          │ • Diversity enforcement │
                                          │ • Token budgeting       │
                                          └──────────┬──────────────┘
                                                     ▼
                              Agentic Loop ◀── Sufficient? ──▶ Answer Generator
                              (rewrite query,       No              │ Yes
                               retrieve again)                      ▼
                                                         Question-Type-Aware
                                                         Prompt + LLM Answer
```

### Storage Layer

```
┌───────────────────────────────────────────────────────────┐
│                   Unified Memory Graph                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Temporal  │  │ Semantic │  │  Causal  │  │  Entity  │  │
│  │ (Allen's  │  │(Embedding│  │ (Cause → │  │(Relations│  │
│  │ Intervals)│  │Similarity│  │  Effect) │  │+Negation)│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
├───────────────────────────────────────────────────────────┤
│                    Memory Hierarchy                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ Working  │   │ Episodic │   │ Semantic │   │ Topic  │ │
│  │ (Decayed │   │(Lossless │   │ (Facts + │   │ Chunks │ │
│  │  Recent) │   │ Messages)│   │Confidence│   │(100msg)│ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Performance

### LoCoMo Benchmark Results

The [LoCoMo benchmark](https://snap-research.github.io/locomo/) evaluates long-term conversational memory across multi-session dialogues with 1,986 questions spanning factual recall, temporal reasoning, multi-hop inference, yes/no, adversarial, and counting question types.

**0GMem Results:**

| Subset | Accuracy | Questions |
|--------|----------|-----------|
| 3-conversation | 96.58% | 585/605 |
| 10-conversation | 85.60% | 1,700/1,986 |

### Comparison with Other Systems

| System | 10-conv Score | Notes |
|--------|---------------|-------|
| **0GMem** | **85.60%** | **Structured memory with multi-graph retrieval** |
| Human Performance | 87.9 F1 | Upper bound ([LoCoMo Paper](https://arxiv.org/abs/2402.17753)) |
| Mem0 | 66.9–68.5% | Graph-enhanced variant ([Mem0 Research](https://mem0.ai/research)) |
| Zep | 58–75% | Results disputed across studies |
| OpenAI Memory | 52.9% | Built-in memory feature |
| MemGPT/Letta | 48–74% | Varies by configuration ([Letta Blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)) |
| Best RAG Baseline | 41.4 F1 | Retrieval-augmented generation |
| GPT-3.5-turbo-16K | 37.8 F1 | Extended context window |
| GPT-4-turbo (4K) | ~32 F1 | Baseline LLM |

*Note: Metrics vary across studies (F1 vs accuracy, different evaluation protocols). Direct comparisons should be interpreted with caution.*

## Project Structure

```
0gmem/
├── src/zerogmem/
│   ├── defaults.py              # Centralized model config & shared constants
│   ├── persistence.py           # State serialization/deserialization
│   ├── mcp_server.py            # MCP server for Claude Code / OpenClaw
│   ├── graph/                   # Unified Memory Graph
│   │   ├── temporal.py          # Allen's interval algebra
│   │   ├── semantic.py          # Embedding-based similarity
│   │   ├── causal.py            # Cause-effect tracking
│   │   ├── entity.py            # Entity relationships & negations
│   │   └── unified.py           # Combined multi-graph
│   ├── memory/                  # Memory hierarchy
│   │   ├── manager.py           # Central orchestrator
│   │   ├── working.py           # Attention-decayed working memory
│   │   ├── episodic.py          # Lossless episode storage
│   │   ├── semantic.py          # Accumulated facts with confidence
│   │   ├── memcell.py           # Atomic memory units
│   │   ├── chunker.py           # LLM-based topic segmentation
│   │   ├── consolidator.py      # Memory consolidation & compression
│   │   └── extractor.py         # MemCell/MemScene extraction
│   ├── encoder/                 # Memory encoding pipeline
│   │   ├── encoder.py           # Main encoder
│   │   ├── embedding_cache.py   # Embedding cache with persistence
│   │   ├── entity_extractor.py  # Named entity recognition
│   │   ├── temporal_extractor.py # Temporal expression parsing
│   │   ├── temporal_resolver.py # Date/time resolution
│   │   ├── fact_extractor.py    # Rule-based fact extraction
│   │   ├── llm_fact_extractor.py # LLM-powered profile & fact extraction
│   │   ├── event_date_index.py  # Event-to-date mapping
│   │   ├── entity_timeline.py   # Per-entity temporal tracking
│   │   ├── session_summarizer.py # Session summary generation
│   │   └── memory_types.py      # Memory type definitions
│   ├── retriever/               # Multi-strategy retrieval
│   │   ├── retriever.py         # Main retriever with RRF fusion
│   │   ├── query_analyzer.py    # Intent classification & query rewriting
│   │   ├── hierarchical_search.py # Session → Chunk → Message tree search
│   │   ├── attention_filter.py  # Precise forgetting & noise removal
│   │   ├── entity_scorer.py     # Entity-aware scoring
│   │   ├── bm25_retriever.py    # BM25 keyword retrieval
│   │   ├── multi_query.py       # Query decomposition
│   │   ├── proposition_index.py # Proposition-level indexing
│   │   ├── reranker.py          # LLM-based reranking
│   │   └── semantic_profile_matcher.py # Profile-based matching
│   ├── reasoning/               # Answer generation & verification
│   │   ├── answer_generator.py  # LLM answer generation & normalization
│   │   ├── answer_verifier.py   # Answer sufficiency checking
│   │   ├── counting.py          # Counting pipeline with evidence dedup
│   │   ├── prompt_templates.py  # Question-type-aware prompts
│   │   └── question_decomposer.py # Compound question splitting
│   └── evaluation/              # Benchmarking
│       ├── locomo.py            # LoCoMo evaluator
│       └── profile_answerer.py  # Profile-based answer generation
├── examples/                    # Usage examples
├── tests/                       # Test suite
├── docs/                        # Documentation
└── scripts/                     # Utility scripts
```

## Key Architectural Features

| Feature | 0GMem Approach |
|---------|----------------|
| Retrieval | 8 strategies fused via Reciprocal Rank Fusion (RRF) with query-type-adaptive weights |
| Context Quality | Attention filter: relevance scoring → semantic dedup → diversity → token budget |
| Temporal Reasoning | Allen's Interval Algebra (13 relations) + event-date index + bitemporal modeling |
| Multi-hop Reasoning | Simultaneous BFS across entity, temporal, and causal graphs |
| Entity Isolation | Graduated scoring (speaker match, first-person, secondary mention — not binary filter) |
| Negation Handling | Extracted at ingestion, stored in entity graph, verified at retrieval |
| Question Awareness | 9 question types with specialized prompts and answer pipelines |
| Agentic Retrieval | Multi-round with sufficiency checking and 5 query rewriting strategies |
| Topic Segmentation | LLM-based chunking every 100 messages with entity/causal/fact extraction |
| Model Portability | Centralized config supporting gpt-4o-mini, gpt-4o, gpt-5.x with automatic parameter handling |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## References

- [LoCoMo Benchmark](https://snap-research.github.io/locomo/) - Long-term conversational memory evaluation
- [LoCoMo Paper (ACL 2024)](https://arxiv.org/abs/2402.17753) - "Evaluating Very Long-Term Conversational Memory of LLM Agents"

## License

MIT License - see [LICENSE](LICENSE) for details.
