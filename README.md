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

0GMem addresses the fundamental limitations of existing memory systems through:

### 1. Unified Memory Graph (UMG)
Four orthogonal graph views for comprehensive memory representation:
- **Temporal Graph**: Allen's interval algebra for precise temporal reasoning
- **Semantic Graph**: Embedding-based similarity with concept relationships
- **Causal Graph**: Cause-effect tracking for "why" questions
- **Entity Graph**: Entity relationships with **negative relation support**

### 2. Memory Hierarchy
Inspired by cognitive science:
- **Working Memory**: Active reasoning workspace with attention-based decay
- **Episodic Memory**: Personal history with lossless compression
- **Semantic Memory**: Accumulated facts with confidence and contradiction tracking

### 3. Advanced Retrieval
- **Multi-hop graph traversal** for complex reasoning
- **Temporal chain reasoning** for time-based questions
- **Position-aware composition** to combat "lost-in-the-middle"
- **Negation checking** for adversarial robustness

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

```
0GMem Architecture
==================

┌─────────────────────────────────────────────────────────────────┐
│                         0GMem System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Encoder  │───▶│ Memory   │───▶│Consolidate│───▶│ Retriever│  │
│  │ Layer    │    │ Manager  │    │  Layer   │    │  Layer   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Unified Memory Graph                     │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │   │
│  │  │Temporal│  │Semantic│  │ Causal │  │ Entity │        │   │
│  │  │ Graph  │  │ Graph  │  │ Graph  │  │ Graph  │        │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Memory Hierarchy                       │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐               │   │
│  │  │ Working │   │Episodic │   │Semantic │               │   │
│  │  │ Memory  │   │ Memory  │   │ Memory  │               │   │
│  │  └─────────┘   └─────────┘   └─────────┘               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
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
│   ├── graph/              # Unified Memory Graph
│   │   ├── temporal.py     # Temporal reasoning
│   │   ├── semantic.py     # Semantic similarity
│   │   ├── causal.py       # Causal reasoning
│   │   ├── entity.py       # Entity relationships
│   │   └── unified.py      # Combined graph
│   ├── memory/             # Memory hierarchy
│   │   ├── working.py      # Working memory
│   │   ├── episodic.py     # Episodic memory
│   │   ├── semantic.py     # Semantic facts
│   │   └── manager.py      # Memory orchestration
│   ├── encoder/            # Memory encoding
│   │   ├── encoder.py      # Main encoder
│   │   ├── entity_extractor.py
│   │   └── temporal_extractor.py
│   ├── retriever/          # Memory retrieval
│   │   ├── retriever.py    # Main retriever
│   │   └── query_analyzer.py
│   └── evaluation/         # Benchmarking
│       └── locomo.py       # LoCoMo evaluator
├── examples/               # Usage examples
├── tests/                  # Test suite
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Key Architectural Features

| Feature | 0GMem Approach |
|---------|----------------|
| Temporal Reasoning | Allen's Interval Algebra for precise time relationships |
| Graph Structure | Multi-graph with 4 orthogonal views |
| Multi-hop Reasoning | Graph traversal (no iterative LLM calls) |
| Negation Handling | First-class support for contradictions |
| Memory Consolidation | Lossless compression with semantic indexing |
| Context Composition | Position-aware to combat "lost-in-the-middle" |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## References

- [LoCoMo Benchmark](https://snap-research.github.io/locomo/) - Long-term conversational memory evaluation
- [LoCoMo Paper (ACL 2024)](https://arxiv.org/abs/2402.17753) - "Evaluating Very Long-Term Conversational Memory of LLM Agents"

## License

MIT License - see [LICENSE](LICENSE) for details.
