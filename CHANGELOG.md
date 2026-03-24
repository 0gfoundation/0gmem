# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-24

### Added

- **Query Planner** — LLM-driven plan-execute-evaluate retrieval loop (`query_planner.py`)
  - Generates retrieval plans selecting from 8 indexed strategies
  - Evaluates result sufficiency and replans if needed (configurable rounds)
  - Falls back to default RRF plan on LLM failure
  - Enable via `RetrieverConfig(use_query_planner=True)`
- **Chunk Fact Extraction** — per-chunk LLM fact extraction with pronoun resolution (`chunk_fact_extractor.py`)
  - Extracts structured facts from conversation chunks during ingestion
  - Resolves pronouns to named entities for better retrieval
  - Enable via `MemoryConfig(use_llm_fact_extraction=True)`
- **BLIP Image Captions** — `[Image shows: ...]` appended to messages during ingestion
  - Cache version bumped to v2 to invalidate stale caches
- **Centralized LLM Config** (`defaults.py`)
  - `DEFAULT_LLM_MODEL` (env: `ZEROGMEM_LLM_MODEL`), `DEFAULT_EMBEDDING_MODEL`
  - `llm_chat_kwargs()` helper for gpt-4o vs gpt-5.x parameter differences
  - Number-word consolidation (`WORD_TO_NUM`, `ORDINAL_TO_NUM`, `build_number_synonyms()`)
- **LLM Query Rewriting** — strategy 5 in agentic retrieval generates focused follow-up queries
- **LLM Reranker Tuning** — multiplier floor 0.3 prevents score zeroing, reduced candidates 30→15
- **LLM Sufficiency Check Tuning** — accepts quantities/ranges as sufficient context
- **Allen's Temporal Expansion** — `_temporal_search()` uses BEFORE/AFTER interval relations
- **Causal Chain Queries** — `_graph_traversal()` follows cause→effect chains for multi-hop
- **Entity Relation Patterns** — origin/homeland patterns in fact and entity extractors
- **Speaker-Enriched Embeddings** — `[Speaker] (date): content` format for better retrieval
- Diagnostic scripts: `trace_questions.py`, `trace_pipeline.py`, `analyze_errors.py`
- Design doc: `docs/QUERY_PLANNER_DESIGN.md`
- Tests: `tests/test_query_planner.py`

### Changed

- All 24 LLM API call sites converted to use `llm_chat_kwargs()` for model portability
- Chunker: JSON parsing bracket-matching fallback, configurable `max_completion_tokens` (default 16384)
- Attention filter threshold tuning for better precision
- Hierarchical search improvements

### Removed

- `reasoning/counting.py` — counting logic consolidated into `answer_generator.py`
- 28 debug/trace scripts from `scripts/`

### Performance

- **10-conversation benchmark: 88.67% (1761/1986)** — up from 80.41%
- **3-conversation benchmark: 96.58% (585/605)** — up from 95.57%

## [0.1.0] - 2024-03-01

### Added

- Initial open-source release of 0GMem
- **Unified Memory Graph** with four orthogonal views:
  - Temporal Graph using Allen's interval algebra
  - Semantic Graph with embedding-based similarity
  - Causal Graph for cause-effect relationships
  - Entity Graph with negative relation support
- **Memory Hierarchy**:
  - Working Memory with attention-based decay
  - Episodic Memory with lossless compression
  - Semantic Memory with confidence tracking
- **Core Components**:
  - `MemoryManager` - Central orchestrator
  - `Encoder` - Text to memory encoding
  - `Retriever` - Multi-strategy retrieval
  - `QueryAnalyzer` - Query understanding
- **LoCoMo Benchmark** evaluation support
- Position-aware context composition
- Multi-hop graph traversal for complex queries
- Examples and documentation

### Performance

- 3-conversation benchmark: 95.57% accuracy
- 10-conversation benchmark: 80.41% accuracy

[Unreleased]: https://github.com/loganionian/0gmem/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/loganionian/0gmem/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/loganionian/0gmem/releases/tag/v0.1.0
