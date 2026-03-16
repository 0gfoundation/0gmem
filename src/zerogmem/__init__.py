"""
0GMem (Zero Gravity Memory): A next-generation AI memory system
designed to achieve SOTA performance on the LoCoMo benchmark.

Key innovations:
- Temporal-first design with explicit temporal reasoning
- Graph-native architecture for multi-hop reasoning
- Hierarchical memory (working -> episodic -> semantic -> meta)
- Negative fact storage for adversarial robustness
- Position-aware composition to combat "lost-in-the-middle"

Quick Start:
    from zerogmem import MemoryManager, Retriever

    memory = MemoryManager()  # Internal encoder auto-created
    retriever = Retriever(memory, embedding_fn=memory.encoder.get_embedding)

    memory.start_session()
    memory.add_message("Alice", "I love hiking in the mountains.")
    memory.end_session()

    result = retriever.retrieve("What does Alice enjoy?")
"""

__version__ = "0.1.0"
__author__ = "0G Labs"

# Core classes
from zerogmem.encoder.encoder import Encoder, EncoderConfig
from zerogmem.memory.manager import MemoryConfig, MemoryManager
from zerogmem.persistence import load_memory_state, save_memory_state
from zerogmem.reasoning.answer_generator import AnswerConfig, AnswerGenerator
from zerogmem.reasoning.counting import CountingConfig, CountingPipeline
from zerogmem.reasoning.prompt_templates import PromptTemplates, QuestionType
from zerogmem.retriever.entity_scorer import EntityScorer, EntityScoringConfig
from zerogmem.retriever.query_analyzer import QueryAnalysis, QueryAnalyzer
from zerogmem.retriever.retriever import (
    RetrievalResponse,
    RetrievalResult,
    Retriever,
    RetrieverConfig,
)

__all__ = [
    # Core orchestrators
    "MemoryManager",
    "Encoder",
    "Retriever",
    # Configuration
    "MemoryConfig",
    "EncoderConfig",
    "RetrieverConfig",
    # Data types
    "RetrievalResult",
    "RetrievalResponse",
    "QueryAnalysis",
    "QueryAnalyzer",
    # Reasoning
    "AnswerGenerator",
    "AnswerConfig",
    "PromptTemplates",
    "QuestionType",
    "CountingPipeline",
    "CountingConfig",
    # Entity scoring
    "EntityScorer",
    "EntityScoringConfig",
    # Persistence
    "save_memory_state",
    "load_memory_state",
    # Meta
    "__version__",
]
