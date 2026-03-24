"""
Reasoning Module: Advanced reasoning components.

This module contains:
- Answer Verification: Generate-verify-refine loop
- Question Decomposition: Break complex questions into sub-questions
- Answer Generation: LLM-based answer generation with normalization
- Prompt Templates: Question-type-aware prompt construction
"""

from zerogmem.reasoning.answer_generator import (
    AnswerConfig,
    AnswerGenerator,
    AnswerNormalizer,
    EvasiveAnswerDetector,
    SelfConsistencyVoter,
)
from zerogmem.reasoning.answer_verifier import (
    AnswerVerifier,
    ConsistencyChecker,
    VerificationResult,
)
from zerogmem.reasoning.prompt_templates import (
    PromptTemplates,
    QuestionType,
    QuestionTypeClassifier,
)
from zerogmem.reasoning.question_decomposer import (
    QuestionDecomposer,
    ReasoningChainExecutor,
    SubQuestion,
)

__all__ = [
    # Answer verification
    "AnswerVerifier",
    "VerificationResult",
    "ConsistencyChecker",
    # Question decomposition
    "QuestionDecomposer",
    "ReasoningChainExecutor",
    "SubQuestion",
    # Answer generation
    "AnswerGenerator",
    "AnswerConfig",
    "AnswerNormalizer",
    "EvasiveAnswerDetector",
    "SelfConsistencyVoter",
    # Prompt templates
    "PromptTemplates",
    "QuestionType",
    "QuestionTypeClassifier",
]
