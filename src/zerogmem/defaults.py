"""Centralized defaults for 0GMem configuration.

All LLM model names and embedding model names are defined here so that
switching models (e.g., gpt-4o-mini, gpt-4o, gpt-5.2) is a one-line change
or environment variable override.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL: str = os.environ.get("ZEROGMEM_LLM_MODEL") or "gpt-5.2"
"""Default LLM model for all chat completion calls.

Override via the ``ZEROGMEM_LLM_MODEL`` environment variable.
Supported values: ``gpt-4o-mini``, ``gpt-4o``, ``gpt-5.2``, etc.
"""

DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-large"
"""Default embedding model for vector representations."""


# ---------------------------------------------------------------------------
# Number word mappings (shared across counting, answer normalization, eval)
# ---------------------------------------------------------------------------

WORD_TO_NUM: dict[str, str] = {
    "zero": "0",
    "none": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "once": "1",
    "twice": "2",
    "thrice": "3",
    "single": "1",
    "a couple": "2",
    "couple of": "2",
    "few": "3",
}
"""Map number words/phrases to digit strings."""

ORDINAL_TO_NUM: dict[str, str] = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "1st": "1",
    "2nd": "2",
    "3rd": "3",
    "4th": "4",
    "5th": "5",
}
"""Map ordinal words to digit strings."""


def build_number_synonyms() -> dict[str, list[str]]:
    """Build reverse mapping: digit -> list of synonyms (for eval scoring)."""
    synonyms: dict[str, list[str]] = {}
    for word, digit in WORD_TO_NUM.items():
        synonyms.setdefault(digit, [digit])
        if word not in synonyms[digit]:
            synonyms[digit].append(word)
    return synonyms


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def llm_chat_kwargs(
    model: str,
    max_tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    """Build keyword arguments for ``chat.completions.create()``.

    Different model families use different parameter names for the token
    budget:
    - ``gpt-5.*`` uses ``max_completion_tokens``
    - ``gpt-4o*`` and others use ``max_tokens``

    This helper abstracts that difference so callers don't need to care.

    Usage::

        response = client.chat.completions.create(
            **llm_chat_kwargs(model, max_tokens=500, temperature=0),
            messages=[...],
        )
    """
    kwargs: dict[str, Any] = {"model": model, **extra}
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    return kwargs
