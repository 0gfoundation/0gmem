"""
Proposition-level index for fine-grained semantic search.

Splits long messages into sentence-level propositions, embeds each
independently, and maps hits back to parent memory items.  This solves
the "incidental fact" problem where a key detail (e.g. "2 younger kids")
is buried in a message whose embedding is dominated by a different topic.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Proposition:
    """A sentence-level fragment of a parent memory item."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_memory_id: str = ""
    content: str = ""
    embedding: np.ndarray | None = None


def split_into_propositions(text: str, min_length: int = 20) -> list[str]:
    """Split text into sentence-level propositions.

    Uses simple regex splitting on sentence-ending punctuation.
    Only returns non-trivial sentences (>= min_length chars).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]


class PropositionIndex:
    """Parallel embedding index at the sentence level.

    Stores proposition embeddings alongside their parent memory IDs.
    At search time, results are mapped back to parent memories.
    """

    def __init__(self, embedding_dim: int = 3072):
        self.embedding_dim = embedding_dim
        self.propositions: dict[str, Proposition] = {}
        self._embeddings: list[np.ndarray] = []
        self._prop_ids: list[str] = []
        # Reverse index: parent_memory_id -> set of proposition IDs
        self._parent_to_props: dict[str, set[str]] = {}

    def add(self, prop: Proposition) -> None:
        """Add a proposition with its embedding."""
        self.propositions[prop.id] = prop
        if prop.embedding is not None:
            self._embeddings.append(prop.embedding)
            self._prop_ids.append(prop.id)
        # Track parent mapping
        if prop.parent_memory_id not in self._parent_to_props:
            self._parent_to_props[prop.parent_memory_id] = set()
        self._parent_to_props[prop.parent_memory_id].add(prop.id)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[tuple[Proposition, float]]:
        """Cosine similarity search over propositions."""
        if not self._embeddings:
            return []

        emb_matrix = np.vstack(self._embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_normed = emb_matrix / (norms + 1e-9)

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        sims = emb_normed @ q_norm

        top_indices = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_indices:
            prop_id = self._prop_ids[idx]
            prop = self.propositions.get(prop_id)
            if prop:
                results.append((prop, float(sims[idx])))
        return results

    def remove_by_parent(self, memory_id: str) -> int:
        """Remove all propositions for a parent memory. Returns count removed."""
        prop_ids = self._parent_to_props.pop(memory_id, set())
        if not prop_ids:
            return 0

        for pid in prop_ids:
            self.propositions.pop(pid, None)

        # Rebuild embedding arrays (removing the deleted entries)
        new_embeddings = []
        new_ids = []
        for i, pid in enumerate(self._prop_ids):
            if pid not in prop_ids:
                new_embeddings.append(self._embeddings[i])
                new_ids.append(pid)
        self._embeddings = new_embeddings
        self._prop_ids = new_ids

        return len(prop_ids)

    def clear(self) -> None:
        """Clear all propositions."""
        self.propositions.clear()
        self._embeddings.clear()
        self._prop_ids.clear()
        self._parent_to_props.clear()

    def __len__(self) -> int:
        return len(self.propositions)

    # ==================== Serialization ====================

    def to_dict(self) -> dict[str, Any]:
        """Serialize (without embeddings)."""
        return {
            "embedding_dim": self.embedding_dim,
            "propositions": [
                {
                    "id": p.id,
                    "parent_memory_id": p.parent_memory_id,
                    "content": p.content,
                }
                for p in self.propositions.values()
            ],
        }

    def get_embeddings_map(self) -> dict[str, np.ndarray]:
        """Get prop_id -> embedding map for persistence."""
        result = {}
        for pid, prop in self.propositions.items():
            if prop.embedding is not None:
                result[pid] = prop.embedding
        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        embeddings_map: dict[str, np.ndarray] | None = None,
    ) -> PropositionIndex:
        """Deserialize from dictionary."""
        embeddings_map = embeddings_map or {}
        index = cls(embedding_dim=data.get("embedding_dim", 3072))

        for pd in data.get("propositions", []):
            prop = Proposition(
                id=pd["id"],
                parent_memory_id=pd["parent_memory_id"],
                content=pd.get("content", ""),
                embedding=embeddings_map.get(pd["id"]),
            )
            index.add(prop)

        return index
