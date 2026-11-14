"""Semantic search over the FAISS index and its aligned corpus records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


class SearchIndex(Protocol):
    ntotal: int
    d: int

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class VectorRetriever:
    """Return the nearest corpus passages, optionally filtered by vehicle/category."""

    def __init__(
        self,
        index: SearchIndex,
        embedder: TextEncoder,
        records: Sequence[dict[str, Any]],
    ) -> None:
        if not records:
            raise ValueError("Le corpus ne doit pas être vide")
        if index.ntotal != len(records):
            raise ValueError(
                f"L'index contient {index.ntotal} vecteurs, mais le corpus "
                f"contient {len(records)} passages"
            )
        self.index = index
        self.embedder = embedder
        self.records = list(records)

    def search(
        self,
        question: str,
        top_k: int = 5,
        *,
        modele: str | None = None,
        categorie: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search by cosine similarity and apply exact metadata filters.

        The FAISS index is searched completely before filters are applied. This
        ensures that filtered searches still return up to ``top_k`` matches.
        Returned records are copies with their cosine similarity in ``score``.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question ne doit pas être vide")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k doit être un entier supérieur ou égal à 1")
        if modele is not None and (not isinstance(modele, str) or not modele.strip()):
            raise ValueError("modele doit être une chaîne non vide ou None")
        if categorie is not None and (
            not isinstance(categorie, str) or not categorie.strip()
        ):
            raise ValueError("categorie doit être une chaîne non vide ou None")

        allowed = {
            position
            for position, record in enumerate(self.records)
            if (modele is None or _same_value(record.get("modele"), modele))
            and (categorie is None or _same_value(record.get("categorie"), categorie))
        }
        if not allowed:
            return []

        vectors = np.asarray(self.embedder.encode([question.strip()]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != 1:
            raise ValueError("L'encodeur doit retourner une matrice de vecteurs à une ligne")
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Dimension de la question ({vectors.shape[1]}) incompatible "
                f"avec l'index ({self.index.d})"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("L'encodeur a retourné des valeurs non finies")

        scores, indices = self.index.search(
            np.ascontiguousarray(vectors, dtype=np.float32),
            self.index.ntotal,
        )
        ranked = [
            (float(score), int(position))
            for score, position in zip(scores[0], indices[0])
            if position >= 0 and position in allowed
        ][:top_k]
        return [
            {**self.records[position], "score": score}
            for score, position in ranked
        ]


def _same_value(value: Any, requested: str) -> bool:
    return isinstance(value, str) and value.strip().casefold() == requested.strip().casefold()
