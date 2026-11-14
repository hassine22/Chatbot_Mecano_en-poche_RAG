"""Sentence-BERT embeddings used by the FAISS index."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Encode corpus passages and questions with one shared model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Le modèle d'embeddings n'expose aucune dimension")
        return int(dimension)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("Impossible d'encoder une liste vide")
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return np.asarray(vectors, dtype="float32")
