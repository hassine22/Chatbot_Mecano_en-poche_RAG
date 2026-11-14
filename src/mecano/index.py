"""FAISS index construction and persistence."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an inner-product index for normalized embeddings."""
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Les vecteurs doivent former une matrice non vide")
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def load_index(path: str | Path) -> faiss.Index:
    index_path = Path(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")
    return faiss.read_index(str(index_path))
