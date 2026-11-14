"""Load and validate the merged JSONL corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mecano.config import get_project_root, load_config

_CACHED_CORPUS: list[dict[str, Any]] | None = None


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load one JSON object per line and reject malformed or empty corpora."""
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus introuvable : {corpus_path}")

    records: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {corpus_path}, ligne {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Le passage de la ligne {line_number} doit être un objet JSON"
                )
            for field in ("id", "texte_embedding"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(
                        f"Champ '{field}' manquant ou vide à la ligne {line_number}"
                    )
            records.append(record)

    if not records:
        raise ValueError(f"Corpus vide : {corpus_path}")
    return records


def load_all_corpus(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Charge le corpus fusionné en mémoire (avec cache par défaut)."""
    global _CACHED_CORPUS
    if _CACHED_CORPUS is not None and path is None:
        return _CACHED_CORPUS

    if path is None:
        cfg = load_config()
        rel_path = cfg.get("corpus", {}).get("path", "corpus/output/all_corpus.jsonl")
        corpus_path = get_project_root() / rel_path
    else:
        corpus_path = Path(path)
        if not corpus_path.is_absolute():
            corpus_path = get_project_root() / corpus_path

    records = load_corpus(corpus_path)
    if path is None:
        _CACHED_CORPUS = records
    return records
