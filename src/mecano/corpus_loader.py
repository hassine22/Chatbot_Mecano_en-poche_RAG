"""
Chargement du corpus fusionné (corpus/output/all_corpus.jsonl).
"""
import json
from pathlib import Path
from mecano.config import get_project_root, load_config

_CACHED_CORPUS = None


def load_all_corpus(path: str = None) -> list[dict]:
    """Charge le corpus JSONL en mémoire (avec cache)."""
    global _CACHED_CORPUS
    if _CACHED_CORPUS is not None and path is None:
        return _CACHED_CORPUS
        
    if path is None:
        cfg = load_config()
        rel_path = cfg.get("corpus", {}).get("path", "corpus/output/all_corpus.jsonl")
        p = get_project_root() / rel_path
    else:
        p = Path(path)
        if not p.is_absolute():
            p = get_project_root() / p
            
    if not p.exists():
        raise FileNotFoundError(f"Corpus introuvable : {p}")
        
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    if path is None:
        _CACHED_CORPUS = records
    return records
