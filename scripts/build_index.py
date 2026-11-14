"""Generate embeddings and build the FAISS index.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus" / "output" / "all_corpus.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "index")
    parser.add_argument("--model", help="Nom du modèle Sentence-BERT")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size doit être supérieur ou égal à 1")
    from mecano.corpus_loader import load_corpus
    from mecano.embeddings import DEFAULT_MODEL, EmbeddingModel
    from mecano.index import build_index, save_index

    model_name = args.model or DEFAULT_MODEL
    records = load_corpus(args.corpus)
    model = EmbeddingModel(model_name)
    vectors = model.encode(
        [record["texte_embedding"] for record in records],
        batch_size=args.batch_size,
    )
    index = build_index(vectors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "corpus.faiss"
    metadata_path = args.output_dir / "corpus_metadata.json"
    save_index(index, index_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "dimension": model.dimension,
                "count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(records)} passages indexés dans {index_path}")


if __name__ == "__main__":
    main()
