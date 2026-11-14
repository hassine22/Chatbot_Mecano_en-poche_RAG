"""Compare top-k semantic search results and filter by vehicle/category."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question à rechercher dans le corpus")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "index")
    parser.add_argument(
        "--embedding-model",
        help="Modèle utilisé pour construire l'index (par défaut : métadonnées)",
    )
    parser.add_argument("--modele", help="Filtrer par véhicule, par ex. « Clio 4 »")
    parser.add_argument("--categorie", help="Filtrer par catégorie, par ex. « voyant »")
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        help="Valeur de k à afficher (par défaut : comparer 3 et 5)",
    )
    parser.add_argument(
        "--relevant-id",
        action="append",
        default=[],
        help="ID de passage pertinent, répétable pour calculer précision/rappel",
    )
    return parser.parse_args()


def _print_results(
    results: list[dict[str, Any]],
    top_k: int,
    relevant_ids: set[str],
) -> tuple[float, float, float, float]:
    print(f"\n--- top-k = {top_k} ---")
    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. [{result['id']}] score={result['score']:.4f} "
            f"{result['modele']} · {result['categorie']} · "
            f"p. {', '.join(result['pages'])}"
        )
        print(f"   {result['titre']}")
        print(f"   {result['texte'].replace(chr(10), ' ')}")

    if not relevant_ids:
        return 0.0, 0.0, 0.0, 0.0
    hits = [
        rank
        for rank, result in enumerate(results, 1)
        if result["id"] in relevant_ids
    ]
    precision = len(hits) / top_k
    recall = len(hits) / len(relevant_ids)
    reciprocal_rank = 1 / hits[0] if hits else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        f"   Précision@{top_k}: {precision:.3f} · "
        f"Rappel@{top_k}: {recall:.3f} · MRR@{top_k}: {reciprocal_rank:.3f} · "
        f"F1@{top_k}: {f1:.3f}"
    )
    return precision, recall, reciprocal_rank, f1


def main() -> None:
    args = parse_args()
    from mecano.embeddings import DEFAULT_MODEL, EmbeddingModel
    from mecano.index import load_index
    from mecano.retriever import VectorRetriever

    metadata_path = args.index_dir / "corpus_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Métadonnées d'index introuvables : {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Aucun passage dans les métadonnées : {metadata_path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Les métadonnées contiennent un passage invalide : {metadata_path}")

    indexed_model = metadata.get("model", DEFAULT_MODEL)
    model_name = args.embedding_model or indexed_model
    if model_name != indexed_model:
        raise ValueError(
            "Le modèle d'embeddings doit être identique à celui de l'index "
            f"({indexed_model!r})"
        )
    index = load_index(args.index_dir / "corpus.faiss")
    embedder = EmbeddingModel(model_name)
    retriever = VectorRetriever(index, embedder, records)

    top_ks = args.top_k or [3, 5]
    if any(isinstance(k, bool) or k < 1 for k in top_ks):
        raise ValueError("Chaque valeur de --top-k doit être supérieure ou égale à 1")
    top_ks = list(dict.fromkeys(top_ks))
    relevant_ids = set(args.relevant_id)

    print(f"Question : {args.question}")
    print(f"Filtres : véhicule={args.modele or 'tous'}, catégorie={args.categorie or 'toutes'}")
    metrics = {}
    for top_k in top_ks:
        results = retriever.search(
            args.question,
            top_k=top_k,
            modele=args.modele,
            categorie=args.categorie,
        )
        metrics[top_k] = _print_results(results, top_k, relevant_ids)

    if relevant_ids and len(metrics) > 1:
        best_k = max(metrics, key=lambda k: (metrics[k][3], metrics[k][2]))
        print(
            f"\nMeilleur top-k pour cette question selon F1@k, "
            f"puis MRR@k : {best_k}"
        )
    elif not relevant_ids:
        print(
            "\nPour choisir k selon la pertinence, fournissez les passages attendus "
            "avec --relevant-id. Sans référence, les scores seuls ne permettent pas "
            "de conclure quel k est meilleur."
        )


if __name__ == "__main__":
    main()
