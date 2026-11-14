"""Evaluate top-k semantic retrieval against manually relevant passage IDs."""

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
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "evaluation" / "retrieval_questions.jsonl",
    )
    parser.add_argument("--index-dir", type=Path, default=ROOT / "index")
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        help="Valeur de k à évaluer (par défaut : 3 et 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "results" / "retrieval_topk.json",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de questions introuvable : {path}")
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalide dans {path}, ligne {line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"La ligne {line_number} doit être un objet JSON")
            for field in ("id", "question", "modele", "categorie", "relevant_ids"):
                if field not in item:
                    raise ValueError(f"Champ '{field}' absent dans la question ligne {line_number}")
            if not all(
                isinstance(item[field], str) and item[field].strip()
                for field in ("id", "question", "modele", "categorie")
            ):
                raise ValueError(f"Champs texte vides ou invalides ligne {line_number}")
            relevant_ids = item["relevant_ids"]
            if (
                not isinstance(relevant_ids, list)
                or not relevant_ids
                or not all(isinstance(value, str) and value for value in relevant_ids)
            ):
                raise ValueError(
                    f"'relevant_ids' doit être une liste non vide d'IDs ligne {line_number}"
                )
            questions.append(item)
    if not questions:
        raise ValueError(f"Aucune question de test dans {path}")
    return questions


def score_results(
    results: list[dict[str, Any]],
    relevant_ids: set[str],
    top_k: int,
) -> dict[str, float]:
    ranks = [
        rank
        for rank, record in enumerate(results, 1)
        if record["id"] in relevant_ids
    ]
    true_positives = len(ranks)
    precision = true_positives / top_k
    recall = true_positives / len(relevant_ids)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": 1 / ranks[0] if ranks else 0.0,
        "hit_rate": 1.0 if ranks else 0.0,
    }


def main() -> None:
    args = parse_args()
    top_ks = args.top_k or [3, 5]
    if any(isinstance(k, bool) or k < 1 for k in top_ks):
        raise ValueError("Chaque valeur de --top-k doit être supérieure ou égale à 1")
    top_ks = list(dict.fromkeys(top_ks))
    questions = load_questions(args.questions)

    metadata_path = args.index_dir / "corpus_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Métadonnées d'index introuvables : {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Métadonnées JSON invalides : {metadata_path}") from exc
    records = metadata.get("records")
    if not isinstance(records, list) or not records or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Les métadonnées ne contiennent aucun passage valide : {metadata_path}")

    records_by_id = {record.get("id"): record for record in records}
    if len(records_by_id) != len(records) or None in records_by_id:
        raise ValueError("Les métadonnées doivent contenir des IDs de passage uniques")

    for question in questions:
        for relevant_id in question["relevant_ids"]:
            record = records_by_id.get(relevant_id)
            if record is None:
                raise ValueError(
                    f"{question['id']} référence un passage absent de l'index : {relevant_id}"
                )
            if (
                record.get("modele", "").casefold() != question["modele"].casefold()
                or record.get("categorie", "").casefold()
                != question["categorie"].casefold()
            ):
                raise ValueError(
                    f"{question['id']} : le passage {relevant_id} ne correspond pas "
                    "aux filtres modèle/catégorie"
                )

    from mecano.embeddings import DEFAULT_MODEL, EmbeddingModel
    from mecano.index import load_index
    from mecano.retriever import VectorRetriever

    model_name = metadata.get("model", DEFAULT_MODEL)
    index = load_index(args.index_dir / "corpus.faiss")
    retriever = VectorRetriever(index, EmbeddingModel(model_name), records)
    max_k = max(top_ks)
    per_question: list[dict[str, Any]] = []
    totals = {
        k: {metric: 0.0 for metric in ("precision", "recall", "f1", "mrr", "hit_rate")}
        for k in top_ks
    }

    for question in questions:
        results = retriever.search(
            question["question"],
            top_k=max_k,
            modele=question["modele"],
            categorie=question["categorie"],
        )
        question_result: dict[str, Any] = {
            "id": question["id"],
            "question": question["question"],
            "modele": question["modele"],
            "categorie": question["categorie"],
            "relevant_ids": question["relevant_ids"],
            "retrieved_ids": [record["id"] for record in results],
            "metrics": {},
        }
        for k in top_ks:
            metrics = score_results(results[:k], set(question["relevant_ids"]), k)
            question_result["metrics"][str(k)] = metrics
            for metric, value in metrics.items():
                totals[k][metric] += value
        per_question.append(question_result)

    averages = {
        str(k): {
            metric: value / len(questions)
            for metric, value in totals[k].items()
        }
        for k in top_ks
    }
    best_k = max(
        top_ks,
        key=lambda k: (
            averages[str(k)]["f1"],
            averages[str(k)]["mrr"],
            -k,
        ),
    )
    report = {
        "index_model": model_name,
        "question_count": len(questions),
        "top_k_values": top_ks,
        "averages": averages,
        "best_top_k": best_k,
        "questions": per_question,
    }

    print(f"Questions évaluées : {len(questions)}")
    for k in top_ks:
        metrics = averages[str(k)]
        print(
            f"top-k={k}: précision={metrics['precision']:.3f}, "
            f"rappel={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
            f"MRR={metrics['mrr']:.3f}, hit-rate={metrics['hit_rate']:.3f}"
        )
    print(f"Meilleur top-k (F1, puis MRR, puis k plus petit) : {best_k}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Résultats détaillés : {args.output}")


if __name__ == "__main__":
    main()
