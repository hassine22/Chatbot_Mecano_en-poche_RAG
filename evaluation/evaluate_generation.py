"""
Évalue le pipeline RAG de bout en bout sur `evaluation/questions.jsonl` (10 questions : 8 dans les notices + 2 hors-corpus) :
1. Pertinence du retrieval (hit-rate et rappel pour top-k=3 et top-k=5 sur les passages attendus)
2. Fidélité des réponses générées (présence des points clés attendus de la notice)
3. Reconnaissance de l'information absente sur les questions pièges (q09, q10)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mecano.pipeline import ask, retrieve_passages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "evaluation" / "questions.jsonl",
        help="Chemin du fichier questions.jsonl",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de passages utilisés pour la génération RAG (défaut: 5)",
    )
    parser.add_argument(
        "--no-multiquery",
        action="store_true",
        help="Désactiver la reformulation multi-requêtes lors de l'évaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "results" / "generation_evaluation.json",
        help="Fichier JSON de sortie contenant le rapport complet",
    )
    return parser.parse_args()


def load_eval_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    items = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        raise ValueError(f"Aucune question trouvée dans {path}")
    return items


def check_key_points(answer: str, key_points: list[str]) -> tuple[float, list[str], list[str]]:
    """
    Vérifie quels points_cles (chaînes ou expressions régulières séparées par '|')
    apparaissent dans la réponse générée (en tolérant l'impératif/infinitif français, ex: arrêter / arrêtez).
    Renvoie (score_fidelite, points_trouves, points_manquants).
    """
    if not key_points:
        return 1.0, [], []
    matched = []
    missing = []
    for pattern in key_points:
        # Tolérer les conjugaisons impératif/infinitif courantes (ex: arrêter -> arrêt(er|ez), couper -> coup(er|ez))
        flex_pattern = re.sub(r"(\w+)er\b", r"\1(?:er|ez|é)", pattern)
        if re.search(pattern, answer, flags=re.IGNORECASE) or re.search(flex_pattern, answer, flags=re.IGNORECASE):
            matched.append(pattern)
        else:
            missing.append(pattern)
    return len(matched) / len(key_points), matched, missing


def evaluate_retrieval_at_k(
    question: str,
    modele: str | None,
    categorie: str | None,
    expected_ids: list[str],
    k: int,
) -> dict[str, Any]:
    """Mesure le hit-rate et le rappel à top-k pour une question dotée de passages_attendus."""
    if not expected_ids:
        return {"retrieved_ids": [], "hit": None, "recall": None}
    results = retrieve_passages(question, modele=modele, top_k=k)
    ret_ids = [r.get("id") for r in results if r.get("id")]
    expected_set = set(expected_ids)
    hits = [pid for pid in ret_ids if pid in expected_set]
    return {
        "retrieved_ids": ret_ids,
        "hit": 1.0 if hits else 0.0,
        "recall": len(hits) / len(expected_set),
    }


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    args = parse_args()
    questions = load_eval_questions(args.questions)

    per_question = []
    in_corpus_fidelite = []
    out_of_corpus_correct = []
    hit_k3_list = []
    hit_k5_list = []
    rag_hit_list = []

    print("=" * 75)
    print(f"ÉVALUATION DU PIPELINE RAG COMPLET ({len(questions)} questions)")
    print("=" * 75)

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        modele = q.get("modele")
        q_type = q.get("type")
        expected_absent = bool(q.get("info_absente", False))
        expected_ids = q.get("passages_attendus", [])
        key_points = q.get("points_cles", [])

        # 1. Mesure de pertinence directe top-k=3 et top-k=5 (si passages_attendus non vide)
        ret_k3 = evaluate_retrieval_at_k(q_text, modele, q_type, expected_ids, k=3)
        ret_k5 = evaluate_retrieval_at_k(q_text, modele, q_type, expected_ids, k=5)
        if ret_k3["hit"] is not None:
            hit_k3_list.append(ret_k3["hit"])
            hit_k5_list.append(ret_k5["hit"])

        # 2. Exécution du pipeline RAG complet (Multi-Query + RRF + Gemini)
        res = ask(
            question=q_text,
            modele=modele,
            top_k=args.top_k,
            use_multi_query=not args.no_multiquery,
        )
        answer = res["reponse"]
        detected_absent = bool(res["info_absente"])
        rag_sources = [s.get("id") for s in res.get("sources", []) if s.get("id")]

        rag_hit = None
        if expected_ids:
            rag_hit = 1.0 if any(pid in set(expected_ids) for pid in rag_sources) else 0.0
            rag_hit_list.append(rag_hit)

        kp_score, kp_matched, kp_missing = check_key_points(answer, key_points)

        if expected_absent:
            absent_ok = detected_absent or kp_score > 0
            out_of_corpus_correct.append(1.0 if absent_ok else 0.0)
            status_str = "OK (Refus hors-notice)" if absent_ok else "ÉCHEC (Hallucination)"
        else:
            in_corpus_fidelite.append(kp_score)
            status_str = f"Fidélité={kp_score:.0%} ({len(kp_matched)}/{len(key_points)} points clés)"

        print(f"[{qid}] ({modele or 'Tous'} | {q_type}) {q_text}")
        print(f"   -> Statut : {status_str} | Sources RAG : {rag_sources}")
        if kp_missing and not expected_absent:
            print(f"   -> Points clés manquants : {kp_missing}")

        per_question.append({
            "id": qid,
            "question": q_text,
            "modele": modele,
            "type": q_type,
            "expected_info_absente": expected_absent,
            "detected_info_absente": detected_absent,
            "expected_passages": expected_ids,
            "retrieval_top3": ret_k3,
            "retrieval_top5": ret_k5,
            "rag_sources": rag_sources,
            "rag_hit": rag_hit,
            "key_points_score": round(kp_score, 3),
            "key_points_matched": kp_matched,
            "key_points_missing": kp_missing,
            "reponse": answer,
        })

    summary = {
        "total_questions": len(questions),
        "in_corpus_questions": len(in_corpus_fidelite),
        "out_of_corpus_questions": len(out_of_corpus_correct),
        "retrieval_hit_rate_top3": round(sum(hit_k3_list) / len(hit_k3_list), 3) if hit_k3_list else 0.0,
        "retrieval_hit_rate_top5": round(sum(hit_k5_list) / len(hit_k5_list), 3) if hit_k5_list else 0.0,
        "rag_hit_rate_with_multiquery_rrf": round(sum(rag_hit_list) / len(rag_hit_list), 3) if rag_hit_list else 0.0,
        "mean_fidelity_key_points": round(sum(in_corpus_fidelite) / len(in_corpus_fidelite), 3) if in_corpus_fidelite else 0.0,
        "absent_info_detection_accuracy": round(sum(out_of_corpus_correct) / len(out_of_corpus_correct), 3) if out_of_corpus_correct else 0.0,
    }

    print("\n" + "=" * 75)
    print("SYNTHÈSE DE L'ÉVALUATION")
    print("=" * 75)
    print(f"- Pertinence Retrieval direct (Hit-Rate @ top-k=3) : {summary['retrieval_hit_rate_top3']:.1%}")
    print(f"- Pertinence Retrieval direct (Hit-Rate @ top-k=5) : {summary['retrieval_hit_rate_top5']:.1%}")
    print(f"- Pertinence RAG Multi-Query + RRF (Hit-Rate)      : {summary['rag_hit_rate_with_multiquery_rrf']:.1%}")
    print(f"- Fidélité moyenne sur points clés (8 questions)   : {summary['mean_fidelity_key_points']:.1%}")
    print(f"- Détection information absente (2 questions pièges): {summary['absent_info_detection_accuracy']:.1%}")

    report = {
        "summary": summary,
        "questions": per_question,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nRapport complet sauvegardé dans : {args.output}")


if __name__ == "__main__":
    main()
