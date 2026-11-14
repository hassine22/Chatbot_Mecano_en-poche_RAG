"""
Pose une question au chatbot Mécano de poche en ligne de commande.
Supporte le filtrage, le mode Multi-Query + RRF et la comparaison de gain.
"""
import argparse
import sys
import os

# Assure que src/ est dans le sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mecano.pipeline import ask


def main():
    parser = argparse.ArgumentParser(
        description="Mécano de poche - Assistant RAG pour notices automobiles"
    )
    parser.add_argument(
        "question",
        type=str,
        help="La question à poser (ex: 'Que faire si le voyant moteur s'allume sur ma i10 ?')"
    )
    parser.add_argument(
        "--modele",
        type=str,
        default=None,
        help="Filtrer par modèle (ex: 'clio4', 'i10', 'ecosport', 'picanto')"
    )
    parser.add_argument(
        "--categorie",
        type=str,
        default=None,
        help="Filtrer par catégorie (ex: 'voyant', 'entretien', 'pneus_roues')"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Nombre de passages à conserver (défaut: 5)"
    )
    parser.add_argument(
        "--no-multiquery",
        action="store_true",
        help="Désactiver la reformulation multi-requêtes + fusion RRF"
    )
    parser.add_argument(
        "--compare-multiquery",
        action="store_true",
        help="Comparer la recherche directe et avec Multi-Query + RRF pour mesurer le gain"
    )

    args = parser.parse_args()

    # Reconfigurer la sortie console en UTF-8 pour Windows
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("\n" + "=" * 70)
    print("MÉCANO DE POCHE — QUESTION DU CONDUCTEUR :")
    print(f"« {args.question} »")
    if args.modele or args.categorie:
        print(f"Filtres : modèle={args.modele or 'tous'} | catégorie={args.categorie or 'toutes'}")
    print("=" * 70)

    if args.compare_multiquery:
        print("\n[TEST COMPARATIF MULTI-QUERY + RRF]")
        print("-" * 50)
        print("1. Recherche directe (Baseline sans reformulation) :")
        res_baseline = ask(
            question=args.question,
            modele=args.modele,
            categorie=args.categorie,
            top_k=args.top_k or 5,
            use_multi_query=False
        )
        for i, s in enumerate(res_baseline["sources"], 1):
            print(f"   [{i}] ID: {s['id']} | Pages: {', '.join(s['pages'])} | Score: {s.get('score')} | {s['titre']}")

        print("\n2. Recherche avec Multi-Query + Fusion RRF :")
        res_mq = ask(
            question=args.question,
            modele=args.modele,
            categorie=args.categorie,
            top_k=args.top_k or 5,
            use_multi_query=True
        )
        print("   Reformulations générées :")
        for ref in res_mq.get("reformulations", []):
            print(f"   - « {ref} »")
        print("   Passages après fusion RRF :")
        for i, s in enumerate(res_mq["sources"], 1):
            print(f"   [{i}] ID: {s['id']} | Pages: {', '.join(s['pages'])} | Score RRF: {s.get('score')} | {s['titre']}")

        print("\n" + "=" * 70)
        print("RÉPONSE DU CHATBOT (AVEC MULTI-QUERY + RRF) :")
        print("=" * 70)
        print(res_mq["answer"])
        print("=" * 70 + "\n")
        return

    # Exécution standard
    res = ask(
        question=args.question,
        modele=args.modele,
        categorie=args.categorie,
        top_k=args.top_k,
        use_multi_query=not args.no_multiquery
    )

    if res.get("multi_query_used") and res.get("reformulations"):
        print("\n[Reformulations multi-requêtes (Gemini)] :")
        for ref in res["reformulations"]:
            print(f"  - {ref}")

    print(f"\n[Sources retenues ({len(res['sources'])} passages)] :")
    for i, s in enumerate(res["sources"], 1):
        score_info = f"Score RRF: {s['score']}" if res.get("multi_query_used") else f"Score: {s.get('score')}"
        print(f"  [{i}] {s['marque']} {s['modele']} | Page(s) {', '.join(s['pages'])} | {score_info} | {s['titre']} (ID: {s['id']})")

    print("\n" + "=" * 70)
    print("RÉPONSE :")
    print("=" * 70)
    print(res["answer"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
