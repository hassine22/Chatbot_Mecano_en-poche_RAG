"""
Fusionne tous les corpus validés en un seul fichier.

Usage : python corpus/merge_corpus.py
Sortie : corpus/output/all_corpus.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

from validate_corpus import check_file

OUT_DIR = Path(__file__).parent / "output"


def main():
    files = sorted(p for p in OUT_DIR.glob("*_corpus.jsonl") if p.name != "all_corpus.jsonl")
    if not files:
        sys.exit("Aucun corpus trouvé dans corpus/output/")

    records, seen = [], set()
    for p in files:
        errors, _, _, n = check_file(p)
        if errors:
            sys.exit(f"{p.name} contient {len(errors)} erreur(s) : lancez validate_corpus.py dessus d'abord.")
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in seen:
                sys.exit(f"id en double entre fichiers : {r['id']}")
            seen.add(r["id"])
            records.append(r)
        print(f"{p.name:35s} {n:4d} passages")

    out = OUT_DIR / "all_corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    par_modele = Counter(f"{r['marque']} {r['modele']}" for r in records)
    print(f"\n{len(records)} passages au total → {out}")
    for k, v in par_modele.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
