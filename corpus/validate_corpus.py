"""
Vérifie qu'un corpus respecte le format commun (FORMAT_CORPUS.md).

Usage :
    python corpus/validate_corpus.py corpus/output/ecosport_corpus.jsonl
    python corpus/validate_corpus.py corpus/output/*.jsonl

Affiche les erreurs (bloquantes) et les avertissements (à vérifier).
Code de sortie 1 s'il y a au moins une erreur.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCHEMA = json.loads((Path(__file__).parent / "corpus_schema.json").read_text(encoding="utf-8"))
REQUIRED = SCHEMA["required"]
PROPS = SCHEMA["properties"]
CATEGORIES = set(PROPS["categorie"]["enum"])
LANGUES = set(PROPS["langue"]["enum"])
MIN_MOTS, MAX_MOTS = 12, 400


def check_file(path):
    errors, warnings = [], []
    slug = Path(path).name.replace("_corpus.jsonl", "")
    ids = Counter()
    stats = Counter()
    constants = {}

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return [f"{path} : fichier vide"], [], stats, 0

    for n, line in enumerate(lines, 1):
        where = f"ligne {n}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{where} : JSON invalide ({e})")
            continue
        where = f"ligne {n} ({r.get('id', '?')})"

        # champs manquants / en trop
        missing = [k for k in REQUIRED if k not in r]
        extra = [k for k in r if k not in PROPS]
        if missing:
            errors.append(f"{where} : champs manquants {missing}")
        if extra:
            errors.append(f"{where} : champs non prévus {extra}")
        if missing:
            continue

        # types
        for k in REQUIRED:
            t = PROPS[k].get("type")
            v = r[k]
            ok = True
            if t == "string":
                ok = isinstance(v, str)
            elif t == ["string", "null"]:
                ok = v is None or isinstance(v, str)
            elif t == "integer":
                ok = isinstance(v, int)
            elif t == "array":
                ok = isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) > 0
            if not ok:
                errors.append(f"{where} : '{k}' a un mauvais type ({type(v).__name__})")

        # valeurs
        if not re.fullmatch(rf"{re.escape(slug)}-\d{{3}}", str(r["id"])):
            errors.append(f"{where} : id '{r['id']}' doit être de la forme '{slug}-001'")
        ids[r["id"]] += 1
        if r["categorie"] not in CATEGORIES:
            errors.append(f"{where} : catégorie '{r['categorie']}' non autorisée")
        if r["langue"] not in LANGUES:
            errors.append(f"{where} : langue '{r['langue']}' non autorisée")
        if not re.fullmatch(r"\d{4}-\d{4}", str(r["annees"])):
            errors.append(f"{where} : annees '{r['annees']}' doit être au format AAAA-AAAA")
        if not str(r["source_url"]).startswith("http"):
            errors.append(f"{where} : source_url doit commencer par http")

        texte = r["texte"] if isinstance(r["texte"], str) else ""
        mots = len(texte.split())
        if isinstance(r["nb_mots"], int) and r["nb_mots"] != mots:
            errors.append(f"{where} : nb_mots={r['nb_mots']} mais le texte a {mots} mots")
        if mots < MIN_MOTS:
            warnings.append(f"{where} : passage très court ({mots} mots)")
        if mots > MAX_MOTS:
            warnings.append(f"{where} : passage trop long ({mots} mots), à découper")

        titre_attendu = r["section"] + (f" > {r['sous_section']}" if r["sous_section"] else "")
        if r["titre"] != titre_attendu:
            errors.append(f"{where} : titre doit être '{titre_attendu}'")
        debut = f"{r['marque']} {r['modele']} ({r['annees']}) – {r['titre']}\n"
        if not str(r["texte_embedding"]).startswith(debut):
            errors.append(f"{where} : texte_embedding doit commencer par '{debut.strip()}' + retour à la ligne")

        # champs qui doivent être identiques sur tout le fichier
        for k in ("marque", "modele", "phase", "annees", "langue", "document", "source_url"):
            constants.setdefault(k, set()).add(r[k])

        # traces de nettoyage incomplet
        if re.search(r"\b\d\.\d{1,2}\s*$", texte):
            warnings.append(f"{where} : le texte finit par un numéro de page ?")
        if re.search(r"[a-zà-ÿ]- [a-zà-ÿ]", texte):
            warnings.append(f"{where} : césure mal recollée ? « {re.search(r'.{15}[a-zà-ÿ]- [a-zà-ÿ].{10}', texte).group(0) if re.search(r'.{15}[a-zà-ÿ]- [a-zà-ÿ].{10}', texte) else ''} »")
        if re.search(r"[-]", texte):
            warnings.append(f"{where} : caractère de pictogramme non remplacé")

        stats[r["categorie"]] += 1

    for i, c in ids.items():
        if c > 1:
            errors.append(f"id '{i}' utilisé {c} fois")
    for k, vals in constants.items():
        if len(vals) > 1:
            errors.append(f"'{k}' doit être identique sur tout le fichier, valeurs trouvées : {sorted(map(str, vals))}")
    if stats and "voyant" not in stats:
        warnings.append("aucun passage 'voyant' : c'est pourtant le cœur du projet")
    return errors, warnings, stats, len(lines)


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        sys.exit(2)
    total_err = 0
    for path in files:
        errors, warnings, stats, n = check_file(path)
        print(f"\n=== {path} : {n} passages")
        for e in errors:
            print(f"  ERREUR  {e}")
        for w in warnings[:30]:
            print(f"  attention {w}")
        if len(warnings) > 30:
            print(f"  … et {len(warnings) - 30} autres avertissements")
        print("  Catégories : " + ", ".join(f"{k}={v}" for k, v in stats.most_common()))
        print(f"  Résultat : {len(errors)} erreur(s), {len(warnings)} avertissement(s)")
        total_err += len(errors)
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
