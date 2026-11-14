"""
Backend de DÉMONSTRATION pour développer l'interface avant que le vrai pipeline
(embeddings + FAISS + LLM, fichiers src/mecano/) soit prêt.

- Recherche : BM25 par mots-clés (pas d'embeddings), avec une petite traduction
  français → anglais des mots courants, car 3 notices sur 4 sont en anglais.
- Génération : pas de LLM, on renvoie l'extrait le plus pertinent.

Il respecte EXACTEMENT le même contrat que src/mecano/pipeline.py (voir CONTRAT
ci-dessous) : quand le vrai pipeline sera prêt, l'interface l'utilisera
automatiquement et ce fichier ne servira plus.

CONTRAT
-------
ask(question, modele=None, categories=None, top_k=5) -> dict
    question   : str              texte de l'utilisateur
    modele     : str | None       valeur du champ "modele" du corpus (ex. "Clio 4"), None = toutes
    categories : list[str] | None valeurs du champ "categorie", None = toutes
    top_k      : int              nombre de passages à récupérer
  renvoie {
    "reponse":      str           texte à afficher (markdown accepté)
    "info_absente": bool          True si les sources ne permettent pas de répondre
    "sources": [                  passages utilisés, du plus pertinent au moins pertinent
        { ...tous les champs du passage du corpus..., "score": float entre 0 et 1 }
    ]
  }
"""
import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "output" / "all_corpus.jsonl"
SEUIL_ABSENT = 3.0  # score BM25 brut en dessous duquel on considère l'information absente
COUVERTURE_MIN = 0.5  # part minimale des mots de la question retrouvés dans le meilleur passage
MOTS_IGNORES = {"clio", "ecosport", "picanto", "i10", "renault", "ford", "kia", "hyundai",
                "code", "voiture", "probleme", "question", "aide"}

STOP = set("""
le la les un une des du de d l au aux et ou a à en dans sur pour par avec sans ce ces cet cette
mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs je tu il elle on nous vous ils elles
que qui quoi quand comment pourquoi est sont suis es etre être fait faire faut dois doit peut peux
y ne pas plus quel quelle quels quelles si se s qu c j n m t ai as avez ont alors donc mais
the a an of to and or in on for with is are be it this that my your what how why when do does can
voiture vehicule véhicule car vehicle
""".split())

# Traduction minimale français → anglais pour chercher dans les notices anglaises
FR_EN = {
    "voyant": "warning light lamp indicator", "temoin": "warning light lamp indicator",
    "allume": "illuminates on comes", "clignote": "blinks flashing", "eteint": "off",
    "huile": "oil", "pression": "pressure", "moteur": "engine", "batterie": "battery",
    "frein": "brake", "freins": "brake brakes", "freinage": "braking brake", "pneu": "tire tyre",
    "pneus": "tires tyres", "crevaison": "flat tire puncture", "roue": "wheel", "roues": "wheels",
    "fusible": "fuse", "fusibles": "fuses", "carburant": "fuel", "essence": "fuel gasoline petrol",
    "gazole": "diesel", "reservoir": "tank", "temperature": "temperature coolant",
    "refroidissement": "coolant cooling", "liquide": "fluid", "vidange": "oil change",
    "entretien": "maintenance service", "demarrage": "starting start", "demarrer": "start",
    "demarre": "start starts", "cle": "key", "direction": "steering", "volant": "steering wheel",
    "ceinture": "seat belt", "airbag": "airbag", "remorquage": "towing tow", "remorquer": "tow",
    "boite": "transmission", "automatique": "automatic", "vitesse": "speed gear",
    "phare": "headlamp headlight", "ampoule": "bulb", "essuie": "wiper", "climatisation": "air conditioning",
    "surchauffe": "overheating overheat", "chauffe": "overheating hot", "fumee": "smoke",
    "bruit": "noise", "panne": "malfunction failure breakdown", "defaut": "malfunction fault",
    "message": "message", "affiche": "displayed display", "tableau": "cluster dashboard",
    "bord": "instrument", "gonfler": "inflate", "gonflage": "inflation inflator",
    "anomalie": "malfunction", "abs": "abs", "esp": "esc stability", "antipollution": "emission",
    "cigare": "cigarette lighter power outlet", "changer": "change changing replace",
    "remplacer": "replace replacing", "crevee": "flat tire puncture", "verifier": "check",
    "niveau": "level", "reste": "remains stays", "grave": "serious", "plat": "discharged dead flat",
    "cables": "cables jumper", "cable": "cable jumper", "clignotant": "turn signal", "plaquette": "pad pads", "secours": "spare",
    "allume-cigare": "power outlet cigarette lighter", "prise": "outlet socket",
}


def _norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _tokens(t):
    return [w for w in re.findall(r"[a-z0-9]+", _norm(t)) if w not in STOP and len(w) > 1]


@lru_cache(maxsize=1)
def _index():
    passages = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    docs = [Counter(_tokens(p["titre"] + " " + p["titre"] + " " + p["texte"])) for p in passages]
    df = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    lens = [sum(d.values()) for d in docs]
    return passages, docs, idf, lens, sum(lens) / n


def modeles_disponibles():
    """Liste des voitures du corpus : [(marque, modele, annees, langue, nb_passages)]."""
    passages = _index()[0]
    c = Counter((p["marque"], p["modele"], p["annees"], p["langue"]) for p in passages)
    return [(m, mo, a, l, n) for (m, mo, a, l), n in sorted(c.items())]


def categories_disponibles():
    return sorted({p["categorie"] for p in _index()[0]})


def _bm25(query_tokens, k1=1.5, b=0.75):
    passages, docs, idf, lens, avg = _index()
    scores = []
    for i, d in enumerate(docs):
        s = 0.0
        for w in query_tokens:
            f = d.get(w)
            if f:
                s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
        scores.append(s)
    return scores


def search(question, modele=None, categories=None, top_k=5):
    toks = _tokens(question)
    extra = []
    for t in toks:
        if t in FR_EN:
            extra += FR_EN[t].split()
    query = toks + extra
    passages = _index()[0]
    scores = _bm25(query)
    ranked = []
    for i in sorted(range(len(passages)), key=lambda i: -scores[i]):
        p = passages[i]
        if modele and p["modele"] != modele:
            continue
        if categories and p["categorie"] not in categories:
            continue
        if scores[i] <= 0:
            break
        ranked.append((scores[i], p))
        if len(ranked) == top_k:
            break
    return ranked


def _couverture(question, passage):
    """Part des mots de la question (ou de leur traduction) présents dans le passage."""
    mots = [t for t in _tokens(question) if t not in MOTS_IGNORES]
    if not mots:
        return 0.0
    contenu = set(_tokens(passage["titre"] + " " + passage["texte"]))
    trouves = sum(1 for t in mots if t in contenu or set(FR_EN.get(t, "").split()) & contenu)
    return trouves / len(mots)


def ask(question, modele=None, categories=None, top_k=5):
    ranked = search(question, modele, categories, top_k)
    best = ranked[0][0] if ranked else 0.0
    sources = [dict(p, score=round(s / best, 3) if best else 0.0) for s, p in ranked]
    if best < SEUIL_ABSENT or _couverture(question, ranked[0][1]) < COUVERTURE_MIN:
        return {
            "reponse": "Je n'ai pas trouvé cette information dans les notices disponibles. "
                       "Pour un diagnostic fiable, consultez un garagiste ou le réseau de la marque.",
            "info_absente": True,
            "sources": sources,
        }
    top = sources[0]
    extrait = top["texte"].strip()
    if len(extrait) > 900:
        extrait = extrait[:900].rsplit(" ", 1)[0] + "…"
    pages = ", ".join(top["pages"])
    reponse = (
        f"*Mode démo (sans LLM) : voici l'extrait le plus pertinent de la notice "
        f"{top['marque']} {top['modele']}, page {pages}.*\n\n"
        f"**{top['titre']}**\n\n" + extrait.replace("\n", "  \n")
    )
    return {"reponse": reponse, "info_absente": False, "sources": sources}
