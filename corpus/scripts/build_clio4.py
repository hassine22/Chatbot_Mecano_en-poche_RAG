"""
Construction du corpus RAG « Mécano de poche » pour la Renault Clio 4 (phase 2).

Entrée  : notice d'utilisation Renault Clio 4 (PDF, édition 06/2017, NU 1189-4)
Sorties : clio4_corpus.jsonl (1 passage par ligne, prêt pour les embeddings)
          clio4_corpus.md    (version lisible pour relecture humaine)

Usage (depuis la racine du dépôt) :
    python corpus/scripts/build_clio4.py data/pdf/Clio4.pdf corpus/output

------------------------------------------------------------------------------
POUR ADAPTER CE SCRIPT À UNE AUTRE VOITURE
  1. Copiez-le en corpus/scripts/build_<slug>.py
  2. Modifiez META (marque, modèle, années, langue, document, source_url)
  3. Modifiez KEEP : les pages imprimées de VOTRE notice à garder
     (voir FORMAT_CORPUS.md §5 pour savoir quoi garder)
  4. Changez SLUG (ex. "ecosport") : il sert aux ids et aux noms de fichiers
  5. Les fonctions de tableaux (anomalies, messages, fusibles) dépendent de la
     mise en page Renault : adaptez-les ou supprimez-les selon votre notice
  6. Vérifiez le .md produit, puis lancez corpus/validate_corpus.py
------------------------------------------------------------------------------
"""
import json
import re
import sys
from pathlib import Path

import pymupdf  # pip install pymupdf

# ---------------------------------------------------------------- métadonnées
SLUG = "clio4"  # préfixe des ids et des noms de fichiers (voir FORMAT_CORPUS.md)
META = {
    "marque": "Renault",
    "modele": "Clio 4",
    "phase": "phase 2",
    "annees": "2016-2020",
    "langue": "fr",
    "document": "Notice d'utilisation Renault Clio (NU 1189-4, édition 06/2017)",
    "source_url": "https://www.user-manual.renault.com/fr/notice/9378/index",
}

# ------------------------------------------------- pages retenues (étiquettes)
# (chapitre, première page, dernière page, catégorie)
KEEP = [
    ("1 - Faites connaissance avec votre véhicule", "1.51", "1.51", "direction"),
    ("1 - Faites connaissance avec votre véhicule", "1.56", "1.62", "voyant"),
    ("1 - Faites connaissance avec votre véhicule", "1.63", "1.64", "message_tableau_de_bord"),
    ("1 - Faites connaissance avec votre véhicule", "1.65", "1.69", "ordinateur_tableau"),
    ("1 - Faites connaissance avec votre véhicule", "1.70", "1.72", "messages_tableau"),
    ("1 - Faites connaissance avec votre véhicule", "1.85", "1.87", "carburant"),
    ("2 - La conduite", "2.3", "2.11", "demarrage_moteur"),
    ("2 - La conduite", "2.18", "2.18", "entretien"),
    ("2 - La conduite", "2.20", "2.26", "aides_conduite"),
    ("2 - La conduite", "2.44", "2.46", "boite_automatique"),
    ("4 - Entretien", "4.2", "4.12", "entretien"),
    ("5 - Conseils pratiques", "5.2", "5.15", "pneus_roues"),
    ("5 - Conseils pratiques", "5.27", "5.27", "fusibles"),
    ("5 - Conseils pratiques", "5.29", "5.29", "fusibles_tableau"),
    ("5 - Conseils pratiques", "5.31", "5.31", "fusibles_tableau"),
    ("5 - Conseils pratiques", "5.32", "5.34", "electricite_batterie"),
    ("5 - Conseils pratiques", "5.39", "5.40", "remorquage"),
    ("5 - Conseils pratiques", "5.41", "5.46", "anomalie"),
    ("6 - Caractéristiques techniques", "6.2", "6.3", "identification"),
    ("6 - Caractéristiques techniques", "6.6", "6.6", "caracteristiques_moteur"),
]
TABLE_PAGES = {"5.41", "5.42", "5.43", "5.44", "5.45", "5.46"}

# --------------------------------------------------------------- nettoyage
NOISE = [
    re.compile(r"L.affichage des informations décrites ci-après DÉPEND", re.I),
]
KEEP_HYPHEN_NEXT = {"vous", "le", "la", "les", "il", "elle", "ils", "en", "y", "lui",
                    "nous", "eux", "moi", "même", "là", "ci", "t-il", "dessus", "dessous"}
KEEP_HYPHEN_PREV = {"essuie", "lave", "lève", "porte", "après", "arrière", "demi", "mi", "semi",
                    "sous", "non", "garde", "pare", "avant", "auto", "contre", "court"}
MAX_WORDS = 320
# mots coupés par une espace dans le PDF source
FIXES = {"assu rez": "assurez", "vites se": "vitesse", "Régu lateur": "Régulateur",
         "laté ralement": "latéralement", "para gra phe": "paragraphe",
         "paragra phe": "paragraphe", "sytème": "système", "alter nance": "alternance",
         "accom pagnés": "accompagnés", "routiè re": "routière", "acompagné": "accompagné"}


def label_index(doc):
    """Associe l'étiquette imprimée (ex. '1.58') à l'index de page PDF."""
    idx = {}
    for i, page in enumerate(doc):
        found = re.findall(r"^\s*(\d\.\d{1,2})\s*$", page.get_text(), re.M)
        if found:
            idx[found[-1]] = i
    return idx


def label_range(idx, first, last):
    a, b = idx[first], idx[last]
    labels = {v: k for k, v in idx.items()}
    return [(i, labels.get(i, "?")) for i in range(a, b + 1)]


def join_lines(lines):
    """Recolle les lignes d'un paragraphe en gérant les césures."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not out:
            out = ln
        elif out.endswith("-"):
            nxt = re.match(r"[\wÀ-ÿ’'-]+", ln)
            word = nxt.group(0).lower() if nxt else ""
            prev = re.search(r"([\wÀ-ÿ]+)-$", out)
            prev = prev.group(1).lower() if prev else ""
            if word in KEEP_HYPHEN_NEXT or prev in KEEP_HYPHEN_PREV or not ln[:1].islower():
                out += ln
            else:
                out = out[:-1] + ln
        else:
            out += " " + ln
    out = re.sub(r"[\ue000-\uf8ff®]", "", out)
    out = re.sub(r"\b(" + "|".join(KEEP_HYPHEN_PREV) + r")- (?=[a-zà-ÿ])", r"\1-", out)
    for bad, good in FIXES.items():
        out = out.replace(bad, good)
    out = out.replace(" / ", " / ")
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:)])", r"\1", out)
    out = out.replace("( ", "(")
    return out.strip()


ICONS = {("NuP", "®"): "arrêt impératif STOP", ("NuP", "©"): "d’alerte (clé)"}


def is_icon(span):
    return span["font"].startswith(("NuN", "NuP"))


def icon_key(span):
    return (span["font"][:3], span["text"].strip())


def build_icon_map(doc, idx):
    """Associe chaque pictogramme de témoin au titre en gras placé à sa droite."""
    for lab in ["1.56", "1.57", "1.58", "1.59", "1.60", "1.61"]:
        page = doc[idx[lab]]
        icons, bold = [], []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for ln in b["lines"]:
                for sp in ln["spans"]:
                    if is_icon(sp) and sp["text"].strip():
                        icons.append((sp["bbox"], icon_key(sp)))
                txt = "".join(sp["text"] for sp in ln["spans"] if not is_icon(sp)).strip()
                if txt and all(sp["flags"] & 16 for sp in ln["spans"] if not is_icon(sp) and sp["text"].strip()):
                    bold.append((ln["bbox"], txt))
        for (ix0, iy0, ix1, iy1), key in icons:
            if key in ICONS:
                continue
            first = sorted([l for l in bold if abs(l[0][1] - iy0) < 14 and ix1 - 8 <= l[0][0] <= ix1 + 60],
                           key=lambda l: abs(l[0][1] - iy0))
            if not first:
                continue
            (x0, y0, _, _), t = first[0]
            parts, last_y = [t], y0
            for (bx0, by0, _, _), bt in sorted(bold, key=lambda r: r[0][1]):
                if 0 < by0 - last_y <= 12 and ix0 - 6 <= bx0 <= ix0 + 190:
                    parts.append(bt)
                    last_y = by0
            name = join_lines(parts)
            name = re.sub(r"^Témoins? (d’|d'|de |des |du )?", "", name).strip()
            if 3 < len(name) < 90:
                ICONS[key] = name


def page_lines(page):
    """Lignes de texte utiles d'une page : (x0, y0, x1, texte, gras, taille)."""
    res = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            words = [s for s in spans if not is_icon(s)]
            if not words:  # ligne ne contenant qu'un pictogramme
                continue
            parts = []
            for s in ln["spans"]:
                if is_icon(s):
                    name = ICONS.get(icon_key(s))
                    if name and s is not spans[0]:
                        parts.append(f"[voyant {name}]")
                else:
                    parts.append(s["text"])
            text = "".join(parts)
            x0, y0, x1, _ = ln["bbox"]
            size = max(s["size"] for s in words)
            bold = all(s["flags"] & 16 for s in words)
            t = text.strip()
            if len(t) <= 2 and (bold or size >= 10 or not t.islower()):
                continue  # repères de figure (A, B, 1, 2…)
            if re.fullmatch(r"\d\.\d{1,2}", t):  # numéro de page
                continue
            if any(p.search(t) for p in NOISE):
                continue
            res.append((x0, y0, x1, text, bold, size))
    return res


def column(x0, x1):
    if x0 < 195:
        return 0
    if x0 < 375:
        return 1
    return 2


def page_paragraphs(page):
    """Renvoie (titre, [ (texte, est_sous_titre) ]) dans l'ordre de lecture."""
    lines = page_lines(page)
    title = None
    body = []
    for x0, y0, x1, text, bold, size in lines:
        if y0 < 25 and size >= 10.5 and bold:
            title = (title + " " if title else "") + text.strip()
        else:
            body.append((column(x0, x1), y0, x0, text, bold, size))
    body.sort(key=lambda r: (r[0], round(r[1]), r[2]))
    # fusion des fragments situés sur la même ligne (texte justifié)
    fused = []
    for r in body:
        if fused and fused[-1][0] == r[0] and abs(fused[-1][1] - r[1]) < 1.5:
            col, y0, x0, text, bold, size = fused[-1]
            fused[-1] = (col, y0, x0, text.rstrip() + " " + r[3].lstrip(), bold and r[4], size)
        else:
            fused.append(r)
    body = fused

    paras, cur, prev, gaps = [], [], None, []
    for col, y0, x0, text, bold, size in body:
        new = (prev is None or col != prev[0] or y0 - prev[1] > 12.5
               or bold != prev[2])
        if new and cur:
            paras.append(cur)
            cur = []
        if new:
            # grand espace vertical dans la même colonne = encadré / nouvel item
            gaps.append(prev is not None and col == prev[0] and y0 - prev[1] > 24)
        cur.append((text, bold, size))
        prev = (col, y0, bold)
    if cur:
        paras.append(cur)

    out = []
    for p, gap in zip(paras, gaps):
        txt = join_lines([t for t, _, _ in p])
        is_bold = all(b for _, b, _ in p)
        is_sub = (is_bold and len(txt) < 90 and len(txt.split()) >= 2
                  and not re.search(r"[.,;:\-]$", txt))
        if txt:
            out.append((txt, is_sub, gap))
    return title, out


def clean_title(t):
    t = re.sub(r"\(\s*\d+\s*/\s*\d+\s*\)", "", t or "").strip(" :")
    t = re.sub(r"\s+", " ", t)
    return t


# ----------------------------------------------- tableau « anomalies »
def parse_anomaly_page(page, state):
    """Tableau à 3 colonnes : symptôme | causes possibles | que faire."""
    lines = page_lines(page)
    hdr = [l for l in lines if l[3].strip().upper() == "CAUSES POSSIBLES"]
    c1 = hdr[0][0] - 5 if hdr else 195
    act = [l for l in lines if l[3].strip().upper() == "QUE FAIRE"]
    c2 = act[0][0] - 5 if act else 365
    rows = []  # (y, col, text, bold)
    title = None
    for x0, y0, x1, text, bold, size in lines:
        t = text.strip()
        if y0 < 25 and size >= 10.5:
            title = t
            continue
        if t.upper() in ("CAUSES POSSIBLES", "QUE FAIRE"):
            continue
        if re.match(r"(Les conseils suivants|possible un Représentant de la marque)", t):
            continue
        if x0 < c1 and x1 > c2 + 60:  # remarque pleine largeur sous le tableau
            col = 3
        else:
            col = 0 if x0 < c1 else (1 if x0 < c2 else 2)
        rows.append((y0, col, t, bold, x0))
    rows.sort(key=lambda r: (round(r[0]), r[1], r[4]))
    notes = [t for y, c, t, b, x in rows if c == 3]
    rows = [(y, c, t, b, x) for y, c, t, b, x in rows if c != 3]

    # 1) symptômes et catégories (colonne 0) avec leur position verticale
    col0 = [(y, t, b, x) for y, c, t, b, x in rows if c == 0]
    sym_blocks = []  # [y_debut, y_fin, [lignes], indenté, est_categorie]
    for y, t, b, x in col0:
        if sym_blocks and y - sym_blocks[-1][1] <= 12.5 and sym_blocks[-1][4] == b:
            sym_blocks[-1][2].append(t)
            sym_blocks[-1][1] = y
        else:
            sym_blocks.append([y, y, [t], x > 35, b])
    category = state.get("category")
    symptom = state.get("symptom")
    marks = []  # (y, categorie, symptome)
    parent = None
    for y0, y1, lines_, indented, is_cat in sym_blocks:
        txt = join_lines(lines_)
        if is_cat:
            category = txt
            symptom = parent = None
        elif indented and parent:
            symptom = f"{parent} {txt}"
        else:
            symptom = txt
            parent = txt if txt.endswith(":") else None
        marks.append((y0, category, symptom))

    def context(y):
        cat, sym = state.get("category"), state.get("symptom")
        for my, mc, ms in marks:
            if my <= y + 2:
                cat, sym = mc, ms
        return cat, sym

    # 2) causes (colonne 1) puis actions (colonne 2)
    causes = []
    for y, col, t, bold, _ in rows:
        if col != 1:
            continue
        cat, sym = context(y)
        if (not causes or y - causes[-1]["y_last"] > 12.5
                or causes[-1]["symptom"] != sym):
            causes.append({"y": y, "y_last": y, "cause": [t], "action": [],
                           "symptom": sym, "category": cat})
        else:
            causes[-1]["cause"].append(t)
            causes[-1]["y_last"] = y
    # actions regroupées en paragraphes, rattachées à la cause alignée
    acts = []
    for y, col, t, bold, _ in rows:
        if col != 2:
            continue
        if acts and y - acts[-1][1] <= 12.5:
            acts[-1][2].append(t)
            acts[-1][1] = y
        else:
            acts.append([y, y, [t]])
    for y0, _, lines_ in acts:
        tgt = [c for c in causes if c["y"] <= y0 + 2]
        (tgt[-1] if tgt else causes[0])["action"] += lines_ if causes else []
    # une cause sans action partage l'action de la cause précédente
    merged = []
    for c in causes:
        if (not c["action"] and merged and merged[-1]["symptom"] == c["symptom"]):
            merged[-1]["cause"] = merged[-1]["cause"] + [" / "] + c["cause"]
        else:
            merged.append(c)
    causes = merged
    if notes and causes:
        causes[-1]["note"] = join_lines(notes)
    state["category"], state["symptom"] = category, symptom
    return title, causes


def anomaly_chunks(doc, pages, chapter):
    groups = {}
    order = []
    state = {}
    for i, lab in pages:
        _, causes = parse_anomaly_page(doc[i], state)
        for c in causes:
            key = (c["category"], c["symptom"])
            if key not in groups:
                groups[key] = {"pages": [], "items": []}
                order.append(key)
            if lab not in groups[key]["pages"]:
                groups[key]["pages"].append(lab)
            groups[key]["items"].append((join_lines(c["cause"]), join_lines(c["action"])))
            if c.get("note"):
                groups[key].setdefault("notes", []).append(c["note"])
    chunks = []
    for cat, sym in order:
        g = groups[(cat, sym)]
        lines = [f"Problème : {sym}"]
        for k, (cause, action) in enumerate(g["items"], 1):
            lines.append(f"Cause possible {k} : {cause}")
            if action:
                lines.append(f"Que faire : {action}")
        for note in g.get("notes", []):
            lines.append(f"Remarque : {note}")
        chunks.append({
            "chapitre": chapter,
            "section": "Anomalies de fonctionnement",
            "sous_section": f"{cat} – {sym}" if cat else sym,
            "categorie": "anomalie",
            "pages": g["pages"],
            "texte": "\n".join(lines),
        })
    return chunks


# ----------------------------------------------- tableau des fusibles
def fuse_chunks(doc, pages, chapter):
    chunks = []
    for i, lab in pages:
        raw = []
        for b in doc[i].get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for ln in b["lines"]:
                t = "".join(s["text"] for s in ln["spans"]).strip()
                if t:
                    x0, y0, x1, y1 = ln["bbox"]
                    raw.append((x0, y0, y1, t))
        heads = sorted(x0 for x0, y0, y1, t in raw if t == "Numéro")
        nums, aff = [], []
        for x0, y0, y1, t in raw:
            if t in ("Numéro", "Affectation") or y0 < 25 or re.fullmatch(r"\d\.\d{1,2}", t):
                continue
            g = max([k for k, h in enumerate(heads) if h - 5 <= x0] or [0])
            if re.fullmatch(r"\d{1,2}", t) and x0 - heads[g] < 30:
                nums.append((g, (y0 + y1) / 2, t))
            else:
                aff.append((g, y0, y1, t))
        aff.sort()
        paras = []
        for g, y0, y1, t in aff:
            if paras and paras[-1][0] == g and y0 - paras[-1][2] <= 6:
                paras[-1][3].append(t)
                paras[-1][2] = y1
            else:
                paras.append([g, y0, y1, [t]])
        rows = []
        for g, y0, y1, lines_ in paras:
            cands = [n for n in nums if n[0] == g and y0 - 8 <= n[1] <= y1 + 8]
            if not cands:
                continue
            c = (y0 + y1) / 2
            num = min(cands, key=lambda n: abs(n[1] - c))[2]
            txt = join_lines(lines_).strip(" ,")
            rows.append((int(num), txt))
        merged = {}
        for n, t in rows:
            merged[n] = (merged[n] + ", " + t) if n in merged else t
        if merged:
            lo, hi = min(merged), max(merged)
            chunks.append({
                "chapitre": chapter, "section": "FUSIBLES",
                "sous_section": f"Affectation des fusibles de l’habitacle (n° {lo} à {hi})",
                "categorie": "fusibles", "pages": [lab],
                "texte": "Boîtier de fusibles de l’habitacle (trappe A). La présence des fusibles "
                         "dépend du niveau d’équipement du véhicule.\n"
                         + "\n".join(f"Fusible n° {n} : {t}" for n, t in sorted(merged.items())),
            })
    return chunks


# ------------------------------- ordinateur de bord : tableaux à 2 colonnes
HEADER_RE = re.compile(r"Interprétation de l.affichage sélectionné")
SEVERITY = {
    "information": "Message d’information (pas d’anomalie).",
    "anomalies": "Message d’anomalie : il apparaît avec le témoin d’alerte (clé) ; arrêt au plus tôt "
                 "chez un Représentant de la marque en conduisant avec ménagement.",
    "alerte": "Message d’alerte : il apparaît avec le témoin STOP ; arrêt impératif et immédiat, "
              "arrêtez le moteur et ne le redémarrez pas, faites appel à un Représentant de la marque.",
}


def split_table(page):
    lines = sorted(page_lines(page), key=lambda r: (round(r[1]), r[0]))
    title = " ".join(t.strip() for x0, y0, x1, t, b, s in lines if y0 < 25 and s >= 10.5)
    hdr = [r for r in lines if HEADER_RE.search(r[3])]
    y_h = hdr[0][1] if hdr else 0
    above = [r for r in lines if 25 <= r[1] < y_h - 5 and not r[3].strip().startswith("Exemples")]
    below = [r for r in lines if r[1] > y_h + 5]
    return clean_title(title), above, below, (hdr[0][0] if hdr else 250)


def message_chunks(doc, pages, chapter):
    chunks = []
    for i, lab in pages:
        title, above, below, hx = split_table(doc[i])
        kind = next((k for k in SEVERITY if k in title), "information")
        intro = join_lines([r[3] for r in above])
        if intro:
            chunks.append({"chapitre": chapter, "section": title, "sous_section": None,
                           "categorie": "message_tableau_de_bord", "pages": [lab], "texte": intro})
        # 1) éclate les lignes « message » Interprétation
        cells = []
        for x0, y0, x1, t, b, s in below:
            m = re.match(r"\s*(«[^»]*»)\s*(.*)", t)
            if m and m.group(2).strip():
                cells.append((y0, "msg", m.group(1)))
                cells.append((y0, "txt", m.group(2)))
            elif x0 < hx - 60 and (b or "«" in t or "»" in t):
                cells.append((y0, "msg", t))
            else:
                cells.append((y0, "txt", t))
        msgs = []  # [y, [lignes message]]
        for y, kind_, t in cells:
            if kind_ != "msg":
                continue
            if msgs and y - msgs[-1][1] <= 12 and not msgs[-1][2][-1].strip().endswith("»"):
                msgs[-1][2].append(t)
                msgs[-1][1] = y
            else:
                msgs.append([y, y, [t]])
        for k, (y0, y1, mlines) in enumerate(msgs):
            nxt = msgs[k + 1][0] if k + 1 < len(msgs) else 1e9
            txt = [t for y, kd, t in cells if kd == "txt" and y0 - 16 <= y < nxt - 16]
            msg = join_lines(mlines)
            chunks.append({
                "chapitre": chapter, "section": title, "sous_section": msg,
                "categorie": "message_tableau_de_bord", "pages": [lab],
                "texte": f"Message affiché au tableau de bord : {msg}\n"
                         f"Signification : {join_lines(txt)}\n{SEVERITY[kind]}",
            })
    return chunks


def trip_chunks(doc, pages, chapter):
    """Pages « paramètres de voyage » : on garde la colonne d'explication,
    pas les exemples d'écran dessinés à gauche."""
    chunks = []
    for i, lab in pages:
        title, above, below, hx = split_table(doc[i])
        right = [r for r in below if r[0] >= hx - 50]
        y_right = max([r[1] for r in right] or [0])
        wide = [r for r in below if r[0] < 60 and r[1] > y_right]
        paras, cur, prev_y = [], [], None
        for x0, y0, x1, t, b, s in sorted(right, key=lambda r: r[1]):
            if prev_y is not None and (y0 - prev_y > 12.5 or b) and cur:
                if not (b and cur and cur[-1][1]):
                    paras.append(cur)
                    cur = []
            cur.append((t, b))
            prev_y = y0
        if cur:
            paras.append(cur)
        texts = [join_lines([t for t, _ in p]) for p in paras]
        if wide:
            texts.append(join_lines([r[3] for r in sorted(wide, key=lambda r: r[1])]))
        body = "\n".join(t for t in texts if t)
        if len(body.split()) >= 12:
            chunks.append({"chapitre": chapter, "section": title, "sous_section": None,
                           "categorie": "message_tableau_de_bord", "pages": [lab], "texte": body})
    return chunks


# ------------------------------------------------------ pages en prose
def prose_chunks(doc, pages, chapter, category):
    chunks = []
    cur = None
    section = None
    orphans = []  # titres sans texte (ex. témoins d'éclairage)

    def emit_orphans():
        if len(orphans) >= 2:
            chunks.append({
                "chapitre": chapter, "section": section, "sous_section": "Liste des témoins",
                "categorie": category, "pages": sorted({p for _, p in orphans}),
                "texte": "Témoins présents au tableau de bord (sans description détaillée) : "
                         + " ; ".join(t for t, _ in orphans) + ".",
            })
        orphans.clear()

    def flush():
        nonlocal cur
        if cur and cur["sub"] and not cur["paras"] and cur["sub"].startswith("Témoin"):
            orphans.append((cur["sub"], cur["pages"][0]))
        if cur and cur["paras"]:
            text = "\n".join(cur["paras"])
            if len(text.split()) >= 12:
                chunks.append({
                    "chapitre": chapter,
                    "section": cur["section"],
                    "sous_section": cur["sub"],
                    "categorie": category,
                    "pages": cur["pages"],
                    "texte": text,
                })
        cur = None

    for i, lab in pages:
        title, paras = page_paragraphs(doc[i])
        if title:
            t = clean_title(title)
            if t != section:
                flush()
                emit_orphans()
                section = t
        for txt, is_sub, gap in paras:
            if is_sub:
                flush()
                cur = {"section": section, "sub": txt, "paras": [], "pages": [lab]}
                continue
            if gap and cur is not None:
                flush()  # encadré : passage autonome
                cur = {"section": section, "sub": None, "paras": [], "pages": [lab]}
            if cur is None:
                cur = {"section": section, "sub": None, "paras": [], "pages": [lab]}
            if lab not in cur["pages"]:
                cur["pages"].append(lab)
            n = sum(len(p.split()) for p in cur["paras"])
            if n + len(txt.split()) > MAX_WORDS and cur["paras"]:
                sub, sec = cur["sub"], cur["section"]
                flush()
                cur = {"section": sec, "sub": sub, "paras": [], "pages": [lab]}
            cur["paras"].append(txt)
    flush()
    emit_orphans()

    # fusionne les passages trop courts avec le précédent de la même section
    merged = []
    for c in chunks:
        small = len(c["texte"].split()) < 30 and c["categorie"] != "voyant"
        if (merged and small and merged[-1]["section"] == c["section"]
                and len(merged[-1]["texte"].split()) < 220):
            merged[-1]["texte"] += "\n" + c["texte"]
            merged[-1]["pages"] += [p for p in c["pages"] if p not in merged[-1]["pages"]]
        else:
            merged.append(c)
    return merged


# ------------------------------------------------------------------- main
def main():
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "Clio4.pdf")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    idx = label_index(doc)
    build_icon_map(doc, idx)

    chunks = []
    for chapter, first, last, cat in KEEP:
        pages = label_range(idx, first, last)
        if cat == "anomalie":
            chunks += anomaly_chunks(doc, pages, chapter)
        elif cat == "messages_tableau":
            chunks += message_chunks(doc, pages, chapter)
        elif cat == "ordinateur_tableau":
            chunks += trip_chunks(doc, pages, chapter)
        elif cat == "fusibles_tableau":
            chunks += fuse_chunks(doc, pages, chapter)
        else:
            chunks += prose_chunks(doc, pages, chapter, cat)

    records = []
    for n, c in enumerate(chunks, 1):
        titre = c["section"] + (f" > {c['sous_section']}" if c["sous_section"] else "")
        rec = {
            "id": f"{SLUG}-{n:03d}",
            **{k: META[k] for k in ("marque", "modele", "phase", "annees", "langue")},
            "chapitre": c["chapitre"],
            "section": c["section"],
            "sous_section": c["sous_section"],
            "categorie": c["categorie"],
            "pages": c["pages"],
            "titre": titre,
            "texte": c["texte"],
            # texte à vectoriser : contexte + contenu
            "texte_embedding": f"{META['marque']} {META['modele']} ({META['annees']}) – {titre}\n{c['texte']}",
            "nb_mots": len(c["texte"].split()),
            "document": META["document"],
            "source_url": META["source_url"],
        }
        records.append(rec)

    with open(out / f"{SLUG}_corpus.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out / f"{SLUG}_corpus.md", "w", encoding="utf-8") as f:
        f.write(f"# Corpus RAG – {META['marque']} {META['modele']} {META['phase']} ({META['annees']})\n\n")
        f.write(f"Source : {META['document']} – {META['source_url']}\n\n")
        f.write(f"{len(records)} passages.\n\n")
        for r in records:
            f.write(f"## [{r['id']}] {r['titre']}\n")
            f.write(f"*{r['chapitre']} · pages {', '.join(r['pages'])} · catégorie : {r['categorie']}*\n\n")
            f.write(r["texte"].replace("\n", "\n\n") + "\n\n")

    words = sum(r["nb_mots"] for r in records)
    print(f"{len(records)} passages, {words} mots, depuis {len(doc)} pages")


if __name__ == "__main__":
    main()
