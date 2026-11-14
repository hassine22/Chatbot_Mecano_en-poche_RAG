"""
Construction du corpus RAG « Mécano de poche » pour la Hyundai i10 (AC3 / 3e génération).

Entrée  : notice d'utilisation Hyundai i10 (data/pdf/I10.pdf)
Sorties : corpus/output/i10_corpus.jsonl (1 passage par ligne)
          corpus/output/i10_corpus.md    (version lisible pour relecture)

Usage :
    python corpus/scripts/build_i10.py data/pdf/I10.pdf corpus/output
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

SLUG = "i10"
META = {
    "marque": "Hyundai",
    "modele": "i10",
    "phase": "AC3",
    "annees": "2019-2024",
    "langue": "en",
    "document": "Hyundai i10 Owner's Manual (AC3, 2020)",
    "source_url": "https://www.manualslib.com/manual/3292100/Hyundai-I10.html",
}

KEEP = [
    ("1. Foreword", "1-5", "1-7", "carburant"),
    ("4. Instrument cluster", "4-8", "4-20", "voyant"),
    ("4. Instrument cluster", "4-21", "4-24", "message_tableau_de_bord"),
    ("5. Convenience features", "5-34", "5-36", "carburant"),
    ("6. Whilst driving", "6-6", "6-13", "demarrage_moteur"),
    ("6. Whilst driving", "6-17", "6-31", "boite_automatique"),
    ("6. Whilst driving", "6-32", "6-43", "aides_conduite"),
    ("6. Whilst driving", "6-44", "6-48", "demarrage_moteur"),
    ("8. Emergency situations", "8-3", "8-9", "anomalie"),
    ("8. Emergency situations", "8-10", "8-15", "pneus_roues"),
    ("8. Emergency situations", "8-16", "8-24", "pneus_roues"),
    ("8. Emergency situations", "8-25", "8-32", "pneus_roues"),
    ("8. Emergency situations", "8-33", "8-36", "remorquage"),
    ("9. Maintenance", "9-4", "9-19", "entretien"),
    ("9. Maintenance", "9-26", "9-29", "electricite_batterie"),
    ("9. Maintenance", "9-30", "9-38", "pneus_roues"),
    ("9. Maintenance", "9-39", "9-55", "fusibles"),
]

def label_index(doc):
    idx = {}
    for i, page in enumerate(doc):
        text = page.get_text()
        found = re.findall(r"^\s*([1-9]-\d{1,2})\s*$", text, re.M)
        if found:
            idx[found[-1]] = i
    return idx

def label_range(idx, first, last):
    a, b = idx[first], idx[last]
    labels = {v: k for k, v in idx.items()}
    return [(i, labels.get(i, f"p.{i+1}")) for i in range(a, b + 1)]

def decode_span(span):
    if "chars" in span:
        chars = span["chars"]
        font = span.get("font", "")
        if "Helvetica" in font and any(c["c"] in "<>H[U" for c in chars) and any(ord(c["c"]) == 3 or ord(c["c"]) < 32 for c in chars):
            res = []
            for c in chars:
                oc = ord(c["c"])
                res.append(chr(oc + 29) if oc < 128 else c["c"])
            return "".join(res)
        return "".join(c["c"] for c in chars)
    return span.get("text", "")

def clean_text(text):
    text = re.sub(r"[\ue000-\uf8ff®™„\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Bullets & symbols normalization
    text = re.sub(r"[\x80-\x9f\u048b\u2022\u00b7]", " - ", text)
    text = text.replace("Ҩ", "'").replace("Ô", "'").replace("Õ", "'")
    text = text.replace("´", "'").replace("`", "'")
    text = text.replace("=to", "to").replace("=yourself", "yourself").replace("=whenever", "whenever")
    text = text.replace("=performing", "performing").replace("=any", "any").replace("=maintenance", "maintenance")
    text = text.replace("=or", "or").replace("=inspection", "inspection").replace("=proce-", "proce-")
    text = text.replace("=repairer", "repairer").replace("=meets", "meets").replace("=HYUNDAI", "HYUNDAI").replace("=high", "high")
    text = text.replace("=", " ")
    
    # Re-stitch split hyphens (e.g. immedi- ately -> immediately)
    text = re.sub(r"\b([a-zA-Z]+)-\s+([a-z][a-zA-Z]+)\b", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = text.replace("( ", "(")
    return text.strip()

def extract_page_lines(page):
    """Extrait les lignes décodées avec position et métadonnées."""
    raw = page.get_text("rawdict")
    lines = []
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue
            line_text = "".join(decode_span(s) for s in spans).strip()
            if not line_text:
                continue
            # Header / footer filters
            if re.match(r"^[1-9]-\d{1,2}$", line_text) or re.match(r"^0[1-9]$", line_text):
                continue
            if line_text in ["Instrument cluster", "Whilst driving", "Emergency situations", "Maintenance", "Foreword", "Convenience features"]:
                continue
            if re.match(r"^[A-Z0-9/]{8,}$", line_text):
                continue
            
            x0, y0, x1, y1 = l.get("bbox", (0, 0, 0, 0))
            max_size = max(s.get("size", 9) for s in spans)
            flags = spans[0].get("flags", 0)
            is_bold = bool(flags & 16) or "Bold" in spans[0].get("font", "")
            
            lines.append({
                "text": line_text,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "size": max_size,
                "bold": is_bold,
                "col": 0 if x0 < 220 else 1
            })
    lines.sort(key=lambda r: (r["col"], round(r["y0"], 1)))
    return lines

def join_block_lines(line_list):
    """Recolle proprement les lignes d'un bloc en gérant les césures en fin de ligne."""
    out = ""
    for ln in line_list:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-"):
            if re.match(r"^[a-z]", ln):
                out = out[:-1] + ln
            else:
                out = out + " " + ln
        else:
            out = out + " " + ln
    return out

def parse_page_sections(lines):
    """Regroupe les lignes par bloc ou sous-section logique."""
    blocks = []
    curr = []
    curr_heading = None

    for ln in lines:
        t = ln["text"].strip()
        is_heading = (ln["bold"] and ln["size"] >= 10.3 and len(t) < 70 
                      and not t.endswith(".") and not t.startswith("•") and not t.startswith("-")
                      and not re.match(r"^(i Information|NOTICE|WARNING|CAUTION|DANGER)\b", t, re.I))
        
        if is_heading:
            if curr:
                blocks.append((curr_heading, curr))
                curr = []
            curr_heading = t
        else:
            curr.append(t)

    if curr:
        blocks.append((curr_heading, curr))
    return blocks

def build_passages(doc, idx):
    chunks = []
    
    for chapter, first, last, cat in KEEP:
        pages = label_range(idx, first, last)
        default_section = chapter.split(". ", 1)[-1]
        current_section = default_section
        
        for pno, label in pages:
            lines = extract_page_lines(doc[pno])
            blocks = parse_page_sections(lines)
            
            for heading, content_lines in blocks:
                if not content_lines and not heading:
                    continue
                
                raw_body = join_block_lines(content_lines)
                body = clean_text(raw_body)
                
                sub_sec = None
                sec = current_section
                
                if heading:
                    h_clean = clean_text(heading)
                    if h_clean.isupper() and len(h_clean.split()) <= 5:
                        current_section = h_clean
                        sec = current_section
                    else:
                        sub_sec = h_clean
                
                words = body.split()
                if len(words) < 12:
                    if chunks and chunks[-1]["categorie"] == cat and len(chunks[-1]["texte"].split()) + len(words) <= 350:
                        chunks[-1]["texte"] += " " + body
                        chunks[-1]["nb_mots"] = len(chunks[-1]["texte"].split())
                        if label not in chunks[-1]["pages"]:
                            chunks[-1]["pages"].append(label)
                    continue
                
                if len(words) > 380:
                    parts = []
                    curr_part = []
                    for w in words:
                        curr_part.append(w)
                        if len(curr_part) >= 240 and curr_part[-1].endswith("."):
                            parts.append(" ".join(curr_part))
                            curr_part = []
                    if curr_part:
                        parts.append(" ".join(curr_part))
                    
                    for part_idx, part in enumerate(parts):
                        part_clean = clean_text(part)
                        if len(part_clean.split()) >= 12:
                            part_sub = f"{sub_sec} (part {part_idx+1})" if sub_sec else f"Part {part_idx+1}"
                            chunks.append({
                                "chapitre": chapter,
                                "section": sec,
                                "sous_section": part_sub,
                                "categorie": cat,
                                "pages": [label],
                                "texte": part_clean,
                                "nb_mots": len(part_clean.split()),
                            })
                else:
                    chunks.append({
                        "chapitre": chapter,
                        "section": sec,
                        "sous_section": sub_sec,
                        "categorie": cat,
                        "pages": [label],
                        "texte": body,
                        "nb_mots": len(words),
                    })

    # Merge very small chunks (< 25 words) if adjacent with same section
    merged = []
    for c in chunks:
        if merged and c["nb_mots"] < 25 and merged[-1]["section"] == c["section"] and (merged[-1]["nb_mots"] + c["nb_mots"]) <= 350:
            merged[-1]["texte"] += " " + c["texte"]
            merged[-1]["nb_mots"] = len(merged[-1]["texte"].split())
            for p in c["pages"]:
                if p not in merged[-1]["pages"]:
                    merged[-1]["pages"].append(p)
        else:
            merged.append(c)

    return merged

def main():
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pdf/I10.pdf")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "corpus/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    doc = pymupdf.open(pdf_path)
    idx = label_index(doc)
    
    chunks = build_passages(doc, idx)
    
    records = []
    for n, c in enumerate(chunks, 1):
        titre = c["section"] + (f" > {c['sous_section']}" if c["sous_section"] else "")
        rec = {
            "id": f"{SLUG}-{n:03d}",
            "marque": META["marque"],
            "modele": META["modele"],
            "phase": META["phase"],
            "annees": META["annees"],
            "langue": META["langue"],
            "chapitre": c["chapitre"],
            "section": c["section"],
            "sous_section": c["sous_section"],
            "titre": titre,
            "categorie": c["categorie"],
            "pages": c["pages"],
            "texte": c["texte"],
            "texte_embedding": f"{META['marque']} {META['modele']} ({META['annees']}) – {titre}\n{c['texte']}",
            "nb_mots": len(c["texte"].split()),
            "document": META["document"],
            "source_url": META["source_url"],
        }
        records.append(rec)
        
    jsonl_path = out_dir / f"{SLUG}_corpus.jsonl"
    md_path = out_dir / f"{SLUG}_corpus.md"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Corpus RAG – {META['marque']} {META['modele']} {META['phase']} ({META['annees']})\n\n")
        f.write(f"Source : {META['document']} – {META['source_url']}\n\n")
        f.write(f"{len(records)} passages.\n\n")
        for r in records:
            f.write(f"## [{r['id']}] {r['titre']}\n")
            f.write(f"*{r['chapitre']} · pages {', '.join(r['pages'])} · catégorie : {r['categorie']}*\n\n")
            f.write(r["texte"].replace("\n", "\n\n") + "\n\n")
            
    words = sum(r["nb_mots"] for r in records)
    print(f"Génération terminée : {len(records)} passages, {words} mots générés.")

if __name__ == "__main__":
    main()
