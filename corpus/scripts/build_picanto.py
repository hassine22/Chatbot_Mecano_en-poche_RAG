"""Build the Kia Picanto RAG corpus from its owner manual.

Usage:
    python corpus/scripts/build_picanto.py data/pdf/2023-kia-picanto--2.pdf corpus/output
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

SLUG = "picanto"
META = {
    "marque": "Kia",
    "modele": "Picanto",
    "phase": "",
    "annees": "2023-2023",
    "langue": "en",
    "document": "Kia Picanto Owner's Manual (2023)",
    "source_url": "https://www.kia.com/",
}

TOPICS = [
    ("fuse|fuse box", "fusibles"),
    ("battery|jump starting|smart key battery|battery replacement", "electricite_batterie"),
    ("towing|emergency towing|roadside assistance", "remorquage"),
    ("tire|tyre|puncture|flat tire|flat tyre|tire pressure|tyre pressure|wheel", "pneus_roues"),
    ("engine oil|coolant|maintenance service|air cleaner|spark plug|drive belt|engine compartment", "entretien"),
    ("fuel requirements|fuel filler|fuel gauge|running out of fuel|fuel", "carburant"),
    ("engine start|starting the engine|stop and start|engine ignition", "demarrage_moteur"),
    ("warning and indicator lights|warning lights|indicator lights|instrument cluster|dashboard", "voyant"),
    ("message|warning message|master warning", "message_tableau_de_bord"),
    ("engine overheating|overheating|engine does not start|if the engine", "anomalie"),
    ("vehicle identification|vin|engine number|technical specifications", "identification"),
    ("abs|esc|electronic stability|traction control|lane keeping|parking sensor|driver assistance", "aides_conduite"),
]


def printed_page(text, fallback):
    """Return the printed section-page label when it is present."""
    labels = re.findall(r"\b(?:[1-9]\d?|10)-\d{1,2}\b", text)
    return labels[0] if labels else str(fallback)


def clean_text(text):
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(?m)^\s*(?:\d+\s*)$", "", text)
    text = re.sub(r"(?m)^\s*[A-Z0-9]{6,}\s*$", "", text)
    return text.strip()


def category_for(text):
    lower = text.lower()
    for pattern, category in TOPICS:
        if re.search(pattern, lower):
            return category
    return None


def section_for(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:12]:
        if 3 <= len(line.split()) <= 12 and not re.search(r"\.{3,}", line):
            return line
    return "Kia Picanto owner manual"


def split_passage(text, limit=380):
    words = text.split()
    if len(words) <= limit:
        return [text]
    parts = []
    while words:
        part = " ".join(words[:limit])
        words = words[limit:]
        if words and len(words) < 12:
            part += " " + " ".join(words)
            words = []
        parts.append(part)
    return parts


def main():
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pdf/2023-kia-picanto--2.pdf")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "corpus/output")
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    records = []

    for page_number, page in enumerate(doc, 1):
        if page_number < 121:
            continue
        raw = clean_text(page.get_text())
        category = category_for(raw)
        if not category:
            continue
        words = raw.split()
        if len(words) < 12:
            continue
        label = printed_page(raw, page_number)
        section = section_for(raw)
        for passage_number, passage in enumerate(split_passage(raw), 1):
            subsection = f"Page {label}" if passage_number == 1 else f"Page {label} (part {passage_number})"
            title = f"{section} > {subsection}"
            records.append({
                "id": f"{SLUG}-{len(records) + 1:03d}",
                "marque": META["marque"],
                "modele": META["modele"],
                "phase": META["phase"],
                "annees": META["annees"],
                "langue": META["langue"],
                "chapitre": section,
                "section": section,
                "sous_section": subsection,
                "titre": title,
                "categorie": category,
                "pages": [label],
                "texte": passage,
                "texte_embedding": f"{META['marque']} {META['modele']} ({META['annees']}) – {title}\n{passage}",
                "nb_mots": len(passage.split()),
                "document": META["document"],
                "source_url": META["source_url"],
            })

    jsonl = out / f"{SLUG}_corpus.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    markdown = out / f"{SLUG}_corpus.md"
    with markdown.open("w", encoding="utf-8") as handle:
        handle.write(f"# Corpus RAG – {META['marque']} {META['modele']} ({META['annees']})\n\n")
        handle.write(f"Source : {META['document']} – {META['source_url']}\n\n")
        handle.write(f"{len(records)} passages.\n\n")
        for record in records:
            handle.write(f"## [{record['id']}] {record['titre']}\n")
            handle.write(f"*pages {', '.join(record['pages'])} · catégorie : {record['categorie']}*\n\n")
            handle.write(record["texte"] + "\n\n")

    print(f"{len(records)} passages, {sum(r['nb_mots'] for r in records)} mots, depuis {len(doc)} pages")


if __name__ == "__main__":
    main()
