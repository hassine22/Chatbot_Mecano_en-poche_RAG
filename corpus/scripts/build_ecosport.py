"""
Construit le corpus RAG de la notice Ford EcoSport 2018 fournie dans data/pdf/.

Usage depuis la racine du dépôt :
    python corpus/scripts/build_ecosport.py
    python corpus/scripts/build_ecosport.py data/pdf/ford_ecosport_2018.pdf corpus/output
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SLUG = "ecosport"
META = {
    "marque": "Ford",
    "modele": "EcoSport",
    "phase": "",
    "annees": "2018-2018",
    "langue": "en",
    "document": "Ford EcoSport Owner's Manual (GN15-19G219-ZFB, first printing, 2018)",
    "source_url": (
        "https://www.fordservicecontent.com/Ford_Content/Catalog/"
        "owner_information/B515_BW_Owner_Manual_SA.pdf"
    ),
}

# Pages PDF (1-based), catégorie du schéma commun.
KEEP = [
    (3, 7, "voyant"),
    (12, 20, "message_tableau_de_bord"),
    (21, 22, "demarrage_moteur"),
    (27, 31, "aides_conduite"),
    (32, 33, "remorquage"),
    (34, 36, "electricite_batterie"),
    (37, 45, "fusibles"),
    (46, 53, "entretien"),
    (54, 54, "electricite_batterie"),
    (55, 64, "pneus_roues"),
    (65, 71, "entretien"),
]

CHAPTERS = [
    (3, 7, "Instrument Cluster"),
    (12, 20, "Information Displays"),
    (21, 22, "Unique Driving Characteristics"),
    (27, 29, "Brakes"),
    (30, 30, "Traction Control (If Equipped)"),
    (31, 31, "Stability Control (If Equipped)"),
    (32, 36, "Roadside Emergencies"),
    (37, 45, "Fuses"),
    (46, 54, "Maintenance"),
    (55, 64, "Wheels and Tires"),
    (65, 71, "Scheduled Maintenance"),
]
MAX_WORDS = 320


def printed_page_number(page):
    """Lit le numéro imprimé dans le pied de page, distinct de l'index PDF."""
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                y0 = span["bbox"][1]
                if 490 < y0 < 520 and re.fullmatch(r"\d{1,3}", text):
                    return text
    raise ValueError("Numéro de page imprimé introuvable dans le PDF.")


def chapter_for(pdf_page):
    for first, last, chapter in CHAPTERS:
        if first <= pdf_page <= last:
            return chapter
    raise ValueError(f"Aucun chapitre défini pour la page PDF {pdf_page}.")


def join_lines(lines):
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
        elif text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text += " " + line
    text = text.replace("\ufffd", "•")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    return text.strip()


def page_paragraphs(page):
    """Extrait les paragraphes en conservant l'ordre vertical des colonnes Ford."""
    columns = {0: [], 1: []}
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0 = line["bbox"][:2]
            if y0 < 45 or y0 > 490:
                continue
            text = "".join(s["text"] for s in spans).strip()
            bold = all(s["flags"] & 16 for s in spans)
            column = 0 if x0 < 195 else 1
            columns[column].append((y0, x0, text, bold))

    paragraphs = []
    for column in columns.values():
        column.sort(key=lambda row: (round(row[0]), row[1]))
        fused = []
        for row in column:
            if fused and abs(row[0] - fused[-1][0]) < 1.5:
                y0, x0, text, bold = fused[-1]
                fused[-1] = (y0, x0, text + " " + row[2], bold and row[3])
            else:
                fused.append(row)

        current = []
        previous = None
        for y0, x0, text, bold in fused:
            if previous and (y0 - previous[0] > 12.5 or bold != previous[1]):
                paragraphs.append((
                    join_lines([item[0] for item in current]),
                    all(item[1] for item in current),
                ))
                current = []
            current.append((text, bold))
            previous = (y0, bold)
        if current:
            paragraphs.append((join_lines([item[0] for item in current]),
                               all(item[1] for item in current)))
    return [item for item in paragraphs if item[0]]


def page_lines(page):
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0 = line["bbox"][:2]
            if 45 <= y0 <= 490:
                text = "".join(s["text"] for s in spans).strip()
                lines.append((y0, x0, text, all(s["flags"] & 16 for s in spans)))
    return sorted(lines, key=lambda row: (row[0], row[1]))


def cell_paragraphs(lines):
    groups = []
    current = []
    previous_y = None
    for y0, _, text, _ in lines:
        if current and y0 - previous_y > 12.5:
            groups.append(current)
            current = []
        current.append((y0, text))
        previous_y = y0
    if current:
        groups.append(current)
    return [
        {
            "start": group[0][0],
            "end": group[-1][0],
            "center": (group[0][0] + group[-1][0]) / 2,
            "text": join_lines([text for _, text in group]),
        }
        for group in groups
    ]


def message_chunks(page, pdf_page, label):
    chapter = chapter_for(pdf_page)
    chunks = []
    section = chapter
    messages = []
    actions = []
    intro = []
    table_started = False

    def flush_section():
        message_groups = cell_paragraphs(messages)
        action_groups = cell_paragraphs(actions)
        if not message_groups and not action_groups:
            return
        paired = {}
        for message in message_groups:
            matching_actions = [
                action for action in action_groups
                if max(0, action["start"] - message["end"],
                       message["start"] - action["end"]) <= 4
            ]
            if matching_actions:
                action = min(
                    matching_actions,
                    key=lambda item: abs(item["center"] - message["center"]),
                )
                paired.setdefault(id(action), (action, []))[1].append(message)
            elif not action_groups:
                paired.setdefault(id(message), (message, []))[1].append(message)

        for action, linked_messages in paired.values():
            action_text = action["text"] if action_groups else ""
            message_text = " / ".join(message["text"] for message in linked_messages)
            text = f"Message: {message_text}"
            if action_text:
                text += f"\nAction: {action_text}"
            if len(text.split()) >= 6:
                chunks.append({
                    "chapitre": chapter,
                    "section": chapter,
                    "sous_section": f"{section} — {message_text}",
                    "categorie": "message_tableau_de_bord",
                    "pages": [label],
                    "texte": text,
                })
        messages.clear()
        actions.clear()

    for y0, x0, text, bold in page_lines(page):
        if text.lower() in {"message", "action"}:
            table_started = True
            continue
        is_section_heading = (
            bold and x0 < 140 and len(text.split()) <= 10
            and text.upper() not in {"WARNING", "NOTE"}
        )
        if not table_started and is_section_heading:
            flush_section()
            section = text
            continue
        if not table_started:
            intro.append((y0, x0, text))
            continue
        if is_section_heading:
            flush_section()
            section = text
        elif x0 < 140:
            messages.append((y0, x0, text, bold))
        else:
            actions.append((y0, x0, text, bold))

    flush_section()
    intro_columns = {0: [], 1: []}
    for _, x0, text in intro:
        intro_columns[0 if x0 < 195 else 1].append(text)
    intro_text = "\n".join(
        join_lines(column) for column in intro_columns.values() if column
    )
    if len(intro_text.split()) >= 12:
        chunks.insert(0, {
            "chapitre": chapter,
            "section": chapter,
            "sous_section": "Information display message notes",
            "categorie": "message_tableau_de_bord",
            "pages": [label],
            "texte": intro_text,
        })
    return chunks


def fuse_chunks(page, pdf_page, label):
    lines = page_lines(page)
    headers = [
        row for row in lines
        if row[2].strip().lower() in {"fuse", "relay"}
    ]
    chunks = []
    for index, header in enumerate(headers):
        y_header, _, table_type, _ = header
        y_end = headers[index + 1][0] if index + 1 < len(headers) else 490
        section_endings = [
            row[0] for row in lines
            if row[3] and row[1] < 150 and y_header + 5 < row[0] < y_end
            and row[2].strip().lower() not in {
                "fuse", "relay", "protected components", "fuse amp rating",
                "circuits switched",
            }
        ]
        if section_endings:
            y_end = min(y_end, min(section_endings))
        table_lines = [row for row in lines if y_header < row[0] < y_end]
        identifiers = [
            row for row in table_lines
            if row[1] < 100 and re.fullmatch(r"(?:F\d{2}|\d{1,2})", row[2])
        ]
        if not identifiers:
            continue

        title_candidates = [
            row[2] for row in lines
            if row[3] and row[1] < 150 and 45 <= row[0] < y_header
            and row[2].strip().lower() not in {
                "fuses", "fuse", "fuse amp rating", "fuse box locations",
                "protected components", "circuits switched",
            }
        ]
        table_title = title_candidates[-1] if title_candidates else (
            "Relay assignments" if table_type.lower() == "relay" else "Fuse assignments"
        )
        rows = {id(item): {"identifier": item[2], "rating": "", "components": []}
                for item in identifiers}

        def nearest_identifier(y0):
            return min(identifiers, key=lambda item: abs(item[0] - y0))

        for item in table_lines:
            y0, x0, text, _ = item
            if x0 < 100 or text.strip().lower() in {
                "protected components", "fuse amp rating", "circuits switched"
            }:
                continue
            identifier = nearest_identifier(y0)
            target = rows[id(identifier)]
            if table_type.lower() == "fuse" and 100 <= x0 < 180:
                target["rating"] = text

        descriptions = [
            row for row in table_lines
            if row[1] >= (180 if table_type.lower() == "fuse" else 100)
            and row[2].strip().lower() not in {
                "protected components", "fuse amp rating", "circuits switched"
            }
        ]
        for paragraph in cell_paragraphs(descriptions):
            identifier = nearest_identifier(paragraph["center"])
            rows[id(identifier)]["components"].append(paragraph["text"])

        lines_out = []
        for identifier in identifiers:
            row = rows[id(identifier)]
            details = []
            if row["rating"]:
                details.append(row["rating"])
            details.extend(row["components"])
            suffix = f" ({', '.join(details)})" if details else ""
            lines_out.append(f"{table_type.title()} {row['identifier']}{suffix}")

        if lines_out:
            chapter = chapter_for(pdf_page)
            chunks.append({
                "chapitre": chapter,
                "section": chapter,
                "sous_section": table_title,
                "categorie": "fusibles",
                "pages": [label],
                "texte": "\n".join(lines_out),
            })
    return chunks


def split_text(text, max_words=MAX_WORDS):
    words = text.split()
    parts = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    if len(parts) > 1 and len(parts[-1].split()) < 12:
        tail = parts.pop()
        parts[-1] += " " + tail
    return parts


def page_chunks(page, pdf_page, label, category):
    chapter = chapter_for(pdf_page)
    paragraphs = page_paragraphs(page)
    chunks = []
    sub_section = None
    body = []

    def flush():
        nonlocal body
        text = "\n".join(body).strip()
        if len(text.split()) >= 12:
            for part in split_text(text):
                chunks.append({
                    "chapitre": chapter,
                    "section": chapter,
                    "sous_section": sub_section,
                    "categorie": category,
                    "pages": [label],
                    "texte": part,
                })
        body = []

    for text, bold in paragraphs:
        is_subheading = (
            bold
            and len(text.split()) <= 14
            and len(text) < 100
            and not re.search(r"[.,;:!?-]$", text)
            and text.upper() not in {"WARNING", "WARNINGS", "NOTE", "NOTES", "CAUTION"}
        )
        if is_subheading:
            flush()
            sub_section = text
        else:
            body.append(text)
    flush()

    if not chunks and paragraphs:
        text = join_lines([text for text, _ in paragraphs])
        if len(text.split()) >= 12:
            chunks.append({
                "chapitre": chapter,
                "section": chapter,
                "sous_section": None,
                "categorie": category,
                "pages": [label],
                "texte": text,
            })
    return chunks


def main():
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "pdf" / "ford_ecosport_2018.pdf"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "corpus" / "output"
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf}")
    out.mkdir(parents=True, exist_ok=True)

    chunks = []
    with pymupdf.open(pdf) as doc:
        for first, last, category in KEEP:
            for pdf_page in range(first, last + 1):
                if pdf_page > len(doc):
                    raise ValueError(f"La page PDF {pdf_page} dépasse les {len(doc)} pages du document.")
                page = doc[pdf_page - 1]
                label = printed_page_number(page)
                if category == "message_tableau_de_bord":
                    chunks.extend(message_chunks(page, pdf_page, label))
                elif category == "fusibles":
                    parsed = fuse_chunks(page, pdf_page, label)
                    chunks.extend(parsed or page_chunks(page, pdf_page, label, category))
                else:
                    chunks.extend(page_chunks(page, pdf_page, label, category))

    records = []
    for number, chunk in enumerate(chunks, 1):
        title = chunk["section"] + (
            f" > {chunk['sous_section']}" if chunk["sous_section"] else ""
        )
        text = chunk["texte"]
        records.append({
            "id": f"{SLUG}-{number:03d}",
            **{key: META[key] for key in ("marque", "modele", "phase", "annees", "langue")},
            "chapitre": chunk["chapitre"],
            "section": chunk["section"],
            "sous_section": chunk["sous_section"],
            "categorie": chunk["categorie"],
            "pages": chunk["pages"],
            "titre": title,
            "texte": text,
            "texte_embedding": (
                f"{META['marque']} {META['modele']} ({META['annees']}) – {title}\n{text}"
            ),
            "nb_mots": len(text.split()),
            "document": META["document"],
            "source_url": META["source_url"],
        })

    jsonl_path = out / f"{SLUG}_corpus.jsonl"
    markdown_path = out / f"{SLUG}_corpus.md"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(
            f"# Corpus RAG – {META['marque']} {META['modele']} ({META['annees']})\n\n"
            f"Source : {META['document']} – {META['source_url']}\n\n"
            f"{len(records)} passages.\n\n"
        )
        for record in records:
            file.write(f"## [{record['id']}] {record['titre']}\n")
            file.write(
                f"*{record['chapitre']} · pages {', '.join(record['pages'])} · "
                f"catégorie : {record['categorie']}*\n\n"
            )
            file.write(record["texte"].replace("\n", "\n\n") + "\n\n")

    total_words = sum(record["nb_mots"] for record in records)
    print(f"{len(records)} passages, {total_words} mots, depuis {len(KEEP)} sélections de pages")


if __name__ == "__main__":
    main()
