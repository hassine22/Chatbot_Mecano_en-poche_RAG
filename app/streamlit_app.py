"""
Interface web de « Mécano de poche ».

Lancement (depuis la racine du dépôt) :
    streamlit run app/streamlit_app.py

L'interface appelle mecano.pipeline.ask() (le vrai RAG : embeddings + FAISS + LLM).
Tant que ce pipeline n'est pas prêt, elle bascule sur app/demo_backend.py
(recherche par mots-clés, sans LLM), qui respecte le même contrat.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import demo_backend  # noqa: E402
from mecano import ner  # noqa: E402


# ---------------------------------------------------------------- backend
def charger_backend():
    """Renvoie (fonction ask, nom du mode). Utilise le vrai pipeline s'il est prêt."""
    try:
        from mecano import pipeline
        if callable(getattr(pipeline, "ask", None)):
            return pipeline.ask, "RAG Actif (FAISS + Gemini)"
    except Exception:
        pass
    return demo_backend.ask, "Recherche lexicale (sans LLM)"


ASK, MODE = charger_backend()
DEMO = ASK is demo_backend.ask

# ---------------------------------------------------------------- libellés sobres
CATEGORIES = {
    "voyant": "Voyants du tableau de bord",
    "message_tableau_de_bord": "Messages affichés",
    "anomalie": "Pannes et anomalies",
    "demarrage_moteur": "Démarrage et moteur",
    "aides_conduite": "ABS, ESP, pression des pneus",
    "boite_automatique": "Boîte automatique",
    "direction": "Direction",
    "carburant": "Carburant",
    "entretien": "Entretien et niveaux",
    "pneus_roues": "Pneus et roues",
    "fusibles": "Fusibles",
    "electricite_batterie": "Batterie et électricité",
    "remorquage": "Remorquage",
    "identification": "Identification du véhicule",
    "caracteristiques_moteur": "Caractéristiques moteur",
    "code_obd": "Codes défaut OBD-II",
}
LANGUE_CODE = {"fr": "FR", "en": "EN", "ar": "AR"}

EXEMPLES = [
    ("Le voyant de pression d'huile s'allume sur ma Clio 4, que faire ?", None),
    ("Mon moteur chauffe, que faire ?", "Clio 4"),
    ("Comment changer une roue crevée sur l'EcoSport ?", None),
    ("Comment démarrer avec des câbles si la batterie de ma i10 est à plat ?", None),
    ("Comment vérifier le niveau d'huile moteur de la Picanto ?", None),
    ("Combien coûte une vidange à Tunis ?", None),
]

# Mots indiquant le niveau d'urgence dans les passages (français et anglais)
URGENT = [
    "arrêt impératif", "arrêtez-vous", "arrêtez votre moteur", "stop the vehicle",
    "stop safely", "stop immediately", "do not drive", "ne redémarrez pas", "risque casse",
]
ATTENTION = [
    "consultez", "représentant de la marque", "faites contrôler", "have the vehicle",
    "authorized", "dealer", "as soon as possible", "au plus tôt", "service now",
]


def niveau_urgence(resultat):
    """Indication heuristique, déduite du passage le plus pertinent."""
    if resultat.get("info_absente") or resultat.get("needs_clarification") or not resultat.get("sources"):
        return None
    texte = (resultat["sources"][0]["titre"] + " " + resultat["sources"][0]["texte"]).lower()
    if any(m in texte for m in URGENT):
        return "urgent"
    if any(m in texte for m in ATTENTION):
        return "attention"
    return "info"


# ---------------------------------------------------------------- configuration page & CSS pro
st.set_page_config(page_title="Mécano de poche — Documentation Technique", page_icon=None, layout="wide")
st.markdown("""
<style>
/* Typographie et sobriété générale */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1080px;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
    font-weight: 600;
}

/* Badges d'entités (Modèle / OBD) */
.entity-bar {
    margin-bottom: 0.65rem;
}
.chip {
    display: inline-block;
    padding: 3px 10px;
    margin: 0 6px 6px 0;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    background: #f8fafc;
    color: #334155;
    border: 1px solid #cbd5e1;
}
.chip-obd {
    background: #fffbeb;
    color: #92400e;
    border-color: #fcd34d;
}
.chip-auto {
    background: #f0f9ff;
    color: #0369a1;
    border-color: #bae6fd;
}

/* Bandeaux d'état techniques */
.status-banner {
    padding: 0.75rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.85rem;
    font-size: 0.9rem;
    line-height: 1.45;
    border-left: 4px solid;
}
.status-urgent {
    background-color: #fef2f2;
    border-left-color: #dc2626;
    color: #7f1d1d;
}
.status-attention {
    background-color: #fffbeb;
    border-left-color: #d97706;
    color: #78350f;
}
.status-absent {
    background-color: #f8fafc;
    border-left-color: #64748b;
    color: #334155;
}
.status-obd {
    background-color: #f0f9ff;
    border-left-color: #0284c7;
    color: #0c4a6e;
    margin-top: 0.75rem;
}

/* Cartes de sources documentaires */
.src-card {
    padding: 0.75rem 0;
}
.src-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.25rem;
}
.src-title {
    font-weight: 600;
    font-size: 0.92rem;
    color: #0f172a;
}
.src-meta {
    color: #64748b;
    font-size: 0.8rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.src-excerpt {
    font-size: 0.86rem;
    line-height: 1.55;
    color: #334155;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.65rem 0.85rem;
    margin: 0.45rem 0;
    white-space: pre-wrap;
}

/* Cartes d'exemples */
div[data-testid="stButton"] > button {
    text-align: left;
    font-weight: 400;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
    border-color: #94a3b8;
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

if "historique" not in st.session_state:
    st.session_state.historique = []
if "question_exemple" not in st.session_state:
    st.session_state.question_exemple = None

# ---------------------------------------------------------------- barre latérale
modeles = demo_backend.modeles_disponibles()
with st.sidebar:
    st.markdown("### Mécano de poche")
    st.caption("Système RAG d'assistance technique sur notices constructeurs")

    st.markdown("**Véhicule**")
    options = ["Détection automatique"] + [m[1] for m in modeles]

    def libelle(o):
        for marque, modele, annees, langue, n in modeles:
            if o == modele:
                debut, fin = annees.split("-")
                periode = debut if debut == fin else annees
                code_lang = LANGUE_CODE.get(langue, langue.upper())
                return f"{marque} {modele} ({periode}) [{code_lang}]"
        return o

    choix_modele = st.selectbox("Modèle", options, format_func=libelle, label_visibility="collapsed")

    st.markdown("**Section de la notice**")
    cats_dispo = [c for c in CATEGORIES if c in demo_backend.categories_disponibles()]
    choix_cats = st.multiselect(
        "Catégories",
        cats_dispo,
        format_func=lambda c: CATEGORIES[c],
        placeholder="Toutes les sections",
        label_visibility="collapsed",
    )

    st.markdown("**Paramètres de recherche**")
    top_k = st.select_slider("Passages extraits (top-k)", options=[3, 5, 8, 10], value=5)
    voir_sources = st.toggle("Afficher les extraits sources", value=True)

    st.divider()
    total = sum(m[4] for m in modeles)
    st.caption(f"**Moteur :** {MODE}")
    st.caption(f"**Index :** {total} passages · {len(modeles)} manuels officiels")
    if st.button("Réinitialiser la session", use_container_width=True):
        st.session_state.historique = []
        st.rerun()
    st.caption(
        "Note de responsabilité : cet outil consulte les manuels d'utilisation officiels et ne "
        "se substitue pas au diagnostic d'un professionnel qualifié."
    )


# ---------------------------------------------------------------- affichage d'une réponse
def afficher_entites(ent, filtre_auto):
    chips = []
    if ent["modele"]:
        cls = "chip chip-auto" if filtre_auto else "chip"
        txt = f"Véhicule : {ent['modele'][0]} {ent['modele'][1]}"
        if filtre_auto:
            txt += " (détecté)"
        chips.append(f"<span class='{cls}'>{txt}</span>")
    for c in ent["codes_obd"]:
        chips.append(f"<span class='chip chip-obd'>OBD-II {c['code']} · {c['famille']}</span>")
    if chips:
        st.markdown(f"<div class='entity-bar'>{''.join(chips)}</div>", unsafe_allow_html=True)


def afficher_reponse(tour):
    res = tour["resultat"]
    afficher_entites(tour["entites"], tour["filtre_auto"])

    niveau = niveau_urgence(res)
    if res["info_absente"]:
        st.markdown(
            f"<div class='status-banner status-absent'>"
            f"<strong>Information hors notice :</strong> {res['reponse']}</div>",
            unsafe_allow_html=True,
        )
    else:
        if niveau == "urgent":
            st.markdown(
                "<div class='status-banner status-urgent'>"
                "<strong>Arrêt impératif recommandé :</strong> la documentation constructeur indique un risque "
                "pour la sécurité ou le moteur. Immobilisez le véhicule dès que possible en sécurité."
                "</div>",
                unsafe_allow_html=True,
            )
        elif niveau == "attention":
            st.markdown(
                "<div class='status-banner status-attention'>"
                "<strong>Contrôle atelier recommandé :</strong> la notice préconise de faire vérifier le véhicule "
                "par un réparateur qualifié."
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown(res["reponse"])

    if tour["codes_sans_source"]:
        st.markdown(
            "<div class='status-banner status-obd'>"
            "<strong>Note OBD-II :</strong> la table complète des codes défaut DTC ne figure pas dans le manuel "
            "conducteur. Un relevé par outil de diagnostic en atelier est nécessaire."
            "</div>",
            unsafe_allow_html=True,
        )

    if voir_sources and res["sources"]:
        suffixe = " (non retenus)" if res["info_absente"] else ""
        titre_exp = f"Extraits des notices consultés ({len(res['sources'])}{suffixe})"
        with st.expander(titre_exp, expanded=False):
            for i, s in enumerate(res["sources"], 1):
                pages = ", ".join(s["pages"])
                code_lang = LANGUE_CODE.get(s.get("langue", ""), s.get("langue", "").upper())
                cat_label = CATEGORIES.get(s.get("categorie", ""), s.get("categorie", ""))
                score_pct = min(max(float(s.get("score", 0.0)), 0.0), 1.0)

                st.markdown(
                    f"<div class='src-header'>"
                    f"<span class='src-title'>{i}. {s['titre']}</span>"
                    f"<span class='src-meta'>Pertinence : {score_pct:.0%}</span>"
                    f"</div>"
                    f"<div class='src-meta'>{s['marque']} {s['modele']} [{code_lang}] · "
                    f"Page(s) {pages} · Section : {cat_label}</div>",
                    unsafe_allow_html=True,
                )
                texte = s["texte"]
                extrait = texte if len(texte) < 1500 else texte[:1500] + "…"
                st.markdown(f"<div class='src-excerpt'>{extrait}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<span class='src-meta'>Document : <a href='{s['source_url']}' target='_blank'>{s['document']}</a></span>",
                    unsafe_allow_html=True,
                )
                if i < len(res["sources"]):
                    st.divider()


# ---------------------------------------------------------------- traitement d'une question
def traiter(question, modele_force=None):
    ent = ner.analyser(question)
    filtre_auto = False
    if choix_modele != "Détection automatique":
        modele = choix_modele
    elif modele_force:
        modele = modele_force
    elif ent["modele"]:
        modele, filtre_auto = ent["modele"][1], True
    else:
        modele = None
    with st.spinner("Recherche dans les notices et génération de la réponse…"):
        res = ASK(question, modele=modele, categories=choix_cats or None, top_k=top_k)
    codes_sans_source = bool(ent["codes_obd"]) and not any(
        s.get("categorie") == "code_obd" for s in res["sources"]
    )
    st.session_state.historique.append({
        "question": question,
        "resultat": res,
        "entites": ent,
        "filtre_auto": filtre_auto,
        "modele": modele,
        "codes_sans_source": codes_sans_source,
    })


# ---------------------------------------------------------------- page principale
st.subheader("Assistance technique sur notices constructeurs")
st.caption(
    "Interrogez les manuels d'utilisation officiels (Renault Clio 4, Hyundai i10, Ford EcoSport, Kia Picanto). "
    "Chaque réponse est sourcée avec le numéro de page correspondant."
)

if DEMO:
    st.info(
        "Mode de secours actif : recherche lexicale directe sans génération LLM.",
        icon=None,
    )

if not st.session_state.historique:
    st.markdown("##### Questions fréquentes")
    cols = st.columns(3)
    for i, (q, m) in enumerate(EXEMPLES):
        if cols[i % 3].button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.question_exemple = (q, m)
            st.rerun()

for tour in st.session_state.historique:
    with st.chat_message("user"):
        st.markdown(tour["question"])
        if tour["modele"] and not tour["filtre_auto"]:
            st.caption(f"Filtre véhicule : {tour['modele']}")
    with st.chat_message("assistant"):
        afficher_reponse(tour)

question = st.chat_input("Posez une question technique (ex. : Que faire si le témoin de pression d'huile s'allume sur Clio 4 ?)")
exemple = st.session_state.question_exemple
if exemple:
    st.session_state.question_exemple = None
    question_a_traiter, modele_force = exemple
elif question:
    question_a_traiter, modele_force = question, None
else:
    question_a_traiter = None

if question_a_traiter:
    traiter(question_a_traiter, modele_force)
    st.rerun()
