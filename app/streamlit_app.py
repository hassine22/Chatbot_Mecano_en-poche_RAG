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
            return pipeline.ask, "RAG complet (embeddings + LLM)"
    except Exception:
        pass
    return demo_backend.ask, "Démo (mots-clés, sans LLM)"


ASK, MODE = charger_backend()
DEMO = ASK is demo_backend.ask

# ---------------------------------------------------------------- libellés
CATEGORIES = {
    "voyant": "🚨 Voyants du tableau de bord",
    "message_tableau_de_bord": "💬 Messages affichés",
    "anomalie": "🔍 Pannes et anomalies",
    "demarrage_moteur": "🔑 Démarrage et moteur",
    "aides_conduite": "🛞 ABS, ESP, pression des pneus",
    "boite_automatique": "⚙️ Boîte automatique",
    "direction": "🎯 Direction",
    "carburant": "⛽ Carburant",
    "entretien": "🛢️ Entretien et niveaux",
    "pneus_roues": "🛞 Pneus et roues",
    "fusibles": "🔌 Fusibles",
    "electricite_batterie": "🔋 Batterie et électricité",
    "remorquage": "🚛 Remorquage",
    "identification": "🪪 Identification du véhicule",
    "caracteristiques_moteur": "📋 Caractéristiques moteur",
    "code_obd": "🧾 Codes défaut OBD-II",
}
DRAPEAU = {"fr": "🇫🇷", "en": "🇬🇧", "ar": "🇹🇳"}

EXEMPLES = [
    ("Le voyant de pression d'huile s'allume sur ma Clio 4, que faire ?", None),
    ("Mon moteur chauffe, que faire ?", "Clio 4"),
    ("Comment changer une roue crevée sur l'EcoSport ?", None),
    ("Comment démarrer avec des câbles si la batterie de ma i10 est à plat ?", None),
    ("Comment vérifier le niveau d'huile moteur de la Picanto ?", None),
    ("Combien coûte une vidange à Tunis ?", None),
]

# Mots indiquant le niveau d'urgence dans les passages (français et anglais)
URGENT = ["arrêt impératif", "arrêtez-vous", "arrêtez votre moteur", "stop the vehicle",
          "stop safely", "stop immediately", "do not drive", "ne redémarrez pas", "risque casse"]
ATTENTION = ["consultez", "représentant de la marque", "faites contrôler", "have the vehicle",
             "authorized", "dealer", "as soon as possible", "au plus tôt", "service now"]


def niveau_urgence(resultat):
    """Indication heuristique, déduite du passage le plus pertinent."""
    if resultat["info_absente"] or not resultat["sources"]:
        return None
    texte = (resultat["sources"][0]["titre"] + " " + resultat["sources"][0]["texte"]).lower()
    if any(m in texte for m in URGENT):
        return "urgent"
    if any(m in texte for m in ATTENTION):
        return "attention"
    return "info"


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Mécano de poche", page_icon="🔧", layout="wide")
st.markdown("""
<style>
.chip {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:12px;
       font-size:0.82rem; background:#f1f5f9; border:1px solid #e2e8f0;}
.chip-obd {background:#fff7ed; border-color:#fdba74;}
.chip-auto {background:#eff6ff; border-color:#93c5fd;}
.src-meta {color:#64748b; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

if "historique" not in st.session_state:
    st.session_state.historique = []
if "question_exemple" not in st.session_state:
    st.session_state.question_exemple = None

# ---------------------------------------------------------------- barre latérale
modeles = demo_backend.modeles_disponibles()
with st.sidebar:
    st.title("🔧 Mécano de poche")
    st.caption("Votre notice de voiture, en questions-réponses.")

    st.subheader("Ma voiture")
    options = ["Détection automatique"] + [m[1] for m in modeles]

    def libelle(o):
        for marque, modele, annees, langue, n in modeles:
            if o == modele:
                debut, fin = annees.split("-")
                periode = debut if debut == fin else annees
                return f"{marque} {modele} ({periode}) {DRAPEAU.get(langue, '')}"
        return "🔎 " + o

    choix_modele = st.selectbox("Modèle", options, format_func=libelle, label_visibility="collapsed")

    st.subheader("Rechercher dans")
    cats_dispo = [c for c in CATEGORIES if c in demo_backend.categories_disponibles()]
    choix_cats = st.multiselect("Catégories", cats_dispo, format_func=lambda c: CATEGORIES[c],
                                placeholder="Toute la notice", label_visibility="collapsed")

    st.subheader("Réglages")
    top_k = st.select_slider("Nombre de passages consultés (top-k)", options=[3, 5, 8, 10], value=5)
    voir_sources = st.toggle("Afficher les passages des notices", value=True)

    st.divider()
    st.caption(f"**Mode :** {MODE}")
    total = sum(m[4] for m in modeles)
    st.caption(f"**Corpus :** {total} passages, {len(modeles)} notices officielles")
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.historique = []
        st.rerun()
    st.caption("⚠️ Cet assistant ne remplace pas un garagiste. En cas de doute sur la sécurité, "
               "arrêtez-vous et faites appel à un professionnel.")

# ---------------------------------------------------------------- affichage d'une réponse
def afficher_entites(ent, filtre_auto):
    chips = []
    if ent["modele"]:
        cls = "chip chip-auto" if filtre_auto else "chip"
        txt = f"🚗 {ent['modele'][0]} {ent['modele'][1]}"
        if filtre_auto:
            txt += " · filtre appliqué"
        chips.append(f"<span class='{cls}'>{txt}</span>")
    for c in ent["codes_obd"]:
        chips.append(f"<span class='chip chip-obd'>🧾 Code {c['code']} · {c['famille']}</span>")
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def afficher_reponse(tour):
    res = tour["resultat"]
    afficher_entites(tour["entites"], tour["filtre_auto"])

    niveau = niveau_urgence(res)
    if res["info_absente"]:
        st.warning("**Information absente des notices.** " + res["reponse"], icon="🤷")
    else:
        if niveau == "urgent":
            st.error("**Urgent** : la notice demande de s'arrêter dès que possible, "
                     "en sécurité. Ne continuez pas à rouler.", icon="🛑")
        elif niveau == "attention":
            st.warning("**À faire vérifier** : la notice recommande de consulter "
                       "un professionnel rapidement.", icon="🔧")
        st.markdown(res["reponse"])

    if tour["codes_sans_source"]:
        st.info("Les codes défaut OBD-II ne figurent pas dans les notices du constructeur : "
                "faites lire le code complet par un garagiste équipé d'une valise de diagnostic.",
                icon="🧾")

    if voir_sources and res["sources"]:
        titre = "📚 Passages consultés" + (" (aucun ne répond à la question)" if res["info_absente"] else "")
        with st.expander(f"{titre} — {len(res['sources'])}", expanded=False):
            for i, s in enumerate(res["sources"], 1):
                pages = ", ".join(s["pages"])
                st.markdown(f"**{i}. {s['titre']}**")
                st.markdown(
                    f"<span class='src-meta'>{s['marque']} {s['modele']} "
                    f"{DRAPEAU.get(s['langue'], '')} · page {pages} · "
                    f"{CATEGORIES.get(s['categorie'], s['categorie'])} · pertinence {s['score']:.0%}</span>",
                    unsafe_allow_html=True)
                st.progress(min(max(float(s["score"]), 0.0), 1.0))
                texte = s["texte"]
                st.text(texte if len(texte) < 1500 else texte[:1500] + "…")
                st.markdown(f"<span class='src-meta'>Source : [{s['document']}]({s['source_url']})</span>",
                            unsafe_allow_html=True)
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
    with st.spinner("Je consulte les notices…"):
        res = ASK(question, modele=modele, categories=choix_cats or None, top_k=top_k)
    codes_sans_source = bool(ent["codes_obd"]) and not any(
        s.get("categorie") == "code_obd" for s in res["sources"])
    st.session_state.historique.append({
        "question": question, "resultat": res, "entites": ent,
        "filtre_auto": filtre_auto, "modele": modele, "codes_sans_source": codes_sans_source,
    })


# ---------------------------------------------------------------- page principale
st.header("Un voyant s'allume ? Une question sur votre voiture ?")
st.caption("Les réponses viennent uniquement des notices officielles des constructeurs, "
           "avec la page à consulter.")
if DEMO:
    st.info("Mode démo : la recherche se fait par mots-clés et la réponse est l'extrait le plus "
            "pertinent de la notice. Le vrai RAG (embeddings + LLM) sera branché automatiquement "
            "dès que `src/mecano/pipeline.py` sera prêt.", icon="🧪")

if not st.session_state.historique:
    st.markdown("**Exemples de questions :**")
    cols = st.columns(3)
    for i, (q, m) in enumerate(EXEMPLES):
        if cols[i % 3].button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.question_exemple = (q, m)
            st.rerun()

for tour in st.session_state.historique:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(tour["question"])
        if tour["modele"] and not tour["filtre_auto"]:
            st.caption(f"Voiture : {tour['modele']}")
    with st.chat_message("assistant", avatar="🔧"):
        afficher_reponse(tour)

question = st.chat_input("Ex. : Le voyant batterie reste allumé sur ma Clio 4, c'est grave ?")
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
