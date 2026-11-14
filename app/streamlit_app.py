"""
Interface web de « Mécano de poche » (membre 3).

Lancement, depuis la racine du dépôt :
    streamlit run app/streamlit_app.py

Parcours : choisir sa voiture → cliquer sur le voyant allumé ou décrire le problème
→ lire la réponse (verdict d'urgence, consignes, pages de la notice citées).

Moteur : mecano.pipeline.ask() (embeddings + FAISS + Gemini + mémoire des 3 derniers échanges). Si le pipeline est
absent ou en erreur (quota, pas de réseau…), l'interface se replie sur
app/demo_backend.py (recherche par mots-clés, sans LLM), qui respecte le même contrat.
Variable d'environnement MECANO_HORS_LIGNE=1 : forcer ce mode de repli.
"""
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(APP))

import demo_backend  # noqa: E402
from mecano import ner  # noqa: E402
from telltales import VOYANTS, svg  # noqa: E402

FEEDBACK = ROOT / "evaluation" / "results" / "feedback.jsonl"


# ===================================================================== moteur RAG
def charger_pipeline():
    if os.environ.get("MECANO_HORS_LIGNE") == "1":
        return None
    try:
        from mecano import pipeline
        if callable(getattr(pipeline, "ask", None)):
            return pipeline.ask
    except Exception:
        pass
    return None


PIPELINE = charger_pipeline()

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
LANGUE = {"fr": "en français", "en": "en anglais", "ar": "en arabe"}
TOUTES = "__toutes__"

EXEMPLES = [
    "Mon moteur chauffe, que faire ?",
    "Comment vérifier le niveau d'huile ?",
    "Comment changer une roue crevée ?",
    "Comment démarrer avec des câbles si la batterie est à plat ?",
]

# Mots qui indiquent le niveau d'urgence (passage le plus pertinent + réponse)
URGENT = ["arrêt impératif", "arrêtez-vous", "arrêtez le moteur", "arrêtez votre moteur", "stoppez",
          "immobilisez", "stop the vehicle", "stop safely", "stop immediately", "do not drive",
          "ne redémarrez pas", "risque casse", "do not continue driving"]
ATTENTION = ["consultez", "représentant de la marque", "faites contrôler", "faites vérifier",
             "have the vehicle", "authorised", "authorized", "dealer", "as soon as possible",
             "au plus tôt", "service now", "garagiste"]

VERDICTS = {
    "urgent": ("Arrêtez-vous dès que possible",
               "La notice signale un risque pour votre sécurité ou pour le moteur."),
    "attention": ("Faites vérifier la voiture rapidement",
                  "Vous pouvez rouler prudemment jusqu'à un garage."),
    "info": ("Pas d'urgence signalée", "Suivez les consignes ci-dessous."),
    "absent": ("Ce n'est pas dans la notice",
               "Pour ne pas vous induire en erreur, l'assistant ne répond qu'à partir de la notice."),
    "question": ("Il manque une précision", "Répondez à la question ci-dessous."),
}


# ===================================================================== état
def init_state():
    ss = st.session_state
    ss.setdefault("voiture", None)        # valeur du champ "modele" ou TOUTES
    ss.setdefault("historique", [])
    ss.setdefault("en_attente", None)     # question posée avant le choix de la voiture
    ss.setdefault("a_traiter", None)      # question à envoyer au moteur à ce tour
    ss.setdefault("selftest", True)       # test d'allumage des voyants, une seule fois


init_state()
st.set_page_config(page_title="Mécano de poche", page_icon="🔧", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"<style>{(APP / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

VOITURES = demo_backend.modeles_disponibles()  # [(marque, modele, annees, langue, nb)]
INFO_VOITURE = {m[1]: m for m in VOITURES}


def periode(annees):
    debut, _, fin = annees.partition("-")
    return debut if debut == fin else f"{debut}–{fin}"


def nom_voiture(modele):
    if modele == TOUTES or modele is None:
        return "toutes les voitures"
    marque, mod, annees, _, _ = INFO_VOITURE[modele]
    return f"{marque} {mod}"


# ===================================================================== barre latérale
with st.sidebar:
    st.markdown("### Réglages de la recherche")
    choix_cats = st.multiselect("Chercher seulement dans", list(CATEGORIES),
                                format_func=lambda c: CATEGORIES[c], placeholder="Toute la notice")
    top_k = st.select_slider("Passages de la notice consultés (top-k)", options=[3, 5, 8, 10], value=5)
    voir_extraits = st.toggle("Montrer les extraits de la notice", value=True)
    hors_ligne = st.toggle("Mode hors ligne (sans LLM)", value=PIPELINE is None,
                           disabled=PIPELINE is None,
                           help="Recherche par mots-clés et extrait brut de la notice, sans appel à Gemini. "
                                "Utile si le quota de l'API est atteint pendant une démonstration.")
    st.divider()
    if PIPELINE is not None and not hors_ligne:
        st.markdown("<div class='etat' style='--c:#3DDC84'><span class='led'></span>"
                    "Assistant complet : FAISS + Gemini</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='etat' style='--c:#FFB020'><span class='led'></span>"
                    "Mode hors ligne : recherche par mots-clés</div>", unsafe_allow_html=True)
    total = sum(v[4] for v in VOITURES)
    st.markdown(f"<p class='side-note'>{total} passages issus de {len(VOITURES)} notices officielles.</p>",
                unsafe_allow_html=True)
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.historique = []
        st.session_state.en_attente = None
        st.rerun()
    st.markdown("<p class='side-note'>Cet assistant ne remplace pas un garagiste. En cas de doute "
                "sur la sécurité, arrêtez-vous et appelez un professionnel.</p>", unsafe_allow_html=True)


# ===================================================================== moteur : appel + repli
def historique_pour_pipeline(n=3):
    """Les derniers échanges, au format attendu par la mémoire conversationnelle du pipeline."""
    tours = [t for t in st.session_state.historique if t.get("type") == "reponse"][-n:]
    return [{"question": t["question"], "resultat": t["res"],
             "modele": None if t["voiture"] == TOUTES else t["voiture"]} for t in tours] or None


def interroger(question, modele):
    cats = choix_cats or None
    filtre = None if modele in (None, TOUTES) else modele
    historique = historique_pour_pipeline()
    erreur = None
    if PIPELINE is not None and not hors_ligne:
        try:
            try:
                res = PIPELINE(question, modele=filtre, categories=cats, top_k=top_k, historique=historique)
            except TypeError:  # ancienne version du pipeline, sans mémoire
                res = PIPELINE(question, modele=filtre, categories=cats, top_k=top_k)
            texte = reponse_de(res)
            if not texte.startswith("Erreur lors de la génération"):
                return res, None
            # Le pipeline renvoie l'erreur Gemini comme une réponse : on garde ses passages
            # et on affiche l'extrait brut le plus pertinent à la place
            erreur = texte
            if res.get("sources"):
                return dict(res, reponse=extrait_brut(res["sources"][0]), answer=None), erreur
        except Exception as e:  # quota, réseau, clé invalide…
            erreur = str(e)
    return demo_backend.ask(question, modele=filtre, categories=cats, top_k=top_k), erreur


def extrait_brut(passage):
    texte = passage.get("texte", "").strip()
    if len(texte) > 900:
        texte = texte[:900].rsplit(" ", 1)[0] + "…"
    return (f"*Extrait de la notice {passage.get('marque', '')} {passage.get('modele', '')}, "
            f"page {', '.join(passage.get('pages', []))} :*\n\n**{passage.get('titre', '')}**\n\n"
            + texte.replace("\n", "  \n"))


def reponse_de(res):
    return res.get("reponse") or res.get("answer") or ""


def niveau(res):
    if res.get("needs_clarification"):
        return "question"
    if res.get("info_absente") or not res.get("sources"):
        return "absent"
    top = res["sources"][0]
    texte = " ".join([top.get("titre", ""), top.get("texte", ""), reponse_de(res)]).lower()
    if any(m in texte for m in URGENT):
        return "urgent"
    if any(m in texte for m in ATTENTION):
        return "attention"
    return "info"


# ===================================================================== traitement d'une question
def demander(question, origine="saisie"):
    """Enregistre la question ; elle est traitée en fin de script, après l'affichage."""
    st.session_state.a_traiter = (question, origine)


def choisir_voiture(modele):
    st.session_state.voiture = modele
    if st.session_state.en_attente and modele is not None:
        demander(st.session_state.en_attente)
        st.session_state.en_attente = None


def traiter(question, origine):
    ent = ner.analyser(question)
    voiture = st.session_state.voiture
    filtre_auto = False
    if voiture is None and ent["modele"]:
        voiture, filtre_auto = ent["modele"][1], True
        st.session_state.voiture = voiture
    if voiture is None:
        # On demande la voiture avant de chercher : évite de mélanger les notices
        st.session_state.en_attente = question
        st.session_state.historique.append({"question": question, "type": "choix_voiture"})
        return
    debut = time.time()
    with st.spinner(f"Lecture de la notice {nom_voiture(voiture)}…"):
        res, erreur = interroger(question, voiture)
    st.session_state.historique.append({
        "type": "reponse", "question": question, "voiture": voiture, "res": res,
        "entites": ent, "filtre_auto": filtre_auto, "erreur": erreur, "top_k": top_k,
        "duree": round(time.time() - debut, 1),
    })


def enregistrer_avis(i):
    tour = st.session_state.historique[i]
    vote = st.session_state.get(f"avis_{i}")
    if vote is None:
        return
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "question": tour["question"],
            "voiture": tour["voiture"], "top_k": tour["top_k"],
            "avis": "utile" if vote == 1 else "pas utile",
            "sources": [s.get("id") for s in tour["res"].get("sources", [])],
            "info_absente": tour["res"].get("info_absente"),
        }, ensure_ascii=False) + "\n")


# ===================================================================== rendu
def esc(t):
    return html.escape(str(t), quote=True)


def nettoyer(texte):
    """Puces mal converties depuis certains PDF (ex. notice Picanto)."""
    return re.sub(r"\s*僅\s*", "\n• ", texte or "").strip()


REFS_RE = re.compile(r"\n\s*(?:\*\*|#+\s*)?\s*(Références|References|Sources)\s*:?\s*(?:\*\*)?\s*:?\s*\n.*\Z",
                     re.S | re.I)
CITE_RE = re.compile(r"\[(p\.\s?[^\[\]]{1,40})\]")


def mise_en_forme(texte):
    """Retire la liste de références (affichée à part) et transforme [p. 5.44] en étiquette."""
    texte = REFS_RE.sub("", "\n" + nettoyer(texte)).strip()
    return CITE_RE.sub(lambda m: f"<span class='cite'>{esc(m.group(1))}</span>", texte)


def afficher_verdict(niv):
    titre, sous = VERDICTS[niv]
    st.markdown(f"<div class='verdict {niv}'><span class='lampe'></span>"
                f"<div><b>{titre}</b><span>{sous}</span></div></div>", unsafe_allow_html=True)


def afficher_extraits(sources, absent=False):
    plusieurs = len({s.get("modele") for s in sources}) > 1
    titre = (f"Voir les {len(sources)} passages consultés (aucun ne répond à la question)" if absent
             else f"Voir les {len(sources)} extraits de la notice utilisés")
    with st.expander(titre):
        for s in sources:
            pages = ", ".join(s.get("pages", []))
            score = max(0.0, min(1.0, float(s.get("score", 0) or 0)))
            voiture = f"<span>{esc(s.get('marque', ''))} {esc(s.get('modele', ''))}</span>" if plusieurs else ""
            texte = nettoyer(s.get("texte", ""))
            if len(texte) > 1400:
                texte = texte[:1400].rsplit(" ", 1)[0] + "…"
            st.markdown(
                f"<div class='src'><div class='src-titre'>{esc(s.get('titre', ''))}</div>"
                f"<div class='src-meta'><span class='cite'>p. {esc(pages)}</span>{voiture}"
                f"<span title='Pertinence'><span class='jauge'><i style='width:{score * 100:.0f}%'></i></span>"
                f" pertinence {score:.0%}</span></div>"
                f"<div class='src-texte'>{esc(texte)}</div>"
                f"<a href='{esc(s.get('source_url', '#'))}' target='_blank'>Ouvrir la notice officielle</a></div>",
                unsafe_allow_html=True)


def afficher_tour(i, tour):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(tour["question"])
    with st.chat_message("assistant", avatar=":material/build:"):
        if tour["type"] == "choix_voiture":
            afficher_verdict("question")
            if st.session_state.en_attente == tour["question"]:
                st.markdown("Pour quelle voiture ? Les consignes changent d'un modèle à l'autre.")
                cols = st.columns(len(VOITURES) + 1)
                for c, (marque, mod, *_r) in zip(cols, VOITURES):
                    c.button(f"{marque} {mod}", key=f"ask_{i}_{mod}", use_container_width=True,
                             on_click=choisir_voiture, args=(mod,))
                cols[-1].button("Toutes", key=f"ask_{i}_all", use_container_width=True,
                                on_click=choisir_voiture, args=(TOUTES,))
            else:
                st.markdown("Voiture choisie : la réponse est juste en dessous.")
            return

        res, ent = tour["res"], tour["entites"]
        chips = [f"<span class='chip'>{esc(nom_voiture(tour['voiture']))}"
                 f"{' (reconnue dans la question)' if tour['filtre_auto'] else ''}</span>"]
        for c in ent["codes_obd"]:
            chips.append(f"<span class='chip obd'>Code {esc(c['code'])} : {esc(c['famille'])}</span>")
        st.markdown(f"<div class='chips'>{''.join(chips)}</div>", unsafe_allow_html=True)

        afficher_verdict(niveau(res))
        st.markdown(mise_en_forme(reponse_de(res)), unsafe_allow_html=True)
        if ent["codes_obd"] and not any(s.get("categorie") == "code_obd" for s in res.get("sources", [])):
            st.markdown("<p class='side-note'>Les codes défaut OBD-II ne figurent pas dans les notices des "
                        "constructeurs. Un garagiste équipé d'une valise de diagnostic peut le lire.</p>",
                        unsafe_allow_html=True)
        if tour["erreur"]:
            err = tour["erreur"]
            if "GEMINI_API_KEY" in err or "API key" in err:
                cause = ("La clé Gemini est absente ou invalide : vérifiez la ligne GEMINI_API_KEY "
                         "du fichier .env, puis relancez l'interface.")
            elif "429" in err or "quota" in err.lower():
                cause = "Le quota gratuit de Gemini est atteint : réessayez dans une minute."
            else:
                cause = "Gemini ne répond pas (connexion ?) : réessayez dans une minute."
            st.markdown(f"<p class='note-repli'>{esc(cause)} En attendant, voici l'extrait le plus "
                        "proche de la notice, sans reformulation.</p>", unsafe_allow_html=True)
        if voir_extraits and res.get("sources"):
            afficher_extraits(res["sources"], absent=bool(res.get("info_absente")))
        st.feedback("thumbs", key=f"avis_{i}", on_change=enregistrer_avis, args=(i,))


def panneau_voyants():
    classe = "selftest" if st.session_state.selftest else ""
    voit = st.session_state.voiture
    cible = "" if voit in (None, TOUTES) else f" sur ma {nom_voiture(voit)}"
    with st.container(key="telltales"):
        for ligne in (VOYANTS[:4], VOYANTS[4:]):
            cols = st.columns(4)
            for col, (cle, label, couleur, formulation) in zip(cols, ligne):
                with col.container(key=f"tt_{cle}"):
                    st.markdown(f"<div class='{classe}'><div class='tt' style='--c:{couleur}'>{svg(cle)}"
                                f"<div class='tt-label'>{esc(label)}</div></div></div>", unsafe_allow_html=True)
                    question = f"{formulation[0].upper()}{formulation[1:]} s'allume{cible}, que faire ?"
                    st.button(f"Voyant {label}", key=f"btn_tt_{cle}", on_click=demander,
                              args=(question, "voyant"))
    st.session_state.selftest = False


# ===================================================================== page
st.markdown(
    "<div class='mdp-hero'><h1 class='mdp-wordmark'>Mécano de poche</h1>"
    "<p class='mdp-tagline'>Un voyant s'allume ? Montrez-le, et l'assistant vous dit quoi faire, "
    "en s'appuyant uniquement sur la notice officielle de votre voiture.</p></div>",
    unsafe_allow_html=True)

# ---------------------------------------------------------------- 1. la voiture
voiture = st.session_state.voiture
if voiture is None:
    st.markdown("<div class='mdp-step'>Quelle voiture conduisez-vous ?</div>", unsafe_allow_html=True)
    with st.container(key="cars"):
        cols = st.columns(len(VOITURES))
        for col, (marque, mod, annees, langue, nb) in zip(cols, VOITURES):
            with col.container(key=f"car_{re.sub(r'[^a-z0-9]', '', mod.lower())}"):
                st.markdown(
                    f"<div class='car'><div class='car-marque'>{esc(marque)}</div>"
                    f"<div class='car-modele'>{esc(mod)}</div>"
                    f"<div class='car-info'>{periode(annees)}<br>Notice {LANGUE.get(langue, langue)}</div></div>",
                    unsafe_allow_html=True)
                st.button(f"Choisir {marque} {mod}", key=f"btn_car_{mod}", on_click=choisir_voiture, args=(mod,))
    st.button("Je ne sais pas, chercher dans toutes les notices", type="tertiary",
              on_click=choisir_voiture, args=(TOUTES,))
else:
    if voiture == TOUTES:
        titre, detail = "Toutes les voitures", "Les réponses peuvent mélanger plusieurs notices."
    else:
        marque, mod, annees, langue, _ = INFO_VOITURE[voiture]
        titre = f"{marque} {mod}"
        detail = f"{periode(annees)}, notice {LANGUE.get(langue, langue)}" + (
            " : posez vos questions en français, l'assistant traduit." if langue != "fr" else "")
    c1, c2 = st.columns([5, 1.4], vertical_alignment="center")
    c1.markdown(f"<div class='car-strip'><span class='led'></span><b>{esc(titre)}</b>"
                f"<span>{esc(detail)}</span></div>", unsafe_allow_html=True)
    c2.button("Changer de voiture", use_container_width=True, on_click=choisir_voiture, args=(None,))

# ---------------------------------------------------------------- 2. le voyant
historique = st.session_state.historique
if not historique:
    st.markdown("<div class='mdp-step'>Quel voyant s'est allumé ?"
                "<small>Rouge : danger. Orange : à faire vérifier.</small></div>", unsafe_allow_html=True)
    panneau_voyants()
    st.markdown("<div class='mdp-step'>Ou choisissez une question fréquente</div>", unsafe_allow_html=True)
    # Streamlit fait défiler les pages de chat vers le bas : on garde l'accueil en haut
    components.html("<script>setTimeout(() => {const m = window.parent.document.querySelector("
                    "'[data-testid=stAppScrollToBottomContainer]'); if (m) m.scrollTo({top: 0});}, 150);"
                    "</script>", height=0)
    cols = st.columns(2)
    for i, q in enumerate(EXEMPLES):
        with cols[i % 2].container(key=f"ex_{i}"):
            st.markdown(f"<div class='ex'>{esc(q)}</div>", unsafe_allow_html=True)
            st.button(q, key=f"btn_ex_{i}", on_click=demander, args=(q, "exemple"))
else:
    with st.expander("Un autre voyant s'est allumé ?"):
        panneau_voyants()

# ---------------------------------------------------------------- 3. la conversation
for i, tour in enumerate(historique):
    afficher_tour(i, tour)

saisie = st.chat_input("Décrivez le problème, ou posez une question de suivi (ex. : et s'il clignote ?)")
if saisie:
    demander(saisie)

if st.session_state.a_traiter:
    question, origine = st.session_state.a_traiter
    st.session_state.a_traiter = None
    traiter(question, origine)
    st.rerun()
