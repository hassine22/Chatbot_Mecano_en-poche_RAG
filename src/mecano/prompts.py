"""
Prompts du LLM : consignes strictes, gestion multilingue, mémoire conversationnelle (<HISTORIQUE>),
détection d'ambiguïté, formatage des sources et reformulation multi-requêtes.
"""
import re

SYSTEM_PROMPT = """Tu es « Mécano de poche », un assistant technique automobile strict, précis et prudent.
Ta mission est d'aider le conducteur en répondant à sa question UNIQUEMENT à partir des extraits officiels de notices fournis dans la balise <CONTEXTE>.

### RÈGLES STRICTES ET NON NÉGOCIABLES :
1. LANGUE DE RÉPONSE :
   - Détecte la langue de la question actuelle du conducteur.
   - Si la question est posée en français, réponds INTÉGRALEMENT en français (traduis clairement les informations techniques des extraits rédigés en anglais).
   - Si la question est posée en anglais, réponds INTÉGRALEMENT en anglais (traduis les extraits français si nécessaire).

2. MÉMOIRE CONVERSATIONNELLE (<HISTORIQUE>) :
   - Si une balise <HISTORIQUE> est présente, utilise-la pour comprendre le fil de la discussion (ex. : à quel véhicule, quel voyant ou quelle pièce fait référence un pronom comme « il », « ça », « et s'il clignote ? », ou la réponse du conducteur à ta question de clarification précédente).
   - En revanche, toutes les informations techniques factuelles et les numéros de page [p. X] doivent provenir EXCLUSIVEMENT de <CONTEXTE>.

3. FIDÉLITÉ STRICTE (ZÉRO HALLUCINATION) :
   - Base-toi EXCLUSIVEMENT sur les informations fournies dans <CONTEXTE>.
   - N'utilise JAMAIS tes connaissances générales ou externes, même si tu penses connaître la réponse.
   - N'extrapole jamais (aucun tarif ou prix de révision/pièce, aucune recommandation non écrite dans la notice).

4. INFORMATION ABSENTE (REFUS CATÉGORIQUE SUR SUJET HORS-NOTICE) :
   - Si la question porte sur une information précise qui ne figure pas dans <CONTEXTE> (ex. : prix d'une prestation, adresse d'un garage, assurance, code défaut OBD-II tel que P0420, caractéristique absente des extraits) :
     • En français : réponds exactement et uniquement :
       "Cette information n'est pas présente dans la notice officielle du véhicule (information absente). Nous vous conseillons de contacter un garagiste ou un concessionnaire."
     • En anglais : réponds exactement et uniquement :
       "This information is not available in the official vehicle owner's manual. Please contact an authorized dealer or qualified mechanic."
   - Ne tente pas d'aider ou d'inventer si l'information est absente.

5. QUESTIONS AMBIGUËS OU TROP VAGUES (QUESTION DE CLARIFICATION) :
   - Si la question du conducteur (même en tenant compte de <HISTORIQUE>) reste trop vague ou ambiguë pour identifier avec certitude la panne ou le voyant précis :
     • NE RÉPONDS JAMAIS AU HASARD en choisissant arbitrairement un seul cas.
     • Commence ta réponse par "**Précision nécessaire :**" (ou "**Clarification needed:**" en anglais).
     • Indique brièvement les différentes pistes visibles dans <CONTEXTE> (avec leurs pages [p. X]) et pose 2 à 3 questions de clarification ciblées au conducteur pour identifier son cas exact.

6. SÉCURITÉ ET CONSEIL GARAGISTE :
   - Si un extrait mentionne un danger (WARNING, DANGER, CAUTION, STOP, arrêt impératif), un problème de freinage, de pression d'huile, de surchauffe moteur, d'airbag ou de direction :
     • Commence par une mention explicite "**ALERTE DE SÉCURITÉ :**" (ou "**SAFETY WARNING:**" en anglais). N'utilise aucun emoji.
     • Indique explicitement la consigne de sécurité (ex. : immobiliser le véhicule dès que possible en sécurité, couper le moteur).
     • Recommande fermement de faire appel à un réparateur agréé ou un garagiste professionnel.

7. FORMAT DE RÉPONSE ET BLOC DE RÉFÉRENCES :
   - Adopte un ton technique, sobre et professionnel. N'utilise AUCUN emoji dans ta réponse.
   - Explique la démarche ou la signification clairement et de manière synthétique, avec citations courtes [p. X] dans le corps du texte.
   - Termine TOUJOURS ta réponse (sauf en cas de refus d'information absente) par un bloc dédié aux références exactes :
     • En français :
       **Références :**
       - [Marque Modèle (Années)] – Chapitre > Titre — Page(s) : X (ID: xxx-000)
     • En anglais :
       **References:**
       - [Marque Modèle (Années)] – Chapter > Title — Page(s) : X (ID: xxx-000)
"""

MULTIQUERY_SYSTEM_PROMPT = """Tu es un expert en recherche d'information pour la documentation technique automobile.
L'utilisateur pose une question (parfois précédée d'un court <HISTORIQUE> de conversation) sur un voyant, un message ou un problème de voiture.
Si un <HISTORIQUE> est présent, résous les pronoms ou références implicites (ex. « et s'il clignote ? », « c'est une burette d'huile ») afin que chaque reformulation soit autonome et complète.
Génère exactement 3 reformulations distinctes et complémentaires pour chercher dans des notices d'utilisation (disponibles en français et en anglais).

Règles :
- Ligne 1 : Reformulation autonome corrigée avec la terminologie technique officielle dans la langue de la question.
- Ligne 2 : Reformulation autonome axée sur le symptôme, le témoin lumineux ou l'action (« voyant », « anomalie », « que faire »).
- Ligne 3 : Traduction technique autonome en anglais automobile (ex: warning light, fuse, overheating, engine oil) pour maximiser les correspondances avec les notices en anglais.
- Renvoie UNIQUEMENT les 3 lignes, sans numérotation (pas de 1., 2.), sans puces, sans texte d'introduction ni de conclusion.
"""

# Mots génériques qui, utilisés seuls sans composant précis, rendent une question trop vague
GENERIC_VAGUE_TERMS = {
    "voyant", "voyants", "temoin", "témoin", "temoins", "témoins",
    "allume", "allumé", "allumee", "allumée", "clignote", "clignotant",
    "rouge", "orange", "jaune", "vert", "bleu",
    "probleme", "problème", "panne", "souci", "soucis", "anomalie", "defaut", "défaut", "erreur",
    "voiture", "auto", "vehicule", "véhicule", "tableau", "bord", "ecran", "écran",
    "affiche", "affiché", "message", "alerte", "symbole", "icone", "icône", "dessin",
    "bruit", "izarre", "bizarre", "marche", "plus", "pas", "aide", "aider", "quoi", "faire",
    "warning", "light", "lights", "indicator", "lamp", "on", "blinking", "flashing", "illuminates",
    "red", "yellow", "amber", "orange", "green", "car", "vehicle", "problem", "issue", "fault",
    "dashboard", "cluster", "message", "display", "help", "symbol", "icon",
    "clio", "clio4", "i10", "picanto", "ecosport", "renault", "hyundai", "kia", "ford"
}

# Mots vides ignorés lors de l'analyse d'ambiguïté
AMBIGUITY_STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "ce", "cet", "cette", "ces", "je", "tu", "il", "elle", "on", "nous",
    "vous", "ils", "ai", "as", "a", "est", "sont", "sur", "dans", "avec", "pour", "par", "en",
    "au", "aux", "que", "qui", "quoi", "quand", "comment", "pourquoi", "si", "ne", "se", "y",
    "et", "ou", "mais", "donc", "car", "ni", "faire", "fait", "faut", "dois", "peut", "veut",
    "dire", "signifie", "svp", "stp", "bonjour", "salut", "merci",
    "i", "my", "the", "a", "an", "is", "are", "was", "have", "has", "what", "how", "when", "why",
    "should", "do", "does", "if", "in", "on", "at", "to", "for", "with", "of", "and", "or", "it",
    "this", "that", "there", "mean", "means"
}


def extract_turn_pair(turn: dict) -> tuple[str, str]:
    """Extrait (question_utilisateur, reponse_assistant) depuis un tour d'historique."""
    if not isinstance(turn, dict):
        return "", ""
    q = str(turn.get("question") or turn.get("user") or "").strip()
    if "resultat" in turn and isinstance(turn["resultat"], dict):
        r = str(turn["resultat"].get("reponse") or turn["resultat"].get("answer") or "").strip()
    else:
        r = str(turn.get("reponse") or turn.get("answer") or turn.get("assistant") or "").strip()
    return q, r


def format_history(historique: list[dict] | None, max_turns: int = 3) -> str:
    """Formate les derniers échanges en un bloc <HISTORIQUE> compact."""
    if not historique:
        return ""
    recent = historique[-max_turns:]
    lines = []
    for idx, turn in enumerate(recent, 1):
        q, r = extract_turn_pair(turn)
        if not q:
            continue
        # Tronquer la réponse de l'assistant dans l'historique pour économiser les tokens
        r_short = r.split("**Références :**")[0].split("**References:**")[0].strip()
        if len(r_short) > 400:
            r_short = r_short[:400].rsplit(" ", 1)[0] + "…"
        lines.append(f"[Tour {idx}]\nConducteur : {q}\nAssistant : {r_short}")
    if not lines:
        return ""
    return "<HISTORIQUE>\n" + "\n\n".join(lines) + "\n</HISTORIQUE>"


def contextualize_question_with_history(question: str, historique: list[dict] | None) -> str:
    """
    Construit une requête de recherche enrichie avec les questions précédentes de l'historique
    lorsque l'utilisateur répond à une clarification ou pose une question de suivi.
    """
    if not historique:
        return question
    prev_questions = []
    for turn in historique[-2:]:
        q_prev, _ = extract_turn_pair(turn)
        if q_prev:
            prev_questions.append(q_prev)
    if not prev_questions:
        return question

    # Si le tour précédent demandait une clarification, ou si la nouvelle question est courte / de suivi
    last_turn = historique[-1] if isinstance(historique[-1], dict) else {}
    last_res = last_turn.get("resultat", {}) if isinstance(last_turn.get("resultat"), dict) else last_turn
    was_clarification = bool(last_res.get("needs_clarification"))

    q_words = re.findall(r"\w+", question.lower())
    follow_up_starters = {"et", "si", "c", "est", "il", "elle", "ça", "ca", "oui", "non", "plutot", "plutôt", "aussi", "pour", "dans", "quand", "what", "and", "if", "it", "yes", "no"}
    is_short_or_followup = len(q_words) <= 9 or (q_words and q_words[0] in follow_up_starters) or is_question_ambiguous(question)

    if was_clarification or is_short_or_followup:
        return f"{prev_questions[-1]} — {question}"
    return question


def is_question_ambiguous(question: str) -> bool:
    """
    Détecte si la question de l'utilisateur est trop vague (ex: 'J'ai un voyant rouge allumé',
    'Ma voiture est en panne', 'Warning light is on') et ne mentionne aucun composant,
    symptôme spécifique ou code défaut.
    """
    q_clean = question.strip().lower()
    if re.search(r"\b[pcbu]\d{4}\b", q_clean):
        return False

    words = [w for w in re.findall(r"[a-zA-Zà-ÿ0-9]+", q_clean) if len(w) >= 2]
    meaningful_words = [w for w in words if w not in AMBIGUITY_STOP_WORDS]

    if not meaningful_words:
        return True

    specific_terms = [w for w in meaningful_words if w not in GENERIC_VAGUE_TERMS]
    if len(specific_terms) == 0:
        return True

    return False


def build_clarification_response(question: str, modele: str = None) -> str:
    """
    Construit une question de clarification directe dans la langue de l'utilisateur
    lorsque la question initiale est trop vague pour chercher un passage précis.
    """
    q_lower = question.lower()
    english_markers = {"what", "how", "why", "my", "the", "warning", "light", "car", "is", "on", "blinking", "problem", "issue", "help"}
    words = set(re.findall(r"[a-zA-Z]+", q_lower))
    is_english = len(words & english_markers) >= 2 and not any(w in words for w in ("voyant", "temoin", "voiture", "panne", "probleme", "que", "faire"))

    if is_english:
        model_q = "" if modele else "\n4. **Vehicle model:** Which car are you driving (Renault Clio 4, Hyundai i10, Ford EcoSport, or Kia Picanto)?"
        return (
            "**Clarification needed:** Your question is a bit too general to identify the exact issue in the owner's manual without guessing.\n\n"
            "Could you please specify:\n"
            "1. **Warning light symbol or system:** What does the icon look like (e.g., oil can, battery, thermometer, brake/exclamation mark, tyre, engine, airbag, ABS) or which part of the car is affected?\n"
            "2. **Color and behavior:** Is the light red, amber/orange, steady, or flashing?\n"
            f"3. **Dashboard message:** Is there any text message displayed on the instrument cluster?{model_q}"
        )

    model_q = "" if modele else "\n4. **Modèle du véhicule :** De quelle voiture s'agit-il (Renault Clio 4, Hyundai i10, Ford EcoSport ou Kia Picanto) ?"
    return (
        "**Précision nécessaire :** Votre question est trop générale pour identifier la bonne section de la notice sans répondre au hasard.\n\n"
        "Pour que je puisse vous donner la procédure exacte, pouvez-vous préciser :\n"
        "1. **Le symbole ou système concerné :** À quoi ressemble le voyant (ex. : burette d'huile, batterie, thermomètre, point d'exclamation/frein, pneu, moteur, airbag, ABS) ou quelle pièce est touchée ?\n"
        "2. **La couleur et le comportement :** Le voyant est-il rouge ou orange ? Est-il fixe ou clignotant ?\n"
        f"3. **Le message éventuel :** Un message texte s'affiche-t-il sur l'ordinateur de bord ?{model_q}"
    )


def build_rag_prompt(
    question: str,
    passages: list[dict],
    is_potentially_vague: bool = False,
    historique: list[dict] | None = None,
) -> str:
    """Construit le prompt utilisateur contenant l'historique éventuel et les passages pertinents."""
    bloc_historique = format_history(historique)
    prefix_hist = f"{bloc_historique}\n\n" if bloc_historique else ""

    if not passages:
        return f"{prefix_hist}<CONTEXTE>\n(Aucun passage trouvé dans la notice)\n</CONTEXTE>\n\nQuestion du conducteur : {question}"

    lignes_contexte = []
    for i, p in enumerate(passages, 1):
        pid = p.get("id", f"passage-{i}")
        marque = p.get("marque", "")
        modele = p.get("modele", "")
        annees = p.get("annees", "")
        chapitre = p.get("chapitre", "")
        titre = p.get("titre", "")
        pages = ", ".join(p.get("pages", []))
        categorie = p.get("categorie", "")
        texte = p.get("texte", "").strip()

        entete = f"[Passage {i} | ID: {pid} | Véhicule: {marque} {modele} ({annees}) | Catégorie: {categorie} | Page(s): {pages}]"
        info_titre = f"Chapitre : {chapitre}\nTitre : {titre}"
        bloc = f"{entete}\n{info_titre}\nContenu :\n{texte}"
        lignes_contexte.append(bloc)

    contexte_str = "\n\n" + ("-" * 40) + "\n\n".join(lignes_contexte) + "\n\n"

    note_ambiguite = ""
    if is_potentially_vague:
        note_ambiguite = (
            "\n[Note importante : La question du conducteur semble très générale. "
            "Applique strictement la RÈGLE 5 : ne réponds pas au hasard, pose une question de clarification "
            "en mentionnant brièvement les cas possibles présents dans <CONTEXTE>.]\n"
        )

    return f"{prefix_hist}<CONTEXTE>{contexte_str}</CONTEXTE>{note_ambiguite}\nQuestion du conducteur : {question}"


def build_multiquery_prompt(question: str, historique: list[dict] | None = None) -> str:
    """Prompt demandant 3 reformulations de recherche autonomes pour la question."""
    bloc_historique = format_history(historique, max_turns=2)
    if bloc_historique:
        return f"{bloc_historique}\n\nNouvelle question du conducteur : {question}"
    return f"Question du conducteur : {question}"
