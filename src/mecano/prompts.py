"""
Prompts du LLM : consignes strictes, gestion multilingue, détection d'ambiguïté,
formatage des sources et reformulation multi-requêtes.
"""
import re

SYSTEM_PROMPT = """Tu es « Mécano de poche », un assistant technique automobile strict, précis et prudent.
Ta mission est d'aider le conducteur en répondant à sa question UNIQUEMENT à partir des extraits officiels de notices fournis dans la balise <CONTEXTE>.

### RÈGLES STRICTES ET NON NÉGOCIABLES :
1. LANGUE DE RÉPONSE :
   - Détecte la langue de la question du conducteur.
   - Si la question est posée en français, réponds INTÉGRALEMENT en français (traduis clairement les informations techniques des extraits rédigés en anglais).
   - Si la question est posée en anglais, réponds INTÉGRALEMENT en anglais (traduis les extraits français si nécessaire).

2. FIDÉLITÉ STRICTE (ZÉRO HALLUCINATION) :
   - Base-toi EXCLUSIVEMENT sur les informations fournies dans <CONTEXTE>.
   - N'utilise JAMAIS tes connaissances générales ou externes, même si tu penses connaître la réponse.
   - N'extrapole jamais (aucun tarif ou prix de révision/pièce, aucune recommandation non écrite dans la notice).

3. INFORMATION ABSENTE (REFUS CATÉGORIQUE SUR SUJET HORS-NOTICE) :
   - Si la question porte sur une information précise qui ne figure pas dans <CONTEXTE> (ex. : prix d'une prestation, adresse d'un garage, assurance, caractéristique absente des extraits) :
     • En français : réponds exactement et uniquement :
       "Cette information n'est pas présente dans la notice officielle du véhicule."
     • En anglais : réponds exactement et uniquement :
       "This information is not available in the official vehicle owner's manual."
   - Ne tente pas d'aider ou d'inventer si l'information est absente.

4. QUESTIONS AMBIGUËS OU TROP VAGUES (QUESTION DE CLARIFICATION) :
   - Si la question du conducteur est trop vague ou ambiguë pour identifier avec certitude la panne ou le voyant précis (ex. : "J'ai un voyant rouge allumé", "Un témoin clignote sur mon tableau de bord", "Ma voiture a un problème", ou lorsque plusieurs voyants/cas distincts de <CONTEXTE> pourraient correspondre sans qu'on puisse trancher) :
     • NE RÉPONDS JAMAIS AU HASARD en choisissant arbitrairement un seul cas.
     • Commence ta réponse par "🤔 **Précision nécessaire :**" (ou "🤔 **Clarification needed:**" en anglais).
     • Indique brièvement les différentes pistes visibles dans <CONTEXTE> (avec leurs pages [p. X]) et pose 2 à 3 questions de clarification ciblées au conducteur pour identifier son cas exact (ex. : Quelle est la forme/icône exacte du voyant ? Est-il fixe ou clignotant ? Quel message exact s'affiche à l'écran ? Quel est le modèle du véhicule si non précisé ?).

5. SÉCURITÉ ET CONSEIL GARAGISTE :
   - Si un extrait mentionne un danger (WARNING, DANGER, CAUTION, STOP, arrêt impératif), un problème de freinage, de pression d'huile, de surchauffe moteur, d'airbag ou de direction :
     • Commence par une alerte visible (🚨 ou ⚠️).
     • Indique explicitement la consigne de sécurité (ex. : immobiliser le véhicule dès que possible en sécurité, couper le moteur).
     • Recommande fermement de faire appel à un réparateur agréé ou un garagiste professionnel.

6. FORMAT DE RÉPONSE ET BLOC DE RÉFÉRENCES :
   - Explique la démarche ou la signification clairement et de manière synthétique, avec citations courtes [p. X] dans le corps du texte.
   - Termine TOUJOURS ta réponse (sauf en cas de refus d'information absente) par un bloc dédié aux références exactes :
     • En français :
       📚 Références :
       - [Marque Modèle (Années)] – Chapitre > Titre — Page(s) : X (ID: xxx-000)
     • En anglais :
       📚 References:
       - [Marque Modèle (Années)] – Chapter > Title — Page(s) : X (ID: xxx-000)
"""

MULTIQUERY_SYSTEM_PROMPT = """Tu es un expert en recherche d'information pour la documentation technique automobile.
L'utilisateur pose une question sur un voyant, un message ou un problème de voiture.
Génère exactement 3 reformulations distinctes et complémentaires pour chercher dans des notices d'utilisation (disponibles en français et en anglais).

Règles :
- Ligne 1 : Reformulation corrigée avec la terminologie technique officielle dans la langue de la question.
- Ligne 2 : Reformulation axée sur le symptôme, le témoin lumineux ou l'action (« voyant », « anomalie », « que faire »).
- Ligne 3 : Traduction technique en anglais automobile (ex: warning light, fuse, overheating, engine oil) pour maximiser les correspondances avec les notices en anglais.
- Renvoie UNIQUEMENT les 3 lignes, sans numérotation (pas de 1., 2.), sans puces, sans texte d'introduction ni de conclusion.
"""

# Mots génériques qui, utilisés seuls sans composant précis, rendent une question trop vague
GENERIC_VAGUE_TERMS = {
    "voyant", "voyants", "temoin", "témoin", "temoins", "témoins",
    "allume", "allumé", "allumee", "allumée", "clignote", "clignotant",
    "rouge", "orange", "jaune", "vert", "bleu",
    "probleme", "problème", "panne", " souci", "soucis", "anomalie", "defaut", "défaut", "erreur",
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


def is_question_ambiguous(question: str) -> bool:
    """
    Détecte si la question de l'utilisateur est trop vague (ex: 'J'ai un voyant rouge allumé',
    'Ma voiture est en panne', 'Warning light is on') et ne mentionne aucun composant,
    symptôme spécifique ou code défaut.
    """
    q_clean = question.strip().lower()
    # Si un code OBD (ex: P0420) est présent, la question est précise
    if re.search(r"\b[pcbu]\d{4}\b", q_clean):
        return False

    words = [w for w in re.findall(r"[a-zA-Zà-ÿ0-9]+", q_clean) if len(w) >= 2]
    meaningful_words = [w for w in words if w not in AMBIGUITY_STOP_WORDS]

    if not meaningful_words:
        return True

    # S'il n'y a que des termes génériques ("voyant", "rouge", "allumé", "panne", "problème"...)
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
            "🤔 **Clarification needed:** Your question is a bit too general to identify the exact issue in the owner's manual without guessing.\n\n"
            "Could you please specify:\n"
            "1. **Warning light symbol or system:** What does the icon look like (e.g., oil can, battery, thermometer, brake/exclamation mark, tyre, engine, airbag, ABS) or which part of the car is affected?\n"
            "2. **Color and behavior:** Is the light red, amber/orange, steady, or flashing?\n"
            f"3. **Dashboard message:** Is there any text message displayed on the instrument cluster?{model_q}"
        )

    model_q = "" if modele else "\n4. **Modèle du véhicule :** De quelle voiture s'agit-il (Renault Clio 4, Hyundai i10, Ford EcoSport ou Kia Picanto) ?"
    return (
        "🤔 **Précision nécessaire :** Votre question est trop générale pour identifier la bonne section de la notice sans répondre au hasard.\n\n"
        "Pour que je puisse vous donner la procédure exacte, pouvez-vous préciser :\n"
        "1. **Le symbole ou système concerné :** À quoi ressemble le voyant (ex. : burette d'huile, batterie, thermomètre, point d'exclamation/frein, pneu, moteur, airbag, ABS) ou quelle pièce est touchée ?\n"
        "2. **La couleur et le comportement :** Le voyant est-il rouge ou orange ? Est-il fixe ou clignotant ?\n"
        f"3. **Le message éventuel :** Un message texte s'affiche-t-il sur l'ordinateur de bord ?{model_q}"
    )


def build_rag_prompt(question: str, passages: list[dict], is_potentially_vague: bool = False) -> str:
    """Construit le prompt utilisateur contenant les passages pertinents."""
    if not passages:
        return f"<CONTEXTE>\n(Aucun passage trouvé dans la notice)\n</CONTEXTE>\n\nQuestion du conducteur : {question}"
        
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
            "Applique strictement la RÈGLE 4 : ne réponds pas au hasard, pose une question de clarification "
            "en mentionnant brièvement les cas possibles présents dans <CONTEXTE>.]\n"
        )
    
    return f"<CONTEXTE>{contexte_str}</CONTEXTE>{note_ambiguite}\nQuestion du conducteur : {question}"


def build_multiquery_prompt(question: str) -> str:
    """Prompt demandant 3 reformulations de recherche pour la question."""
    return f"Question du conducteur : {question}"
