"""
Détection d'entités dans la question de l'utilisateur (bonus NER).

- codes défaut OBD-II : P0420, p0301, C1234, U0100…
- modèle de voiture cité : « ma clio », « Picanto », « i 10 », « eco sport »…

Utilisé par l'interface pour appliquer automatiquement le filtre par modèle
et pour afficher les entités détectées.
"""
import re
import unicodedata

# Code OBD-II : lettre (P moteur, C châssis, B carrosserie, U réseau) + 4 caractères hexadécimaux
OBD_RE = re.compile(r"\b([PCBU])\s?([0-3][0-9A-F]{3})\b", re.IGNORECASE)

OBD_FAMILLES = {
    "P": "groupe motopropulseur (moteur, boîte)",
    "C": "châssis (freins, direction, suspension)",
    "B": "carrosserie (airbags, confort)",
    "U": "réseau de communication entre calculateurs",
}

# Valeur = (marque, modele) tels qu'ils sont écrits dans le corpus
MODELES = {
    "clio": ("Renault", "Clio 4"),
    "ecosport": ("Ford", "EcoSport"),
    "picanto": ("Kia", "Picanto"),
    "i10": ("Hyundai", "i10"),
}
# Variantes d'écriture → clé de MODELES (comparées sans accents ni espaces)
ALIAS = {
    "clio": "clio", "clio4": "clio", "cliov4": "clio", "cliophase2": "clio",
    "ecosport": "ecosport", "ecosports": "ecosport",
    "picanto": "picanto", "kiapicanto": "picanto",
    "i10": "i10", "grandi10": "i10", "hyundaii10": "i10",
}


def _normaliser(texte):
    texte = unicodedata.normalize("NFD", texte.lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def detecter_codes_obd(question):
    """Renvoie la liste des codes OBD-II trouvés, en majuscules et sans doublon."""
    codes = []
    for lettre, chiffres in OBD_RE.findall(question):
        code = f"{lettre.upper()}{chiffres.upper()}"
        if code not in codes:
            codes.append(code)
    return codes


def detecter_modele(question):
    """Renvoie (marque, modele) du premier modèle cité, ou None."""
    q = _normaliser(question)
    compact = re.sub(r"[\s\-_]", "", q)
    meilleur = None
    for alias, cle in ALIAS.items():
        pos = compact.find(alias)
        if pos != -1 and (meilleur is None or pos < meilleur[0]):
            meilleur = (pos, cle)
    return MODELES[meilleur[1]] if meilleur else None


def analyser(question):
    """Point d'entrée : toutes les entités détectées dans la question."""
    codes = detecter_codes_obd(question)
    return {
        "modele": detecter_modele(question),
        "codes_obd": [{"code": c, "famille": OBD_FAMILLES[c[0]]} for c in codes],
    }
