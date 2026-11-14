"""
Voyants du tableau de bord : pictogrammes SVG dessinés pour le projet
(formes génériques inspirées de la norme ISO 2575, pas de logo constructeur).

Chaque voyant a une couleur réelle de tableau de bord :
  rouge  = danger, s'arrêter
  orange = anomalie, faire vérifier
"""

ROUGE = "#FF4D3D"
ORANGE = "#FFB020"

_SVG = '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.6" ' \
       'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>'

ICONES = {
    # burette d'huile avec une goutte
    "huile": '<path d="M6 22h6l4-4h14l8 6 6-3-8 11H14l-2-4H6z"/><path d="M22 18v-4h4"/>'
             '<path d="M42 30c0 2-1.4 3.5-3 3.5S36 32 36 30c0-1.6 3-5 3-5s3 3.4 3 5z"/>',
    # thermomètre plongé dans le liquide
    "temperature": '<path d="M24 6v20"/><circle cx="24" cy="30" r="5"/><path d="M24 12h6M24 18h6"/>'
                   '<path d="M6 40c3 0 3-2 6-2s3 2 6 2 3-2 6-2 3 2 6 2 3-2 6-2 3 2 6 2"/>',
    # batterie avec bornes + et −
    "batterie": '<rect x="6" y="14" width="36" height="24" rx="2"/><path d="M12 14v-4h6v4M30 14v-4h6v4"/>'
                '<path d="M12 26h6M15 23v6M30 26h6"/>',
    # cercle avec point d'exclamation entre parenthèses (freins)
    "freins": '<circle cx="24" cy="24" r="12"/><path d="M8 13a19 19 0 0 0 0 22M40 13a19 19 0 0 1 0 22"/>'
              '<path d="M24 17v9"/><path d="M24 31v.5"/>',
    # silhouette de moteur
    "moteur": '<path d="M8 20h4v-4h8v-4h10v4h6l4 4h4v14h-4l-4 4H18l-6-6H8z"/><path d="M4 24v8M18 12h10"/>',
    # pneu en coupe avec point d'exclamation
    "pneus": '<path d="M12 38c-4-4-6-9-6-14 0-6 3-11 6-14h24c3 3 6 8 6 14 0 5-2 10-6 14"/>'
             '<path d="M10 38h28"/><path d="M24 16v10"/><path d="M24 31v.5"/>',
    # ABS dans un cercle
    "abs": '<circle cx="24" cy="24" r="13"/><path d="M8 13a19 19 0 0 0 0 22M40 13a19 19 0 0 1 0 22"/>'
           '<text x="24" y="28.5" text-anchor="middle" font-size="11" font-weight="700" '
           'fill="currentColor" stroke="none" font-family="Saira, sans-serif">ABS</text>',
    # passager assis + coussin gonflable
    "airbag": '<circle cx="17" cy="9" r="3.5"/><path d="M17 15v12h10l4 10"/><path d="M11 18v13h10"/>'
              '<circle cx="34" cy="21" r="7"/>',
}

# (clé, libellé affiché, couleur, formulation utilisée pour la question)
VOYANTS = [
    ("huile", "Pression d'huile", ROUGE, "le voyant de pression d'huile"),
    ("temperature", "Température moteur", ROUGE, "le voyant de température du liquide de refroidissement"),
    ("batterie", "Batterie", ROUGE, "le voyant de charge de la batterie"),
    ("freins", "Freins", ROUGE, "le voyant de freinage"),
    ("airbag", "Airbag", ROUGE, "le voyant d'airbag"),
    ("moteur", "Moteur / antipollution", ORANGE, "le voyant moteur (antipollution)"),
    ("pneus", "Pression des pneus", ORANGE, "le voyant de pression des pneus"),
    ("abs", "ABS", ORANGE, "le voyant ABS"),
]


def svg(cle):
    return _SVG.format(ICONES[cle])
