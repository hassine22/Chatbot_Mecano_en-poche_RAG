# Règles communes pour les corpus

**À lire avant de commencer.** Les 4 corpus seront fusionnés en un seul index FAISS. S'ils n'ont pas exactement le même format, la fusion, les filtres et l'affichage des sources ne fonctionneront pas.

Le corpus de référence est `corpus/output/clio4_corpus.jsonl` : en cas de doute, faites comme lui.

---

## 1. Répartition

| Voiture | Identifiant (`slug`) | Responsable | Langue de la notice |
|---|---|---|---|
| Renault Clio 4 (phase 2, 2016-2020) | `clio4` | … | fr |
| Ford EcoSport | `ecosport` | … | en |
| … | … | … | … |
| … | … | … | … |

Le `slug` est court, en minuscules, sans espace ni accent. Il sert de préfixe aux identifiants et aux noms de fichiers.

## 2. Fichiers à produire (par voiture)

```
corpus/scripts/build_<slug>.py      # le script qui génère le corpus
corpus/output/<slug>_corpus.jsonl   # le corpus (1 passage = 1 ligne JSON)
corpus/output/<slug>_corpus.md      # version lisible pour la relecture
```

Les PDF ne vont **pas** sur GitHub : mettez-les dans `data/pdf/` (dossier ignoré par git) et ajoutez le lien de téléchargement dans `data/pdf/SOURCES.md`.

## 3. Format d'un passage

Chaque ligne du `.jsonl` est un objet JSON avec **exactement** ces champs :

| Champ | Type | Obligatoire | Exemple | Règle |
|---|---|---|---|---|
| `id` | texte | oui | `"ecosport-007"` | `<slug>-` + numéro sur 3 chiffres, unique |
| `marque` | texte | oui | `"Ford"` | Majuscule au début |
| `modele` | texte | oui | `"EcoSport"` | Identique sur tous les passages de la voiture |
| `phase` | texte | oui | `"phase 2"` ou `""` | Vide si non applicable |
| `annees` | texte | oui | `"2013-2017"` | Format `AAAA-AAAA` |
| `langue` | texte | oui | `"en"` | `fr`, `en` ou `ar` |
| `chapitre` | texte | oui | `"Warning lamps and indicators"` | Chapitre de la notice |
| `section` | texte | oui | `"TÉMOINS LUMINEUX"` | Titre de la page/section |
| `sous_section` | texte ou `null` | oui | `"Témoin de pression d’huile"` | `null` s'il n'y en a pas |
| `titre` | texte | oui | `"TÉMOINS LUMINEUX > Témoin de pression d’huile"` | `section` + ` > ` + `sous_section` |
| `categorie` | texte | oui | `"voyant"` | **Uniquement** une valeur de la liste §4 |
| `pages` | liste de textes | oui | `["1.57"]` | Numéros **imprimés** dans la notice |
| `texte` | texte | oui | `"Il doit s'allumer…"` | Le contenu nettoyé (12 à 400 mots) |
| `texte_embedding` | texte | oui | voir ci-dessous | Le texte à vectoriser |
| `nb_mots` | entier | oui | `57` | Nombre de mots de `texte` |
| `document` | texte | oui | `"Ford EcoSport Owner's Manual (CG3590en, 08/2014)"` | Nom et édition de la notice |
| `source_url` | texte | oui | `"https://www.fordservicecontent.com/…pdf"` | Lien officiel de la notice |

**`texte_embedding`** se construit toujours de la même façon :

```
<marque> <modele> (<annees>) – <titre>
<texte>
```

Exemple : `"Renault Clio 4 (2016-2020) – TÉMOINS LUMINEUX > Témoin de pression d’huile\nIl doit s’allumer à la mise sous contact…"`

## 4. Catégories autorisées

| `categorie` | Ce qu'on y met |
|---|---|
| `voyant` | Un passage **par voyant** du tableau de bord (signification + que faire) |
| `message_tableau_de_bord` | Messages texte affichés par l'ordinateur de bord |
| `anomalie` | Tableau « problème / cause / que faire » (un passage par problème) |
| `demarrage_moteur` | Démarrage, arrêt, Stop & Start, particularités essence/diesel |
| `aides_conduite` | ABS, ESP/ESC, antipatinage, surveillance de pression des pneus |
| `boite_automatique` | Boîte automatique et ses messages |
| `direction` | Direction assistée |
| `carburant` | Réservoir, mauvais carburant, panne sèche |
| `entretien` | Huile, vidange, niveaux, filtres, intervalles d'entretien |
| `pneus_roues` | Pression, crevaison, kit de gonflage, changement de roue |
| `fusibles` | Remplacement et affectation des fusibles |
| `electricite_batterie` | Batterie, démarrage avec câbles, pile de la clé |
| `remorquage` | Remorquage / dépannage |
| `identification` | Plaques d'identification, numéro de série (VIN) |
| `caracteristiques_moteur` | Tableau des moteurs, capacités |
| `code_obd` | Réservé à la base des codes défaut OBD-II (P0420…) |

Besoin d'une nouvelle catégorie ? On en discute dans le groupe **avant**, puis on l'ajoute ici **et** dans `corpus_schema.json`.

## 5. Que garder dans la notice ?

**À garder :** voyants, messages, anomalies/pannes, démarrage, entretien et niveaux, pneus/crevaison, fusibles, batterie, remorquage, ABS/ESP, boîte auto, identification du moteur.

**À retirer :** radio/multimédia, GPS, climatisation, sièges, rangements, sièges enfants, ceintures, éclairage intérieur, carrosserie, éco-conduite, garantie, carnet d'entretien vierge, index, sommaires, mentions légales.

## 6. Règles de nettoyage et de découpage

1. **Un sujet = un passage** : un voyant, un message ou un problème par passage.
2. **Longueur** : entre 12 et 400 mots (idéalement 30 à 300).
3. **Recoller les césures** : `refroidis-` + `sement` → `refroidissement`, sans casser les mots composés (`essuie-vitre`).
4. **Remettre l'ordre de lecture** des pages en colonnes.
5. **Supprimer** les numéros de page, les repères de figures (`A`, `B`, `1`, `2` seuls) et les en-têtes répétés.
6. **Pictogrammes** : quand une icône apparaît au milieu d'une phrase, la remplacer par `[voyant <nom>]`, par ex. `[voyant arrêt impératif STOP]`.
7. **Tableaux** : écrire une ligne par information, par ex.
   ```
   Problème : Le moteur chauffe.
   Cause possible 1 : Ventilateur défectueux.
   Que faire : Arrêtez le véhicule et consultez un garagiste.
   ```
8. **Ne rien inventer ni traduire** : le texte reste celui de la notice, dans sa langue d'origine.

## 7. Avant de pousser sur GitHub

```bash
python corpus/validate_corpus.py corpus/output/<slug>_corpus.jsonl
```

Le script doit afficher **0 erreur**. Relisez aussi 10 passages au hasard dans le `.md` pour vérifier que le texte est propre.

Puis, quand tous les corpus sont prêts :

```bash
python corpus/merge_corpus.py      # → corpus/output/all_corpus.jsonl
```
