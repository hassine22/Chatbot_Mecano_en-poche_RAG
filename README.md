# Mécano de poche — Système RAG d'Assistance Technique Automobile

**Mécano de poche** est un assistant technique automobile basé sur une architecture **RAG (Retrieval-Augmented Generation)**. Il aide un conducteur à comprendre un voyant allumé, un message d'alerte, une anomalie ou une procédure d'entretien en s'appuyant **exclusivement** sur les notices d'utilisation officielles des constructeurs (Renault Clio 4, Ford EcoSport, Hyundai i10, Kia Picanto). Il cite systématiquement les pages sources, refuse de répondre lorsque l'information ne figure pas dans la notice, pose une question de clarification si la demande est trop vague, et recommande un arrêt immédiat ou la consultation d'un garagiste lorsque la sécurité est engagée.

Mini-projet NLP / RAG & LLM — ENSI.

---

## Table des Matières

1. [Architecture Globale du Pipeline RAG](#1-architecture-globale-du-pipeline-rag)
2. [Structure Complète du Dépôt](#2-structure-complète-du-dépôt)
3. [Détail Exhaustif des Étapes du Projet](#3-détail-exhaustif-des-étapes-du-projet)
   - [Étape 0 : Extraction, Nettoyage, Validation et Fusion du Corpus](#étape-0--extraction-nettoyage-validation-et-fusion-du-corpus-corpus)
   - [Étape 1 : Recherche Vectorielle Multilingue et Index FAISS](#étape-1--recherche-vectorielle-multilingue-et-index-faiss-srcmecanoretrieverpy)
   - [Étape 2 : Génération LLM, Multi-Query + RRF et Questions Ambiguës](#étape-2--génération-llm-multi-query--rrf-et-questions-ambiguës-srcmecano)
   - [Étape 3 : Interface Web Streamlit et Détection NER / OBD-II](#étape-3--interface-web-streamlit-et-détection-ner--obd-ii-app)
   - [Étape 4 : Évaluation Quantitative et Analyse Critique](#étape-4--évaluation-quantitative-et-analyse-critique-evaluation)
4. [Installation et Configuration](#4-installation-et-configuration)
5. [Guide Complet des Commandes (Corpus, Index, CLI, Interface, Évaluation, Tests)](#5-guide-complet-des-commandes)
6. [État d'Avancement et Livrables](#6-état-davancement-et-livrables)

---

## 1. Architecture Globale du Pipeline RAG

Lorsqu'un conducteur pose une question depuis l'interface web (`app/streamlit_app.py`) ou en ligne de commande (`scripts/ask.py`), la requête traverse les 5 étages suivants :

```text
Question du conducteur (+ filtres optionnels : modèle, catégories, top-k)
   │
   ├──► ÉTAGE 1 : Analyse d'Entités (NER) & Filtre d'Ambiguïté (`ner.py`, `prompts.py`)
   │       ├─ Détection automatique du véhicule mentionné (Clio 4, EcoSport, i10, Picanto)
   │       ├─ Détection des codes défaut OBD-II (ex. P0420, C0123, B..., U...) et identification de la famille
   │       └─ Détection d'ambiguïté (`is_question_ambiguous`) : si la question ne contient que des termes
   │          génériques ("J'ai un voyant rouge allumé, que faire ?"), retour immédiat d'une demande de
   │          clarification ciblée (`needs_clarification = True`) sans répondre au hasard.
   │
   ├──► ÉTAGE 2 : Reformulation Multi-Requêtes (`llm.py`, `prompts.py`)
   │       └─ Le LLM (Gemini) génère 3 reformulations complémentaires :
   │          (1) Terminologie technique officielle corrigée dans la langue de la question
   │          (2) Formulation orientée symptôme, témoin lumineux ou action immédiate
   │          (3) Traduction technique en anglais automobile (indispensable pour interroger en français
   │              les notices officielles en anglais des modèles EcoSport, i10 et Picanto)
   │
   ├──► ÉTAGE 3 : Recherche Sémantique Multilingue (`embeddings.py`, `index.py`, `retriever.py`)
   │       ├─ Encodage en vecteurs de dimension 384 via `paraphrase-multilingual-MiniLM-L12-v2` (normalisés L2)
   │       ├─ Recherche par similarité cosinus dans l'index FAISS (`index/corpus.faiss`, `IndexFlatIP`)
   │       ├─ Filtrage exact pré-top-k par véhicule (`modele`) et par section (`categorie`) via `VectorRetriever`
   │       └─ Fallback lexical automatique (`_fallback_search`) si FAISS n'est pas disponible localement
   │
   ├──► ÉTAGE 4 : Fusion Reciprocal Rank Fusion — RRF (`pipeline.py`)
   │       ├─ Fusion des listes de passages issues de la requête originale + des 3 reformulations (k = 60)
   │       └─ Dédoublonnage par identifiant de passage et normalisation du score de pertinence dans [0, 1]
   │
   └──► ÉTAGE 5 : Génération Contrainte & Sourcée (`llm.py`, `prompts.py`)
           ├─ Réponse dans la langue de la question (FR ou EN), avec traduction technique à la volée
           ├─ Ancrage strict sur la balise <CONTEXTE> (zéro connaissance externe, zéro hallucination)
           ├─ Refus normalisé si l'information est absente (`info_absente = True`)
           ├─ En-tête `**ALERTE DE SÉCURITÉ :**` et recommandation d'un professionnel pour les cas graves
           └─ Citations `[p. X]` dans le corps du texte + bloc final structuré `**Références :**`
```

---

## 2. Structure Complète du Dépôt

```text
Chatbot_Mecano_en-poche_RAG/
├── README.md                          ← Documentation générale complète du projet
├── FORMAT_CORPUS.md                   ← Spécification du format JSONL et règles de découpage du corpus
├── requirements.txt                   ← Dépendances Python du projet
├── config.yaml                        ← Configuration centrale (modèles, top-k, RRF, paramètres LLM, chemins)
├── .env.example                       ← Modèle pour configurer la clé GEMINI_API_KEY dans .env
├── .streamlit/
│   └── config.toml                    ← Thème sombre « tableau de bord de nuit » et service des polices statiques
│
├── data/                              ← Données sources
│   └── pdf/                           ← Notices constructeurs PDF (SOURCES.md contient les références)
│
├── corpus/                            ← ÉTAPE 0 : Extraction, nettoyage, validation et fusion des notices
│   ├── corpus_schema.json             ← Schéma JSON officiel d'un passage
│   ├── validate_corpus.py             ← Validation stricte d'un fichier JSONL (0 erreur, 0 avertissement)
│   ├── merge_corpus.py                ← Fusionne les 4 corpus individuels dans all_corpus.jsonl
│   ├── scripts/
│   │   ├── build_clio4.py             ← Extraction et découpage Renault Clio 4 (FR)
│   │   ├── build_ecosport.py          ← Extraction et découpage Ford EcoSport (EN)
│   │   ├── build_i10.py               ← Extraction et découpage Hyundai i10 (EN)
│   │   └── build_picanto.py           ← Extraction et découpage Kia Picanto (EN)
│   └── output/
│       ├── clio4_corpus.jsonl         ← 246 passages validés (Renault Clio 4)
│       ├── ecosport_corpus.jsonl      ← 219 passages validés (Ford EcoSport)
│       ├── i10_corpus.jsonl           ← 294 passages validés (Hyundai i10)
│       ├── picanto_corpus.jsonl       ← 360 passages validés (Kia Picanto)
│       └── all_corpus.jsonl           ← Corpus complet fusionné (1 119 passages)
│
├── src/mecano/                        ← ÉTAPES 1 & 2 : Moteur RAG (Recherche + Génération + NER)
│   ├── config.py                      ← Lecture de config.yaml et chargement sécurisé de .env
│   ├── corpus_loader.py               ← Chargement et filtrage des passages JSONL
│   ├── embeddings.py                  ← Encodage Sentence-BERT multilingue (normalisé L2)
│   ├── index.py                       ← Construction, sauvegarde et chargement de l'index FAISS
│   ├── retriever.py                   ← Classe VectorRetriever (recherche cosinus + filtres modèle/catégorie)
│   ├── ner.py                         ← Détection automatique du modèle automobile et des codes OBD-II
│   ├── prompts.py                     ← Prompts système RAG et Multi-Query, détection d'ambiguïté
│   ├── llm.py                         ← Client REST Google Gemini avec retry automatique sur code 429
│   └── pipeline.py                    ← Orchestration complète : Question → Multi-Query + RRF → Réponse
│
├── index/                             ← Index vectoriel FAISS partagé
│   ├── corpus.faiss                   ← Vecteurs d'embeddings indexés (IndexFlatIP, dim 384)
│   └── corpus_metadata.json           ← Métadonnées alignées des passages indexés
│
├── scripts/                           ← Outils en ligne de commande (CLI)
│   ├── build_index.py                 ← Construit index/corpus.faiss et index/corpus_metadata.json
│   ├── search.py                      ← Teste la recherche vectorielle et compare top-k=3 et top-k=5
│   └── ask.py                         ← Pose une question au RAG complet (inclut --compare-multiquery)
│
├── app/                               ← ÉTAPE 3 : Interface Web (Thème « Tableau de bord de nuit »)
│   ├── streamlit_app.py               ← Application Streamlit (sélection voiture, voyants SVG, mémoire, repli)
│   ├── telltales.py                   ← Pictogrammes SVG des 8 voyants du tableau de bord cliquables
│   ├── style.css                      ← Feuille de style du thème tableau de bord, animations et responsive mobile
│   ├── static/fonts/                  ← Police embarquée Saira (woff2 + licence SIL OFL, utilisable hors ligne)
│   ├── demo_backend.py                ← Backend lexical BM25 de secours (même contrat que pipeline.py)
│   └── README.md                      ← Documentation de l'interface et du contrat pipeline ↔ frontend
│
├── evaluation/                        ← ÉTAPE 4 : Évaluation quantitative (Retrieval & Génération)
│   ├── retrieval_questions.jsonl      ← 10 questions annotées pour évaluer top-k=3 vs top-k=5
│   ├── questions.jsonl                ← 10 questions de test bout-en-bout (8 notices + 2 hors-corpus)
│   ├── evaluate_retrieval.py          ← Calcule Précision, Rappel, F1, MRR et Hit-Rate pour top-k=3 et 5
│   ├── evaluate_generation.py         ← Évalue la pertinence RAG, la fidélité (points clés) et le refus
│   └── results/
│       ├── generation_evaluation.json ← Rapport JSON détaillé de l'évaluation de bout en bout
│       └── feedback.jsonl             ← Retours utilisateurs (👍 / 👎) enregistrés depuis l'interface
│
├── docs/images/                       ← Captures d'écran de l'interface (accueil, voyants, réponse, sources, mobile)
│
└── tests/                             ← Suite de 17 tests unitaires automatisés
    ├── test_retriever.py              ← Tests unitaires de VectorRetriever et des filtres
    ├── test_evaluate_retrieval.py     ← Tests unitaires des métriques d'évaluation du retrieval
    └── test_generation.py             ← Tests unitaires RRF, prompts, mémoire (<HISTORIQUE>), ambiguïté et contrat
```

---

## 3. Détail Exhaustif des Étapes du Projet

### Étape 0 : Extraction, Nettoyage, Validation et Fusion du Corpus (`corpus/`)

Chaque membre a extrait et structuré la notice officielle d'un véhicule en appliquant les règles définies dans [`FORMAT_CORPUS.md`](FORMAT_CORPUS.md) :
- Suppression du bruit (sommaires, index alphabétiques, numéros de page isolés, en-têtes répétitifs).
- Conservation intégrale des avertissements de sécurité (`WARNING`, `CAUTION`, `DANGER`, `STOP`) et des tableaux de fusibles ou d'entretien.
- Découpage sémantique en passages autonomes de **40 à 350 mots** enrichis d'un champ `texte_embedding` préfixé par le modèle et le titre de section.

| Véhicule | Années | Langue | Script de construction | Passages | Validation (`validate_corpus.py`) |
|---|---|---|---|---|---|
| **Renault Clio 4** | 2016–2020 | Français (`fr`) | `corpus/scripts/build_clio4.py` | 246 | 0 erreur, 0 avertissement |
| **Ford EcoSport** | 2018–2022 | Anglais (`en`) | `corpus/scripts/build_ecosport.py` | 219 | 0 erreur, 0 avertissement |
| **Hyundai i10** | 2014–2019 | Anglais (`en`) | `corpus/scripts/build_i10.py` | 294 | 0 erreur, 0 avertissement |
| **Kia Picanto** | 2017–2023 | Anglais (`en`) | `corpus/scripts/build_picanto.py` | 360 | 0 erreur, 0 avertissement |
| **Corpus Fusionné** | — | `fr` + `en` | `corpus/merge_corpus.py` | **1 119** | `corpus/output/all_corpus.jsonl` |

---

### Étape 1 : Recherche Vectorielle Multilingue et Index FAISS (`src/mecano/retriever.py`)

- **Modèle d'embeddings multilingue :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`src/mecano/embeddings.py`). Chaque passage est encodé à partir de son champ `texte_embedding` (ou d'un assemblage structuré véhicule + chapitre + titre + texte) en un vecteur dense de dimension 384 normalisé ($L_2$).
- **Index FAISS :** les vecteurs sont indexés dans une structure `faiss.IndexFlatIP` (`src/mecano/index.py`), où le produit scalaire correspond exactement à la similarité cosinus.
- **Recherche filtrée (`VectorRetriever` dans `src/mecano/retriever.py`) :**
  - Recherche sur l'intégralité de l'index FAISS puis application des filtres exacts insensibles à la casse (`modele` et `categorie`) **avant** la troncature à `top_k`, garantissant jusqu'à `top_k` passages pertinents même sur un sous-ensemble restreint du corpus.

---

### Étape 2 : Génération LLM, Multi-Query + RRF et Questions Ambiguës (`src/mecano/`)

- **Modèle de génération :** Google Gemini (`gemini-flash-lite-latest`) appelé via l'API REST (`src/mecano/llm.py`) avec une température basse (`0.1`) et gestion automatique des limites de quota Free Tier (retry automatique avec attente sur code HTTP `429`).
- **Règles strictes du Prompt Système (`src/mecano/prompts.py`) :**
  1. **Adaptation linguistique automatique (FR / EN) :** répond en français à une question en français et en anglais à une question en anglais, en traduisant fidèlement les extraits techniques des notices anglaises.
  2. **Fidélité stricte (zéro hallucination) :** interdiction formelle d'utiliser des connaissances hors `<CONTEXTE>`.
  3. **Refus catégorique sur information absente :** si la réponse ne figure pas dans les extraits (ex. prix d'une vidange, assurance, code OBD absent du manuel), renvoie la réponse normalisée :
     > *"Cette information n'est pas présente dans la notice officielle du véhicule (information absente). Nous vous conseillons de contacter un garagiste ou un concessionnaire."*
  4. **Sécurité et recommandation garagiste :** commence par `**ALERTE DE SÉCURITÉ :**` dès qu'un danger moteur ou sécurité est mentionné (pression d'huile, surchauffe, freinage, direction, airbag), indique la consigne d'arrêt et recommande un professionnel.
  5. **Traçabilité des sources :** citations `[p. X]` au fil du texte et bloc final structuré `**Références :**` (Véhicule, Chapitre, Titre, Pages et ID du passage).
- **Bonus — Reformulation Multi-Requêtes + Reciprocal Rank Fusion (RRF, $k=60$) :**
  - Avant l'étape de recherche, `generate_reformulations()` génère 3 reformulations complémentaires de la question (terminologie corrigée, formulation symptôme/action, et traduction technique en anglais automobile).
  - Les classements issus de la question initiale et des 3 reformulations sont fusionnés par `reciprocal_rank_fusion()` selon la formule :
    $$\text{Score}_{\text{RRF}}(d) = \sum_{q \in Q} \frac{1}{60 + \text{rang}_q(d)}$$
  - Ce mécanisme corrige à la fois les fautes d'orthographe/formulations vagues et le fossé linguistique entre les questions posées en français et les 3 notices rédigées en anglais (`EcoSport`, `i10`, `Picanto`).
- **Bonus — Détection et Clarification des Questions Ambiguës :**
  - La fonction `is_question_ambiguous()` détecte les questions qui ne mentionnent aucun système, symbole ou code précis (ex. *"J'ai un voyant rouge allumé sur mon tableau de bord, que faire ?"*).
  - Au lieu de répondre au hasard, `build_clarification_response()` invite le conducteur à préciser : (1) la forme du symbole ou le système concerné, (2) la couleur et le comportement (fixe/clignotant), (3) le message affiché à l'écran, et (4) le modèle du véhicule.
- **Bonus — Mémoire Conversationnelle Multi-Tours (`<HISTORIQUE>`) :**
  - **Persistance du véhicule en session (`infer_modele_from_history`) :** si le conducteur mentionne sa voiture au tour 1 (ex. `Clio 4`) puis pose une question de suivi au tour 2 sans répéter le modèle (*"Et quel fusible correspond à l'allume-cigare ?"*), le pipeline hérite automatiquement du véhicule actif. Dans l'interface, la voiture choisie reste affichée en haut de la page tant que le conducteur ne clique pas sur « Changer de voiture ».
  - **Contextualisation des questions de suivi (`contextualize_question_with_history`) :** lorsqu'un conducteur répond à une demande de clarification (ex. Tour 1 : *"J'ai un voyant rouge allumé"* → Tour 2 : *"C'est une burette d'huile"*) ou utilise des pronoms (*"Et s'il clignote ?"*), la requête de recherche et les reformulations Multi-Query intègrent automatiquement le contexte des 3 derniers échanges.
  - **Bloc `<HISTORIQUE>` dans le prompt RAG :** les 3 derniers échanges sont injectés dans `build_rag_prompt()` afin que le LLM comprenne la continuité du dialogue tout en restant strictement ancré sur `<CONTEXTE>` pour les faits techniques et les numéros de page.

---

### Étape 3 : Interface Web Streamlit et Détection NER / OBD-II (`app/`)

L'application web [`app/streamlit_app.py`](app/streamlit_app.py) propose une expérience pensée pour un conducteur face à son tableau de bord (détails complets et captures dans [`app/README.md`](app/README.md)) :
- **Thème « Tableau de bord de nuit » (`.streamlit/config.toml`, `app/style.css`, `app/static/fonts/`) :** fond bleu nuit (`#0A1220`), police `Saira` embarquée (fonctionne hors connexion) et codes couleurs rouge / orange / vert réservés aux voyants et au niveau d'urgence, avec adaptation responsive sur mobile.
- **Panneau de 8 voyants SVG cliquables (`app/telltales.py`) :** pictogrammes vectoriels (pression d'huile, température moteur, batterie, freins, moteur/OBD, pneus TPMS, ABS, ESP) avec animation d'allumage au chargement (*self-test*). Le conducteur peut cliquer directement sur le symbole allumé sans avoir à connaître son nom technique.
- **Sélection guidée ou détection automatique du véhicule (`src/mecano/ner.py`) :**
  - Le conducteur choisit sa voiture parmi les 4 cartes en haut de l'écran, ou la mentionne directement dans sa question (détection NER automatique avec badge *« reconnue dans la question »*).
  - Si une question est posée sans qu'aucune voiture ne soit sélectionnée ni détectée, l'interface met la question en attente et invite l'utilisateur à cliquer sur son véhicule avant de lancer la recherche afin de ne jamais mélanger les notices.
- **Détection des codes défaut OBD-II (`src/mecano/ner.py`) :** reconnaît les codes normalisés (`P0420`, `C0123`, `B...`, `U...`), affiche le badge `Code P0420 : Groupe motopropulseur (moteur / boîte)` et rappelle qu'une lecture à la valise de diagnostic en atelier est nécessaire.
- **Verdict d'urgence immédiat (`niveau()`) :** bandeau visuel avec témoin lumineux indiquant immédiatement la gravité (*Arrêtez-vous dès que possible*, *Faites vérifier la voiture rapidement*, *Pas d'urgence signalée*, *Ce n'est pas dans la notice*, ou *Il manque une précision*).
- **Citations en étiquettes et visualisation des sources :** les renvois `[p. X]` dans la réponse sont transformés en badges visuels (`mise_en_forme()`), et un panneau dépliable affiche les extraits de la notice utilisés avec page, jauge visuelle de pertinence en `%`, texte nettoyé et lien vers le PDF officiel.
- **Collecte d'avis (`evaluation/results/feedback.jsonl`) :** boutons 👍 / 👎 sous chaque réponse enregistrant la question, le véhicule, le `top-k`, les sources et l'avis de l'utilisateur.
- **Mode hors ligne et repli automatique (`MECANO_HORS_LIGNE=1`) :** bascule automatiquement sur l'extrait brut le plus pertinent ou sur le moteur lexical `app/demo_backend.py` si le quota de l'API Gemini est atteint ou en absence de connexion réseau.

---

### Étape 4 : Évaluation Quantitative et Analyse Critique (`evaluation/`)

#### 4.1 Évaluation du Retrieval Sémantique (`top-k = 3` vs `top-k = 5`)
Exécutée via `python evaluation/evaluate_retrieval.py` sur les 10 questions annotées de `evaluation/retrieval_questions.jsonl` avec l'index FAISS (`paraphrase-multilingual-MiniLM-L12-v2`) :

| Métrique | `top-k = 3` | `top-k = 5` | Analyse |
|---|---|---|---|
| **Précision moyenne** | **0.233** | 0.160 | `top-k=3` concentre davantage de passages pertinents en tête de liste |
| **Rappel moyen** | 0.700 | **0.800** | `top-k=5` retrouve +10 points de passages attendus supplémentaires |
| **Score F1 moyen** | **0.350** | 0.267 | Meilleur compromis précision/rappel obtenu à `top-k=3` |
| **MRR (Mean Reciprocal Rank)** | **0.650** | **0.650** | Le premier passage pertinent apparaît en moyenne entre le rang 1 et le rang 2 |
| **Hit-Rate (au moins 1 bon passage)** | 70.0 % | **80.0 %** | `top-k=5` maximise la couverture pour le prompt du LLM |

**Conclusion sur le choix de `top-k` :**
- En recherche pure (`evaluate_retrieval.py`), **`top-k = 3`** obtient le meilleur score **F1 (`0.350` contre `0.267`)**.
- Pour la génération RAG complète (`config.yaml` et `streamlit_app.py`), **`top-k = 5`** est retenu par défaut car il porte le **Hit-Rate à 80 %** (et **100 %** avec Multi-Query + RRF), tandis que le LLM ignore facilement les 1 à 2 passages annexes.

#### 4.2 Évaluation de Bout en Bout du Pipeline RAG (`evaluation/evaluate_generation.py`)
Exécutée sur les **10 questions de test** de `evaluation/questions.jsonl` (8 questions portant sur les 4 voitures + 2 questions pièges sans réponse dans les notices : `q09` tarif d'une vidange à Tunis et `q10` signification du code OBD `P0420`) :

| Question | Véhicule | Type | Passages attendus retrouvés (RAG Multi-Query + RRF) | Fidélité / Statut |
|---|---|---|---|---|
| **q01** — Voyant pression d'huile en roulant | Clio 4 | `voyant` | `clio4-231`, `clio4-230`, `clio4-012` (3/3 retrouvés) | **100 %** (4/4 points clés) |
| **q02** — Fusible allume-cigare | Clio 4 | `fusibles` | `clio4-207` (retrouvé au rang 1) | **100 %** (Fusible n° 17, p. 5.29) |
| **q03** — Voyant pression des pneus (TPMS) | EcoSport | `voyant` | `ecosport-011`, `ecosport-035` (retrouvés) | **100 %** (3/3 points clés) |
| **q04** — Vérifier le niveau d'huile moteur | EcoSport | `entretien` | `ecosport-150` (retrouvé au rang 1) | **100 %** (sol plat, 15 min, jauge, p. 206) |
| **q05** — Aiguille température vers H | Picanto | `voyant` | `picanto-003` (retrouvé) | **67 %** (*"Ne continuez pas à conduire"*, p. 7-7) |
| **q06** — Démarrer batterie à plat avec câbles | Picanto | `electricite_batterie` | `picanto-221` (retrouvé au rang 1) | **67 %**–**100 %** (12 V, ordre des câbles, p. 337) |
| **q07** — Voyant pression d'huile allumé | i10 | `voyant` | `i10-019` (retrouvé au rang 1) | **100 %** (3/3 points clés, p. 4-12) |
| **q08** — Kit de réparation de pneu (TMK) | i10 | `pneus_roues` | `i10-165`, `i10-166`, `i10-168` (3/3 retrouvés) | **100 %** (4/4 points clés, p. 8-27 à 8-29) |
| **q09** — Prix d'une vidange chez Renault à Tunis | Tous | `hors_corpus` | — (Question piège hors-notice) | **100 % OK** (Refus strict sans hallucination) |
| **q10** — Signification du code défaut P0420 | Clio 4 | `hors_corpus` | — (Question piège hors-notice) | **100 % OK** (Refus strict + renvoi garagiste) |

**Synthèse globale (`evaluation/results/generation_evaluation.json`) :**
- **Pertinence RAG avec Multi-Query + RRF (Hit-Rate) :** **100.0 %** (contre 50.0 % sans Multi-Query sur les questions posées en français sur des notices en anglais, soit un **gain de +50 points**).
- **Fidélité moyenne sur les points clés :** **91.7 %** (100 % des réponses sont strictement fidèles aux notices ; les seuls écarts proviennent de variations de formulation comme *"Ne continuez pas à conduire"* au lieu de *"ne pas continuer"*).
- **Reconnaissance de l'information absente (`q09`, `q10`) :** **100.0 %** (aucune invention de prix ni de signification du code `P0420`).

---

## 4. Installation et Configuration

### Prérequis
- Python 3.10+
- Une clé API Google Gemini (`GEMINI_API_KEY`)

### 1. Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/hassine22/Chatbot_Mecano_en-poche_RAG.git
cd Chatbot_Mecano_en-poche_RAG
pip install -r requirements.txt
```

### 2. Configurer le fichier `.env`

Créez un fichier `.env` à la racine du projet à partir de `.env.example` :

```ini
GEMINI_API_KEY=votre_cle_api_gemini
```

---

## 5. Guide Complet des Commandes

### 1. Valider et fusionner les corpus (`corpus/`)

```bash
python corpus/validate_corpus.py corpus/output/clio4_corpus.jsonl
python corpus/validate_corpus.py corpus/output/ecosport_corpus.jsonl
python corpus/validate_corpus.py corpus/output/i10_corpus.jsonl
python corpus/validate_corpus.py corpus/output/picanto_corpus.jsonl
python corpus/merge_corpus.py
```

### 2. Construire ou reconstruire l'index FAISS (`scripts/build_index.py`)

```bash
python scripts/build_index.py
```

### 3. Tester la recherche vectorielle seule (`scripts/search.py`)

```bash
# Compare automatiquement top-k=3 et top-k=5 avec filtres
python scripts/search.py "Que faire si le voyant ABS reste allumé ?" --modele "Clio 4" --categorie voyant
```

### 4. Interroger le pipeline RAG complet en CLI (`scripts/ask.py`)

```bash
# Question standard avec filtre par véhicule
python -X utf8 scripts/ask.py "Que faire si le voyant de pression d'huile s'allume sur ma Clio 4 ?" --modele "Clio 4"

# Question en anglais
python -X utf8 scripts/ask.py "How do I check the engine oil level on the Hyundai i10?" --modele "i10"

# Question trop vague (déclenche la demande de clarification)
python -X utf8 scripts/ask.py "J'ai un voyant rouge qui clignote sur mon tableau de bord, que faire ?"

# Question hors-notice (déclenche le refus strict)
python -X utf8 scripts/ask.py "Combien coûte une vidange chez Renault à Tunis ?"

# Comparer la recherche directe vs Multi-Query + RRF sur une question bruitée / avec fautes
python -X utf8 scripts/ask.py "voyan batri rouj allumé que fair" --modele "i10" --compare-multiquery
```

### 5. Lancer l'application web Streamlit (`app/streamlit_app.py`)

```bash
python -m streamlit run app/streamlit_app.py
```
L'application s'ouvre dans le navigateur sur **`http://localhost:8501`**.

### 6. Exécuter les évaluations (`evaluation/`)

```bash
# Évaluation du retrieval (top-k=3 vs top-k=5 sur evaluation/retrieval_questions.jsonl)
python -X utf8 evaluation/evaluate_retrieval.py

# Évaluation de bout en bout (pertinence + fidélité + refus hors-notice sur evaluation/questions.jsonl)
python -X utf8 evaluation/evaluate_generation.py
```

### 7. Exécuter la suite de tests unitaires (`tests/`)

```bash
python -X utf8 -m unittest discover -s tests -v
```

---

## 6. État d'Avancement et Livrables

- [x] **Étape 0 — Corpus :** 4 notices extraites, nettoyées, validées (`0 erreur`) et fusionnées (`1 119` passages dans `corpus/output/all_corpus.jsonl`).
- [x] **Étape 1 — Recherche :** Embeddings Sentence-BERT multilingues (`paraphrase-multilingual-MiniLM-L12-v2`), index FAISS (`index/corpus.faiss`), filtres modèle/catégorie (`VectorRetriever`) et comparaison `top-k=3` vs `top-k=5`.
- [x] **Étape 2 — Génération :** Intégration Google Gemini (`src/mecano/llm.py`), prompt strict FR/EN, refus hors-notice, alertes de sécurité, citations des pages, **Multi-Query + RRF ($k=60$)** et **clarification des questions ambiguës**.
- [x] **Étape 3 — Interface :** Application web Streamlit professionnelle (`app/streamlit_app.py`) avec détection automatique du véhicule et des codes défaut OBD-II (`src/mecano/ner.py`), bandeaux d'urgence et visualisation des sources.
- [x] **Étape 4 — Évaluation :** Scripts et rapports d'évaluation du retrieval (`evaluation/evaluate_retrieval.py`) et de la génération de bout en bout (`evaluation/evaluate_generation.py` -> `evaluation/results/generation_evaluation.json`).

## Droits d'auteur

Les notices d'utilisation appartiennent à leurs constructeurs respectifs (Renault, Ford, Hyundai, Kia) et sont utilisées exclusivement dans un cadre pédagogique et académique.
