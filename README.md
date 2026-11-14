# Mécano de poche — Système RAG d'Assistance Technique Automobile

**Mécano de poche** est un assistant de diagnostic et d'aide à la conduite basé sur une architecture **RAG (Retrieval-Augmented Generation)**. Il répond aux questions des conducteurs (voyants allumés, messages d'alerte, procédures d'entretien, anomalies) en s'appuyant **exclusivement** sur les manuels d'utilisation officiels des constructeurs, avec citation systématique des pages sources et recommandation d'arrêt ou de contrôle en atelier lorsque la sécurité est engagée.

Mini-projet NLP / RAG & LLM — ENSI.

---

## 1. Vue d'ensemble et Architecture du Pipeline

Lorsqu'un conducteur pose une question depuis l'interface web (`app/streamlit_app.py`) ou en ligne de commande (`scripts/ask.py`), le système exécute la chaîne suivante :

```text
Question du conducteur (+ filtres optionnels : modèle, catégorie, top-k)
   │
   ├──► 1. Analyse d'entités (NER) & Ambiguïté (`ner.py`, `prompts.py`)
   │       ├─ Détection automatique du modèle de voiture (Clio 4, i10, EcoSport, Picanto)
   │       ├─ Détection des codes défaut OBD-II (ex. P0420, C0123...)
   │       └─ Vérification d'ambiguïté : si la question est trop vague (ex. "Voyant rouge allumé"),
   │          retour immédiat d'une demande de clarification ciblée sans répondre au hasard.
   │
   ├──► 2. Reformulation Multi-Requêtes (`llm.py`, `prompts.py`)
   │       └─ Le LLM génère 3 reformulations complémentaires de la question :
   │          (1) Terminologie technique officielle dans la langue de l'utilisateur
   │          (2) Formulation orientée symptôme / voyant / action
   │          (3) Traduction technique en anglais automobile (pour interroger les notices EN)
   │
   ├──► 3. Recherche Vectorielle Multilingue & Fusion RRF (`embeddings.py`, `index.py`, `pipeline.py`)
   │       ├─ Encodage via Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)
   │       ├─ Recherche dans l'index FAISS (`index/corpus.faiss`) avec filtrage par modèle/catégorie
   │       └─ Fusion des classements par Reciprocal Rank Fusion (RRF, k = 60) et normalisation des scores [0, 1]
   │
   └──► 4. Génération Contrainte & Sourcée (`llm.py`, `prompts.py`)
           ├─ Réponse dans la langue de la question (Français ou Anglais), même si la notice source est dans l'autre langue
           ├─ Ancrage strict sur <CONTEXTE> (zéro hallucination, refus strict si hors-notice)
           ├─ Alerte de sécurité explicite et recommandation d'un professionnel pour les pannes critiques
           └─ Citations [p. X] dans le texte et bloc final structuré "Références :"
```

---

## 2. Structure du Dépôt

```text
Chatbot_Mecano_en-poche_RAG/
├── README.md                    ← Documentation complète du projet, architecture et guide d'utilisation
├── FORMAT_CORPUS.md             ← Spécification du schéma JSONL et règles de découpage du corpus
├── requirements.txt             ← Dépendances Python
├── config.yaml                  ← Configuration centrale (embeddings, LLM Gemini, top-k, Multi-Query RRF, chemins)
├── .env.example                 ← Modèle de fichier d'environnement pour la clé API Gemini
│
├── data/                        ← Données brutes
│   └── pdf/                     ← Notices constructeurs au format PDF
│       └── SOURCES.md           ← Origine et métadonnées des manuels officiels
│
├── corpus/                      ← ÉTAPE 0 : Extraction, nettoyage, validation et fusion du corpus
│   ├── corpus_schema.json       ← Schéma de validation d'un passage
│   ├── validate_corpus.py       ← Script de contrôle qualité (longueur, doublons, catégories, pages)
│   ├── merge_corpus.py          ← Fusion des 4 notices en un corpus unifié
│   ├── scripts/
│   │   ├── build_clio4.py       ← Extraction Renault Clio 4 (FR)
│   │   ├── build_ecosport.py    ← Extraction Ford EcoSport (EN)
│   │   ├── build_i10.py         ← Extraction Hyundai i10 (EN)
│   │   └── build_picanto.py     ← Extraction Kia Picanto (EN)
│   └── output/
│       ├── clio4_corpus.jsonl   ← 246 passages validés
│       ├── ecosport_corpus.jsonl← 219 passages validés
│       ├── i10_corpus.jsonl     ← 294 passages validés
│       ├── picanto_corpus.jsonl ← 360 passages validés
│       └── all_corpus.jsonl     ← Corpus fusionné (1 119 passages)
│
├── src/mecano/                  ← ÉTAPES 1 & 2 : Moteur de recherche et de génération RAG
│   ├── config.py                ← Chargement de config.yaml et des variables d'environnement (.env)
│   ├── corpus_loader.py         ← Lecture et filtrage des fichiers JSONL
│   ├── embeddings.py            ← Vectorisation Sentence-BERT multilingue
│   ├── index.py                 ← Construction et chargement de l'index FAISS
│   ├── retriever.py             ← Recherche sémantique top-k avec filtres
│   ├── ner.py                   ← Extraction d'entités : modèle de véhicule et codes défaut OBD-II
│   ├── prompts.py               ← Prompts système, détection d'ambiguïté, formatage du contexte
│   ├── llm.py                   ← Client API Google Gemini (génération et reformulation multi-requêtes)
│   └── pipeline.py              ← Orchestration complète : Question → Multi-Query + RRF → Réponse + Sources
│
├── index/                       ← Fichiers d'index générés (corpus.faiss, corpus_metadata.json)
│
├── scripts/                     ← Outils en ligne de commande (CLI)
│   ├── build_index.py           ← Génère les embeddings et sauvegarde l'index FAISS
│   ├── search.py                ← Teste la recherche vectorielle seule (comparaison top-k=3 et top-k=5)
│   └── ask.py                   ← Interroge le pipeline RAG complet et compare le gain Multi-Query + RRF
│
├── app/                         ← ÉTAPE 3 : Interface utilisateur
│   ├── streamlit_app.py         ← Application web Streamlit (interface technique sobre et professionnelle)
│   ├── demo_backend.py          ← Moteur de secours lexical (fallback hors-ligne)
│   └── README.md                ← Documentation dédiée à l'interface et au contrat d'API
│
├── evaluation/                  ← ÉTAPE 4 : Évaluation du système
│   ├── questions.jsonl          ← Jeu de questions de test (avec questions pièges hors-notice)
│   ├── evaluate_retrieval.py    ← Évaluation de la pertinence de recherche (top-k=3 vs top-k=5)
│   └── evaluate_generation.py   ← Évaluation de la fidélité et de la détection d'information absente
│
└── tests/                       ← Tests unitaires automatisés
    └── test_generation.py       ← Tests RRF, formatage des prompts, détection d'ambiguïté et contrat Streamlit
```

---

## 3. Détail du Travail Réalisé par Étape

### Étape 0 : Constitution et Validation du Corpus (`corpus/`)
Les notices officielles de 4 véhicules ont été extraites, nettoyées (suppression des en-têtes/pieds de page répétitifs, sommaires et index, préservation des tableaux techniques et des avertissements de sécurité) et découpées en passages cohérents de 40 à 350 mots :

| Véhicule | Années | Langue de la notice | Script d'extraction | Passages validés |
|---|---|---|---|---|
| **Renault Clio 4** | 2016–2020 | Français (`fr`) | `corpus/scripts/build_clio4.py` | 246 |
| **Ford EcoSport** | 2018–2022 | Anglais (`en`) | `corpus/scripts/build_ecosport.py` | 219 |
| **Hyundai i10** | 2014–2019 | Anglais (`en`) | `corpus/scripts/build_i10.py` | 294 |
| **Kia Picanto** | 2017–2021 | Anglais (`en`) | `corpus/scripts/build_picanto.py` | 360 |
| **Total fusionné** | — | Multilingue (`fr` / `en`) | `corpus/merge_corpus.py` | **1 119** |

Chaque passage respecte strictement `corpus/corpus_schema.json` (validé avec `0 erreur, 0 avertissement` via `corpus/validate_corpus.py`) et contient : `id`, `marque`, `modele`, `annees`, `langue`, `document`, `source_url`, `chapitre`, `titre`, `categorie`, `pages` et `texte`.

### Étape 1 : Recherche Vectorielle Multilingue et Index FAISS (`src/mecano/embeddings.py`, `index.py`, `retriever.py`)
- **Modèle d'embeddings :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions), permettant d'interroger en français des notices rédigées en anglais (et inversement).
- **Indexation FAISS :** vecteurs normalisés ($L_2$) indexés dans `IndexFlatIP` (produit scalaire équivalent à la similarité cosinus).
- **Filtrage et Top-$k$ :** supporte le filtrage par modèle (`Clio 4`, `i10`, `EcoSport`, `Picanto`) et par catégorie (`voyant`, `entretien`, `fusibles`, `pneus_roues`, `demarrage_moteur`, etc.), ainsi que le réglage dynamique de `top_k` ($3, 5, 8, 10$).

### Étape 2 : Génération LLM, Multi-Query + RRF et Gestion de l'Ambiguïté (`src/mecano/llm.py`, `prompts.py`, `pipeline.py`)
- **Modèle LLM :** Google Gemini (`gemini-flash-lite-latest`) configuré avec une température basse (`0.1`) pour garantir des réponses déterministes et factuelles.
- **Bilinguisme automatique (FR / EN) :** le système détecte la langue de la question du conducteur et répond intégralement dans cette langue, en traduisant à la volée les termes techniques si la notice source est en anglais.
- **Fidélité stricte & Refus hors-notice :**
  - Le LLM répond uniquement à partir des passages fournis dans `<CONTEXTE>`.
  - Si l'information demandée ne figure pas dans la notice (ex. : prix d'une vidange à Tunis, tarif d'une pièce, assurance), le système renvoie la phrase normalisée : *"Cette information n'est pas présente dans la notice officielle du véhicule."* et positionne l'indicateur `info_absente = True`.
- **Sécurité & Recommandation garagiste :** pour toute anomalie critique (pression d'huile, surchauffe moteur, système de freinage, airbag, direction), la réponse commence par une mention explicite `**ALERTE DE SÉCURITÉ :**`, rappelle la consigne d'immobilisation du véhicule et recommande de consulter un réparateur qualifié.
- **Citations et bloc de références :** chaque réponse cite les pages `[p. X]` dans le texte et se termine par un bloc structuré `**Références :**` détaillant le modèle, le chapitre, le titre, la page et l'identifiant du passage.
- **Bonus — Reformulation Multi-Requêtes + Reciprocal Rank Fusion (RRF) :**
  - Avant la recherche, le LLM génère 3 reformulations complémentaires de la question initiale (correction orthographique/terminologique, approche symptôme, et traduction technique en anglais).
  - Les listes de résultats issues de la question originale et des 3 reformulations sont fusionnées via la formule **Reciprocal Rank Fusion** ($k = 60$) :
    $$\text{Score}_{\text{RRF}}(d) = \sum_{q \in Q} \frac{1}{k + \text{rang}_q(d)}$$
  - Les scores sont ensuite normalisés dans $[0, 1]$ pour l'affichage dans l'interface.
- **Bonus — Détection des questions ambiguës :**
  - Lorsqu'une question est trop vague pour identifier un passage précis sans deviner (ex. : *"J'ai un voyant rouge allumé sur mon tableau de bord, que faire ?"*), le système détecte l'ambiguïté (`is_question_ambiguous`) et pose une question de clarification structurée (`needs_clarification = True`) demandant la forme du symbole, son comportement (fixe/clignotant), le message affiché et le modèle du véhicule.

### Étape 3 : Interface Web Streamlit & Détection NER (`app/streamlit_app.py`, `src/mecano/ner.py`)
- **Interface sobre et professionnelle :** interface épurée sans emojis superflus, pensée comme un outil de documentation technique constructeur.
- **Détection automatique d'entités (`src/mecano/ner.py`) :**
  - Détecte automatiquement le modèle de voiture mentionné dans la question et applique le filtre correspondant.
  - Détecte les codes défaut **OBD-II** (`P0420`, `B...`, `C...`, `U...`), identifie la famille du calculateur concerné et affiche un bandeau technique explicatif.
- **Bandeaux d'état et d'urgence :** affichage automatique d'un bandeau d'alerte (*Arrêt impératif recommandé*, *Contrôle atelier recommandé*, ou *Information hors notice*) selon le contenu des sources.
- **Visualisation des sources :** panneau dépliable affichant pour chaque extrait consulté son titre, le véhicule, la langue (`[FR]` / `[EN]`), la page exacte, la section, le score de pertinence en pourcentage, l'extrait textuel et le lien vers le PDF officiel.

---

## 4. Installation et Configuration

### Prérequis
- Python 3.10 ou supérieur
- Une clé API Google Gemini (gratuite via Google AI Studio)

### 1. Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/hassine22/Chatbot_Mecano_en-poche_RAG.git
cd Chatbot_Mecano_en-poche_RAG
pip install -r requirements.txt
```

### 2. Configurer la clé API Gemini

Copiez le fichier `.env.example` vers `.env` à la racine du dépôt (le fichier `.env` est ignoré par Git) :

```ini
GEMINI_API_KEY=votre_cle_api_gemini
```

---

## 5. Guide d'Exécution

### A. Valider et fusionner les corpus (déjà généré dans `corpus/output/all_corpus.jsonl`)

```bash
python corpus/validate_corpus.py corpus/output/clio4_corpus.jsonl
python corpus/validate_corpus.py corpus/output/ecosport_corpus.jsonl
python corpus/validate_corpus.py corpus/output/i10_corpus.jsonl
python corpus/validate_corpus.py corpus/output/picanto_corpus.jsonl
python corpus/merge_corpus.py
```

### B. Construire l'index vectoriel FAISS

Sous Windows PowerShell :
```powershell
$env:PYTHONPATH = "src"
python scripts\build_index.py
```

Sous Linux / macOS :
```bash
PYTHONPATH=src python scripts/build_index.py
```
Cela génère `index/corpus.faiss` et `index/corpus_metadata.json`. (Si l'index FAISS n'est pas encore construit sur une machine, `pipeline.py` bascule automatiquement sur un moteur de recherche lexical de secours afin que l'application reste toujours fonctionnelle).

### C. Lancer l'interface web Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```
L'application est accessible dans le navigateur à l'adresse : **`http://localhost:8501`**

### D. Utiliser le chatbot en ligne de commande (`scripts/ask.py`)

```bash
# 1. Question précise avec filtre de modèle
python -X utf8 scripts/ask.py "Que faire si le voyant de pression d'huile s'allume sur ma Clio 4 ?" --modele "Clio 4" --top-k 5

# 2. Question en anglais
python -X utf8 scripts/ask.py "How do I jump start my i10 if the battery is flat?" --modele "i10"

# 3. Question ambiguë (déclenche la demande de clarification)
python -X utf8 scripts/ask.py "J'ai un voyant rouge qui clignote sur mon tableau de bord, que faire ?"

# 4. Question hors-notice (déclenche le refus strict)
python -X utf8 scripts/ask.py "Combien coûte une vidange chez Renault à Tunis ?" --modele "Clio 4"

# 5. Comparer la recherche directe vs Multi-Query + RRF sur une question comportant des fautes
python -X utf8 scripts/ask.py "voyan batri rouj allumé que fair" --modele "i10" --compare-multiquery
```

### E. Exécuter les tests unitaires

```bash
python -X utf8 -m unittest discover -s tests -v
```

---

## 6. État d'Avancement du Projet

- [x] **Étape 0 — Corpus :** Extraction, nettoyage, validation (`0 erreur`) et fusion des 4 notices constructeurs (`1 119` passages dans `all_corpus.jsonl`).
- [x] **Étape 1 — Recherche :** Embeddings multilingues Sentence-BERT + indexation FAISS + filtres par modèle et catégorie.
- [x] **Étape 2 — Génération :** Intégration Google Gemini, prompt d'ancrage strict bilingue FR/EN, refus hors-notice, alertes de sécurité, citations des pages, **Multi-Query + RRF** et **clarification des questions ambiguës**.
- [x] **Étape 3 — Interface :** Application Streamlit professionnelle connectée au pipeline RAG, détection automatique du véhicule et des codes OBD-II, visualisation des sources et du niveau d'urgence.
- [ ] **Étape 4 — Évaluation :** Exécution du protocole d'évaluation sur les 10+ questions de test (`evaluation/`).
- [ ] **Étape 5 — Rapport & Soutenance :** Rédaction finale et support de présentation.

## Droits d'auteur

Les notices d'utilisation appartiennent à leurs constructeurs respectifs (Renault, Ford, Hyundai, Kia). Elles sont utilisées exclusivement dans un cadre pédagogique et académique.
