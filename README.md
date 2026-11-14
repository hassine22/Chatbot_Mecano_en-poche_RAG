# Mécano de poche 🔧

Assistant RAG qui aide un conducteur à comprendre un voyant ou une panne, en s'appuyant **uniquement** sur la notice officielle de son véhicule. Il cite la page source et conseille un garagiste quand c'est nécessaire.

Mini-projet RAG & LLM – ENSI.

## Structure du dépôt

```
mecano-de-poche/
├── README.md                    ← présentation, installation, utilisation
├── FORMAT_CORPUS.md             ← règles communes pour les corpus
├── requirements.txt             ← dépendances Python
├── config.yaml                  ← paramètres (modèle d'embeddings, LLM, top-k, chemins)
├── .env.example                 ← modèle du fichier .env (clé d'API du LLM)
├── .gitignore
│
├── data/                        ← données brutes (non versionnées)
│   ├── pdf/                     ← notices PDF des voitures
│   │   └── SOURCES.md           ← liens de téléchargement des notices
│   └── obd/                     ← base des codes défaut OBD-II (bonus)
│
├── corpus/                      ← ÉTAPE 2 : préparation du corpus
│   ├── corpus_schema.json       ← format exact d'un passage
│   ├── validate_corpus.py       ← vérifie le format d'un corpus
│   ├── merge_corpus.py          ← fusionne les corpus des 4 voitures
│   ├── scripts/
│   │   ├── build_clio4.py       ← corpus Renault Clio 4 (référence)
│   │   ├── build_<slug>.py      ← un script par voiture
│   │   └── build_obd.py         ← corpus des codes défaut (bonus)
│   └── output/
│       ├── clio4_corpus.jsonl
│       ├── clio4_corpus.md
│       └── all_corpus.jsonl     ← corpus fusionné (généré)
│
├── src/mecano/                  ← ÉTAPES 2 à 4 : le moteur RAG
│   ├── config.py                ← lecture de config.yaml et .env
│   ├── corpus_loader.py         ← chargement du corpus
│   ├── embeddings.py            ← embeddings Sentence-BERT
│   ├── index.py                 ← index FAISS
│   ├── retriever.py             ← recherche top-k, filtres, recherche hybride
│   ├── ner.py                   ← détection des codes défaut et du modèle (bonus)
│   ├── prompts.py               ← prompts du LLM, réponse prudente
│   ├── llm.py                   ← appel du LLM (local ou API)
│   └── pipeline.py              ← question → recherche → réponse + sources
│
├── index/                       ← index FAISS généré (non versionné)
│
├── scripts/                     ← commandes en ligne de commande
│   ├── build_index.py           ← construit l'index
│   ├── search.py                ← teste la recherche (plusieurs top-k)
│   └── ask.py                   ← pose une question
│
├── app/
│   └── streamlit_app.py         ← interface web pour la démo
│
├── evaluation/                  ← ÉTAPE 5 : évaluation
│   ├── questions.jsonl          ← 10+ questions de test (dont 2 sans réponse)
│   ├── evaluate_retrieval.py    ← pertinence des passages retrouvés
│   ├── evaluate_generation.py   ← fidélité des réponses, info absente
│   └── results/                 ← résultats des évaluations
│
├── tests/                       ← tests unitaires
│   ├── test_corpus.py
│   └── test_retriever.py
│
├── notebooks/                   ← expérimentations (comparaison d'embeddings…)
│
└── docs/
    ├── rapport/                 ← rapport final
    ├── soutenance/              ← slides de la soutenance
    └── images/                  ← schémas, captures d'écran
```

## Installation

```bash
git clone <url-du-depot>
cd mecano-de-poche
pip install -r requirements.txt
```

## Faire le corpus de sa voiture (étape actuelle)

1. Lire **`FORMAT_CORPUS.md`** en entier.
2. Télécharger la notice dans `data/pdf/` et ajouter son lien dans `data/pdf/SOURCES.md`.
3. Copier le script de référence :
   ```bash
   cp corpus/scripts/build_clio4.py corpus/scripts/build_<slug>.py
   ```
4. Adapter `SLUG`, `META` et `KEEP` en haut du script (les instructions sont en tête du fichier).
5. Générer le corpus :
   ```bash
   python corpus/scripts/build_<slug>.py data/pdf/<notice>.pdf corpus/output
   ```
6. Relire `corpus/output/<slug>_corpus.md`, puis valider :
   ```bash
   python corpus/validate_corpus.py corpus/output/<slug>_corpus.jsonl
   ```
   → il faut **0 erreur** avant de pousser.
7. Pousser sur une branche `corpus-<slug>` et ouvrir une Pull Request.

## Répartition

| Membre | Voiture | Branche |
|---|---|---|
| … | Renault Clio 4 | `corpus-clio4` ✅ |
| … | Ford EcoSport | `corpus-ecosport` |
| … | … | … |
| … | … | … |

## Étapes suivantes

- [ ] Corpus des 4 voitures validés et fusionnés (`merge_corpus.py`)
- [ ] Embeddings (Sentence-BERT multilingue) + index FAISS
- [ ] Prompt LLM : réponse fondée sur les passages, citation des pages, refus si l'info est absente
- [ ] Interface (Streamlit) avec filtre par modèle
- [ ] 10 questions de test (dont 2 sans réponse) + évaluation
- [ ] Rapport

## Droits d'auteur

Les notices appartiennent aux constructeurs. Les PDF ne sont pas versionnés. Gardez ce dépôt **privé** tant qu'il contient les corpus extraits.
