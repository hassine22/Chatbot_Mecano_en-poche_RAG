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
│   │   ├── build_ecosport.py    ← corpus Ford EcoSport
│   │   ├── build_i10.py         ← corpus Hyundai i10
│   │   ├── build_picanto.py     ← corpus Kia Picanto
│   │   └── build_obd.py         ← corpus des codes défaut (bonus)
│   └── output/
│       ├── clio4_corpus.jsonl
│       ├── ecosport_corpus.jsonl
│       ├── i10_corpus.jsonl
│       ├── picanto_corpus.jsonl
│       └── all_corpus.jsonl     ← corpus fusionné (généré)
│
├── src/mecano/                  ← ÉTAPES 2 à 4 : le moteur RAG
│   ├── config.py                ← lecture de config.yaml et .env
│   ├── corpus_loader.py         ← chargement du corpus
│   ├── embeddings.py            ← embeddings Sentence-BERT
│   ├── index.py                 ← index FAISS
│   ├── retriever.py             ← recherche top-k, filtres, recherche hybride
│   ├── ner.py                   ← détection des codes défaut et du modèle (bonus)
│   ├── prompts.py               ← prompts du LLM, réponse prudente, clarification
│   ├── llm.py                   ← appel du LLM (Google Gemini)
│   └── pipeline.py              ← question → Multi-Query + RRF → réponse + sources
│
├── index/                       ← index FAISS généré (non versionné)
│
├── scripts/                     ← commandes en ligne de commande
│   ├── build_index.py           ← construit l'index
│   ├── search.py                ← teste la recherche (plusieurs top-k)
│   └── ask.py                   ← pose une question (CLI + comparateur Multi-Query RRF)
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
│   ├── test_retriever.py
│   └── test_generation.py
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

Créer un fichier `.env` à la racine du dépôt à partir de `.env.example` :

```ini
GEMINI_API_KEY=votre_cle_api_ici
```

## Construire les embeddings et l'index FAISS

Les quatre corpus validés sont fusionnés dans `corpus/output/all_corpus.jsonl`.
Chaque passage est transformé en vecteur avec Sentence-BERT, puis ces vecteurs
sont rangés dans un index FAISS pour permettre une recherche sémantique rapide.
Le modèle utilisé est multilingue et convient aux notices françaises et anglaises :

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Depuis la racine du dépôt :

```bash
python corpus/merge_corpus.py
set PYTHONPATH=src
python scripts/build_index.py
```

Sous PowerShell :

```powershell
python corpus\merge_corpus.py
$env:PYTHONPATH = "src"
python scripts\build_index.py
```

Les fichiers générés sont :

```text
index/corpus.faiss             # vecteurs et structure de recherche FAISS
index/corpus_metadata.json     # passages, IDs, modèles et pages sources
```

Le fichier FAISS n'est pas lisible directement. Le résultat de recherche utilise
la position renvoyée par FAISS pour retrouver le passage correspondant dans
`corpus_metadata.json`. Les fichiers de l'index sont générés localement et ne
sont pas versionnés.

Pour reconstruire avec un autre corpus ou un autre modèle :

```powershell
$env:PYTHONPATH = "src"
python scripts\build_index.py `
  --corpus corpus\output\all_corpus.jsonl `
  --output-dir index `
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --batch-size 32
```

## Génération LLM, Multi-Query + RRF et Questions Ambiguës

Le pipeline de génération (`src/mecano/pipeline.py`, `src/mecano/prompts.py`, `src/mecano/llm.py`) intègre :
- **Détection de langue automatique (FR / EN) :** répond en français si la question est en français, et en anglais si la question est en anglais (en traduisant les notices anglaises ou françaises).
- **Reformulation multi-requêtes + Reciprocal Rank Fusion (RRF, $k=60$) :** le LLM génère 3 reformulations complémentaires (dont une en anglais technique automobile) fusionnées avec la requête originale via RRF pour maximiser le rappel sur les questions mal formulées ou comportant des fautes.
- **Questions ambiguës :** si la question est trop vague (ex. *« J'ai un voyant rouge allumé, que faire ? »*), l'assistant pose une question de clarification ciblée au lieu de répondre au hasard.
- **Ancrage strict, sécurité & références :** refus systématique si l'information est absente de la notice, alerte `🚨` et conseil d'un garagiste pour les voyants critiques, et bloc final `📚 Références :` citant le modèle, le chapitre, la page et l'identifiant du passage.

### Poser une question en ligne de commande (`scripts/ask.py`)

```bash
# Question standard avec filtre par modèle
python -X utf8 scripts/ask.py "Que faire si le voyant de pression d'huile s'allume sur ma i10 ?" --modele "i10" --top-k 5

# Comparer la recherche directe vs Multi-Query + RRF sur une question mal formulée
python -X utf8 scripts/ask.py "voyan batri rouj allumé que fair" --modele "i10" --compare-multiquery
```

## Lancer l'interface

```bash
streamlit run app/streamlit_app.py
```

Détails, fonctionnalités et contrat avec le pipeline : [`app/README.md`](app/README.md).

## Répartition

| Membre | Voiture | Branche | Passages |
|---|---|---|---|
| … | Renault Clio 4 | `corpus-clio4` ✅ | 246 |
| … | Ford EcoSport | `corpus-ecosport` ✅ | 219 |
| … | Hyundai i10 | `corpus-i10` ✅ | 294 |
| … | Kia Picanto | `corpus-picanto` ✅ | 360 |

## Étapes suivantes

- [x] Corpus des 4 voitures validés et fusionnés (`merge_corpus.py` → 1119 passages)
- [x] Embeddings (Sentence-BERT multilingue) + index FAISS
- [x] Prompt LLM : réponse fondée sur les passages, citation des pages, refus si l'info est absente, Multi-Query RRF & clarification des questions ambiguës
- [x] Interface (Streamlit) avec filtre par modèle
- [ ] 10 questions de test (dont 2 sans réponse) + évaluation
- [ ] Rapport

## Droits d'auteur

Les notices appartiennent aux constructeurs. Les PDF ne sont pas versionnés. Gardez ce dépôt **privé** tant qu'il contient les corpus extraits.
