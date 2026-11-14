# Interface – Mécano de poche (membre 3)

## Lancer l'interface

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

L'interface s'ouvre sur http://localhost:8501.

## Fonctionnalités

| Fonction | Où | Barème |
|---|---|---|
| Question libre ou exemples cliquables | page principale | démo |
| Choix de la voiture, ou **détection automatique** du modèle cité dans la question | barre latérale + `src/mecano/ner.py` | bonus fonctionnalité innovante |
| Filtre par catégorie (voyants, pannes, fusibles…) | barre latérale | bonus filtres utiles |
| Réglage du **top-k** (3, 5, 8, 10) | barre latérale | « tester au moins deux top-k » |
| **Détection des codes défaut** OBD-II (P0420…) avec leur famille | `src/mecano/ner.py` | bonus NER |
| **Niveau d'urgence** (🛑 urgent / 🔧 à faire vérifier), déduit du passage | `streamlit_app.py` | bonus expérience utilisateur |
| Message clair quand **l'information est absente** des notices | `streamlit_app.py` | « réponse prudente » |
| **Sources visualisées** : titre, voiture, langue, page, catégorie, pertinence, lien vers la notice | panneau « Passages consultés » | bonus sources visualisées |

Captures d'écran : `docs/images/interface_*.png`.

## Mode démo et branchement du vrai RAG

Tant que `src/mecano/pipeline.py` ne contient pas de fonction `ask`, l'interface utilise
`app/demo_backend.py` :
- recherche **par mots-clés** (BM25), avec une petite traduction français → anglais ;
- **pas de LLM** : la réponse est l'extrait le plus pertinent.

Dès que `mecano.pipeline.ask` existe, l'interface l'utilise **automatiquement**. Le mode
affiché dans la barre latérale passe alors à « RAG complet ».

## Contrat à respecter par `src/mecano/pipeline.py` (membre 2)

```python
def ask(question: str, modele: str | None = None,
        categories: list[str] | None = None, top_k: int = 5) -> dict:
    ...
    return {
        "reponse": "texte de la réponse (markdown accepté)",
        "info_absente": False,      # True si les sources ne permettent pas de répondre
        "sources": [                # du plus pertinent au moins pertinent
            {**passage_du_corpus, "score": 0.83},   # score entre 0 et 1
        ],
    }
```

- `modele` : la valeur du champ `modele` du corpus (`"Clio 4"`, `"EcoSport"`, `"Picanto"`, `"i10"`), ou `None` pour toutes les voitures.
- `categories` : des valeurs du champ `categorie`, ou `None` pour toutes.
- Chaque source doit garder **tous les champs du passage** (`titre`, `texte`, `pages`, `marque`, `modele`, `langue`, `categorie`, `document`, `source_url`) et ajouter `score`.
