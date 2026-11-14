# Interface Web — Mécano de poche (`app/streamlit_app.py`)

## Lancer l'interface

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

L'interface s'ouvre dans le navigateur à l'adresse : **`http://localhost:8501`**.

---

## Fonctionnalités de l'Interface

| Fonctionnalité | Implémentation | Objectif / Barème |
|---|---|---|
| **Saisie libre et questions fréquentes** | Page principale (`app/streamlit_app.py`) | Prise en main immédiate lors de la démonstration |
| **Sélection du véhicule ou détection automatique** | Barre latérale + `src/mecano/ner.py` | Filtrage automatique du modèle mentionné dans la question |
| **Filtrage par section de la notice** | Barre latérale (`multiselect` sur 16 catégories) | Restreindre la recherche aux voyants, fusibles, entretien, etc. |
| **Réglage dynamique du `top-k`** | Barre latérale (`select_slider` : 3, 5, 8, 10) | Tester et comparer l'impact du nombre de passages extraits |
| **Détection des codes défaut OBD-II** | `src/mecano/ner.py` (ex. `P0420`, `C0123`) | Identification automatique du code et de la famille calculateur |
| **Bandeaux techniques de niveau d'urgence** | `niveau_urgence()` dans `app/streamlit_app.py` | Mise en évidence sobre des consignes d'arrêt impératif ou de contrôle atelier |
| **Signalement d'information hors notice** | Indicateur `info_absente` | Refus explicite lorsque la question sort du périmètre des manuels |
| **Clarification des questions ambiguës** | `src/mecano/prompts.py` + `pipeline.py` | Demande de précision ciblée au lieu de répondre au hasard |
| **Reformulation Multi-Query + Fusion RRF** | `src/mecano/pipeline.py` + `llm.py` | Robustesse face aux fautes de frappe et au croisement de langues FR/EN |
| **Visualisation détaillée des sources** | Panneau dépliable `Extraits des notices consultés` | Affiche titre, modèle, langue (`[FR]`/`[EN]`), page, section, score `%`, extrait et lien PDF |

---

## Architecture et Contrat avec le Pipeline RAG (`src/mecano/pipeline.py`)

Au démarrage, `app/streamlit_app.py` charge automatiquement `mecano.pipeline.ask` et active le moteur **`RAG Actif (FAISS + Gemini)`**. En cas d'absence du module, il bascule sur `app/demo_backend.py` qui respecte exactement la même signature.

### Signature et format de retour de `mecano.pipeline.ask`

```python
def ask(
    question: str,
    modele: str | None = None,
    categorie: str | None = None,
    categories: list[str] | None = None,
    top_k: int = 5,
    use_multi_query: bool | None = None,
) -> dict:
    return {
        "reponse": "Texte de la réponse générée en Markdown avec citations [p. X] et bloc Références",
        "answer": "Alias identique à reponse (compatibilité CLI)",
        "info_absente": False,          # True si l'information ne figure pas dans les notices
        "needs_clarification": False,   # True si la question est trop vague et demande précision
        "sources": [                    # Passages triés par pertinence décroissante
            {
                **passage_du_corpus,
                "score": 0.89,          # Score de pertinence normalisé dans [0.0, 1.0]
            }
        ],
        "reformulations": [...],        # Les 3 reformulations générées par Gemini (si Multi-Query actif)
        "multi_query_used": True,
    }
```

- **`modele`** : accepte aussi bien le nom d'affichage (`"Clio 4"`, `"EcoSport"`, `"Picanto"`, `"i10"`) que l'identifiant court (`"clio4"`, `"ecosport"`, `"picanto"`, `"i10"`), ou `None` pour chercher dans toutes les notices.
- **`categories`** : liste de clés de catégories (`["voyant", "anomalie"]`) ou `None` pour toutes les sections.
- **`sources`** : conserve l'intégralité des métadonnées du schéma JSONL (`id`, `titre`, `texte`, `pages`, `marque`, `modele`, `annees`, `langue`, `categorie`, `document`, `source_url`) enrichies du champ `score` normalisé entre `0` et `1`.
