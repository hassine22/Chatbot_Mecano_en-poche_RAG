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

Thème « tableau de bord de nuit » : fond bleu nuit, police Saira embarquée (`app/static/fonts/`,
licence SIL OFL, fonctionne hors connexion), couleurs rouge / orange / vert réservées aux voyants
et au niveau d'urgence.

| Fonctionnalité | Implémentation | Objectif / Barème |
|---|---|---|
| **Choix de la voiture en premier** (4 cartes) | `app/streamlit_app.py` | Évite les réponses qui mélangent plusieurs notices |
| **Demande de la voiture si elle manque** | `traiter()` : la question est mise en attente, 4 boutons de choix | Pas de réponse au hasard quand le modèle est inconnu |
| **Panneau de voyants cliquables** (8 pictogrammes, test d'allumage au chargement) | `app/telltales.py` + `style.css` | Le conducteur clique sur ce qu'il voit, sans connaître le nom du voyant (bonus UX) |
| **Détection automatique du modèle et des codes défaut OBD-II** | `src/mecano/ner.py` | Bonus NER |
| **Verdict d'urgence** : arrêtez-vous / faites vérifier / pas d'urgence / pas dans la notice | `niveau()` | Réponse prudente, lisible en une seconde |
| **Pages citées en étiquettes** dans la réponse (`[p. 5.44]`) | `mise_en_forme()` | Sources visualisées, sans liste de références en double |
| **Extraits de la notice** : page, jauge de pertinence, texte, lien vers la notice | `afficher_extraits()` | Transparence de la recherche |
| **Avis 👍 / 👎** enregistré dans `evaluation/results/feedback.jsonl` | `enregistrer_avis()` | Matière pour l'évaluation |
| **Mode hors ligne** (sans LLM) + **repli automatique** si Gemini ne répond pas | `interroger()`, `app/demo_backend.py` | La démo continue même si le quota est atteint |
| **Réglages avancés** : catégories de la notice, top-k (3, 5, 8, 10) | barre latérale | Comparer plusieurs valeurs de top-k |
| **Affichage mobile** | `style.css` (media queries) | Démo sur téléphone possible |

Captures : `docs/images/interface_*.png`.

Forcer le mode hors ligne au lancement : `MECANO_HORS_LIGNE=1` (Windows : `set MECANO_HORS_LIGNE=1`).

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
    historique: list[dict] | None = None,
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
