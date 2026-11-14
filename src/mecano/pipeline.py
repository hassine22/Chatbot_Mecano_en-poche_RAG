"""
Enchaîne recherche + génération avec mémoire conversationnelle (<HISTORIQUE>) :
question (+ historique) → contextualisation & détection d'ambiguïté → reformulations (Multi-Query) →
recherche (VectorRetriever FAISS ou secours lexical) → fusion RRF → LLM → réponse + sources.

Respecte le contrat attendu par app/streamlit_app.py (Membre 3), scripts/ask.py (Membre 2) et evaluation/ (Membre 4).
"""
import json
import re
from mecano.config import get_project_root, load_config
from mecano.corpus_loader import load_all_corpus
from mecano.prompts import (
    build_rag_prompt,
    is_question_ambiguous,
    build_clarification_response,
    contextualize_question_with_history,
)
from mecano.llm import generate_response, generate_reformulations

_RETRIEVER_CACHE = None

MODELE_ALIASES = {
    "clio4": "Clio 4",
    "clio 4": "Clio 4",
    "clio": "Clio 4",
    "renault clio 4": "Clio 4",
    "ecosport": "EcoSport",
    "ford ecosport": "EcoSport",
    "i10": "i10",
    "hyundai i10": "i10",
    "picanto": "Picanto",
    "kia picanto": "Picanto",
}


def normalize_modele(modele: str | None) -> str | None:
    """Normalise un identifiant de modèle (ex: 'clio4' -> 'Clio 4') pour VectorRetriever."""
    if modele is None:
        return None
    clean = str(modele).strip()
    if not clean:
        return None
    return MODELE_ALIASES.get(clean.lower(), clean)


def infer_modele_from_history(historique: list[dict] | None) -> str | None:
    """Récupère le dernier modèle de véhicule actif dans l'historique de conversation."""
    if not historique:
        return None
    for turn in reversed(historique):
        if not isinstance(turn, dict):
            continue
        mod = turn.get("modele")
        if mod:
            return normalize_modele(mod)
        filters = turn.get("filters") or (turn.get("resultat", {}) if isinstance(turn.get("resultat"), dict) else {}).get("filters")
        if isinstance(filters, dict) and filters.get("modele"):
            return normalize_modele(filters["modele"])
    return None


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5
) -> list[dict]:
    """
    Combine plusieurs listes ordonnées de passages avec la formule RRF :
    Score_RRF(d) = sum(1 / (k + rang))
    Normalise également 'score' entre 0 et 1 pour l'affichage Streamlit,
    tout en conservant 'score_rrf' (score RRF brut).
    """
    if not results_lists:
        return []

    scores_rrf = {}
    best_sim_score = {}
    passages_by_id = {}
    n_lists = len(results_lists)
    max_possible_rrf = n_lists * (1.0 / (k + 1))

    for r_list in results_lists:
        for rank, item in enumerate(r_list, 1):
            pid = item.get("id")
            if not pid:
                continue
            if pid not in passages_by_id:
                passages_by_id[pid] = dict(item)
            scores_rrf[pid] = scores_rrf.get(pid, 0.0) + (1.0 / (k + rank))
            raw_s = float(item.get("score", 0.0))
            if raw_s <= 1.0 and raw_s > best_sim_score.get(pid, 0.0):
                best_sim_score[pid] = raw_s

    sorted_ids = sorted(scores_rrf.keys(), key=lambda pid: scores_rrf[pid], reverse=True)

    fused = []
    for pid in sorted_ids[:top_k]:
        p = dict(passages_by_id[pid])
        rrf_val = round(scores_rrf[pid], 4)
        p["score_rrf"] = rrf_val
        norm_rrf = min(1.0, scores_rrf[pid] / max_possible_rrf) if max_possible_rrf > 0 else 0.0
        if pid in best_sim_score and best_sim_score[pid] > 0:
            p["score"] = round(0.5 * norm_rrf + 0.5 * best_sim_score[pid], 4)
        else:
            p["score"] = round(norm_rrf, 4)
        fused.append(p)

    return fused


STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "sur", "un", "une", "que", "qui", "faire", "fait",
    "est", "pour", "dans", "par", "en", "et", "ou", "si", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "ce", "cet", "cette", "ces", "au", "aux", "avec", "sans", "sous", "vers",
    "quand", "comment", "quel", "quelle", "quels", "quelles", "quoi", "dois", "je", "il", "elle",
    "on", "nous", "vous", "ils", "elles", "ne", "pas", "plus", "grave", "monte", "allume", "roulant",
    "pendant", "roule", "veut", "dire", "signifie", "faut",
    "the", "of", "in", "on", "at", "to", "for", "with", "is", "are", "was", "a", "an", "and", "or",
    "what", "how", "do", "does", "if", "my", "your", "its", "their", "this", "that", "when", "while",
    "should", "can", "it", "mean", "means"
}

CAR_NAMES = {"clio", "clio4", "i10", "picanto", "ecosport", "renault", "hyundai", "kia", "ford"}


def _match_filters(item: dict, modele: str | None = None, categories: list[str] | None = None) -> bool:
    """Vérifie si un passage correspond aux filtres de modèle et de catégories."""
    if modele:
        norm_mod = normalize_modele(modele)
        mod_clean = str(norm_mod).lower().replace(" ", "")
        item_mod = str(item.get("modele", "")).lower().replace(" ", "")
        item_marque = str(item.get("marque", "")).lower()
        if mod_clean not in item_mod and mod_clean not in item_marque:
            return False

    if categories:
        allowed = {str(c).lower() for c in categories}
        if str(item.get("categorie", "")).lower() not in allowed:
            return False

    return True


def _try_vector_retriever(
    query: str,
    modele: str | None = None,
    categories: list[str] | None = None,
    top_k: int = 5,
    config: dict | None = None
) -> list[dict] | None:
    """Utilise VectorRetriever (src/mecano/retriever.py) et l'index FAISS (index/corpus.faiss)."""
    global _RETRIEVER_CACHE
    root = get_project_root()
    cfg = config or load_config()
    idx_dir = root / cfg.get("index", {}).get("dir", "index")
    faiss_path = idx_dir / "corpus.faiss"
    meta_path = idx_dir / "corpus_metadata.json"

    if not faiss_path.is_file() or not meta_path.is_file():
        return None

    try:
        if _RETRIEVER_CACHE is None:
            from mecano.embeddings import DEFAULT_MODEL, EmbeddingModel
            from mecano.index import load_index
            from mecano.retriever import VectorRetriever

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            records = meta.get("records", [])
            emb_model = EmbeddingModel(meta.get("model", DEFAULT_MODEL))
            faiss_index = load_index(faiss_path)
            _RETRIEVER_CACHE = VectorRetriever(faiss_index, emb_model, records)

        retriever = _RETRIEVER_CACHE
        norm_modele = normalize_modele(modele)

        if not categories or len(categories) == 1:
            single_cat = categories[0] if categories else None
            raw_results = retriever.search(
                query,
                top_k=top_k,
                modele=norm_modele,
                categorie=single_cat,
            )
        else:
            allowed_cats = {str(c).strip().casefold() for c in categories if str(c).strip()}
            all_matches = retriever.search(
                query,
                top_k=len(retriever.records),
                modele=norm_modele,
                categorie=None,
            )
            raw_results = [
                r for r in all_matches
                if str(r.get("categorie", "")).strip().casefold() in allowed_cats
            ][:top_k]

        results = []
        for item in raw_results:
            item_copy = dict(item)
            item_copy["score"] = round(max(0.0, min(1.0, float(item_copy.get("score", 0.0)))), 4)
            results.append(item_copy)
        return results
    except Exception:
        return None


def _fallback_search(
    query: str,
    corpus: list[dict],
    modele: str | None = None,
    categories: list[str] | None = None,
    top_k: int = 5
) -> list[dict]:
    """Recherche lexicale autonome de secours si faiss/sentence-transformers ne sont pas installés."""
    all_words = re.findall(r"\w+", query.lower())
    query_terms = [w for w in all_words if len(w) >= 2 and w not in STOP_WORDS and w not in CAR_NAMES]
    if not query_terms:
        query_terms = [w for w in all_words if len(w) >= 2 and w not in STOP_WORDS]
    if not query_terms:
        return []

    max_possible = float(len(query_terms) * 4.0)
    scored = []
    for item in corpus:
        if not _match_filters(item, modele=modele, categories=categories):
            continue

        titre = str(item.get("titre", "")).lower()
        texte = str(item.get("texte", "")).lower()
        raw_score = 0.0
        for w in query_terms:
            if w in titre:
                raw_score += 3.0
            if w in texte:
                raw_score += 1.0

        if raw_score > 0:
            item_copy = dict(item)
            item_copy["score"] = round(min(1.0, raw_score / max_possible), 4)
            item_copy["_raw_score"] = raw_score
            scored.append(item_copy)

    scored.sort(key=lambda x: x["_raw_score"], reverse=True)
    for s in scored:
        s.pop("_raw_score", None)
    return scored[:top_k]


def retrieve_passages(
    query: str,
    modele: str | None = None,
    categories: list[str] | None = None,
    categorie: str | None = None,
    top_k: int = 5,
    config: dict | None = None
) -> list[dict]:
    """
    Appelle dans l'ordre :
    1. VectorRetriever (src/mecano/retriever.py + index/corpus.faiss)
    2. La recherche lexicale autonome de secours sur all_corpus.jsonl
    """
    cats_list = list(categories) if categories else ([categorie] if categorie else None)

    vec_res = _try_vector_retriever(
        query=query,
        modele=modele,
        categories=cats_list,
        top_k=top_k,
        config=config
    )
    if vec_res is not None:
        return vec_res

    corpus = load_all_corpus()
    return _fallback_search(
        query=query,
        corpus=corpus,
        modele=modele,
        categories=cats_list,
        top_k=top_k
    )


def _is_refusal_response(text: str) -> bool:
    """Détecte si la réponse du LLM indique que l'information est absente de la notice."""
    t_low = text.lower()
    return (
        "cette information n'est pas présente dans la notice" in t_low
        or "this information is not available in the official" in t_low
        or "n'est pas dans la notice" in t_low
        or "information absente" in t_low
    )


def ask(
    question: str,
    modele: str | None = None,
    categories: list[str] | None = None,
    top_k: int | None = None,
    categorie: str | None = None,
    use_multi_query: bool | None = None,
    config: dict | None = None,
    historique: list[dict] | None = None,
) -> dict:
    """
    Pipeline RAG complet avec mémoire conversationnelle, compatible avec app/streamlit_app.py et scripts/ask.py :
    Renvoie {
        "reponse": str,
        "answer": str,
        "info_absente": bool,
        "needs_clarification": bool,
        "sources": list[dict],
        "question": str,
        "contextualized_question": str,
        "reformulations": list[str],
        "filters": dict,
        "multi_query_used": bool,
    }
    """
    cfg = config or load_config()
    retrieval_cfg = cfg.get("retrieval", {})

    k = top_k if top_k is not None else int(retrieval_cfg.get("top_k", 5))
    do_multi_query = use_multi_query if use_multi_query is not None else bool(retrieval_cfg.get("use_multi_query", True))
    rrf_k = int(retrieval_cfg.get("rrf_k", 60))
    n_reforms = int(retrieval_cfg.get("num_reformulations", 3))
    cats_list = list(categories) if categories else ([categorie] if categorie else None)

    # 1. Mémoire du véhicule : si non spécifié sur ce tour, hériter du modèle des tours précédents
    norm_modele = normalize_modele(modele)
    if norm_modele is None and historique:
        norm_modele = infer_modele_from_history(historique)

    # 2. Contextualisation de la question avec l'historique récent (suivi ou réponse à une clarification)
    search_question = contextualize_question_with_history(question, historique)

    # 3. Détection de question trop vague / ambiguë (sur la question contextualisée)
    if is_question_ambiguous(search_question):
        clarification = build_clarification_response(search_question, modele=norm_modele)
        return {
            "reponse": clarification,
            "answer": clarification,
            "info_absente": False,
            "needs_clarification": True,
            "sources": [],
            "question": question,
            "contextualized_question": search_question,
            "reformulations": [],
            "filters": {"modele": norm_modele, "categories": cats_list, "categorie": categorie, "top_k": k},
            "multi_query_used": False,
        }

    # 4. Génération des reformulations Multi-Query (tenant compte de l'historique)
    reformulations = []
    if do_multi_query:
        reformulations = generate_reformulations(
            search_question,
            n=n_reforms,
            config=cfg,
            historique=historique,
        )

    all_queries = [search_question] + [r for r in reformulations if r != search_question]

    # 5. Exécuter la recherche pour chaque requête
    results_per_query = []
    for q in all_queries:
        res = retrieve_passages(
            query=q,
            modele=norm_modele,
            categories=cats_list,
            top_k=k * 2 if len(all_queries) > 1 else k,
            config=cfg
        )
        if res:
            results_per_query.append(res)

    # 6. Fusion RRF des résultats
    if not results_per_query:
        passages_finaux = []
    elif len(results_per_query) == 1:
        passages_finaux = results_per_query[0][:k]
    else:
        passages_finaux = reciprocal_rank_fusion(results_per_query, k=rrf_k, top_k=k)

    if not passages_finaux:
        refusal = (
            "Cette information n'est pas présente dans la notice officielle du véhicule "
            "(information absente). Nous vous conseillons de contacter un garagiste ou un concessionnaire."
        )
        return {
            "reponse": refusal,
            "answer": refusal,
            "info_absente": True,
            "needs_clarification": False,
            "sources": [],
            "question": question,
            "contextualized_question": search_question,
            "reformulations": reformulations,
            "filters": {"modele": norm_modele, "categories": cats_list, "categorie": categorie, "top_k": k},
            "multi_query_used": bool(reformulations),
        }

    # 7. Construction du prompt (avec <HISTORIQUE> et <CONTEXTE>) et génération LLM
    user_prompt = build_rag_prompt(question, passages_finaux, historique=historique)
    answer = generate_response(user_prompt=user_prompt, config=cfg)
    info_absente = _is_refusal_response(answer)
    needs_clarif = "Précision nécessaire" in answer or "Clarification needed" in answer

    return {
        "reponse": answer,
        "answer": answer,
        "info_absente": info_absente,
        "needs_clarification": needs_clarif,
        "sources": passages_finaux,
        "question": question,
        "contextualized_question": search_question,
        "reformulations": reformulations,
        "filters": {"modele": norm_modele, "categories": cats_list, "categorie": categorie, "top_k": k},
        "multi_query_used": bool(reformulations),
    }
