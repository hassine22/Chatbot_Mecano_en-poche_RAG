"""
Enchaîne recherche + génération : question → détection d'ambiguïté → recherche (Multi-Query + RRF) → réponse + sources.
"""
import re
from mecano.config import load_config
from mecano.corpus_loader import load_all_corpus
from mecano.prompts import (
    build_rag_prompt,
    is_question_ambiguous,
    build_clarification_response,
)
from mecano.llm import generate_response, generate_reformulations


def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5
) -> list[dict]:
    """
    Combine plusieurs listes ordonnées de passages avec la formule RRF :
    Score_RRF(d) = sum(1 / (k + rang))
    """
    scores = {}
    passages_by_id = {}
    
    for r_list in results_lists:
        for rank, item in enumerate(r_list, 1):
            pid = item.get("id")
            if not pid:
                continue
            if pid not in passages_by_id:
                passages_by_id[pid] = dict(item)
            scores[pid] = scores.get(pid, 0.0) + (1.0 / (k + rank))
            
    sorted_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
    
    fused = []
    for pid in sorted_ids[:top_k]:
        p = dict(passages_by_id[pid])
        p["score"] = round(scores[pid], 4)
        fused.append(p)
        
    return fused


STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "sur", "un", "une", "que", "qui", "faire", "fait",
    "est", "pour", "dans", "par", "en", "et", "ou", "si", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "ce", "cet", "cette", "ces", "au", "aux", "avec", "sans", "sous",
    "the", "of", "in", "on", "at", "to", "for", "with", "is", "are", "was", "a", "an", "and", "or",
    "what", "how", "do", "does", "if", "my", "your", "its", "their", "this", "that"
}

CAR_NAMES = {"clio", "clio4", "i10", "picanto", "ecosport", "renault", "hyundai", "kia", "ford"}


def _fallback_search(
    query: str,
    corpus: list[dict],
    modele: str = None,
    categorie: str = None,
    top_k: int = 5
) -> list[dict]:
    """Recherche lexicale autonome de secours si l'index FAISS n'est pas encore prêt."""
    all_words = re.findall(r"\w+", query.lower())
    query_terms = [w for w in all_words if len(w) >= 2 and w not in STOP_WORDS and w not in CAR_NAMES]
    if not query_terms:
        query_terms = [w for w in all_words if len(w) >= 2 and w not in STOP_WORDS]
    if not query_terms:
        return []
        
    scored = []
    for item in corpus:
        # Filtre modèle
        if modele:
            mod_clean = str(modele).lower().replace(" ", "")
            item_mod = str(item.get("modele", "")).lower().replace(" ", "")
            item_marque = str(item.get("marque", "")).lower()
            if mod_clean not in item_mod and mod_clean not in item_marque:
                continue
                
        # Filtre catégorie
        if categorie:
            if str(item.get("categorie", "")).lower() != str(categorie).lower():
                continue
                
        # Calcul du score d'occurrence pondéré
        titre = str(item.get("titre", "")).lower()
        texte = str(item.get("texte", "")).lower()
        score = 0.0
        for w in query_terms:
            if w in titre:
                score += 3.0
            if w in texte:
                score += 1.0
                
        if score > 0:
            item_copy = dict(item)
            item_copy["score"] = round(score, 2)
            scored.append(item_copy)
            
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def retrieve_passages(
    query: str,
    modele: str = None,
    categorie: str = None,
    top_k: int = 5,
    config: dict = None
) -> list[dict]:
    """
    Appelle le retriever de Membre 1 (src.mecano.retriever) s'il est prêt,
    sinon bascule sur la recherche lexicale autonome.
    """
    # 1. Tenter d'utiliser le module retriever officiel s'il est implémenté
    try:
        from mecano import retriever
        for fn_name in ("retrieve", "search", "rechercher"):
            if hasattr(retriever, fn_name):
                fn = getattr(retriever, fn_name)
                return fn(query=query, modele=modele, categorie=categorie, top_k=top_k)
    except Exception:
        pass
        
    # 2. Fallback de secours autonome sur le corpus
    corpus = load_all_corpus()
    return _fallback_search(
        query=query,
        corpus=corpus,
        modele=modele,
        categorie=categorie,
        top_k=top_k
    )


def ask(
    question: str,
    modele: str = None,
    categorie: str = None,
    top_k: int = None,
    use_multi_query: bool = None,
    config: dict = None
) -> dict:
    """
    Pipeline RAG complet :
    question → détection d'ambiguïté → reformulations (optionnel) → recherche(s) → fusion RRF → LLM → réponse + sources.
    """
    cfg = config or load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    
    k = top_k if top_k is not None else int(retrieval_cfg.get("top_k", 5))
    do_multi_query = use_multi_query if use_multi_query is not None else bool(retrieval_cfg.get("use_multi_query", True))
    rrf_k = int(retrieval_cfg.get("rrf_k", 60))
    n_reforms = int(retrieval_cfg.get("num_reformulations", 3))

    # 1. Détection de question trop vague / ambiguë avant de répondre au hasard
    if is_question_ambiguous(question):
        clarification = build_clarification_response(question, modele=modele)
        return {
            "question": question,
            "reformulations": [],
            "answer": clarification,
            "sources": [],
            "filters": {"modele": modele, "categorie": categorie, "top_k": k},
            "multi_query_used": False,
            "needs_clarification": True,
        }
    
    reformulations = []
    if do_multi_query:
        reformulations = generate_reformulations(question, n=n_reforms, config=cfg)
        
    all_queries = [question] + reformulations
    
    # 2. Exécuter la recherche pour chaque requête
    results_per_query = []
    for q in all_queries:
        res = retrieve_passages(
            query=q,
            modele=modele,
            categorie=categorie,
            top_k=k * 2 if len(all_queries) > 1 else k,
            config=cfg
        )
        if res:
            results_per_query.append(res)
            
    # 3. Fusion des résultats
    if not results_per_query:
        passages_finaux = []
    elif len(results_per_query) == 1:
        passages_finaux = results_per_query[0][:k]
    else:
        passages_finaux = reciprocal_rank_fusion(results_per_query, k=rrf_k, top_k=k)
        
    # Si aucun passage n'est trouvé
    if not passages_finaux:
        answer = "Cette information n'est pas présente dans la notice officielle du véhicule."
        return {
            "question": question,
            "reformulations": reformulations,
            "answer": answer,
            "sources": [],
            "filters": {"modele": modele, "categorie": categorie, "top_k": k},
            "multi_query_used": bool(reformulations),
            "needs_clarification": False,
        }
        
    # 4. Construction du prompt et génération
    user_prompt = build_rag_prompt(question, passages_finaux)
    answer = generate_response(user_prompt=user_prompt, config=cfg)
    needs_clarif = "Précision nécessaire" in answer or "Clarification needed" in answer
    
    return {
        "question": question,
        "reformulations": reformulations,
        "answer": answer,
        "sources": passages_finaux,
        "filters": {"modele": modele, "categorie": categorie, "top_k": k},
        "multi_query_used": bool(reformulations),
        "needs_clarification": needs_clarif,
    }
