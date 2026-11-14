"""
Tests unitaires pour le module de génération (Membre 2), la mémoire conversationnelle (<HISTORIQUE>),
et la compatibilité avec l'interface Streamlit (Membre 3) :
- RRF (Reciprocal Rank Fusion) et normalisation des scores dans [0, 1]
- Prompts, formatage du contexte et de l'historique conversationnel
- Détection des questions ambiguës / trop vagues (clarification) et résolution au tour suivant
- Persistance du modèle de véhicule dans l'historique multi-tours
- Pipeline RAG, refus sur information absente et contrat Streamlit (reponse, info_absente, sources, categories)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mecano.pipeline import reciprocal_rank_fusion, ask, infer_modele_from_history
from mecano.prompts import (
    build_rag_prompt,
    build_multiquery_prompt,
    is_question_ambiguous,
    contextualize_question_with_history,
)


class TestGeneration(unittest.TestCase):

    def test_rrf_ranking_and_deduplication(self):
        """Vérifie que RRF fusionne correctement les rangs, dédouble et normalise score dans [0, 1]."""
        list1 = [
            {"id": "doc1", "titre": "Titre 1", "pages": ["1.10"], "score": 0.9},
            {"id": "doc2", "titre": "Titre 2", "pages": ["1.20"], "score": 0.8},
        ]
        list2 = [
            {"id": "doc2", "titre": "Titre 2", "pages": ["1.20"], "score": 0.85},
            {"id": "doc3", "titre": "Titre 3", "pages": ["1.30"], "score": 0.7},
        ]

        fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["id"], "doc2")
        self.assertIn("score", fused[0])
        self.assertIn("score_rrf", fused[0])
        expected_rrf = round(1.0 / 62 + 1.0 / 61, 4)
        self.assertAlmostEqual(fused[0]["score_rrf"], expected_rrf, places=3)
        self.assertTrue(0.0 <= fused[0]["score"] <= 1.0)

    def test_build_rag_prompt_and_history(self):
        """Vérifie que le prompt RAG emballe bien l'historique <HISTORIQUE>, le contexte et les métadonnées."""
        passages = [
            {
                "id": "clio4-001",
                "marque": "Renault",
                "modele": "Clio 4",
                "annees": "2016-2020",
                "chapitre": "1. Faites connaissance",
                "titre": "Voyants",
                "categorie": "voyant",
                "pages": ["1.56"],
                "texte": "Le témoin s'allume au démarrage."
            }
        ]
        historique = [
            {
                "question": "J'ai un voyant rouge sur ma Clio 4",
                "resultat": {"reponse": "**Précision nécessaire :** Quel est le symbole ?", "needs_clarification": True},
                "modele": "Clio 4",
            }
        ]
        prompt = build_rag_prompt("C'est une burette d'huile", passages, historique=historique)
        self.assertIn("<HISTORIQUE>", prompt)
        self.assertIn("J'ai un voyant rouge sur ma Clio 4", prompt)
        self.assertIn("<CONTEXTE>", prompt)
        self.assertIn("clio4-001", prompt)
        self.assertIn("Page(s): 1.56", prompt)
        self.assertIn("Question du conducteur : C'est une burette d'huile", prompt)

    def test_build_multiquery_prompt(self):
        """Vérifie la construction du prompt de reformulation avec et sans historique."""
        p = build_multiquery_prompt("voyant huile clignote")
        self.assertIn("voyant huile clignote", p)

        hist = [{"question": "Comment vérifier l'huile sur i10 ?", "reponse": "Utilisez la jauge."}]
        p_hist = build_multiquery_prompt("Et quelle quantité ajouter ?", historique=hist)
        self.assertIn("<HISTORIQUE>", p_hist)
        self.assertIn("Comment vérifier l'huile sur i10 ?", p_hist)
        self.assertIn("Et quelle quantité ajouter ?", p_hist)

    def test_multi_turn_clarification_resolution_and_model_memory(self):
        """Vérifie qu'un second tour résout l'ambiguïté du premier tour et conserve le modèle en mémoire."""
        # Tour 1 : Question vague sur Clio 4 -> demande de clarification
        res_t1 = ask("J'ai un voyant rouge allumé sur mon tableau de bord, que faire ?", modele="Clio 4")
        self.assertTrue(res_t1["needs_clarification"])

        historique = [
            {
                "question": "J'ai un voyant rouge allumé sur mon tableau de bord, que faire ?",
                "resultat": res_t1,
                "modele": "Clio 4",
            }
        ]
        # Vérifie l'héritage du modèle depuis l'historique
        self.assertEqual(infer_modele_from_history(historique), "Clio 4")

        # Vérifie que la question de suivi combinée à l'historique n'est plus ambiguë
        contextualized = contextualize_question_with_history("C'est une burette d'huile", historique)
        self.assertFalse(is_question_ambiguous(contextualized))
        self.assertIn("burette d'huile", contextualized)

    def test_ambiguous_question_detection_and_clarification(self):
        """Vérifie que les questions trop vagues demandent une clarification au lieu de répondre au hasard."""
        self.assertTrue(is_question_ambiguous("J'ai un voyant rouge allumé que faire ?"))
        self.assertTrue(is_question_ambiguous("Ma voiture a un problème sur le tableau de bord"))
        self.assertTrue(is_question_ambiguous("My warning light is on what should I do?"))
        self.assertFalse(is_question_ambiguous("Que faire si le voyant de pression d'huile s'allume ?"))
        self.assertFalse(is_question_ambiguous("Erreur P0420 sur ma voiture"))

        res_fr = ask("J'ai un voyant rouge allumé sur mon tableau de bord, que faire ?", modele="Clio 4")
        self.assertTrue(res_fr["needs_clarification"])
        self.assertFalse(res_fr["info_absente"])
        self.assertIn("Précision nécessaire", res_fr["reponse"])
        self.assertEqual(res_fr["reponse"], res_fr["answer"])

        res_en = ask("A red warning light is blinking on my dashboard, what should I do?", modele="i10")
        self.assertTrue(res_en["needs_clarification"])
        self.assertFalse(res_en["info_absente"])
        self.assertIn("Clarification needed", res_en["reponse"])

    def test_refusal_and_streamlit_contract(self):
        """Vérifie le contrat Streamlit (reponse, info_absente, categories, sources)."""
        res = ask(
            question="Question introuvable xyz123456",
            modele="modele_inexistant_999",
            categories=["voyant"],
            top_k=3,
            use_multi_query=False
        )
        self.assertIn("reponse", res)
        self.assertIn("info_absente", res)
        self.assertIn("sources", res)
        self.assertTrue(res["info_absente"])
        self.assertFalse(res["needs_clarification"])
        self.assertEqual(len(res["sources"]), 0)
        self.assertIn("Cette information n'est pas présente dans la notice officielle", res["reponse"])


if __name__ == "__main__":
    unittest.main()
