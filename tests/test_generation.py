"""
Tests unitaires pour le module de génération (Membre 2) :
- RRF (Reciprocal Rank Fusion)
- Prompts et formatage du contexte
- Détection des questions ambiguës / trop vagues (clarification)
- Pipeline RAG et refus sur information absente
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mecano.pipeline import reciprocal_rank_fusion, ask
from mecano.prompts import (
    build_rag_prompt,
    build_multiquery_prompt,
    is_question_ambiguous,
)


class TestGeneration(unittest.TestCase):

    def test_rrf_ranking_and_deduplication(self):
        """Vérifie que RRF fusionne correctement les rangs et dédouble les passages."""
        list1 = [
            {"id": "doc1", "titre": "Titre 1", "pages": ["1.10"]},
            {"id": "doc2", "titre": "Titre 2", "pages": ["1.20"]},
        ]
        list2 = [
            {"id": "doc2", "titre": "Titre 2", "pages": ["1.20"]},
            {"id": "doc3", "titre": "Titre 3", "pages": ["1.30"]},
        ]
        
        fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["id"], "doc2")
        self.assertIn("score", fused[0])
        expected_score = round(1.0 / 62 + 1.0 / 61, 4)
        self.assertAlmostEqual(fused[0]["score"], expected_score, places=3)

    def test_build_rag_prompt(self):
        """Vérifie que le prompt RAG emballe bien le contexte et les métadonnées."""
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
        prompt = build_rag_prompt("Que signifie ce témoin ?", passages)
        self.assertIn("<CONTEXTE>", prompt)
        self.assertIn("clio4-001", prompt)
        self.assertIn("Page(s): 1.56", prompt)
        self.assertIn("Question du conducteur : Que signifie ce témoin ?", prompt)

    def test_build_multiquery_prompt(self):
        """Vérifie la construction du prompt de reformulation."""
        p = build_multiquery_prompt("voyant huile clignote")
        self.assertIn("voyant huile clignote", p)

    def test_ambiguous_question_detection_and_clarification(self):
        """Vérifie que les questions trop vagues demandent une clarification au lieu de répondre au hasard."""
        self.assertTrue(is_question_ambiguous("J'ai un voyant rouge allumé que faire ?"))
        self.assertTrue(is_question_ambiguous("Ma voiture a un problème sur le tableau de bord"))
        self.assertTrue(is_question_ambiguous("My warning light is on what should I do?"))
        self.assertFalse(is_question_ambiguous("Que faire si le voyant de pression d'huile s'allume ?"))
        self.assertFalse(is_question_ambiguous("Erreur P0420 sur ma voiture"))

        res_fr = ask("J'ai un voyant rouge allumé sur mon tableau de bord, que faire ?", modele="clio4")
        self.assertTrue(res_fr["needs_clarification"])
        self.assertIn("Précision nécessaire", res_fr["answer"])

        res_en = ask("A red warning light is blinking on my dashboard, what should I do?", modele="i10")
        self.assertTrue(res_en["needs_clarification"])
        self.assertIn("Clarification needed", res_en["answer"])

    def test_refusal_when_no_passages(self):
        """Vérifie que le pipeline refuse poliment si aucun passage n'est trouvé."""
        res = ask(
            question="Question introuvable xyz123456",
            modele="modele_inexistant_999",
            use_multi_query=False
        )
        self.assertEqual(len(res["sources"]), 0)
        self.assertFalse(res["needs_clarification"])
        self.assertIn("Cette information n'est pas présente dans la notice officielle", res["answer"])


if __name__ == "__main__":
    unittest.main()
