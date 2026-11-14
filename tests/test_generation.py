"""
Tests unitaires pour le module de génération (Membre 2) :
- RRF (Reciprocal Rank Fusion)
- Prompts et formatage du contexte
- Pipeline RAG et refus sur information absente
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mecano.pipeline import reciprocal_rank_fusion, ask
from mecano.prompts import build_rag_prompt, build_multiquery_prompt


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
        
        # doc2 est présent dans les deux listes (rang 2 puis rang 1), son score RRF doit être le plus élevé
        fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["id"], "doc2")
        self.assertIn("score", fused[0])
        # Score doc2 : 1/(60+2) + 1/(60+1)
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

    def test_refusal_when_no_passages(self):
        """Vérifie que le pipeline refuse poliment si aucun passage n'est trouvé."""
        res = ask(
            question="Question introuvable xyz123456",
            modele="modele_inexistant_999",
            use_multi_query=False
        )
        self.assertEqual(len(res["sources"]), 0)
        self.assertIn("Cette information n'est pas présente dans la notice officielle", res["answer"])


if __name__ == "__main__":
    unittest.main()
