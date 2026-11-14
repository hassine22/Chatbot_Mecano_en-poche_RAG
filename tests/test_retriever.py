import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mecano.retriever import VectorRetriever


class FakeEncoder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts, _batch_size=32):
        return np.repeat(self.vector[None, :], len(texts), axis=0)


class FakeIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = len(self.vectors)
        self.d = self.vectors.shape[1]

    def search(self, query, k):
        scores = self.vectors @ query[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return scores[order][None, :], order.astype(np.int64)[None, :]


def make_retriever():
    records = [
        {"id": "clio4-001", "modele": "Clio 4", "categorie": "voyant"},
        {"id": "ecosport-001", "modele": "EcoSport", "categorie": "voyant"},
        {"id": "clio4-002", "modele": "Clio 4", "categorie": "entretien"},
        {"id": "clio4-003", "modele": "Clio 4", "categorie": "voyant"},
    ]
    vectors = [
        [1.0, 0.0],
        [0.8, 0.6],
        [0.0, 1.0],
        [0.7, 0.7],
    ]
    encoder = FakeEncoder([1.0, 0.0])
    return VectorRetriever(FakeIndex(vectors), encoder, records)


class VectorRetrieverTests(unittest.TestCase):
    def test_returns_nearest_passages_with_scores(self):
        results = make_retriever().search("question", top_k=3)

        self.assertEqual(
            [result["id"] for result in results],
            ["clio4-001", "ecosport-001", "clio4-003"],
        )
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_supports_both_requested_top_k_values(self):
        retriever = make_retriever()

        self.assertEqual(len(retriever.search("question", top_k=3)), 3)
        self.assertEqual(len(retriever.search("question", top_k=5)), 4)

    def test_applies_vehicle_and_category_filters_before_top_k(self):
        results = make_retriever().search(
            "question",
            top_k=3,
            modele="clio 4",
            categorie="VOYANT",
        )

        self.assertEqual(
            [result["id"] for result in results],
            ["clio4-001", "clio4-003"],
        )
        self.assertTrue(all(result["modele"] == "Clio 4" for result in results))
        self.assertTrue(all(result["categorie"] == "voyant" for result in results))

    def test_returns_empty_list_when_filters_match_nothing(self):
        self.assertEqual(
            make_retriever().search("question", modele="Unknown car"),
            [],
        )

    def test_rejects_empty_question_and_invalid_top_k(self):
        retriever = make_retriever()
        with self.assertRaises(ValueError):
            retriever.search("  ")
        for top_k in (0, -1, True):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                retriever.search("question", top_k=top_k)

    def test_rejects_index_corpus_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "index contient"):
            VectorRetriever(FakeIndex([[1.0, 0.0]]), FakeEncoder([1.0, 0.0]), [{}, {}])

    def test_rejects_embedding_dimension_mismatch(self):
        retriever = VectorRetriever(
            FakeIndex([[1.0, 0.0]]),
            FakeEncoder([1.0, 0.0, 0.0]),
            [{"id": "clio4-001"}],
        )
        with self.assertRaisesRegex(ValueError, "Dimension de la question"):
            retriever.search("question")


if __name__ == "__main__":
    unittest.main()
