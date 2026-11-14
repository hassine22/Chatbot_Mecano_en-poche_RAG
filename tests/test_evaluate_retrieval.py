import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from evaluation.evaluate_retrieval import load_questions, score_results


class RetrievalEvaluationTests(unittest.TestCase):
    def test_computes_top_k_metrics(self):
        results = [
            {"id": "passage-1"},
            {"id": "passage-2"},
            {"id": "passage-3"},
        ]

        metrics = score_results(results, {"passage-2"}, top_k=3)

        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["hit_rate"], 1.0)

    def test_handles_no_relevant_result(self):
        metrics = score_results([{"id": "other"}], {"expected"}, top_k=3)

        self.assertEqual(metrics, {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
        })

    def test_loads_annotated_questions(self):
        questions = load_questions(ROOT / "evaluation" / "retrieval_questions.jsonl")

        self.assertEqual(len(questions), 10)
        self.assertTrue(all(question["relevant_ids"] for question in questions))

    def test_rejects_empty_relevant_ids(self):
        question = {
            "id": "q",
            "question": "Question?",
            "modele": "Clio 4",
            "categorie": "voyant",
            "relevant_ids": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(json.dumps(question) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relevant_ids"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()
