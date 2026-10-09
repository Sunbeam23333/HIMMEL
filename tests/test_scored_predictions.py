import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / "tools/analyze_scored_predictions.py"
spec = importlib.util.spec_from_file_location("scored", path)
scored = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scored)


class CountTests(unittest.TestCase):
    def test_exact_counts(self):
        rows = {str(i): i < 1136 for i in range(1319)}
        self.assertEqual(scored.summarize(rows), {
            "correct": 1136, "total": 1319, "accuracy_fraction": "1136/1319",
            "accuracy_percent": "86.125853"})

    def test_paired(self):
        report = scored.compare({"a": True, "b": False, "c": True},
                                {"c": False, "a": True, "b": True})
        self.assertEqual(report["question_ids"]["lost"], ["c"])
        self.assertEqual(report["question_ids"]["recovered"], ["b"])
        self.assertEqual(report["net_correct_change"], 0)

    def test_mismatched_ids(self):
        with self.assertRaises(ValueError):
            scored.compare({"a": True}, {"b": True})

    def load_rows(self, rows, expected):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "scored.jsonl"
            fixture.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            return scored.load_scored(fixture, expected)

    def test_reject_bad_data(self):
        for rows, expected in [
            ([{"id": "a", "correct": 1}], 1),
            ([{"id": "a", "correct": True}] * 2, 2),
            ([{"id": "a", "correct": True}], 1000),
            ([{"id": "", "correct": False}], 1),
            ([[]], 1),
        ]:
            with self.subTest(rows=rows, expected=expected), self.assertRaises(ValueError):
                self.load_rows(rows, expected)

    def test_valid_rows(self):
        self.assertEqual(self.load_rows([{"id": "a", "correct": False}], 1), {"a": False})


if __name__ == "__main__":
    unittest.main()
