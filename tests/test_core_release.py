"""CPU-only release checks; these do not exercise GPU model training."""
import ast
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
from data import compressed_video_loader as loader


def isolated_functions(path, names):
    """Execute only pure parser functions, without importing the GPU stack."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"re": re, "List": list}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class CoreReleaseTests(unittest.TestCase):
    def test_all_python_syntax(self):
        for path in CODE.rglob("*.py"):
            with self.subTest(path=path.relative_to(CODE)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_config_safety_and_provenance(self):
        for path in (CODE / "train/configs").glob("*.json"):
            cfg = json.loads(path.read_text())
            self.assertFalse(cfg["allow_compressed_proxy_fallback"])
            self.assertEqual(cfg["compressed_backend"], "coviar")
            self.assertIn("Archived", cfg["_comment"])
        stage2 = json.loads((CODE / "train/configs/stage2_qwen_compressed_tristream.json").read_text())
        self.assertEqual(stage2["lora_r"], 64)

    def test_even_sampling_is_ordered_and_unique(self):
        result = loader._sample_evenly(list(range(31)), 7)
        self.assertEqual(result, sorted(set(result)))
        self.assertEqual(len(result), 7)
        self.assertEqual((result[0], result[-1]), (0, 30))
        self.assertEqual(loader._sample_evenly([], 3), [])
        self.assertEqual(loader._sample_evenly([1, 2], 10), [1, 2])

    def test_motion_map_shape_dtype_and_zero(self):
        raw = np.zeros((8, 9, 2), dtype=np.int32)
        encoded = loader.encode_motion_vector_map(raw)
        self.assertEqual(encoded.shape, (8, 9, 3))
        self.assertEqual(encoded.dtype, np.uint8)
        np.testing.assert_array_equal(encoded, 128)

    def test_motion_clipping(self):
        raw = np.empty((8, 9, 2), dtype=np.int32)
        raw[..., 0], raw[..., 1] = 100, -100
        encoded = loader.encode_motion_vector_map(raw, clip_value=20)
        np.testing.assert_array_equal(encoded[..., 0], 255)
        np.testing.assert_array_equal(encoded[..., 1], 1)
        self.assertGreaterEqual(int(encoded[..., 2].min()), 254)

    def test_residual_signed_encoding(self):
        raw = np.zeros((8, 9), dtype=np.int16)
        encoded = loader.encode_residual_map(raw)
        self.assertEqual(encoded.shape, (8, 9, 3))
        np.testing.assert_array_equal(encoded, 128)

    def test_unknown_backend_fails(self):
        with self.assertRaises(ValueError):
            loader.validate_compressed_backend("nonexistent")

    def test_empty_answer_not_guessed(self):
        names = {"extract_answer_letter", "_check_mc_answer", "_check_free_form_answer"}
        parsers = isolated_functions(CODE / "eval/eval_himmel.py", names)
        self.assertEqual(parsers["extract_answer_letter"]("", 4), "")
        self.assertEqual(parsers["extract_answer_letter"]("B", 4), "B")
        self.assertFalse(parsers["_check_mc_answer"]("", "A", ["x", "y"]))
        self.assertFalse(parsers["_check_free_form_answer"]("12", "2"))
        self.assertTrue(parsers["_check_free_form_answer"]("2.0", "2"))


if __name__ == "__main__":
    unittest.main()
