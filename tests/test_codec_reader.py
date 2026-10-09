"""Reader contract tests; synthetic fixtures only, no FFmpeg needed."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "codec" / "read_signals.py"
SPEC = importlib.util.spec_from_file_location("read_signals", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CodecReaderTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is required for the array reader")
        self.temp = tempfile.TemporaryDirectory(prefix="himmel-reader-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "motion_vectors").mkdir()
        (self.root / "residuals").mkdir()
        (self.root / "summary.csv").write_text(
            "frame,width,height,mv_count,mv_origin\n1,2,2,1,bitstream\n")
        (self.root / "motion_vectors" / "frame_000001.csv").write_text(
            "origin,source,block_w,block_h,src_x,src_y,dst_x,dst_y,motion_x,motion_y,motion_scale,flags\n"
            "bitstream,-1,2,2,1,1,1,1,1,-3,4,0x0\n")
        (self.root / "residuals" / "frame_000001.s16le").write_bytes(struct.pack("<4h", -255, -1, 0, 255))

    def test_fractional_motion_and_signed_little_endian_residual(self):
        result = MODULE.load_frame(self.root, 1)
        self.assertEqual(result["motion"].shape, (2, 2, 2))
        self.assertEqual(result["motion"][0, 0].tolist(), [0.25, -0.75])
        self.assertEqual(result["residual"].tolist(), [[-255, -1], [0, 255]])
        self.assertEqual(result["metadata"]["mv_origin"], "bitstream")

    def test_missing_frame(self):
        with self.assertRaises(ValueError):
            MODULE.load_frame(self.root, 2)

    def test_invalid_frame_index(self):
        with self.assertRaises(ValueError):
            MODULE.load_frame(self.root, 0)

    def test_truncated_residual(self):
        (self.root / "residuals" / "frame_000001.s16le").write_bytes(b"\x00")
        with self.assertRaisesRegex(ValueError, "byte count"):
            MODULE.load_frame(self.root, 1)


if __name__ == "__main__":
    unittest.main()
