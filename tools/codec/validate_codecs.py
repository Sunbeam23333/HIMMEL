#!/usr/bin/env python3
"""Synthetic extraction smoke tests; no datasets, checkpoints, or model scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


CODECS = {
    "h264": ("libx264", "mp4", ["-bf", "0", "-g", "10"]),
    "hevc": ("libx265", "mp4", ["-x265-params", "bframes=0:keyint=10:log-level=error"]),
    "vp9": ("libvpx-vp9", "webm", ["-g", "10", "-deadline", "realtime", "-cpu-used", "8"]),
    "av1": ("libaom-av1", "mkv", ["-g", "10", "-cpu-used", "8"]),
    "mpeg4": ("mpeg4", "mp4", ["-bf", "0", "-g", "10"]),
}


def run(args, *, success=True):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if success != (result.returncode == 0):
        raise RuntimeError(f"Unexpected exit {result.returncode}: {args}\n{result.stderr[-3000:]}")
    return result


def rows(folder):
    with (folder / "summary.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def tree_digest(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", type=Path, default=Path(__file__).with_name("extract_codec_signals"))
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--output-dir", type=Path, help="New directory only; defaults to a fresh temporary directory.")
    args = parser.parse_args()
    extractor = args.extractor.resolve()
    if not extractor.is_file():
        parser.error("Build the extractor first: make -C tools/codec")
    for option in ("--help", "-h"):
        help_result = run([extractor, option])
        assert "Usage:" in help_result.stdout + help_result.stderr
    if args.output_dir:
        work = args.output_dir.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        work = Path(tempfile.mkdtemp(prefix="himmel-codec-validation-"))
    report = {"ffmpeg": run([args.ffmpeg, "-version"]).stdout.splitlines()[0], "codecs": {}}

    def sample(name, encoder, extension, options):
        video = work / f"{name}.{extension}"
        run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "testsrc2=size=160x96:rate=10:duration=1", "-c:v", encoder, *options, video])
        output = work / f"out_{name}"
        run([extractor, video, output, "10", "16", "4"])
        result = rows(output)
        assert len(result) == 10, (name, "frame count")
        assert all(int(row["mv_count"]) > 0 for row in result[1:]), (name, "missing motion")
        assert all(float(row["mean_abs_residual"]) > 0 for row in result[1:]), (name, "empty residual")
        for index in range(1, 11):
            assert (output / "residuals" / f"frame_{index:06d}.s16le").stat().st_size == 160 * 96 * 2
        report["codecs"][name] = {"frames": len(result), "inter_frames_with_motion": 9,
                                 "origins": sorted({row["mv_origin"] for row in result[1:]}),
                                 "raw_residual_bytes_per_frame": 30720}
        return video, output, result

    for name, (encoder, extension, options) in CODECS.items():
        _, _, result = sample(name, encoder, extension, options)
        expected = "bitstream" if name in {"h264", "mpeg4"} else "estimated"
        assert {row["mv_origin"] for row in result[1:]} == {expected}, (name, "decoder support changed")

    _, _, bframes = sample("h264_bframes", "libx264", "mp4", ["-bf", "2", "-g", "10", "-x264-params", "b-adapt=0"])
    assert any(row["fallback_reason"] == "future_reference" for row in bframes)
    assert all(row["mv_origin"] == "estimated" for row in bframes if row["fallback_reason"] == "future_reference")

    repeat = work / "out_av1_repeat"
    run([extractor, work / "av1.mkv", repeat, "10", "16", "4"])
    assert tree_digest(work / "out_av1") == tree_digest(repeat), "Fallback is not deterministic"
    report["deterministic_repeat"] = "av1: byte-identical CSV, PGM, and int16 files"

    video = work / "h264.mp4"
    for name, tail in [("negative_frames", ["-1"]), ("zero_block", ["0", "0"]),
                       ("bad_number", ["abc"]), ("negative_radius", ["0", "16", "-1"]),
                       ("overflow_number", ["99999999999999999999999999"])]:
        run([extractor, video, work / name, *tail], success=False)
        assert not (work / name).exists()
    run([extractor, work / "missing.mp4", work / "missing_output"], success=False)
    before = tree_digest(work / "out_h264")
    run([extractor, video, work / "out_h264"], success=False)
    assert tree_digest(work / "out_h264") == before, "Existing output was modified"
    audio = work / "audio.wav"
    run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=0.1", audio])
    run([extractor, audio, work / "audio_output"], success=False)
    report["negative_tests"] = "PASS: invalid integers, missing input, audio-only input, existing output"
    report["artifacts"] = str(work)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
