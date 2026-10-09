#!/usr/bin/env python3
"""Read audited C-extractor outputs. Requires NumPy for the array conversion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_frame(directory, frame_number):
    """Return metadata, raw block records, float32 H×W×2 MVs, int16 H×W residual.

    MV components are source-minus-destination pixel offsets (motion / scale).
    Overlapping exported blocks are rasterized in CSV order, last block wins;
    retain `motion_records` when block structure matters. These offsets and
    this diagnostic residual are not a reconstruction of codec internals.
    """
    import numpy as np

    directory = Path(directory)
    if not isinstance(frame_number, int) or frame_number < 1:
        raise ValueError("frame_number must be a positive, one-based integer")
    with (directory / "summary.csv").open(newline="") as handle:
        row = next((r for r in csv.DictReader(handle) if int(r["frame"]) == frame_number), None)
    if row is None:
        raise ValueError(f"Frame {frame_number} is not in summary.csv")
    width, height = int(row["width"]), int(row["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid dimensions")
    raw_path = directory / "residuals" / f"frame_{frame_number:06d}.s16le"
    if raw_path.stat().st_size != width * height * 2:
        raise ValueError("Residual byte count does not match summary dimensions")
    residual = np.fromfile(raw_path, dtype="<i2").reshape(height, width)
    with (directory / "motion_vectors" / f"frame_{frame_number:06d}.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != int(row["mv_count"]):
        raise ValueError("Motion record count does not match summary")
    motion = np.zeros((height, width, 2), dtype=np.float32)
    for record in records:
        block_w, block_h = int(record["block_w"]), int(record["block_h"])
        left = int(record["dst_x"]) - block_w // 2
        top = int(record["dst_y"]) - block_h // 2
        right, bottom = min(width, left + block_w), min(height, top + block_h)
        scale = int(record["motion_scale"])
        if scale <= 0 or block_w <= 0 or block_h <= 0:
            raise ValueError("Invalid motion block")
        if record["origin"] != row["mv_origin"]:
            raise ValueError("Motion provenance does not match summary")
        if right <= max(0, left) or bottom <= max(0, top):
            continue
        motion[max(0, top):bottom, max(0, left):right] = (
            int(record["motion_x"]) / scale, int(record["motion_y"]) / scale)
    return {"metadata": row, "motion_records": records, "motion": motion, "residual": residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--frame", type=int, default=2)
    args = parser.parse_args()
    frame = load_frame(args.directory, args.frame)
    print(json.dumps({"metadata": frame["metadata"], "motion_shape": frame["motion"].shape,
                      "motion_dtype": str(frame["motion"].dtype),
                      "residual_shape": frame["residual"].shape,
                      "residual_dtype": str(frame["residual"].dtype)}, indent=2))


if __name__ == "__main__":
    main()
