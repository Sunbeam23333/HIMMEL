#!/usr/bin/env python3
"""Exact counts and paired cases from already-scored, complete predictions.

This is an aggregation utility, NOT a model runner or an answer-extraction
implementation. Use the benchmark's official scoring protocol first.
Each JSONL line must contain a nonempty string `id` and a boolean `correct`.
All conditions must cover exactly the same unique question IDs.
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path


DENOMINATORS = {"mathvista_testmini": 1000, "gsm8k_test": 1319,
                "video_mme_full": 2700}


def load_scored(path, expected):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"line {line_number}: id must be a nonempty string")
            if item_id in rows:
                raise ValueError(f"line {line_number}: duplicate id {item_id!r}")
            if type(row.get("correct")) is not bool:
                raise ValueError(f"line {line_number}: correct must be true or false")
            rows[item_id] = row["correct"]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} unique questions, found {len(rows)}")
    return rows


def summarize(rows):
    total = len(rows)
    if not total:
        raise ValueError("empty prediction set")
    correct = sum(rows.values())
    return {"correct": correct, "total": total,
            "accuracy_fraction": f"{correct}/{total}",
            "accuracy_percent": format(Decimal(correct) * 100 / total, ".6f")}


def compare(base, other):
    if base.keys() != other.keys():
        raise ValueError("conditions must contain exactly the same question IDs")
    categories = {"both_correct": [], "lost": [], "recovered": [],
                  "both_incorrect": []}
    for item_id in sorted(base):
        if base[item_id]:
            key = "both_correct" if other[item_id] else "lost"
        else:
            key = "recovered" if other[item_id] else "both_incorrect"
        categories[key].append(item_id)
    return {"counts": {key: len(ids) for key, ids in categories.items()},
            "question_ids": categories,
            "net_correct_change": len(categories["recovered"]) - len(categories["lost"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(DENOMINATORS), required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=JSONL",
                        help="repeat for baseline, video LoRA and math rehearsal; first is reference")
    args = parser.parse_args()
    runs = {}
    try:
        for item in args.run:
            label, separator, path = item.partition("=")
            if not separator or not label or not path or label in runs:
                raise ValueError("each --run must have a unique LABEL=JSONL")
            runs[label] = load_scored(path, DENOMINATORS[args.benchmark])
        reference = next(iter(runs))
        report = {"benchmark": args.benchmark, "reference": reference,
                  "metrics": {label: summarize(rows) for label, rows in runs.items()},
                  "paired_vs_reference": {
                      label: compare(runs[reference], rows)
                      for label, rows in runs.items() if label != reference}}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
