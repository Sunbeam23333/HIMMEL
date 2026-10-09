"""Derive an explicit experiment config from a supplied recipe (no training run)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE")
    args = parser.parse_args()
    config = json.loads(args.base.read_text(encoding="utf-8"))
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator or key not in config:
            parser.error(f"Unknown key or invalid override: {key}")
        config[key] = json.loads(value)
    config["_comment"] = "New derived experiment recipe; not an archived checkpoint configuration."
    # Exclusive creation protects an existing recipe from accidental overwrite.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
