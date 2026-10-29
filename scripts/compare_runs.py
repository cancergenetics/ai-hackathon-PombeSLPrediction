#!/usr/bin/env python
"""Paired gene-bootstrap comparison of two registered runs (src.evaluate.compare), optionally with the
alternative-split pair, appended to results/comparisons.tsv.

    python scripts/compare_runs.py RUN_A RUN_B [--alt RUN_A_ALT RUN_B_ALT] [--no-register]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--alt", nargs=2, metavar=("RUN_A_ALT", "RUN_B_ALT"))
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args(argv)
    res = evaluate.compare(a.run_a, a.run_b, alt=tuple(a.alt) if a.alt else None, register=not a.no_register)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
