#!/usr/bin/env python
"""B0 random baseline (DESIGN.md §5), run end-to-end through src/evaluate.py as a smoke test.

Scores are iid Uniform(0,1) per (seed, fold, pair); evaluate() averages over seeds. Expected primary
metric ~= the weighted prevalence (fold mean of sum_c w_c * prevalence_c).
Usage: run_b0.py [--seeds ladder|all] [--split-seed S] [--compare-to RUN_ID] [--no-register]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, protocol  # noqa: E402
from src.evaluate import compare, evaluate  # noqa: E402
from src.timing import Timings, set_threads  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="ladder", choices=["ladder", "all"])
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--compare-to", default=None)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    set_threads(8)
    P = protocol.load()
    seeds = P["seeds"][a.seeds]
    T = Timings()
    rows = []
    with T.stage("predict"):
        for k in range(P["dev_cv"]["n_folds"]):
            _, ev = data.fold_split(a.split_seed, k)
            for s in seeds:
                rng = np.random.default_rng([s, k])
                rows.append(ev[["gene_a", "gene_b", "fold", "part"]].assign(score=rng.random(len(ev)), seed=s))
    preds = pd.concat(rows, ignore_index=True)
    # runtime = prediction stages; evaluate() records its own eval_wall_s
    res = evaluate(preds, model="B0_random", split_seed=a.split_seed, runtime=T.as_dict(),
                   notes=f"random baseline smoke test, seeds={a.seeds}", register=not a.no_register)
    m = res["metrics_mean"]
    b = res["bootstrap"]["primary"]
    print(f"run_id {res['run_id']}  split_seed {res['split_seed']}  seeds {res['seeds']}")
    print(f"primary (pooled universe-weighted AUPRC, fold mean) = {res['primary']:.5f} "
          f"(bootstrap SE {b['se']:.5f}, 95% CI {b['ci'][0]:.5f}–{b['ci'][1]:.5f})")
    print(f"weighted prevalence = {m['prevalence_primary']:.5f}; lift = {m['lift']:.3f}; AUROC = {m['auroc']:.4f}")
    print("per-class AP:", {c: round(m[f'ap_{c}'], 5) for c in ('C1', 'C2', 'C3')},
          "per-class prevalence:", {c: round(m[f'pos_{c}'] / m[f'n_{c}'], 5) for c in ('C1', 'C2', 'C3')})
    print(f"runtime: {T.as_dict()['wall_s']} s wall, eval {res['eval_wall_s']} s")
    if a.compare_to:
        c = compare(res["run_id"], a.compare_to, register=not a.no_register)
        print(f"compare vs {a.compare_to}: delta {c['delta']:+.5f} CI [{c['ci_lo']:+.5f}, {c['ci_hi']:+.5f}] "
              f"adopt={c['adopt']} ({c['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
