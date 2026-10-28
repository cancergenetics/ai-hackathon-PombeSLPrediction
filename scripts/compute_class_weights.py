#!/usr/bin/env python
"""Compute the frozen C1/C2/C3 class weights (DESIGN.md §3) and write them into config/protocol.yaml.

Universe genes U (universe_genes.tsv); R = universe genes that are Ryan S2 genes (>=1 S2 pair).
For every universe pair, class = number of genes in R: 2 -> C1, 1 -> C2, 0 -> C3.
  * unmeasured_universe (PRIMARY): class fractions among universe pairs WITHOUT an S2 value
    (the pairs the deliverable actually has to predict);
  * all_pairs: class fractions among all universe pairs (report-only weighting).
Computed analytically from |U|, |R| and the number of measured (S2) universe pairs; all measured
pairs are C1 by construction (both genes are S2 genes), which is asserted.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import protocol  # noqa: E402


def class_counts(n_universe: int, n_seen: int, n_measured: int) -> dict:
    c2 = lambda n: n * (n - 1) // 2  # noqa: E731
    allp = {"C1": c2(n_seen), "C2": n_seen * (n_universe - n_seen), "C3": c2(n_universe - n_seen)}
    unm = dict(allp, C1=allp["C1"] - n_measured)
    return {"all_pairs": allp, "unmeasured_universe": unm}


def fractions(counts: dict) -> dict:
    tot = sum(counts.values())
    return {k: round(v / tot, 6) for k, v in counts.items()}


def main() -> int:
    root = protocol.data_root()
    u = pd.read_csv(root / "universe_genes.tsv", sep="\t")
    U = set(u.gene[u.in_universe])
    lab = protocol.label_column()
    pairs = pd.read_parquet(root / "ryan_pairs.parquet", columns=["gene_a", "gene_b", lab])
    pairs = pairs[pairs[lab].notna()]
    R = (set(pairs.gene_a) | set(pairs.gene_b)) & U
    inU = pairs[pairs.gene_a.isin(U) & pairs.gene_b.isin(U)]
    assert inU.gene_a.isin(R).all() and inU.gene_b.isin(R).all()
    counts = class_counts(len(U), len(R), len(inU))
    payload = {"class_weights": {
        "primary": "unmeasured_universe",
        "unmeasured_universe": fractions(counts["unmeasured_universe"]),
        "all_pairs": fractions(counts["all_pairs"]),
        "derivation": {"n_universe_genes": len(U), "n_universe_ryan_s2_genes": len(R),
                       "n_measured_universe_pairs": int(len(inU)),
                       "pair_counts": {k: {c: int(n) for c, n in v.items()} for k, v in counts.items()},
                       "script": "scripts/compute_class_weights.py"}}}
    protocol.write_derived_block("class_weights", payload, "scripts/compute_class_weights.py")
    cw = payload["class_weights"]
    print(f"|U|={len(U)} |R|={len(R)} measured universe pairs={len(inU):,}")
    print("unmeasured-universe weights (primary):", cw["unmeasured_universe"])
    print("all-pairs weights:", cw["all_pairs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
