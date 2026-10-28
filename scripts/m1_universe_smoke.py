#!/usr/bin/env python
"""Universe-scoring smoke test for M1 (DESIGN.md §7): score every universe pair (~7.1M) with one saved fold
model, chunked (1M pairs), and record runtime + peak RSS. Scores are NOT saved (only summary statistics).

Usage: m1_universe_smoke.py --models-dir <data root>/models/m1/<name> [--model fold0_seed0.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pairs as PA  # noqa: E402
from src import protocol  # noqa: E402
from src.models.lgbm import safe_names  # noqa: E402
from src.timing import Timings, hardware, peak_rss_mb, set_threads  # noqa: E402

N_JOBS = 8


def universe_indices(spec: PA.FeatureSpec) -> np.ndarray:
    f = protocol.data_root() / "universe_genes.tsv"
    if f.exists():
        u = pd.read_csv(f, sep="\t")
        genes = u.gene[u.in_universe].tolist()
    else:                                         # toy data: all feature-store genes
        genes = spec.genes
    return np.sort(PA.gene_indices(spec, genes))


def pair_chunks(U: np.ndarray, chunk: int):
    """Yield (ia, ib) index arrays of all i<j pairs of U, about `chunk` pairs at a time."""
    A, B, n = [], [], 0
    for r in range(len(U) - 1):
        m = len(U) - r - 1
        A.append(np.full(m, U[r]))
        B.append(U[r + 1:])
        n += m
        if n >= chunk:
            yield np.concatenate(A), np.concatenate(B)
            A, B, n = [], [], 0
    if n:
        yield np.concatenate(A), np.concatenate(B)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--model", default="fold0_seed0.txt")
    ap.add_argument("--chunk", type=int, default=PA.CHUNK)
    a = ap.parse_args()
    set_threads(N_JOBS)
    t0 = time.perf_counter()
    b = lgb.Booster(model_file=str(Path(a.models_dir) / a.model))
    spec = PA.load_spec()
    names = spec.feature_names()
    pos = {n: i for i, n in enumerate(safe_names(names))}
    missing = [n for n in b.feature_name() if n not in pos]
    assert not missing, f"model features missing from the current store: {missing[:5]}"
    cols = np.array([pos[n] for n in b.feature_name()])
    groups = sorted({spec.feature_groups()[names[i]] for i in cols})
    sub = {n: i for i, n in enumerate(safe_names(spec.feature_names(groups)))}
    idx = np.array([sub[n] for n in b.feature_name()])
    reorder = not np.array_equal(idx, np.arange(len(sub)))
    U = universe_indices(spec)
    T = Timings()
    n_pairs, s1, s2, top = 0, 0.0, 0.0, []
    for ia, ib in pair_chunks(U, a.chunk):
        with T.stage("assemble"):
            X = PA.assemble_arrays(ia, ib, spec, groups=groups)
            if reorder:
                X = X[:, idx]
        with T.stage("predict"):
            s = b.predict(X, num_threads=N_JOBS)
        n_pairs += len(s)
        s1 += float(s.sum())
        s2 += float((s ** 2).sum())
        del X, s
    rt = T.as_dict()
    agg = {st: round(sum(x["wall_s"] for x in rt["stages"] if x["stage"] == st), 2) for st in ("assemble", "predict")}
    out = {"models_dir": str(a.models_dir), "model": a.model, "n_universe_genes": int(len(U)), "n_pairs": n_pairs,
           "chunk": a.chunk, "n_chunks": len(rt["stages"]) // 2, "n_features": int(len(cols)),
           "wall_s_total": round(time.perf_counter() - t0, 2), "wall_s_by_stage": agg,
           "cpu_s": rt["cpu_s"], "peak_rss_mb": round(peak_rss_mb(), 1), "threads": N_JOBS, "hardware": hardware(),
           "score_mean": s1 / n_pairs, "score_sd": float(np.sqrt(max(s2 / n_pairs - (s1 / n_pairs) ** 2, 0))),
           "note": "scores not saved (smoke test)"}
    d = protocol.results_root() / "m1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "universe_smoke.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
