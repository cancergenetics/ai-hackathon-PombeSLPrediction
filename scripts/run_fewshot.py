#!/usr/bin/env python
"""E-FEWSHOT (config/fewshot.yaml, docs/rarely_seen.md): reveal k measured pairs per held-out gene into the
outer training set, re-run the incumbent pipeline (scripts/run_m2plus.py, M2r variant, Huber stacker), score the
remaining C2/C3 pairs. Report-only. No src/ or other-script edits: the reveal is a monkeypatch of
src.data.fold_split (training rows added, revealed eval rows removed) and src.data.cache_path (separate cache).

  python scripts/run_fewshot.py --folds 0,1 --ks 5,20          # run (resumable; per-fold/k prediction files)
  python scripts/run_fewshot.py --selftest                      # reveal-set sanity checks only (no training)
Per-seed predictions are saved under <data root>/preds/fewshot/ (git-ignored, pair-level); only runtime and
aggregates go to results/rarely_seen/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data, protocol  # noqa: E402
from src.timing import Timings, peak_rss_mb, set_threads  # noqa: E402

FCFG = yaml.safe_load((ROOT / "config" / "fewshot.yaml").read_text())
FS = FCFG["fewshot"]
KMAX = max(FS["ks"])
_ORIG_SPLIT = data.fold_split
_ORIG_CACHE = data.cache_path
_STATE = {"k": 0}


def preds_dir() -> Path:
    d = protocol.preds_dir() / "fewshot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir() -> Path:
    d = ROOT / "results" / "rarely_seen"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================================================ reveal sets
_RANK_CACHE: dict = {}


def reveal_ranks(split_seed, fold) -> tuple[pd.DataFrame, pd.Series]:
    """(eval C2/C3 pairs of the fold, rank) where rank = the smallest position of the pair in the reveal-order of
    either of its held-out genes (0-based). Pair is revealed at k iff rank < k."""
    key = (data._seed(split_seed), fold)
    if key in _RANK_CACHE:
        return _RANK_CACHE[key]
    _, ev = _ORIG_SPLIT(split_seed, fold)
    ev = ev[ev.part.isin(["C2", "C3"])].reset_index(drop=True)
    held = set(data.fold_genes(split_seed, fold))
    long = pd.concat([ev[["gene_a"]].rename(columns={"gene_a": "gene"}).assign(row=np.arange(len(ev))),
                      ev[["gene_b"]].rename(columns={"gene_b": "gene"}).assign(row=np.arange(len(ev)))])
    long = long[long.gene.isin(held)].sort_values(["gene", "row"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng([int(FS["reveal_seed"]), int(fold)])
    long["u"] = rng.random(len(long))
    long["pos"] = long.groupby("gene").u.rank(method="first").astype(int) - 1
    rank = long.groupby("row").pos.min().reindex(np.arange(len(ev)))
    assert rank.notna().all(), "every C2/C3 eval pair has a held-out gene"
    _RANK_CACHE[key] = (ev, rank.astype(int).reset_index(drop=True))
    return _RANK_CACHE[key]


def reveal_pairs(split_seed, fold, k) -> pd.DataFrame:
    ev, rank = reveal_ranks(split_seed, fold)
    return ev.loc[rank.to_numpy() < k, ["gene_a", "gene_b"]]


def eval_set(split_seed, fold) -> pd.DataFrame:
    """E_f: C2+C3 eval pairs of the fold minus reveal(KMAX); identical for every k."""
    ev, rank = reveal_ranks(split_seed, fold)
    return ev[rank.to_numpy() >= KMAX].reset_index(drop=True)


def patched_fold_split(split_seed=None, fold: int = 0):
    tr, ev = _ORIG_SPLIT(split_seed, fold)
    k = _STATE["k"]
    if k == 0:
        return tr, ev
    rev = reveal_pairs(split_seed, fold, k)
    lab = data.dev_pairs()  # training-eligible dev pairs (train_drop excluded) via the sanctioned helper
    add = rev.merge(lab, on=["gene_a", "gene_b"], how="left")
    assert len(add) == len(rev) and add.y.notna().all()  # also proves no revealed pair is train_drop
    tr2 = pd.concat([tr, add], ignore_index=True)
    return tr2, eval_set(split_seed, fold)


def patched_cache_path(name, split_seed=None, fold=0, inner=None):
    p = _ORIG_CACHE(name, split_seed, fold, inner)
    cr = protocol.cache_root()
    return cr / "fewshot" / f"k{_STATE['k']}_r{FS['reveal_seed']}" / p.relative_to(cr)


def install():
    data.fold_split = patched_fold_split
    data.cache_path = patched_cache_path


def selftest():
    for fold in (0, 1):
        tr0, ev0 = _ORIG_SPLIT(None, fold)
        E = eval_set(None, fold)
        r5, r20 = reveal_pairs(None, fold, 5), reveal_pairs(None, fold, 20)
        k5 = set(zip(r5.gene_a, r5.gene_b))
        k20 = set(zip(r20.gene_a, r20.gene_b))
        e = set(zip(E.gene_a, E.gene_b))
        held = set(data.fold_genes(None, fold))
        assert k5 <= k20 and not (k20 & e)
        assert len(k20) + len(e) == (ev0.part.isin(["C2", "C3"])).sum()
        cnt5 = pd.concat([r5.gene_a, r5.gene_b]).value_counts()
        cnt5 = cnt5[cnt5.index.isin(held)]
        print(f"fold {fold}: reveal5 {len(r5)} reveal20 {len(r20)} eval {len(E)} (was {ev0.part.isin(['C2','C3']).sum()}); "
              f"held genes {len(held)}; per-gene revealed@5 min {cnt5.min()} median {cnt5.median()}")
        _STATE["k"] = 5
        install()
        tr, ev = data.fold_split(None, fold)
        assert len(tr) == len(tr0) + len(r5) and len(ev) == len(E)
        assert set(zip(ev.gene_a, ev.gene_b)) == e
        assert not ({(a, b) for a, b in zip(ev.gene_a, ev.gene_b)} & {(a, b) for a, b in zip(tr.gene_a, tr.gene_b)})
        data.fold_split, data.cache_path = _ORIG_SPLIT, _ORIG_CACHE
        _STATE["k"] = 0
    print("selftest ok")


# ============================================================================================ run
def run_fold_k(M, R, split_seed, fold, k, spec, sims) -> dict:
    from src.models import lgbm
    install()
    _STATE["k"] = k
    seeds = [int(s) for s in FS["seeds"]]
    T = Timings()
    t0 = time.perf_counter()
    tfe = {}
    D = M.FoldDataP(split_seed, fold, spec, sims, T, FCFG["incumbent"]["variant"])
    cols = D.cols(M.FAMILIES)
    params = lgbm.params_for("default", M.THREADS)
    preds, fits = [], []
    for s in seeds:
        j = s % D.n_inner
        with T.stage(f"train_f{fold}_s{s}"):
            b, inf = M.fit_stacker_huber(D, cols, params, s, j)
        sc = R.predict(b, D, cols, False)
        fits.append({"seed": s, **inf})
        preds.append(D.ev[["gene_a", "gene_b", "fold", "part"]].assign(score=sc, seed=s))
        print(f"  fold {fold} k {k} seed {s}: best_iter {inf['best_iteration']} inner wAP {inf['valid_ap_weighted']:.4f}",
              flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(preds_dir() / f"k{k}_fold{fold}.parquet", index=False)
    st = T.as_dict()["stages"]
    rec = {"fold": fold, "k": k, "n_train_stacker_rows": int(len(D.Xtr)), "n_eval_rows": int(len(D.ev)),
           "wall_s": round(time.perf_counter() - t0, 1),
           "feature_load_build_s": round(sum(s["wall_s"] for s in st if not s["stage"].startswith("train")), 1),
           "train_s": round(sum(s["wall_s"] for s in st if s["stage"].startswith("train")), 1),
           "best_iters": [f["best_iteration"] for f in fits], "peak_rss_mb": peak_rss_mb(), "threads": M.THREADS}
    del D
    _STATE["k"] = 0
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=",".join(map(str, FS["folds_order"])))
    ap.add_argument("--ks", default=",".join(map(str, FS["ks"])))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    thr = int(FS["threads"])
    set_threads(thr)
    import torch
    torch.set_num_threads(thr)
    import run_m2plus as M
    import run_m2 as R
    M.THREADS = thr
    R.N_JOBS = thr
    from src import pairs as PA
    from src.ryanfeat import SimStore
    spec = PA.load_spec()
    sims = SimStore(names=SimStore().names)
    log_p = out_dir() / "fewshot_runtime.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {"runs": []}
    for fold in [int(f) for f in args.folds.split(",")]:
        for k in [int(x) for x in args.ks.split(",")]:
            if (preds_dir() / f"k{k}_fold{fold}.parquet").exists():
                print(f"skip fold {fold} k {k} (exists)", flush=True)
                continue
            print(f"== fold {fold} k {k}", flush=True)
            rec = run_fold_k(M, R, None, fold, k, spec, sims)
            print(rec, flush=True)
            log["runs"].append(rec)
            log_p.write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
