"""Check 2: label-permutation control (and a clean reproduction) of the full M2 pipeline.

mode 'perm' : for each outer fold given, the labels (s_score, y, s_raw_mean, y_raw jointly, row-wise PLAIN
              shuffle, seed 99+k) of the outer TRAINING pairs are permuted in memory (data.fold_split
              patched); then the unmodified production path runs: drop_noncoding -> inner_splits -> nested
              RYAN/NBR/GP features -> run_m2.FoldData / fit_stacker / predict (seeds 0,1,2, default config),
              no feature cache. Eval labels are never touched; metric = my own pooled weighted AP (check3).
mode 'repro': same stacker code on the registered run's cached clean features (read-only), to confirm the
              per-fold primary is reproduced from features -> predictions.
Prints aggregate metrics only."""
import common
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd

from src.timing import Timings, set_threads
set_threads(common.THREADS)
import torch  # noqa: E402
torch.set_num_threads(common.THREADS)

from src import data  # noqa: E402
from src import pairs as PA  # noqa: E402
from src.models import lgbm, m2  # noqa: E402
from src.ryanfeat import SimStore  # noqa: E402
import check3_metric as C3  # noqa: E402

spec_ = importlib.util.spec_from_file_location("run_m2", common.REPO / "scripts" / "run_m2.py")
R = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(R)
R.N_JOBS = common.THREADS
mode = sys.argv[1]
FOLDS = [int(x) for x in sys.argv[2].split(",")]
PSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 99
ORIG_FS, ORIG_FT = data.fold_split, m2.fold_tables_m2


def perm_fold_split(split_seed=None, fold=0):
    tr, ev = ORIG_FS(split_seed, fold)
    rng = np.random.default_rng(PSEED + fold)
    p = rng.permutation(len(tr))
    cols = ["s_score", "y", "s_raw_mean", "y_raw"]
    for c in cols:
        tr[c] = tr[c].to_numpy()[p]
    return tr, ev


def nocache_tables(split_seed=None, fold=0, sims=None, cfg=None, gp_cfg=None, use_cache=True, timings=None):
    from src import stacking
    train, ev, drop = m2.fold_pairs(split_seed, fold)
    inner = data.inner_splits(train)
    builder = m2.make_builder(m2.gene_table(), gp_cfg)
    rows = [stacking.inner_rows(f, sims, cfg, builder) for f in inner]
    return pd.concat(rows, ignore_index=True), stacking.outer_rows(train, ev, sims, cfg, builder), inner, drop


def registered_cache_tables(split_seed=None, fold=0, sims=None, cfg=None, gp_cfg=None, use_cache=True, timings=None):
    train, ev, drop = m2.fold_pairs(split_seed, fold)
    inner = data.inner_splits(train)
    d = common.REPO / "data" / "processed" / "cache" / "20261028" / f"fold{fold}"
    nm = "m2feat_3d35f8c314e9"
    rows = [pd.read_parquet(d / f"inner{f.j}" / f"{nm}.parquet") for f in inner]
    return pd.concat(rows, ignore_index=True), pd.read_parquet(d / f"{nm}.parquet"), inner, drop


if mode == "perm":
    data.fold_split = perm_fold_split
    m2.fold_tables_m2 = nocache_tables
else:
    m2.fold_tables_m2 = registered_cache_tables

lab = pd.read_parquet(common.FROZEN / "splits" / "labels_dev.parquet", columns=["gene_a", "gene_b", "s_score"])
lab["y"] = (lab.s_score < C3.THR).astype(np.int8)
lab = lab.drop(columns="s_score")
spec = PA.load_spec()
sims = SimStore(names=SimStore().names)
params = lgbm.params_for("default", common.THREADS)
out = {"mode": mode, "folds": {}}
t0 = time.time()
for k in FOLDS:
    D = R.FoldData(None, k, spec, sims, Timings())
    if mode == "perm":
        out["folds"][k] = {"stacker_pos_rate": float(D.y.mean())}
    cols = D.cols(R.VARIANTS["M2"]["families"])
    sc, fits = [], []
    for s in (0, 1, 2):
        b, inf = R.fit_stacker(D, cols, params, s, False, s % D.n_inner)
        sc.append(R.predict(b, D, cols, False))
        fits.append({"best_iter": inf["best_iteration"], "valid_wAP": round(inf["valid_ap_weighted"], 4)})
        print(f"fold {k} seed {s} {inf['best_iteration']} {time.time()-t0:.0f}s", flush=True)
    ev = D.ev[["gene_a", "gene_b", "fold", "part"]].assign(score=np.mean(sc, axis=0))
    df = ev.merge(lab, on=["gene_a", "gene_b"], how="left")
    pm, _ = C3.primary(df, C3.CW)
    prev = float(np.sum(C3.weights(df.part.to_numpy(), C3.CW) * df.y))
    cls = {c: round(C3.my_ap(df.y.to_numpy()[df.part == c], df.score.to_numpy()[df.part == c],
                             np.ones((df.part == c).sum())), 4) for c in ("C1", "C2", "C3")}
    from sklearn.metrics import roc_auc_score
    neg = C3.primary(df.assign(score=-df.score), C3.CW)[0]
    auc = float(roc_auc_score(df.y, df.score, sample_weight=C3.weights(df.part.to_numpy(), C3.CW)))
    out["folds"][k] = {**out["folds"].get(k, {}), "primary": round(pm, 4), "primary_of_negated_score": round(neg, 4),
                       "weighted_auroc": round(auc, 4), "weighted_prevalence": round(prev, 4),
                       "class_AP": cls, "fits": fits}
    print(json.dumps(out["folds"][k]), flush=True)
    del D
out["wall_s"] = round(time.time() - t0)
print(json.dumps(out, indent=1))
(common.OUT / f"check2_{mode}_{'_'.join(map(str, FOLDS))}_s{PSEED}.json").write_text(json.dumps(out, indent=1))
