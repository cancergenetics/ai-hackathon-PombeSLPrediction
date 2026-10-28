"""Check 2: label-permutation control of the full M2r + Huber-stacker pipeline.

For each outer fold given: the labels of the outer TRAINING pairs (s_score, y, s_raw_mean, y_raw jointly,
row-wise plain shuffle, seed PSEED + k) are permuted in memory (data.fold_split patched), so the Huber target
-min(S2, 0) stays consistent with y. Then the unmodified production path runs with NO feature cache:
drop_noncoding -> inner_splits -> nested RYAN/NBR/GP features -> run_m2plus.FoldDataP -> fit_stacker_huber
(seeds = config ladder 0,1,2; inner fold s mod 5 early stopping) -> run_m2.predict, seed-averaged.
Eval labels are never touched; metric = my own fold pooled weighted AP (check3_metric.Fold).
Expected: primary ~ weighted prevalence (~0.021). Prints aggregates only."""
import common
import json
import sys
import time

import numpy as np
import pandas as pd

from src.timing import Timings, set_threads
set_threads(common.THREADS)
import torch  # noqa: E402
torch.set_num_threads(common.THREADS)

import run_m2plus as RP  # noqa: E402
from src import data  # noqa: E402
from src import pairs as PA  # noqa: E402
from src.models import lgbm, m2  # noqa: E402
from src.ryanfeat import SimStore  # noqa: E402
import check3_metric as C3  # noqa: E402

RP.THREADS = common.THREADS
RP.R.N_JOBS = common.THREADS
FOLDS = [int(x) for x in sys.argv[1].split(",")]
PSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
SEEDS = [int(s) for s in RP.CFG["seeds"]["ladder"]]
ORIG_FS, ORIG_FT = data.fold_split, m2.fold_tables_m2


def perm_fold_split(split_seed=None, fold=0):
    tr, ev = ORIG_FS(split_seed, fold)
    p = np.random.default_rng(PSEED + fold).permutation(len(tr))
    for c in ("s_score", "y", "s_raw_mean", "y_raw"):
        tr[c] = tr[c].to_numpy()[p]
    return tr, ev


data.fold_split = perm_fold_split
m2.fold_tables_m2 = lambda *a, **kw: ORIG_FT(*a, **{**kw, "use_cache": False})
data.cache_save = lambda *a, **kw: None

lab = pd.read_parquet(common.LIVE / "splits" / "labels_dev.parquet", columns=["gene_a", "gene_b", "s_score"])
lab["y"] = (lab.s_score < C3.THR).astype(np.int8)
lab = lab.drop(columns="s_score")
spec = PA.load_spec()
sims = SimStore(names=SimStore().names)
params = lgbm.params_for("default", common.THREADS)
out = {"seeds": SEEDS, "perm_seed": PSEED, "folds": {}}
t0 = time.time()
for k in FOLDS:
    D = RP.FoldDataP(None, k, spec, sims, Timings(), "M2r")
    # sanity: the stacker target is consistent with the (permuted) stacker label
    cons = bool(np.array_equal(D.y.astype(bool), D.s < C3.THR))
    cols = D.cols(RP.FAMILIES)
    sc, fits = [], []
    for s in SEEDS:
        b, inf = RP.fit_stacker_huber(D, cols, params, s, s % D.n_inner)
        sc.append(RP.R.predict(b, D, cols, False))
        fits.append({"seed": s, "best_iter": inf["best_iteration"], "inner_wAP": round(inf["valid_ap_weighted"], 4)})
        print(f"fold {k} seed {s} {fits[-1]} {time.time()-t0:.0f}s", flush=True)
    ev = D.ev[["gene_a", "gene_b", "fold", "part"]].assign(score=np.mean(sc, axis=0))
    df = ev.merge(lab, on=["gene_a", "gene_b"], how="left", validate="one_to_one")
    genes = pd.Index(sorted(set(df.gene_a) | set(df.gene_b)))
    F = C3.Fold(df, genes)
    w = np.zeros(len(df))
    for c, v in C3.CW.items():
        m = (df.part == c).to_numpy()
        w[m] = v / m.sum()
    cls = {c: round(C3.ap(df.y.to_numpy()[df.part == c].astype(float), df.score.to_numpy()[df.part == c],
                          np.ones((df.part == c).sum())), 4) for c in ("C1", "C2", "C3")}
    out["folds"][k] = {"primary": round(F.ap(C3.CW), 4), "weighted_prevalence": round(float(np.sum(w * df.y)), 4),
                       "class_AP": cls, "class_prevalence": {c: round(float(df.y[df.part == c].mean()), 4)
                                                             for c in ("C1", "C2", "C3")},
                       "stacker_pos_rate": round(float(D.y.mean()), 4), "target_consistent_with_y": cons,
                       "fits": fits}
    print(json.dumps(out["folds"][k]), flush=True)
    del D
out["mean_primary"] = float(np.mean([v["primary"] for v in out["folds"].values()]))
out["wall_s"] = round(time.time() - t0)
print(json.dumps(out, indent=1))
(common.OUT / f"check2_perm_{'_'.join(map(str, FOLDS))}_s{PSEED}.json").write_text(json.dumps(out, indent=1))
