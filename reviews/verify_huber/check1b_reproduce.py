"""Check 1b: is the registered M2r_bag5_huber fold-K output reproducible, and where does a rebuilt-feature run
differ from it?
  (a) FoldDataP on the REGISTERED cached tables (data/processed/cache, read-only; the run reused the incumbent
      cache m2feat_3d35f8c314e9) -> fit_stacker_huber seeds 0-4 at 2 LightGBM threads -> compare with the
      registered per-seed predictions (expect bit-identical) and fold-K primary.
  (b) FoldDataP rebuilt from scratch (no cache, torch 1 thread, as in check1) -> column-wise comparison with (a)
      (which features differ and by how much) -> same 5 fits -> fold-K primary.
Prints aggregates only."""
import common
import json
import sys
import time

import numpy as np
import pandas as pd

from src.timing import Timings, set_threads
set_threads(common.THREADS)
import torch  # noqa: E402
torch.set_num_threads(1)

import run_m2plus as RP  # noqa: E402
from src import data, protocol  # noqa: E402
from src import pairs as PA  # noqa: E402
from src.models import lgbm, m2  # noqa: E402
from src.ryanfeat import SimStore  # noqa: E402
import check3_metric as C3  # noqa: E402

RP.THREADS = common.THREADS
RP.R.N_JOBS = common.THREADS
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEEDS = [int(s) for s in RP.CFG["seeds"]["bag"]]
REG = "M2r_bag5_huber__20261028T202603__f8bde5"
ORIG_FT = m2.fold_tables_m2
LIVE_CACHE = common.LIVE / "cache" / str(data._seed(None)) / f"fold{K}"
NAME = "m2feat_3d35f8c314e9"
data.cache_save = lambda *a, **kw: None


def cached_tables(split_seed=None, fold=0, sims=None, cfg=None, gp_cfg=None, use_cache=True, timings=None, **kw):
    assert m2.m2_cache_name(m2.resolve_cfg(cfg), sims, gp_cfg, kw.get("exclude_groups", ()), kw.get("extra_key")) == NAME
    train, ev, drop = m2.fold_pairs(split_seed, fold)
    inner = data.inner_splits(train)
    rows = [pd.read_parquet(LIVE_CACHE / f"inner{f.j}" / f"{NAME}.parquet") for f in inner]
    return pd.concat(rows, ignore_index=True), pd.read_parquet(LIVE_CACHE / f"{NAME}.parquet"), inner, drop


lab = pd.read_parquet(common.LIVE / "splits" / "labels_dev.parquet", columns=["gene_a", "gene_b", "s_score"])
lab["y"] = (lab.s_score < C3.THR).astype(np.int8)
lab = lab.drop(columns="s_score")
spec = PA.load_spec()
sims = SimStore(names=SimStore().names)
params = lgbm.params_for("default", common.THREADS)
q = pd.read_parquet(common.REG_PREDS / "m2plus" / f"{REG}_perseed.parquet")
q = q[q.fold == K]
reg_fits = [f for f in json.load(open(common.REPO / "results" / "m2plus" / "summary.json"))["runs"]["M2r_bag5_huber"]["fits"]
            if f["fold"] == K]
out = {"fold": K, "registered_best_iter": [f["best_iteration"] for f in reg_fits]}
t0 = time.time()


def fit_all(D, tag):
    cols = D.cols(RP.FAMILIES)
    preds, fits = [], []
    for s in SEEDS:
        b, inf = RP.fit_stacker_huber(D, cols, params, s, s % D.n_inner)
        preds.append(RP.R.predict(b, D, cols, False))
        fits.append(inf["best_iteration"])
        print(f"{tag} seed {s} best_iter {fits[-1]} {time.time()-t0:.0f}s", flush=True)
    ev = D.ev[["gene_a", "gene_b", "fold", "part"]].assign(score=np.mean(preds, axis=0))
    df = ev.merge(lab, on=["gene_a", "gene_b"], how="left", validate="one_to_one")
    prim = C3.Fold(df, pd.Index(sorted(set(df.gene_a) | set(df.gene_b)))).ap(C3.CW)
    cmp = {}
    for i, s in enumerate(SEEDS):
        g = D.ev[["gene_a", "gene_b"]].assign(mine=preds[i]).merge(q[q.seed == s][["gene_a", "gene_b", "score"]],
                                                                   on=["gene_a", "gene_b"], validate="one_to_one")
        cmp[s] = {"bit_identical": bool((g.mine.to_numpy() == g.score.to_numpy()).all()),
                  "max_abs": float(np.max(np.abs(g.mine - g.score)))}
    return {"best_iter": fits, "fold_primary_seedavg": round(prim, 5), "vs_registered": cmp}


m2.fold_tables_m2 = cached_tables
Dc = RP.FoldDataP(None, K, spec, sims, Timings(), "M2r")
out["a_cached"] = fit_all(Dc, "cached")
print(json.dumps(out["a_cached"]), flush=True)
Xtr_c, Xev_c, names = Dc.Xtr, Dc.Xev, list(Dc.names)
s_c, w_c, y_c = Dc.s, Dc.w, Dc.y
del Dc
m2.fold_tables_m2 = lambda *a, **kw: ORIG_FT(*a, **{**kw, "use_cache": False})
Df = RP.FoldDataP(None, K, spec, sims, Timings(), "M2r")
assert list(Df.names) == names
diff = {}
for nm, A, B in (("Xtr", Xtr_c, Df.Xtr), ("Xev", Xev_c, Df.Xev)):
    d = {}
    for i, n in enumerate(names):
        a, b = A[:, i], B[:, i]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            d[n] = {"n_diff": int((~same).sum()), "max_abs": float(np.nanmax(np.abs(a - b)))}
    diff[nm] = d
out["fresh_vs_cached_feature_diff"] = diff
out["fresh_vs_cached_target_s_identical"] = bool(np.array_equal(s_c, Df.s))
out["fresh_vs_cached_w_identical"] = bool(np.array_equal(w_c, Df.w))
out["fresh_vs_cached_y_identical"] = bool(np.array_equal(y_c, Df.y))
print(json.dumps({k: out[k] for k in out if k.startswith("fresh")}, default=str)[:3000], flush=True)
del Xtr_c, Xev_c
out["b_fresh"] = fit_all(Df, "fresh")
out["wall_s"] = round(time.time() - t0)
print(json.dumps(out, indent=1))
(common.OUT / f"check1b_reproduce_fold{K}.json").write_text(json.dumps(out, indent=1))
