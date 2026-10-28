"""Check 1b: the only canary difference was mf_score. Show it is torch CPU non-determinism, not a leak:
fit the MF (production fit_mf + mf_score, production cfg) on fold-0 outer training pairs built from
(a) clean and (b) eval-corrupted labels_dev, repeatedly, with 1 / 2 / 3 torch threads; report max |Δ|
on the eval pairs and whether results are bit-identical. Aggregates only."""
import common
import json

import numpy as np
import pandas as pd
import torch

from src import data
from src import splits as S
from src.models import m2
from src.ryanfeat import DEFAULT_CFG, _clean_train, fit_mf, mf_score, resolve_cfg

C = resolve_cfg({k: DEFAULT_CFG[k] for k in ("b1_C", "mf_k", "mf_wd", "mf_epochs", "mf_lr")})
CLEAN = data._labels_dev(str(data.protocol.splits_dir())).copy()
K = 0
folds = pd.read_csv(common.FROZEN / "splits" / "dev_folds_seed20261028.tsv", sep="\t").set_index("gene").fold
hold = pd.read_parquet(common.FROZEN / "splits" / "c1_holdout_seed20261028.parquet")
hold = hold[hold.fold == K]
mask = np.asarray((CLEAN.gene_a.map(folds) == K) | (CLEAN.gene_b.map(folds) == K) |
                  S.pair_keys(CLEAN.gene_a, CLEAN.gene_b).isin(S.pair_keys(hold.gene_a, hold.gene_b)))
BAD = CLEAN.copy()
BAD.loc[mask, ["s_score", "s_raw_mean"]] = -10.0
BAD.loc[mask, ["y", "y_raw"]] = True


def mf_eval(labels, threads):
    data._labels_dev = lambda sd, _d=labels: _d
    torch.set_num_threads(threads)
    tr, ev, _ = m2.fold_pairs(None, K)
    P = fit_mf(_clean_train(tr), C["mf_k"], C["mf_wd"], C["mf_epochs"], C["mf_lr"], C["mf_seed"])
    return mf_score(P, ev.gene_a.to_numpy(object), ev.gene_b.to_numpy(object)).astype(np.float32)


cache = pd.read_parquet(common.REPO / "data/processed/cache/20261028/fold0/m2feat_3d35f8c314e9.parquet",
                        columns=["mf_score"]).mf_score.to_numpy()
out = {}
res = {}
for t in (1, 2, 3):
    res[t] = [mf_eval(CLEAN, t), mf_eval(CLEAN, t), mf_eval(BAD, t)]
    a, b, c = res[t]
    out[f"threads{t}"] = {"clean_vs_clean_identical": bool(a.tobytes() == b.tobytes()),
                          "clean_vs_clean_maxabs": float(np.abs(a - b).max()),
                          "clean_vs_corrupt_identical": bool(a.tobytes() == c.tobytes()),
                          "clean_vs_corrupt_maxabs": float(np.abs(a - c).max()),
                          "vs_registered_cache_identical": bool(a.tobytes() == cache.tobytes()),
                          "vs_registered_cache_maxabs": float(np.abs(a - cache).max())}
    print(t, out[f"threads{t}"], flush=True)
out["mf_score_sd"] = float(cache.std())
print(json.dumps(out, indent=1))
(common.OUT / "check1b_mf_determinism.json").write_text(json.dumps(out, indent=1))
