"""Check 1: nesting-leak canary for M2's Ryan-derived features (RYAN-fold, NBR, GP).

Outer canary (fold K): rebuild the full nested tables of the production path (m2.fold_pairs ->
data.inner_splits -> stacking.inner_rows / outer_rows with the bound M2Builder; = m2.fold_tables_m2
minus the cache) twice: (a) clean labels, (b) labels_dev corrupted IN MEMORY for every pair touching a
fold-K gene or in fold K's C1 hold-out (s_score=-10, y=1, raw too). Every eval-row feature and every
stacker-row column must be bit-identical.
Inner canary (outer fold K, inner folds J): corrupt (in the outer training set) the labels of every
pair touching a T_j gene or in H_j, rebuild the inner fold's training set with the SAME T_j/H_j (the
generator's stratification itself reads labels, so it is held fixed), pass the corrupted inner eval
rows too, and rebuild inner_rows; features must be bit-identical. Also asserts f.train equals my own
re-filter (neither gene in T_j, not in H_j) of the clean outer training set.
Positive controls (in-memory mutants; no file is modified):
  P1 outer: data.fold_split patched to append the fold's C1 eval pairs WITH labels to the training set.
  P2 inner: inner training set = pairs with neither gene in T_j, H_j NOT removed.
Also: clean fold-K tables are compared with the cached tables the registered M2 run used.
Prints only aggregate counts / max abs differences (no pair-level scores)."""
import common
import json
import sys
import time

import numpy as np
import pandas as pd

from src.timing import set_threads
set_threads(common.THREADS)
import torch  # noqa: E402
torch.set_num_threads(common.THREADS)

from src import data, stacking  # noqa: E402
from src import splits as S  # noqa: E402
from src.models import m2  # noqa: E402
from src.ryanfeat import DEFAULT_CFG, SimStore, resolve_cfg  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
JS = [0, 3]
LABCOLS = ["s_score", "y", "s_raw_mean", "y_raw"]
CFG = resolve_cfg({k: DEFAULT_CFG[k] for k in ("b1_C", "mf_k", "mf_wd", "mf_epochs", "mf_lr")})
SIMS = SimStore(names=SimStore().names)
BUILDER = m2.make_builder(m2.gene_table(), None)
CLEAN = data._labels_dev(str(data.protocol.splits_dir())).copy()
ORIG_LABELS_DEV, ORIG_FOLD_SPLIT = data._labels_dev, data.fold_split
log = {"fold": K}


def corrupt(df, mask):
    d = df.copy()
    d.loc[mask, "s_score"] = -10.0
    d.loc[mask, "s_raw_mean"] = -10.0
    d.loc[mask, "y"] = True
    d.loc[mask, "y_raw"] = True
    return d


def set_labels(df):
    data._labels_dev = lambda sd, _d=df: _d          # module attribute used by _parts and labels_dev()


def compare(a: pd.DataFrame, b: pd.DataFrame, skip=()):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    diff = {}
    for c in a.columns:
        if c in skip:
            continue
        x, z = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            same = x.tobytes() == z.tobytes()
            if not same:
                with np.errstate(invalid="ignore"):
                    nd = int((~((x == z) | (np.isnan(x) & np.isnan(z)))).sum())
                diff[c] = nd
        elif not np.array_equal(x, z):
            diff[c] = int((x != z).sum())
    return diff


def outer_tables(inner_too=True):
    tr, ev, _ = m2.fold_pairs(None, K)
    inner = data.inner_splits(tr)
    rows = [stacking.inner_rows(f, SIMS, CFG, BUILDER) for f in inner] if inner_too else []
    evr = stacking.outer_rows(tr, ev, SIMS, CFG, BUILDER)
    return tr, ev, inner, rows, evr


t0 = time.time()
# ------------------------------------------------------------------ clean
set_labels(CLEAN)
tr_c, ev_c, inner_c, rows_c, evr_c = outer_tables()
log["n_eval_rows"], log["n_stacker_rows"] = len(evr_c), int(sum(len(r) for r in rows_c))
print(f"clean built {time.time()-t0:.0f}s", flush=True)

# compare to the cached tables of the registered run (read-only; data/processed/cache)
cdir = common.REPO / "data" / "processed" / "cache" / "20261028" / f"fold{K}"
name = "m2feat_3d35f8c314e9"
cev = pd.read_parquet(cdir / f"{name}.parquet")
log["clean_vs_registered_cache_eval"] = compare(evr_c, cev)
log["clean_vs_registered_cache_inner"] = {j: compare(rows_c[j], pd.read_parquet(cdir / f"inner{j}" / f"{name}.parquet"))
                                          for j in range(len(rows_c))}

# ------------------------------------------------------------------ outer canary
folds = pd.read_csv(common.FROZEN / "splits" / "dev_folds_seed20261028.tsv", sep="\t").set_index("gene").fold
hold = pd.read_parquet(common.FROZEN / "splits" / "c1_holdout_seed20261028.parquet")
hold = hold[hold.fold == K]
in_fold = (CLEAN.gene_a.map(folds) == K) | (CLEAN.gene_b.map(folds) == K)
in_hold = S.pair_keys(CLEAN.gene_a, CLEAN.gene_b).isin(S.pair_keys(hold.gene_a, hold.gene_b))
mask = np.asarray(in_fold | in_hold)
log["outer_corrupted_pairs"] = int(mask.sum())
BAD = corrupt(CLEAN, mask)
set_labels(BAD)
tr_b, ev_b, inner_b, rows_b, evr_b = outer_tables()
print(f"corrupt built {time.time()-t0:.0f}s", flush=True)
log["outer_canary_train_pairs_identical"] = bool(tr_b.equals(tr_c))
log["outer_canary_eval_diff"] = compare(evr_c, evr_b)
log["outer_canary_stacker_diff"] = {j: compare(rows_c[j], rows_b[j]) for j in range(len(rows_c))}
set_labels(CLEAN)


# ------------------------------------------------------------------ inner canary
def refilter(train, f, exclude_h=True):
    T = set(f.genes)
    hk = S.pair_keys(f.holdout.gene_a, f.holdout.gene_b)
    keep = ~train.gene_a.isin(T) & ~train.gene_b.isin(T)
    if exclude_h:
        keep &= ~S.pair_keys(train.gene_a, train.gene_b).isin(hk)
    return train[np.asarray(keep)]


log["inner"] = {}
for j in JS:
    f = inner_c[j]
    T = set(f.genes)
    hk = S.pair_keys(f.holdout.gene_a, f.holdout.gene_b)
    mine = refilter(tr_c, f)
    e = {"f.train_equals_my_refilter": bool(S.pair_keys(mine.gene_a, mine.gene_b).sort_values().equals(
        S.pair_keys(f.train.gene_a, f.train.gene_b).sort_values()))}
    m_in = np.asarray(tr_c.gene_a.isin(T) | tr_c.gene_b.isin(T) | S.pair_keys(tr_c.gene_a, tr_c.gene_b).isin(hk))
    e["corrupted_pairs"] = int(m_in.sum())
    trB = corrupt(tr_c, m_in)
    evB = corrupt(f.eval, np.ones(len(f.eval), bool))
    fB = S.InnerFold(j=f.j, genes=f.genes, train=refilter(trB, f), eval=evB, holdout=f.holdout)
    rB = stacking.inner_rows(fB, SIMS, CFG, BUILDER)
    e["feature_diff"] = compare(rows_c[j], rB, skip=("y", "s_score"))
    # P2 positive control: inner training set keeps H_j
    fM = S.InnerFold(j=f.j, genes=f.genes, train=refilter(tr_c, f, exclude_h=False), eval=f.eval, holdout=f.holdout)
    fMB = S.InnerFold(j=f.j, genes=f.genes, train=refilter(trB, f, exclude_h=False), eval=evB, holdout=f.holdout)
    a, b = stacking.inner_rows(fM, SIMS, CFG, BUILDER), stacking.inner_rows(fMB, SIMS, CFG, BUILDER)
    dd = compare(a, b, skip=("y", "s_score"))
    e["P2_mutant_diff_n_columns"] = len(dd)
    e["P2_mutant_diff_example"] = dict(list(dd.items())[:6])
    e["P2_detected"] = len(dd) > 0
    log["inner"][j] = e
    print(f"inner {j} done {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ P1 positive control (outer)
def mutant_fold_split(split_seed=None, fold=0):
    tr, ev = ORIG_FOLD_SPLIT(split_seed, fold)
    lab = data.labels_dev()
    c1 = ev[ev.part == "C1"]
    add = lab[S.pair_keys(lab.gene_a, lab.gene_b).isin(S.pair_keys(c1.gene_a, c1.gene_b))][data.TRAIN_COLUMNS]
    return pd.concat([tr, add], ignore_index=True), ev


data.fold_split = mutant_fold_split
set_labels(CLEAN)
_, _, _, _, evr_mc = outer_tables(inner_too=False)
set_labels(BAD)
_, _, _, _, evr_mb = outer_tables(inner_too=False)
set_labels(CLEAN)
data.fold_split = ORIG_FOLD_SPLIT
dd = compare(evr_mc, evr_mb)
log["P1_mutant_diff_n_columns"] = len(dd)
log["P1_mutant_diff_example"] = dict(list(dd.items())[:6])
log["P1_detected"] = len(dd) > 0
log["wall_s"] = round(time.time() - t0)
print(json.dumps(log, indent=1, default=str))
(common.OUT / f"check1_canary_fold{K}.json").write_text(json.dumps(log, indent=1, default=str))
