"""Pure split-generation and split-checking functions (no file I/O; DESIGN.md §2, §4 rule 2).

Used by scripts/make_splits.py (outer splits), src/data.py (fold views, nested inner splits) and the
tests (each checker has deliberately broken variants in tests/ that it must flag).

Conventions: pair tables have columns gene_a < gene_b (str) and a boolean label column `y`.
Parts: C1 = both genes seen in training, C2 = exactly one, C3 = neither (Park & Marcotte 2012).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PARTS = ("C1", "C2", "C3")


# ------------------------------------------------------------------------------------------- utils
def pair_keys(a, b) -> pd.Index:
    """Order-insensitive string key per pair (gene IDs never contain '|')."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    lo = np.where(a < b, a, b)
    hi = np.where(a < b, b, a)
    return pd.Index(pd.Series(lo).astype(str) + "|" + pd.Series(hi).astype(str))


def gene_hit_rates(pairs: pd.DataFrame) -> pd.Series:
    long = pd.concat([pairs[["gene_a", "y"]].rename(columns={"gene_a": "g"}),
                      pairs[["gene_b", "y"]].rename(columns={"gene_b": "g"})])
    return long.groupby("g").y.mean().rename("hit_rate")


def stratify_genes(hit_rate: pd.Series, role: pd.Series | None, hub_quantile: float,
                   n_quantiles: int) -> pd.Series:
    """Stratum per gene: role x within-role hit-rate quantile; the top (1-hub_quantile) genes by hit
    rate form a separate 'hub' stratum. Same scheme as scripts/draw_lockbox.py."""
    df = pd.DataFrame({"hit_rate": hit_rate})
    df["role"] = (role.reindex(df.index).fillna("unknown") if role is not None else "all")
    df["stratum"] = df.role
    hub = df.hit_rate >= df.hit_rate.quantile(hub_quantile)
    for r, sub in df[~hub].groupby("role"):
        nq = min(n_quantiles, len(sub))
        q = pd.qcut(sub.hit_rate.rank(method="first"), nq, labels=False)
        df.loc[sub.index, "stratum"] = r + "_q" + q.astype(str)
    df.loc[hub, "stratum"] = "hub"
    return df.stratum


def assign_folds(strata: pd.Series, n_folds: int, seed) -> pd.Series:
    """Stratified gene -> fold. Within each stratum (sorted by name, genes sorted) genes are permuted
    and dealt round-robin; the dealing position carries over between strata so fold sizes balance."""
    rng = np.random.default_rng(seed)
    fold = pd.Series(-1, index=strata.index, dtype=int)
    pos = 0
    for s in sorted(strata.unique()):
        ids = np.sort(strata.index[strata == s].values)
        ids = ids[rng.permutation(len(ids))]
        fold.loc[ids] = (pos + np.arange(len(ids))) % n_folds
        pos = (pos + len(ids)) % n_folds
    return fold.sort_index()


def stratified_pair_sample(y: np.ndarray, eligible: np.ndarray, frac: float, rng) -> np.ndarray:
    """Boolean mask selecting round(frac * n) eligible rows within each label stratum."""
    y = np.asarray(y, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    out = np.zeros(len(y), dtype=bool)
    for lab in (False, True):
        idx = np.flatnonzero(eligible & (y == lab))
        k = int(round(frac * len(idx)))
        if k:
            out[rng.choice(idx, k, replace=False)] = True
    return out


# ------------------------------------------------------------------------------------------- outer
def fold_parts(fold_a: np.ndarray, fold_b: np.ndarray, k: int, c1_mask: np.ndarray,
               eligible: np.ndarray, train_ok: np.ndarray) -> np.ndarray:
    """Part per pair for outer fold k: 'C1'/'C2'/'C3' (eval rows, eligible only), 'train', or '' (neither).
    c1_mask: the fold's C1 hold-out pairs (must have neither gene in fold k)."""
    n_in = (np.asarray(fold_a) == k).astype(int) + (np.asarray(fold_b) == k).astype(int)
    part = np.full(len(n_in), "", dtype=object)
    part[(n_in == 0) & ~c1_mask & train_ok] = "train"
    part[(n_in == 0) & c1_mask & eligible] = "C1"
    part[(n_in == 1) & eligible] = "C2"
    part[(n_in == 2) & eligible] = "C3"
    return part


def check_fold_split(train: pd.DataFrame, ev: pd.DataFrame, fold_genes) -> list[str]:
    """Violations of the outer split invariants (empty list = OK)."""
    v = []
    fold_genes = set(fold_genes)
    tg = set(train.gene_a) | set(train.gene_b)
    leaked = tg & fold_genes
    if leaked:
        v.append(f"{len(leaked)} held-out (C2/C3) fold genes appear in training pairs")
    for part, n_req in (("C3", 2), ("C2", 1), ("C1", 0)):
        sub = ev[ev.part == part]
        n_in = sub.gene_a.isin(fold_genes).astype(int) + sub.gene_b.isin(fold_genes).astype(int)
        if (n_in != n_req).any():
            v.append(f"{int((n_in != n_req).sum())} {part} rows have the wrong number of fold genes")
    c1 = ev[ev.part == "C1"]
    unseen = ~(c1.gene_a.isin(tg) & c1.gene_b.isin(tg))
    if unseen.any():
        v.append(f"{int(unseen.sum())} C1 rows have a gene not seen in training")
    overlap = pair_keys(train.gene_a, train.gene_b).intersection(pair_keys(ev.gene_a, ev.gene_b))
    if len(overlap):
        v.append(f"{len(overlap)} pairs are in both train and eval")
    if pair_keys(ev.gene_a, ev.gene_b).duplicated().any():
        v.append("duplicate eval pairs")
    return v


def check_dev_labels(dev: pd.DataFrame, lockbox_genes, lockbox_c1: pd.DataFrame) -> list[str]:
    v = []
    lb = set(lockbox_genes)
    bad = dev.gene_a.isin(lb) | dev.gene_b.isin(lb)
    if bad.any():
        v.append(f"{int(bad.sum())} dev pairs involve a lockbox gene")
    ov = pair_keys(dev.gene_a, dev.gene_b).intersection(pair_keys(lockbox_c1.gene_a, lockbox_c1.gene_b))
    if len(ov):
        v.append(f"{len(ov)} lockbox-C1 pairs are in the dev labels")
    return v


# ------------------------------------------------------------------------------------------- inner
@dataclass
class InnerFold:
    j: int
    genes: np.ndarray                 # T_j
    train: pd.DataFrame               # inner-training pairs: neither gene in T_j, not in H_j
    eval: pd.DataFrame                # inner eval rows (stacker training rows) with column `part`
    holdout: pd.DataFrame = field(repr=False, default=None)   # H_j pairs (gene_a, gene_b)


def inner_split_generator(train_pairs: pd.DataFrame, n_folds: int, c1_fraction: float, seed,
                          role: pd.Series | None, hub_quantile: float, n_quantiles: int) -> list[InnerFold]:
    """Nested C1/C2/C3 generator inside an outer training set (DESIGN §4 rule 2).

    Inner gene folds T_j are stratified like the outer folds (hit rates from `train_pairs` only).
    H_j = label-stratified `c1_fraction` hold-out among pairs with neither gene in T_j (restricted to
    `eval_eligible` rows if that column exists, mirroring the outer C1 rule).
    C2-like pairs (genes in two different inner folds) are kept in exactly one of the two folds,
    chosen by a seeded coin flip."""
    tp = train_pairs.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    strata = stratify_genes(gene_hit_rates(tp), role, hub_quantile, n_quantiles)
    gfold = assign_folds(strata, n_folds, rng.integers(2**32))
    fa = tp.gene_a.map(gfold).to_numpy()
    fb = tp.gene_b.map(gfold).to_numpy()
    y = tp.y.to_numpy(dtype=bool)
    eligible = tp.eval_eligible.to_numpy(dtype=bool) if "eval_eligible" in tp else np.ones(len(tp), bool)
    keep_c2_in = np.where(rng.random(len(tp)) < 0.5, fa, fb)
    out = []
    for j in range(n_folds):
        a_in, b_in = fa == j, fb == j
        neither = ~a_in & ~b_in
        h = stratified_pair_sample(y, neither & eligible, c1_fraction, rng)
        c3 = a_in & b_in
        c2 = (a_in ^ b_in) & (keep_c2_in == j)
        part = np.full(len(tp), "", dtype=object)
        part[h], part[c2], part[c3] = "C1", "C2", "C3"
        ev = tp[part != ""].copy()
        ev["part"] = part[part != ""]
        ev["inner_fold"] = j
        out.append(InnerFold(j=j, genes=np.sort(gfold.index[gfold == j].values),
                             train=tp[neither & ~h], eval=ev, holdout=tp.loc[h, ["gene_a", "gene_b"]]))
    return out


def check_inner_split(inner: list[InnerFold]) -> list[str]:
    v = []
    for f in inner:
        T = set(f.genes)
        if (f.train.gene_a.isin(T) | f.train.gene_b.isin(T)).any():
            v.append(f"inner fold {f.j}: training pairs include a T_j gene")
        hk = pair_keys(f.holdout.gene_a, f.holdout.gene_b)
        if len(pair_keys(f.train.gene_a, f.train.gene_b).intersection(hk)):
            v.append(f"inner fold {f.j}: training pairs include an H_j pair")
        if len(pair_keys(f.train.gene_a, f.train.gene_b).intersection(pair_keys(f.eval.gene_a, f.eval.gene_b))):
            v.append(f"inner fold {f.j}: a pair is both inner-train and inner-eval")
    c2 = pd.concat([f.eval[f.eval.part == "C2"] for f in inner])
    if pair_keys(c2.gene_a, c2.gene_b).duplicated().any():
        v.append("a C2-like pair is kept in more than one inner fold")
    return v


def gene_grouped_validation(inner: list[InnerFold]) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Stacker CV over the union of inner eval rows (DESIGN §4 rule 2, v3.1): validating on inner fold
    j's rows uses as training only rows from other inner folds that involve NO T_j gene.
    Returns [(j, stacker_train_rows, stacker_valid_rows)]."""
    rows = pd.concat([f.eval for f in inner], ignore_index=True)
    out = []
    for f in inner:
        T = set(f.genes)
        valid = rows[rows.inner_fold == f.j]
        tr = rows[(rows.inner_fold != f.j) & ~rows.gene_a.isin(T) & ~rows.gene_b.isin(T)]
        # a pair can be an eval row in several inner folds (e.g. in H_j and H_l): never train on it
        tr = tr[~pair_keys(tr.gene_a, tr.gene_b).isin(pair_keys(valid.gene_a, valid.gene_b))]
        out.append((f.j, tr, valid))
    return out


def check_gene_grouped(groups, inner: list[InnerFold]) -> list[str]:
    v = []
    genes = {f.j: set(f.genes) for f in inner}
    for j, tr, va in groups:
        if (tr.gene_a.isin(genes[j]) | tr.gene_b.isin(genes[j])).any():
            v.append(f"stacker fold {j}: training rows involve a T_j gene")
        if len(pair_keys(tr.gene_a, tr.gene_b).intersection(pair_keys(va.gene_a, va.gene_b))):
            v.append(f"stacker fold {j}: a validation pair is also a training row")
    return v
