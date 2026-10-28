"""Nested stacking of Ryan-derived features (DESIGN.md §4 rule 2).

For an outer fold k (training pairs T, eval pairs E from src.data.fold_split):
* inner = src.data.inner_splits(T): 5 inner gene folds T_j + a 10% inner pair hold-out H_j.
* Stacker TRAINING rows = the inner eval rows of every j (C3-like: both genes in T_j; C2-like: one gene
  in T_j, one seeded copy kept; C1-like: H_j). Their Ryan features are computed by
  src.ryanfeat.build_ryan_features from inner fold j's training pairs only (neither gene in T_j, not
  in H_j). Each row carries its label (y, s_score), part, inner_fold, hypomorph flags and a class
  weight `w` = protocol class weight of its part / number of rows of that part in inner fold j.
* Outer EVAL rows: identical feature computation from all outer-training pairs T.
* Gene-grouped stacker validation: `stacker_folds(train_rows, inner)` = src.data.gene_grouped_validation.

    from src.stacking import fold_tables
    tr_rows, ev_rows, inner = fold_tables(split_seed=None, fold=0)       # cached

Cache (src.data cache helpers; key = dev-label-file hash): <cache>/<split_seed>/fold<k>/<name>.parquet
for the outer eval rows and .../fold<k>/inner<j>/<name>.parquet for the stacker training rows, with
name = "ryanfeat_<cfg hash>" (the hash covers the feature-set version, the config and the sha256 of every
similarity matrix used).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data, protocol
from src.ryanfeat import RyanFeatureBuilder, SimStore, cfg_hash, resolve_cfg

META_TRAIN = ["gene_a", "gene_b", "y", "s_score", "part", "inner_fold", "hypomorph_a", "hypomorph_b"]
META_EVAL = ["gene_a", "gene_b", "fold", "part", "hypomorph_a", "hypomorph_b"]


def class_weight_column(part: np.ndarray) -> np.ndarray:
    cw = protocol.class_weights()
    w = np.zeros(len(part))
    for c, wc in cw.items():
        m = part == c
        if m.any():
            w[m] = wc / m.sum()
    return w


def inner_rows(f, sims, cfg, builder=RyanFeatureBuilder) -> pd.DataFrame:
    """Stacker training rows of inner fold f (features from f.train only)."""
    ev = f.eval.reset_index(drop=True)
    F = builder(f.train, ev[["gene_a", "gene_b"]], sims, cfg).build()
    assert (F.gene_a.to_numpy() == ev.gene_a.to_numpy()).all()
    meta = ev[[c for c in META_TRAIN if c in ev]].copy()
    meta["w"] = class_weight_column(meta.part.to_numpy())
    return pd.concat([meta, F.drop(columns=["gene_a", "gene_b"])], axis=1)


def outer_rows(train, ev, sims, cfg, builder=RyanFeatureBuilder) -> pd.DataFrame:
    ev = ev.reset_index(drop=True)
    F = builder(train, ev[["gene_a", "gene_b"]], sims, cfg).build()
    return pd.concat([ev[[c for c in META_EVAL if c in ev]], F.drop(columns=["gene_a", "gene_b"])], axis=1)


def nested_tables(train: pd.DataFrame, ev: pd.DataFrame, sims, cfg=None, inner=None, seed=None,
                  builder=RyanFeatureBuilder, outer: bool = True):
    """Pure (no cache) nested stacking tables: (stacker training rows, outer eval rows, inner folds)."""
    inner = inner if inner is not None else data.inner_splits(train, seed)
    tr_rows = pd.concat([inner_rows(f, sims, cfg, builder) for f in inner], ignore_index=True)
    ev_rows = outer_rows(train, ev, sims, cfg, builder) if outer else None
    return tr_rows, ev_rows, inner


def fold_tables(split_seed=None, fold: int = 0, sims: SimStore | None = None, cfg: dict | None = None,
                use_cache: bool = True):
    """Cached nested tables for outer fold `fold`: (stacker training rows, outer eval rows, inner folds)."""
    sims = sims if sims is not None else SimStore()
    cfg = resolve_cfg(cfg)
    name = f"ryanfeat_{cfg_hash(cfg, sims)}"
    train, ev = data.fold_split(split_seed, fold)
    inner = data.inner_splits(train)
    parts = []
    for f in inner:
        x = data.cache_load(name, split_seed, fold, inner=f.j) if use_cache else None
        if x is None:
            x = inner_rows(f, sims, cfg)
            data.cache_save(x, name, split_seed, fold, inner=f.j)
        parts.append(x)
    ev_rows = data.cache_load(name, split_seed, fold) if use_cache else None
    if ev_rows is None:
        ev_rows = outer_rows(train, ev, sims, cfg)
        data.cache_save(ev_rows, name, split_seed, fold)
    return pd.concat(parts, ignore_index=True), ev_rows, inner


def stacker_folds(tr_rows: pd.DataFrame, inner) -> list:
    """Gene-grouped stacker CV (DESIGN §4 rule 2 v3.1) on the feature rows, via
    src.data.gene_grouped_validation: [(j, stacker training rows, validation rows)]. Relies on tr_rows
    being the concatenation of the inner folds' eval rows in order (as built by nested_tables/fold_tables)."""
    assert len(tr_rows) == sum(len(f.eval) for f in inner)
    return [(j, tr_rows.loc[t.index], tr_rows.loc[v.index]) for j, t, v in data.gene_grouped_validation(inner)]
