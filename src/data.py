"""Fold-scoped label API (DESIGN.md §2, §4 rule 2). The ONLY way models obtain Ryan labels.

* `fold_split(split_seed, fold)` -> (train, eval): training pairs WITH labels; eval pairs WITHOUT
  labels (gene_a, gene_b, fold, part in C1/C2/C3), already filtered by the evaluation masks.
* `inner_splits(train, seed)` -> nested inner C1/C2/C3 splits of an outer training set.
* `gene_grouped_validation(inner)` -> stacker CV that never trains on rows involving T_j genes.
* `cache_path(...)` -> fold-scoped feature-cache path keyed by the dev-label-file hash.

This module never reads the lockbox label file (lint test + runtime guard in `_read`).
Eval labels are joined only inside src/evaluate.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import protocol
from src import splits as S

# Columns of labels_dev exposed on TRAINING rows. Design-revealing columns (both_orientations,
# per-orientation raw scores, measurement counts) are deliberately not exposed (§4 rule 1).
TRAIN_COLUMNS = ["gene_a", "gene_b", "s_score", "y", "s_raw_mean", "y_raw", "in_universe", "eval_eligible",
                 "viable_viable", "hypomorph_a", "hypomorph_b"]
EVAL_COLUMNS = ["gene_a", "gene_b", "fold", "part", "hypomorph_a", "hypomorph_b"]


def _read(path: Path) -> pd.DataFrame:
    if "lockbox" in Path(path).name:
        raise PermissionError("src/data.py must never read lockbox files")
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, sep="\t")


@lru_cache(maxsize=2)
def _labels_dev(splits_dir: str) -> pd.DataFrame:
    return _read(Path(splits_dir) / "labels_dev.parquet")


def labels_dev() -> pd.DataFrame:
    return _labels_dev(str(protocol.splits_dir()))


def labels_hash() -> str:
    """sha256 of the dev label file (cache key)."""
    return protocol.sha256_file(protocol.splits_dir() / "labels_dev.parquet")


def split_seeds() -> tuple[int, int]:
    D = protocol.load()["dev_cv"]
    return D["split_seed"], D["alt_split_seed"]


def _seed(split_seed):
    return protocol.load()["dev_cv"]["split_seed"] if split_seed is None else int(split_seed)


@lru_cache(maxsize=4)
def _folds(splits_dir: str, seed: int) -> pd.Series:
    f = _read(Path(splits_dir) / f"dev_folds_seed{seed}.tsv")
    return f.set_index("gene").fold


def dev_folds(split_seed=None) -> pd.Series:
    """gene -> fold for the dev genes."""
    return _folds(str(protocol.splits_dir()), _seed(split_seed))


def fold_genes(split_seed=None, fold=0) -> np.ndarray:
    g = dev_folds(split_seed)
    return np.sort(g.index[g == fold].values)


def _parts(split_seed, fold) -> np.ndarray:
    sd = str(protocol.splits_dir())
    dev = _labels_dev(sd)
    gfold = _folds(sd, _seed(split_seed))
    hold = _read(Path(sd) / f"c1_holdout_seed{_seed(split_seed)}.parquet")
    hold = hold[hold.fold == fold]
    c1 = S.pair_keys(dev.gene_a, dev.gene_b).isin(S.pair_keys(hold.gene_a, hold.gene_b))
    return S.fold_parts(dev.gene_a.map(gfold).to_numpy(), dev.gene_b.map(gfold).to_numpy(), fold, c1,
                        dev.eval_eligible.to_numpy(), ~dev.train_drop.to_numpy())


def fold_split(split_seed=None, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, eval) for outer fold `fold` of the dev CV with seed `split_seed` (default: primary).

    train: TRAIN_COLUMNS for pairs with neither gene in the fold, minus the fold's C1 hold-out, minus
           pairs involving drop-from-training genes.
    eval:  EVAL_COLUMNS (no labels) for eval-eligible C1/C2/C3 pairs; hypomorph_a/b are 0 by definition
           (the deliverable predicts deletion x deletion interactions).
    """
    part = _parts(split_seed, fold)
    dev = labels_dev()
    train = dev.loc[part == "train", TRAIN_COLUMNS].reset_index(drop=True)
    m = np.isin(part, S.PARTS)
    ev = dev.loc[m, ["gene_a", "gene_b"]].reset_index(drop=True)
    ev["fold"] = fold
    ev["part"] = part[m]
    ev["hypomorph_a"] = np.int8(0)
    ev["hypomorph_b"] = np.int8(0)
    return train, ev[EVAL_COLUMNS]


def fold_labels(split_seed=None, fold: int = 0, part: str = "train") -> pd.DataFrame:
    """DESIGN §2 name for the label API. Only training labels are available to models."""
    if part != "train":
        raise PermissionError("eval labels are only joined inside src/evaluate.py")
    return fold_split(split_seed, fold)[0]


def protein_coding_genes() -> set:
    """PomBase protein-coding genes = the feature-store gene order (<data root>/features/genes.txt,
    docs/INTERFACES.md)."""
    f = protocol.data_root() / "features" / "genes.txt"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: protein-coding gene list (feature store) not built")
    return {g for g in f.read_text().split() if g}


def drop_noncoding(pairs: pd.DataFrame, genes=None) -> pd.DataFrame:
    """Drop pairs involving a gene that is not protein-coding (pseudogenes, ncRNAs, unmapped IDs;
    DESIGN.md §1). `genes` defaults to `protein_coding_genes()`. Returns a new frame with a fresh
    RangeIndex; `.attrs['noncoding_dropped']` = {'pairs': n, 'positives': n_pos (if y present),
    'genes': sorted dropped genes}. Eval pairs never need this (both genes are in the universe)."""
    g = protein_coding_genes() if genes is None else set(genes)
    keep = pairs.gene_a.isin(g).to_numpy() & pairs.gene_b.isin(g).to_numpy()
    out = pairs[keep].reset_index(drop=True)
    bad = pairs.loc[~keep, ["gene_a", "gene_b"]].to_numpy().ravel()
    out.attrs["noncoding_dropped"] = {
        "pairs": int((~keep).sum()),
        "positives": int(pairs.y[~keep].sum()) if "y" in pairs else None,
        "genes": sorted({x for x in bad if x not in g})}
    return out


def all_eval_pairs(split_seed=None) -> pd.DataFrame:
    n = protocol.load()["dev_cv"]["n_folds"]
    return pd.concat([fold_split(split_seed, k)[1] for k in range(n)], ignore_index=True)


# ------------------------------------------------------------------------------------------- inner
@lru_cache(maxsize=1)
def _roles(root: str) -> pd.Series | None:
    f = Path(root) / "ryan_genes.tsv"
    if not f.exists():
        return None
    g = pd.read_csv(f, sep="\t").dropna(subset=["pombase_id"]).groupby("pombase_id").agg(
        q=("is_query", "any"), a=("is_array", "any"))
    return pd.Series(np.where(g.q & g.a, "both", np.where(g.q, "query", "array")), index=g.index)


def inner_splits(train_pairs: pd.DataFrame, seed=None, n_folds: int | None = None,
                 c1_fraction: float | None = None) -> list[S.InnerFold]:
    """Nested inner C1/C2/C3 splits of an outer training set (DESIGN §4 rule 2).

    Returns a list of InnerFold(j, genes=T_j, train=inner-training pairs (neither gene in T_j, not in
    H_j), eval=inner eval rows with `part` (C3-like: both in T_j; C2-like: one in T_j, one seeded copy
    kept; C1-like: H_j) and `inner_fold`, holdout=H_j). Inner-eval rows keep their true hypomorph flags
    (they are stacker TRAINING rows)."""
    P = protocol.load()
    I, St = P["inner"], P["stratification"]
    return S.inner_split_generator(train_pairs, n_folds or I["n_folds"], c1_fraction or I["c1_fraction"],
                                   I["seed"] if seed is None else seed, _roles(str(protocol.data_root())),
                                   St["hub_quantile"], St["n_hit_rate_quantiles"])


gene_grouped_validation = S.gene_grouped_validation


# ------------------------------------------------------------------------------------------- cache
def cache_path(name: str, split_seed=None, fold: int = 0, inner: int | None = None) -> Path:
    """data/processed/cache/<split_seed>/fold<k>/[inner<j>/]<name>.parquet (docs/INTERFACES.md)."""
    d = protocol.cache_root() / str(_seed(split_seed)) / f"fold{fold}"
    if inner is not None:
        d = d / f"inner{inner}"
    return d / f"{name}.parquet"


def cache_save(df: pd.DataFrame, name: str, split_seed=None, fold: int = 0, inner: int | None = None) -> Path:
    p = cache_path(name, split_seed, fold, inner)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    p.with_suffix(".key").write_text(labels_hash())
    return p


def cache_load(name: str, split_seed=None, fold: int = 0, inner: int | None = None) -> pd.DataFrame | None:
    """Cached frame, or None if missing or built from a different dev-label file."""
    p = cache_path(name, split_seed, fold, inner)
    k = p.with_suffix(".key")
    if not (p.exists() and k.exists() and k.read_text() == labels_hash()):
        return None
    return pd.read_parquet(p)
