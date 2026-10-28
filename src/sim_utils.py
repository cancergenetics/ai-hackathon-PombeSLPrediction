"""Small numeric helpers shared by the tier-2 feature builders (scripts/features/build_{expr,koprot,loc,seq,cons}.py).

All functions work on a sub-matrix for the n genes that have data (rows of X / B, positions `idx` in
genes.txt) and scatter the result into a float32 G x G matrix with NaN elsewhere and on the diagonal,
as required by docs/INTERFACES.md.  No file I/O here; unit-tested in tests/features/test_tier2_features.py.
"""
from __future__ import annotations

import numpy as np

_QUIET = dict(over="ignore", divide="ignore", invalid="ignore")   # spurious Apple Accelerate matmul flags


def scatter(sub: np.ndarray, idx: np.ndarray, G: int, fill: float = np.nan) -> np.ndarray:
    """Place an n x n block (rows/cols = genes.txt positions `idx`) into a G x G float32 matrix."""
    idx = np.asarray(idx)
    assert sub.shape == (len(idx), len(idx)), (sub.shape, len(idx))
    assert len(np.unique(idx)) == len(idx), "duplicate gene positions"
    M = np.full((G, G), fill, dtype=np.float32)
    M[np.ix_(idx, idx)] = sub.astype(np.float32, copy=False)
    np.fill_diagonal(M, np.nan)
    return M


def _symmetrize(S: np.ndarray) -> np.ndarray:
    S = (S + S.T) / 2                                  # removes float round-off asymmetry of X @ X.T
    np.fill_diagonal(S, np.nan)
    return S


def row_pearson(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between the rows of a complete (no NaN) matrix X (n x d).
    Returns (S, ok): S is n_ok x n_ok (NaN diagonal), ok = rows with non-zero variance (others dropped)."""
    X = np.asarray(X, dtype=np.float64)
    assert np.isfinite(X).all(), "row_pearson needs a complete matrix (impute first)"
    V = X - X.mean(1, keepdims=True)
    nrm = np.linalg.norm(V, axis=1)
    ok = nrm > 1e-12
    V = V[ok] / nrm[ok, None]
    with np.errstate(**_QUIET):
        S = np.clip(V @ V.T, -1.0, 1.0)
    return _symmetrize(S), ok


def row_cosine(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity between rows of X (n x d). Rows with zero norm are dropped (ok mask)."""
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1)
    ok = nrm > 1e-12
    V = X[ok] / nrm[ok, None]
    with np.errstate(**_QUIET):
        S = np.clip(V @ V.T, -1.0, 1.0)
    return _symmetrize(S), ok


def incidence_overlap(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B: n x k 0/1 incidence (gene x item).  Returns (intersection counts, Jaccard) both n x n float64;
    Jaccard is NaN where the union is empty (both genes have no item); diagonals NaN."""
    B = np.asarray(B, dtype=np.float64)
    with np.errstate(**_QUIET):
        inter = B @ B.T
    sz = B.sum(1)
    union = sz[:, None] + sz[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / np.where(union > 0, union, 1), np.nan)
    inter = inter.copy()
    np.fill_diagonal(inter, np.nan)
    np.fill_diagonal(jac, np.nan)
    return inter, jac


def mutual_rank_score(S: np.ndarray) -> np.ndarray:
    """Mutual-rank normalisation of a symmetric n x n similarity (NaN diagonal; no other NaN).
    rank_ij = rank of j among i's partners by descending S (1 = best); MR_ij = sqrt(rank_ij * rank_ji);
    returned score = 1 - log(MR) / log(n - 1) in [0, 1] (1 = mutual best partners)."""
    n = S.shape[0]
    A = np.where(np.isnan(S), -np.inf, S)
    np.fill_diagonal(A, -np.inf)
    order = np.argsort(-A, axis=1, kind="stable")
    R = np.empty((n, n), dtype=np.float64)
    rows = np.arange(n)[:, None]
    R[rows, order] = np.arange(1, n + 1)[None, :]
    MR = np.sqrt(R * R.T)
    out = 1.0 - np.log(MR) / np.log(n - 1)
    np.fill_diagonal(out, np.nan)
    return np.clip(out, 0.0, 1.0)


def dist_summary(M: np.ndarray, rows: np.ndarray | None = None, max_n: int = 2_000_000, seed: int = 0) -> dict:
    """Distribution of the finite upper-triangle values of M (optionally restricted to `rows` x `rows`).
    Flags heavy skew (|skewness| > 2, or > 50% of values within 1% of the range of the minimum)."""
    sub = np.asarray(M if rows is None else M[np.ix_(rows, rows)])
    iu = np.triu_indices(sub.shape[0], 1)
    v = sub[iu]
    v = v[np.isfinite(v)].astype(np.float64)
    if v.size == 0:
        return {"n": 0}
    if v.size > max_n:
        v = np.random.default_rng(seed).choice(v, max_n, replace=False)
    m, s = v.mean(), v.std()
    skew = float(((v - m) ** 3).mean() / s ** 3) if s > 0 else 0.0
    rng_ = v.max() - v.min()
    frac_at_min = float((v <= v.min() + 0.01 * rng_).mean()) if rng_ > 0 else 1.0
    q = np.quantile(v, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {"n_sampled": int(v.size), "mean": float(m), "sd": float(s), "skewness": skew,
            "quantiles": {k: float(x) for k, x in zip(["min", "q01", "q05", "q25", "median", "q75", "q95", "q99",
                                                       "max"], q)},
            "frac_at_min": frac_at_min, "n_distinct_sampled": int(min(np.unique(v).size, 10_000)),
            "heavily_skewed": bool(abs(skew) > 2 or frac_at_min > 0.5)}
