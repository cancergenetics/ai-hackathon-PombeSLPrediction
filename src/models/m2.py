"""M2: nested-stacked LightGBM on label-free + Ryan-derived (RYAN-fold, NBR, gene-propensity) features
(DESIGN.md §4 rule 2, §5 M2).

Feature families (each one ablation unit; `family_of(name)`):
* LF    label-free pair features from src/pairs.assemble (all manifest groups + HYPO), as in M1.
* RYAN  RYAN-fold features of src/ryanfeat: seen flags, B1 logit and beta, MF score, profile correlation,
        gene mean -S.
* NBR   similarity-transfer features of src/ryanfeat (`nbr_*`).
* GP    gene-propensity model (new here; Ryan-derived, therefore nested exactly like RYAN/NBR):
        within a training pair set T
          1. beta_g = B1 propensity of every training gene (L2-shrunk joint additive logit, fitted on T);
          2. gene-level inputs x_g = label-free gene tables (feature store gene/*.parquet) + leave-own-pairs-out
             NBR gene rates per similarity source (similarity-weighted mean training hit rate of g's
             neighbours c, where c's rate excludes the pair (c, g) itself) + their cross-source mean;
          3. a small regressor beta ~ x (default: mean of a LightGBM regressor and a ridge regression on
             median-imputed, standardised inputs + missing indicators, RidgeCV alpha) is fitted with K-fold
             cross-fitting over the training genes:
             a training gene's beta_hat comes from the model whose training genes exclude it (out-of-fold),
             every other gene gets the mean prediction of the K models;
          4. pair features: gp_hat_{min,max,sum} (beta_hat), gp_beta_{min,max} (actual beta of seen genes,
             NaN if unseen; min is NaN if either gene is unseen), gp_best_{min,max,sum} (beta if seen else
             beta_hat).
        Nothing about the query pairs except their gene IDs is used (same contract as src/ryanfeat).

`M2Builder` = RyanFeatureBuilder + GP features; it plugs into src/stacking.nested_tables/inner_rows/
outer_rows (builder=...). `make_builder(gene_table, gp_cfg)` binds the label-free gene table.
`fold_tables_m2(...)` = cached nested tables for one outer fold (training pairs passed through
src.data.drop_noncoding first), cache name `m2feat_<hash>` under src.data cache helpers.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import data, protocol
from src.ryanfeat import RyanFeatureBuilder, SimStore, _clean_train, cfg_hash, resolve_cfg

M2_FEATURE_VERSION = "m2feat-v2"
RYAN_COLS = ["seen_min", "seen_max", "b1_logit", "beta_min", "beta_max", "mf_score", "prof_corr",
             "negs_mean_min", "negs_mean_max"]
GP_COLS = ["gp_hat_min", "gp_hat_max", "gp_hat_sum", "gp_beta_min", "gp_beta_max",
           "gp_best_min", "gp_best_max", "gp_best_sum"]
FAMILIES = ("LF", "RYAN", "NBR", "GP")

GP_DEFAULT = {
    "k_folds": 5,
    "seed": 0,
    "nbr_gene": True,          # add leave-own-pairs-out NBR gene rates as GP inputs
    "regressor": "mean",       # 'lgbm' | 'ridge' | 'mean' (average of the two predictions)
    "ridge_alphas": [float(a) for a in np.logspace(-1, 4, 20)],
    "rounds": 300,
    "params": dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
                   feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbosity=-1, deterministic=True, force_row_wise=True, num_threads=2),
}


def family_of(name: str) -> str:
    if name.startswith("gp_"):
        return "GP"
    if name.startswith("nbr_"):
        return "NBR"
    if name in RYAN_COLS:
        return "RYAN"
    return "LF"


def resolve_gp(gp: dict | None) -> dict:
    g = {**GP_DEFAULT, **(gp or {})}
    g["params"] = {**GP_DEFAULT["params"], **((gp or {}).get("params") or {})}
    return g


# =========================================================================================== gene table
@lru_cache(maxsize=2)
def _gene_table(fdir: str) -> pd.DataFrame:
    from src import pairs as PA
    spec = PA.load_spec()
    parts = [pd.DataFrame(spec.table(rel, cols), index=spec.genes, columns=cols)
             for _, rel, cols in spec.gene_tables]
    return pd.concat(parts, axis=1)


def gene_table() -> pd.DataFrame:
    """All label-free gene-level columns of the feature store (genes.txt index, NaN = missing)."""
    return _gene_table(str(protocol.data_root() / "features"))


def gene_table_hashes() -> dict:
    from src import pairs as PA
    spec = PA.load_spec()
    return {rel: spec.manifest[rel].get("sha256", "") for _, rel, _ in spec.gene_tables}


# =========================================================================================== GP model
def _ridge_design(A: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Median-impute (medians of the fitting rows), add missing indicators, standardise (fitting rows)."""
    miss = np.isnan(A)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(A[fit_rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Z = np.where(miss, med, A)
    keep = miss.any(axis=0)
    Z = np.hstack([Z, miss[:, keep].astype(float)])
    mu, sd = Z[fit_rows].mean(0), Z[fit_rows].std(0)
    sd[sd == 0] = 1.0
    return (Z - mu) / sd


def _fit_predict(g: dict, Xv: np.ndarray, fit_rows: np.ndarray, y: np.ndarray, w, pred_rows: np.ndarray,
                 seed: int) -> np.ndarray:
    preds = []
    if g["regressor"] in ("lgbm", "mean"):
        ds = lgb.Dataset(Xv[fit_rows].astype(np.float32), label=y, weight=w, free_raw_data=True)
        m = lgb.train({**g["params"], "seed": seed}, ds, num_boost_round=g["rounds"])
        preds.append(m.predict(Xv[pred_rows].astype(np.float32)))
    if g["regressor"] in ("ridge", "mean"):
        from sklearn.linear_model import RidgeCV
        Z = _ridge_design(Xv.astype(float), fit_rows)
        with np.errstate(all="ignore"):             # spurious FP flags from macOS Accelerate BLAS
            r = RidgeCV(alphas=g["ridge_alphas"]).fit(Z[fit_rows], y, sample_weight=w)
            preds.append(r.predict(Z[pred_rows]))
    if not preds:
        raise ValueError(f"unknown GP regressor {g['regressor']}")
    return np.mean(preds, axis=0)


def fit_gene_propensity(beta: pd.Series, X: pd.DataFrame, gp: dict | None = None,
                        weight: pd.Series | None = None) -> tuple[pd.Series, dict]:
    """Cross-fitted gene-propensity regressor. beta: target for training genes (index subset of X.index).
    Returns (beta_hat over X.index, info). A training gene's beta_hat is predicted by the one model whose
    training genes exclude it; every other gene gets the mean of the K models."""
    g = resolve_gp(gp)
    tg = X.index[X.index.isin(beta.index)]
    other = X.index[~X.index.isin(beta.index)]
    rng = np.random.default_rng(g["seed"])
    fold = rng.permutation(len(tg)) % g["k_folds"]
    Xv = X.to_numpy(np.float64)
    pos = pd.Series(np.arange(len(X)), index=X.index)
    ptg, pot = pos[tg].to_numpy(), pos[other].to_numpy()
    out = pd.Series(np.nan, index=X.index, dtype=float)
    acc = np.zeros(len(other))
    info = {"n_train_genes": int(len(tg)), "n_other_genes": int(len(other)), "oof_r": None,
            "regressor": g["regressor"]}
    y_all = beta.reindex(tg).to_numpy(float)
    w_all = None if weight is None else weight.reindex(tg).to_numpy(float)
    for k in range(g["k_folds"]):
        fit_g = fold != k
        rows = np.r_[ptg[~fit_g], pot]
        p = _fit_predict(g, Xv, ptg[fit_g], y_all[fit_g], None if w_all is None else w_all[fit_g], rows,
                         g["seed"] + k)
        out.iloc[ptg[~fit_g]] = p[:(~fit_g).sum()]
        acc += p[(~fit_g).sum():]
    if len(other):
        out.iloc[pot] = acc / g["k_folds"]
    if len(tg) > 2:
        info["oof_r"] = float(np.corrcoef(out[tg].to_numpy(), y_all)[0, 1])
    return out, info


def loo_nbr_gene_rate(W, Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Similarity-weighted mean of neighbours' training hit rates, each neighbour's rate excluding its
    pair with the gene itself: r_g = sum_c W[g,c] (s_c - Y[g,c])/(n_c - M[g,c]) / sum_c W[g,c] over
    neighbours with n_c - M[g,c] > 0 (NaN if none). W: csr (n x n), Y/M: dense symmetric (n x n)."""
    s = Y.sum(1)
    n = M.sum(1)
    Wc = W.tocoo()
    r, c, w = Wc.row, Wc.col, Wc.data
    den = n[c] - M[r, c]
    ok = den > 0
    val = np.zeros(len(r))
    val[ok] = (s[c][ok] - Y[r, c][ok]) / den[ok]
    num = np.bincount(r[ok], weights=w[ok] * val[ok], minlength=W.shape[0])
    sw = np.bincount(r[ok], weights=w[ok], minlength=W.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, num / sw, np.nan)


class M2Builder(RyanFeatureBuilder):
    """RyanFeatureBuilder + gene-propensity (GP) features. `gene_table` / `gp_cfg` are bound by
    make_builder; hooks for planted mutants (tests only): `gp_oof` (False = in-sample regressor)."""

    gene_table: pd.DataFrame | None = None
    gp_cfg: dict | None = None
    gp_oof = True

    def build(self) -> pd.DataFrame:
        F = super().build()
        gp = resolve_gp(self.gp_cfg)
        tr = _clean_train(self.stat_pairs("gp"))
        beta = self.info["b1"]["beta"]
        q = self.query
        qa, qb = q.gene_a.to_numpy(object), q.gene_b.to_numpy(object)
        idx = self.info.get("idx")
        if idx is None:
            idx = pd.Index(sorted(set(tr.gene_a) | set(tr.gene_b) | set(qa) | set(qb)))
        X = pd.DataFrame(index=idx)
        if self.gene_table is not None:
            X = self.gene_table.reindex(idx)
        if gp["nbr_gene"] and "W" in self.info:
            n = len(idx)
            ta, tb = idx.get_indexer(tr.gene_a), idx.get_indexer(tr.gene_b)
            Y = np.zeros((n, n), np.float64)
            M = np.zeros((n, n), np.float64)
            Y[ta, tb] = Y[tb, ta] = tr.y.to_numpy(float)
            M[ta, tb] = M[tb, ta] = 1.0
            cols = {}
            for s, W in self.info["W"].items():
                cols[f"gpin_nbr_{s}"] = loo_nbr_gene_rate(W, Y, M)
            nb = pd.DataFrame(cols, index=idx)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                nb["gpin_nbr_mean"] = np.nanmean(nb.to_numpy(), axis=1)
            X = pd.concat([X, nb], axis=1)
        if X.shape[1] == 0:
            raise ValueError("GP model has no inputs (no gene table and no NBR sources)")
        if self.gp_oof:
            bhat, info = fit_gene_propensity(beta, X, gp)
        else:                                   # planted mutant: in-sample (each gene sees its own beta)
            bhat, info = _in_sample_gp(beta, X, gp)
        self.info["gp"] = {"beta_hat": bhat, "X_columns": list(X.columns), **info}
        ha, hb = bhat.reindex(qa).to_numpy(), bhat.reindex(qb).to_numpy()
        seen = set(beta.index)
        sa, sb = np.isin(qa, list(seen)), np.isin(qb, list(seen))
        ba = np.where(sa, beta.reindex(qa).to_numpy(), np.nan)
        bb = np.where(sb, beta.reindex(qb).to_numpy(), np.nan)
        best_a, best_b = np.where(sa, ba, ha), np.where(sb, bb, hb)
        out = {"gp_hat_min": np.minimum(ha, hb), "gp_hat_max": np.maximum(ha, hb), "gp_hat_sum": ha + hb,
               "gp_beta_min": np.minimum(ba, bb), "gp_beta_max": np.fmax(ba, bb),
               "gp_best_min": np.minimum(best_a, best_b), "gp_best_max": np.maximum(best_a, best_b),
               "gp_best_sum": best_a + best_b}
        for k, v in out.items():
            F[k] = np.asarray(v, dtype=np.float32)
        return F


def _in_sample_gp(beta, X, gp):
    """Planted mutant (tests only): one regressor fitted on ALL training genes predicts every gene."""
    g = resolve_gp(gp)
    pos = pd.Series(np.arange(len(X)), index=X.index)
    tg = X.index[X.index.isin(beta.index)]
    p = _fit_predict(g, X.to_numpy(np.float64), pos[tg].to_numpy(), beta.reindex(tg).to_numpy(float), None,
                     np.arange(len(X)), g["seed"])
    return pd.Series(p, index=X.index), {}


def make_builder(gene_tab: pd.DataFrame | None, gp_cfg: dict | None = None, base=M2Builder) -> type:
    return type("BoundM2Builder", (base,), {"gene_table": gene_tab, "gp_cfg": resolve_gp(gp_cfg)})


# =========================================================================================== nested tables
def m2_cache_name(cfg: dict, sims: SimStore, gp_cfg: dict | None) -> str:
    payload = {"v": M2_FEATURE_VERSION, "ryanfeat": cfg_hash(cfg, sims), "gp": resolve_gp(gp_cfg),
               "gene_tables": gene_table_hashes(), "train_filter": "drop_noncoding"}
    return "m2feat_" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def fold_pairs(split_seed, fold):
    """(training pairs with non-protein-coding genes dropped, eval pairs, drop info)."""
    tr, ev = data.fold_split(split_seed, fold)
    trd = data.drop_noncoding(tr)
    return trd, ev, trd.attrs["noncoding_dropped"]


def fold_tables_m2(split_seed=None, fold: int = 0, sims: SimStore | None = None, cfg: dict | None = None,
                   gp_cfg: dict | None = None, use_cache: bool = True, timings: dict | None = None):
    """Cached nested M2 tables for outer fold `fold`: (stacker training rows, outer eval rows, inner folds,
    drop info). Stacker rows: features from each inner fold's training pairs; eval rows: from all outer
    training pairs (src/stacking)."""
    import time
    from src import stacking
    sims = sims if sims is not None else SimStore()
    cfg = resolve_cfg(cfg)
    name = m2_cache_name(cfg, sims, gp_cfg)
    train, ev, drop = fold_pairs(split_seed, fold)
    inner = data.inner_splits(train)
    builder = make_builder(gene_table(), gp_cfg)
    parts = []
    timings = timings if timings is not None else {}
    for f in inner:
        x = data.cache_load(name, split_seed, fold, inner=f.j) if use_cache else None
        if x is None:
            t0 = time.perf_counter()
            x = stacking.inner_rows(f, sims, cfg, builder)
            timings[f"inner{f.j}"] = round(time.perf_counter() - t0, 1)
            data.cache_save(x, name, split_seed, fold, inner=f.j)
        parts.append(x)
    ev_rows = data.cache_load(name, split_seed, fold) if use_cache else None
    if ev_rows is None:
        t0 = time.perf_counter()
        ev_rows = stacking.outer_rows(train, ev, sims, cfg, builder)
        timings["outer"] = round(time.perf_counter() - t0, 1)
        data.cache_save(ev_rows, name, split_seed, fold)
    return pd.concat(parts, ignore_index=True), ev_rows, inner, drop
