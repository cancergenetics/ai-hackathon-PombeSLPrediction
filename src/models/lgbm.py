"""LightGBM binary classifier on assembled pair features (M1, DESIGN.md §5).

* `gene_holdout(pairs, frac, seed)` -> (fit_mask, valid_mask): a gene-grouped inner validation split of an
  OUTER TRAINING set: hold out `frac` of its genes; validation = pairs touching a held-out gene; fitting
  rows = pairs touching none (pairs between a held-out and a kept gene are validation only).
  Early stopping never sees the reported fold.
* `fit(X, y, fit_mask, valid_mask, params, seed)` -> (Booster, info) with early stopping on validation AP.
* `CONFIGS`: the default plus <= 5 alternatives for the capped fold-0 inner-split check (DESIGN §3 caps).
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DEFAULT = dict(objective="binary", learning_rate=0.05, num_leaves=63, feature_fraction=0.7, bagging_fraction=0.8,
               bagging_freq=1, min_data_in_leaf=200, lambda_l2=1.0, max_bin=255)
MAX_ROUNDS = 2000
EARLY_STOP = 100
VALID_GENE_FRAC = 0.15

# <= 6 configs in total (DESIGN §3: no large search for M1); selected on the fold-0 inner split only.
CONFIGS = {
    "default": {},
    "leaves31_mdl500": dict(num_leaves=31, min_data_in_leaf=500),
    "leaves127_mdl100": dict(num_leaves=127, min_data_in_leaf=100),
    "ff0.4": dict(feature_fraction=0.4),
    "mdl1000_l2_10": dict(min_data_in_leaf=1000, lambda_l2=10.0),
    "lr0.02": dict(learning_rate=0.02),
}


def params_for(name: str = "default", n_jobs: int = 8, **override) -> dict:
    return {**DEFAULT, **CONFIGS[name], "num_threads": n_jobs, "verbosity": -1, "force_col_wise": True,
            "metric": "average_precision", **override}


def gene_holdout(pairs: pd.DataFrame, frac: float = VALID_GENE_FRAC, seed=0) -> tuple[np.ndarray, np.ndarray]:
    genes = np.sort(pd.unique(np.r_[pairs.gene_a.to_numpy(), pairs.gene_b.to_numpy()]))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(genes, int(round(frac * len(genes))), replace=False))
    touch = pairs.gene_a.isin(held).to_numpy() | pairs.gene_b.isin(held).to_numpy()
    return ~touch, touch


def safe_names(names) -> list:
    out = [re.sub(r"[^A-Za-z0-9_.\-]", "_", n) for n in names]
    assert len(set(out)) == len(out), "feature names collide after sanitising"
    return out


def fit(X: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, valid_mask: np.ndarray, params: dict, seed: int,
        feature_names=None, weight: np.ndarray | None = None) -> tuple[lgb.Booster, dict]:
    p = {**params, "seed": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2,
         "data_random_seed": seed + 3}
    names = safe_names(feature_names) if feature_names is not None else "auto"
    dtr = lgb.Dataset(X[fit_mask], label=y[fit_mask], weight=None if weight is None else weight[fit_mask],
                      feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=dtr, free_raw_data=True)
    ev = {}
    b = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.record_evaluation(ev)])
    pv = b.predict(X[valid_mask], num_iteration=b.best_iteration)
    info = {"best_iteration": int(b.best_iteration), "valid_ap": float(average_precision_score(y[valid_mask], pv)),
            "n_fit": int(fit_mask.sum()), "n_valid": int(valid_mask.sum()),
            "pos_fit": int(y[fit_mask].sum()), "pos_valid": int(y[valid_mask].sum())}
    return b, info


def predict(b: lgb.Booster, X: np.ndarray, n_jobs: int = 8) -> np.ndarray:
    return b.predict(X, num_iteration=b.best_iteration, num_threads=n_jobs)


def gain_importance(b: lgb.Booster) -> np.ndarray:
    g = b.feature_importance(importance_type="gain", iteration=b.best_iteration).astype(float)
    return g / g.sum() if g.sum() > 0 else g
