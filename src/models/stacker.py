"""LightGBM stacker fits shared by dev CV (scripts/run_m2.py, scripts/run_m2plus.py) and full-training-set
retrains (src/deliverable.py, hence scripts/predict_all_pairs.py and scripts/lockbox_eval.py).

Both objectives fit on the stacker rows of one inner split (fit rows / validation rows chosen by the caller,
src.stacking.stacker_folds) with class-weighted rows and early stopping on the validation rows:

* `fit_binary`  objective = binary on the protocol label y (params' metric, default class-weighted AP);
                optional init_score (B1 logit). Score = raw margin (+ init).
* `fit_huber`   E-TARGET (docs/m2plus.md step 4, config/m2plus.yaml `etarget`): Huber regression on
                t = -min(S2, 0) (`etarget`) with delta = |protocol label threshold| (`huber_delta`); early
                stopping on the class-weighted binary AP of the protocol label on the validation rows (custom
                metric, LightGBM metric disabled). Score = predicted t.

`fit_stacker(objective, ...)` dispatches. Seeds: `seed_params` (seed, bagging_seed = seed+1,
feature_fraction_seed = seed+2, data_random_seed = seed+3), as in every M2 run.
The callers pass already-sliced matrices; these functions do no feature selection or slicing, so a dev-CV fold
and a deliverable retrain on the same rows give bit-identical boosters.
"""
from __future__ import annotations

import numpy as np

from src import protocol

OBJECTIVES = ("binary", "huber")
TARGETS = ("neg_min_s2_0",)
DELTAS = ("label_threshold_abs",)


def seed_params(params: dict, seed: int) -> dict:
    s = int(seed)
    return {**params, "seed": s, "bagging_seed": s + 1, "feature_fraction_seed": s + 2, "data_random_seed": s + 3}


def etarget(s: np.ndarray, target: str = "neg_min_s2_0") -> np.ndarray:
    """E-TARGET regression target from the protocol label column (S2): t = -min(S2, 0)."""
    if target != "neg_min_s2_0":
        raise ValueError(f"unknown E-TARGET target {target!r}; known {TARGETS}")
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("E-TARGET: non-finite label values in the stacker rows")
    return -np.minimum(s, 0.0)


def huber_delta(delta: str = "label_threshold_abs") -> float:
    """Huber transition point; 'label_threshold_abs' = |protocol label threshold| (read from protocol.yaml)."""
    if delta != "label_threshold_abs":
        raise ValueError(f"unknown Huber delta rule {delta!r}; known {DELTAS}")
    return abs(float(protocol.threshold()))


def _wap(y, p, w) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p, sample_weight=w))


def fit_binary(X_fit, y_fit, w_fit, X_valid, y_valid, w_valid, params: dict, seed: int, feature_names,
               init_fit=None, init_valid=None, max_rounds: int | None = None, early_stop: int | None = None):
    """Binary LightGBM stacker (M2). Returns (booster, info)."""
    import lightgbm as lgb
    from src.models import lgbm
    p = seed_params(params, seed)
    dtr = lgb.Dataset(X_fit, label=y_fit, weight=w_fit, init_score=init_fit, feature_name=list(feature_names),
                      free_raw_data=True)
    dva = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, init_score=init_valid, reference=dtr,
                      free_raw_data=True)
    b = lgb.train(p, dtr, num_boost_round=max_rounds or lgbm.MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(early_stop or lgbm.EARLY_STOP, verbose=False)])
    pv = b.predict(X_valid, num_iteration=b.best_iteration, raw_score=True) + (0.0 if init_valid is None else init_valid)
    info = {"objective": "binary", "best_iteration": int(b.best_iteration),
            "valid_ap_weighted": _wap(y_valid, pv, w_valid), "n_fit": int(len(y_fit)), "n_valid": int(len(y_valid)),
            "pos_fit": int(np.sum(y_fit)), "pos_valid": int(np.sum(y_valid))}
    return b, info


def fit_huber(X_fit, s_fit, w_fit, X_valid, s_valid, y_valid, w_valid, params: dict, seed: int, feature_names,
              target: str = "neg_min_s2_0", delta: str = "label_threshold_abs", y_fit=None,
              max_rounds: int | None = None, early_stop: int | None = None):
    """Huber regression stacker on etarget(S2) (E-TARGET). s_* = protocol label column (S2) of the rows, y_valid =
    protocol label of the validation rows (early-stopping metric). Returns (booster, info)."""
    import lightgbm as lgb
    from src.models import lgbm
    d = huber_delta(delta)
    p = {**params, "objective": "huber", "alpha": d, "metric": "None"}
    p = seed_params(p, seed)
    dtr = lgb.Dataset(X_fit, label=etarget(s_fit, target), weight=w_fit, feature_name=list(feature_names),
                      free_raw_data=True)
    dva = lgb.Dataset(X_valid, label=etarget(s_valid, target), weight=w_valid, reference=dtr, free_raw_data=True)

    def wap(preds, _ds):
        return "weighted_ap", _wap(y_valid, preds, w_valid), True

    b = lgb.train(p, dtr, num_boost_round=max_rounds or lgbm.MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"],
                  feval=wap, callbacks=[lgb.early_stopping(early_stop or lgbm.EARLY_STOP, verbose=False)])
    pv = b.predict(X_valid, num_iteration=b.best_iteration)
    info = {"objective": "huber", "best_iteration": int(b.best_iteration),
            "valid_ap_weighted": _wap(y_valid, pv, w_valid), "n_fit": int(len(s_fit)), "n_valid": int(len(y_valid)),
            "pos_fit": None if y_fit is None else int(np.sum(y_fit)), "pos_valid": int(np.sum(y_valid)),
            "huber_delta": d, "target": target}
    return b, info


def fit_stacker(objective: str, X: np.ndarray, y: np.ndarray, s: np.ndarray | None, w: np.ndarray,
                fit_rows: np.ndarray, valid_rows: np.ndarray, params: dict, seed: int, feature_names,
                init: np.ndarray | None = None, etarget_cfg: dict | None = None):
    """Fit one stacker seed on boolean row masks of the full stacker matrix X (rows x selected features).
    objective 'binary' (init = optional init score per row) or 'huber' (s = protocol label column per row;
    etarget_cfg = {target, delta}). Returns (booster, info)."""
    if objective == "binary":
        return fit_binary(X[fit_rows], y[fit_rows], w[fit_rows], X[valid_rows], y[valid_rows], w[valid_rows],
                          params, seed, feature_names,
                          init_fit=None if init is None else init[fit_rows],
                          init_valid=None if init is None else init[valid_rows])
    if objective == "huber":
        if init is not None:
            raise ValueError("init_score is not defined for the Huber stacker")
        if s is None:
            raise ValueError("the Huber stacker needs the label column s")
        e = etarget_cfg or {}
        return fit_huber(X[fit_rows], s[fit_rows], w[fit_rows], X[valid_rows], s[valid_rows], y[valid_rows],
                         w[valid_rows], params, seed, feature_names, target=e.get("target", "neg_min_s2_0"),
                         delta=e.get("delta", "label_threshold_abs"), y_fit=y[fit_rows])
    raise ValueError(f"unknown stacker objective {objective!r}; known {OBJECTIVES}")


def predict(booster, X: np.ndarray, objective: str, threads: int, init: np.ndarray | None = None) -> np.ndarray:
    """Stacker score: raw margin (binary, + init) or predicted target (huber; identity link, raw = predicted)."""
    s = booster.predict(X, num_iteration=booster.best_iteration, raw_score=True, num_threads=threads)
    if init is not None:
        if objective != "binary":
            raise ValueError("init_score only for the binary stacker")
        s = s + init
    return s
