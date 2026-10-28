"""src/models/stacker.py: the shared stacker fits (binary M2, Huber E-TARGET) used by scripts/run_m2.py,
scripts/run_m2plus.py and src/deliverable.py."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from src import protocol
from src.models import lgbm, stacker

ROOT = protocol.ROOT


def _data(n=3000, f=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    s = (-1.5 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(0, 1.5, n)).astype(float)
    y = protocol.binarize(s).astype(np.int8)
    w = rng.uniform(0.5, 2.0, n)
    fm = rng.random(n) < 0.7
    return X, s, y, w, fm, ~fm


def test_etarget_and_delta_from_protocol():
    s = np.array([-5.0, protocol.threshold(), 0.0, 3.0])
    assert np.array_equal(stacker.etarget(s), [5.0, -protocol.threshold(), 0.0, 0.0])
    assert stacker.huber_delta() == abs(protocol.threshold()) > 0
    with pytest.raises(ValueError):
        stacker.etarget(s, "raw_s2")
    with pytest.raises(ValueError):
        stacker.huber_delta("one")
    with pytest.raises(ValueError):
        stacker.etarget(np.array([np.nan, -1.0]))


def test_seed_params():
    p = stacker.seed_params({"learning_rate": 0.1}, 4)
    assert (p["seed"], p["bagging_seed"], p["feature_fraction_seed"], p["data_random_seed"]) == (4, 5, 6, 7)
    assert p["learning_rate"] == 0.1


def test_huber_fit_uses_huber_objective_target_and_ap_early_stopping():
    X, s, y, w, fm, vm = _data()
    params = lgbm.params_for("default", 2)
    names = [f"f{i}" for i in range(X.shape[1])]
    b, inf = stacker.fit_huber(X[fm], s[fm], w[fm], X[vm], s[vm], y[vm], w[vm], params, 0, names)
    cfg = b.params
    assert cfg["objective"] == "huber" and cfg["alpha"] == stacker.huber_delta() and cfg["metric"] == "None"
    assert inf["objective"] == "huber" and 1 <= inf["best_iteration"] < lgbm.MAX_ROUNDS
    # early-stopping metric = class-weighted AP of the protocol label on the validation rows
    from sklearn.metrics import average_precision_score
    pv = b.predict(X[vm], num_iteration=b.best_iteration)
    assert inf["valid_ap_weighted"] == pytest.approx(average_precision_score(y[vm], pv, sample_weight=w[vm]))
    assert b.best_score["valid"]["weighted_ap"] == pytest.approx(inf["valid_ap_weighted"])
    # the score is the predicted target (identity link): raw == transformed, larger for more negative S2
    assert np.array_equal(stacker.predict(b, X[vm], "huber", 2), pv)
    assert np.corrcoef(pv, stacker.etarget(s[vm]))[0, 1] > 0.3
    # deterministic
    b2, inf2 = stacker.fit_huber(X[fm], s[fm], w[fm], X[vm], s[vm], y[vm], w[vm], params, 0, names)
    assert np.array_equal(b2.predict(X[vm], num_iteration=b2.best_iteration), pv) and inf2 == inf


def test_fit_stacker_dispatch_equals_direct_fits():
    X, s, y, w, fm, vm = _data(seed=1)
    params = lgbm.params_for("default", 2)
    names = [f"f{i}" for i in range(X.shape[1])]
    bh, ih = stacker.fit_stacker("huber", X, y, s, w, fm, vm, params, 3, names)
    bd, idd = stacker.fit_huber(X[fm], s[fm], w[fm], X[vm], s[vm], y[vm], w[vm], params, 3, names, y_fit=y[fm])
    assert ih == idd and np.array_equal(bh.predict(X), bd.predict(X))
    init = np.random.default_rng(2).normal(size=len(y))
    bb, ib = stacker.fit_stacker("binary", X, y, None, w, fm, vm, params, 3, names, init=init)
    bd, idb = stacker.fit_binary(X[fm], y[fm], w[fm], X[vm], y[vm], w[vm], params, 3, names, init_fit=init[fm],
                                 init_valid=init[vm])
    assert ib == idb and np.array_equal(stacker.predict(bb, X, "binary", 2, init), stacker.predict(bd, X, "binary", 2, init))
    assert bb.params["objective"] == "binary"
    with pytest.raises(ValueError, match="init_score"):
        stacker.fit_stacker("huber", X, y, s, w, fm, vm, params, 0, names, init=init)
    with pytest.raises(ValueError, match="label column"):
        stacker.fit_stacker("huber", X, y, None, w, fm, vm, params, 0, names)
    with pytest.raises(ValueError, match="objective"):
        stacker.fit_stacker("poisson", X, y, s, w, fm, vm, params, 0, names)


def test_huber_seeds_change_the_model():
    X, s, y, w, fm, vm = _data(seed=3)
    params = {**lgbm.params_for("default", 2), "bagging_fraction": 0.7, "bagging_freq": 1}
    names = [f"f{i}" for i in range(X.shape[1])]
    p0 = stacker.fit_huber(X[fm], s[fm], w[fm], X[vm], s[vm], y[vm], w[vm], params, 0, names)[0].predict(X[vm])
    p1 = stacker.fit_huber(X[fm], s[fm], w[fm], X[vm], s[vm], y[vm], w[vm], params, 1, names)[0].predict(X[vm])
    assert not np.array_equal(p0, p1)


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_scripts_delegate_to_the_shared_stacker():
    """run_m2plus.fit_stacker_huber / run_m2.fit_stacker are thin wrappers (no second implementation)."""
    import inspect
    MP = _load("run_m2plus")
    src_h = inspect.getsource(MP.fit_stacker_huber)
    assert "stacker.fit_huber(" in src_h and "lgb.train" not in src_h
    assert MP.etarget(np.array([-3.0, 1.0])).tolist() == [3.0, 0.0] and MP.huber_delta() == stacker.huber_delta()
    src_b = inspect.getsource(MP.R.fit_stacker)
    assert "stacker.fit_binary(" in src_b and "lgb.train" not in src_b
    from src import deliverable as DL
    src_d = inspect.getsource(DL.TrainedModel._fit_m2_stack)
    assert "stacker.fit_stacker(" in src_d and "lgb.train" not in src_d
