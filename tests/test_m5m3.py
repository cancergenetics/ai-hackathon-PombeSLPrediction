"""Tests for the M5 rank-average rules (scripts/run_m5.py) and the M3 network (src/models/m3.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_m5 as M5  # noqa: E402
from src.models import m3  # noqa: E402


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"fold": np.repeat([0, 1, 2], n // 3),
                       "part": rng.choice(["C1", "C2", "C3"], n),
                       "INC": rng.normal(size=n), "B": rng.normal(size=n), "C": rng.integers(0, 5, n).astype(float)})
    df.loc[df.fold == 1, "INC"] += 100.0          # fold-level shift must not matter (ranks per fold)
    return df


def test_pooled_single_member_is_within_fold_pct_rank():
    df = _frame()
    s = M5.rank_average(df, ["INC"], "pooled", "INC")
    for f, g in df.groupby("fold"):
        np.testing.assert_allclose(s[g.index], M5.pct_rank(g.INC.to_numpy()))
    assert s.max() <= 1.0 and s.min() > 0


def test_pooled_member_order_irrelevant_and_equal_weights():
    df = _frame()
    a = M5.rank_average(df, ["INC", "B", "C"], "pooled", "INC")
    b = M5.rank_average(df, ["C", "INC", "B"], "pooled", "INC")
    np.testing.assert_allclose(a, b)
    g = df[df.fold == 0]
    exp = (M5.pct_rank(g.INC.to_numpy()) + M5.pct_rank(g.B.to_numpy()) + M5.pct_rank(g.C.to_numpy())) / 3
    np.testing.assert_allclose(a[g.index], exp)


def test_class_preserving_keeps_incumbent_scores_per_class():
    df = _frame()
    s = M5.rank_average(df, ["INC", "B", "C"], "class_preserving", "INC")
    for _, g in df.groupby(["fold", "part"]):
        np.testing.assert_array_equal(np.sort(s[g.index]), np.sort(g.INC.to_numpy()))
    # with the incumbent alone the rule is the identity
    np.testing.assert_array_equal(M5.rank_average(df, ["INC"], "class_preserving", "INC"), df.INC.to_numpy())
    # a constant member (as B4 / NBR on C3) leaves the order unchanged
    df["K"] = 1.0
    np.testing.assert_array_equal(M5.rank_average(df, ["INC", "K"], "class_preserving", "INC"), df.INC.to_numpy())


def test_unknown_rule():
    with pytest.raises(ValueError):
        M5.rank_average(_frame(), ["INC"], "mean", "INC")


def test_preprocessor_fit_rows_only_and_indicators():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4)).astype(np.float32)
    X[:50, 1] = np.nan
    X[:, 3] = np.nan                                # all-NaN column
    pp = m3.Preprocessor(n_quantiles=100, subsample=1000, clip=5.0).fit(X[:400])
    Z = pp.transform(X)
    assert Z.shape == (500, 4 + 2) and pp.n_out == 6
    assert np.isfinite(Z).all() and np.abs(Z).max() <= 5.0
    np.testing.assert_array_equal(Z[:, 4], np.isnan(X[:, 1]).astype(np.float32))
    np.testing.assert_array_equal(Z[:, 5], 1.0)
    # an extreme value outside the fit range is clipped, not extrapolated
    Y = X.copy()
    Y[0, 0] = 1e6
    assert pp.transform(Y)[0, 0] <= 5.0


def _synthetic(n=6000, G=300, seed=0):
    rng = np.random.default_rng(seed)
    gx = rng.normal(size=(G, 6)).astype(np.float32)
    prop = gx[:, 0]
    ia, ib = rng.integers(0, G, n), rng.integers(0, G, n)
    P = np.c_[np.minimum(prop[ia], prop[ib]) + rng.normal(size=n) * 0.0, rng.normal(size=(n, 3))].astype(np.float32)
    logit = 1.5 * (prop[ia] + prop[ib]) - 4
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    s = np.where(y > 0, -3.0 - rng.random(n), rng.normal(size=n))
    return gx, ia, ib, P, y, s


def test_m3_symmetric_and_learns():
    import torch
    torch.set_num_threads(1)
    gx, ia, ib, P, y, s = _synthetic()
    t = -np.minimum(s, 0)
    w = np.ones(len(y))
    f, v = slice(0, 4500), slice(4500, None)
    cfg = {"max_epochs": 30, "patience": 30, "batch": 256, "lr": 3e-3, "gene_tower": [16, 8], "pair_mlp": [32, 8]}
    net, info = m3.fit_m3(gx, ia[f], ib[f], P[f], t[f], y[f], w[f], ia[v], ib[v], P[v], y[v], w[v], cfg, 0, 2.3)
    prev = y[v].mean()
    assert info["valid_ap_weighted"] > 2 * prev, (info["valid_ap_weighted"], prev)
    s1 = m3.predict_m3(net, gx, ia[v], ib[v], P[v])
    s2 = m3.predict_m3(net, gx, ib[v], ia[v], P[v])
    assert np.max(np.abs(s1 - s2)) <= 1e-5
    # determinism given the seed
    net2, info2 = m3.fit_m3(gx, ia[f], ib[f], P[f], t[f], y[f], w[f], ia[v], ib[v], P[v], y[v], w[v], cfg, 0, 2.3)
    assert info2["valid_ap_weighted"] == info["valid_ap_weighted"]


def test_m3_symmetry_check_catches_asymmetric_encoder(monkeypatch):
    """Planted mutant: concatenating (e_a, e_b) instead of (e_a + e_b, e_a * e_b) breaks score(a,b)==score(b,a)."""
    import torch
    torch.set_num_threads(1)
    gx, ia, ib, P, y, s = _synthetic(n=500)
    cfg = m3.resolve_cfg({"gene_tower": [16, 8], "pair_mlp": [32, 8]})
    net = m3.build_net(gx.shape[1], P.shape[1], cfg)

    def fwd(ga, gb, p):
        ea, eb = net.tower(ga), net.tower(gb)
        h = net.trunk(torch.cat([ea, eb, p], dim=1))
        return net.head_huber(h).squeeze(1), net.head_bce(h).squeeze(1)

    monkeypatch.setattr(net, "forward", fwd)
    s1 = m3.predict_m3(net, gx, ia, ib, P)
    s2 = m3.predict_m3(net, gx, ib, ia, P)
    assert np.max(np.abs(s1 - s2)) > 1e-5
