"""Tests for M2 (src/models/m2.py, scripts/run_m2.py) and src.data.drop_noncoding.

| test                                         | production must        | planted mutant it must catch            |
|----------------------------------------------|------------------------|-----------------------------------------|
| perturb a training gene's own beta           | its beta_hat unchanged | in-sample GP regressor (gp_oof=False)   |
| change a gene's own pairs -> LOO NBR rate    | unchanged              | neighbour rates including own pairs     |
| flip eval labels -> eval features (M2)       | bit-identical          | GP inputs computed with eval rows       |
| flip a training row -> its nested M2 features| unchanged              | non-nested stacker rows                 |
| query-set invariance (M2)                    | unchanged              |                                         |
| symmetry, unseen-gene semantics, families    |                        |                                         |
Synthetic data and helpers are shared with tests/test_ryanfeat.py.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src import data, protocol
from src import splits as S
from src.models import m2
from src.ryanfeat import _clean_train, feature_columns
from src.stacking import nested_tables

from test_ryanfeat import _relabel_inner, cfg, flipped, make_data, same  # noqa: E402

GP_FAST = {"rounds": 60, "params": {"num_threads": 2}}


@pytest.fixture(scope="module")
def D():
    return make_data()


def gene_tab(D, seed=0):
    rng = np.random.default_rng(seed)
    g = D["genes"]
    t = pd.DataFrame({"X__a": rng.normal(size=len(g)), "X__b": rng.normal(size=len(g))}, index=g)
    t.iloc[::7, 0] = np.nan
    return t


def builder(D, base=m2.M2Builder, **gp):
    return m2.make_builder(gene_tab(D), {**GP_FAST, **gp}, base=base)


# ============================================================================================ drop_noncoding
def test_drop_noncoding_synthetic():
    p = pd.DataFrame({"gene_a": ["A", "A", "B", "N1"], "gene_b": ["B", "N2", "C", "C"], "y": [True, True, False, False]})
    d = data.drop_noncoding(p, genes={"A", "B", "C"})
    assert list(zip(d.gene_a, d.gene_b)) == [("A", "B"), ("B", "C")]
    assert d.index.tolist() == [0, 1]
    assert d.attrs["noncoding_dropped"] == {"pairs": 2, "positives": 1, "genes": ["N1", "N2"]}
    assert len(p) == 4                                   # input untouched


def test_drop_noncoding_real(real_env):
    genes = data.protein_coding_genes()
    tr, ev = data.fold_split(None, 0)
    d = data.drop_noncoding(tr)
    info = d.attrs["noncoding_dropped"]
    assert info["pairs"] > 0 and info["genes"] and not set(info["genes"]) & genes
    assert d.gene_a.isin(genes).all() and d.gene_b.isin(genes).all()
    assert len(d) + info["pairs"] == len(tr)
    assert data.drop_noncoding(ev).attrs["noncoding_dropped"]["pairs"] == 0     # eval pairs: universe genes


# ============================================================================================ GP regressor
def gp_data(n=300, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"G{i}" for i in range(n)])
    z = rng.normal(size=n)
    X = pd.DataFrame({"inf": z + 0.5 * rng.normal(size=n) if signal else rng.normal(size=n),
                      "noise": rng.normal(size=n)}, index=idx)
    X.iloc[::11, 0] = np.nan
    beta = pd.Series(z + 0.3 * rng.normal(size=n), index=idx)[: int(0.7 * n)]   # 30% unseen genes
    return beta, X, pd.Series(z, index=idx)


@pytest.mark.parametrize("reg", ["lgbm", "ridge", "mean"])
def test_gp_recovers_signal_and_ignores_noise(reg):
    beta, X, z = gp_data()
    bh, info = m2.fit_gene_propensity(beta, X, {**GP_FAST, "regressor": reg})
    unseen = X.index[~X.index.isin(beta.index)]
    assert bh.notna().all() and info["oof_r"] > 0.5
    assert spearmanr(bh[unseen], z[unseen]).correlation > 0.5
    beta0, X0, z0 = gp_data(signal=False)
    bh0, _ = m2.fit_gene_propensity(beta0, X0, {**GP_FAST, "regressor": reg})
    assert abs(spearmanr(bh0[unseen], z0[unseen]).correlation) < 0.3


def own_beta_moves_hat(fit) -> bool:
    beta, X, _ = gp_data()
    b0, _ = fit(beta, X, GP_FAST)
    moved = False
    for g in beta.index[:5]:
        b2 = beta.copy()
        b2[g] += 5.0
        b1, _ = fit(b2, X, GP_FAST)
        moved |= not np.isclose(b0[g], b1[g], rtol=0, atol=1e-9)
    return moved


def test_gp_training_gene_never_sees_own_beta():
    assert not own_beta_moves_hat(m2.fit_gene_propensity)


def test_gp_oof_check_catches_in_sample_mutant():
    assert own_beta_moves_hat(m2._in_sample_gp)


# ============================================================================================ LOO NBR rate
def _rate_inputs(D):
    tr = _clean_train(D["train"])
    b = m2.make_builder(None, GP_FAST)(tr, D["eval"].iloc[:50], D["sims"], cfg(D, mf=False, profile=False,
                                                                                 sims=["INF"]))
    b.build()
    idx = b.info["idx"]
    n = len(idx)
    ta, tb = idx.get_indexer(tr.gene_a), idx.get_indexer(tr.gene_b)
    Y, M = np.zeros((n, n)), np.zeros((n, n))
    Y[ta, tb] = Y[tb, ta] = tr.y.to_numpy(float)
    M[ta, tb] = M[tb, ta] = 1.0
    return b.info["W"]["INF"], Y, M, ta, tb


def _non_loo_rate(W, Y, M):
    rate = np.divide(Y.sum(1), M.sum(1), out=np.zeros(len(Y)), where=M.sum(1) > 0)
    sw = np.asarray(W.sum(1)).ravel()
    return np.where(sw > 0, (W @ rate) / np.where(sw > 0, sw, 1), np.nan)


@pytest.mark.parametrize("fn,expect_moved", [(m2.loo_nbr_gene_rate, False), (_non_loo_rate, True)])
def test_loo_nbr_rate_ignores_own_pairs(D, fn, expect_moved):
    W, Y, M, ta, tb = _rate_inputs(D)
    r0 = fn(W, Y, M)
    moved = False
    for g in np.unique(ta)[:15]:                         # flip every label of gene g's own pairs
        Y2 = Y.copy()
        m = M[g] > 0
        Y2[g, m] = 1 - Y2[g, m]
        Y2[m, g] = Y2[g, m]
        moved |= not np.isclose(fn(W, Y2, M)[g], r0[g], rtol=0, atol=1e-12, equal_nan=True)
    assert moved == expect_moved


# ============================================================================================ builder leakage
class GPEvalMutant(m2.M2Builder):
    """Planted leak: GP inputs (LOO NBR rates) computed from training + EVAL labelled pairs."""
    eval_labelled = None

    def stat_pairs(self, component):
        if component == "gp":
            return pd.concat([self.train, self.eval_labelled[["gene_a", "gene_b", "y", "s_score"]]],
                             ignore_index=True)
        return self.train


def m2_eval_leak(D, base) -> bool:
    ev, evf = D["eval"], flipped(D["eval"])
    c = cfg(D, mf=False)
    out = []
    for e in (ev, evf):
        B = builder(D, base=type("B", (base,), {"eval_labelled": e}))
        out.append(B(D["train"], e[["gene_a", "gene_b"]], D["sims"], c).build())
    return not same(out[0], out[1])


def test_m2_flip_eval_labels_bit_identical(D):
    assert not m2_eval_leak(D, m2.M2Builder)


def test_m2_flip_eval_labels_catches_gp_eval_mutant(D):
    assert m2_eval_leak(D, GPEvalMutant)


def _own_row_changes(D, nested: bool) -> bool:
    tr = D["train"]
    inner = S.inner_split_generator(tr, 5, 0.1, 0, D["roles"], 0.98, 5)
    c = cfg(D, mf=False)
    B = builder(D)

    def tables(t, inn):
        if nested:
            return nested_tables(t, D["eval"], D["sims"], c, inner=inn, builder=B, outer=False)[0]
        rows = []                                         # mutant: stacker features from ALL training pairs
        for f in inn:
            F = B(t, f.eval[["gene_a", "gene_b"]], D["sims"], c).build()
            rows.append(pd.concat([f.eval.reset_index(drop=True)[["gene_a", "gene_b", "part"]],
                                   F.drop(columns=["gene_a", "gene_b"])], axis=1))
        return pd.concat(rows, ignore_index=True)

    T0 = tables(tr, inner)
    cols = [x for x in T0.columns if x.startswith("gp_")]
    changed = False
    for p in ("C1", "C2", "C3"):
        r = T0[T0.part == p].sort_values("gene_a").index[0]
        ga, gb = T0.loc[r, ["gene_a", "gene_b"]]
        trf = flipped(tr, tr.index[(tr.gene_a == ga) & (tr.gene_b == gb)])
        T1 = tables(trf, _relabel_inner(inner, trf))
        s0 = T0[(T0.gene_a == ga) & (T0.gene_b == gb)].reset_index(drop=True)
        s1 = T1[(T1.gene_a == ga) & (T1.gene_b == gb)].reset_index(drop=True)
        changed |= not same(s0, s1, cols)
    return changed


def test_m2_flip_training_row_leaves_its_nested_gp_features(D):
    assert not _own_row_changes(D, nested=True)


def test_m2_flip_training_row_catches_non_nested(D):
    assert _own_row_changes(D, nested=False)


def test_m2_query_set_invariance(D):
    ev = D["eval"]
    c = cfg(D, mf=False)
    B = builder(D)
    F_all = B(D["train"], ev[["gene_a", "gene_b"]], D["sims"], c).build().iloc[:300].reset_index(drop=True)
    F_sub = B(D["train"], ev.iloc[:300][["gene_a", "gene_b"]], D["sims"], c).build()
    for x in [x for x in feature_columns(F_sub) if x.startswith("gp_")]:
        assert np.allclose(F_all[x], F_sub[x], rtol=1e-6, atol=1e-9, equal_nan=True), x


def test_m2_features_symmetric_and_unseen_semantics(D):
    ev = D["eval"]
    c = cfg(D, mf=False)
    B = builder(D)
    F = B(D["train"], ev[["gene_a", "gene_b"]], D["sims"], c).build()
    G = B(D["train"], ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})[["gene_a", "gene_b"]],
          D["sims"], c).build()
    for x in feature_columns(F):
        assert np.allclose(F[x], G[x], rtol=1e-6, atol=1e-9, equal_nan=True), x
    c3, c2, c1 = [(ev.part == p).to_numpy() for p in ("C3", "C2", "C1")]
    assert F.loc[c3, ["gp_beta_min", "gp_beta_max"]].isna().all().all()
    assert np.allclose(F.gp_best_sum[c3], F.gp_hat_sum[c3])
    assert F.gp_beta_min[c2].isna().all() and F.gp_beta_max[c2].notna().all()
    assert F.loc[c1, ["gp_beta_min", "gp_beta_max", "gp_hat_min"]].notna().all().all()
    assert F[[x for x in F if x.startswith("gp_hat")]].notna().all().all()


def test_families_and_variants():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_m2", protocol.ROOT / "scripts" / "run_m2.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    assert m2.family_of("gp_hat_sum") == "GP" and m2.family_of("nbr_mean__hit_dmean") == "NBR"
    assert all(m2.family_of(x) == "RYAN" for x in m2.RYAN_COLS)
    assert m2.family_of("PHEN__n_pheno__min") == "LF" and m2.family_of("HYPO__max") == "LF"
    V = R.VARIANTS
    assert set(V["M1N"]["families"]) == {"LF", "NBR"}
    assert set(V["M2"]["families"]) == set(m2.FAMILIES) and not V["M2"]["init"] and V["M2_init"]["init"]
    assert "GP" not in V["M2_minus_GP"]["families"] and "NBR" not in V["M2_minus_NBR"]["families"]


# ============================================================================================ toy known answer
TOY = protocol.ROOT / "data" / "toy"


@pytest.mark.skipif(not ((TOY / "_truth").exists() and (TOY / "splits" / "dev_folds_seed20261028.tsv").exists()),
                    reason="toy data / toy splits not built")
def test_toy_gp_tracks_true_propensity(monkeypatch):
    """Regression guard (bounds set after the pre-stated check in scripts/verify_m2_toy.py, see docs/m2.md):
    unseen-gene beta_hat must track the planted propensity, and clearly beat a NOISE-only GP."""
    monkeypatch.setenv("PSL_DATA_ROOT", str(TOY))
    truth = pd.read_csv(TOY / "_truth" / "gene_truth.tsv", sep="\t").set_index("gene").beta
    tr, ev, _ = m2.fold_pairs(None, 0)
    held = set(data.fold_genes(None, 0))
    gt = m2.gene_table()
    rho = {}
    for name, tab, sims, gp in (("full", gt, __import__("src.ryanfeat", fromlist=["SimStore"]).SimStore(), {}),
                                ("noise", gt[[x for x in gt if x.startswith("NOISE")]], None, {"nbr_gene": False})):
        b = m2.make_builder(tab, gp)(tr, ev[["gene_a", "gene_b"]], sims,
                                     {"mf": False, "profile": False, "sims": None if sims is not None else []})
        b.build()
        bh = b.info["gp"]["beta_hat"]
        ug = [g for g in bh.index if g in held]
        rho[name] = spearmanr(bh[ug], truth[ug]).correlation
    assert rho["full"] > 0.4 and abs(rho["noise"]) < 0.2 and rho["full"] > rho["noise"] + 0.3, rho
