"""Tests for the E-CONT additive options (config/m2cont.yaml): continuous NBR columns (src/ryanfeat
cfg['cont_nbr']), continuous GP target (src/models/m2 gp_cfg['cont']), the gene-permuted control
(cfg['cont_perm_seed']) and scripts/run_m2cont.py helpers.

| test                                              | production must          | planted mutant it must catch          |
|---------------------------------------------------|--------------------------|---------------------------------------|
| options off -> features / cache hash              | identical to incumbent   |                                       |
| options on  -> binary (incumbent) columns         | bit-identical            |                                       |
| CANARY: corrupt S2 of held-out (eval) pairs       | new columns bit-identical| cont components fed eval S2           |
| CANARY: corrupt S2 of a stacker row's own pair    | its nested new columns   | non-nested stacker rows               |
|         (labels y unchanged, only S2 moves)       | unchanged                |                                       |
| query-set invariance, symmetry                    | unchanged                |                                       |
| additive ridge theta                              | recovers planted theta   |                                       |
| permutation control                               | roles/labels kept, new   |                                       |
|                                                   | columns change, binary   |                                       |
|                                                   | columns identical        |                                       |
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src import protocol
from src import splits as S
from src.models import m2
from src.ryanfeat import _clean_train, cfg_hash, cont_target, feature_columns, permute_genes_within_roles
from src.stacking import nested_tables

from test_ryanfeat import _relabel_inner, cfg, make_data, same  # noqa: E402

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


def builder(D, base=m2.M2Builder, cont=True):
    return m2.make_builder(gene_tab(D), {**GP_FAST, "cont": cont}, base=base)


def ccfg(D, **kw):
    return cfg(D, mf=False, cont_nbr=True, **kw)


def new_cols(F):
    return [c for c in feature_columns(F) if m2.family_of(c) in ("GPC", "NBRC")]


def corrupt_s(df, rows=None, seed=0):
    """Move S2 of `rows` (default all) while keeping the binary label y: positives stay below the
    threshold, negatives stay at/above it. Only continuous features can react."""
    out = df.copy()
    m = np.zeros(len(df), bool)
    m[rows if rows is not None else slice(None)] = True
    rng = np.random.default_rng(seed)
    thr = protocol.threshold()
    y = out.y.to_numpy(bool)
    new = np.where(y, thr - 1.0 - 4.0 * rng.random(len(df)), thr * rng.random(len(df)))
    out.loc[m, "s_score"] = new[m]
    assert (protocol.binarize(out.s_score) == out.y.to_numpy(bool)).all()
    assert (out.s_score.to_numpy()[m] != df.s_score.to_numpy()[m]).all()
    return out


# ============================================================================================ off / on
def test_options_off_leave_incumbent_unchanged(D):
    c0 = cfg(D, mf=False)
    assert cfg_hash(c0) == cfg_hash({**c0, "cont_nbr": False, "cont_perm_seed": None})
    assert m2.resolve_gp({"cont": False}) == m2.resolve_gp(None)
    F0 = m2.make_builder(gene_tab(D), GP_FAST)(D["train"], D["eval"][["gene_a", "gene_b"]], D["sims"], c0).build()
    assert not new_cols(F0)
    F1 = builder(D)(D["train"], D["eval"][["gene_a", "gene_b"]], D["sims"], ccfg(D)).build()
    assert len(new_cols(F1)) == 8 + 5 * (len(D["sims"].names) + 1)
    assert same(F0, F1, feature_columns(F0))                 # binary columns bit-identical
    assert cfg_hash(ccfg(D)) != cfg_hash(c0)


def test_families():
    assert m2.family_of("gpc_best_sum") == "GPC" and m2.family_of("nbrc_mean__dmax") == "NBRC"
    assert m2.family_of("gp_best_sum") == "GP" and m2.family_of("nbr_mean__hit_dmax") == "NBR"
    assert m2.GPC_COLS[0] == "gpc_hat_min" and len(m2.GPC_COLS) == len(m2.GP_COLS)


# ============================================================================================ canaries
class ContEvalMutant(m2.M2Builder):
    """Planted leak: continuous components (nbrc, gpc) fed training + EVAL labelled pairs."""
    eval_labelled = None

    def stat_pairs(self, component):
        if component in ("nbrc", "gpc"):
            return pd.concat([self.train, self.eval_labelled[["gene_a", "gene_b", "y", "s_score"]]],
                             ignore_index=True)
        return self.train


def eval_canary_moves(D, base) -> bool:
    ev, evc = D["eval"], corrupt_s(D["eval"])
    out = []
    for e in (ev, evc):
        B = builder(D, base=type("B", (base,), {"eval_labelled": e}))
        out.append(B(D["train"], e[["gene_a", "gene_b"]], D["sims"], ccfg(D)).build())
    return not same(out[0], out[1], new_cols(out[0]))


def test_canary_corrupt_heldout_s2_new_features_identical(D):
    assert not eval_canary_moves(D, m2.M2Builder)


def test_canary_catches_cont_eval_mutant(D):
    assert eval_canary_moves(D, ContEvalMutant)


def _own_row_changes(D, nested: bool) -> bool:
    tr = D["train"]
    inner = S.inner_split_generator(tr, 5, 0.1, 0, D["roles"], 0.98, 5)
    c = ccfg(D)
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
    cols = new_cols(T0)
    changed = False
    for p in ("C1", "C2", "C3"):
        r = T0[T0.part == p].sort_values("gene_a").index[0]
        ga, gb = T0.loc[r, ["gene_a", "gene_b"]]
        trc = corrupt_s(tr, tr.index[(tr.gene_a == ga) & (tr.gene_b == gb)])
        T1 = tables(trc, _relabel_inner(inner, trc))
        s0 = T0[(T0.gene_a == ga) & (T0.gene_b == gb)].reset_index(drop=True)
        s1 = T1[(T1.gene_a == ga) & (T1.gene_b == gb)].reset_index(drop=True)
        changed |= not same(s0, s1, cols)
    return changed


def test_canary_corrupt_stacker_row_s2_leaves_its_nested_cont_features(D):
    assert not _own_row_changes(D, nested=True)


def test_canary_catches_non_nested_cont(D):
    assert _own_row_changes(D, nested=False)


def test_cont_features_react_to_training_s2(D):
    """Sanity of the canary: corrupting TRAINING S2 (labels unchanged) does move the new columns."""
    B = builder(D)
    q = D["eval"][["gene_a", "gene_b"]]
    F0 = B(D["train"], q, D["sims"], ccfg(D)).build()
    F1 = B(corrupt_s(D["train"]), q, D["sims"], ccfg(D)).build()
    assert same(F0, F1, [c for c in feature_columns(F0) if c.startswith(("gp_", "nbr_")) and "negs" not in c])
    assert not same(F0, F1, new_cols(F0))


def test_query_set_invariance_and_symmetry(D):
    ev = D["eval"]
    B = builder(D)
    F = B(D["train"], ev[["gene_a", "gene_b"]], D["sims"], ccfg(D)).build()
    Fs = B(D["train"], ev.iloc[:300][["gene_a", "gene_b"]], D["sims"], ccfg(D)).build()
    G = B(D["train"], ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})[["gene_a", "gene_b"]],
          D["sims"], ccfg(D)).build()
    for x in new_cols(F):
        assert np.allclose(F[x].iloc[:300], Fs[x], rtol=1e-6, atol=1e-9, equal_nan=True), x
        assert np.allclose(F[x], G[x], rtol=1e-6, atol=1e-9, equal_nan=True), x
    c3 = (ev.part == "C3").to_numpy()
    assert F.loc[c3, ["gpc_beta_min", "gpc_beta_max"]].isna().all().all()
    assert np.allclose(F.gpc_best_sum[c3], F.gpc_hat_sum[c3])
    assert F[[x for x in F if x.startswith("gpc_hat")]].notna().all().all()


# ============================================================================================ theta
def test_additive_cont_recovers_theta():
    rng = np.random.default_rng(0)
    G = 120
    genes = np.array([f"G{i:03d}" for i in range(G)], dtype=object)
    th = rng.gamma(1.0, 0.5, G)
    a, b = np.triu_indices(G, 1)
    keep = rng.random(len(a)) < 0.5
    a, b = a[keep], b[keep]
    sig = th[a] + th[b] + rng.normal(0, 1.0, len(a))
    tr = pd.DataFrame({"gene_a": genes[a], "gene_b": genes[b], "s_score": -sig})
    tr["y"] = protocol.binarize(tr.s_score)
    theta, info = m2.fit_additive_cont(_clean_train(tr))
    assert info["alpha"] > 0 and theta.index.equals(pd.Index(genes))
    assert spearmanr(theta[genes], th).correlation > 0.85
    tr0 = tr.assign(s_score=-rng.normal(0, 1.0, len(tr)) - 0.0)       # no gene signal -> strong shrinkage
    th0, info0 = m2.fit_additive_cont(_clean_train(tr0.assign(y=protocol.binarize(tr0.s_score))))
    assert info0["alpha"] > info["alpha"] * 10


def test_cont_target():
    assert np.array_equal(cont_target([-3.0, 0.5, 0.0, -0.1]), [3.0, 0.0, 0.0, 0.1])


# ============================================================================================ permutation control
def test_permute_genes_within_roles(D):
    tr = _clean_train(D["train"])
    P1 = permute_genes_within_roles(tr, 7, D["roles"])
    P2 = permute_genes_within_roles(tr, 7, D["roles"])
    assert P1.equals(P2) and len(P1) == len(tr)
    assert sorted(P1.s_score) == sorted(tr.s_score) and P1.y.sum() == tr.y.sum()
    ga = set(tr.gene_a) | set(tr.gene_b)
    assert set(P1.gene_a) | set(P1.gene_b) == ga
    na = pd.concat([tr.gene_a, tr.gene_b]).value_counts()
    nb = pd.concat([P1.gene_a, P1.gene_b]).value_counts()
    assert sorted(na.to_numpy()) == sorted(nb.to_numpy()) and not na.reindex(nb.index).equals(nb)
    assert not P1.equals(permute_genes_within_roles(tr, 8, D["roles"]))


def test_perm_control_changes_only_new_columns(D):
    q = D["eval"][["gene_a", "gene_b"]]
    F0 = builder(D)(D["train"], q, D["sims"], ccfg(D)).build()
    Fp = builder(D)(D["train"], q, D["sims"], ccfg(D, cont_perm_seed=20261028)).build()
    assert same(F0, Fp, [c for c in feature_columns(F0) if c not in new_cols(F0)])
    moved = [c for c in new_cols(F0) if not np.allclose(F0[c], Fp[c], equal_nan=True)]
    assert len(moved) >= len(new_cols(F0)) - 2
    # the permuted theta_hat no longer tracks the planted gene structure as well as the real one
    assert np.corrcoef(F0.gpc_hat_sum, F0.gp_hat_sum)[0, 1] > np.corrcoef(Fp.gpc_hat_sum, Fp.gp_hat_sum)[0, 1]
