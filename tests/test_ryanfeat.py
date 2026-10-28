"""Leakage tests for src/ryanfeat.py and src/stacking.py (DESIGN.md §6 out-of-fold leakage tests).

Each invariance test runs on the production builder AND on planted leaky mutants (flags of the test-only
wrapper `mutant()` below); the production builder must pass and every mutant must be flagged:

| test                                   | production must | mutants it must catch                         |
|----------------------------------------|-----------------|-----------------------------------------------|
| flip eval labels -> eval features      | bit-identical   | b1_eval, mf_eval, prof_eval, own_rate          |
| flip a training row -> its nested feats| unchanged       | non_nested (stacker rows from all train pairs) |
| own label -> NBR directional at (a,b)  | unchanged       | self (self not excluded)                       |
| query-set invariance                   | unchanged       | pool_eval (neighbours drawn from eval genes)   |
| random-label canary (B1/MF AP)         | ~ prevalence    | b1_eval                                        |

Synthetic data: Ryan-like query x array design with planted gene propensities and module-pair effects;
an informative similarity (module structure) and a noise similarity.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import protocol
from src import splits as S
from src.ryanfeat import RyanFeatureBuilder, SimStore, build_ryan_features, feature_columns
from src.stacking import nested_tables, stacker_folds

CFG = {"k": 5, "mf_k": 4, "mf_epochs": 60, "mf_lr": 0.05, "mf_wd": 1e-3, "b1_C": 0.1, "min_overlap": 5}
FLAGS = ("b1_eval", "mf_eval", "prof_eval", "own_rate", "self", "pool_eval")


# ============================================================================================ data
def make_data(seed=3, n_q=40, n_both=20, n_a=160, fill=0.85):
    rng = np.random.default_rng(seed)
    n = n_q + n_both + n_a
    genes = np.array([f"G{i:04d}" for i in range(n)], dtype=object)
    role = np.array(["query"] * n_q + ["both"] * n_both + ["array"] * n_a, dtype=object)
    beta = rng.normal(0, 0.8, n)
    mod = rng.integers(0, 6, n)
    sl_mod = rng.random((6, 6)) < 0.25
    sl_mod = sl_mod | sl_mod.T
    isq = np.isin(role, ["query", "both"])
    isa = np.isin(role, ["array", "both"])
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if not ((isq[i] and isa[j]) or (isq[j] and isa[i])) or rng.random() > fill:
                continue
            z = beta[i] + beta[j] + 1.5 * sl_mod[mod[i], mod[j]]
            rows.append((genes[i], genes[j], -(z + rng.normal(0, 0.8)) * 1.2))
    df = pd.DataFrame(rows, columns=["gene_a", "gene_b", "s_score"])
    thr = np.quantile(df.s_score, 0.05)                      # ~5% positives
    df["s_score"] = df.s_score - thr + protocol.threshold() - 1e-9
    df["y"] = protocol.binarize(df.s_score)
    # informative similarity: same module -> high; noise similarity: iid
    inf = (mod[:, None] == mod[None, :]) * 0.6 + rng.normal(0, 0.15, (n, n))
    inf = (inf + inf.T) / 2
    noise = rng.normal(0, 0.3, (n, n))
    noise = (noise + noise.T) / 2
    miss = rng.random(n) < 0.05
    inf[miss] = np.nan
    inf[:, miss] = np.nan
    sims = SimStore.from_arrays(list(genes), {"INF": inf, "NOISE": noise})
    # split: 20% held-out genes (C2/C3), 10% C1 pair hold-out among the rest
    held = set(rng.choice(genes, int(0.2 * n), replace=False))
    na = df.gene_a.isin(held).astype(int) + df.gene_b.isin(held).astype(int)
    c1 = (na == 0) & (rng.random(len(df)) < 0.1)
    part = np.where(na == 2, "C3", np.where(na == 1, "C2", np.where(c1, "C1", "train")))
    train = df[part == "train"].reset_index(drop=True)
    ev = df[part != "train"].assign(part=part[part != "train"]).reset_index(drop=True)
    roles = pd.Series(role, index=genes)
    return {"train": train, "eval": ev, "sims": sims, "roles": roles, "genes": genes}


@pytest.fixture(scope="module")
def D():
    return make_data()


def cfg(D, **kw):
    return {**CFG, "roles": D["roles"], **kw}


# ============================================================================================ mutants
def mutant(flags=(), eval_labelled=None):
    """Test-only wrapper: the production builder with planted leaks switched on by `flags`."""
    flags = set(flags)
    assert flags <= set(FLAGS)

    class Mutant(RyanFeatureBuilder):
        exclude_self = "self" not in flags

        def stat_pairs(self, component):
            leak = {"b1": "b1_eval", "mf": "mf_eval", "profile": "prof_eval", "rate": "own_rate"}.get(component)
            if leak in flags:
                return pd.concat([self.train, eval_labelled[["gene_a", "gene_b", "y", "s_score"]]],
                                 ignore_index=True)
            return self.train

        def pool_genes(self, role, train_genes):
            p = super().pool_genes(role, train_genes)
            if "pool_eval" in flags:
                p |= set(self.query.gene_a) | set(self.query.gene_b)
            return p

        def gene_nbr_rate(self, W, rate, idx):
            if "own_rate" not in flags:
                return super().gene_nbr_rate(W, rate, idx)
            sw = np.asarray(W.sum(1)).ravel()
            return (W @ rate + rate) / (sw + 1)               # neighbour rate including the gene's own rate

        def _neighbour_matrix(self, Wd, pools):
            if "self" in flags:
                np.fill_diagonal(Wd, 1.0)
            return super()._neighbour_matrix(Wd, pools)

    return Mutant


def flipped(df, rows=None):
    """Flip labels (and S consistently) of `rows` (default: all)."""
    out = df.copy()
    m = np.zeros(len(df), bool)
    m[rows if rows is not None else slice(None)] = True
    y = out.y.to_numpy(bool)
    out.loc[m, "s_score"] = np.where(y[m], 0.0, protocol.threshold() - 3.0)
    out["y"] = protocol.binarize(out.s_score)
    assert (out.y.to_numpy(bool)[m] != y[m]).all()
    return out


def same(F1, F2, cols=None):
    cols = cols or feature_columns(F1)
    return all(np.array_equal(F1[c].to_numpy(), F2[c].to_numpy(), equal_nan=True) for c in cols)


# ============================================================================================ tests
def test_mutant_wrapper_without_flags_equals_production(D):
    F0 = build_ryan_features(D["train"], D["eval"], D["sims"], cfg(D))
    F1 = mutant((), D["eval"])(D["train"], D["eval"], D["sims"], cfg(D)).build()
    assert same(F0, F1)
    assert F0.shape[1] > 40 and F0.prof_corr.notna().any() and F0.nbr_INF__hit_block.notna().any()


def leaks_eval_labels(D, flags) -> bool:
    """Build eval features twice, with eval labels as-is and flipped; True if any feature changes."""
    ev, evf = D["eval"], flipped(D["eval"])
    B1 = mutant(flags, ev)(D["train"], ev[["gene_a", "gene_b"]], D["sims"], cfg(D)).build()
    B2 = mutant(flags, evf)(D["train"], evf[["gene_a", "gene_b"]], D["sims"], cfg(D)).build()
    return not same(B1, B2)


def test_flip_eval_labels_production_bit_identical(D):
    assert not leaks_eval_labels(D, ())


@pytest.mark.parametrize("flag", ["b1_eval", "mf_eval", "prof_eval", "own_rate"])
def test_flip_eval_labels_catches_mutant(D, flag):
    assert leaks_eval_labels(D, (flag,))


def _relabel_inner(inner, new_train):
    """Same inner gene folds / hold-outs, labels taken from `new_train` (by pair)."""
    lab = new_train.set_index(["gene_a", "gene_b"])[["y", "s_score"]]
    out = []
    for f in inner:
        def rl(x):
            x = x.copy()
            v = lab.loc[pd.MultiIndex.from_frame(x[["gene_a", "gene_b"]])]
            x["y"], x["s_score"] = v.y.to_numpy(), v.s_score.to_numpy()
            return x
        out.append(S.InnerFold(j=f.j, genes=f.genes, train=rl(f.train), eval=rl(f.eval), holdout=f.holdout))
    return out


def non_nested_tables(train, ev, sims, c, inner):
    """Mutant stacker: training-row features from ALL outer-training pairs (not the inner fold's)."""
    rows = []
    for f in inner:
        F = RyanFeatureBuilder(train, f.eval[["gene_a", "gene_b"]], sims, c).build()
        rows.append(pd.concat([f.eval.reset_index(drop=True)[["gene_a", "gene_b", "part", "inner_fold"]],
                               F.drop(columns=["gene_a", "gene_b"])], axis=1))
    return pd.concat(rows, ignore_index=True), None, inner


def own_row_changes(D, tables_fn) -> bool:
    tr = D["train"]
    inner = S.inner_split_generator(tr, 5, 0.1, 0, D["roles"], 0.98, 5)
    T0, _, _ = tables_fn(tr, D["eval"], D["sims"], cfg(D), inner)
    # one stacker-training row of each part (C1-like, C2-like, C3-like), all positives if possible
    picks = []
    for p in ("C1", "C2", "C3"):
        cand = T0[T0.part == p]
        picks.append(cand.sort_values("gene_a").index[0])
    changed = False
    for r in picks:
        ga, gb = T0.loc[r, ["gene_a", "gene_b"]]
        ti = tr.index[(tr.gene_a == ga) & (tr.gene_b == gb)]
        trf = flipped(tr, ti)
        T1, _, _ = tables_fn(trf, D["eval"], D["sims"], cfg(D), _relabel_inner(inner, trf))
        sel0 = T0[(T0.gene_a == ga) & (T0.gene_b == gb)]
        sel1 = T1[(T1.gene_a == ga) & (T1.gene_b == gb)]
        assert len(sel0) == len(sel1) >= 1
        changed |= not same(sel0.reset_index(drop=True), sel1.reset_index(drop=True),
                            [c for c in feature_columns(T0) if c not in ("y", "s_score", "part", "inner_fold",
                                                                          "w", "hypomorph_a", "hypomorph_b")])
    return changed


def test_flip_training_row_leaves_its_nested_features(D):
    def prod(tr, ev, sims, c, inner):
        return nested_tables(tr, ev, sims, c, inner=inner, outer=False)
    assert not own_row_changes(D, prod)


def test_flip_training_row_catches_non_nested(D):
    assert own_row_changes(D, non_nested_tables)


DIR_COLS = ("hit_dmax", "hit_dmean", "negs_dmax", "negs_dmean", "support_max", "support_mean")


def own_label_moves_dir(D, flags, n_rows=12) -> bool:
    """Query = training pairs themselves (non-nested). For a training row r, swap (y, S) with a row of the
    opposite label that shares no gene with r: the global priors p0 are unchanged and no other path
    links the partner row to r's directional features, so only a self-path can move them."""
    tr = D["train"]
    B = mutant(flags)
    c = cfg(D, mf=False, profile=False, sims=["INF"])
    rows = tr.index[tr.y][:n_rows // 2].tolist() + tr.index[~tr.y][:n_rows // 2].tolist()
    q = tr.loc[rows, ["gene_a", "gene_b"]].reset_index(drop=True)
    F0 = B(tr, q, D["sims"], c).build()
    cols = [x for x in F0 if x.startswith("nbr_INF__") and x.split("__")[1] in DIR_COLS]
    moved = False
    for i, r in enumerate(rows):
        ga, gb = tr.loc[r, ["gene_a", "gene_b"]]
        ok = (tr.y != tr.y[r]) & ~tr.gene_a.isin([ga, gb]) & ~tr.gene_b.isin([ga, gb])
        r2 = tr.index[ok][0]
        t2 = tr.copy()
        t2.loc[[r, r2], ["y", "s_score"]] = tr.loc[[r2, r], ["y", "s_score"]].to_numpy()
        F1 = B(t2, q, D["sims"], c).build()
        moved |= not same(F0.iloc[[i]], F1.iloc[[i]], cols)
    return moved


def test_nbr_directional_excludes_self(D):
    assert not own_label_moves_dir(D, ())


def test_nbr_directional_catches_self_not_excluded(D):
    assert own_label_moves_dir(D, ("self",))


def query_set_dependence(D, flags) -> bool:
    ev = D["eval"]
    sub = ev.iloc[:300]
    B = mutant(flags)
    c = cfg(D, mf=False)
    F_all = B(D["train"], ev[["gene_a", "gene_b"]], D["sims"], c).build().iloc[:300].reset_index(drop=True)
    F_sub = B(D["train"], sub[["gene_a", "gene_b"]], D["sims"], c).build()
    cols = feature_columns(F_sub)
    return not all(np.allclose(F_all[x], F_sub[x], rtol=1e-6, atol=1e-9, equal_nan=True) for x in cols)


def test_query_set_invariance(D):
    assert not query_set_dependence(D, ())


def test_query_set_invariance_catches_eval_gene_neighbours(D):
    assert query_set_dependence(D, ("pool_eval",))


def test_neighbours_are_training_genes(D):
    b = RyanFeatureBuilder(D["train"], D["eval"], D["sims"], cfg(D, mf=False))
    b.build()
    tg = set(D["train"].gene_a) | set(D["train"].gene_b)
    for W in b.info["W"].values():
        nb = set(b.info["idx"][np.unique(W.indices)])
        assert nb and nb <= tg
        assert (W.diagonal() == 0).all()


def canary_ap(D, flags):
    """Train-label permutation canary: AP of b1_logit / mf_score on the TRUE eval labels."""
    rng = np.random.default_rng(11)
    tr = D["train"].copy()
    perm = rng.permutation(len(tr))
    tr[["y", "s_score"]] = tr[["y", "s_score"]].to_numpy()[perm]
    ev = D["eval"]
    F = mutant(flags, ev)(tr, ev[["gene_a", "gene_b"]], D["sims"], cfg(D, sims=[])).build()
    y = ev.y.to_numpy(bool)
    return {c: average_precision_score(y, F[c]) for c in ("b1_logit", "mf_score")}, y.mean()


def test_random_label_canary(D):
    ap, prev = canary_ap(D, ())
    assert all(v < 1.6 * prev for v in ap.values()), (ap, prev)
    # positive control: with the real training labels B1 and MF clearly beat prevalence
    F = build_ryan_features(D["train"], D["eval"], D["sims"], cfg(D, sims=[]))
    y = D["eval"].y.to_numpy(bool)
    assert average_precision_score(y, F.b1_logit) > 3 * prev
    assert average_precision_score(y, F.mf_score) > 2 * prev


def test_random_label_canary_catches_b1_with_eval_rows(D):
    ap, prev = canary_ap(D, ("b1_eval",))
    assert ap["b1_logit"] > 2 * prev, (ap, prev)


def test_features_symmetric(D):
    ev = D["eval"]
    F = build_ryan_features(D["train"], ev, D["sims"], cfg(D))
    sw = ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})[["gene_a", "gene_b"]]
    G = build_ryan_features(D["train"], sw, D["sims"], cfg(D))
    for c in feature_columns(F):
        assert np.allclose(F[c], G[c], rtol=1e-6, atol=1e-9, equal_nan=True), c


def test_unseen_genes_fall_back(D):
    F = build_ryan_features(D["train"], D["eval"], D["sims"], cfg(D))
    c3 = (D["eval"].part == "C3").to_numpy()
    assert (F.seen_max[c3] == 0).all() and (F.beta_max[c3] == 0).all()
    assert np.ptp(F.b1_logit[c3]) == 0 and np.ptp(F.mf_score[c3]) < 1e-5
    assert F.prof_corr[c3].isna().all()


def test_informative_sim_beats_noise(D):
    F = build_ryan_features(D["train"], D["eval"], D["sims"], cfg(D, mf=False))
    ev = D["eval"]
    for part in ("C2", "C3"):
        m = (ev.part == part).to_numpy()
        y = ev.y.to_numpy(bool)[m]
        inf = average_precision_score(y, F.loc[m, "nbr_INF__hit_block"].fillna(0))
        noi = average_precision_score(y, F.loc[m, "nbr_NOISE__hit_block"].fillna(0))
        assert inf > noi, (part, inf, noi)


def test_stacker_rows_weights_and_grouping(D):
    tr = D["train"]
    inner = S.inner_split_generator(tr, 5, 0.1, 0, D["roles"], 0.98, 5)
    T, _, _ = nested_tables(tr, D["eval"], D["sims"], cfg(D, mf=False, sims=["INF"]), inner=inner, outer=False)
    for j, d in T.groupby("inner_fold"):
        tot = d.groupby("part").w.sum()
        cw = protocol.class_weights()
        assert np.allclose(tot.to_numpy(), [cw[p] for p in tot.index])
    # C2-like / C3-like rows have seen_min = 0 (T_j genes are unseen in inner training)
    assert (T[T.part != "C1"].seen_min == 0).all() and (T[T.part == "C1"].seen_min == 1).all()
    for j, t, v in stacker_folds(T, inner):
        Tj = set(inner[j].genes)
        assert not (t.gene_a.isin(Tj) | t.gene_b.isin(Tj)).any()
        assert (v.inner_fold == j).all()


# ============================================================================================ toy known answers
TOY = protocol.ROOT / "data" / "toy"


def _toy_root(tmp_path_factory, src, monkeypatch):
    import shutil
    import subprocess
    import sys
    root = tmp_path_factory.mktemp(src.name)
    shutil.copytree(src, root / "data", ignore=shutil.ignore_patterns("_truth", "splits", "cache", "preds"))
    env = {**__import__("os").environ, "PSL_DATA_ROOT": str(root / "data"), "PSL_RESULTS_ROOT": str(root / "res")}
    subprocess.run([sys.executable, str(protocol.ROOT / "scripts" / "make_splits.py")], env=env, check=True,
                   capture_output=True)
    monkeypatch.setenv("PSL_DATA_ROOT", str(root / "data"))
    monkeypatch.setenv("PSL_RESULTS_ROOT", str(root / "res"))
    return root / "data"


def _toy_b1_c1(root):
    """Fold-mean C1 AP of B1 (tuned default C) and of the TRUE latent propensity beta_a + beta_b on the same
    CV C1 rows (the known_answers.json oracle is computed on a different pair set)."""
    from src import data
    from src.ryanfeat import DEFAULT_CFG, fit_b1
    truth = pd.read_csv(TOY / "_truth" / "gene_truth.tsv", sep="\t").set_index("gene").beta
    lab = pd.read_parquet(root / "splits" / "labels_dev.parquet")[["gene_a", "gene_b", "y"]]
    b1, orc, prev = [], [], []
    for k in range(protocol.load()["dev_cv"]["n_folds"]):
        tr, ev = data.fold_split(None, k)
        c1 = ev[ev.part == "C1"].merge(lab, on=["gene_a", "gene_b"])
        _, beta = fit_b1(tr, DEFAULT_CFG["b1_C"])
        s = beta.reindex(c1.gene_a).fillna(0).to_numpy() + beta.reindex(c1.gene_b).fillna(0).to_numpy()
        b1.append(average_precision_score(c1.y, s))
        orc.append(average_precision_score(c1.y, truth[c1.gene_a].to_numpy() + truth[c1.gene_b].to_numpy()))
        prev.append(c1.y.mean())
    return np.mean(b1), np.mean(orc), np.mean(prev)


@pytest.mark.skipif(not (TOY / "_truth").exists(), reason="toy data not built")
def test_toy_b1_approaches_true_propensity_oracle(tmp_path_factory, monkeypatch):
    b1, orc, prev = _toy_b1_c1(_toy_root(tmp_path_factory, TOY, monkeypatch))
    assert 0.8 * orc <= b1 <= orc + 0.02, (b1, orc)       # approaches the oracle from below (estimation noise)
    assert b1 > 5 * prev


@pytest.mark.skipif(not (protocol.ROOT / "data" / "toy_permuted").exists(), reason="permuted toy not built")
def test_toy_permuted_labels_b1_at_prevalence(tmp_path_factory, monkeypatch):
    b1, _, prev = _toy_b1_c1(_toy_root(tmp_path_factory, protocol.ROOT / "data" / "toy_permuted", monkeypatch))
    assert b1 < 1.5 * prev, (b1, prev)
