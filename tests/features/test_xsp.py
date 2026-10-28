"""Tests for the cross-species transfer (XSP) group: src/xsp.py unit tests on toy data with known answers,
plus shape/symmetry/manifest checks of the staged outputs (skipped until build_xsp.py has run)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import xsp  # noqa: E402

STAGE = ROOT / "data/processed/features_staging"
ALLOW = ROOT / "config/data_allowlist.yaml"


# ----------------------------------------------------------------------------------- unit tests
def test_rank_normalise_within_table():
    df = pd.DataFrame({"a": [10.0, 20.0, 20.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    r = xsp.rank_normalise(df)
    assert np.isnan(r.a[3])
    assert r.a[1] == r.a[2] == pytest.approx(2.5 / 3)
    np.testing.assert_allclose(r.b, [0.25, 0.5, 0.75, 1.0])
    # invariant to monotone transforms -> species scale differences vanish
    pd.testing.assert_frame_equal(xsp.rank_normalise(df * 1000 + 7), r)


def test_ecdf_midrank_and_symmetry():
    rng = np.random.default_rng(0)
    M = rng.random((60, 60)).astype(np.float32)
    M = (M + M.T) / 2
    np.fill_diagonal(M, np.nan)
    E = xsp.ecdf_matrix(M)
    assert np.isnan(np.diagonal(E)).all()
    assert np.array_equal(E, E.T, equal_nan=True)
    v = E[np.triu_indices(60, 1)]
    assert 0 <= v.min() and v.max() <= 1 and abs(np.median(v) - 0.5) < 0.05
    # ties get the mid-rank: a constant-0 matrix maps to 0.5
    Z = np.zeros((10, 10), np.float32)
    np.fill_diagonal(Z, np.nan)
    assert np.allclose(xsp.ecdf_matrix(Z)[np.triu_indices(10, 1)], 0.5)


def test_ppi_pair_matrices_known_graph():
    # path 0-1-2-3, triangle 4-5-6, isolated 7
    A = np.zeros((8, 8), np.float32)
    for a, b in [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (4, 6)]:
        A[a, b] = A[b, a] = 1
    m = xsp.ppi_pair_matrices(A)
    assert m["edge"][0, 1] == 1 and m["edge"][0, 2] == 0
    assert m["shared"][0, 2] == 1 and m["shared"][4, 5] == 1
    assert m["sp"][0, 3] == 3 and m["sp"][0, 4] == 5 and np.isnan(m["sp"][0, 7])
    assert np.isnan(m["jaccard"][7, 0]) and m["jaccard"][0, 2] == pytest.approx(1 / 2)
    assert np.isnan(m["rwr"][7, 1]) and m["rwr"][0, 1] > m["rwr"][0, 3] > 0
    for M in m.values():
        assert np.array_equal(M, M.T, equal_nan=True) and np.isnan(np.diagonal(M)).all()
    g = xsp.ppi_gene_features(A)
    assert list(g.ppi_degree) == [1, 2, 2, 1, 2, 2, 2, 0]
    assert g.ppi_clustering[4] == 1 and g.ppi_core[4] == 2 and g.ppi_nbr_degree[7] == 0


def test_subsample_edges_fraction_and_symmetry():
    rng = np.random.default_rng(3)
    A = (rng.random((300, 300)) < 0.1).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T
    B = xsp.subsample_edges(A, 0.2, seed=0)
    assert np.array_equal(B, B.T) and (B <= A).all()
    assert abs(B.sum() / A.sum() - 0.2) < 0.03


def test_esm_joint_pca_removes_species_offset():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 20))
    pcs, evr = xsp.esm_joint_pca({"p": base[:100] + 50.0, "c": base[100:] - 50.0}, n_pc=5)
    # per-species centring: a pure species offset must not appear as a principal component
    assert abs(pcs["p"].mean(0)).max() < 1e-4 and abs(pcs["c"].mean(0)).max() < 1e-4
    assert evr[0] < 0.5


def test_paralog_and_mutual_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 16))
    X[1] = X[0] + 0.01 * rng.normal(size=16)            # near-duplicate pair (0, 1)
    S = xsp.centred_cosine(X)
    p = xsp.paralog_features(S)
    assert p.para_top1[0] > 0.95 and p.para_top1[1] > 0.95 and p.para_top1[5] < 0.9
    MR = xsp.mutual_rank(S)
    assert MR[0, 1] == pytest.approx(1.0) and np.array_equal(MR, MR.T, equal_nan=True)


def test_seq_stats():
    s = xsp.seq_stats({"g1": "AAAK", "g2": "DDEE"}, ["g1", "g2", "g3"])
    assert s.loc["g1", "aa_A"] == 0.75 and s.loc["g1", "net_charge"] == 0.25
    assert s.loc["g2", "net_charge"] == -1 and s.loc["g2", "charged"] == 1
    assert s.loc["g3"].isna().all()


def test_assemble_is_symmetric():
    rng = np.random.default_rng(2)
    n = 30
    gene = pd.DataFrame(rng.random((n, 3)), columns=["x", "y", "z"], index=[f"g{i}" for i in range(n)])
    gene.iloc[3, 0] = np.nan
    M = rng.random((n, n)).astype(np.float32)
    M = (M + M.T) / 2
    np.fill_diagonal(M, np.nan)
    sp = xsp.Species("toy", list(gene.index), gene.astype(np.float32), {"m": M})
    extra = rng.random((n, 2)).astype(np.float32)
    i, j = rng.integers(0, n, 200), rng.integers(0, n, 200)
    keep = i != j
    i, j = i[keep], j[keep]
    cols = ["x", "y", "z", "h1", "h2"]
    a = xsp.assemble(sp, i, j, cols, ["m"], extra)
    b = xsp.assemble(sp, j, i, cols, ["m"], extra)
    assert np.array_equal(a, b, equal_nan=True)
    assert a.shape[1] == len(xsp.pair_feature_names(cols, ["m"]))
    r = np.where(i == 3)[0][0]                             # gene 3 has NaN x: min NaN, max = partner's x
    assert np.isnan(a[r, 0]) and a[r, 5] == pytest.approx(gene.x.iloc[j[r]])


def test_gene_folds_keep_groups_together():
    genes = [f"Y{i}" for i in range(100)]
    f = xsp.gene_folds(genes, [("Y1", "Y2"), ("Y2", "Y50"), ("Y7", "Y8")], k=5, seed=0)
    assert f[1] == f[2] == f[50] and f[7] == f[8]
    assert set(f) == set(range(5))


def test_smoothed_rate():
    r = xsp.smoothed_rate(np.array([0, 10, 0]), np.array([100, 100, 0]), prior=0.05, strength=20)
    assert r[0] == pytest.approx(1 / 120) and r[1] == pytest.approx(11 / 120) and np.isnan(r[2])


def test_builder_is_pombe_label_free():
    pat = re.compile(r"ryan_pairs|ryan_measurements|ryan_genes|results/eda|data_inventory|FYPOviability|splits/|preds/")
    for p in [ROOT / "scripts/features/build_xsp.py", ROOT / "src/xsp.py"]:
        assert not pat.search(p.read_text()), f"{p} references pombe label / forbidden files"


# ----------------------------------------------------------------------------------- staged outputs
staged = pytest.mark.skipif(not (STAGE / "manifest_xsp.json").exists(), reason="XSP not built yet")


@staged
def test_staged_pair_hat_shape_symmetry():
    G = len(fs.genes())
    P = np.load(STAGE / "pair/XSP__pair_hat.npy", mmap_mode="r")
    assert P.shape == (G, G) and P.dtype == np.float32
    fs.check_square(np.asarray(P), G, "XSP__pair_hat", sim=False)
    v = np.asarray(P[:500, 500:1000])
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()


@staged
def test_staged_gene_table():
    g = pd.read_parquet(STAGE / "gene/XSP.parquet")
    assert list(g.index) == fs.genes()
    assert set(g.columns) == {"XSP__prop_hat", "XSP__prop_nn_hat", "XSP__ess_hat"}
    assert (g.dtypes == np.float32).all() and g.notna().all().all()
    assert g.XSP__ess_hat.between(0, 1).all()


@staged
def test_staged_manifest_sources_allowlisted_and_hashes():
    import yaml
    allow = yaml.safe_load(ALLOW.read_text())["datasets"]
    man = json.loads((STAGE / "manifest_xsp.json").read_text())
    assert set(man) == {"gene/XSP.parquet", "pair/XSP__pair_hat.npy"}
    for key, e in man.items():
        for s in e["sources"]:
            assert allow[s]["use"] == "feature", (key, s)
        assert fs.sha256(STAGE / key) == e["sha256"], key
