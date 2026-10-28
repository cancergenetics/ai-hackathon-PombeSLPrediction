"""Tests for the tier-2 label-free feature groups (EXPR, KOPROT, LOC, SEQ, CONS):
src/sim_utils.py, scripts/features/build_{expr,koprot,loc,seq,cons}.py, scripts/features/report_tier2.py.

Unit tests run on synthetic inputs with known answers; store tests skip when the (git-ignored) feature store or
a group's outputs are absent.
Run: .venv/bin/python -m pytest tests/features -q
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import sim_utils as su  # noqa: E402

FEAT = fs.feat_dir()
MANIFEST = FEAT / "manifest.json"
GROUPS = {"EXPR", "KOPROT", "LOC", "SEQ", "CONS"}
TIER2_SCRIPTS = [ROOT / f"scripts/features/build_{n}.py" for n in ("expr", "koprot", "loc", "seq", "cons")]
EXPECTED = {
    "EXPR": ["gene/EXPR.parquet", "sim/expr_rnaseq.npy", "sim/expr_array.npy", "pair/EXPR__mr_rnaseq.npy",
             "pair/EXPR__mr_array.npy"],
    "KOPROT": ["gene/KOPROT.parquet", "sim/koprot_corr.npy", "pair/KOPROT__a_changes_b.npy",
               "pair/KOPROT__min_z.npy", "pair/KOPROT__changed.npy", "directed/KOPROT__z_ko_row_protein_col.npy"],
    "LOC": ["gene/LOC.parquet", "pair/LOC__same_primary.npy", "pair/LOC__jaccard.npy"],
    "SEQ": ["gene/SEQ.parquet", "pair/SEQ__shared_pfam.npy", "pair/SEQ__shared_panther.npy",
            "pair/SEQ__domain_jaccard.npy", "pair/SEQ__paralog_identity.npy", "pair/SEQ__paralog.npy",
            "pair/SEQ__paralog_recent.npy"],
    "CONS": ["gene/CONS.parquet", "sim/cons_phylo.npy"],
}
FORBIDDEN = re.compile(r"fypo|gene.?ontology|go[ _]?slim|interpro2go|kegg|string|text.?min|interactions\.tsv|"
                       r"ryan|koch|data_inventory|results/eda|funfam|ncbifam|pirsf|hamap|sfld", re.I)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/features/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _feature_datasets() -> set[str]:
    al = yaml.safe_load((ROOT / "config/data_allowlist.yaml").read_text())
    return {k for k, v in al["datasets"].items() if isinstance(v, dict) and v.get("use") == "feature"}


def _group_built(group: str) -> bool:
    man = _manifest()
    return (FEAT / "genes.txt").exists() and all(k in man and (FEAT / k).exists() for k in EXPECTED[group])


# ======================================================================= label-free / allowlist (static)
def test_tier2_scripts_are_label_free():
    """Tier-2 builders, the report and the helper module must never reference Ryan labels, splits or label EDA.
    (The report reads data/processed/ryan_genes.tsv, a gene list, for coverage denominators only.)"""
    pat = re.compile(r"ryan_pairs|ryan_measurements|results/eda|data_inventory|processed/splits|\bsplits/|"
                     r"labels_dev|labels_lockbox|lockbox|FYPOviability|src\.data\b|from src import data|"
                     r"src/data\.py|ryan_s2_transform")
    files = TIER2_SCRIPTS + [ROOT / "scripts/features/report_tier2.py", ROOT / "src/sim_utils.py"]
    for p in files:
        assert p.exists(), p
        assert not pat.search(p.read_text()), f"{p} references label/forbidden files"
    for p in TIER2_SCRIPTS:
        assert "ryan_genes" not in p.read_text(), f"{p}: builders must not even read the Ryan gene list"


def test_tier2_sources_constants_allowlisted():
    """Every *_SRC constant in the builders is a use=feature dataset id of config/data_allowlist.yaml."""
    allowed = _feature_datasets()
    for p in TIER2_SCRIPTS:
        consts = re.findall(r'(?:^|[ ,(])[A-Z_]*SRC\s*(?:,\s*[A-Z_]*SRC\s*)*=\s*([^\n]+)', p.read_text(), re.M)
        found = set(re.findall(r'"([a-z0-9_]+)"', " ".join(consts)))
        assert found, f"{p}: no source constants found"
        assert found <= allowed, f"{p}: {found - allowed} not use=feature in the allowlist"


# ======================================================================= unit tests: sim_utils
def test_row_pearson_and_cosine_match_numpy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 12))
    X[5] = 3.0                                         # constant row -> dropped
    S, ok = su.row_pearson(X)
    assert not ok[5] and ok.sum() == 29
    R = np.corrcoef(X[ok])
    np.fill_diagonal(R, np.nan)
    assert np.allclose(S, R, equal_nan=True)
    assert np.array_equal(S, S.T, equal_nan=True) and np.isnan(np.diagonal(S)).all()
    C, okc = su.row_cosine(X)
    V = X / np.linalg.norm(X, axis=1, keepdims=True)
    ref = V @ V.T
    np.fill_diagonal(ref, np.nan)
    assert okc.all() and np.allclose(C, ref, equal_nan=True)


def test_incidence_overlap_known_answer():
    B = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]])
    inter, jac = su.incidence_overlap(B)
    assert inter[0, 1] == 1 and inter[0, 4] == 1 and inter[1, 4] == 0
    assert jac[0, 1] == pytest.approx(0.5) and jac[0, 4] == pytest.approx(1 / 3)
    assert jac[1, 2] == 0                              # one empty set -> 0
    assert np.isnan(jac[2, 3])                         # both empty -> NaN
    assert np.isnan(np.diagonal(jac)).all() and np.isnan(np.diagonal(inter)).all()


def test_mutual_rank_known_answer():
    S = np.array([[np.nan, 0.9, 0.1, 0.0],
                  [0.9, np.nan, 0.5, 0.2],
                  [0.1, 0.5, np.nan, 0.3],
                  [0.0, 0.2, 0.3, np.nan]])
    M = su.mutual_rank_score(S)
    # 0-1: mutual rank 1 -> score 1 ; 0-3: rank 3 for gene 0, rank 3 for gene 3 -> MR 3 -> score 0
    assert M[0, 1] == pytest.approx(1.0) and M[0, 3] == pytest.approx(0.0)
    # 1-2: rank of 2 for gene 1 is 2, rank of 1 for gene 2 is 1 -> MR sqrt(2)
    assert M[1, 2] == pytest.approx(1 - np.log(np.sqrt(2)) / np.log(3))
    assert np.allclose(M, M.T, equal_nan=True) and np.isnan(np.diagonal(M)).all()


def test_scatter_and_dist_summary():
    sub = np.array([[np.nan, 0.5], [0.5, np.nan]])
    M = su.scatter(sub, np.array([3, 1]), 5)
    assert M.dtype == np.float32 and M[3, 1] == 0.5 and M[1, 3] == 0.5 and np.isnan(M[0, 1])
    fs.check_square(M, 5, "toy", sim=True)
    with pytest.raises(AssertionError):
        su.scatter(sub, np.array([1, 1]), 5)
    rng = np.random.default_rng(1)
    n = 400
    A = rng.normal(size=(n, n))
    A = (A + A.T) / 2
    assert not su.dist_summary(A)["heavily_skewed"]
    Z = np.zeros((n, n))
    Z[:5, :5] = 1                                      # mostly zeros (like a sparse paralogue matrix)
    assert su.dist_summary(Z)["heavily_skewed"]


# ======================================================================= unit tests: parsers / ESM
def test_loc_grammar_parser():
    loc = _load_script("build_loc")
    st, allc, prim, _ = loc.parse("nucleolus>nucleus=cytosol")
    assert st == "observed" and allc == {"nucleolus", "nucleus", "cytosol"} and prim == {"nucleolus"}
    st, allc, prim, _ = loc.parse("cytosol=nucleus; cytoplasmic dots")
    assert allc == {"cytosol", "nucleus", "cytoplasmic_dots"} and prim == {"cytosol", "nucleus"}
    st, allc, prim, _ = loc.parse("SPB; nucleus>cytosol")
    assert prim == {"spb"} and "nucleus" in allc
    assert loc.parse("ER; Golgi?")[1] == {"er", "golgi"}
    assert loc.parse("periphery at cell tip and site of septum formation")[1] == {"periphery"}
    assert loc.parse("no apparent signal")[0] == "no_signal"
    for miss in ("NA", "no expression clone", "not cloned", None, float("nan")):
        assert loc.parse(miss)[0] == "missing"
    st, allc, prim, unm = loc.parse("ambiguous structure")
    assert st == "observed" and not allc and unm == ["ambiguous structure"]


def test_esm_features_on_random_matrix():
    seq = _load_script("build_seq")
    rng = np.random.default_rng(2)
    genes = [f"G{i}" for i in range(50)]
    gi = {g: i for i, g in enumerate(genes)}
    E = pd.DataFrame(rng.normal(size=(40, 64)) + 5.0, index=genes[5:45])     # large shared offset
    pcs, M, info = seq.esm_features(E, gi, len(genes), n_pc=8)
    assert pcs.shape == (40, 8) and list(pcs.index) == genes[5:45]
    assert info["n_pc"] == 8 and 0 < info["explained_variance_ratio_cum"] < 1
    fs.check_square(M, len(genes), "esm_toy", sim=True)
    assert np.isnan(M[:5]).all() and np.isnan(M[45:]).all()
    X = E.values - E.values.mean(0)
    V = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.allclose(M[5, 6], V[0] @ V[1], atol=1e-5)
    assert abs(np.nanmean(M)) < 0.1                   # centring removes the shared offset
    # PCs are uncorrelated and ordered by variance
    v = pcs.var().values
    assert (np.diff(v) <= 1e-9).all()


def test_koprot_sig_mask():
    ko = _load_script("build_koprot")
    Z = np.array([[3.5, -4.0, 2.0, np.nan]])
    R = np.array([[1.0, -0.3, 2.0, 1.0]])
    assert ko.sig_mask(Z, R).tolist() == [[True, False, False, False]]


# ======================================================================= store tests
@pytest.mark.parametrize("group", sorted(GROUPS))
def test_group_outputs_and_manifest(group):
    if not _group_built(group):
        pytest.skip(f"{group} not built")
    man = _manifest()
    allowed = _feature_datasets()
    G = len(fs.genes())
    for k in EXPECTED[group]:
        e = man[k]
        assert e["group"] == group, k
        assert set(e["sources"]) <= allowed, f"{k}: sources {set(e['sources']) - allowed} not use=feature"
        for s in list(e["sources"]) + list(e.get("source_files", [])):
            assert not FORBIDDEN.search(s), f"{k}: forbidden source '{s}'"
        if "columns" in e:
            for c, ce in e["columns"].items():
                assert set(ce.get("sources", e["sources"])) <= allowed, (k, c)
        p = FEAT / k
        if k.startswith(("pair/", "sim/")):
            assert fs.sha256(p) == e["sha256"], f"{k}: sha256 mismatch"
            M = np.load(p)
            fs.check_square(M, G, k, sim=k.startswith("sim/"))     # shape vs genes.txt, symmetry, NaN diag
            assert np.isfinite(M).any(), f"{k}: all NaN"
            if k.startswith("pair/"):
                assert Path(k).stem.split("__")[0] == group
        elif k.startswith("directed/"):
            M = np.load(p, mmap_mode="r")
            assert M.shape == (G, G) and M.dtype == np.float32 and np.isnan(np.diagonal(M)).all()
        else:
            df = pd.read_parquet(p)
            assert set(df.index) <= set(fs.genes()) and df.index.is_unique
            assert all(c.startswith(f"{group}__") for c in df.columns)
            assert all(t == np.float32 for t in df.dtypes)
            assert (df.notna().sum() > 0).all(), f"{k}: an all-NaN column"


def test_koprot_symmetric_features_match_directed():
    if not _group_built("KOPROT"):
        pytest.skip("KOPROT not built")
    D = np.load(FEAT / "directed/KOPROT__z_ko_row_protein_col.npy", mmap_mode="r")
    A = np.load(FEAT / "pair/KOPROT__a_changes_b.npy", mmap_mode="r")
    Zm = np.load(FEAT / "pair/KOPROT__min_z.npy", mmap_mode="r")
    rows = np.flatnonzero(np.isfinite(np.asarray(D[:, :50])).any(1))[:40]
    cols = np.flatnonzero(np.isfinite(np.asarray(D[rows[0]])))[:200]
    for i in rows:
        for j in cols:
            if i == j:
                continue
            d1, d2 = D[i, j], D[j, i]
            exp_max = np.nanmax([abs(d1), abs(d2)]) if np.isfinite([d1, d2]).any() else np.nan
            exp_min = np.nanmin([d1, d2]) if np.isfinite([d1, d2]).any() else np.nan
            assert np.isclose(A[i, j], exp_max, equal_nan=True), (i, j)
            assert np.isclose(Zm[i, j], exp_min, equal_nan=True), (i, j)


def test_seq_paralog_flags_consistent():
    if not _group_built("SEQ"):
        pytest.skip("SEQ not built")
    P = np.load(FEAT / "pair/SEQ__paralog_identity.npy")
    F = np.load(FEAT / "pair/SEQ__paralog.npy")
    Rr = np.load(FEAT / "pair/SEQ__paralog_recent.npy")
    off = ~np.eye(len(P), dtype=bool)
    assert np.array_equal(F[off] == 1, P[off] > 0)
    assert np.array_equal(Rr[off] == 1, P[off] >= 40)
    assert (Rr[off] <= F[off]).all()
