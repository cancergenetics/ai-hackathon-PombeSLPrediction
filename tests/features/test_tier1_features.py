"""Tests for the tier-1 label-free feature store (src/idmap.py, src/feature_store.py, scripts/features/).

Unit tests run on synthetic inputs; store tests skip when the (git-ignored) feature store is absent.
Run: .venv/bin/python -m pytest tests/features -q
"""
from __future__ import annotations

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
from src.idmap import IDMapper  # noqa: E402

FEAT = fs.feat_dir()
MANIFEST = FEAT / "manifest.json"
HAS_STORE = MANIFEST.exists() and (FEAT / "genes.txt").exists()
store = pytest.mark.skipif(not HAS_STORE, reason="feature store not built")

# Exact DATA.md dataset ids (config/data_allowlist.yaml) that may feed features: every use=feature id. Short
# aliases (`biogrid`, `pombase`) were replaced by exact ids on 2026-10-28 (scripts/features/merge_staged.py
# --fix-aliases). genes.txt (kind gene_order, group META) is not a feature: it may also cite universe-only ids.
# Anything else in a manifest is a failure.
def _allow_uses() -> dict:
    import yaml
    return {k: v.get("use") for k, v in
            yaml.safe_load((ROOT / "config" / "data_allowlist.yaml").read_text())["datasets"].items()}


ALLOWED_SOURCES = {k for k, u in _allow_uses().items() if u == "feature"}
META_SOURCES = ALLOWED_SOURCES | {k for k, u in _allow_uses().items() if u == "universe-only"}
FORBIDDEN = re.compile(r"fypo|gene.?ontology|\bgo\b|go[ _]?slim|go enrich|kegg|string|text.?min|"
                       r"interactions\.tsv|net-ff|published phenotype|ryan|koch|biogrid.*genetic.*(pairs|edges)|"
                       r"data_inventory|results/eda", re.I)


# ----------------------------------------------------------------------------------- unit tests
@pytest.fixture()
def toy_pombase(tmp_path):
    df = pd.DataFrame({
        "gene_systematic_id": ["SPAC1.01", "SPAC1.02c", "SPAC2.01", "SPNCRNA.1", "SPAC3.01"],
        "gene_systematic_id_with_prefix": [""] * 5,
        "gene_name": ["abc1", "def1", "", "rna1", "dup1"],
        "chromosome_id": [""] * 5, "gene_product": [""] * 5, "external_id": [""] * 5,
        "gene_type": ["protein coding gene"] * 3 + ["lncRNA gene", "protein coding gene"],
        "synonyms": ["SPAC9.99", "old1,SHARED", "SHARED", "", "dup1x"]})
    p = tmp_path / "genes.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


def test_idmap_rules(toy_pombase, tmp_path):
    m = IDMapper(toy_pombase, log_dir=tmp_path / "log")
    assert m.resolve("spac1.01") == ("SPAC1.01", "exact")
    assert m.resolve("SPAC1.01.1:pep") == ("SPAC1.01", "exact")        # transcript suffix
    assert m.resolve("SPAC9.99") == ("SPAC1.01", "synonym")              # obsolete sysID
    assert m.resolve("SHARED")[0] is None and m.resolve("SHARED")[1].startswith("ambiguous")
    assert m.resolve("SPAC7.77", "def1") == ("SPAC1.02c", "gene_name")   # name rescue
    assert m.resolve("nothing") == (None, "unmapped")
    res = m.map(["SPAC1.01", "SPAC9.99", "SHARED", "SPNCRNA.1", "zzz"], source="toy")
    assert res["SPAC1.01"] == "SPAC1.01" and res["SPAC9.99"] == "SPAC1.01"
    assert res["SHARED"] is None and res["SPNCRNA.1"] is None and res["zzz"] is None
    log = pd.read_csv(tmp_path / "log/toy.tsv", sep="\t")
    st = set(log.status)
    assert {"synonym", "unmapped", "not_protein_coding", "many_to_one"} <= st
    assert any(s.startswith("ambiguous") for s in st)
    summ = pd.read_csv(tmp_path / "log/summary.tsv", sep="\t").set_index("source")
    assert summ.loc["toy", "n_input"] == 5 and summ.loc["toy", "unmapped"] == 1


def test_check_square_catches_broken_matrices():
    G = 6
    M = np.random.default_rng(0).uniform(-1, 1, (G, G)).astype(np.float32)
    M = (M + M.T) / 2
    np.fill_diagonal(M, np.nan)
    fs.check_square(M, G, "ok", sim=True)
    bad = M.copy()
    bad[0, 1] += 0.1                                  # asymmetric
    with pytest.raises(AssertionError):
        fs.check_square(bad, G, "asym", sim=True)
    bad = M.copy()
    bad[2, 2] = 1.0                                   # finite diagonal
    with pytest.raises(AssertionError):
        fs.check_square(bad, G, "diag", sim=True)
    bad = M.copy()
    bad[0, 1] = bad[1, 0] = 1.5                       # similarity outside [-1, 1]
    with pytest.raises(AssertionError):
        fs.check_square(bad, G, "range", sim=True)
    bad = M.copy()
    bad[0, 1] = np.nan                                # asymmetric NaN pattern
    with pytest.raises(AssertionError):
        fs.check_square(bad, G, "nan", sim=False)
    with pytest.raises(AssertionError):
        fs.check_square(M[:5, :5], G, "shape", sim=False)


def test_feature_scripts_are_label_free():
    """Tier-1 builders must not read Ryan labels or label-derived EDA."""
    pat = re.compile(r"ryan_pairs|ryan_measurements|results/eda|data_inventory|FYPOviability")
    for p in sorted((ROOT / "scripts/features").glob("build_*.py")) + [ROOT / "src/idmap.py",
                                                                         ROOT / "src/orthologs.py"]:
        assert not pat.search(p.read_text()), f"{p} references label/forbidden files"


# ----------------------------------------------------------------------------------- store tests
@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def genes():
    return fs.genes()


@store
def test_genes_txt(genes):
    assert genes == sorted(set(genes))
    pb = pd.read_csv(ROOT / "data/raw/pombase/gene_IDs_names_products.tsv", sep="\t", dtype=str)
    assert set(genes) == set(pb.loc[pb.gene_type == "protein coding gene", "gene_systematic_id"])


@store
def test_manifest_sources_allowlisted(manifest):
    assert manifest, "empty manifest"
    for key, e in manifest.items():
        for k in ("group", "sources", "script", "sha256"):
            assert k in e, f"{key}: missing {k}"
        srcs = set(e["sources"])
        assert srcs, key
        ok = META_SOURCES if e.get("group") == "META" else ALLOWED_SOURCES
        assert srcs <= ok, f"{key}: non-allowlisted sources {srcs - ok}"
        for c, ce in (e.get("columns") or {}).items():
            if isinstance(ce, dict) and "sources" in ce:
                assert set(ce["sources"]) <= ALLOWED_SOURCES, f"{key}:{c}: {set(ce['sources']) - ALLOWED_SOURCES}"
        for s in list(srcs) + list(e.get("source_files", [])):
            assert not FORBIDDEN.search(s.replace("biogrid_genetic_pmids", "")), f"{key}: forbidden source '{s}'"


@store
def test_manifest_files_exist_and_hash(manifest):
    for key, e in manifest.items():
        p = FEAT / key
        assert p.exists(), key
        if p.stat().st_size < 50e6 or key.startswith("sim/"):
            assert fs.sha256(p) == e["sha256"], f"{key}: sha256 mismatch"


@store
def test_matrices_shape_symmetry(manifest, genes):
    G = len(genes)
    keys = [k for k in manifest if k.startswith(("pair/", "sim/"))]
    assert keys
    for k in keys:
        M = np.load(FEAT / k, mmap_mode="r")
        assert M.shape == (G, G) and M.dtype == np.float32, k
        assert np.isnan(np.diagonal(M)).all(), f"{k}: diagonal not NaN"
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(G, 300, replace=False))
        a = np.asarray(M[rows])
        b = np.asarray(M[:, rows]).T
        assert np.array_equal(a, b, equal_nan=True), f"{k}: not symmetric"
        if k.startswith("sim/"):
            v = a[np.isfinite(a)]
            assert v.size and v.min() >= -1 and v.max() <= 1, f"{k}: outside [-1, 1]"
        if k.startswith("pair/"):
            assert Path(k).stem.split("__")[0] == manifest[k]["group"], k


@store
def test_gene_tables(manifest, genes):
    gs = set(genes)
    keys = [k for k in manifest if k.startswith("gene/")]
    assert keys
    for k in keys:
        df = pd.read_parquet(FEAT / k)
        grp = manifest[k]["group"]
        assert set(df.index) <= gs and df.index.is_unique, k
        assert all(c.startswith(f"{grp}__") for c in df.columns), k
        assert all(t == np.float32 for t in df.dtypes), k


@store
def test_ppi_topology_consistent(genes):
    """Shared-neighbour counts and capped shortest paths agree with an independent computation."""
    import networkx as nx
    E = np.nan_to_num(np.load(FEAT / "pair/PPI__edge.npy"))
    S = np.load(FEAT / "pair/PPI__shared_nbrs.npy", mmap_mode="r")
    SP = np.load(FEAT / "pair/PPI__sp.npy", mmap_mode="r")
    g = nx.from_numpy_array(E)
    rng = np.random.default_rng(1)
    nodes = [n for n in g if g.degree(n) > 0]
    for s in rng.choice(nodes, 20, replace=False):
        d = nx.single_source_shortest_path_length(g, int(s), cutoff=4)
        nb = set(g[int(s)])
        for t in rng.choice(nodes, 100):
            if t == s:
                continue
            assert SP[s, t] == d.get(int(t), 5)
            assert S[s, t] == len(nb & set(g[int(t)]))
