"""Tests for src/pairs.py (symmetric pair-feature assembler) and the M1 provenance rule.

* assemble(a, b) == assemble(b, a) exactly (synthetic store with NaNs, toy store, real store);
  a deliberately asymmetric assembler is caught by the same check.
* Shape / dtype / NaN semantics of min, max, prod; group selection; HYPO = 0 on eval rows.
* Every feature file's sources are use=feature datasets of config/data_allowlist.yaml; forbidden sources
  (FYPO, Ryan, STRING, GO, viability) planted in a manifest are caught.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pairs as PA
from src import protocol

ROOT = protocol.ROOT
REAL_FEAT = ROOT / "data" / "processed" / "features"
TOY_FEAT = ROOT / "data" / "toy" / "features"


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture()
def store(tmp_path) -> Path:
    """6-gene store: gene table with NaNs + a fitness column, a pair matrix, a similarity matrix."""
    genes = [f"G{i}" for i in range(6)]
    fd = tmp_path / "features"
    (fd / "gene").mkdir(parents=True)
    (fd / "pair").mkdir()
    (fd / "sim").mkdir()
    (fd / "genes.txt").write_text("\n".join(genes) + "\n")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A__x": rng.normal(size=6), "A__smf": rng.uniform(0.5, 1.2, 6)}, index=genes,
                      dtype="float32")
    df.loc["G1", "A__x"] = np.nan
    df.loc["G2", ["A__x", "A__smf"]] = np.nan
    df = df.drop(index="G5")              # gene absent from the table -> all NaN
    df.to_parquet(fd / "gene" / "A.parquet")
    M = rng.normal(size=(6, 6)).astype(np.float32)
    M = (M + M.T) / 2
    M[0, 3] = M[3, 0] = np.nan
    np.fill_diagonal(M, np.nan)
    np.save(fd / "pair" / "P__z.npy", M)
    S = np.abs(M) / 10
    np.save(fd / "sim" / "s1.npy", S.astype(np.float32))
    man = {"gene/A.parquet": {"group": "A", "sources": ["costanzo2016"], "sha256": "a"},
           "pair/P__z.npy": {"group": "P", "sources": ["biogrid"], "sha256": "b"},
           "sim/s1.npy": {"group": "S", "sources": ["pombase"], "sha256": "c"},
           "genes.txt": {"group": "META", "sources": ["pombase"], "sha256": "d"}}
    (fd / "manifest.json").write_text(json.dumps(man))
    return fd


def all_pairs(genes, n=None, seed=0):
    a, b = np.triu_indices(len(genes), 1)
    if n is not None and n < len(a):
        i = np.random.default_rng(seed).choice(len(a), n, replace=False)
        a, b = a[i], b[i]
    g = np.asarray(genes, dtype=object)
    return pd.DataFrame({"gene_a": g[a], "gene_b": g[b]})


def swap(p):
    return p.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a", "hypomorph_a": "hypomorph_b",
                             "hypomorph_b": "hypomorph_a"})


def symmetric(assembler, p) -> bool:
    return np.array_equal(assembler(p).to_numpy(), assembler(swap(p)).to_numpy(), equal_nan=True)


# ------------------------------------------------------------------------------------------ symmetry
def test_symmetry_synthetic(store):
    spec = PA.load_spec(store)
    p = all_pairs(spec.genes)
    p["hypomorph_a"] = np.int8([1, 0] * (len(p) // 2) + [0] * (len(p) % 2))
    p["hypomorph_b"] = np.int8(0)
    assert symmetric(lambda d: PA.assemble(d, spec=spec), p)
    assert symmetric(lambda d: PA.assemble(d, spec=spec, chunk=4), p)            # chunking
    perm = np.random.default_rng(1).permutation(spec.G)
    assert symmetric(lambda d: PA.assemble(d, spec=spec, perm=perm), p)


def test_asymmetric_assembler_is_caught(store):
    """Broken variant: raw a-then-b features (no min/max) must fail the symmetry check."""
    spec = PA.load_spec(store)
    T = pd.read_parquet(store / "gene" / "A.parquet").reindex(spec.genes)

    def broken(d):
        return pd.DataFrame(np.c_[T.loc[d.gene_a].to_numpy(), T.loc[d.gene_b].to_numpy()])
    assert not symmetric(broken, all_pairs(spec.genes))


@pytest.mark.parametrize("fdir", [TOY_FEAT, REAL_FEAT], ids=["toy", "real"])
def test_symmetry_store(fdir):
    if not (fdir / "manifest.json").exists():
        pytest.skip(f"{fdir} not built")
    spec = PA.load_spec(fdir)
    p = all_pairs(spec.genes, n=20_000, seed=3)
    X = PA.assemble(p, spec=spec)
    assert X.shape == (len(p), len(spec.feature_names()))
    assert X.dtypes.eq(np.float32).all()
    assert np.array_equal(X.to_numpy(), PA.assemble(swap(p), spec=spec).to_numpy(), equal_nan=True)
    assert not np.isinf(X.to_numpy()).any()
    assert (X.notna().mean() > 0).all(), "a feature column is entirely NaN on random pairs"


# ------------------------------------------------------------------------------------------ semantics
def test_min_max_prod_semantics(store):
    spec = PA.load_spec(store)
    p = all_pairs(spec.genes)
    X = PA.assemble(p, spec=spec)
    T = pd.read_parquet(store / "gene" / "A.parquet").reindex(spec.genes)
    assert list(X.columns) == ["A__x__min", "A__x__max", "A__smf__min", "A__smf__max", "A__smf__prod", "P__z",
                               "S__sim_s1", "HYPO__min", "HYPO__max"]
    for c in ("A__x", "A__smf"):
        a, b = T.loc[p.gene_a, c].to_numpy(), T.loc[p.gene_b, c].to_numpy()
        mn, mx = X[f"{c}__min"].to_numpy(), X[f"{c}__max"].to_numpy()
        either, both = np.isnan(a) | np.isnan(b), np.isnan(a) & np.isnan(b)
        assert np.array_equal(np.isnan(mn), either)                     # min NaN iff either gene missing
        assert np.array_equal(np.isnan(mx), both)                       # max NaN iff both missing
        assert np.allclose(mn[~either], np.minimum(a, b)[~either])
        assert np.allclose(mx[~both], np.fmax(a, b)[~both])
        assert (mn[~either] <= mx[~either]).all()
    a, b = T.loc[p.gene_a, "A__smf"].to_numpy(), T.loc[p.gene_b, "A__smf"].to_numpy()
    assert np.allclose(X["A__smf__prod"], a * b, equal_nan=True)
    M = np.load(store / "pair" / "P__z.npy")
    gi = {g: i for i, g in enumerate(spec.genes)}
    ref = M[p.gene_a.map(gi), p.gene_b.map(gi)]
    assert np.array_equal(X["P__z"].to_numpy(), ref, equal_nan=True)
    assert (X[["HYPO__min", "HYPO__max"]].to_numpy() == 0).all()        # no hypomorph columns -> 0


def test_hypomorph_passthrough(store):
    spec = PA.load_spec(store)
    p = all_pairs(spec.genes).head(4).assign(hypomorph_a=np.int8([1, 0, 1, 0]), hypomorph_b=np.int8([0, 0, 1, 1]))
    X = PA.assemble(p, spec=spec)
    assert X.HYPO__max.tolist() == [1, 0, 1, 1] and X.HYPO__min.tolist() == [0, 0, 1, 0]


def test_group_selection_and_errors(store):
    spec = PA.load_spec(store)
    p = all_pairs(spec.genes)
    assert spec.groups() == ["A", "HYPO", "P", "S"]
    assert list(PA.assemble(p, ["P", "S"], spec=spec).columns) == ["P__z", "S__sim_s1"]
    assert "P__z" not in PA.assemble(p, spec=spec, exclude=["P"]).columns
    full = PA.assemble(p, spec=spec)
    sub = PA.assemble(p, spec=spec, exclude=["A"])
    assert np.array_equal(full[sub.columns].to_numpy(), sub.to_numpy(), equal_nan=True)
    with pytest.raises(KeyError):
        PA.assemble(p, ["NOPE"], spec=spec)
    with pytest.raises(KeyError):
        PA.assemble(pd.DataFrame({"gene_a": ["G0"], "gene_b": ["UNKNOWN"]}), spec=spec)
    ident = np.arange(spec.G)
    assert np.array_equal(PA.assemble(p, spec=spec, perm=ident).to_numpy(), full.to_numpy(), equal_nan=True)


def test_permutation_moves_features_consistently(store):
    spec = PA.load_spec(store)
    perm = np.array([1, 0, 2, 3, 4, 5])          # swap G0 <-> G1
    p = pd.DataFrame({"gene_a": ["G0", "G2"], "gene_b": ["G3", "G3"]})
    q = pd.DataFrame({"gene_a": ["G1", "G2"], "gene_b": ["G3", "G3"]})
    assert np.array_equal(PA.assemble(p, spec=spec, perm=perm).to_numpy(), PA.assemble(q, spec=spec).to_numpy(),
                          equal_nan=True)


def test_hypomorph_zero_on_eval_rows():
    if not (protocol.splits_dir() / "labels_dev.parquet").exists() or not (PA.fs.feat_dir() / "manifest.json").exists():
        pytest.skip("splits / feature store not built for this data root")
    from src import data
    _, ev = data.fold_split(None, 0)
    X = PA.assemble(ev.head(5000), [PA.HYPO_GROUP])
    assert (X.to_numpy() == 0).all()


# ------------------------------------------------------------------------------------------ provenance
def test_real_store_sources_allowlisted():
    if not (REAL_FEAT / "manifest.json").exists():
        pytest.skip("real feature store not built")
    spec = PA.load_spec(REAL_FEAT)
    assert spec.gene_tables or spec.matrices
    assert PA.source_violations(spec) == []


@pytest.mark.parametrize("bad", [["fypo_phenotype_annotations"], ["ryan2012"], ["string"], ["gene_ontology"],
                                 ["pombase_ids_viability"], ["biogrid", "biogrid_genetic"], ["made_up_source"]])
def test_forbidden_sources_are_caught(store, bad):
    man = json.loads((store / "manifest.json").read_text())
    man["pair/P__z.npy"]["sources"] = bad
    (store / "manifest.json").write_text(json.dumps(man))
    assert PA.source_violations(PA.load_spec(store))


def test_forbidden_source_file_is_caught(store):
    man = json.loads((store / "manifest.json").read_text())
    man["gene/A.parquet"]["source_files"] = ["pombase:FYPOviability.tsv"]
    (store / "manifest.json").write_text(json.dumps(man))
    assert PA.source_violations(PA.load_spec(store))


def test_clean_synthetic_store_passes(store):
    assert PA.source_violations(PA.load_spec(store)) == []
