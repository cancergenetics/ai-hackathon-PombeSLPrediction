"""scripts/features/merge_staged.py: checksum-verified merge of a staged feature group, alias fix; and the
live store after the XSP merge (XSP visible to src/pairs and the M2 gene table)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402

_spec = importlib.util.spec_from_file_location("merge_staged", ROOT / "scripts/features/merge_staged.py")
MS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MS)


def _store(tmp_path):
    fdir, stg = tmp_path / "features", tmp_path / "staging"
    (fdir / "gene").mkdir(parents=True)
    (stg / "gene").mkdir(parents=True)
    (stg / "pair").mkdir(parents=True)
    pd.DataFrame({"A__x": np.float32([1, 2])}, index=["g1", "g2"]).to_parquet(fdir / "gene/A.parquet")
    live = {"gene/A.parquet": {"group": "A", "sources": ["biogrid", "pombase"], "script": "scripts/features/build_ppi.py",
                               "sha256": fs.sha256(fdir / "gene/A.parquet"),
                               "columns": {"A__x": {"sources": ["biogrid"]}}},
            "genes.txt": {"group": "META", "sources": ["pombase"], "script": "scripts/features/build_genes.py",
                          "sha256": "z"}}
    (fdir / "manifest.json").write_text(json.dumps(live))
    pd.DataFrame({"X__p": np.float32([0.1, 0.2])}, index=["g1", "g2"]).to_parquet(stg / "gene/X.parquet")
    np.save(stg / "pair/X__q.npy", np.eye(2, dtype=np.float32))
    sman = {k: {"group": "X", "sources": ["costanzo2016", "esm2_weights"], "script": "s.py",
                "sha256": fs.sha256(stg / k)} for k in ("gene/X.parquet", "pair/X__q.npy")}
    (stg / "m.json").write_text(json.dumps(sman))
    return fdir, stg, sman


def test_merge_copies_verifies_and_merges(tmp_path):
    fdir, stg, sman = _store(tmp_path)
    rec = MS.merge(stg, "m.json", fdir)
    assert rec["files"] == {"gene/X.parquet": "copied", "pair/X__q.npy": "copied"}
    man = json.loads((fdir / "manifest.json").read_text())
    for k, e in sman.items():
        assert man[k]["sha256"] == e["sha256"] == fs.sha256(fdir / k)
    assert man["gene/A.parquet"]["sources"] == ["biogrid", "pombase"]        # untouched without --fix-aliases
    rec2 = MS.merge(stg, "m.json", fdir)                                      # idempotent
    assert set(rec2["files"].values()) == {"identical"}


def test_merge_rejects_bad_checksum_bad_source_and_conflicts(tmp_path):
    fdir, stg, sman = _store(tmp_path)
    bad = {**sman, "gene/X.parquet": {**sman["gene/X.parquet"], "sha256": "0" * 64}}
    (stg / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        MS.merge(stg, "bad.json", fdir)
    alias = {k: {**e, "sources": ["biogrid"]} for k, e in sman.items()}       # alias, not an exact id
    (stg / "alias.json").write_text(json.dumps(alias))
    with pytest.raises(SystemExit, match="exact use=feature"):
        MS.merge(stg, "alias.json", fdir)
    (fdir / "pair").mkdir()
    np.save(fdir / "pair/X__q.npy", np.zeros((2, 2), np.float32))           # conflicting destination
    with pytest.raises(SystemExit, match="different sha256"):
        MS.merge(stg, "m.json", fdir)
    assert not (fdir / "gene/X.parquet").exists() or fs.sha256(fdir / "gene/X.parquet") == sman["gene/X.parquet"]["sha256"]


def test_fix_aliases(tmp_path):
    fdir, stg, _ = _store(tmp_path)
    rec = MS.merge(stg, "m.json", fdir, do_fix=True)
    man = json.loads((fdir / "manifest.json").read_text())
    assert man["gene/A.parquet"]["sources"] == ["biogrid_physical", "pombase_orthologs"]
    assert man["gene/A.parquet"]["columns"]["A__x"]["sources"] == ["biogrid_physical"]
    assert man["genes.txt"]["sources"] == ["pombase_ids_viability"]
    assert man["gene/A.parquet"]["sha256"] == fs.sha256(fdir / "gene/A.parquet")
    assert len(rec["alias_changes"]) == 3
    assert list(fdir.glob("manifest.json.bak-*"))
    assert MS.merge(stg, "m.json", fdir, do_fix=True)["alias_changes"] == []   # idempotent


# ------------------------------------------------------------------------------- live store (after merge)
FEAT = ROOT / "data" / "processed" / "features"
LIVE = (FEAT / "manifest.json").exists() and "gene/XSP.parquet" in json.loads((FEAT / "manifest.json").read_text())


@pytest.mark.skipif(not LIVE, reason="XSP not merged into the live feature store")
def test_live_xsp_registered_in_pairs_and_m2(monkeypatch):
    monkeypatch.setenv("PSL_DATA_ROOT", str(ROOT / "data" / "processed"))
    from src import pairs as PA
    from src.models import m2
    spec = PA.load_spec()
    assert "XSP" in spec.groups()
    names = spec.feature_names(groups=["XSP"])
    assert set(names) == {"XSP__prop_hat__min", "XSP__prop_hat__max", "XSP__prop_nn_hat__min",
                          "XSP__prop_nn_hat__max", "XSP__ess_hat__min", "XSP__ess_hat__max", "XSP__pair_hat"}
    assert all(m2.family_of(n) == "LF" for n in names)
    # excluding XSP gives exactly the pre-merge feature list / gene table / cache identity
    assert not any(n.startswith("XSP__") for n in spec.feature_names(exclude=["XSP"]))
    gt, gtx = m2.gene_table(), m2.gene_table(exclude_groups=("XSP",))
    assert {c for c in gt.columns if c.startswith("XSP__")} == {"XSP__prop_hat", "XSP__prop_nn_hat", "XSP__ess_hat"}
    assert list(gtx.columns) == [c for c in gt.columns if not c.startswith("XSP__")]
    assert "gene/XSP.parquet" in m2.gene_table_hashes()
    assert "gene/XSP.parquet" not in m2.gene_table_hashes(exclude_groups=("XSP",))
    pairs = pd.DataFrame({"gene_a": spec.genes[:50], "gene_b": spec.genes[50:100]})
    X = PA.assemble(pairs, spec=spec, groups=["XSP"])
    Xs = PA.assemble(pairs.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"}), spec=spec, groups=["XSP"])
    assert np.array_equal(X.to_numpy(), Xs.to_numpy(), equal_nan=True) and X.notna().all().all()


@pytest.mark.skipif(not LIVE, reason="XSP not merged into the live feature store")
def test_live_manifest_exact_source_ids():
    man = json.loads((FEAT / "manifest.json").read_text())
    for k, e in man.items():
        assert not MS.has_alias(e.get("sources", []), MS.feature_ids()), k
