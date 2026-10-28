"""Default model inputs are pinned (config/models.yaml default_inputs; src/model_inputs.py): a group (or
similarity source) merged into the feature store later must NOT change the default inputs of run_m1 / run_m2 /
the deliverable specs. This is the XSP default leak: after the XSP merge, bare PA.load_spec() / m2.gene_table()
include XSP."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import model_inputs as MI
from src import pairs as PA
from src import protocol

ROOT = protocol.ROOT
REAL = ROOT / "data" / "processed"
FROZEN = ROOT / "data" / "frozen_m2"
HAVE_REAL = (REAL / "features" / "manifest.json").exists() and (REAL / "splits").exists()
real_only = pytest.mark.skipif(not HAVE_REAL, reason="real feature store / splits not available")


# ============================================================================================ synthetic store
def _write_store(fdir: Path, groups, genes, extra_group=None, seed=0):
    rng = np.random.default_rng(seed)
    (fdir / "gene").mkdir(parents=True, exist_ok=True)
    (fdir / "pair").mkdir(exist_ok=True)
    (fdir / "sim").mkdir(exist_ok=True)
    (fdir / "genes.txt").write_text("\n".join(genes) + "\n")
    G = len(genes)
    man = {"genes.txt": {"group": "META", "sources": ["pombase_ids_viability"]}}
    for g in list(groups) + ([extra_group] if extra_group else []):
        if g == PA.HYPO_GROUP:
            continue
        pd.DataFrame({f"{g}__x": rng.normal(size=G).astype(np.float32)}, index=genes).to_parquet(fdir / f"gene/{g}.parquet")
        man[f"gene/{g}.parquet"] = {"group": g, "sources": ["costanzo_2016"], "sha256": f"h_gene_{g}"}
        M = rng.normal(size=(G, G)).astype(np.float32)
        np.save(fdir / f"pair/{g}__p.npy", (M + M.T) / 2)
        man[f"pair/{g}__p.npy"] = {"group": g, "sources": ["costanzo_2016"], "sha256": f"h_pair_{g}"}
    if extra_group:
        M = rng.normal(size=(G, G)).astype(np.float32)
        np.save(fdir / f"sim/{extra_group.lower()}_sim.npy", (M + M.T) / 2)
        man[f"sim/{extra_group.lower()}_sim.npy"] = {"group": extra_group, "sources": ["costanzo_2016"], "sha256": "h_sim"}
    (fdir / "manifest.json").write_text(json.dumps(man))


def test_new_group_does_not_change_default_inputs(tmp_path):
    groups = MI.default_inputs()["lf_groups"]
    genes = [f"SPAC{i:03d}" for i in range(12)]
    a, b = tmp_path / "a", tmp_path / "b"
    _write_store(a, groups, genes)
    _write_store(b, groups, genes, extra_group="NEWGRP")
    sa, sb = MI.default_spec(a, real=True), MI.default_spec(b, real=True)
    raw = PA.load_spec(b)
    assert "NEWGRP" in raw.groups() and any(n.startswith("NEWGRP") for n in raw.feature_names())  # the hazard
    assert sb.groups() == sa.groups() == sorted(groups)
    assert sb.feature_names() == sa.feature_names()
    assert sb.hashes() == sa.hashes()
    assert not any(n.startswith("NEWGRP") for n in sb.feature_names())
    assert MI.gene_table_exclude(raw, MI.default_lf_groups(raw, real=True)) == ("NEWGRP",)
    assert MI.gene_table_exclude(PA.load_spec(a), MI.default_lf_groups(PA.load_spec(a), real=True)) == ()
    # assembled features of the restricted spec equal the raw spec's subset of groups
    ia, ib = np.array([0, 3, 5]), np.array([1, 7, 11])
    Xr = PA.assemble_arrays(ia, ib, sb)
    Xg = PA.assemble_arrays(ia, ib, raw, groups=groups)
    assert np.array_equal(Xr, Xg, equal_nan=True)
    # toy / synthetic stores (not the real data): the default is that store's own groups
    assert MI.default_lf_groups(raw, real=False) == raw.groups()
    assert MI.default_sims(real=False) is None and MI.default_sims(real=True) == sorted(MI.default_inputs()["sims"])


def test_missing_default_group_is_an_error(tmp_path):
    groups = [g for g in MI.default_inputs()["lf_groups"] if g != "PHEN"]
    _write_store(tmp_path, groups, [f"G{i}" for i in range(5)])
    with pytest.raises(KeyError, match="PHEN"):
        MI.default_spec(tmp_path, real=True)
    with pytest.raises(ValueError, match="HYPO"):
        MI.restrict(PA.load_spec(tmp_path), ["CER"])


def test_default_inputs_are_the_one_config_entry():
    """The incumbent and M2 specs use exactly default_inputs (YAML anchors), and the config has one entry."""
    from src import deliverable as DL
    raw = yaml.safe_load(DL.MODELS_PATH.read_text())
    d = MI.default_inputs()
    assert "HYPO" in d["lf_groups"] and len(set(d["lf_groups"])) == len(d["lf_groups"])
    assert not any(k.startswith("_real") for k in raw), "use default_inputs, not ad-hoc anchors"
    for mid in (DL.default_model(), "M2", "M2r_bag5_huber", "M1"):
        s = DL.model_spec(mid, real=True)
        assert s["lf_groups"] == d["lf_groups"], mid
        if s["kind"] == "m2_stack":
            assert s["sims"] == d["sims"], mid


def test_model_scripts_take_defaults_from_model_inputs():
    """run_m1 / run_m2 must not build their model inputs from the raw 'everything in the store' accessors."""
    for f in ("scripts/run_m1.py", "scripts/run_m2.py"):
        src = (ROOT / f).read_text()
        assert "MI.default_spec()" in src, f
        assert "PA.load_spec()" not in src, f"{f}: bare PA.load_spec() (every group in the store)"
        assert "SimStore()" not in src, f"{f}: bare SimStore() (every similarity in the store)"
        assert "m2.gene_table(" not in src, f
        for call in _calls(src, "fold_tables_m2("):
            assert "exclude_groups=" in call, f"{f}: fold_tables_m2 without exclude_groups: {call}"
        for call in _calls(src, "m2_cache_name("):
            assert "gt_exclude()" in call, f"{f}: m2_cache_name without the GP-input exclusion: {call}"


def _calls(src: str, head: str) -> list:
    """Argument text of every call `head(...)` (balanced parentheses)."""
    out = []
    for m in re.finditer(re.escape(head), src):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.end():i - 1])
    return out


# ============================================================================================ live store
@pytest.fixture
def real_root(monkeypatch):
    monkeypatch.setenv("PSL_DATA_ROOT", str(REAL))
    yield REAL


@real_only
def test_live_default_inputs_equal_the_adopted_m2_snapshot(real_root):
    """On today's store (XSP merged) the defaults are exactly the inputs of the adopted M2 run (data/frozen_m2:
    the feature-store snapshot of M2__20261028T183713__024416)."""
    if not (FROZEN / "features" / "manifest.json").exists():
        pytest.skip("data/frozen_m2 snapshot not available")
    from src.ryanfeat import SimStore
    spec = MI.default_spec()
    ref = PA.load_spec(FROZEN / "features")
    assert spec.groups() == ref.groups()
    assert spec.feature_names() == ref.feature_names()
    assert spec.hashes() == ref.hashes()
    assert MI.default_sim_store().names == SimStore(root=FROZEN).names
    assert MI.default_sim_store().hashes() == SimStore(root=FROZEN).hashes()


@real_only
def test_live_m2_cache_name_is_the_incumbents(real_root):
    sys.path.insert(0, str(ROOT / "scripts"))
    s = importlib.util.spec_from_file_location("run_m2", ROOT / "scripts" / "run_m2.py")
    R = importlib.util.module_from_spec(s)
    s.loader.exec_module(R)
    fb = ROOT / "results" / "m2" / "feature_build_20261028.json"
    if not fb.exists():
        pytest.skip("incumbent feature-build record missing")
    assert R.cache_name(MI.default_sim_store()) == json.loads(fb.read_text())["cache_name"]


def _injected_root(tmp: Path) -> Path:
    """A real-like data root (real splits -> is_real_data) whose feature store = the live store + one new group
    (gene table, pair matrix and similarity matrix)."""
    root = tmp / "processed"
    f = root / "features"
    for sub in ("gene", "pair", "sim"):
        (f / sub).mkdir(parents=True)
        for p in sorted((REAL / "features" / sub).iterdir()):
            os.symlink(p.resolve(), f / sub / p.name)
    os.symlink((REAL / "features" / "genes.txt").resolve(), f / "genes.txt")
    os.symlink((REAL / "splits").resolve(), root / "splits")
    genes = (REAL / "features" / "genes.txt").read_text().split()
    G = len(genes)
    man = json.loads((REAL / "features" / "manifest.json").read_text())
    pd.DataFrame({"ZNEW__score": np.arange(G, dtype=np.float32)}, index=genes).to_parquet(f / "gene" / "ZNEW.parquet")
    man["gene/ZNEW.parquet"] = {"group": "ZNEW", "sources": ["costanzo_2016"], "sha256": "x"}
    M = np.lib.format.open_memmap(f / "pair" / "ZNEW__p.npy", mode="w+", dtype=np.float32, shape=(G, G))
    M[:] = 0.5
    del M
    man["pair/ZNEW__p.npy"] = {"group": "ZNEW", "sources": ["costanzo_2016"], "sha256": "y"}
    os.symlink(f / "pair" / "ZNEW__p.npy", f / "sim" / "znew_sim.npy")
    man["sim/znew_sim.npy"] = {"group": "ZNEW", "sources": ["costanzo_2016"], "sha256": "z"}
    (f / "manifest.json").write_text(json.dumps(man))
    return root


@real_only
def test_live_injected_group_does_not_change_defaults(tmp_path, monkeypatch):
    from src import universe as U
    from src.ryanfeat import SimStore
    monkeypatch.setenv("PSL_DATA_ROOT", str(REAL))
    ref_spec, ref_sims = MI.default_spec(), MI.default_sim_store()
    ref_excl = MI.default_gene_table_exclude()
    from src.models import m2
    ref_gt_hashes = m2.gene_table_hashes(ref_excl)
    root = _injected_root(tmp_path)
    monkeypatch.setenv("PSL_DATA_ROOT", str(root))
    assert U.is_real_data()
    assert "ZNEW" in PA.load_spec().groups() and "znew_sim" in SimStore().names        # the hazard is present
    spec = MI.default_spec()
    assert spec.feature_names() == ref_spec.feature_names() and spec.hashes() == ref_spec.hashes()
    assert MI.default_sim_store().names == ref_sims.names
    assert MI.default_gene_table_exclude() == tuple(sorted(set(ref_excl) | {"ZNEW"}))
    gt = m2.gene_table(MI.default_gene_table_exclude())
    assert not any(c.startswith(("ZNEW", "XSP")) for c in gt.columns)
    assert m2.gene_table_hashes(MI.default_gene_table_exclude()) == ref_gt_hashes   # -> same M2 cache name
