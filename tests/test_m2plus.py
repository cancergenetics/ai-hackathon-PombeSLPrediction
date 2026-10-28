"""scripts/run_m2plus.py helpers: C3 blend rule, XSP permutation control, E-TARGET target, variant caches."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("run_m2plus", ROOT / "scripts/run_m2plus.py")
MP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MP)


def test_blend_c3_keeps_score_multiset_and_extremes():
    rng = np.random.default_rng(0)
    m2 = rng.normal(size=500)
    out = MP.blend_c3(m2, m2 * 2 + 1)                 # same order -> unchanged
    assert np.allclose(out, m2)
    lr = rng.normal(size=500)
    out = MP.blend_c3(m2, lr)
    assert np.allclose(np.sort(out), np.sort(m2))     # multiset (pooled position of C3) unchanged
    w = MP.CFG["c3"]["blend_weight_m2"]
    assert 0 < w < 1
    out1 = MP.blend_c3(m2, -m2)                       # opposite orders: ranks tie -> still a permutation
    assert np.allclose(np.sort(out1), np.sort(m2))
    # a row that is top in both keeps the top score
    m2b, lrb = m2.copy(), lr.copy()
    m2b[7], lrb[7] = 99.0, 99.0
    assert MP.blend_c3(m2b, lrb)[7] == 99.0


def test_xsp_permutation_moves_only_xsp_columns():
    genes = [f"g{i}" for i in range(50)]
    gt = pd.DataFrame({"XSP__prop_hat": np.arange(50.0), "PHEN__pc1": np.arange(50.0) * 10}, index=genes)
    p = MP.permute_xsp_rows(gt)
    perm = MP.xsp_perm(50)
    assert np.array_equal(p.XSP__prop_hat.to_numpy(), gt.XSP__prop_hat.to_numpy()[perm])
    assert np.array_equal(p.PHEN__pc1.to_numpy(), gt.PHEN__pc1.to_numpy())
    assert not np.array_equal(perm, np.arange(50))
    assert MP.tables_kw("M2_XSPperm")["extra_key"] and "extra_key" not in MP.tables_kw("M2_XSP")


def test_etarget_target_and_delta():
    s = np.array([-5.0, -2.3, 0.0, 3.0])
    assert np.array_equal(MP.etarget(s), [5.0, 2.3, 0.0, 0.0])
    from src import protocol
    assert MP.huber_delta() == abs(float(protocol.threshold())) > 0


def test_variant_definitions():
    V = MP.VARIANTS
    assert V["M2r"]["lf_exclude"] == ["XSP"] and V["M2r"]["gt_exclude"] == ["XSP"]
    assert V["M2_XSP"]["lf_exclude"] == [] and V["M2_XSP"]["gt_exclude"] == []
    assert V["M2_XSP_LF"]["lf_exclude"] == [] and V["M2_XSP_LF"]["gt_exclude"] == ["XSP"]
    assert MP.model_name("M2_XSP", "bag", "huber", None) == "M2_XSP_bag5_huber"


FEAT = ROOT / "data" / "processed" / "features"
LIVE = (FEAT / "manifest.json").exists() and "gene/XSP.parquet" in json.loads((FEAT / "manifest.json").read_text())


@pytest.mark.skipif(not LIVE, reason="XSP not merged into the live feature store")
def test_m2r_reuses_incumbent_cache_and_xsp_variants_do_not(monkeypatch):
    monkeypatch.setenv("PSL_DATA_ROOT", str(ROOT / "data" / "processed"))
    from src.ryanfeat import SimStore
    sims = SimStore()
    names = {v: MP.cache_name(v, sims) for v in MP.VARIANTS}
    assert names["M2r"] == names["M2_XSP_LF"]                 # same GP inputs
    assert len({names["M2r"], names["M2_XSP"], names["M2_XSPperm"]}) == 3
    reg = ROOT / "results" / "m2" / "feature_build_20261028.json"
    if reg.exists():                                         # incumbent M2's cache (docs/m2.md)
        assert names["M2r"] == json.loads(reg.read_text())["cache_name"]
