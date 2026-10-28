"""evaluate()/compare() end-to-end on the synthetic root (registry/runs/preds written to a temp dir)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import data, evaluate as E, protocol
from src.timing import Timer, Timings


def preds_for(score_fn, seeds=(0, 1)):
    rows = []
    for k in range(protocol.load()["dev_cv"]["n_folds"]):
        _, ev = data.fold_split(None, k)
        for s in seeds:
            rows.append(ev[["gene_a", "gene_b", "fold", "part"]].assign(score=score_fn(ev, s), seed=s))
    return pd.concat(rows, ignore_index=True)


def oracle(ev, s):   # test-only: reads labels directly to build a perfect predictor
    lab = data.labels_dev().set_index(["gene_a", "gene_b"]).y
    return lab.loc[pd.MultiIndex.from_frame(ev[["gene_a", "gene_b"]])].to_numpy(float)


def rand(ev, s):
    return np.random.default_rng(s).random(len(ev))


def test_evaluate_random_and_oracle_and_registry(synth_env):
    rr = protocol.results_root()
    t = Timings()
    with t.stage("predict"):
        p_rand = preds_for(rand)
    r = E.evaluate(p_rand, model="B0_test", runtime=t.as_dict(), n_boot=60)
    assert abs(r["primary"] - r["metrics_mean"]["prevalence_primary"]) < 0.06
    o = E.evaluate(preds_for(oracle), model="oracle_test", n_boot=60)
    assert o["primary"] == pytest.approx(1.0)
    assert o["bootstrap"]["primary"]["se"] < 1e-9
    reg = pd.read_csv(rr / "registry.tsv", sep="\t")
    assert {r["run_id"], o["run_id"]} <= set(reg.run_id)
    assert reg.set_index("run_id").loc[r["run_id"], "wall_s"] >= 0
    assert json.loads((rr / "runs" / f"{o['run_id']}.json").read_text())["primary"] == pytest.approx(1.0)
    assert (protocol.preds_dir() / f"{o['run_id']}.parquet").exists()
    c = E.compare(o["run_id"], r["run_id"])
    assert c["delta"] > 0.5 and c["ci_lo"] > 0
    assert not c["clean"] or c["adopt"]          # adopted unless the git tree is dirty
    c0 = E.compare(r["run_id"], r["run_id"])
    assert c0["delta"] == 0 and not c0["adopt"]
    assert len(pd.read_csv(rr / "comparisons.tsv", sep="\t")) >= 2


def test_evaluate_rejects_bad_predictions(synth_env):
    p = preds_for(rand, seeds=(0,))
    with pytest.raises(ValueError):
        E.evaluate(p.iloc[1:], model="x", register=False, n_boot=2)          # missing eval row
    bad = p.copy()
    bad.loc[bad.index[0], "part"] = "C3" if bad.part.iloc[0] != "C3" else "C1"
    with pytest.raises(ValueError):
        E.evaluate(bad, model="x", register=False, n_boot=2)                 # wrong part label
    sw = p.copy()
    sw[["gene_a", "gene_b"]] = sw[["gene_b", "gene_a"]].to_numpy()
    with pytest.raises(ValueError):
        E.evaluate(sw, model="x", register=False, n_boot=2)                  # unsorted pair
    nan = p.copy()
    nan.loc[nan.index[0], "score"] = np.nan
    with pytest.raises(ValueError):
        E.evaluate(nan, model="x", register=False, n_boot=2)


def test_seed_averaging(synth_env):
    p = preds_for(rand, seeds=(0, 1, 2))
    df, seeds = E.prepare(p)
    assert seeds == [0, 1, 2]
    exp = p.groupby(["gene_a", "gene_b", "fold"]).score.mean()
    got = df.set_index(["gene_a", "gene_b", "fold"]).score
    assert np.allclose(got.loc[exp.index], exp)


def test_timer():
    import time
    with Timer("x") as t:
        time.sleep(0.05)
        sum(range(10**6))
    for k in ("wall_s", "cpu_s", "peak_rss_mb", "threads", "hardware"):
        assert k in t.result
    assert t.result["wall_s"] >= 0.05 and t.result["peak_rss_mb"] > 0
