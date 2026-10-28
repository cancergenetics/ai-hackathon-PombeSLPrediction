"""Tests for the deliverable tooling: src/universe.py, src/deliverable.py, scripts/predict_all_pairs.py,
scripts/lockbox_eval.py, scripts/linkage_diagnostic.py.

Fast unit tests run everywhere. Toy end-to-end tests (PSL_DATA_ROOT=data/toy, results in a temp dir; 2 threads) check:
  * the deliverable M2 path reproduces scripts/run_m2.py's dev-CV fold predictions, and the M2r_huber path
    (default model) reproduces scripts/run_m2plus.py's Huber fold predictions;
  * universe scoring: row count n(n-1)/2, no NaN, rank/sort, flags, determinism with fixed seeds, invariance to
    the block sizes (Ryan-feature blocks and label-free blocks), in-sample vs out-of-fold consistency;
  * lockbox_eval refusal paths (untracked / modified / TODO prereg, one-shot, override without reason), a
    completed run with Holm-adjusted contrasts, the access log, and --train-set all gating.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import protocol
from src import universe as U

ROOT = protocol.ROOT
TOY = ROOT / "data" / "toy"
HAVE_TOY = (TOY / "features" / "manifest.json").exists() and (TOY / "splits" / "labels_lockbox.parquet").exists()
toy_only = pytest.mark.skipif(not HAVE_TOY, reason="toy data / toy splits not built")


def load_script(name):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ============================================================================================ unit
@pytest.mark.parametrize("n,chunk", [(2, 1), (7, 3), (50, 100), (31, 17)])
def test_pair_chunks_cover_triu_in_order(n, chunk):
    ia = np.concatenate([a for a, _ in U.pair_chunks(n, chunk)])
    ib = np.concatenate([b for _, b in U.pair_chunks(n, chunk)])
    ta, tb = np.triu_indices(n, 1)
    assert np.array_equal(ia, ta) and np.array_equal(ib, tb)
    assert len(ia) == U.n_pairs(n)


def test_pos_of_key_matches_enumeration():
    pap = load_script("predict_all_pairs")
    n = 23
    ia, ib = np.triu_indices(n, 1)
    assert np.array_equal(pap.pos_of_key(ia.astype(np.int64) * n + ib, n), np.arange(len(ia)))
    idx = pd.Index([f"g{i:02d}" for i in range(n)])
    k = pap.linear_keys(idx, ["g05", "g09", "zz", "g03"], ["g02", "g09", "g01", "g04"], n)
    assert list(k) == [2 * n + 5, -1, -1, 3 * n + 4]


def test_covering_pairs_and_class():
    g = [f"G{i}" for i in range(9)]
    c = U.covering_pairs(g)
    assert set(c.gene_a) | set(c.gene_b) == set(g) and (c.gene_a < c.gene_b).all()
    assert list(U.pair_class([1, 1, 0, 0], [1, 0, 1, 0])) == ["C1", "C2", "C2", "C3"]


def test_gap_and_linked():
    coords = pd.DataFrame({"chrom": [1, 1, 1, 2], "start": [100, 1000, 500_000, 100], "end": [900, 2000, 501_000, 900]},
                          index=["a", "b", "c", "d"])
    same, gap = U.pair_distance(coords, ["a", "a", "a", "b", "x"], ["b", "c", "d", "a", "a"])
    assert list(same) == [True, True, False, True, False]
    assert list(gap) == [100, 499_100, -1, 100, -1]
    # overlapping genes -> gap 0
    s2, g2 = U.pair_distance(pd.DataFrame({"chrom": [3, 3], "start": [10, 50], "end": [100, 60]}, index=["p", "q"]),
                             ["p"], ["q"])
    assert s2[0] and g2[0] == 0


def test_linkage_window_from_config():
    assert U.linked_window_bp() == int(round(float(U.deliverable_cfg()["linkage"]["window_kb"]) * 1000))


def test_coords_files_match_pinned_sha256():
    d = U.coords_dir()
    if not (d / "chromosome_1.gene.coords.tsv").exists():
        pytest.skip("PomBase coordinates not downloaded")
    for f, h in U.deliverable_cfg()["genome"]["sha256"].items():
        assert protocol.sha256_file(d / f) == h
    c = U.load_coords()
    assert set(c.chrom.unique()) == {1, 2, 3} and (c.end >= c.start).all()


def test_holm_and_bootstrap_p():
    lb = load_script("lockbox_eval")
    adj = lb.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert lb.holm({"x": 0.6, "y": 0.5}) == pytest.approx({"y": 1.0, "x": 1.0})
    assert lb.boot_p(np.ones(100)) == 0.0
    assert lb.boot_p(np.r_[np.ones(90), -np.ones(10)]) == pytest.approx(0.2)


def test_todo_detection():
    lb = load_script("lockbox_eval")
    assert lb._has_todo({"a": [1, {"b": "TODO"}]}) and not lb._has_todo({"a": ["M2", 3]})
    P = yaml.safe_load((ROOT / "config" / "lockbox_prereg.yaml").read_text())
    assert lb._has_todo(P), "the committed prereg must stay a template until the PI fills it"
    assert P["protocol_sha256"] == protocol.protocol_sha256()
    assert P["lockbox_genes_sha256"] == protocol.load()["lockbox"]["genes_sha256"]


def test_model_specs_resolve():
    from src import deliverable as DL
    from src.models import stacker
    for mid in yaml.safe_load(DL.MODELS_PATH.read_text())["models"]:
        s = DL.model_spec(mid, real=True)
        assert s["kind"] in ("m2_stack", "lf_lgbm", "b1", "b4")
        assert s.get("objective", "binary") in stacker.OBJECTIVES
        if s.get("objective") == "huber":
            assert not s.get("init_score") and s["etarget"] == {"target": "neg_min_s2_0", "delta": "label_threshold_abs"}
        if s["kind"] in ("m2_stack", "lf_lgbm"):
            assert isinstance(s["lf_groups"], list) and "HYPO" in s["lf_groups"]
        t = DL.model_spec(mid, real=False)
        if "lf_groups" in t:
            assert t["lf_groups"] == "all"


def test_default_model_is_the_huber_incumbent():
    """PI 2026-10-28: incumbent = M2r_huber (Huber stacker, ladder seeds); M2r_bag5_huber and M2 selectable."""
    from src import deliverable as DL
    assert DL.default_model() == "M2r_huber"
    s = DL.model_spec("M2r_huber", real=True)
    assert s["objective"] == "huber" and s["seeds"] == protocol.load()["seeds"]["ladder"]
    assert s["families"] == ["LF", "RYAN", "NBR", "GP"] and "XSP" not in s["lf_groups"]
    assert DL.dev_run_id(s) == "M2r_huber__20261028T205545__7b5e9e"
    b = DL.model_spec("M2r_bag5_huber", real=True)
    assert b["seeds"] == protocol.load()["seeds"]["all"] and b["objective"] == "huber"
    m = DL.model_spec("M2", real=True)
    assert m.get("objective", "binary") == "binary"
    assert DL.model_spec("M2r_huber", real=False).get("dev_run") is None     # toy: latest toy run, not the pin
    pap = (ROOT / "scripts" / "predict_all_pairs.py").read_text()
    assert 'ap.add_argument("--model", default=None' in pap and "DL.default_model(" in pap


def test_lockbox_seed_rule_and_model_validation():
    from src import deliverable as DL
    lb = load_script("lockbox_eval")
    assert lb.resolve_seeds([0, 1, 2], "model") == [0, 1, 2]
    assert lb.resolve_seeds([0, 1, 2], "all") == protocol.load()["seeds"]["all"]
    assert lb.resolve_seeds([0, 1, 2, 3, 4], [7]) == [7]
    P = {**_prereg({"models": DL.MODELS_PATH}, models=("M2r_huber",), contrasts=(("M2r_huber", "B1_additive_logit"),)),
         "seeds": "model"}
    lb.validate_prereg(P, real=False)
    s = lb.model_spec_for("M2r_huber", P)
    assert s["objective"] == "huber" and s["seeds"] == protocol.load()["seeds"]["ladder"]
    assert lb.model_spec_for("M2r_huber", {**P, "seeds": "all"})["seeds"] == protocol.load()["seeds"]["all"]
    with pytest.raises(lb.Refused, match="not in"):
        lb.validate_prereg({**P, "shortlist": ["M2r_huber", "M9"]}, real=False)
    with pytest.raises(lb.Refused, match="seeds"):
        lb.validate_prereg({**P, "seeds": "some"}, real=False)


def test_universe_genes_synthetic(synth_env):
    genes, df = U.universe_genes(feature_genes=None)
    u = pd.read_csv(synth_env["root"] / "data" / "universe_genes.tsv", sep="\t")
    assert genes == sorted(u.gene[u.in_universe]) and not df.viability_unknown.any()


def test_dev_pairs_helper(synth_env):
    from src import data
    d = data.dev_pairs()
    dd = data.dev_pairs(include_train_drop=True)
    assert len(dd) > len(d) and list(d.columns) == data.TRAIN_COLUMNS
    drop = set(synth_env["drop"])
    assert not (d.gene_a.isin(drop) | d.gene_b.isin(drop)).any()
    lb = set((synth_env["root"] / "data" / "lockbox_genes.txt").read_text().split())
    assert not (dd.gene_a.isin(lb) | dd.gene_b.isin(lb)).any()


# ============================================================================================ toy fixtures
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    if not HAVE_TOY:
        pytest.skip("toy data not built")
    tmp = tmp_path_factory.mktemp("deliv")
    old = {k: os.environ.get(k) for k in ("PSL_DATA_ROOT", "PSL_RESULTS_ROOT")}
    os.environ["PSL_DATA_ROOT"] = str(TOY)
    os.environ["PSL_RESULTS_ROOT"] = str(tmp / "results")
    M = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    for k in M["models"]:
        if "seeds" in M["models"][k]:
            M["models"][k]["seeds"] = [0]                 # one seed keeps the toy tests fast
    mf = tmp / "models_toy.yaml"
    mf.write_text(yaml.safe_dump(M))
    yield {"tmp": tmp, "models": mf}
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for p in (TOY / "lockbox").glob("preds_*.parquet"):
        p.unlink()


@toy_only
def test_m2_path_reproduces_dev_cv_fold(toy):
    """Deliverable training on an outer fold's training pairs == scripts/run_m2.py fold predictions."""
    from src import data
    from src import deliverable as DL
    from src import pairs as PA
    from src.models import lgbm
    from src.ryanfeat import SimStore
    from src.timing import Timings
    R = load_script("run_m2")
    R.N_JOBS = 2
    spec = PA.load_spec()
    sims = SimStore(names=SimStore().names)
    D = R.FoldData(None, 1, spec, sims, Timings())
    cols = D.cols(["LF", "RYAN", "NBR", "GP"])
    b, _ = R.fit_stacker(D, cols, lgbm.params_for("default", 2), 0, False, 0)
    ref = R.predict(b, D, cols, False)
    tr, ev = data.fold_split(None, 1)
    ms = DL.model_spec("M2", toy["models"])
    m = DL.TrainedModel(ms, tr, threads=2, sims=sims, lfspec=spec, use_cache=False).fit()
    s = m.predict(ev)
    # identical procedure; the only numerical difference allowed is the MF fit, which the deliverable runs with
    # deterministic torch (run_m2 does not: ~1e-6 run-to-run jitter), so scores agree except for at most a few
    # rows whose MF value sits on a tree threshold
    diff = np.abs(s - ref)
    assert np.mean(diff > 1e-9) < 1e-3 and np.corrcoef(s, ref)[0, 1] > 0.99999, (np.mean(diff > 1e-9), diff.max())
    # with the universe covering genes (deliverable setting) features agree to float32 rounding
    m.cover_genes = spec.genes
    s2 = m.predict(ev)
    assert np.corrcoef(s, s2)[0, 1] > 0.9999 and np.max(np.abs(s - s2)) < 1e-2


@toy_only
def test_huber_path_reproduces_m2plus_fold(toy):
    """Deliverable M2r_huber trained on an outer fold's training pairs == scripts/run_m2plus.py (variant M2r,
    --objective huber) fold predictions: same shared stacker (src/models/stacker.fit_huber), same rows."""
    from src import data
    from src import deliverable as DL
    from src import pairs as PA
    from src.models import lgbm
    from src.ryanfeat import SimStore
    from src.timing import Timings
    sys.path.insert(0, str(ROOT / "scripts"))
    MP = load_script("run_m2plus")
    MP.R.N_JOBS = 2
    spec = PA.load_spec()
    sims = SimStore(names=SimStore().names)
    D = MP.FoldDataP(None, 1, spec, sims, Timings(), "M2r")
    cols = D.cols(MP.FAMILIES)
    b, inf = MP.fit_stacker_huber(D, cols, lgbm.params_for("default", 2), 0, 0)
    ref = MP.R.predict(b, D, cols, False)
    tr, ev = data.fold_split(None, 1)
    ms = DL.model_spec("M2r_huber", toy["models"])
    m = DL.TrainedModel(ms, tr, threads=2, sims=sims, lfspec=spec, use_cache=False).fit()
    assert m.info["objective"] == "huber" and m.info["fits"][0]["huber_delta"] == inf["huber_delta"]
    s = m.predict(ev)
    diff = np.abs(s - ref)
    assert np.mean(diff > 1e-9) < 1e-3 and np.corrcoef(s, ref)[0, 1] > 0.99999, (np.mean(diff > 1e-9), diff.max())
    # the score is the predicted -min(S2, 0) (>= 0 up to the regression's extrapolation), not a log-odds margin
    assert np.median(s) > -0.5


def _run_pap(args):
    pap = load_script("predict_all_pairs")
    assert pap.main(args) == 0


@pytest.fixture(scope="module")
def toy_universe(toy):
    """Two runs on the first 150 toy genes: default blocks (cache on) and tiny blocks without cache."""
    from src import data
    from src import deliverable as DL
    tmp = toy["tmp"]
    # out-of-fold scores for dev pairs, produced by the same code path fold by fold (= run_m2 OOF, see above)
    spec = DL.model_spec(DL.default_model(), toy["models"])     # M2r_huber (the default of predict_all_pairs)
    oof = []
    for k in range(protocol.load()["dev_cv"]["n_folds"]):
        tr, ev = data.fold_split(None, k)
        m = DL.TrainedModel(spec, tr, threads=2, log=lambda *a: None).fit()
        oof.append(ev[["gene_a", "gene_b"]].assign(score=m.predict(ev)))
    pd.concat(oof).to_parquet(tmp / "oof.parquet")
    base = ["--models-file", str(toy["models"]), "--limit-genes", "150", "--oof-file", str(tmp / "oof.parquet"),
            "--threads", "2"]
    _run_pap(base + ["--out", str(tmp / "u1.parquet"), "--summary", str(tmp / "s1.json")])
    _run_pap(base + ["--out", str(tmp / "u2.parquet"), "--summary", str(tmp / "s2.json"), "--no-cache",
                     "--ryan-chunk", "2500", "--lf-chunk", "700"])
    return {"u1": pd.read_parquet(tmp / "u1.parquet"), "u2": pd.read_parquet(tmp / "u2.parquet"),
            "s1": json.loads((tmp / "s1.json").read_text()), "s2": json.loads((tmp / "s2.json").read_text())}


@toy_only
def test_universe_rows_scores_and_flags(toy_universe):
    u, s = toy_universe["u1"], toy_universe["s1"]
    assert s["model"] == "M2r_huber" and s["fit_info"]["objective"] == "huber"
    n = 150
    assert len(u) == n * (n - 1) // 2 == s["checks"]["expected_rows"]
    assert np.isfinite(u.score).all()
    assert (u.gene_a.astype(str) < u.gene_b.astype(str)).all()
    assert not pd.Series(u.gene_a.astype(str) + "|" + u.gene_b.astype(str)).duplicated().any()
    assert (u["rank"].to_numpy() == np.arange(1, len(u) + 1)).all() and (np.diff(u.score) <= 0).all()
    assert set(u["class"].unique()) <= {"C1", "C2", "C3"}
    # in_training pairs are measured and C1
    assert (u.measured_in_ryan[u.in_training] == True).all()  # noqa: E712
    assert (u["class"][u.in_training] == "C1").all()
    assert s["checks"]["cover_rows_identical_across_blocks"]
    # lockbox-touching pairs are never "measured" in dev mode: <NA> (both genes Ryan S2 genes) or False
    lb = set(U.lockbox_gene_list())
    touch = u.gene_a.astype(str).isin(lb) | u.gene_b.astype(str).isin(lb)
    mt = u.measured_in_ryan[touch]
    assert not mt.fillna(False).any() and mt.isna().any()
    assert not u.in_training[touch].any()


@toy_only
def test_universe_deterministic_and_block_invariant(toy_universe):
    """Rerun without cache and with tiny Ryan/LF blocks: identical scores (fixed seeds, exact block padding)."""
    a = toy_universe["u1"].sort_values(["gene_a", "gene_b"]).reset_index(drop=True)
    b = toy_universe["u2"].sort_values(["gene_a", "gene_b"]).reset_index(drop=True)
    assert (a.gene_a.astype(str) == b.gene_a.astype(str)).all()
    assert np.array_equal(a.score.to_numpy(), b.score.to_numpy())
    assert toy_universe["s2"]["checks"]["cover_rows_identical_across_blocks"]


@toy_only
def test_in_sample_vs_oof_consistency(toy_universe):
    c = toy_universe["s1"]["in_sample_vs_oof"]
    assert c is not None and c["n_pairs"] > 1000
    assert c["ap_in_sample"] >= c["ap_out_of_fold"] - 0.01      # in-sample is optimistic (or equal)
    assert c["spearman_in_sample_vs_oof"] > 0.3                 # same signal, scored two ways
    u = toy_universe["u1"]
    assert (u.oof_source[np.isfinite(u.oof_score)] == "dev_cv").all()


# ============================================================================================ lockbox
def _git(d, *a):
    return subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True, check=True)


def _prereg(toy, models=("M2r_huber",), seeds=(0,), contrasts=(("M2r_huber", "B1_additive_logit"),)) -> dict:
    from src import deliverable as DL
    return {"prereg_version": 1, "protocol_sha256": protocol.protocol_sha256(),
            "lockbox_genes_sha256": "toy", "models_file": str(toy["models"]),
            "models_sha256": DL.models_sha256(toy["models"]), "code_commit": "TOY-ONLY-SKIP",
            "shortlist": list(models), "headline_model": models[0], "baselines": ["B1_additive_logit"],
            "seeds": list(seeds), "boosting_rounds": "dev_cv_inner_early_stopping", "metric": {"primary": "pooled"},
            "holm_family": {"alpha": 0.05, "contrasts": [list(c) for c in contrasts]}}


@pytest.fixture(scope="module")
def lockbox_repo(toy):
    d = toy["tmp"] / "prereg_repo"
    d.mkdir()
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@example.org")
    _git(d, "config", "user.name", "t")
    return d


def _lb(args):
    lb = load_script("lockbox_eval")
    return lb.main(args + ["--n-boot", "50", "--threads", "2"]), lb


@toy_only
def test_lockbox_sequence(toy, lockbox_repo):
    d = lockbox_repo
    f = d / "prereg.yaml"
    # 1. untracked prereg -> refused
    f.write_text(yaml.safe_dump(_prereg(toy)))
    rc, lb = _lb(["--prereg", str(f)])
    assert rc == 2
    L = lb.read_log()
    assert L.event.tolist() == ["refused"] and "not committed" in L.detail.iloc[-1]
    # 2. committed then modified -> refused
    _git(d, "add", "prereg.yaml")
    _git(d, "commit", "-q", "-m", "prereg")
    f.write_text(f.read_text() + "# edit\n")
    rc, lb = _lb(["--prereg", str(f)])
    assert rc == 2 and "uncommitted modifications" in lb.read_log().detail.iloc[-1]
    _git(d, "checkout", "--", "prereg.yaml")
    # 3. TODO left -> refused
    P = _prereg(toy)
    P["shortlist"] = ["TODO"]
    t = d / "todo.yaml"
    t.write_text(yaml.safe_dump(P))
    _git(d, "add", "todo.yaml")
    _git(d, "commit", "-q", "-m", "todo")
    rc, lb = _lb(["--prereg", str(t)])
    assert rc == 2 and "TODO" in lb.read_log().detail.iloc[-1]
    # 4. --train-set all before any completed lockbox run -> refused
    with pytest.raises(PermissionError):
        _run_pap(["--models-file", str(toy["models"]), "--train-set", "all", "--limit-genes", "60", "--threads", "2",
                  "--out", str(toy["tmp"] / "all0.parquet"), "--summary", str(toy["tmp"] / "all0.json")])
    # 5. valid -> completed
    rc, lb = _lb(["--prereg", str(f)])
    assert rc == 0
    L = lb.read_log()
    assert L.event.tolist()[-2:] == ["started", "completed"]
    sha = protocol.sha256_file(f)
    assert (L.prereg_sha256.iloc[-1] == sha) and L.user.iloc[-1] and L.git_commit.iloc[-1]
    res = json.loads((protocol.results_root() / "lockbox" / f"lockbox_result_{sha[:12]}.json").read_text())
    for m in ("M2r_huber", "B1_additive_logit"):
        r = res["models"][m]
        assert 0 <= r["primary"] <= 1 and r["primary_ci"][0] <= r["primary"] + 1e-9
        assert all(f"ap_{c}_ci" in r for c in ("C1", "C2", "C3"))
        assert r["n_C1"] > 0 and r["n_C2"] > 0 and r["n_C3"] > 0
    c = res["contrasts"]["M2r_huber - B1_additive_logit"]
    assert c["p_holm"] >= c["p"] and res["shipped_model"] == "M2r_huber"
    assert res["models"]["M2r_huber"]["primary"] > res["models"]["B1_additive_logit"]["primary"]   # toy has signal
    assert res["fits"]["M2r_huber"]["objective"] == "huber"            # lockbox_eval trains the Huber stacker
    assert lb.lockbox_preds_path("M2r_huber").exists()
    # 6. second access -> refused; override without reason -> refused; with reason -> completed + logged
    rc, lb = _lb(["--prereg", str(f)])
    assert rc == 2 and "one-shot" in lb.read_log().detail.iloc[-1]
    rc, lb = _lb(["--prereg", str(f), "--i-understand-second-access"])
    assert rc == 2 and "reason" in lb.read_log().detail.iloc[-1]
    rc, lb = _lb(["--prereg", str(f), "--i-understand-second-access", "--reason", "test of the override path"])
    assert rc == 0
    L = lb.read_log()
    assert L.event.tolist()[-2:] == ["started", "completed"]
    assert L.override.iloc[-1] == "True" and L.reason.iloc[-1] == "test of the override path"
    assert "second access" in L.detail.iloc[-2]
    assert (L.event == "completed").sum() == 2
    # 7. --train-set all now allowed: lockbox pairs become training pairs, lockbox OOF attached
    _run_pap(["--models-file", str(toy["models"]), "--train-set", "all", "--limit-genes", "120", "--threads", "2",
              "--out", str(toy["tmp"] / "all.parquet"), "--summary", str(toy["tmp"] / "all.json"), "--no-oof"])
    u = pd.read_parquet(toy["tmp"] / "all.parquet")
    lbg = set(U.lockbox_gene_list())
    touch = u.gene_a.astype(str).isin(lbg) | u.gene_b.astype(str).isin(lbg)
    assert len(u) == 120 * 119 // 2 and np.isfinite(u.score).all()
    assert u.in_training[touch].any() and u.measured_in_ryan.notna().all()
    assert (u.oof_source == "lockbox").any()


@toy_only
def test_lockbox_toy_skip_refused_on_real_data(toy):
    lb = load_script("lockbox_eval")
    P = {**_prereg(toy), "lockbox_genes_sha256": protocol.load()["lockbox"]["genes_sha256"]}
    with pytest.raises(lb.Refused, match="only allowed on toy data"):
        lb.validate_prereg(P, real=True)
    with pytest.raises(lb.Refused, match="headline_model"):
        lb.validate_prereg({**P, "headline_model": "B4_logistic_MF"}, real=False)
    with pytest.raises(lb.Refused, match="outside shortlist"):
        lb.validate_prereg({**P, "holm_family": {"contrasts": [["M2r_huber", "M1"]]}}, real=False)
    with pytest.raises(lb.Refused, match="not an ancestor"):
        lb.validate_prereg({**P, "code_commit": "0" * 40}, real=False)


# ============================================================================================ linkage
def test_linkage_bin_table_and_wilson():
    ld = load_script("linkage_diagnostic")
    lo, hi = ld.wilson(np.array([0, 5]), np.array([10, 10]))
    assert lo[0] == pytest.approx(0, abs=1e-12) and 0 < hi[0] < 0.35 and lo[1] < 0.5 < hi[1]
    T = ld.bin_table(np.array([5.0, 15.0, 15.0, 2000.0]), np.array([True, False, True, False]), [0, 10, 1000])
    assert T.n_pairs.tolist() == [1, 2, 1] and T.hits.tolist() == [1, 1, 0]
