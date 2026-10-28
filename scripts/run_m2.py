#!/usr/bin/env python
"""M2: nested-stacked LightGBM (DESIGN.md §4 rule 2, §5) on label-free (LF) + RYAN-fold + NBR + gene-
propensity (GP) features, evaluated through src/evaluate.py. Feature construction: src/models/m2.py.

Per outer fold k (split seed from protocol.yaml unless --split-seed):
  training pairs = data.fold_split(k) training pairs -> data.drop_noncoding (DESIGN §1);
  stacker TRAINING rows = inner eval rows of data.inner_splits (C1/C2/C3-like, 5 inner folds) with
    Ryan-derived features (RYAN, NBR, GP) computed from that inner fold's training pairs only;
  outer EVAL rows = the fold's eval pairs with the same features computed from all outer training pairs;
  LF features from src/pairs.assemble for both (true hypomorph flags on stacker rows, 0 on eval rows).
  Row weights = protocol class weights per inner fold (src/stacking `w`), rescaled to mean 1.
  Model seed s: early stopping on stacker rows of inner fold j = s mod n_inner (class-weighted AP), fitted on
  the gene-grouped training rows for j (src.data.gene_grouped_validation: no row touching a T_j gene). The
  outer eval fold is never used for fitting, early stopping or selection.
Variants (ablations reuse the parent configuration; DESIGN §3):
  M2            LF + RYAN + NBR + GP                     M2_init        M2 + inner-fold B1 logit as init_score
  M1N           LF + NBR  (M2 minus RYAN-fold and GP)    M2_minus_GP    LF + RYAN + NBR
  M2_minus_NBR  LF + RYAN + GP
Stages:
  features   nested tables for every fold, one process per fold (cached; src.data cache helpers)
  tune       <= 6 LightGBM configs (src/models/lgbm.CONFIGS) for M2 on fold 0, inner fold 0 validation
             -> results/m2/tuning.json; rule fixed in advance: keep 'default' unless another config's
             inner-validation AP is higher by >= adoption min_delta
  check      fold-0 inner-validation AP of every variant (development sanity check; no outer eval)
  run        all variants x seeds x folds -> evaluate() per variant (registered)
  shift      train-vs-eval feature shift report -> results/m2/feature_shift_<seed>.json
  compare    paired compare() of the latest clean runs at HEAD: M2 vs M1 (alt = split-seed-7 M2/M1 runs),
             M2 vs B4, M2 vs B1, M2 vs each ablation, M2_init vs M2 -> results/m2/comparisons.json
Usage: run_m2.py --stages features,run,shift [--variants M2,M1N,...] [--split-seed S] [--seeds ladder]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, protocol  # noqa: E402
from src import pairs as PA  # noqa: E402
from src.timing import Timer, Timings, peak_rss_mb, set_threads  # noqa: E402

N_JOBS = 8
SYM_SAMPLE = 20_000
VARIANTS = {
    "M2": {"families": ["LF", "RYAN", "NBR", "GP"], "init": False},
    "M2_init": {"families": ["LF", "RYAN", "NBR", "GP"], "init": True},
    "M1N": {"families": ["LF", "NBR"], "init": False},
    "M2_minus_GP": {"families": ["LF", "RYAN", "NBR"], "init": False},
    "M2_minus_NBR": {"families": ["LF", "RYAN", "GP"], "init": False},
}
SHIFT_FEATURES = ["seen_min", "seen_max", "b1_logit", "beta_max", "mf_score", "prof_corr", "gp_hat_sum",
                  "gp_hat_max", "gp_best_sum", "nbr_mean__support_mean", "nbr_mean__hit_dmean",
                  "nbr_mean__generate_max", "nbr_phen_corr__hit_dmean"]


def out_dir() -> Path:
    d = protocol.results_root() / "m2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_tag(split_seed) -> str:
    s = data._seed(split_seed)
    return "" if s == protocol.load()["dev_cv"]["split_seed"] else f"_split{s}"


def ryan_cfg() -> dict:
    """RYAN/NBR settings = the values tuned once for the baselines (results/baselines/tuning.json)."""
    from src.ryanfeat import DEFAULT_CFG
    cfg = {k: DEFAULT_CFG[k] for k in ("b1_C", "mf_k", "mf_wd", "mf_epochs", "mf_lr")}
    tj = protocol.results_root() / "baselines" / "tuning.json"
    if tj.exists() and protocol.data_root().resolve() == (ROOT / "data" / "processed").resolve():
        ch = json.loads(tj.read_text())["chosen"]
        assert all(abs(float(ch[k]) - float(cfg[k])) < 1e-12 for k in cfg), (ch, cfg)
    return cfg


# ============================================================================================ features
def _fold_features(args):
    split_seed, k, threads, names, rebuild = args
    set_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from src.models import m2
    from src.ryanfeat import SimStore
    tim = {}
    with Timer(f"features_fold{k}") as t:
        sims = SimStore(names=names)
        tr, ev, inner, drop = m2.fold_tables_m2(split_seed, k, sims, ryan_cfg(), use_cache=not rebuild, timings=tim)
    return k, t.result, tim, len(tr), len(ev), drop


def stage_features(split_seed, workers, rebuild=False) -> dict:
    from src.models import m2
    from src.ryanfeat import SimStore
    n = protocol.load()["dev_cv"]["n_folds"]
    names = SimStore().names
    thr = max(1, 10 // workers)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_fold_features, [(split_seed, k, thr, names, rebuild) for k in range(n)]))
    sims = SimStore(names=names)
    out = {"split_seed": data._seed(split_seed), "cache_name": m2.m2_cache_name(m2.resolve_cfg(ryan_cfg()), sims, None),
           "sims": names, "gp_cfg": m2.resolve_gp(None), "ryan_cfg": ryan_cfg(), "workers": workers,
           "threads_per_worker": thr, "rebuilt_ignoring_cache": bool(rebuild),
           "folds": {k: {"timing": t, "builds_s": tim, "n_stacker_rows": a, "n_eval_rows": b, "noncoding_dropped": d}
                     for k, t, tim, a, b, d in res}}
    (out_dir() / f"feature_build_{data._seed(split_seed)}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v["timing"]["wall_s"], v["builds_s"]) for k, v in out["folds"].items()}))
    return out


# ============================================================================================ data
class FoldData:
    """Feature matrices of one outer fold: stacker rows (X, y, w, init) and eval rows."""

    def __init__(self, split_seed, k, spec, sims, T: Timings):
        from src.models import m2
        from src.stacking import stacker_folds
        with T.stage(f"load_features_f{k}"):
            tr, ev, inner, drop = m2.fold_tables_m2(split_seed, k, sims, ryan_cfg())
        self.drop = drop
        rcols = [c for c in tr.columns if m2.family_of(c) != "LF" and c in ev.columns and c not in
                 ("gene_a", "gene_b", "y", "s_score", "part", "inner_fold", "hypomorph_a", "hypomorph_b", "w",
                  "fold")]
        with T.stage(f"assemble_f{k}"):
            Ltr = PA.assemble(tr, spec=spec).to_numpy()
            Lev = PA.assemble(ev, spec=spec).to_numpy()
        self.names = list(spec.feature_names()) + rcols
        self.fam = np.array([m2.family_of(n) for n in self.names])
        self.Xtr = np.hstack([Ltr, tr[rcols].to_numpy(np.float32)])
        self.Xev = np.hstack([Lev, ev[rcols].to_numpy(np.float32)])
        del Ltr, Lev
        self.y = tr.y.to_numpy(np.int8)
        self.w = tr.w.to_numpy(float) / tr.w.mean()
        self.init_tr = tr.b1_logit.to_numpy(float)
        self.init_ev = ev.b1_logit.to_numpy(float)
        self.ev = ev
        self.tr_meta = tr[["gene_a", "gene_b", "part", "inner_fold", "y"]]
        self.groups = {}
        for j, t, v in stacker_folds(tr, inner):
            fm = np.zeros(len(tr), bool)
            vm = np.zeros(len(tr), bool)
            fm[t.index.to_numpy()] = True
            vm[v.index.to_numpy()] = True
            self.groups[j] = (fm, vm)
        self.n_inner = len(inner)
        self.tr_rows, self.ev_rows = tr, ev

    def cols(self, families) -> np.ndarray:
        return np.flatnonzero(np.isin(self.fam, list(families)))


def fit_stacker(D: FoldData, cols, params, seed, init: bool, j: int):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from src.models import lgbm
    fm, vm = D.groups[j]
    p = {**params, "seed": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2,
         "data_random_seed": seed + 3}
    names = lgbm.safe_names(np.array(D.names)[cols])
    Xf = D.Xtr[np.ix_(fm, cols)]
    Xv = D.Xtr[np.ix_(vm, cols)]
    dtr = lgb.Dataset(Xf, label=D.y[fm], weight=D.w[fm], init_score=D.init_tr[fm] if init else None,
                      feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, label=D.y[vm], weight=D.w[vm], init_score=D.init_tr[vm] if init else None,
                      reference=dtr, free_raw_data=True)
    b = lgb.train(p, dtr, num_boost_round=lgbm.MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(lgbm.EARLY_STOP, verbose=False)])
    pv = b.predict(Xv, num_iteration=b.best_iteration, raw_score=True) + (D.init_tr[vm] if init else 0.0)
    info = {"best_iteration": int(b.best_iteration), "inner_fold_valid": int(j),
            "valid_ap_weighted": float(average_precision_score(D.y[vm], pv, sample_weight=D.w[vm])),
            "n_fit": int(fm.sum()), "n_valid": int(vm.sum()), "pos_fit": int(D.y[fm].sum()),
            "pos_valid": int(D.y[vm].sum())}
    return b, info


def predict(b, D: FoldData, cols, init: bool, rows=None, X=None) -> np.ndarray:
    """Raw score (+ outer B1 logit if init) for eval rows `rows` (default all); X overrides the features."""
    rows = np.arange(len(D.ev)) if rows is None else rows
    X = D.Xev[np.ix_(rows, cols)] if X is None else X
    return b.predict(X, num_iteration=b.best_iteration, raw_score=True, num_threads=N_JOBS) + \
        (D.init_ev[rows] if init else 0.0)


# ============================================================================================ tune / check
def stage_tune(split_seed, spec, sims) -> dict:
    from src.models import lgbm
    T = Timings()
    D = FoldData(split_seed, 0, spec, sims, T)
    cols = D.cols(VARIANTS["M2"]["families"])
    out = {"fold": 0, "inner_fold_valid": 0, "variant": "M2", "configs": {}}
    for name in lgbm.CONFIGS:
        with Timer(name) as t:
            _, inf = fit_stacker(D, cols, lgbm.params_for(name, N_JOBS), 0, False, 0)
        out["configs"][name] = {**inf, "wall_s": t.result["wall_s"], "params": lgbm.params_for(name, N_JOBS)}
        print(f"  {name}: inner-valid weighted AP {inf['valid_ap_weighted']:.4f} best_iter {inf['best_iteration']} "
              f"{t.result['wall_s']:.0f}s", flush=True)
    md = protocol.load()["adoption"]["min_delta"]
    best = max(out["configs"], key=lambda n: out["configs"][n]["valid_ap_weighted"])
    gain = out["configs"][best]["valid_ap_weighted"] - out["configs"]["default"]["valid_ap_weighted"]
    out["best"] = best
    out["chosen"] = best if gain >= md else "default"
    out["rule"] = (f"keep 'default' (M1 default LightGBM config) unless another config's fold-0 inner-fold-0 "
                   f"weighted AP exceeds it by >= {md}")
    (out_dir() / "tuning.json").write_text(json.dumps(out, indent=1))
    print(f"best {best} (+{gain:.4f}); chosen {out['chosen']}")
    return out


def chosen_config(arg: str) -> str:
    if arg != "tuned":
        return arg
    return json.loads((out_dir() / "tuning.json").read_text())["chosen"]


def stage_check(split_seed, spec, sims, cfg, variants) -> dict:
    from src.models import lgbm
    T = Timings()
    D = FoldData(split_seed, 0, spec, sims, T)
    out = {}
    for v in variants:
        V = VARIANTS[v]
        _, inf = fit_stacker(D, D.cols(V["families"]), lgbm.params_for(cfg, N_JOBS), 0, V["init"], 0)
        out[v] = inf
        print(f"  {v}: inner-valid weighted AP {inf['valid_ap_weighted']:.4f} best_iter {inf['best_iteration']}",
              flush=True)
    (out_dir() / f"check{seed_tag(split_seed)}.json").write_text(json.dumps(out, indent=1))
    return out


# ============================================================================================ run
def stage_run(split_seed, spec, sims, cfg, variants, seeds, args) -> dict:
    from src.evaluate import compare, evaluate
    from src.models import lgbm
    from src.models import m2
    P = protocol.load()
    n = P["dev_cv"]["n_folds"]
    params = lgbm.params_for(cfg, N_JOBS)
    fb = out_dir() / f"feature_build_{data._seed(split_seed)}.json"
    fbj = json.loads(fb.read_text()) if fb.exists() else None
    T = {v: Timings() for v in variants}
    Tshared = Timings()
    preds = {v: [] for v in variants}
    fits = {v: [] for v in variants}
    imp = {v: [] for v in variants}
    names_by_v = {}
    drops = {}
    t0 = time.perf_counter()
    for k in range(n):
        D = FoldData(split_seed, k, spec, sims, Tshared)
        drops[k] = D.drop
        # symmetry: LF features of swapped eval pairs are identical (Ryan/GP features are symmetric by
        # construction and tested in tests/test_m2.py)
        sw = D.ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})
        Lsw = PA.assemble(sw, spec=spec).to_numpy()
        nlf = len(spec.feature_names())
        assert np.array_equal(Lsw, D.Xev[:, :nlf], equal_nan=True), f"fold {k}: LF features not symmetric"
        Xsw = np.hstack([Lsw, D.Xev[:, nlf:]])
        idx = np.random.default_rng(k).choice(len(D.ev), min(SYM_SAMPLE, len(D.ev)), replace=False)
        for v in variants:
            V = VARIANTS[v]
            cols = D.cols(V["families"])
            names_by_v[v] = np.array(D.names)[cols]
            for s in seeds:
                j = int(s) % D.n_inner
                with T[v].stage(f"train_f{k}_s{s}"):
                    b, inf = fit_stacker(D, cols, params, int(s), V["init"], j)
                with T[v].stage(f"predict_f{k}_s{s}"):
                    sc = predict(b, D, cols, V["init"])
                ssw = predict(b, D, cols, V["init"], rows=idx, X=Xsw[np.ix_(idx, cols)])
                assert np.array_equal(ssw, sc[idx]), f"fold {k} {v} seed {s}: score(a,b) != score(b,a)"
                fits[v].append({"fold": k, "seed": int(s), **inf})
                imp[v].append(lgbm.gain_importance(b))
                preds[v].append(D.ev[["gene_a", "gene_b", "fold", "part"]].assign(score=sc, seed=int(s)))
                print(f"  [{v}] fold {k} seed {s}: best_iter {inf['best_iteration']} inner-valid wAP "
                      f"{inf['valid_ap_weighted']:.4f} ({inf['n_fit']} fit / {inf['n_valid']} valid)", flush=True)
        del D
    shared = Tshared.as_dict()["stages"]
    feat_stage = []
    if fbj:
        feat_stage = [{"stage": "nested_feature_build_all_folds(one-off, cached; shared by all variants)",
                       "wall_s": sum(f["timing"]["wall_s"] for f in fbj["folds"].values()),
                       "cpu_s": sum(f["timing"]["cpu_s"] for f in fbj["folds"].values()),
                       "peak_rss_mb": max(f["timing"]["peak_rss_mb"] for f in fbj["folds"].values()),
                       "threads": fbj["threads_per_worker"], "hardware": shared[0]["hardware"]}]
    fh = {**spec.hashes(), **{f"sim/{k}": v for k, v in sims.hashes().items()},
          "m2feat_cache": m2.m2_cache_name(m2.resolve_cfg(ryan_cfg()), sims, None)}
    summary = {"split_seed": data._seed(split_seed), "config": cfg, "params": params, "seeds": seeds,
               "runs": {}, "noncoding_dropped": drops}
    sp = out_dir() / f"summary{seed_tag(split_seed)}.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        summary["runs"] = old.get("runs", {})
    tag = seed_tag(split_seed)
    for v in variants:
        Tv = Timings()
        Tv.stages = feat_stage + shared + T[v].stages
        name = v + tag
        res = evaluate(pd.concat(preds[v], ignore_index=True), model=name, split_seed=split_seed,
                       runtime=Tv.as_dict(), feature_hashes=fh, register=not args.no_register,
                       notes=f"M2 nested stacking; families={VARIANTS[v]['families']} init_score={VARIANTS[v]['init']}; "
                             f"config={cfg}; seeds={seeds}; training pairs drop_noncoding")
        m, bt = res["metrics_mean"], res["bootstrap"]["primary"]
        gi = pd.Series(np.mean(imp[v], axis=0), index=names_by_v[v]).sort_values(ascending=False)
        fam = gi.groupby(gi.index.map(m2.family_of)).sum().sort_values(ascending=False)
        stage_sum = {}
        for st in Tv.stages:
            key = st["stage"].split("_f")[0] if not st["stage"].startswith("nested") else "feature_build"
            stage_sum[key] = round(stage_sum.get(key, 0) + st["wall_s"], 1)
        summary["runs"][name] = {
            "run_id": res["run_id"], "variant": v, "primary": res["primary"], "se": bt["se"], "ci": bt["ci"],
            "metrics_mean": {kk: m[kk] for kk in ("ap_C1", "ap_C2", "ap_C3", "prevalence_primary", "lift", "auroc",
                                                  "pooled_all_pairs", "pooled_equal", "mean_per_class",
                                                  "precision_at_k", "auroc_C1", "auroc_C2", "auroc_C3",
                                                  "pooled_viable_viable", "pooled_excl_damp_mixed",
                                                  "ap_C1_reciprocal", "ap_C1_single") if kk in m},
            "fold_primary": [res["metrics_per_fold"][f]["pooled"] for f in sorted(res["metrics_per_fold"])],
            "git_commit": res["git_commit"], "git_dirty": res["git_dirty"], "n_features": int(len(names_by_v[v])),
            "importance_by_family": fam.to_dict(), "importance_top30": gi.head(30).to_dict(),
            "fits": fits[v], "runtime_stages_s": stage_sum, "runtime_total_s": Tv.as_dict()["wall_s"],
            "peak_rss_mb": Tv.as_dict()["peak_rss_mb"]}
        print(f"== {name}: run_id {res['run_id']} primary {res['primary']:.4f} (SE {bt['se']:.4f}) "
              f"C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} C3 {m['ap_C3']:.4f}; by family "
              f"{ {kk: round(vv, 3) for kk, vv in fam.items()} }", flush=True)
    summary["total_wall_s"] = round(time.perf_counter() - t0, 1)
    summary["peak_rss_mb"] = peak_rss_mb()
    if not args.no_register:
        sp.write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ============================================================================================ shift
def stage_shift(split_seed, sims) -> dict:
    from scipy.stats import ks_2samp
    from src.models import m2
    n = protocol.load()["dev_cv"]["n_folds"]
    out = {"split_seed": data._seed(split_seed), "folds": {}}
    feats = None
    for k in range(n):
        tr, ev, inner, drop = m2.fold_tables_m2(split_seed, k, sims, ryan_cfg())
        feats = [f for f in SHIFT_FEATURES if f in tr.columns]
        out["features"] = feats
        fk = {}
        for part in ("C1", "C2", "C3"):
            a, b = tr[tr.part == part], ev[ev.part == part]
            fk[part] = {"n_train_rows": int(len(a)), "n_eval_rows": int(len(b))}
            for f in feats:
                x, z = a[f].to_numpy(float), b[f].to_numpy(float)
                xf, zf = x[np.isfinite(x)], z[np.isfinite(z)]
                fk[part][f] = {"train_mean": float(xf.mean()) if len(xf) else None,
                               "eval_mean": float(zf.mean()) if len(zf) else None,
                               "train_sd": float(xf.std()) if len(xf) else None,
                               "eval_sd": float(zf.std()) if len(zf) else None,
                               "train_nan": float(np.mean(~np.isfinite(x))), "eval_nan": float(np.mean(~np.isfinite(z))),
                               "ks": float(ks_2samp(xf, zf).statistic) if len(xf) and len(zf) else None}
        out["folds"][k] = fk
    agg = {}
    for part in ("C1", "C2", "C3"):
        agg[part] = {}
        for f in feats:
            vals = [out["folds"][k][part][f] for k in range(n)]
            agg[part][f] = {kk: (float(np.mean([v[kk] for v in vals])) if all(v[kk] is not None for v in vals) else None)
                            for kk in ("train_mean", "eval_mean", "train_sd", "eval_sd", "train_nan", "eval_nan", "ks")}
    out["fold_mean"] = agg
    (out_dir() / f"feature_shift_{data._seed(split_seed)}.json").write_text(json.dumps(out, indent=1))
    for part in ("C1", "C2", "C3"):
        print(part, {f: (round(agg[part][f]["train_mean"] or np.nan, 3), round(agg[part][f]["eval_mean"] or np.nan, 3),
                         round(agg[part][f]["ks"] or np.nan, 3)) for f in feats})
    return out


# ============================================================================================ compare
def latest_run(model: str, split_seed: int, clean: bool = True) -> str | None:
    """Most recent registered run of `model` on `split_seed` at the current git commit (clean if asked)."""
    from src.evaluate import git_info
    reg = pd.read_csv(protocol.results_root() / "registry.tsv", sep="\t", dtype=str)
    head = git_info()["git_commit"]
    m = (reg.model == model) & (reg.split_seed.astype(int) == int(split_seed)) & (reg.git_commit == head)
    if clean:
        m &= reg.git_dirty == "False"
    return reg.loc[m, "run_id"].iloc[-1] if m.any() else None


def stage_compare(args) -> dict:
    """M2 vs M1 (with the seed-7 partition check), M2 vs B4, M2 vs B1, and every ablation vs M2."""
    from src.evaluate import compare
    s0, s7 = data.split_seeds()
    r = {m: latest_run(m, s0) for m in ("M2", "M1", "B4_logistic_MF", "B1_additive_logit", "M2_init", "M1N",
                                        "M2_minus_GP", "M2_minus_NBR")}
    r7 = {"M2": latest_run("M2" + seed_tag(s7), s7), "M1": latest_run("M1" + seed_tag(s7), s7)}
    print("runs:", r, r7)
    out = {"runs": r, "runs_alt_seed": r7, "comparisons": {}}
    alt = (r7["M2"], r7["M1"]) if all(r7.values()) else None
    plan = [("M2_vs_M1", "M2", "M1", alt), ("M2_vs_B4", "M2", "B4_logistic_MF", None),
            ("M2_vs_B1", "M2", "B1_additive_logit", None),
            ("M2_vs_M1N", "M2", "M1N", None),            # Delta > 0: RYAN-fold + GP help
            ("M2_vs_M2_minus_GP", "M2", "M2_minus_GP", None),
            ("M2_vs_M2_minus_NBR", "M2", "M2_minus_NBR", None),
            ("M2_init_vs_M2", "M2_init", "M2", None)]    # candidate = with init_score
    if r7["M2"] and r7["M1"]:
        plan.append(("M2_vs_M1_split7", None, None, None))
    for name, a, b, al in plan:
        if name == "M2_vs_M1_split7":
            ra, rb = r7["M2"], r7["M1"]
        else:
            ra, rb = r[a], r[b]
        if not (ra and rb):
            print(f"skip {name}: missing run(s) {ra} {rb}")
            continue
        c = compare(ra, rb, alt=al, register=not args.no_register)
        out["comparisons"][name] = c
        print(f"{name}: delta {c['delta']:+.4f} CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] all-pairs "
              f"{c['delta_all_pairs']:+.4f} equal {c['delta_equal']:+.4f} alt {c['alt_delta']} -> {c['reason']}",
              flush=True)
    if not args.no_register:
        (out_dir() / "comparisons.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="features,run,shift")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--seeds", default="ladder", choices=["ladder", "all"])
    ap.add_argument("--params", default="tuned", help="tuned (results/m2/tuning.json) | a key of lgbm.CONFIGS")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="features stage: recompute (and overwrite) cached tables")
    args = ap.parse_args()
    set_threads(N_JOBS)
    from src.ryanfeat import SimStore
    st = args.stages.split(",")
    variants = args.variants.split(",")
    assert all(v in VARIANTS for v in variants), variants
    spec = PA.load_spec()
    viol = PA.source_violations(spec)
    real = protocol.data_root().resolve() == (ROOT / "data" / "processed").resolve()
    if viol and real:
        raise SystemExit("feature sources not on the use=feature allowlist:\n" + "\n".join(viol))
    print(f"data root {protocol.data_root()}; split seed {data._seed(args.split_seed)}; {len(spec.feature_names())} LF features")
    if "features" in st:
        stage_features(args.split_seed, args.workers, args.rebuild)
    sims = SimStore(names=SimStore().names)
    if "tune" in st:
        stage_tune(args.split_seed, spec, sims)
    if "check" in st:
        stage_check(args.split_seed, spec, sims, chosen_config(args.params), variants)
    if "run" in st:
        seeds = protocol.load()["seeds"][args.seeds]
        stage_run(args.split_seed, spec, sims, chosen_config(args.params), variants, seeds, args)
    if "shift" in st:
        stage_shift(args.split_seed, sims)
    if "compare" in st:
        stage_compare(args)
    print(f"peak RSS {peak_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
