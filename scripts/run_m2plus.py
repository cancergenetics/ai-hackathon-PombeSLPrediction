#!/usr/bin/env python
"""M2+ rung (docs/m2plus.md; pre-stated rules in config/m2plus.yaml): M2 +/- XSP, gene-permuted XSP control,
seed bagging, E-TARGET (Huber stacker objective) and a C3-only low-capacity blend.

Reuses the M2 machinery unchanged (src/models/m2.py nested features, scripts/run_m2.py fit_stacker/predict);
variants differ only in which label-free groups enter (a) the LF pair features and (b) the GP model's gene
inputs, in the stacker objective and in the seed list.

Variants (VARIANTS; families always LF + RYAN + NBR + GP, no init_score):
  M2r          incumbent M2 re-run at this commit: XSP excluded from LF and GP inputs (same cache as M2)
  M2_XSP       XSP in LF pair features and GP gene inputs (adoption candidate)
  M2_XSP_LF    XSP in LF pair features only (diagnostic)
  M2_XSPperm   M2_XSP with XSP gene identities permuted (one permutation of genes.txt; control)
Options: --seeds ladder|bag (config/m2plus.yaml), --objective binary|huber (E-TARGET).
Registered model name = variant [+ _bag5] [+ _huber] [+ _split<seed>].
Stages:
  features  nested tables for the variants' GP configurations (cached; one process per fold)
  run       variants x seeds x folds -> evaluate() (registered); per-seed predictions saved under
            <data root>/preds/m2plus/ (git-ignored)
  c3blend   C3-only blend of a registered run with the low-capacity logistic model (config c3)
  compare   --pairs A:B[:A7:B7],... paired compare() (model names -> latest clean run at HEAD, or run ids)
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data, protocol  # noqa: E402
from src import pairs as PA  # noqa: E402
from src.timing import Timer, Timings, peak_rss_mb, set_threads  # noqa: E402

import run_m2 as R  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "m2plus.yaml").read_text())
THREADS = int(CFG["compute"]["stacker_threads"])
R.N_JOBS = THREADS                       # run_m2.predict uses this thread count
FAMILIES = ["LF", "RYAN", "NBR", "GP"]
VARIANTS = {
    "M2r": {"lf_exclude": ["XSP"], "gt_exclude": ["XSP"], "xsp_perm": False},
    "M2_XSP": {"lf_exclude": [], "gt_exclude": [], "xsp_perm": False},
    "M2_XSP_LF": {"lf_exclude": [], "gt_exclude": ["XSP"], "xsp_perm": False},
    "M2_XSPperm": {"lf_exclude": [], "gt_exclude": [], "xsp_perm": True},
}


def out_dir() -> Path:
    d = protocol.results_root() / "m2plus"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preds_dir() -> Path:
    d = protocol.preds_dir() / "m2plus"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================================================ XSP permutation
def xsp_perm(G: int) -> np.ndarray:
    return np.random.default_rng(int(CFG["xsp"]["perm_seed"])).permutation(G)


def permute_xsp_rows(gt: pd.DataFrame) -> pd.DataFrame:
    """Gene table (genes.txt order) with the XSP columns of gene i replaced by those of gene perm[i]."""
    from src.models import m2  # noqa: F401
    gt = gt.copy()
    cols = [c for c in gt.columns if c.startswith("XSP__")]
    assert cols, "no XSP columns to permute"
    p = xsp_perm(len(gt))
    gt[cols] = gt[cols].to_numpy()[p]
    return gt


def tables_kw(v: str) -> dict:
    V = VARIANTS[v]
    kw = {"exclude_groups": tuple(V["gt_exclude"])}
    if V["xsp_perm"]:
        kw.update(gt_transform=permute_xsp_rows, extra_key={"xsp_perm_seed": int(CFG["xsp"]["perm_seed"])})
    return kw


def tables_key(v: str) -> tuple:
    V = VARIANTS[v]
    return tuple(V["gt_exclude"]), bool(V["xsp_perm"])


def assemble_lf(df: pd.DataFrame, spec, v: str) -> np.ndarray:
    """LF features of variant v (XSP columns gene-permuted for the control)."""
    V = VARIANTS[v]
    X = PA.assemble(df, spec=spec, exclude=V["lf_exclude"]).to_numpy()
    if V["xsp_perm"]:
        names = spec.feature_names(exclude=V["lf_exclude"])
        grp = spec.feature_groups(exclude=V["lf_exclude"])
        xc = [i for i, n in enumerate(names) if grp[n] == "XSP"]
        Xp = PA.assemble(df, spec=spec, groups=["XSP"], perm=xsp_perm(spec.G))
        assert list(Xp.columns) == [names[i] for i in xc]
        X[:, xc] = Xp.to_numpy()
    return X


# ============================================================================================ features
def _fold_features(args):
    split_seed, k, threads, names, v = args
    set_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from src.models import m2
    from src.ryanfeat import SimStore
    tim = {}
    with Timer(f"features_fold{k}") as t:
        sims = SimStore(names=names)
        tr, ev, inner, drop = m2.fold_tables_m2(split_seed, k, sims, R.ryan_cfg(), timings=tim, **tables_kw(v))
    return k, t.result, tim, len(tr), len(ev)


def cache_name(v: str, sims) -> str:
    from src.models import m2
    kw = tables_kw(v)
    return m2.m2_cache_name(m2.resolve_cfg(R.ryan_cfg()), sims, None, kw["exclude_groups"], kw.get("extra_key"))


def stage_features(split_seed, variants) -> dict:
    from src.ryanfeat import SimStore
    n = protocol.load()["dev_cv"]["n_folds"]
    names = SimStore().names
    workers = int(CFG["compute"]["feature_workers"])
    thr = max(1, THREADS // workers)
    out = {}
    done = set()
    for v in variants:
        key = tables_key(v)
        if key in done:
            continue
        done.add(key)
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_fold_features, [(split_seed, k, thr, names, v) for k in range(n)]))
        sims = SimStore(names=names)
        rec = {"split_seed": data._seed(split_seed), "variant_key": [list(key[0]), key[1]],
               "cache_name": cache_name(v, sims), "workers": workers, "threads_per_worker": thr,
               "folds": {k: {"timing": t, "builds_s": tim, "n_stacker_rows": a, "n_eval_rows": b}
                         for k, t, tim, a, b in res}}
        out[rec["cache_name"]] = rec
        p = out_dir() / f"feature_build_{rec['cache_name']}_{data._seed(split_seed)}.json"
        p.write_text(json.dumps(rec, indent=1, default=str))
        print(v, rec["cache_name"], {k: (r["timing"]["wall_s"], r["builds_s"]) for k, r in rec["folds"].items()},
              flush=True)
    return out


# ============================================================================================ data
class FoldDataP(R.FoldData):
    """run_m2.FoldData for an M2+ variant (LF exclusions / XSP permutation / GP-input exclusions)."""

    def __init__(self, split_seed, k, spec, sims, T: Timings, v: str):
        from src.models import m2
        from src.stacking import stacker_folds
        with T.stage(f"load_features_f{k}"):
            tr, ev, inner, drop = m2.fold_tables_m2(split_seed, k, sims, R.ryan_cfg(), **tables_kw(v))
        self.drop = drop
        rcols = [c for c in tr.columns if m2.family_of(c) != "LF" and c in ev.columns and c not in
                 ("gene_a", "gene_b", "y", "s_score", "part", "inner_fold", "hypomorph_a", "hypomorph_b", "w",
                  "fold")]
        with T.stage(f"assemble_f{k}"):
            Ltr = assemble_lf(tr, spec, v)
            Lev = assemble_lf(ev, spec, v)
        self.lf_names = list(spec.feature_names(exclude=VARIANTS[v]["lf_exclude"]))
        self.names = self.lf_names + rcols
        self.fam = np.array([m2.family_of(n) for n in self.names])
        self.Xtr = np.hstack([Ltr, tr[rcols].to_numpy(np.float32)])
        self.Xev = np.hstack([Lev, ev[rcols].to_numpy(np.float32)])
        del Ltr, Lev
        self.y = tr.y.to_numpy(np.int8)
        self.s = tr[protocol.label_column()].to_numpy(float)
        assert np.isfinite(self.s).all()
        self.w = tr.w.to_numpy(float) / tr.w.mean()
        self.init_tr = tr.b1_logit.to_numpy(float)
        self.init_ev = ev.b1_logit.to_numpy(float)
        self.ev = ev
        self.tr_meta = tr[["gene_a", "gene_b", "part", "inner_fold", "y"]]
        self.groups = {}
        for j, t, vv in stacker_folds(tr, inner):
            fm = np.zeros(len(tr), bool)
            vm = np.zeros(len(tr), bool)
            fm[t.index.to_numpy()] = True
            vm[vv.index.to_numpy()] = True
            self.groups[j] = (fm, vm)
        self.n_inner = len(inner)
        self.tr_rows, self.ev_rows = tr, ev


def etarget(s: np.ndarray) -> np.ndarray:
    """E-TARGET regression target t = -min(S2, 0) (config etarget.target = neg_min_s2_0)."""
    assert CFG["etarget"]["target"] == "neg_min_s2_0"
    return -np.minimum(s, 0.0)


def huber_delta() -> float:
    assert CFG["etarget"]["delta"] == "label_threshold_abs"
    return abs(float(protocol.threshold()))


def fit_stacker_huber(D: FoldDataP, cols, params, seed, j: int):
    """Huber regression stacker on t = -min(S2, 0); early stopping on the class-weighted binary AP of the
    protocol label on inner fold j's validation rows (same rows / weights as run_m2.fit_stacker)."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from src.models import lgbm
    fm, vm = D.groups[j]
    t = etarget(D.s)
    p = {**params, "objective": "huber", "alpha": huber_delta(), "metric": "None", "seed": seed,
         "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2, "data_random_seed": seed + 3}
    names = lgbm.safe_names(np.array(D.names)[cols])
    Xf = D.Xtr[np.ix_(fm, cols)]
    Xv = D.Xtr[np.ix_(vm, cols)]
    yv, wv = D.y[vm], D.w[vm]
    dtr = lgb.Dataset(Xf, label=t[fm], weight=D.w[fm], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, label=t[vm], weight=wv, reference=dtr, free_raw_data=True)

    def wap(preds, _ds):
        return "weighted_ap", float(average_precision_score(yv, preds, sample_weight=wv)), True

    b = lgb.train(p, dtr, num_boost_round=lgbm.MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"], feval=wap,
                  callbacks=[lgb.early_stopping(lgbm.EARLY_STOP, verbose=False)])
    pv = b.predict(Xv, num_iteration=b.best_iteration)
    info = {"best_iteration": int(b.best_iteration), "inner_fold_valid": int(j),
            "valid_ap_weighted": float(average_precision_score(yv, pv, sample_weight=wv)),
            "n_fit": int(fm.sum()), "n_valid": int(vm.sum()), "pos_fit": int(D.y[fm].sum()),
            "pos_valid": int(yv.sum()), "huber_delta": huber_delta()}
    return b, info


# ============================================================================================ run
def model_name(v: str, seeds_kind: str, objective: str, split_seed) -> str:
    name = v + ("_bag5" if seeds_kind == "bag" else "") + ("_huber" if objective == "huber" else "")
    return name + R.seed_tag(split_seed)


def stage_run(split_seed, spec, sims, variants, seeds_kind, objective, args) -> dict:
    from src.evaluate import evaluate
    from src.models import lgbm, m2
    seeds = [int(s) for s in CFG["seeds"][seeds_kind]]
    n = protocol.load()["dev_cv"]["n_folds"]
    params = lgbm.params_for("default", THREADS)
    summ_path = out_dir() / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"runs": {}}
    for v in variants:
        name = model_name(v, seeds_kind, objective, split_seed)
        T = Timings()
        preds, fits, imp = [], [], []
        t0 = time.perf_counter()
        for k in range(n):
            D = FoldDataP(split_seed, k, spec, sims, T, v)
            cols = D.cols(FAMILIES)
            nlf = len(D.lf_names)
            sw = D.ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})
            Lsw = assemble_lf(sw, spec, v)
            assert np.array_equal(Lsw, D.Xev[:, :nlf], equal_nan=True), f"fold {k}: LF features not symmetric"
            Xsw = np.hstack([Lsw, D.Xev[:, nlf:]])
            idx = np.random.default_rng(k).choice(len(D.ev), min(R.SYM_SAMPLE, len(D.ev)), replace=False)
            for s in seeds:
                j = s % D.n_inner
                with T.stage(f"train_f{k}_s{s}"):
                    if objective == "huber":
                        b, inf = fit_stacker_huber(D, cols, params, s, j)
                    else:
                        b, inf = R.fit_stacker(D, cols, params, s, False, j)
                with T.stage(f"predict_f{k}_s{s}"):
                    sc = R.predict(b, D, cols, False)
                ssw = R.predict(b, D, cols, False, rows=idx, X=Xsw[np.ix_(idx, cols)])
                assert np.array_equal(ssw, sc[idx]), f"fold {k} {v} seed {s}: score(a,b) != score(b,a)"
                fits.append({"fold": k, "seed": s, **inf})
                imp.append(pd.Series(lgbm.gain_importance(b), index=np.array(D.names)[cols]))
                preds.append(D.ev[["gene_a", "gene_b", "fold", "part"]].assign(score=sc, seed=s))
                print(f"  [{name}] fold {k} seed {s}: best_iter {inf['best_iteration']} inner-valid wAP "
                      f"{inf['valid_ap_weighted']:.4f}", flush=True)
            del D
        P = pd.concat(preds, ignore_index=True)
        stages = T.as_dict()["stages"]
        fb = sorted(out_dir().glob(f"feature_build_{cache_name(v, sims)}_{data._seed(split_seed)}.json"))
        if fb:
            fbj = json.loads(fb[-1].read_text())
            stages = [{"stage": "nested_feature_build_all_folds(one-off, cached)",
                       "wall_s": sum(f["timing"]["wall_s"] for f in fbj["folds"].values()),
                       "cpu_s": sum(f["timing"]["cpu_s"] for f in fbj["folds"].values()),
                       "peak_rss_mb": max(f["timing"]["peak_rss_mb"] for f in fbj["folds"].values()),
                       "threads": fbj["threads_per_worker"], "hardware": stages[0]["hardware"]}] + stages
        Tv = Timings()
        Tv.stages = stages
        fh = {**spec.hashes(exclude=VARIANTS[v]["lf_exclude"]), **{f"sim/{kk}": vv for kk, vv in sims.hashes().items()},
              "m2feat_cache": cache_name(v, sims)}
        res = evaluate(P, model=name, split_seed=split_seed, runtime=Tv.as_dict(), feature_hashes=fh,
                       register=not args.no_register,
                       notes=f"M2+ variant {v} {VARIANTS[v]}; objective={objective}; seeds={seeds}; "
                             f"threads={THREADS}; config/m2plus.yaml")
        if not args.no_register:
            P.to_parquet(preds_dir() / f"{res['run_id']}_perseed.parquet", index=False)
        m, bt = res["metrics_mean"], res["bootstrap"]["primary"]
        gi = pd.concat(imp, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
        fam = gi.groupby(gi.index.map(m2.family_of)).sum()
        grp = spec.feature_groups()
        xsp_imp = float(gi[[i for i in gi.index if grp.get(i) == "XSP"]].sum())
        stage_sum = {}
        for st in Tv.stages:
            key = "feature_build" if st["stage"].startswith("nested") else st["stage"].split("_f")[0]
            stage_sum[key] = round(stage_sum.get(key, 0) + st["wall_s"], 1)
        summary["runs"][name] = {
            "run_id": res["run_id"], "variant": v, "objective": objective, "seeds": seeds,
            "primary": res["primary"], "se": bt["se"], "ci": bt["ci"],
            "metrics_mean": {kk: m[kk] for kk in ("ap_C1", "ap_C2", "ap_C3", "auroc", "pooled_all_pairs",
                                                  "pooled_equal", "precision_at_k") if kk in m},
            "fold_primary": [res["metrics_per_fold"][f]["pooled"] for f in sorted(res["metrics_per_fold"])],
            "git_commit": res["git_commit"], "git_dirty": res["git_dirty"], "n_features": int(len(cols)),
            "importance_by_family": fam.to_dict(), "importance_XSP": xsp_imp,
            "importance_top20": gi.head(20).to_dict(), "fits": fits, "runtime_stages_s": stage_sum,
            "runtime_total_s": Tv.as_dict()["wall_s"], "run_wall_s": round(time.perf_counter() - t0, 1),
            "peak_rss_mb": peak_rss_mb()}
        print(f"== {name}: run_id {res['run_id']} primary {res['primary']:.4f} (SE {bt['se']:.4f}) "
              f"C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} C3 {m['ap_C3']:.4f}; XSP importance {xsp_imp:.3f}; "
              f"stages {stage_sum}", flush=True)
        if not args.no_register:
            summ_path.write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ============================================================================================ C3 blend
def c3_features(rows: pd.DataFrame, spec) -> np.ndarray:
    feats = CFG["c3"]["features"]
    lf = [f for f in feats if f not in rows.columns]
    L = PA.assemble(rows, spec=spec)[lf] if lf else pd.DataFrame(index=rows.index)
    return pd.concat([rows[[f for f in feats if f in rows.columns]], L], axis=1)[feats].to_numpy(float)


def c3_scores(Xtr: np.ndarray, ytr: np.ndarray, Xev: np.ndarray) -> np.ndarray:
    """Logistic regression (config c3) fitted on Xtr, decision function on Xev; imputation / scaling from Xtr."""
    from sklearn.linear_model import LogisticRegression
    from src.models.m2 import _ridge_design
    A = np.vstack([Xtr, Xev])
    Z = _ridge_design(A, np.arange(len(Xtr)))
    lr = LogisticRegression(C=float(CFG["c3"]["C"]), max_iter=5000).fit(Z[:len(Xtr)], ytr)
    return lr.decision_function(Z[len(Xtr):]), lr


def blend_c3(m2_score: np.ndarray, lr_score: np.ndarray) -> np.ndarray:
    """Pre-stated rule (config c3): order C3 rows by r = w pct_rank(M2) + (1-w) pct_rank(LR) and give them
    M2's own C3 scores in that order (score multiset unchanged)."""
    from scipy.stats import rankdata
    w = float(CFG["c3"]["blend_weight_m2"])
    n = len(m2_score)
    r = w * rankdata(m2_score) / n + (1 - w) * rankdata(lr_score) / n
    out = np.empty(n)
    out[np.argsort(r, kind="stable")] = np.sort(m2_score)
    return out


def stage_c3blend(split_seed, spec, sims, base_model: str, v: str, args) -> dict:
    from sklearn.metrics import average_precision_score
    from src.evaluate import evaluate
    from src.models import m2
    rid = latest(base_model)
    per = pd.read_parquet(preds_dir() / f"{rid}_perseed.parquet")
    n = protocol.load()["dev_cv"]["n_folds"]
    T = Timings()
    out, info = [], {}
    for k in range(n):
        with T.stage(f"load_features_f{k}"):
            tr, ev, inner, drop = m2.fold_tables_m2(split_seed, k, sims, R.ryan_cfg(), **tables_kw(v))
        tr3 = tr[tr.part == "C3"].reset_index(drop=True)
        ev3 = ev[ev.part == "C3"].reset_index(drop=True)
        with T.stage(f"c3_fit_f{k}"):
            Xtr, Xev = c3_features(tr3, spec), c3_features(ev3, spec)
            # inner check (development only): fit on C3-like rows of inner folds != 0, score inner fold 0
            m0 = tr3.inner_fold.to_numpy() == 0
            s0, _ = c3_scores(Xtr[~m0], tr3.y.to_numpy()[~m0], Xtr[m0])
            sev, lr = c3_scores(Xtr, tr3.y.to_numpy(), Xev)
        info[k] = {"n_c3_train": int(len(tr3)), "pos_c3_train": int(tr3.y.sum()), "n_c3_eval": int(len(ev3)),
                   "inner0_c3_ap_lr": float(average_precision_score(tr3.y.to_numpy()[m0], s0)),
                   "inner0_prevalence": float(tr3.y.to_numpy()[m0].mean()),
                   "coef": dict(zip(CFG["c3"]["features"], map(float, lr.coef_[0][:len(CFG["c3"]["features"])])))}
        key = ev3[["gene_a", "gene_b"]].assign(lr=sev)
        pk = per[per.fold == k]
        for s, g in pk.groupby("seed"):
            g = g.copy()
            m3 = (g.part == "C3").to_numpy()
            lr_s = g[m3][["gene_a", "gene_b"]].merge(key, on=["gene_a", "gene_b"], how="left").lr.to_numpy()
            assert np.isfinite(lr_s).all() and len(lr_s) == len(ev3)
            sc = g.score.to_numpy().copy()
            sc[m3] = blend_c3(sc[m3], lr_s)
            out.append(g.assign(score=sc))
        print(f"  fold {k}: {info[k]['n_c3_train']} C3-like rows, inner0 LR AP {info[k]['inner0_c3_ap_lr']:.4f} "
              f"(prev {info[k]['inner0_prevalence']:.4f})", flush=True)
    P = pd.concat(out, ignore_index=True)
    name = base_model.replace(R.seed_tag(split_seed), "") + "_c3blend" + R.seed_tag(split_seed)
    res = evaluate(P, model=name, split_seed=split_seed, runtime=T.as_dict(), register=not args.no_register,
                   notes=f"C3-only rank blend (config/m2plus.yaml c3) of {rid}; w_m2={CFG['c3']['blend_weight_m2']}")
    m = res["metrics_mean"]
    print(f"== {name}: run_id {res['run_id']} primary {res['primary']:.4f} C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} "
          f"C3 {m['ap_C3']:.4f}", flush=True)
    rec = {"run_id": res["run_id"], "base_run": rid, "primary": res["primary"],
           "metrics_mean": {kk: m[kk] for kk in ("ap_C1", "ap_C2", "ap_C3")}, "folds": info,
           "runtime_total_s": T.as_dict()["wall_s"]}
    if not args.no_register:
        (out_dir() / f"c3blend_{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


# ============================================================================================ compare
def latest(token: str) -> str:
    if "__" in token:
        return token
    reg = pd.read_csv(protocol.results_root() / "registry.tsv", sep="\t", dtype=str)
    from src.evaluate import git_info
    head = git_info()["git_commit"]
    m = (reg.model == token) & (reg.git_dirty == "False") & (reg.git_commit == head)
    if not m.any():
        raise SystemExit(f"no clean run of {token} at HEAD {head[:7]}")
    return reg.loc[m, "run_id"].iloc[-1]


def stage_compare(pairs: str, args) -> dict:
    from src.evaluate import compare
    p = out_dir() / "comparisons.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for item in pairs.split(","):
        t = item.split(":")
        ra, rb = latest(t[0]), latest(t[1])
        alt = (latest(t[2]), latest(t[3])) if len(t) == 4 else None
        c = compare(ra, rb, alt=alt, register=not args.no_register)
        out[f"{t[0]}_vs_{t[1]}"] = c
        print(f"{t[0]} - {t[1]}: delta {c['delta']:+.4f} CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] all-pairs "
              f"{c['delta_all_pairs']:+.4f} equal {c['delta_equal']:+.4f} alt {c['alt_delta']} -> {c['reason']}",
              flush=True)
    if not args.no_register:
        p.write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="features,run")
    ap.add_argument("--variants", default="M2r,M2_XSP")
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--seeds", default="ladder", choices=["ladder", "bag"])
    ap.add_argument("--objective", default="binary", choices=["binary", "huber"])
    ap.add_argument("--pairs", default="")
    ap.add_argument("--base-model", default="", help="c3blend: registered model name (or run id) to blend")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    set_threads(THREADS)
    from src.ryanfeat import SimStore
    st = args.stages.split(",")
    variants = args.variants.split(",")
    assert all(v in VARIANTS for v in variants), variants
    spec = PA.load_spec()
    viol = PA.source_violations(spec)
    if viol:
        raise SystemExit("feature sources not on the use=feature allowlist:\n" + "\n".join(viol))
    print(f"data root {protocol.data_root()}; split seed {data._seed(args.split_seed)}; threads {THREADS}", flush=True)
    if "features" in st:
        stage_features(args.split_seed, variants)
    sims = SimStore(names=SimStore().names)
    if "run" in st:
        stage_run(args.split_seed, spec, sims, variants, args.seeds, args.objective, args)
    if "c3blend" in st:
        stage_c3blend(args.split_seed, spec, sims, args.base_model, variants[0], args)
    if "compare" in st:
        stage_compare(args.pairs, args)
    print(f"peak RSS {peak_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
