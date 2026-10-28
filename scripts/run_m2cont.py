#!/usr/bin/env python
"""E-CONT rung (docs/m2cont.md; pre-stated rules in config/m2cont.yaml): gene-propensity target and NBR rates
built from the continuous t = -min(S2, 0) instead of binary hits, on top of the incumbent M2r_huber.

Mechanics
* The incumbent's nested feature tables (cache m2feat_3d35f8c314e9: LF + RYAN + NBR + GP, XSP excluded) are
  reused unchanged. The NEW columns (GPC: gpc_*, NBRC: nbrc_*) are built by the same nested machinery
  (src/stacking inner_rows / outer_rows with src/models/m2.M2Builder, options gp_cfg['cont'] and
  cfg['cont_nbr'] of src/ryanfeat): stacker rows of inner fold j from inner fold j's training pairs, eval rows
  from all outer-training pairs. They are cached as `m2cont_<hash>` add-on tables (same row order as the
  incumbent tables; asserted) and joined onto the incumbent rows. MF / profile correlation are switched off in
  the add-on build (not used by any new column); the recomputed binary NBR/GP/beta columns are compared with
  the incumbent cache and the max |diff| is recorded.
* Stacker = the incumbent's Huber stacker, imported unchanged from scripts/run_m2plus.py (fit_stacker_huber,
  FoldDataP for variant M2r, assemble_lf) and scripts/run_m2.py (predict). 3 seeds, 4 threads.

Stages
  toycheck  pre-stated toy known-answer check (needs PSL_DATA_ROOT=data/toy)
  canary    pipeline-level canary on the toy data (needs PSL_DATA_ROOT=data/toy; outer fold 0): S2 of the pairs
            held out of inner fold 0's training set corrupted (labels y kept) -> inner fold 0's stacker rows keep
            bit-identical new columns; positive control: other inner folds' rows and the eval rows move. (The
            outer-level held-out canary is tests/test_m2cont.py: the builder only ever receives training pairs.)
  features  add-on tables for --split-seed (2 folds in parallel x 2 threads); --perm = gene-permuted control
  run       --variants (config variants, M2r_huber = incumbent comparator, <variant>_perm = control)
  compare   --pairs A:B[:A7:B7],...  model names (incumbent from config, others from results/m2cont/summary.json)
            or run ids
  perclass  diagnostic: paired per-class deltas + CI (shared gene bootstrap, one-class weightings), --pairs A:B
  noise     diagnostic: split-half reliability of B1 beta vs continuous theta on real fold-0 training pairs
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
import run_m2plus as P  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "m2cont.yaml").read_text())
THREADS = int(CFG["compute"]["stacker_threads"])
R.N_JOBS = THREADS                       # run_m2.predict thread count
P.THREADS = THREADS
INC_FAMILIES = ["LF", "RYAN", "NBR", "GP"]
GT_EXCLUDE = ("XSP",)                    # incumbent: XSP excluded from LF and GP inputs (run_m2plus M2r)


def variants() -> dict:
    V = {k: list(v["families"]) for k, v in CFG["variants"].items()}
    V["M2r_huber"] = INC_FAMILIES
    for k in CFG["variants"]:
        V[k + "_perm"] = V[k]
    return V


def out_dir() -> Path:
    d = protocol.results_root() / "m2cont"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preds_dir() -> Path:
    d = protocol.preds_dir() / "m2cont"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================================================ add-on build
def addon_cfg(perm: bool) -> tuple[dict, dict]:
    cfg = {**R.ryan_cfg(), "mf": False, "profile": False, "cont_nbr": True}
    if perm:
        cfg["cont_perm_seed"] = int(CFG["adoption"]["perm_seed"])
    return cfg, {"cont": True}


def addon_name(sims, perm: bool) -> str:
    from src.models import m2
    cfg, gp = addon_cfg(perm)
    return m2.m2_cache_name(m2.resolve_cfg(cfg), sims, gp, GT_EXCLUDE).replace("m2feat_", "m2cont_")


def incumbent_name(sims) -> str:
    from src.models import m2
    return m2.m2_cache_name(m2.resolve_cfg(R.ryan_cfg()), sims, None, GT_EXCLUDE)


def is_new(c: str) -> bool:
    from src.models import m2
    return m2.family_of(c) in ("GPC", "NBRC")


def _diff(x: pd.DataFrame, inc: pd.DataFrame | None) -> dict:
    """max |diff| of recomputed binary columns vs the incumbent cache, per family."""
    from src.models import m2
    if inc is None:
        return {"incumbent_cache": "missing"}
    assert (x.gene_a.to_numpy() == inc.gene_a.to_numpy()).all() and (x.gene_b.to_numpy() == inc.gene_b.to_numpy()).all()
    out = {}
    for c in x.columns:
        if c in inc.columns and m2.family_of(c) in ("RYAN", "NBR", "GP") and c not in ("mf_score", "prof_corr"):
            a, b = x[c].to_numpy(float), inc[c].to_numpy(float)
            nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
            d = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
            f = m2.family_of(c)
            out[f] = max(out.get(f, 0.0), float(d))
            out[f + "_nan_mismatch"] = out.get(f + "_nan_mismatch", 0) + nan_mismatch
    return out


def build_addon(split_seed, k, sims, perm: bool, use_cache=True, timings=None, diffs=None):
    """Add-on tables (stacker rows, eval rows) with gene_a, gene_b, part + new columns for outer fold k."""
    from src import stacking
    from src.models import m2
    cfg, gp = addon_cfg(perm)
    name = addon_name(sims, perm)
    inc = incumbent_name(sims)
    train, ev, drop = m2.fold_pairs(split_seed, k)
    inner = data.inner_splits(train)
    B = m2.make_builder(m2.gene_table(GT_EXCLUDE), gp)
    timings = timings if timings is not None else {}
    diffs = diffs if diffs is not None else {}
    parts = []
    for f in inner:
        x = data.cache_load(name, split_seed, k, inner=f.j) if use_cache else None
        if x is None:
            t0 = time.perf_counter()
            full = stacking.inner_rows(f, sims, cfg, B)
            timings[f"inner{f.j}"] = round(time.perf_counter() - t0, 1)
            if not perm:
                diffs[f"inner{f.j}"] = _diff(full, data.cache_load(inc, split_seed, k, inner=f.j))
            x = full[["gene_a", "gene_b", "part"] + [c for c in full.columns if is_new(c)]]
            if use_cache:
                data.cache_save(x, name, split_seed, k, inner=f.j)
        parts.append(x)
    e = data.cache_load(name, split_seed, k) if use_cache else None
    if e is None:
        t0 = time.perf_counter()
        full = stacking.outer_rows(train, ev, sims, cfg, B)
        timings["outer"] = round(time.perf_counter() - t0, 1)
        if not perm:
            diffs["outer"] = _diff(full, data.cache_load(inc, split_seed, k))
        e = full[["gene_a", "gene_b", "part"] + [c for c in full.columns if is_new(c)]]
        if use_cache:
            data.cache_save(e, name, split_seed, k)
    return pd.concat(parts, ignore_index=True), e


def _fold_addon(args):
    split_seed, k, threads, names, perm = args
    set_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from src.ryanfeat import SimStore
    tim, dif = {}, {}
    with Timer(f"addon_fold{k}") as t:
        sims = SimStore(names=names)
        tr, ev = build_addon(split_seed, k, sims, perm, timings=tim, diffs=dif)
    return k, t.result, tim, dif, len(tr), len(ev)


def stage_features(split_seed, perm: bool) -> dict:
    from src.ryanfeat import SimStore
    n = protocol.load()["dev_cv"]["n_folds"]
    names = SimStore().names
    workers = int(CFG["compute"]["feature_workers"])
    thr = max(1, THREADS // workers)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_fold_addon, [(split_seed, k, thr, names, perm) for k in range(n)]))
    sims = SimStore(names=names)
    rec = {"split_seed": data._seed(split_seed), "perm": perm, "addon_cache": addon_name(sims, perm),
           "incumbent_cache": incumbent_name(sims), "workers": workers, "threads_per_worker": thr,
           "folds": {k: {"timing": t, "builds_s": tim, "max_abs_diff_vs_incumbent": dif, "n_stacker_rows": a,
                         "n_eval_rows": b} for k, t, tim, dif, a, b in res}}
    p = out_dir() / f"feature_build_{rec['addon_cache']}_{data._seed(split_seed)}.json"
    p.write_text(json.dumps(rec, indent=1, default=str))
    for k, r in rec["folds"].items():
        print(f"fold {k}: wall {r['timing']['wall_s']} s builds {r['builds_s']} diff {r['max_abs_diff_vs_incumbent']}",
              flush=True)
    return rec


# ============================================================================================ run
class FoldDataC(P.FoldDataP):
    """Incumbent fold data (run_m2plus FoldDataP, variant M2r) + the E-CONT add-on columns appended."""

    def __init__(self, split_seed, k, spec, sims, T: Timings, perm: bool | None):
        super().__init__(split_seed, k, spec, sims, T, "M2r")
        if perm is None:                                   # incumbent comparator: no add-on
            return
        from src.models import m2
        with T.stage(f"load_addon_f{k}"):
            name = addon_name(sims, perm)
            inner = data.inner_splits(m2.fold_pairs(split_seed, k)[0])
            parts = []
            for f in inner:
                x = data.cache_load(name, split_seed, k, inner=f.j)
                assert x is not None, f"add-on {name} fold {k} inner {f.j} missing: run --stages features"
                parts.append(x)
            atr = pd.concat(parts, ignore_index=True)
            aev = data.cache_load(name, split_seed, k)
            assert aev is not None
        for a, b in ((atr, self.tr_rows), (aev, self.ev_rows)):
            assert len(a) == len(b)
            assert (a.gene_a.to_numpy() == b.gene_a.to_numpy()).all()
            assert (a.gene_b.to_numpy() == b.gene_b.to_numpy()).all()
            assert (a.part.to_numpy() == b.part.to_numpy()).all()
        new = [c for c in atr.columns if is_new(c)]
        assert new == [c for c in aev.columns if is_new(c)]
        self.Xtr = np.hstack([self.Xtr, atr[new].to_numpy(np.float32)])
        self.Xev = np.hstack([self.Xev, aev[new].to_numpy(np.float32)])
        self.names = self.names + new
        self.fam = np.array([m2.family_of(n) for n in self.names])


def stage_run(split_seed, spec, sims, vnames, args) -> dict:
    from src.evaluate import evaluate
    from src.models import lgbm, m2
    V = variants()
    seeds = [int(s) for s in CFG["seeds"]]
    n = protocol.load()["dev_cv"]["n_folds"]
    params = lgbm.params_for("default", THREADS)
    summ_path = out_dir() / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {"runs": {}}
    for v in vnames:
        fams = V[v]
        perm = None if v == "M2r_huber" else v.endswith("_perm")
        name = v + R.seed_tag(split_seed)
        T = Timings()
        preds, fits, imp = [], [], []
        t0 = time.perf_counter()
        for k in range(n):
            D = FoldDataC(split_seed, k, spec, sims, T, perm)
            cols = D.cols(fams)
            nlf = len(D.lf_names)
            sw = D.ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})
            Lsw = P.assemble_lf(sw, spec, "M2r")
            assert np.array_equal(Lsw, D.Xev[:, :nlf], equal_nan=True), f"fold {k}: LF features not symmetric"
            Xsw = np.hstack([Lsw, D.Xev[:, nlf:]])
            idx = np.random.default_rng(k).choice(len(D.ev), min(R.SYM_SAMPLE, len(D.ev)), replace=False)
            for s in seeds:
                j = s % D.n_inner
                with T.stage(f"train_f{k}_s{s}"):
                    b, inf = P.fit_stacker_huber(D, cols, params, s, j)
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
        Pr = pd.concat(preds, ignore_index=True)
        stages = T.as_dict()["stages"]
        fb = None if perm is None else out_dir() / f"feature_build_{addon_name(sims, perm)}_{data._seed(split_seed)}.json"
        if fb is not None and fb.exists():
            fbj = json.loads(fb.read_text())
            stages = [{"stage": "addon_feature_build_all_folds(one-off, cached)",
                       "wall_s": sum(f["timing"]["wall_s"] for f in fbj["folds"].values()),
                       "cpu_s": sum(f["timing"]["cpu_s"] for f in fbj["folds"].values()),
                       "peak_rss_mb": max(f["timing"]["peak_rss_mb"] for f in fbj["folds"].values()),
                       "threads": fbj["threads_per_worker"], "hardware": stages[0]["hardware"]}] + stages
        Tv = Timings()
        Tv.stages = stages
        fh = {**spec.hashes(exclude=list(GT_EXCLUDE)), **{f"sim/{kk}": vv for kk, vv in sims.hashes().items()},
              "m2feat_cache": incumbent_name(sims)}
        if perm is not None:
            fh["m2cont_addon_cache"] = addon_name(sims, perm)
        res = evaluate(Pr, model=name, split_seed=split_seed, runtime=Tv.as_dict(), feature_hashes=fh,
                       register=not args.no_register,
                       notes=f"E-CONT variant {v} families={fams}; Huber stacker (run_m2plus.fit_stacker_huber); "
                             f"seeds={seeds}; threads={THREADS}; config/m2cont.yaml")
        if not args.no_register:
            Pr.to_parquet(preds_dir() / f"{res['run_id']}_perseed.parquet", index=False)
        m, bt = res["metrics_mean"], res["bootstrap"]["primary"]
        gi = pd.concat(imp, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
        fam = gi.groupby(gi.index.map(m2.family_of)).sum()
        stage_sum = {}
        for st in Tv.stages:
            key = "feature_build" if st["stage"].startswith("addon") else st["stage"].split("_f")[0]
            stage_sum[key] = round(stage_sum.get(key, 0) + st["wall_s"], 1)
        summary["runs"][name] = {
            "run_id": res["run_id"], "variant": v, "families": fams, "seeds": seeds,
            "primary": res["primary"], "se": bt["se"], "ci": bt["ci"],
            "metrics_mean": {kk: m[kk] for kk in ("ap_C1", "ap_C2", "ap_C3", "auroc", "pooled_all_pairs",
                                                  "pooled_equal", "precision_at_k") if kk in m},
            "fold_primary": [res["metrics_per_fold"][f]["pooled"] for f in sorted(res["metrics_per_fold"])],
            "git_commit": res["git_commit"], "git_dirty": res["git_dirty"], "n_features": int(len(cols)),
            "importance_by_family": fam.to_dict(), "importance_top20": gi.head(20).to_dict(), "fits": fits,
            "runtime_stages_s": stage_sum, "runtime_total_s": Tv.as_dict()["wall_s"],
            "run_wall_s": round(time.perf_counter() - t0, 1), "peak_rss_mb": peak_rss_mb()}
        print(f"== {name}: run_id {res['run_id']} primary {res['primary']:.4f} (SE {bt['se']:.4f}) "
              f"C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} C3 {m['ap_C3']:.4f}; importance {fam.round(3).to_dict()}; "
              f"stages {stage_sum}", flush=True)
        if not args.no_register:
            summ_path.write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ============================================================================================ compare
def resolve_run(token: str) -> str:
    if "__" in token:
        return token
    if token == CFG["incumbent"]["model"]:
        return CFG["incumbent"]["run_id"]
    s = json.loads((out_dir() / "summary.json").read_text())["runs"]
    return s[token]["run_id"]


def stage_compare(pairs: str, args) -> dict:
    from src.evaluate import compare
    p = out_dir() / "comparisons.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for item in pairs.split(","):
        t = item.split(":")
        ra, rb = resolve_run(t[0]), resolve_run(t[1])
        alt = (resolve_run(t[2]), resolve_run(t[3])) if len(t) == 4 else None
        c = compare(ra, rb, alt=alt, register=not args.no_register)
        out[f"{t[0]}_vs_{t[1]}"] = c
        print(f"{t[0]} - {t[1]}: delta {c['delta']:+.4f} CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] all-pairs "
              f"{c['delta_all_pairs']:+.4f} {c['ci_all_pairs']} equal {c['delta_equal']:+.4f} {c['ci_equal']} "
              f"alt {c['alt_delta']} -> {c['reason']}", flush=True)
    if not args.no_register:
        p.write_text(json.dumps(out, indent=1, default=float))
    return out


# ============================================================================================ toy
def _toy_guard():
    if protocol.data_root().resolve() != (ROOT / "data" / "toy").resolve():
        raise SystemExit("toycheck / canary need PSL_DATA_ROOT=data/toy")


def stage_toycheck() -> dict:
    """Pre-stated known-answer check (config toy_check)."""
    from scipy.stats import spearmanr
    from src.models import m2
    from src.ryanfeat import SimStore, _clean_train, cont_target
    _toy_guard()
    truth = pd.read_csv(protocol.data_root() / "_truth" / "gene_truth.tsv", sep="\t").set_index("gene").beta
    sims = SimStore()
    gt = m2.gene_table()
    n = protocol.load()["dev_cv"]["n_folds"]
    folds = []
    for k in range(n):
        tr, ev, _ = m2.fold_pairs(None, k)
        held = set(data.fold_genes(None, k))
        B = m2.make_builder(gt, {"cont": True})
        b = B(tr, ev[["gene_a", "gene_b"]], sims, {**R.ryan_cfg(), "mf": False, "profile": False})
        b.build()
        beta, theta = b.info["b1"]["beta"], b.info["gpc"]["theta"]
        bh, th = b.info["gp"]["beta_hat"], b.info["gpc"]["theta_hat"]
        ct = _clean_train(tr)
        t = cont_target(ct.s_score)
        long = pd.concat([pd.DataFrame({"g": ct.gene_a, "t": t}), pd.DataFrame({"g": ct.gene_b, "t": t})])
        g = long.groupby("g").t.agg(["mean", "size"])
        al = b.info["gpc"]["alpha"]
        shr = (g["mean"] - t.mean()) * g["size"] / (g["size"] + al)
        tg = [x for x in beta.index if x in truth.index]
        ug = [x for x in bh.index if x in held and x in truth.index]

        def rho(s, genes):
            return float(spearmanr(s.reindex(genes), truth[genes]).correlation)
        r = {"fold": k, "n_train": len(tg), "n_unseen": len(ug), "alpha": al,
             "rho_b1": rho(beta, tg), "rho_theta": rho(theta, tg), "rho_shrunk_mean_t": rho(shr, tg),
             "rho_unseen_gp_b1target": rho(bh, ug), "rho_unseen_gp_conttarget": rho(th, ug),
             "rho_train_oof_gp_b1target": rho(bh, tg), "rho_train_oof_gp_conttarget": rho(th, tg)}
        folds.append(r)
        print(r, flush=True)
    keys = [k for k in folds[0] if k.startswith("rho") or k == "alpha"]
    mean = {k: float(np.mean([f[k] for f in folds])) for k in keys}
    marg = float(CFG["toy_check"]["primary_margin"])
    res = {"folds": folds, "mean": mean, "bar": f"rho_theta >= rho_b1 + {marg} (fold mean)",
           "checks": {"primary_rho_theta_ge_rho_b1_plus_margin": mean["rho_theta"] >= mean["rho_b1"] + marg,
                      "secondary_unseen_gp_cont_gt_b1": mean["rho_unseen_gp_conttarget"] > mean["rho_unseen_gp_b1target"]}}
    p = ROOT / "results" / "m2cont"
    p.mkdir(parents=True, exist_ok=True)
    (p / "toy_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["mean"], indent=1), res["checks"])
    return res


def _corrupt_s(df: pd.DataFrame, mask: np.ndarray, seed: int = 1) -> pd.DataFrame:
    """Move S2 of rows `mask` while keeping the binary label (positives stay < threshold)."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    thr = protocol.threshold()
    y = out.y.to_numpy(bool)
    new = np.where(y, thr - 1.0 - 4.0 * rng.random(len(out)), thr * rng.random(len(out)))
    out.loc[mask, "s_score"] = new[mask]
    assert (protocol.binarize(out.s_score) == y).all()
    return out


def stage_canary(j: int = 0) -> dict:
    """Pipeline canary on the toy data, outer fold 0, add-on tables rebuilt without cache: S2 of every outer-
    training pair HELD OUT of inner fold j's training set (T_j-gene pairs + H_j) is corrupted (labels y kept).
    Pass: inner fold j's stacker rows keep bit-identical new columns, while the other inner folds' rows and the
    outer eval rows move (positive control: they do see those pairs)."""
    from src.models import m2
    from src.ryanfeat import SimStore
    _toy_guard()
    sims = SimStore()
    k = 0
    tr0, ev0 = build_addon(None, k, sims, False, use_cache=False)
    train = m2.fold_pairs(None, k)[0]
    inner = data.inner_splits(train)
    key = lambda d: pd.MultiIndex.from_frame(d[["gene_a", "gene_b"]])   # noqa: E731
    held = ~key(train).isin(key(inner[j].train))
    orig = m2.fold_pairs

    def corrupted(split_seed, fold):
        t, e, d = orig(split_seed, fold)
        assert len(t) == len(train)
        return _corrupt_s(t, held), e, d
    m2.fold_pairs = corrupted
    try:
        inner2 = data.inner_splits(corrupted(None, k)[0])
        assert all(set(a.genes) == set(b.genes) and len(a.eval) == len(b.eval) for a, b in zip(inner, inner2))
        tr1, ev1 = build_addon(None, k, sims, False, use_cache=False)
    finally:
        m2.fold_pairs = orig
    cols = [c for c in tr0.columns if is_new(c)]
    fold_of = np.concatenate([np.full(len(f.eval), f.j) for f in inner])

    def ident(a, b):
        return all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) for c in cols)
    mj = fold_of == j
    res = {"outer_fold": k, "inner_fold": j, "n_pairs_corrupted": int(held.sum()), "n_new_columns": len(cols),
           "inner_j_rows_bit_identical": ident(tr0[mj], tr1[mj]),
           "other_inner_rows_changed": not ident(tr0[~mj], tr1[~mj]),
           "outer_eval_rows_changed": not ident(ev0, ev1)}
    res["pass"] = bool(res["inner_j_rows_bit_identical"] and res["other_inner_rows_changed"]
                       and res["outer_eval_rows_changed"])
    print(res, flush=True)
    p = ROOT / "results" / "m2cont"
    p.mkdir(parents=True, exist_ok=True)
    (p / "canary_toy.json").write_text(json.dumps(res, indent=1))
    return res


def stage_perclass(pairs: str) -> dict:
    """Diagnostic (not an adoption rule): paired per-class deltas with the shared gene bootstrap, using
    one-class weightings ({C: 1, others: 0}) in src/evaluate.bootstrap_replicates -> fold-mean within-class AP."""
    from src.evaluate import _ci, _run_frame, bootstrap_counts, bootstrap_replicates, load_run
    W = {c: {cc: float(cc == c) for cc in protocol.CLASSES} for c in protocol.CLASSES}
    p = out_dir() / "perclass.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for item in pairs.split(","):
        a, b = item.split(":")
        A, B = load_run(resolve_run(a)), load_run(resolve_run(b))
        da, db = _run_frame(A), _run_frame(B)
        key = ["gene_a", "gene_b", "fold"]
        db = db.set_index(key).loc[pd.MultiIndex.from_frame(da[key])].reset_index()
        counts, gidx = bootstrap_counts(data.dev_folds(A["split_seed"]).index, A["n_boot"])
        ones = np.ones((1, counts.shape[1]), counts.dtype)
        rec = {}
        for c in protocol.CLASSES:
            w = {c: W[c]}
            pa = bootstrap_replicates(da, ones, gidx, w)[c][0]
            pb = bootstrap_replicates(db, ones, gidx, w)[c][0]
            ra, rb = bootstrap_replicates(da, counts, gidx, w)[c], bootstrap_replicates(db, counts, gidx, w)[c]
            lo, hi = _ci(ra - rb, protocol.load()["adoption"]["ci_level"])
            rec[c] = {"a": float(pa), "b": float(pb), "delta": float(pa - pb), "ci": [float(lo), float(hi)]}
        out[f"{a}_vs_{b}"] = rec
        print(a, "-", b, {c: f"{r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" for c, r in rec.items()},
              flush=True)
    p.write_text(json.dumps(out, indent=1))
    return out


def stage_noise(split_seed=None, k: int = 0, seed: int = 0) -> dict:
    """Diagnostic: split-half reliability of the GP targets on real outer-training pairs of fold k. Pairs are
    split 50/50 at random; B1 beta and the continuous theta are fitted on each half; Spearman across genes
    with >= 20 pairs in each half. Higher = less noisy target. Aggregate numbers only."""
    from scipy.stats import spearmanr
    from src.models import m2
    from src.ryanfeat import _clean_train, fit_b1
    tr = _clean_train(m2.fold_pairs(split_seed, k)[0])
    h = np.random.default_rng(seed).random(len(tr)) < 0.5
    halves = [tr[h].reset_index(drop=True), tr[~h].reset_index(drop=True)]
    cnt = [pd.concat([x.gene_a, x.gene_b]).value_counts() for x in halves]
    genes = sorted(set(cnt[0].index[cnt[0] >= 20]) & set(cnt[1].index[cnt[1] >= 20]))
    b = [fit_b1(x, R.ryan_cfg()["b1_C"])[1] for x in halves]
    t = [m2.fit_additive_cont(x)[0] for x in halves]
    full_b, full_t = fit_b1(tr, R.ryan_cfg()["b1_C"])[1], m2.fit_additive_cont(tr)[0]
    res = {"split_seed": data._seed(split_seed), "fold": k, "n_pairs": int(len(tr)), "n_genes": len(genes),
           "split_half_rho_b1": float(spearmanr(b[0][genes], b[1][genes]).correlation),
           "split_half_rho_theta": float(spearmanr(t[0][genes], t[1][genes]).correlation),
           "rho_b1_vs_theta_full": float(spearmanr(full_b[genes], full_t[genes]).correlation)}
    print(res, flush=True)
    (out_dir() / "target_noise_real.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="features,run")
    ap.add_argument("--variants", default="M2c_gp,M2c_nbr,M2c_both")
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--perm", action="store_true", help="features: gene-permuted control add-on")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    set_threads(THREADS)
    assert protocol.label_column() == "s_score" and CFG["target"] == "neg_min_s2_0"
    st = args.stages.split(",")
    print(f"data root {protocol.data_root()}; split seed {data._seed(args.split_seed)}; threads {THREADS}", flush=True)
    if "toycheck" in st:
        stage_toycheck()
    if "canary" in st:
        stage_canary()
    if "features" in st:
        stage_features(args.split_seed, args.perm)
    if "run" in st:
        from src.ryanfeat import SimStore
        vn = args.variants.split(",")
        V = variants()
        assert all(v in V for v in vn), vn
        spec = PA.load_spec()
        viol = PA.source_violations(spec)
        if viol:
            raise SystemExit("feature sources not on the use=feature allowlist:\n" + "\n".join(viol))
        stage_run(args.split_seed, spec, SimStore(names=SimStore().names), vn, args)
    if "compare" in st:
        stage_compare(args.pairs, args)
    if "perclass" in st:
        stage_perclass(args.pairs)
    if "noise" in st:
        stage_noise(args.split_seed)
    print(f"peak RSS {peak_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
