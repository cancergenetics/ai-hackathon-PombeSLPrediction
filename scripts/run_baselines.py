#!/usr/bin/env python
"""Baselines B1, B4, B2, B3 + diagnostics through src/evaluate.py (DESIGN.md §5, Phase C).

Stages (default: all, in order):
  tune      B1 L2 strength C and B4 (k, weight decay, epochs) tuned ONCE on outer fold 0, inner fold 0
            (src.data.inner_splits): B1 by class-weighted pooled AP on the inner eval rows, B4 by C1-like AP
            -> results/baselines/tuning.json
  features  nested Ryan feature tables (src.stacking.fold_tables, cached) for every dev fold, one process
            per fold -> results/baselines/feature_build_timing.json
  eval      registered runs (5 dev folds, split seed from protocol):
              B1  b1_logit (deterministic)            B4  logistic MF, seeds = protocol ladder; pairs with
              B2  -(Costanzo ortholog eps min)            an unseen gene fall back to B1
              B3  Costanzo ortholog profile similarity    (missing -> median for B2/B3)
              DIAG_oracle_propensity  true gene hit rates from ALL dev labels (union of the folds'
                                      training labels; uses held-out labels -> DIAGNOSTIC, not a model)
              DIAG_NBR_<src>__<feat>  single NBR feature per source; the feature is picked per source on
                                      fold-0 stacker training rows (inner eval rows), never on outer eval
            -> results/baselines/summary.json
  doc       docs/baselines.md from summary.json
Usage: run_baselines.py [--stages tune,features,eval,doc] [--split-seed S] [--workers 5]
"""
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
from src.timing import Timer, Timings, set_threads  # noqa: E402

OUT = protocol.results_root() / "baselines"
B1_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
MF_K = [8, 16, 32]
MF_WD = [1e-3, 1e-2, 3e-2, 1e-1]   # round 1 (1e-4..1e-2) peaked at the edge 1e-2: tuning_round1.json
MF_EPOCHS, MF_LR = 400, 0.05
NBR_CANDIDATES = ["hit_dmax", "hit_dmean", "hit_block", "negs_dmax", "negs_dmean", "negs_block",
                  "generate_max", "generate_min"]


def wap(y, s, part) -> float:
    """Class-weighted pooled AP (protocol primary weights), as in src/evaluate.py."""
    from sklearn.metrics import average_precision_score
    from src.evaluate import class_row_weights
    return float(average_precision_score(y, s, sample_weight=class_row_weights(part, protocol.class_weights())))


def fill_bottom(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    fin = np.isfinite(s)
    return np.where(fin, s, (s[fin].min() - 1.0) if fin.any() else 0.0)


# ============================================================================================ tune
def stage_tune(split_seed) -> dict:
    from sklearn.metrics import average_precision_score
    from src.ryanfeat import fit_b1, fit_mf
    t0 = time.perf_counter()
    train, _ = data.fold_split(split_seed, 0)
    f = data.inner_splits(train)[0]
    ev = f.eval.reset_index(drop=True)
    tg = set(f.train.gene_a) | set(f.train.gene_b)
    y, part = ev.y.to_numpy(bool), ev.part.to_numpy()
    res = {"split": {"outer_fold": 0, "inner_fold": 0, "n_train": len(f.train), "n_eval": len(ev),
                     "parts": ev.part.value_counts().to_dict()}, "b1": {}, "mf": {}}
    for C in B1_GRID:
        c0, beta = fit_b1(f.train, C)
        s = c0 + (beta.reindex(ev.gene_a).fillna(0).to_numpy() + beta.reindex(ev.gene_b).fillna(0).to_numpy())
        res["b1"][str(C)] = {"pooled": wap(y, s, part),
                             **{p: float(average_precision_score(y[part == p], s[part == p]))
                                for p in ("C1", "C2")}}
        print(f"B1 C={C}: {res['b1'][str(C)]}", flush=True)
    best_C = max(B1_GRID, key=lambda C: res["b1"][str(C)]["pooled"])
    c1 = ev[(ev.part == "C1") & ev.gene_a.isin(tg) & ev.gene_b.isin(tg)]
    valid = (c1.gene_a.to_numpy(object), c1.gene_b.to_numpy(object), c1.y.to_numpy(bool))
    best = None
    for k in MF_K:
        for wd in MF_WD:
            P = fit_mf(f.train, k, wd, MF_EPOCHS, MF_LR, 0, valid=valid, eval_every=25)
            ep, apv = max(P["history"], key=lambda h: h[1])
            res["mf"][f"k{k}_wd{wd}"] = {"history": P["history"], "best_epoch": ep, "best_ap_C1": apv}
            print(f"MF k={k} wd={wd}: best C1-like AP {apv:.4f} at epoch {ep}", flush=True)
            if best is None or apv > best[3]:
                best = (k, wd, ep, apv)
    res["chosen"] = {"b1_C": best_C, "mf_k": best[0], "mf_wd": best[1], "mf_epochs": best[2],
                     "mf_lr": MF_LR, "mf_inner_C1_ap": best[3],
                     "b1_inner_C1_ap": res["b1"][str(best_C)]["C1"]}
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuning.json").write_text(json.dumps(res, indent=1))
    print("chosen:", res["chosen"])
    return res


def tuned_cfg() -> dict:
    ch = json.loads((OUT / "tuning.json").read_text())["chosen"]
    return {k: ch[k] for k in ("b1_C", "mf_k", "mf_wd", "mf_epochs", "mf_lr")}


# ============================================================================================ features
def _fold_features(args):
    split_seed, k, cfg, threads, names = args
    set_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from src.ryanfeat import SimStore
    from src.stacking import fold_tables
    with Timer(f"fold{k}") as t:
        sims = SimStore(names=names)
        tr, ev, _ = fold_tables(split_seed, k, sims, cfg)
    return k, t.result, len(tr), len(ev), sims.names, sims.hashes()


def stage_features(split_seed, workers) -> dict:
    from src.ryanfeat import SimStore, cfg_hash, resolve_cfg
    cfg = tuned_cfg()
    n = protocol.load()["dev_cv"]["n_folds"]
    thr = max(1, 10 // workers)
    names = SimStore().names              # snapshot: tier-2 matrices added later do not change this run
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_fold_features, [(split_seed, k, cfg, thr, names) for k in range(n)]))
    sims = SimStore(names=names)
    out = {"cfg": resolve_cfg(cfg), "cfg_hash": cfg_hash(cfg, sims), "sims": res[0][4], "sim_sha256": res[0][5],
           "folds": {k: {"timing": t, "n_train_rows": a, "n_eval_rows": b} for k, t, a, b, _, _ in res},
           "workers": workers, "threads_per_worker": thr}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_build_timing.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v["timing"]["wall_s"] for k, v in out["folds"].items()}))
    return out


# ============================================================================================ eval
def _mf_fold(args):
    split_seed, k, seed, cfg, threads = args
    set_threads(threads)
    import torch
    torch.set_num_threads(threads)
    from src.ryanfeat import fit_b1, fit_mf, mf_score
    train, ev = data.fold_split(split_seed, k)
    with Timer(f"B4_fold{k}_seed{seed}") as t:
        c0, beta = fit_b1(train, cfg["b1_C"])
        P = fit_mf(train, cfg["mf_k"], cfg["mf_wd"], cfg["mf_epochs"], cfg["mf_lr"], seed)
        tg = set(train.gene_a) | set(train.gene_b)
        both = (ev.gene_a.isin(tg) & ev.gene_b.isin(tg)).to_numpy()
        b1 = c0 + (beta.reindex(ev.gene_a).fillna(0).to_numpy() + beta.reindex(ev.gene_b).fillna(0).to_numpy())
        s = np.where(both, mf_score(P, ev.gene_a.to_numpy(object), ev.gene_b.to_numpy(object)), b1)
    return ev[["gene_a", "gene_b", "fold", "part"]].assign(score=s, seed=seed), t.result


def pair_lookup(path: Path, ev: pd.DataFrame, fill) -> np.ndarray:
    from src import feature_store as fs
    M = np.load(path)
    pos = pd.Series(np.arange(M.shape[0]), index=fs.genes())
    ia, ib = pos.reindex(ev.gene_a).to_numpy(), pos.reindex(ev.gene_b).to_numpy()
    v = np.full(len(ev), np.nan)
    ok = ~np.isnan(ia) & ~np.isnan(ib)
    v[ok] = M[ia[ok].astype(int), ib[ok].astype(int)]
    return np.where(np.isfinite(v), v, fill)


def stage_eval(split_seed, workers) -> dict:
    from src import feature_store as fs
    from src.evaluate import evaluate
    from src.ryanfeat import SimStore, cfg_hash, fit_b1
    from src.stacking import fold_tables
    P = protocol.load()
    n = P["dev_cv"]["n_folds"]
    cfg = tuned_cfg()
    ftime = json.loads((OUT / "feature_build_timing.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    summary.setdefault("runs", {})
    folds = [data.fold_split(split_seed, k) for k in range(n)]
    base_cols = ["gene_a", "gene_b", "fold", "part"]

    def register(model, preds, T, notes, fh=None, kind="baseline"):
        res = evaluate(preds, model=model, split_seed=split_seed, runtime=T.as_dict(), notes=notes,
                       feature_hashes=fh or {})
        m = res["metrics_mean"]
        summary["runs"][model] = {"run_id": res["run_id"], "kind": kind, "primary": res["primary"],
                                  "se": res["bootstrap"]["primary"]["se"], "ci": res["bootstrap"]["primary"]["ci"],
                                  **{f"ap_{c}": m[f"ap_{c}"] for c in ("C1", "C2", "C3")},
                                  "prevalence": m["prevalence_primary"], "auroc": m["auroc"],
                                  "wall_s": T.as_dict()["wall_s"], "seeds": res["seeds"], "notes": notes}
        print(f"{model:40s} {res['primary']:.4f} ± {res['bootstrap']['primary']['se']:.4f}  "
              f"C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} C3 {m['ap_C3']:.4f}  run {res['run_id']}", flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---- B1
    T = Timings()
    rows = []
    for k, (train, ev) in enumerate(folds):
        with T.stage(f"B1_fold{k}"):
            c0, beta = fit_b1(train, cfg["b1_C"])
            s = c0 + (beta.reindex(ev.gene_a).fillna(0).to_numpy() + beta.reindex(ev.gene_b).fillna(0).to_numpy())
        rows.append(ev[base_cols].assign(score=s, seed=0))
    register("B1_additive_logit", pd.concat(rows, ignore_index=True), T,
             f"B1 c+beta_a+beta_b, L2 C={cfg['b1_C']} (tuned fold0/inner0); unseen beta=0")

    # ---- B4 (seeds = ladder)
    seeds = P["seeds"]["ladder"]
    thr = max(1, 10 // workers)
    jobs = [(split_seed, k, s, cfg, thr) for k in range(n) for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_mf_fold, jobs))
    T = Timings()
    T.stages = [t for _, t in out]
    register("B4_logistic_MF", pd.concat([p for p, _ in out], ignore_index=True), T,
             f"symmetric logistic MF k={cfg['mf_k']} wd={cfg['mf_wd']} epochs={cfg['mf_epochs']} lr={cfg['mf_lr']} "
             f"(tuned fold0/inner0); pairs with an unseen gene -> B1 logit; runtime = sum over fold x seed fits "
             f"run in {workers} parallel workers")

    # ---- B2 / B3 (label-free lookups)
    fdir = fs.feat_dir()
    man = json.loads((fdir / "manifest.json").read_text())
    for model, rel, sign, desc in (("B2_costanzo_eps_min", "pair/CER__eps_min.npy", -1.0,
                                    "-(Costanzo ortholog eps min); missing -> median"),
                                   ("B3_costanzo_profile_sim", "sim/cer_profile_corr.npy", 1.0,
                                    "Costanzo ortholog GI-profile similarity; missing -> median")):
        T = Timings()
        rows = []
        with T.stage("lookup"):
            M = np.load(fdir / rel, mmap_mode="r")
            med = float(np.nanmedian(np.asarray(M)))
            for k, (_, ev) in enumerate(folds):
                v = pair_lookup(fdir / rel, ev, med)
                rows.append(ev[base_cols].assign(score=sign * v, seed=0))
        register(model, pd.concat(rows, ignore_index=True), T, desc + f" (median {med:.5f} over all finite G x G)",
                 {rel: man.get(rel, {}).get("sha256")})

    # ---- DIAGNOSTIC: oracle propensity (true gene hit rates from all dev labels)
    T = Timings()
    with T.stage("oracle"):
        allp = pd.concat([tr[["gene_a", "gene_b", "y"]] for tr, _ in folds]).drop_duplicates(["gene_a", "gene_b"])
        long = pd.concat([allp[["gene_a", "y"]].rename(columns={"gene_a": "g"}),
                          allp[["gene_b", "y"]].rename(columns={"gene_b": "g"})])
        g = long.groupby("g").y.agg(["sum", "count"])
        r = (g["sum"] + 0.5) / (g["count"] + 1.0)
        lr = np.log(r / (1 - r))
        rows = []
        for k, (_, ev) in enumerate(folds):
            s = lr.reindex(ev.gene_a).fillna(lr.median()).to_numpy() + lr.reindex(ev.gene_b).fillna(lr.median()).to_numpy()
            rows.append(ev[base_cols].assign(score=s, seed=0))
    summary["oracle_coverage"] = {"n_genes": int(len(g)), "n_pairs": int(len(allp))}
    register("DIAG_oracle_propensity", pd.concat(rows, ignore_index=True), T,
             "DIAGNOSTIC (not a model): logit of true gene hit rates computed from ALL dev labels incl. the "
             "held-out genes' pairs (union of the 5 folds' training labels), (pos+0.5)/(n+1)", kind="diagnostic")

    # ---- DIAGNOSTIC: single NBR feature per source (feature chosen on fold-0 stacker training rows)
    sims = SimStore(names=ftime["sims"])
    assert cfg_hash(cfg, sims) == ftime["cfg_hash"], "similarity set / cfg changed since the feature stage"
    tabs = [fold_tables(split_seed, k, sims, cfg) for k in range(n)]
    tr0 = tabs[0][0]
    srcs = sorted({c.split("__")[0][4:] for c in tr0 if c.startswith("nbr_")})
    sel = {}
    for s_ in srcs:
        aps = {f: wap(tr0.y.to_numpy(bool), fill_bottom(tr0[f"nbr_{s_}__{f}"]), tr0.part.to_numpy())
               for f in NBR_CANDIDATES}
        best = max(aps, key=aps.get)
        sel[s_] = {"feature": best, "inner_pooled_ap": aps}
    summary["nbr_selection"] = sel
    fhash = {"ryanfeat_cfg_hash": cfg_hash(cfg, sims), **{f"sim/{k}": v for k, v in sims.hashes().items()}}
    feat_wall = sum(v["timing"]["wall_s"] for v in ftime["folds"].values())
    for s_ in srcs:
        col = f"nbr_{s_}__{sel[s_]['feature']}"
        T = Timings()
        rows = []
        with T.stage("score"):
            for k in range(n):
                ev = tabs[k][1]
                rows.append(ev[base_cols].assign(score=fill_bottom(ev[col]), seed=0))
        T.stages.append({"stage": "nested_feature_build_all_folds(shared, all features)", "wall_s": feat_wall,
                         "cpu_s": float("nan"), "peak_rss_mb": float("nan"), "threads": ftime["threads_per_worker"],
                         "hardware": T.stages[0]["hardware"]})
        register(f"DIAG_NBR_{s_}__{sel[s_]['feature']}", pd.concat(rows, ignore_index=True), T,
                 f"DIAGNOSTIC single feature {col} (NaN -> bottom); chosen on fold-0 stacker training rows; "
                 f"wall_s includes the full nested feature build (all features, all folds)", fhash, kind="diagnostic")
    summary["cfg"] = cfg
    summary["sims"] = sims.names
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ============================================================================================ doc
INNER_DIAG = ["hit_dmean", "hit_block", "negs_block"]


def inner_class_diag(ft) -> dict:
    from sklearn.metrics import average_precision_score
    from src.ryanfeat import SimStore
    from src.stacking import fold_tables
    tr0 = fold_tables(None, 0, SimStore(names=ft["sims"]), tuned_cfg())[0]
    out = {}
    for s_ in ft["sims"] + ["mean"]:
        out[s_] = {}
        for f in INNER_DIAG:
            out[s_][f] = {}
            for c in ("C1", "C2", "C3"):
                d = tr0[tr0.part == c]
                out[s_][f][c] = float(average_precision_score(d.y.to_numpy(bool), fill_bottom(d[f"nbr_{s_}__{f}"])))
    y = tr0.y.to_numpy(bool)
    out["(prevalence)"] = {f: {c: float(y[tr0.part.to_numpy() == c].mean()) for c in ("C1", "C2", "C3")}
                           for f in INNER_DIAG}
    return out


def stage_doc() -> None:
    S_ = json.loads((OUT / "summary.json").read_text())
    tune = json.loads((OUT / "tuning.json").read_text())
    ft = json.loads((OUT / "feature_build_timing.json").read_text())
    tox = protocol.results_root() / "ryanfeat_toy_verification.json"
    toy = json.loads(tox.read_text()) if tox.exists() else None
    R = S_["runs"]
    L = ["# Baselines (Phase C): B1, B4, B2, B3 and diagnostics",
         "",
         "Generated by `scripts/run_baselines.py --stages doc` from `results/baselines/summary.json` (do not edit by",
         "hand). All runs: 5 dev folds, split seed " + str(data._seed(None)) + ", metrics via `src/evaluate.py`",
         "(primary = fold-mean pooled AUPRC weighted to the unmeasured-universe class mix; SE = shared Poisson gene",
         "bootstrap, 1000 replicates). Per-class AUPRC is unweighted within the class. Runtime = wall seconds of the",
         "model's own fit+predict stages over all folds (and seeds), excluding evaluation; hardware in the run JSON.",
         "",
         "| model | primary ± SE | 95% CI | AP C1 | AP C2 | AP C3 | AUROC | runtime (s) | seeds | run_id |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["B1_additive_logit", "B4_logistic_MF", "B2_costanzo_eps_min", "B3_costanzo_profile_sim",
             "DIAG_oracle_propensity"] + sorted(k for k in R if k.startswith("DIAG_NBR"))
    for m in order:
        if m not in R:
            continue
        r = R[m]
        L.append(f"| {m} | {r['primary']:.4f} ± {r['se']:.4f} | {r['ci'][0]:.4f}–{r['ci'][1]:.4f} | "
                 f"{r['ap_C1']:.4f} | {r['ap_C2']:.4f} | {r['ap_C3']:.4f} | {r['auroc']:.3f} | {r['wall_s']:.1f} | "
                 f"{','.join(map(str, r['seeds']))} | `{r['run_id']}` |")
    prev = next(iter(R.values()))["prevalence"]
    ch = tune["chosen"]
    L += ["", f"Weighted prevalence (= B0 expectation): {prev:.4f}. Rows starting `DIAG_` are **diagnostics, not models**:",
          "`DIAG_oracle_propensity` uses the held-out genes' true hit rates (computed from all dev labels, including",
          "the evaluated pairs themselves) and is an upper reference for any gene-propensity-only score;",
          "`DIAG_NBR_*` score a single NBR feature per similarity source (the feature is chosen per source on fold-0",
          "stacker training rows, never on the outer eval rows; their runtime includes the full nested feature build).",
          "",
          "## Configuration and tuning (once, outer fold 0 / inner fold 0; `results/baselines/tuning.json`)",
          f"- B1: L2 strength C = {ch['b1_C']} (sklearn inverse regularisation; grid {B1_GRID}; selection by class-",
          "  weighted pooled AP on the inner eval rows).",
          f"- B4: k = {ch['mf_k']}, weight decay = {ch['mf_wd']}, epochs = {ch['mf_epochs']} (full-batch Adam, lr {ch['mf_lr']};",
          f"  grid k {MF_K} x wd {MF_WD}, epochs checked every 25 up to {MF_EPOCHS}; selection by C1-like AP,",
          f"  inner C1-like AP {ch['mf_inner_C1_ap']:.4f} vs B1 {ch['b1_inner_C1_ap']:.4f}). Pairs with an unseen gene fall",
          "  back to the B1 logit.",
          "- B2: −(Costanzo ortholog ε min) from `pair/CER__eps_min.npy`; B3: `sim/cer_profile_corr.npy`; missing → median",
          "  of the finite G × G values (label-free).",
          f"- NBR: k = {ft['cfg']['k']} per role pool, λ = {ft['cfg']['lam']}; similarity sources used: {', '.join(ft['sims'])}",
          "  (`ppi_rwr_htp` excluded: report-only ablation). Selected feature per source:"]
    for s_, v in S_.get("nbr_selection", {}).items():
        L.append(f"  - {s_}: `{v['feature']}` (fold-0 stacker-row pooled AP {v['inner_pooled_ap'][v['feature']]:.4f})")
    L += ["", "## Per-class NBR diagnostic on fold-0 stacker training rows (inner eval rows; not outer eval)",
          "Unweighted AP within each inner part (C1-like / C2-like / C3-like); the directional features are constant",
          "(= p0) when both genes are unseen, so C3 signal comes only from the block features.", "",
          "| source | " + " | ".join(f"{f} C1/C2/C3" for f in INNER_DIAG) + " |",
          "|---|" + "---|" * len(INNER_DIAG)]
    for s_, v in inner_class_diag(ft).items():
        L.append(f"| {s_} | " + " | ".join("/".join(f"{v[f][c]:.3f}" for c in ("C1", "C2", "C3")) for f in INNER_DIAG)
                 + " |")
    L += ["", "## Nested feature build (src/stacking.py; cached under `data/processed/cache/<seed>/fold<k>/`)",
          f"- cfg hash `{ft['cfg_hash']}`; {ft['workers']} folds in parallel, {ft['threads_per_worker']} threads each.",
          "- Per fold (1 outer + 5 inner builds): " + ", ".join(
              f"fold {k}: {v['timing']['wall_s']:.0f} s, {v['n_train_rows']:,} stacker rows / {v['n_eval_rows']:,} eval rows"
              for k, v in ft["folds"].items()) + "."]
    if toy:
        fe = toy["features"]
        L += ["", "## Toy known-answer check (`scripts/verify_ryanfeat_toy.py`, `results/ryanfeat_toy_verification.json`)",
              f"- B1 C1 AUPRC {fe['b1_logit']['C1']:.3f} vs propensity-only oracle C1 "
              f"{toy['known_answers']['propensity_only_oracle']['C1']:.3f} (toy prevalence C1 "
              f"{toy['known_answers']['prevalence']['C1']:.3f}).",
              f"- NBR hit_block C2/C3: informative MODSIM {fe['nbr_MODSIM__hit_block']['C2']:.3f}/"
              f"{fe['nbr_MODSIM__hit_block']['C3']:.3f} vs noise NOISESIM {fe['nbr_NOISESIM__hit_block']['C2']:.3f}/"
              f"{fe['nbr_NOISESIM__hit_block']['C3']:.3f}.",
              "- Checks: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in toy["checks"].items()) + ".",
              "- The known-answers oracle is computed on a different pair set (its own held-out split), so the tight",
              "  comparison is done on exactly the CV C1 rows in `tests/test_ryanfeat.py::test_toy_*` (reads the toy",
              "  truth): B1 must lie in [0.8 x, x + 0.02] of the TRUE beta_a + beta_b AP on the same rows (observed",
              "  during development: 0.229 vs 0.234, fold range of the oracle 0.18-0.34), and B1 on the permuted-label",
              "  toy must stay < 1.5 x prevalence (observed 0.019 vs 0.020)."]
    L += ["", "## Leakage tests (`tests/test_ryanfeat.py`, synthetic data)",
          "Flip eval labels → eval features bit-identical (catches β/MF/profile fitted with eval rows and NBR with the",
          "eval gene's own rate); flip a training row → its nested features unchanged (catches non-nested stacker rows);",
          "own label → NBR directional features at (a,b) unchanged (catches self not excluded); query-set invariance",
          "(catches neighbours drawn from eval genes); random-label canary (B1/MF AP ≈ prevalence; catches β fitted",
          "with eval rows); symmetry; unseen-gene fallback; informative sim beats noise sim; stacker weights/grouping."]
    (ROOT / "docs" / "baselines.md").write_text("\n".join(L) + "\n")
    print("wrote docs/baselines.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="tune,features,eval,doc")
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    set_threads(8)
    st = a.stages.split(",")
    if "tune" in st:
        stage_tune(a.split_seed)
    if "features" in st:
        stage_features(a.split_seed, a.workers)
    if "eval" in st:
        stage_eval(a.split_seed, a.workers)
    if "doc" in st:
        stage_doc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
