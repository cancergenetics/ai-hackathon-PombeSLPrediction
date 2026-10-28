#!/usr/bin/env python
"""M5: equal-weight rank-average ensembles of existing out-of-fold predictions (DESIGN §5; pre-stated in
config/m5m3.yaml `m5`; docs/m5m3.md). Nothing is tuned; members are seed-averaged registered predictions.

Rules (per outer fold; src-free pure functions below, tested in tests/test_m5m3.py):
  pooled            percentile ranks over all eval rows of the fold (C1+C2+C3), equal-weight mean
  class_preserving  percentile ranks within (fold, class), equal-weight mean r; the class's rows get the
                    incumbent's own scores of that class re-assigned in the order of r (ties by incumbent score)

Stages:
  members7  seed-7 members that do not exist yet: B4_logistic_MF_split7 (run_baselines._mf_fold, tuned cfg,
            seeds 0-2) and DIAG_NBR_mean__hit_dmean_split7 (M2 split-7 nested eval rows column)
  ensemble  every candidate (or --candidates) on both partitions -> evaluate() (registered)
  compare   candidate vs incumbent with the seed-7 pair as alt -> results/m5m3/comparisons.json
  control   shuffled-member control of --candidates (config adoption.control_if_adopted.M5)
  corr      Spearman correlations of members per class (report-only)
Member run ids: config/m5m3.yaml; `--set NAME=RUN_ID` overrides/fills (incumbent split7, M3 runs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data, protocol  # noqa: E402
from src.timing import Timer, Timings, set_threads  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "m5m3.yaml").read_text())
THREADS = int(CFG["compute"]["threads"])
KEY = ["gene_a", "gene_b", "fold"]
SHUFFLE_SEED = 20261028


# ============================================================================================ rules (pure)
def pct_rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return rankdata(x, method="average") / len(x)


def rank_average(df: pd.DataFrame, members: list, rule: str, inc: str) -> np.ndarray:
    """Ensemble score for rows of df (columns fold, part and one score column per member)."""
    out = np.empty(len(df))
    if rule == "pooled":
        for _, ix in df.groupby("fold", sort=False).indices.items():
            out[ix] = np.mean([pct_rank(df[m].to_numpy()[ix]) for m in members], axis=0)
        return out
    if rule == "class_preserving":
        for _, ix in df.groupby(["fold", "part"], sort=False, observed=True).indices.items():
            r = np.mean([pct_rank(df[m].to_numpy()[ix]) for m in members], axis=0)
            s_inc = df[inc].to_numpy()[ix]
            order = np.lexsort((s_inc, r))          # ascending r, ties by incumbent score
            v = np.empty(len(ix))
            v[order] = np.sort(s_inc)
            out[ix] = v
        return out
    raise ValueError(f"unknown rule {rule!r}")


# ============================================================================================ helpers
def out_dir() -> Path:
    d = protocol.results_root() / "m5m3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry() -> pd.DataFrame:
    return pd.read_csv(protocol.results_root() / "registry.tsv", sep="\t", dtype=str)


def latest_clean(model: str) -> str | None:
    reg = registry()
    m = (reg.model == model) & (reg.git_dirty == "False")
    return reg.loc[m, "run_id"].iloc[-1] if m.any() else None


def member_runs(overrides: dict) -> dict:
    """{member: {'primary': run_id, 'split7': run_id}} from config + registry + --set overrides."""
    M = CFG["m5"]["members"]
    out = {}
    for k, v in M.items():
        p, s7 = v["primary"], v["split7"]
        if k == "INC":
            s7 = CFG["incumbent"]["run_split7"] or latest_clean("M2r_huber_split7")
        elif k == "B4":
            s7 = latest_clean("B4_logistic_MF_split7")
        elif k == "NBR":
            s7 = latest_clean("DIAG_NBR_mean__hit_dmean_split7")
        elif k == "M3":
            p, s7 = latest_clean("M3"), latest_clean("M3_split7")
        out[k] = {"primary": overrides.get(k, p), "split7": overrides.get(f"{k}_split7", s7)}
    return out


def load_member(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(protocol.preds_dir() / f"{run_id}.parquet")[KEY + ["part", "score"]]


def run_info(run_id: str) -> dict:
    from src.evaluate import load_run
    r = load_run(run_id)
    return {"run_id": run_id, "split_seed": r["split_seed"], "primary": r["primary"],
            "wall_s": (r.get("runtime") or {}).get("wall_s"), "git_dirty": r["git_dirty"]}


def frame(runs: dict, members: list) -> pd.DataFrame:
    df = None
    for m in members:
        x = load_member(runs[m]).rename(columns={"score": m})
        df = x if df is None else df.merge(x.drop(columns="part"), on=KEY, how="inner", validate="1:1")
    n = {len(load_member(runs[m])) for m in members}
    assert len(n) == 1 and len(df) == n.pop(), "members cover different eval rows"
    return df


# ============================================================================================ members7
def stage_members7() -> dict:
    import run_baselines as RB
    import run_m2plus as P2
    from src.evaluate import evaluate
    from src.models import m2
    from src.ryanfeat import SimStore
    out = {}
    split = 7
    n = protocol.load()["dev_cv"]["n_folds"]
    cfg = RB.tuned_cfg()
    if latest_clean("B4_logistic_MF_split7") is None:
        seeds = protocol.load()["seeds"]["ladder"]
        res = [RB._mf_fold((split, k, s, cfg, THREADS)) for k in range(n) for s in seeds]
        T = Timings()
        T.stages = [t for _, t in res]
        r = evaluate(pd.concat([p for p, _ in res], ignore_index=True), model="B4_logistic_MF_split7",
                     split_seed=split, runtime=T.as_dict(),
                     notes=f"B4 (run_baselines._mf_fold, tuned cfg {cfg}) on split seed 7 for the M5 seed-7 check; "
                           f"{THREADS} torch threads, sequential")
        out["B4_split7"] = r["run_id"]
        print("B4_split7", r["run_id"], round(r["primary"], 4), flush=True)
    if latest_clean("DIAG_NBR_mean__hit_dmean_split7") is None:
        sims = SimStore(names=SimStore().names)
        T = Timings()
        rows = []
        with T.stage("score"):
            for k in range(n):
                _, ev, _, _ = m2.fold_tables_m2(split, k, sims, __import__("run_m2").ryan_cfg(),
                                               **P2.tables_kw("M2r"))
                rows.append(ev[["gene_a", "gene_b", "fold", "part"]].assign(
                    score=RB.fill_bottom(ev["nbr_mean__hit_dmean"].to_numpy()), seed=0))
        r = evaluate(pd.concat(rows, ignore_index=True), model="DIAG_NBR_mean__hit_dmean_split7", split_seed=split,
                     runtime=T.as_dict(), feature_hashes={"m2feat_cache": P2.cache_name("M2r", sims)},
                     notes="DIAGNOSTIC single feature nbr_mean__hit_dmean (NaN -> bottom) from the M2 split-7 nested "
                           "eval rows (non-coding training pairs dropped, unlike the ryanfeat baseline tables)")
        out["NBR_split7"] = r["run_id"]
        print("NBR_split7", r["run_id"], round(r["primary"], 4), flush=True)
    return out


# ============================================================================================ ensemble
def build(cand: str, runs: dict, split: str, shuffle: bool = False) -> tuple[pd.DataFrame, dict]:
    c = CFG["m5"]["candidates"][cand]
    mem = c["members"]
    ids = {m: runs[m][split] for m in mem}
    missing = [m for m, r in ids.items() if not r]
    if missing:
        raise SystemExit(f"{cand} {split}: no run for members {missing}")
    t0 = time.perf_counter()
    df = frame(ids, mem)
    if shuffle:
        rng = np.random.default_rng(SHUFFLE_SEED)
        for m in mem:
            if m == "INC":
                continue
            v = df[m].to_numpy().copy()
            for _, ix in df.groupby("fold", sort=False).indices.items():
                v[ix] = v[ix][rng.permutation(len(ix))]
            df[m] = v
    df["score"] = rank_average(df, mem, c["rule"], "INC")
    combine_s = time.perf_counter() - t0
    info = {m: run_info(r) for m, r in ids.items()}
    return df[KEY + ["part", "score"]].assign(seed=0), {"members": info, "rule": c["rule"], "combine_s": combine_s}


def stage_ensemble(cands: list, runs: dict, shuffle=False) -> dict:
    from src.evaluate import evaluate
    p = out_dir() / "ensembles.json"
    summ = json.loads(p.read_text()) if p.exists() else {}
    for cand in cands:
        for split in ("primary", "split7"):
            P, info = build(cand, runs, split, shuffle)
            ss = info["members"]["INC"]["split_seed"]
            name = cand + ("_shuffled" if shuffle else "") + ("" if split == "primary" else "_split7")
            member_wall = sum(float(v["wall_s"] or 0) for v in info["members"].values())
            rt = {"wall_s": member_wall + info["combine_s"], "combine_s": info["combine_s"],
                  "member_wall_s": {m: v["wall_s"] for m, v in info["members"].items()}, "threads": THREADS}
            res = evaluate(P, model=name, split_seed=ss, runtime=rt,
                           notes=f"M5 rank-average (config/m5m3.yaml) rule={info['rule']} members="
                                 f"{ {m: v['run_id'] for m, v in info['members'].items()} } shuffle={shuffle}")
            m = res["metrics_mean"]
            summ[name] = {"run_id": res["run_id"], "primary": res["primary"], "se": res["bootstrap"]["primary"]["se"],
                          "ap": {c: m[f"ap_{c}"] for c in ("C1", "C2", "C3")}, "pooled_all_pairs": m["pooled_all_pairs"],
                          "pooled_equal": m["pooled_equal"], **info}
            print(f"== {name}: {res['run_id']} primary {res['primary']:.4f} C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} "
                  f"C3 {m['ap_C3']:.4f} (combine {info['combine_s']:.1f} s)", flush=True)
    p.write_text(json.dumps(summ, indent=1, default=float))
    return summ


def stage_compare(cands: list, runs: dict, shuffle=False) -> dict:
    from src.evaluate import compare
    p = out_dir() / "comparisons.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    ens = json.loads((out_dir() / "ensembles.json").read_text())
    for cand in cands:
        name = cand + ("_shuffled" if shuffle else "")
        if cand in CFG["m5"]["candidates"]:
            a, a7 = ens[name]["run_id"], ens[name + "_split7"]["run_id"]
        else:                                   # a standalone model, e.g. M3
            a, a7 = runs[cand]["primary"], runs[cand]["split7"]
        b, b7 = runs["INC"]["primary"], runs["INC"]["split7"]
        c = compare(a, b, alt=(a7, b7) if a7 and b7 else None)
        out[f"{name}_vs_INC"] = c
        print(f"{name} - INC: delta {c['delta']:+.4f} CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] all-pairs "
              f"{c['delta_all_pairs']:+.4f} equal {c['delta_equal']:+.4f} seed7 {c['alt_delta']} -> {c['reason']}",
              flush=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    return out


def stage_corr(runs: dict) -> dict:
    from scipy.stats import spearmanr
    mem = [m for m in runs if runs[m]["primary"]]
    df = frame({m: runs[m]["primary"] for m in mem}, mem)
    out = {}
    for part, g in df.groupby("part", observed=True):
        rho = spearmanr(g[mem].to_numpy()).correlation
        out[part] = pd.DataFrame(rho, index=mem, columns=mem).round(3).to_dict()
    (out_dir() / "member_correlations.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="members7,ensemble,compare")
    ap.add_argument("--candidates", default="M5_IncB4Nbr,M5_IncB4Nbr_cp,M5_IncM2bM1")
    ap.add_argument("--set", default="", help="MEMBER=RUN_ID[,MEMBER_split7=RUN_ID]")
    a = ap.parse_args()
    set_threads(THREADS)
    ov = dict(x.split("=") for x in a.set.split(",") if x)
    st = a.stages.split(",")
    cands = [c for c in a.candidates.split(",") if c]
    if "members7" in st:
        stage_members7()
    runs = member_runs(ov)
    print(json.dumps(runs, indent=1), flush=True)
    if "ensemble" in st:
        stage_ensemble(cands, runs)
    if "compare" in st:
        stage_compare(cands, runs)
    if "control" in st:
        stage_ensemble(cands, runs, shuffle=True)
        stage_compare(cands, runs, shuffle=True)
    if "corr" in st:
        stage_corr(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
