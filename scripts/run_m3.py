#!/usr/bin/env python
"""M3: inductive neural pair model (DESIGN §5; pre-stated in config/m5m3.yaml `m3`; docs/m5m3.md).

Per outer fold k: the incumbent's nested stacker rows and outer eval rows (run_m2plus.FoldDataP, variant M2r,
cache m2feat_3d35f8c314e9) -> P = the incumbent's 475 pair features; gene tower input = label-free gene-table
columns (src.models.m2.gene_table, XSP excluded as for M2r). Seed s: fit on the gene-grouped fit rows of inner
fold j = s mod 5, early stopping on its validation rows (= the incumbent's rows); preprocessing fitted on the fit
rows only. Score = Huber-head output; seeds averaged in evaluate().

Usage: run_m3.py [--split-seed 7] [--permute-labels] [--folds 0,1] [--seeds 0] [--no-register]
  --permute-labels   control (config adoption.control_if_adopted.M3): (y, S2) of the stacker rows permuted
                     jointly within each outer fold (rng [20261028, k]); registered as M3_permlabels
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
from src import pairs as PA  # noqa: E402
from src.timing import Timings, peak_rss_mb, set_threads  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "m5m3.yaml").read_text())
THREADS = int(CFG["compute"]["threads"])
VARIANT = "M2r"
FAMILIES = ["LF", "RYAN", "NBR", "GP"]
PERM_SEED = 20261028
SYM_SAMPLE = 20_000


def m3_cfg() -> dict:
    """config/m5m3.yaml m3 section -> src.models.m3 cfg dict."""
    c = CFG["m3"]
    a, o, p = c["architecture"], c["optimiser"], c["preprocessing"]
    return {"input_dropout": a["input_dropout"], "gene_tower": a["gene_tower"], "pair_mlp": a["pair_mlp"],
            "hidden_dropout": 0.2, "bce_weight": 1.0, "lr": o["lr"], "weight_decay": o["weight_decay"],
            "batch": o["batch"], "max_epochs": o["max_epochs"], "patience": o["patience"],
            "clip": float(p["clip"][1]), "n_quantiles": 1000, "subsample": 200_000}


def out_dir() -> Path:
    d = protocol.results_root() / "m5m3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preds_dir() -> Path:
    d = protocol.preds_dir() / "m5m3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str) -> None:
    print(msg, flush=True)


def run(split_seed, permute: bool, folds=None, seeds=None, register=True) -> dict:
    import torch
    import run_m2 as R
    import run_m2plus as P2
    from src.evaluate import evaluate
    from src.models import m2, m3, stacker
    from src.ryanfeat import SimStore
    P2.THREADS = THREADS
    R.N_JOBS = THREADS
    torch.set_num_threads(THREADS)
    spec = PA.load_spec()
    sims = SimStore(names=SimStore().names)
    cfg = m3_cfg()
    seeds = seeds if seeds is not None else [int(s) for s in CFG["m3"]["seeds"]]
    n = protocol.load()["dev_cv"]["n_folds"]
    folds = folds if folds is not None else list(range(n))
    name = "M3" + ("_permlabels" if permute else "") + R.seed_tag(split_seed)
    excl = P2.tables_kw(VARIANT)["exclude_groups"]
    GT = m2.gene_table(excl)
    gidx = pd.Index(GT.index)
    delta = stacker.huber_delta()
    T = Timings()
    preds, fits = [], []
    t_all = time.perf_counter()
    for k in folds:
        D = P2.FoldDataP(split_seed, k, spec, sims, T, VARIANT)
        cols = D.cols(FAMILIES)
        nlf = len(D.lf_names)
        assert len(cols) == D.Xtr.shape[1], "M3 P = all incumbent features"
        y, s_ = D.y.astype(np.float32), D.s.astype(float)
        if permute:
            pr = np.random.default_rng([PERM_SEED, k]).permutation(len(y))
            y, s_ = y[pr], s_[pr]
        t = stacker.etarget(s_)
        tr, ev = D.tr_rows, D.ev
        ia_tr, ib_tr = gidx.get_indexer(tr.gene_a), gidx.get_indexer(tr.gene_b)
        ia_ev, ib_ev = gidx.get_indexer(ev.gene_a), gidx.get_indexer(ev.gene_b)
        assert (ia_tr >= 0).all() and (ib_tr >= 0).all() and (ia_ev >= 0).all() and (ib_ev >= 0).all()
        # swapped eval rows for the symmetry check (LF re-assembled with a/b swapped; Ryan-derived symmetric)
        idx = np.random.default_rng(k).choice(len(ev), min(SYM_SAMPLE, len(ev)), replace=False)
        sw = ev.iloc[idx].rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})
        Lsw = P2.assemble_lf(sw, spec, VARIANT)
        assert np.array_equal(Lsw, D.Xev[idx, :nlf], equal_nan=True), f"fold {k}: LF features not symmetric"
        Xsw = np.hstack([Lsw, D.Xev[idx, nlf:]])
        for s in seeds:
            j = s % D.n_inner
            fm, vm = D.groups[j]
            with T.stage(f"prep_f{k}_s{s}"):
                pp = m3.Preprocessor(cfg["n_quantiles"], cfg["subsample"], cfg["clip"]).fit(D.Xtr[fm])
                fit_genes = np.unique(np.r_[ia_tr[fm], ib_tr[fm]])
                gp = m3.Preprocessor(cfg["n_quantiles"], cfg["subsample"], cfg["clip"]).fit(
                    GT.to_numpy(np.float32)[fit_genes])
                G = gp.transform(GT.to_numpy(np.float32))
                Pf, Pv = pp.transform(D.Xtr[fm]), pp.transform(D.Xtr[vm])
            with T.stage(f"train_f{k}_s{s}"):
                net, inf = m3.fit_m3(G, ia_tr[fm], ib_tr[fm], Pf, t[fm], y[fm], D.w[fm], ia_tr[vm], ib_tr[vm], Pv,
                                     y[vm], D.w[vm], cfg, s, delta, log=log)
            del Pf, Pv
            with T.stage(f"predict_f{k}_s{s}"):
                sc = m3.predict_m3(net, G, ia_ev, ib_ev, pp.transform(D.Xev))
            ssw = m3.predict_m3(net, G, ib_ev[idx], ia_ev[idx], pp.transform(Xsw))
            md = float(np.max(np.abs(ssw - sc[idx])))
            assert md <= 1e-5, f"fold {k} seed {s}: score(a,b) != score(b,a) (max |diff| {md})"
            inf.update({"fold": k, "seed": s, "inner_fold_valid": int(j), "sym_max_abs_diff": md,
                        "d_pair_in": int(pp.n_out), "d_gene_in": int(gp.n_out)})
            fits.append(inf)
            preds.append(ev[["gene_a", "gene_b", "fold", "part"]].assign(score=sc, seed=s))
            log(f"  [{name}] fold {k} seed {s}: best_epoch {inf['best_epoch']}/{inf['epochs_run']} inner-valid wAP "
                f"{inf['valid_ap_weighted']:.4f} train {inf['train_s']} s; sym diff {md:.1e}")
        del D
    P = pd.concat(preds, ignore_index=True)
    res = None
    if folds == list(range(n)):
        cache = P2.cache_name(VARIANT, sims)
        fh = {**spec.hashes(exclude=P2.VARIANTS[VARIANT]["lf_exclude"]),
              **{f"sim/{kk}": vv for kk, vv in sims.hashes().items()}, "m2feat_cache": cache,
              **{f"gene_table/{kk}": vv for kk, vv in m2.gene_table_hashes(excl).items()}}
        res = evaluate(P, model=name, split_seed=split_seed, runtime=T.as_dict(), feature_hashes=fh,
                       register=register,
                       notes=f"M3 two-tower MLP (config/m5m3.yaml m3; src/models/m3.py) on {VARIANT} nested stacker "
                             f"rows; seeds={seeds}; torch/cpu threads={THREADS}; permute_labels={permute}")
        m = res["metrics_mean"]
        log(f"== {name}: run_id {res['run_id']} primary {res['primary']:.4f} C1 {m['ap_C1']:.4f} "
            f"C2 {m['ap_C2']:.4f} C3 {m['ap_C3']:.4f}")
    rec = {"name": name, "run_id": res and res["run_id"], "split_seed": data._seed(split_seed), "seeds": seeds,
           "folds": folds, "permute_labels": permute, "cfg": cfg, "threads": THREADS, "fits": fits,
           "runtime": T.as_dict(), "run_wall_s": round(time.perf_counter() - t_all, 1), "peak_rss_mb": peak_rss_mb()}
    if res is not None and register:
        P.to_parquet(preds_dir() / f"{res['run_id']}_perseed.parquet", index=False)
        rec.update(primary=res["primary"], metrics_mean=res["metrics_mean"], bootstrap=res["bootstrap"]["primary"])
        (out_dir() / f"m3_{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--folds", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    set_threads(THREADS)
    folds = [int(x) for x in a.folds.split(",")] if a.folds else None
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
    rec = run(a.split_seed, a.permute_labels, folds, seeds, register=not a.no_register)
    log(f"wall {rec['run_wall_s']} s; peak RSS {rec['peak_rss_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
