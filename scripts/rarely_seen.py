#!/usr/bin/env python
"""Rarely-seen analysis (config/fewshot.yaml, docs/rarely_seen.md). Report-only.

  python scripts/rarely_seen.py degree     # stage 1: AUPRC vs seen-gene degree, from existing OOF predictions
  python scripts/rarely_seen.py fewshot    # stage 2: k = 0/5/20 from scripts/run_fewshot.py predictions
  python scripts/rarely_seen.py figure     # stage 3: results/rarely_seen/rarely_seen.png
Uses src/evaluate.py machinery (AP core, Poisson(1) gene bootstrap counts, pooled_metrics, bootstrap_replicates).
Writes aggregates only to results/rarely_seen/ (no pair-level scores).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data, protocol  # noqa: E402
from src import evaluate as EV  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "fewshot.yaml").read_text())
OUT = ROOT / "results" / "rarely_seen"
OUT.mkdir(parents=True, exist_ok=True)
LVL = protocol.load()["bootstrap"]["ci_level"]
N_FOLDS = protocol.load()["dev_cv"]["n_folds"]


def pred_frame(run_id: str) -> pd.DataFrame:
    p = pd.read_parquet(protocol.preds_dir() / f"{run_id}.parquet")
    return p.merge(EV._labels_for(None), on=["gene_a", "gene_b"], how="left")


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, (1 - LVL) / 2)), float(np.quantile(x, 1 - (1 - LVL) / 2))]


# ============================================================================================ degree
def add_degree(df: pd.DataFrame) -> pd.DataFrame:
    """seen-gene degree (training-pair count) and training-positive count per eval row; NaN for C3."""
    out = []
    for k, d in df.groupby("fold"):
        tr = data.fold_split(None, int(k))[0]
        a = pd.concat([tr[["gene_a", "y"]].rename(columns={"gene_a": "g"}), tr[["gene_b", "y"]].rename(columns={"gene_b": "g"})])
        deg = a.groupby("g").size()
        pos = a.groupby("g").y.sum()
        da, db = d.gene_a.map(deg), d.gene_b.map(deg)
        pa, pb = d.gene_a.map(pos), d.gene_b.map(pos)
        d = d.copy()
        # C2: the seen gene is the one with training pairs; C1: both seen -> the rarer gene (smaller degree)
        d["deg"] = np.fmin(da, db)
        d["npos"] = np.where(da.fillna(1e9) <= db.fillna(1e9), pa, pb)
        d.loc[d.part == "C3", ["deg", "npos"]] = np.nan
        out.append(d)
    return pd.concat(out, ignore_index=True)


def binned_ap(d: pd.DataFrame, counts, gidx, mask_fn, chunk=50):
    """per-fold AP (point + Poisson-bootstrap replicates) on rows mask_fn(d); fold-mean. Returns dict."""
    R = counts.shape[0]
    pt, rep, prv, prv_rep, n, npos = [], [], [], [], 0, 0
    for k, f in d.groupby("fold"):
        f = f[mask_fn(f)]
        if len(f) == 0 or f.y.sum() == 0:
            continue
        order, ends = EV._sort_groups(f.score.to_numpy(float))
        y = f.y.to_numpy(bool)[order]
        ia = gidx.get_indexer(f.gene_a.to_numpy()[order])
        ib = gidx.get_indexer(f.gene_b.to_numpy()[order])
        pt.append(EV.ap_weighted_matrix(y, np.ones(len(y)), ends)[0])
        prv.append(y.mean())
        r_ap, r_pv = np.zeros(R), np.zeros(R)
        for r0 in range(0, R, chunk):
            c = counts[r0:r0 + chunk].T.astype(np.float64)
            W = c[ia] * c[ib]
            r_ap[r0:r0 + chunk] = EV.ap_weighted_matrix(y, W, ends)
            with np.errstate(invalid="ignore", divide="ignore"):
                r_pv[r0:r0 + chunk] = (W * y[:, None]).sum(0) / W.sum(0)
        rep.append(r_ap)
        prv_rep.append(r_pv)
        n += len(f)
        npos += int(y.sum())
    if not pt:
        return None
    ap, pv = float(np.mean(pt)), float(np.mean(prv))
    ra, rp = np.nanmean(rep, axis=0), np.nanmean(prv_rep, axis=0)
    return {"ap": ap, "ap_ci": ci(ra), "prevalence": pv, "lift": ap / pv, "lift_ci": ci(ra / rp),
            "n_pairs": n, "n_pos": npos, "n_folds": len(pt)}


def stage_degree():
    I = CFG["incumbent"]
    runs = {"B1": I["reference_run_ids"]["B1"], "B4": I["reference_run_ids"]["B4"], "M2": I["reference_run_ids"]["M2"],
            "M2r_huber (incumbent)": I["run_id"], "M2r_bag5_huber": I["extra_run_ids"][0]}
    counts, gidx = EV.bootstrap_counts(data.dev_folds(None).index)
    frames = {n: add_degree(pred_frame(r)) for n, r in runs.items()}
    base = frames["M2r_huber (incumbent)"]
    seen = base[base.part.isin(["C1", "C2"])]
    nq = CFG["degree"]["n_quantile_bins"]
    edges = np.unique(np.quantile(seen.deg, np.linspace(0, 1, nq + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pb = list(CFG["degree"]["pos_bins"]) + [np.inf]
    out = {"runs": runs, "deg_edges_inner": [float(e) for e in edges[1:-1]], "pos_edges": [float(p) for p in pb[:-1]],
           "deg_summary": {c: seen[seen.part == c].deg.describe().round(1).to_dict() for c in ("C1", "C2")},
           "bins": []}
    specs = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lab = f"deg [{'min' if lo == -np.inf else int(lo)}, {'max' if hi == np.inf else int(hi)})"
        specs.append(("deg", lab, lambda f, lo=lo, hi=hi: ((f.deg >= lo) & (f.deg < hi)).to_numpy()))
    for i in range(len(pb) - 1):
        lo, hi = pb[i], pb[i + 1]
        lab = f"pos [{int(lo)}, {'inf' if hi == np.inf else int(hi)})"
        specs.append(("npos", lab, lambda f, lo=lo, hi=hi: ((f.npos >= lo) & (f.npos < hi)).to_numpy()))
    for kind, lab, fn in specs:
        for cls in ("C1", "C2"):
            for name, df in frames.items():
                d = df[df.part == cls]
                col = kind
                r = binned_ap(d, counts, gidx, lambda f, fn=fn: fn(f))
                if r is None:
                    continue
                out["bins"].append({"kind": kind, "bin": lab, "class": cls, "model": name, **r})
                print(kind, lab, cls, name, f"AP {r['ap']:.3f} prev {r['prevalence']:.3f} lift {r['lift']:.1f} n {r['n_pairs']} pos {r['n_pos']}", flush=True)
    (OUT / "degree_bins.json").write_text(json.dumps(out, indent=1, default=float))
    pd.DataFrame(out["bins"]).drop(columns=["ap_ci", "lift_ci"]).assign(
        ap_lo=[b["ap_ci"][0] for b in out["bins"]], ap_hi=[b["ap_ci"][1] for b in out["bins"]],
        lift_lo=[b["lift_ci"][0] for b in out["bins"]], lift_hi=[b["lift_ci"][1] for b in out["bins"]]
    ).to_csv(OUT / "degree_bins.tsv", sep="\t", index=False)


# ============================================================================================ fewshot
def avg_seed(P: pd.DataFrame, seeds) -> pd.DataFrame:
    P = P[P.seed.isin(seeds)]
    return P.groupby(["gene_a", "gene_b", "fold", "part"], sort=False, observed=True).score.mean().reset_index()


def stage_fewshot():
    import run_fewshot as F
    I, FS = CFG["incumbent"], CFG["fewshot"]
    seeds = FS["seeds"]
    ks = FS["ks"]
    pdir = F.preds_dir()
    folds = [f for f in FS["folds_order"] if all((pdir / f"k{k}_fold{f}.parquet").exists() for k in ks)]
    print("completed folds:", folds)
    lab = EV._labels_for(None)
    k0 = pd.read_parquet(protocol.preds_dir() / "m2plus" / f"{I['run_id']}_perseed.parquet")
    k0 = avg_seed(k0[k0.fold.isin(folds)], seeds)
    frames = {}
    for f in folds:
        E = F.eval_set(None, f)[["gene_a", "gene_b", "fold", "part"]]
        frames.setdefault(0, []).append(E.merge(k0, on=["gene_a", "gene_b", "fold", "part"], how="left"))
        for k in ks:
            P = avg_seed(pd.read_parquet(pdir / f"k{k}_fold{f}.parquet"), seeds)
            m = E.merge(P, on=["gene_a", "gene_b", "fold", "part"], how="left")
            assert m.score.notna().all(), (f, k)
            frames.setdefault(k, []).append(m)
        assert frames[0][-1].score.notna().all()
    frames = {k: pd.concat(v, ignore_index=True).merge(lab, on=["gene_a", "gene_b"], how="left") for k, v in frames.items()}
    cw = protocol.class_weights()
    W = {"primary": cw, "C2": {"C1": 0, "C2": 1, "C3": 0}, "C3": {"C1": 0, "C2": 0, "C3": 1}}
    counts, gidx = EV.bootstrap_counts(data.dev_folds(None).index)
    res, rep = {}, {}
    for k, df in frames.items():
        met = EV.pooled_metrics(df, W)
        rep[k] = EV.bootstrap_replicates(df, counts, gidx, W)
        m = met["mean"]
        res[k] = {"pooled_C2C3": m["pooled"], "C2": m["pooled_C2"], "C3": m["pooled_C3"],
                  "prev_pooled": m["prevalence_primary"], "n_C2": m["n_C2"] * 1.0, "n_C3": m["n_C3"] * 1.0,
                  "pos_C2": m["pos_C2"], "pos_C3": m["pos_C3"],
                  "per_fold": {int(f): {"pooled_C2C3": met["per_fold"][f]["pooled"], "C2": met["per_fold"][f]["pooled_C2"],
                                       "C3": met["per_fold"][f]["pooled_C3"]} for f in met["per_fold"]}}
        for n, key in (("pooled_C2C3", "primary"), ("C2", "C2"), ("C3", "C3")):
            res[k][n + "_ci"] = ci(rep[k][key])
    out = {"folds": folds, "seeds": seeds, "ks": [0] + ks, "k0_source": I["run_id"], "weights": cw, "by_k": res,
           "paired": {}}
    for k in ks:
        for n, key in (("pooled_C2C3", "primary"), ("C2", "C2"), ("C3", "C3")):
            d = rep[k][key] - rep[0][key]
            out["paired"][f"{n}_k{k}_minus_k0"] = {"delta": res[k][n] - res[0][n], "ci": ci(d)}
    k = max(ks)
    d = rep[k]["primary"] - rep[ks[0]]["primary"]
    out["paired"][f"pooled_C2C3_k{k}_minus_k{ks[0]}"] = {"delta": res[k]["pooled_C2C3"] - res[ks[0]]["pooled_C2C3"], "ci": ci(d)}
    # context: revealed information per held-out gene, and gain for genes that received >= 1 revealed positive
    ctx = {}
    for f in folds:
        held = set(data.fold_genes(None, f))
        for k in ks:
            r = F.reveal_pairs(None, f, k).merge(lab, on=["gene_a", "gene_b"], how="left")
            g = pd.concat([r[["gene_a", "y"]].rename(columns={"gene_a": "g"}), r[["gene_b", "y"]].rename(columns={"gene_b": "g"})])
            g = g[g.g.isin(held)].groupby("g").y.agg(["size", "sum"]).reindex(sorted(held)).fillna(0)
            ctx[f"fold{f}_k{k}"] = {"pairs_per_gene_mean": float(g["size"].mean()), "pos_per_gene_mean": float(g["sum"].mean()),
                                    "frac_genes_with_pos": float((g["sum"] > 0).mean())}
    out["revealed_context"] = ctx
    # C2 pairs whose held-out gene received >=1 revealed positive vs none (k = max): report-only
    k = max(ks)
    strat = {}
    pos_gene = {}
    for f in folds:
        held = set(data.fold_genes(None, f))
        r = F.reveal_pairs(None, f, k).merge(lab, on=["gene_a", "gene_b"], how="left")
        r = r[r.y == 1]
        pos_gene[f] = set(pd.concat([r.gene_a, r.gene_b])) & held
    for nm, flag in (("held-out gene has >=1 revealed positive", True), ("no revealed positive", False)):
        sub = {}
        for kk in (0, k):
            d = frames[kk].copy()
            hit = np.array([(a in pos_gene[f]) or (b in pos_gene[f]) for a, b, f in zip(d.gene_a, d.gene_b, d.fold)])
            d["hit"] = hit
            sub[kk] = d
        r2 = {}
        for kk in (0, k):
            r2[f"k{kk}"] = binned_ap(sub[kk][sub[kk].part == "C2"], counts, gidx, lambda f: (f.hit == flag).to_numpy())
        strat[nm] = r2
    out["c2_by_revealed_positive_k%d" % k] = strat
    (OUT / "fewshot.json").write_text(json.dumps(out, indent=1, default=float))
    for k in out["ks"]:
        r = res[k]
        print(f"k={k}: pooled C2+C3 {r['pooled_C2C3']:.4f} {r['pooled_C2C3_ci']} C2 {r['C2']:.4f} {r['C2_ci']} C3 {r['C3']:.4f} {r['C3_ci']}")
    for n, v in out["paired"].items():
        print(n, v)
    print(json.dumps(strat, indent=1, default=float)[:1500])


# ============================================================================================ figure
def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = json.loads((OUT / "fewshot.json").read_text()) if (OUT / "fewshot.json").exists() else None
    dg = json.loads((OUT / "degree_bins.json").read_text())
    col = {"B1": "#999999", "B4": "#56B4E9", "M2": "#E69F00", "M2r_huber (incumbent)": "#0072B2", "M2r_bag5_huber": "#009E73"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    if fs:
        ks = fs["ks"]
        for i, (nm, key, c) in enumerate((("C2 (one gene held out)", "C2", "#0072B2"), ("C3 (both held out)", "C3", "#D55E00"),
                                          ("C2+C3 pooled", "pooled_C2C3", "#000000"))):
            y = [fs["by_k"][str(k)][key] for k in ks]
            lo = [fs["by_k"][str(k)][key + "_ci"][0] for k in ks]
            hi = [fs["by_k"][str(k)][key + "_ci"][1] for k in ks]
            xs = np.arange(len(ks)) + (i - 1) * 0.04
            ax[0].errorbar(xs, y, yerr=[np.array(y) - lo, np.array(hi) - y], marker="o", color=c, label=nm, capsize=3)
        ax[0].axhline(fs["by_k"]["0"]["prev_pooled"], ls=":", color="grey", label="prevalence (C2+C3)")
        ax[0].set_xticks(range(len(ks)))
        ax[0].set_xticklabels([f"k={k}" for k in ks])
        ax[0].set_ylabel("AUPRC (remaining pairs)")
        ax[0].set_title(f"Few-shot: revealed pairs per new gene\n({len(fs['folds'])} folds, {len(fs['seeds'])} seeds, 95% gene-bootstrap CI)", fontsize=10)
        ax[0].legend(fontsize=8)
    B = pd.DataFrame(dg["bins"])
    for j, cls in enumerate(("C2", "C1")):
        a = ax[1 + j]
        d = B[(B["class"] == cls) & (B.kind == "deg")]
        bins = list(dict.fromkeys(d.bin))
        for mname, c in col.items():
            dm = d[d.model == mname].set_index("bin").reindex(bins)
            y = dm.ap.to_numpy(float)
            lo = np.array([x[0] for x in dm.ap_ci])
            hi = np.array([x[1] for x in dm.ap_ci])
            xs = np.arange(len(bins)) + (list(col).index(mname) - 2) * 0.07
            a.errorbar(xs, y, yerr=[y - lo, hi - y], marker="o", ms=4, color=c, label=mname, capsize=2, lw=1)
        a.plot(np.arange(len(bins)), dm.prevalence.to_numpy(float), ls=":", color="k", label="prevalence")
        a.set_xticks(range(len(bins)))
        a.set_xticklabels([b.replace("deg ", "") for b in bins], fontsize=8)
        a.set_xlabel("seen gene's training-pair count" + (" (smaller of the two)" if cls == "C1" else ""))
        a.set_ylabel("AUPRC")
        a.set_title(f"{cls} AUPRC by seen-gene degree (no retraining)", fontsize=10)
        if j == 0:
            a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "rarely_seen.png", dpi=150)


if __name__ == "__main__":
    {"degree": stage_degree, "fewshot": stage_fewshot, "figure": stage_figure}[sys.argv[1]]()
