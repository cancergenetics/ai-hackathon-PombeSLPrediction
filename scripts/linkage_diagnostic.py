#!/usr/bin/env python
"""Linkage diagnostic (DESIGN.md §1 "Linkage"; report-only, dev genes only, never the lockbox).

Question: are Ryan S2 hits among tightly linked gene pairs (same chromosome, small distance) artefactual?
Double mutants of linked genes are hard to recover by random spore / mating, so colony sizes there may
reflect linkage, not interaction, and E-MAP pipelines filter close pairs. If the hit rate rises (or the
pair coverage drops) at short distance, the deliverable flags such pairs (`linked`).

For same-chromosome dev x dev pairs, by distance bin (gap between gene bodies, PomBase coordinates):
  n measured pairs, hits (S2 label), hit rate (+ Wilson 95% CI);
  coverage = measured pairs / design-testable pairs (one gene a Ryan query, the other a Ryan query or
             array gene); lockbox-C1 pairs (10% of dev x dev eval-eligible pairs, random) are absent from
             the dev labels in every bin alike;
  if an M2 dev-CV run is registered: mean out-of-fold score and the fraction of pairs in the top-q of all
  OOF scores (q = overall OOF-pair hit rate), comparable to the hit rate.
Reference rows: all different-chromosome dev pairs.

Distance is NEVER a model feature. Outputs: results/linkage/linkage_by_distance.tsv, linkage.png,
linkage_summary.json (aggregates only).
Usage: linkage_diagnostic.py [--download] [--oof-run RUN_ID | --no-oof]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, protocol  # noqa: E402
from src import universe as U  # noqa: E402


def wilson(k, n, z=1.959964):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def design_testable(genes: list, roles: pd.Series | None) -> np.ndarray | None:
    if roles is None:
        return None
    r = roles.reindex(genes).fillna("none")
    q = r.isin(["query", "both"]).to_numpy()
    a = r.isin(["array", "both", "query"]).to_numpy()
    return q, a


def bin_table(gap_kb: np.ndarray, y: np.ndarray, bins: list, oof=None, top=None) -> pd.DataFrame:
    edges = list(bins) + [np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gap_kb >= lo) & (gap_kb < hi)
        r = {"bin_kb": f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f">= {lo:g}", "lo_kb": lo, "hi_kb": hi,
             "n_pairs": int(m.sum()), "hits": int(y[m].sum())}
        r["hit_rate"] = r["hits"] / r["n_pairs"] if r["n_pairs"] else np.nan
        r["hit_rate_lo"], r["hit_rate_hi"] = (float(x) for x in wilson(r["hits"], r["n_pairs"]))
        if oof is not None:
            mo = m & np.isfinite(oof)
            r["n_oof"] = int(mo.sum())
            r["oof_mean_score"] = float(oof[mo].mean()) if mo.any() else np.nan
            r["oof_top_q_rate"] = float(top[mo].mean()) if mo.any() else np.nan
            r["hit_rate_oof_pairs"] = float(y[mo].mean()) if mo.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch/verify PomBase coordinate files first")
    ap.add_argument("--oof-run", default=None)
    ap.add_argument("--no-oof", action="store_true")
    ap.add_argument("--out", default=None, help="output dir (default <results>/linkage)")
    a = ap.parse_args()
    C = U.deliverable_cfg()["linkage"]
    if a.download:
        U.download_coords()
    coords = U.load_coords()
    dev = data.dev_pairs(include_train_drop=True)         # dev genes only (lockbox never read)
    y = dev.y.to_numpy(bool)
    same, gap = U.pair_distance(coords, dev.gene_a.to_numpy(object), dev.gene_b.to_numpy(object))
    genes = sorted(set(dev.gene_a) | set(dev.gene_b))
    has = coords.index.intersection(genes)
    # ---- OOF scores
    oof = top = None
    run = None if a.no_oof else (a.oof_run or U.latest_run_id(U.deliverable_cfg()["oof"]["model"]))
    if run and (protocol.preds_dir() / f"{run}.parquet").exists():
        o = U.oof_scores(run)
        j = dev[["gene_a", "gene_b"]].merge(o, on=["gene_a", "gene_b"], how="left")
        oof = j.oof_score.to_numpy(float)
        fin = np.isfinite(oof)
        q = y[fin].mean()
        thr = np.quantile(oof[fin], 1 - q)
        top = np.where(fin, oof >= thr, False)
    else:
        run = None
    # ---- same-chromosome table
    gap_kb = gap / 1000.0
    T = bin_table(gap_kb[same], y[same], C["bins_kb"], None if oof is None else oof[same],
                  None if top is None else top[same])
    # coverage vs design-testable pairs (dev genes, same chromosome)
    roles = data._roles(str(protocol.data_root()))
    dt = design_testable(genes, roles)
    if dt is not None:
        q_, a_ = dt
        chrom, start, end = U.gene_coord_arrays(coords, genes)
        n = len(genes)
        cov_n = np.zeros(len(T), np.int64)
        edges = np.r_[np.asarray(C["bins_kb"], float), np.inf] * 1000
        for i in range(n - 1):
            jj = np.arange(i + 1, n)
            s, g = U.gap_from_arrays(chrom, start, end, np.full(len(jj), i), jj)
            testable = (q_[i] & a_[jj]) | (q_[jj] & a_[i])
            k = s & testable
            if k.any():
                b = np.searchsorted(edges, g[k], side="right") - 1
                cov_n += np.bincount(b, minlength=len(T))[:len(T)]
        T.insert(4, "design_testable_pairs", cov_n)
        T.insert(5, "coverage", T.n_pairs / T.design_testable_pairs.replace(0, np.nan))
    # reference: different chromosomes
    d = ~same & (gap == -1)
    both_coords = dev.gene_a.isin(has).to_numpy() & dev.gene_b.isin(has).to_numpy()
    d &= both_coords
    ref = {"bin_kb": "different chromosome", "n_pairs": int(d.sum()), "hits": int(y[d].sum()),
           "hit_rate": float(y[d].mean())}
    ref["hit_rate_lo"], ref["hit_rate_hi"] = (float(x) for x in wilson(ref["hits"], ref["n_pairs"]))
    if oof is not None:
        md = d & np.isfinite(oof)
        ref.update({"n_oof": int(md.sum()), "oof_mean_score": float(oof[md].mean()),
                    "oof_top_q_rate": float(top[md].mean()), "hit_rate_oof_pairs": float(y[md].mean())})
    T = pd.concat([T, pd.DataFrame([ref])], ignore_index=True)
    # windowed summaries
    W = U.linked_window_bp() / 1000.0
    win = same & (gap_kb < W) & (gap >= 0)
    out = Path(a.out) if a.out else protocol.results_root() / "linkage"
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "linkage_by_distance.tsv", sep="\t", index=False, float_format="%.6g")
    from scipy.stats import fisher_exact
    far = same & (gap_kb >= W)
    tab = [[int(y[win].sum()), int((~y[win]).sum())], [int(y[far].sum()), int((~y[far]).sum())]]
    summ = {"data_root": str(protocol.data_root()), "n_dev_pairs": int(len(dev)), "n_dev_genes": len(genes),
            "dev_genes_with_coords": int(len(has)), "n_same_chrom_pairs": int(same.sum()),
            "window_kb": W, "within_window": {"n": int(win.sum()), "hits": int(y[win].sum()),
                                              "hit_rate": float(y[win].mean()) if win.any() else None},
            "same_chrom_outside_window": {"n": int(far.sum()), "hit_rate": float(y[far].mean())},
            "different_chrom": {"n": ref["n_pairs"], "hit_rate": ref["hit_rate"]},
            "fisher_window_vs_outside": {"odds_ratio": float(fisher_exact(tab)[0]), "p": float(fisher_exact(tab)[1])},
            "min_gap_kb_measured": float(gap_kb[same].min()) if same.any() else None,
            "oof_run": run, "coords_sha256": {f: protocol.sha256_file(U.coords_dir() / f)
                                              for f in U.deliverable_cfg()["genome"]["sha256"]}}
    bands = []
    be = list(C.get("bands_kb", [])) + [np.inf]
    dtab = [int(y[d].sum()), int((~y[d]).sum())]
    for lo, hi in zip(be[:-1], be[1:]):
        m = same & (gap >= 0) & (gap_kb >= lo) & (gap_kb < hi)
        if not m.any():
            continue
        t = [[int(y[m].sum()), int((~y[m]).sum())], dtab]
        orr, p = fisher_exact(t)
        lo_, hi_ = wilson(int(y[m].sum()), int(m.sum()))
        b = {"band_kb": [lo, None if not np.isfinite(hi) else hi], "n": int(m.sum()), "hits": int(y[m].sum()),
             "hit_rate": float(y[m].mean()), "hit_rate_ci": [float(lo_), float(hi_)],
             "odds_ratio_vs_diff_chrom": float(orr), "fisher_p_vs_diff_chrom": float(p)}
        if oof is not None:
            mo = m & np.isfinite(oof)
            b.update({"n_oof": int(mo.sum()), "hit_rate_oof_pairs": float(y[mo].mean()),
                      "oof_top_q_rate": float(top[mo].mean())})
        bands.append(b)
    summ["bands"] = bands
    (out / "linkage_summary.json").write_text(json.dumps(summ, indent=1))
    plot(T, out / "linkage.png", ref, run)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(T.to_string(index=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "coords_sha256"}, indent=1))
    return 0


def plot(T: pd.DataFrame, path: Path, ref: dict, run) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, blue, orange, grid = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e4e3df"
    S = T[T.bin_kb != "different chromosome"]
    x = np.arange(len(S))
    panels = [("hit rate (S2 < threshold)", "hit_rate")]
    if "coverage" in S:
        panels.append(("coverage (measured / design-testable)", "coverage"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    for ax, (title, col) in zip(axes[0], panels):
        if col == "hit_rate":
            lo, hi = S.hit_rate_lo.to_numpy(), S.hit_rate_hi.to_numpy()
            ax.fill_between(x, lo, hi, color=blue, alpha=0.15, linewidth=0)
            ax.plot(x, S.hit_rate, color=blue, lw=2, marker="o", ms=5, label="observed hit rate")
            if "oof_top_q_rate" in S:
                ax.plot(x, S.oof_top_q_rate, color=orange, lw=2, marker="s", ms=5,
                        label="M2 OOF: fraction in top-q")
            ax.axhline(ref["hit_rate"], color=muted, lw=1, ls="--", label="different chromosome")
            ax.legend(frameon=False, fontsize=8, labelcolor=ink)
        else:
            ax.plot(x, S[col], color=blue, lw=2, marker="o", ms=5)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(S.bin_kb, rotation=45, ha="right", fontsize=7, color=muted)
        ax.set_xlabel("gap between gene bodies (kb), same chromosome", fontsize=8, color=muted)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Ryan dev pairs vs chromosomal distance{'' if run is None else ' (OOF: ' + run + ')'}",
                 fontsize=9, color=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
