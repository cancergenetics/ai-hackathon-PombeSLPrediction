#!/usr/bin/env python
"""Render the tables of docs/m1.md from M1 run reports (results/m1/report_*.json, results/toy/m1/...).

Usage: m1_report.py --headline R.json [--ablation R.json] [--toy R.json] > tables.md
Prints markdown tables (headline, per-class, importance, ablation, permutation, runtime) to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def row(r: dict) -> str:
    m = r["metrics_mean"]
    return (f"| {r['model']} | {r['primary']:.4f} ± {r['se']:.4f} | {m['ap_C1']:.4f} | {m['ap_C2']:.4f} | "
            f"{m['ap_C3']:.4f} | {m['prevalence_primary']:.4f} | {m['lift']:.2f} | {m['auroc']:.3f} | "
            f"{r['runtime']['wall_s']:.0f} | {r['runtime']['peak_rss_mb']:.0f} | `{r['run_id']}` |")


HEAD = ("| model | primary ± SE | AP C1 | AP C2 | AP C3 | weighted prev. | lift | AUROC | wall s | peak MB | run_id |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|")


def b1_rows(reg: Path) -> list:
    if not reg.exists():
        return []
    r = pd.read_csv(reg, sep="\t")
    r = r[r.model.astype(str).str.startswith("B1")]
    return [f"| {x.model} | {x.primary:.4f} ± {x.primary_se:.4f} | {x.ap_C1:.4f} | {x.ap_C2:.4f} | {x.ap_C3:.4f} | "
            f"{x.weighted_prevalence:.4f} | {x.lift:.2f} | {x.auroc:.3f} | | | `{x.run_id}` |" for x in r.itertuples()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", required=True)
    ap.add_argument("--ablation")
    ap.add_argument("--toy")
    a = ap.parse_args()
    H = json.loads(Path(a.headline).read_text())
    f = H["full"]
    print("### Headline\n" + HEAD + "\n" + row(f))
    A = json.loads(Path(a.ablation).read_text()) if a.ablation else None
    if A and A.get("permutation"):
        print(row(A["permutation"]))
    print("\n### Gain importance (mean normalised gain over fold × seed models), top 30\n| # | feature | gain |\n|---|---|---|")
    for i, (k, v) in enumerate(f["importance_top30"].items(), 1):
        print(f"| {i} | {k} | {v:.4f} |")
    print("\n### Gain importance by group\n| group | gain share |\n|---|---|")
    for k, v in f["importance_by_group"].items():
        print(f"| {k} | {v:.3f} |")
    print("\n### Early stopping (inner gene-grouped validation)\n| fold | seed | best_iter | inner valid AP |\n|---|---|---|---|")
    for x in f["fits"]:
        print(f"| {x['fold']} | {x['seed']} | {x['best_iteration']} | {x['valid_ap']:.4f} |")
    if A:
        ref = A.get("ablation_reference") or A["full"]
        print(f"\n### Ablation (drop one group; reference `{ref['run_id']}` primary {ref['primary']:.4f}, "
              f"seeds {ref.get('seeds')})\n"
              "| dropped group | primary | Δ = ref − ablated | 95% CI (paired gene bootstrap) | AP C1 | AP C2 | AP C3 | run_id |\n"
              "|---|---|---|---|---|---|---|---|")
        for g, s in A["ablations"].items():
            c = s.get("compare_ref_minus_ablated") or s.get("compare_full_minus_ablated") or {}
            m = s["metrics_mean"]
            print(f"| {g} | {s['primary']:.4f} | {c.get('delta', float('nan')):+.4f} | "
                  f"[{c.get('ci_lo', float('nan')):+.4f}, {c.get('ci_hi', float('nan')):+.4f}] | {m['ap_C1']:.4f} | "
                  f"{m['ap_C2']:.4f} | {m['ap_C3']:.4f} | `{s['run_id']}` |")
    b1 = b1_rows(ROOT / "results" / "registry.tsv")
    if b1:
        print("\n### B1 (registry)\n" + HEAD + "\n" + "\n".join(b1))
    if a.toy:
        T = json.loads(Path(a.toy).read_text())
        print("\n### Toy\n" + HEAD + "\n" + row(T["full"]) + ("\n" + row(T["permutation"]) if T.get("permutation") else ""))
        if T.get("ablations"):
            print("\n| dropped group | primary | Δ | 95% CI | run_id |\n|---|---|---|---|---|")
            for g, s in T["ablations"].items():
                c = s.get("compare_ref_minus_ablated") or s.get("compare_full_minus_ablated") or {}
                print(f"| {g} | {s['primary']:.4f} | {c.get('delta', 0):+.4f} | [{c.get('ci_lo', 0):+.4f}, "
                      f"{c.get('ci_hi', 0):+.4f}] | `{s['run_id']}` |")
        tb1 = b1_rows(ROOT / "results" / "toy" / "registry.tsv")
        if tb1:
            print("\n" + HEAD + "\n" + "\n".join(tb1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
