#!/usr/bin/env python
"""Generated tables of docs/m2.md (between the '<!-- BEGIN generated -->' / '<!-- END generated -->'
markers; prose outside the markers is hand-written). Inputs: results/m2/*.json, results/runs/<run_id>.json,
results/m2_toy_verification.json. Usage: .venv/bin/python scripts/m2_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import protocol  # noqa: E402

R = protocol.results_root()
M2 = R / "m2"
DOC = ROOT / "docs" / "m2.md"
BEGIN, END = "<!-- BEGIN generated -->", "<!-- END generated -->"


def run(rid):
    return json.loads((R / "runs" / f"{rid}.json").read_text()) if rid else None


def j(p):
    p = M2 / p if not Path(p).is_absolute() else Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, d=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def stage_runtime(r):
    st = r["runtime"].get("stages", [])
    out = {"feature_build": 0.0, "load_features": 0.0, "assemble": 0.0, "train": 0.0, "predict": 0.0,
           "other": 0.0}
    for s in st:
        n = s["stage"]
        key = ("feature_build" if n.startswith("nested") else "load_features" if n.startswith("load_features")
               else "assemble" if n.startswith("assemble") else "train" if n.startswith("train")
               else "predict" if n.startswith("predict") else "other")
        out[key] += s["wall_s"]
    return out


def main() -> int:
    cmp = j("comparisons.json")
    S0, S7 = j("summary.json"), j("summary_split7.json")
    L = []
    runs = cmp["runs"] if cmp else {}
    order = ["B1_additive_logit", "B4_logistic_MF", "M1", "M1N", "M2_minus_GP", "M2_minus_NBR", "M2", "M2_init"]
    L += ["### Dev CV results (split seed %s; seeds averaged inside evaluate)" % protocol.load()["dev_cv"]["split_seed"],
          "", "| model | primary ± SE | 95% CI | AP C1 | AP C2 | AP C3 | AUROC | all-pairs | equal | P@1k | seeds | "
          "git clean | run_id |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = run(runs.get(m))
        if r is None:
            continue
        mm, b = r["metrics_mean"], r["bootstrap"]["primary"]
        L.append(f"| {m} | {r['primary']:.4f} ± {b['se']:.4f} | {b['ci'][0]:.3f}–{b['ci'][1]:.3f} | {mm['ap_C1']:.4f} | "
                 f"{mm['ap_C2']:.4f} | {mm['ap_C3']:.4f} | {mm['auroc']:.3f} | {mm['pooled_all_pairs']:.4f} | "
                 f"{mm['pooled_equal']:.4f} | {mm['precision_at_k']:.3f} | {','.join(map(str, r['seeds']))} | "
                 f"{not r['git_dirty']} | `{r['run_id']}` |")
    prev = run(runs.get("M2"))["metrics_mean"]["prevalence_primary"] if runs.get("M2") else None
    L += ["", f"Weighted prevalence (B0) = {fmt(prev)}. Per-class AP is unweighted within the class.", ""]
    if cmp and cmp.get("runs_alt_seed"):
        L += ["### Seed-7 dev partition (DESIGN §3 ladder check)", "",
              "| model | primary ± SE | AP C1 | AP C2 | AP C3 | run_id |", "|---|---|---|---|---|---|"]
        for m, rid in cmp["runs_alt_seed"].items():
            r = run(rid)
            if r:
                mm = r["metrics_mean"]
                L.append(f"| {r['model']} | {r['primary']:.4f} ± {r['bootstrap']['primary']['se']:.4f} | "
                         f"{mm['ap_C1']:.4f} | {mm['ap_C2']:.4f} | {mm['ap_C3']:.4f} | `{rid}` |")
        L.append("")
    if cmp:
        L += ["### Paired comparisons (src/evaluate.compare; Δ = primary(a) − primary(b), shared gene bootstrap)", "",
              "| comparison (a − b) | Δ primary | 95% CI | Δ all-pairs [CI] | Δ equal [CI] | seed-7 Δ | clean | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for name, c in cmp["comparisons"].items():
            ap_, eq_ = c["ci_all_pairs"], c["ci_equal"]
            L.append(f"| {c['model_a']} − {c['model_b']} | {c['delta']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
                     f"{c['delta_all_pairs']:+.4f} [{ap_[0]:+.3f}, {ap_[1]:+.3f}] | {c['delta_equal']:+.4f} "
                     f"[{eq_[0]:+.3f}, {eq_[1]:+.3f}] | {fmt(c['alt_delta'])} | {c['clean']} | "
                     f"{'ADOPT' if c['adopt'] else 'not adopted: ' + c['reason']} |")
        L.append("")
    if S0:
        L += ["### Per-run details (M2 family)", "",
              "| run | n features | best_iter (min–median–max) | inner-valid wAP (mean) | importance by family | "
              "fold primaries |", "|---|---|---|---|---|---|"]
        for name, v in S0["runs"].items():
            bi = [f["best_iteration"] for f in v["fits"]]
            va = np.mean([f["valid_ap_weighted"] for f in v["fits"]])
            fam = ", ".join(f"{k} {x:.2f}" for k, x in v["importance_by_family"].items())
            L.append(f"| {name} | {v['n_features']} | {min(bi)}–{int(np.median(bi))}–{max(bi)} | {va:.4f} | {fam} | "
                     + ", ".join(f"{x:.3f}" for x in v["fold_primary"]) + " |")
        top = list(S0["runs"].get("M2", {}).get("importance_top30", {}).items())[:20]
        if top:
            L += ["", "Top-20 gain importance, M2 (mean normalised gain over 15 models): " +
                  "; ".join(f"`{k}` {x:.3f}" for k, x in top) + "."]
        L.append("")
    # runtime
    L += ["### Runtime (wall seconds; Apple M4 10 cores; LightGBM n_jobs = 8)", "",
          "| run | nested feature build (one-off, 5 folds, cached) | load cached features | LF assembly | training "
          "(15 models) | prediction (15 × eval rows) | total |", "|---|---|---|---|---|---|---|"]
    rids = [runs.get(m) for m in order] + list(((cmp or {}).get("runs_alt_seed") or {}).values())
    for rid in rids:
        r = run(rid)
        if r is None:
            continue
        s = stage_runtime(r)
        L.append(f"| {r['model']} | {s['feature_build']:.0f} | {s['load_features']:.0f} | {s['assemble']:.0f} | "
                 f"{s['train']:.0f} | {s['predict']:.0f} | {r['runtime'].get('wall_s', 0):.0f} |")
    fb = j(f"feature_build_{protocol.load()['dev_cv']['split_seed']}.json")
    if fb:
        L += ["", "Nested feature build per fold (1 outer + 5 inner builds, 5 folds in parallel, "
              f"{fb['threads_per_worker']} threads each; cache `{fb['cache_name']}`): " + "; ".join(
                  f"fold {k}: {v['timing']['wall_s']:.0f} s ({v['n_stacker_rows']:,} stacker / {v['n_eval_rows']:,} eval rows; "
                  f"dropped {v['noncoding_dropped']['pairs']:,} non-coding training pairs, "
                  f"{len(v['noncoding_dropped']['genes'])} genes)" for k, v in fb["folds"].items()) + "."]
        allg = sorted({g for v in fb["folds"].values() for g in v["noncoding_dropped"]["genes"]})
        L += ["", f"Non-protein-coding dev genes dropped from training ({len(allg)}): " + ", ".join(f"`{g}`" for g in allg) + "."]
    L.append("")
    # shift
    sh = j(f"feature_shift_{protocol.load()['dev_cv']['split_seed']}.json")
    if sh:
        L += ["### Train-vs-eval feature shift (stacker training rows vs outer eval rows; fold means)", "",
              "| part | feature | stacker mean | eval mean | stacker sd | eval sd | NaN stacker / eval | KS |",
              "|---|---|---|---|---|---|---|---|"]
        for part, d in sh["fold_mean"].items():
            for f, v in d.items():
                if v["train_mean"] is None and v["eval_mean"] is None:
                    continue
                L.append(f"| {part} | `{f}` | {fmt(v['train_mean'], 3)} | {fmt(v['eval_mean'], 3)} | {fmt(v['train_sd'], 3)} | "
                         f"{fmt(v['eval_sd'], 3)} | {fmt(v['train_nan'], 2)} / {fmt(v['eval_nan'], 2)} | {fmt(v['ks'], 3)} |")
        L.append("")
    tune = j("tuning.json")
    if tune:
        L += ["### LightGBM config check (fold 0, early-stopping on inner fold 0; `results/m2/tuning.json`)", "",
              "| config | inner-valid weighted AP | best_iter | wall s |", "|---|---|---|---|"]
        for k, v in tune["configs"].items():
            L.append(f"| {k} | {v['valid_ap_weighted']:.4f} | {v['best_iteration']} | {v['wall_s']:.0f} |")
        L += ["", f"Rule (fixed before the check): {tune['rule']}. Best: `{tune['best']}`; chosen: `{tune['chosen']}`.", ""]
    toy = json.loads((R / "m2_toy_verification.json").read_text()) if (R / "m2_toy_verification.json").exists() else None
    if toy:
        L += ["### Gene-propensity model on the toy data (`scripts/verify_m2_toy.py`)", "",
              "| GP inputs | Spearman(β̂, true β) unseen genes | Spearman(OOF β̂, true β) training genes | "
              "Spearman(B1 β, true β) training genes | OOF r(β̂, B1 β) |", "|---|---|---|---|---|"]
        for v in ("full", "INFB_only", "NOISE_only"):
            m = toy[v]["mean"]
            L.append(f"| {v} | {m['rho_unseen']:.3f} | {m['rho_train_oof']:.3f} | {m['rho_train_b1beta']:.3f} | "
                     f"{m['oof_r_vs_beta']:.3f} |")
        ap = toy["gp_hat_sum_ap"]
        L += ["", f"`gp_hat_sum` alone (full GP): AP C1/C2/C3 = {ap['C1']:.3f}/{ap['C2']:.3f}/{ap['C3']:.3f} "
              f"(prevalence {toy['prevalence']['C3']:.3f}; known-answers propensity-only oracle C3 "
              f"{toy['known_answers_propensity_only_oracle']['C3']:.3f}). Pre-stated checks: " +
              ", ".join(f"{k} = {'PASS' if v else 'FAIL'}" for k, v in toy["checks"].items()) + ".", ""]
    text = DOC.read_text() if DOC.exists() else f"# M2\n\n{BEGIN}\n{END}\n"
    a, b = text.index(BEGIN) + len(BEGIN), text.index(END)
    DOC.write_text(text[:a] + "\n" + "\n".join(L) + "\n" + text[b:])
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
