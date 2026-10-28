#!/usr/bin/env python
"""M1: LightGBM on label-free symmetric pair features (DESIGN.md §5), evaluated through src/evaluate.py.

Per outer fold k (split seed from protocol.yaml unless --split-seed):
  training rows = data.fold_split(k) training pairs (minus pairs with a gene outside genes.txt: pseudogenes
  etc., DESIGN §1), uniform weights; early stopping on a gene-grouped inner validation split of those
  training pairs (15% of training genes held out, src/models/lgbm.gene_holdout) — never the reported fold.
  One model per seed (protocol seeds.ladder by default); evaluate() averages seeds.
Modes:
  --tune                 <= 6 configs (lgbm.CONFIGS) on the fold-0 inner split only -> <results>/m1/tuning.json
  (default)              full M1 run, registered; symmetry assertion on predictions; gain importance
  --ablate               + one run per feature group dropped (same config), compare(full, ablated)
  --permute              + gene-permutation negative control (dev-gene identities permuted in the store map)
Usage: run_m1.py [--params default|tuned|<config>] [--seeds ladder|all] [--ablate] [--permute] [--save-models]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, protocol  # noqa: E402
from src import pairs as PA  # noqa: E402
from src.evaluate import compare, evaluate  # noqa: E402
from src.models import lgbm  # noqa: E402
from src.timing import Timer, Timings, peak_rss_mb, set_threads  # noqa: E402

N_JOBS = 8
SYM_SAMPLE = 20_000


def out_dir() -> Path:
    d = protocol.results_root() / "m1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_real_root() -> bool:
    return protocol.data_root().resolve() == (ROOT / protocol.load()["paths"]["data_root_default"]).resolve()


class Folds:
    """Per-fold training/eval pairs and assembled features (full feature set), cached in memory."""

    def __init__(self, spec: PA.FeatureSpec, split_seed, T: Timings | None = None):
        self.spec, self.split_seed, self.T = spec, split_seed, T
        self.n_folds = protocol.load()["dev_cv"]["n_folds"]
        self.cache: dict = {}
        self.dropped = {}

    def pairs(self, k):
        tr, ev = data.fold_split(self.split_seed, k)
        g = set(self.spec.genes)
        keep = tr.gene_a.isin(g) & tr.gene_b.isin(g)
        self.dropped[k] = {"pairs": int((~keep).sum()), "positives": int(tr.y[~keep].sum())}
        assert ev.gene_a.isin(g).all() and ev.gene_b.isin(g).all(), "eval gene outside genes.txt"
        return tr[keep].reset_index(drop=True), ev

    def get(self, k, perm=None):
        key = (k, None if perm is None else "perm")
        if key not in self.cache:
            tr, ev = self.pairs(k)
            ctx = self.T.stage(f"assemble_f{k}") if self.T else Timer("assemble")
            with ctx:
                Xtr = PA.assemble(tr, spec=self.spec, perm=perm).to_numpy()
                Xev = PA.assemble(ev, spec=self.spec, perm=perm).to_numpy()
            self.cache[key] = (tr, tr.y.to_numpy(np.int8), Xtr, ev, Xev)
        return self.cache[key]


def holdout_seed(k, s):
    return [protocol.load()["inner"]["seed"], int(k), int(s)]


def run_model(F: Folds, model: str, cols: np.ndarray, seeds, params: dict, T: Timings, perm=None,
              check_symmetry=True, save_models: Path | None = None) -> tuple[pd.DataFrame, dict]:
    names = np.array(F.spec.feature_names())[cols]
    rows, info, imp = [], [], []
    full = len(cols) == len(F.spec.feature_names())
    for k in range(F.n_folds):
        tr, y, Xtr, ev, Xev = F.get(k, perm)
        if not full:
            with T.stage(f"select_cols_f{k}"):
                Xtr, Xev = Xtr[:, cols], Xev[:, cols]
        boosters = []
        for s in seeds:
            fit_m, val_m = lgbm.gene_holdout(tr, lgbm.VALID_GENE_FRAC, holdout_seed(k, s))
            with T.stage(f"train_f{k}_s{s}"):
                b, inf = lgbm.fit(Xtr, y, fit_m, val_m, params, seed=int(s), feature_names=names)
            with T.stage(f"predict_f{k}_s{s}"):
                sc = lgbm.predict(b, Xev, N_JOBS)
            info.append({"fold": k, "seed": int(s), **inf})
            imp.append(lgbm.gain_importance(b))
            rows.append(ev[["gene_a", "gene_b", "fold", "part"]].assign(score=sc, seed=int(s)))
            boosters.append((s, b, sc))
            if save_models is not None:
                save_models.mkdir(parents=True, exist_ok=True)
                b.save_model(str(save_models / f"fold{k}_seed{s}.txt"), num_iteration=b.best_iteration)
            print(f"  [{model}] fold {k} seed {s}: best_iter {inf['best_iteration']} valid AP {inf['valid_ap']:.4f} "
                  f"({inf['n_fit']} fit / {inf['n_valid']} valid rows)", flush=True)
        if check_symmetry:
            sw = ev.rename(columns={"gene_a": "gene_b", "gene_b": "gene_a"})
            Xsw = PA.assemble(sw, spec=F.spec, perm=perm).to_numpy()[:, cols]
            assert np.array_equal(Xsw, Xev, equal_nan=True), f"fold {k}: features not symmetric"
            idx = np.random.default_rng(k).choice(len(ev), min(SYM_SAMPLE, len(ev)), replace=False)
            for s, b, sc in boosters:
                assert np.array_equal(lgbm.predict(b, Xsw[idx], N_JOBS), sc[idx]), f"fold {k} seed {s}: score(a,b) != score(b,a)"
    preds = pd.concat(rows, ignore_index=True)
    gi = np.mean(imp, axis=0)
    importance = pd.Series(gi, index=names).sort_values(ascending=False)
    return preds, {"fits": info, "importance": importance}


def register(model, preds, F: Folds, cols, T: Timings, notes, args, extra: dict) -> dict:
    groups = sorted({F.spec.feature_groups()[n] for n in np.array(F.spec.feature_names())[cols]})
    excl = [g for g in F.spec.groups() if g not in groups]
    res = evaluate(preds, model=model, split_seed=args.split_seed, runtime=T.as_dict(),
                   feature_hashes=F.spec.hashes(exclude=excl), notes=notes, register=not args.no_register)
    m, b = res["metrics_mean"], res["bootstrap"]["primary"]
    print(f"== {model}: run_id {res['run_id']} primary {res['primary']:.4f} (SE {b['se']:.4f}) "
          f"C1 {m['ap_C1']:.4f} C2 {m['ap_C2']:.4f} C3 {m['ap_C3']:.4f} prev {m['prevalence_primary']:.4f} "
          f"wall {T.as_dict()['wall_s']:.0f}s peak {T.as_dict()['peak_rss_mb']:.0f} MB", flush=True)
    summary = {"run_id": res["run_id"], "model": model, "primary": res["primary"], "se": b["se"], "ci": b["ci"],
               "metrics_mean": {k: m[k] for k in ("ap_C1", "ap_C2", "ap_C3", "prevalence_primary", "lift", "auroc",
                                                  "pooled_all_pairs", "pooled_equal", "mean_per_class",
                                                  "precision_at_k") if k in m},
               "groups": groups, "n_features": int(len(cols)), "runtime": T.as_dict(), **extra}
    if not args.no_register:
        (out_dir() / f"{res['run_id']}.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def tune(F: Folds, args) -> None:
    tr, y, Xtr, ev, Xev = F.get(0)
    fit_m, val_m = lgbm.gene_holdout(tr, lgbm.VALID_GENE_FRAC, holdout_seed(0, 0))
    out = {"fold": 0, "holdout_seed": holdout_seed(0, 0), "configs": {}}
    for name in lgbm.CONFIGS:
        with Timer(name) as t:
            _, inf = lgbm.fit(Xtr, y, fit_m, val_m, lgbm.params_for(name, N_JOBS), seed=0,
                              feature_names=F.spec.feature_names())
        out["configs"][name] = {**inf, "wall_s": t.result["wall_s"], "params": lgbm.params_for(name, N_JOBS)}
        print(f"  {name}: valid AP {inf['valid_ap']:.4f} best_iter {inf['best_iteration']} {t.result['wall_s']:.0f}s",
              flush=True)
    best = max(out["configs"], key=lambda n: out["configs"][n]["valid_ap"])
    out["chosen"] = best
    out["rule"] = "max inner-validation AP on fold 0 (gene-grouped 15% hold-out of fold-0 training genes)"
    (out_dir() / "tuning.json").write_text(json.dumps(out, indent=1))
    print(f"chosen config: {best}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="default", help="default | tuned | a key of lgbm.CONFIGS")
    ap.add_argument("--seeds", default="ladder", choices=["ladder", "all"])
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--permute-seed", type=int, default=0)
    ap.add_argument("--permute-seeds", default="first", choices=["first", "ladder"],
                    help="model seeds for the permutation control")
    ap.add_argument("--ablate-seeds", default="first", choices=["first", "ladder"],
                    help="model seeds for ablations; 'first' also registers a matching seed-0 full reference")
    ap.add_argument("--skip-full", action="store_true", help="skip the headline run (ablation/permutation only)")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--tag", default="M1")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()
    set_threads(N_JOBS)
    P = protocol.load()
    spec = PA.load_spec()
    viol = PA.source_violations(spec)
    if viol and is_real_root():
        raise SystemExit("feature sources not on the use=feature allowlist:\n" + "\n".join(viol))
    if viol:
        print(f"note: non-allowlisted sources on a non-default data root (toy): {sorted(set(v.split(': ')[1] for v in viol))}")
    print(f"data root {protocol.data_root()}; groups {spec.groups()}; {len(spec.feature_names())} features")

    T = Timings()
    F = Folds(spec, args.split_seed, T)
    if args.tune:
        tune(F, args)
        return 0

    if args.params == "tuned":
        tj = json.loads((out_dir() / "tuning.json").read_text())
        cfg = tj["chosen"]
    else:
        cfg = args.params
    params = lgbm.params_for(cfg, N_JOBS)
    seeds = P["seeds"][args.seeds]
    names = spec.feature_names()
    gmap = spec.feature_groups()
    cols = np.array([i for i, n in enumerate(names) if gmap[n] not in set(args.exclude)])
    model = args.tag + ("" if not args.exclude else "_minus_" + "+".join(args.exclude))
    t0 = time.perf_counter()
    save = (protocol.data_root() / "models" / "m1" / f"{model}_{cfg}") if args.save_models else None
    report = {"full": None, "ablation_reference": None, "ablations": {}, "permutation": None}

    def full_run(sd, name, T_):
        preds, info = run_model(F, name, cols, sd, params, T_, save_models=save)
        extra = {"config": cfg, "params": params, "seeds": sd, "fits": info["fits"],
                 "importance_top30": info["importance"].head(30).to_dict(),
                 "importance_by_group": info["importance"].groupby(info["importance"].index.map(gmap)).sum()
                 .sort_values(ascending=False).to_dict(),
                 "dropped_training_pairs_not_in_genes_txt": F.dropped, "symmetry_checked": True,
                 "models_dir": str(save) if save else None}
        r = register(name, preds, F, cols, T_, f"M1 label-free LightGBM, config={cfg}, seeds={sd}", args, extra)
        print("top-15 gain importance:", {k: round(v, 4) for k, v in list(r["importance_top30"].items())[:15]})
        print("importance by group:", {k: round(v, 3) for k, v in r["importance_by_group"].items()})
        return r

    if not args.skip_full:
        report["full"] = full_run(seeds, model, T)

    if args.ablate:
        aseeds = seeds[:1] if args.ablate_seeds == "first" else seeds
        if report["full"] is not None and aseeds == seeds:
            ref = report["full"]
        else:
            save = None
            ref = full_run(aseeds, f"{model}_s{'+'.join(map(str, aseeds))}", Timings())
        report["ablation_reference"] = ref
        for g in spec.groups():
            if g in args.exclude:
                continue
            Ta = Timings()
            c = np.array([i for i, n in enumerate(names) if gmap[n] not in set(args.exclude) | {g}])
            print(f"-- ablation: drop {g} ({len(cols) - len(c)} features), seeds {aseeds}", flush=True)
            nm = f"{model}_minus_{g}" + ("" if aseeds == seeds else f"_s{'+'.join(map(str, aseeds))}")
            p, inf = run_model(F, nm, c, aseeds, params, Ta, check_symmetry=False)
            s = register(nm, p, F, c, Ta, f"M1 ablation drop {g}, config={cfg}, seeds={aseeds}; "
                         f"reference {ref['run_id']}", args, {"config": cfg, "dropped_group": g, "fits": inf["fits"]})
            cmp = compare(ref["run_id"], s["run_id"], register=not args.no_register) if not args.no_register else {}
            report["ablations"][g] = {**s, "compare_ref_minus_ablated": cmp}
            if cmp:
                print(f"   delta(ref - minus_{g}) = {cmp['delta']:+.4f} CI [{cmp['ci_lo']:+.4f}, {cmp['ci_hi']:+.4f}]",
                      flush=True)

    if args.permute:
        dev = data.dev_folds(args.split_seed).index
        gi = PA.gene_indices(spec, [g for g in dev if g in set(spec.genes)])
        rng = np.random.default_rng(args.permute_seed)
        perm = np.arange(spec.G)
        perm[gi] = gi[rng.permutation(len(gi))]
        F.cache.clear()   # free the unpermuted feature cache
        Tp = Timings()
        Fp = Folds(spec, args.split_seed, Tp)
        ps = seeds[:1] if args.permute_seeds == "first" else seeds
        p, inf = run_model(Fp, f"{model}_genepermuted", cols, ps, params, Tp, perm=perm)
        s = register(f"{model}_genepermuted", p, Fp, cols, Tp,
                     f"NEGATIVE CONTROL: dev-gene identities permuted in the feature-store map (seed {args.permute_seed}), "
                     f"model seeds {ps}", args, {"config": cfg, "permute_seed": args.permute_seed, "fits": inf["fits"],
                                                 "n_permuted_genes": int(len(gi))})
        report["permutation"] = s

    report["total_wall_s"] = round(time.perf_counter() - t0, 1)
    report["peak_rss_mb"] = peak_rss_mb()
    rid = (report["full"] or report["ablation_reference"] or report["permutation"] or {}).get("run_id", "none")
    if not args.no_register:
        (out_dir() / f"report_{rid}.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"total wall {report['total_wall_s']} s; peak RSS {report['peak_rss_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
