#!/usr/bin/env python
"""Known-answer check of src/ryanfeat.py on the toy data (data/toy, written by scripts/toy/make_toy.py).

Copies data/toy to a temporary data root (the toy directory is never modified), builds the toy splits
with scripts/make_splits.py, computes outer-fold Ryan features for the 5 dev folds and scores single
features through src/evaluate.py (registry in the temporary results root). Pre-stated expectations
(data/toy/known_answers.json):
  * B1 (b1_logit) C1 AUPRC approaches the propensity-only oracle C1 (>= half of it here; the known-answers
    oracle is computed on a different pair set, so the tight same-pair comparison -- B1 vs the TRUE
    beta_a + beta_b on exactly these CV C1 rows, and the permuted-label toy -- lives in
    tests/test_ryanfeat.py::test_toy_*, which may read data/toy/_truth);
  * NBR features with the informative similarity (MODSIM) beat the noise similarity (NOISESIM) on C2 and
    C3, and NOISESIM features stay near prevalence.
  * no class AUPRC exceeds the oracle hard ceiling (leak flag).
Writes results/ryanfeat_toy_verification.json (aggregate numbers only).
Usage: .venv/bin/python scripts/verify_ryanfeat_toy.py [--keep DIR]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES = ["b1_logit", "mf_score", "prof_corr",
            "nbr_MODSIM__hit_block", "nbr_MODSIM__hit_dmax", "nbr_MODSIM__generate_max",
            "nbr_NOISESIM__hit_block", "nbr_NOISESIM__hit_dmax", "nbr_NOISESIM__generate_max"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="use/keep this directory instead of a temp dir")
    a = ap.parse_args()
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="ryanfeat_toy_"))
    data_root, res_root = tmp / "data", tmp / "results"
    if not (data_root / "splits").exists():
        shutil.copytree(ROOT / "data" / "toy", data_root, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("_truth", "splits", "cache", "preds"))
        env = {**os.environ, "PSL_DATA_ROOT": str(data_root), "PSL_RESULTS_ROOT": str(res_root)}
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_splits.py")], env=env, check=True,
                       capture_output=True)
    os.environ["PSL_DATA_ROOT"], os.environ["PSL_RESULTS_ROOT"] = str(data_root), str(res_root)

    from src import data, protocol
    from src.evaluate import evaluate
    from src.ryanfeat import SimStore, build_ryan_features
    from src.timing import Timings, set_threads
    set_threads(8)
    ka = json.loads((ROOT / "data" / "toy" / "known_answers.json").read_text())
    sims = SimStore(data_root)
    T = Timings()
    rows = []
    with T.stage("features"):
        for k in range(protocol.load()["dev_cv"]["n_folds"]):
            tr, ev = data.fold_split(None, k)
            F = build_ryan_features(tr, ev, sims, {})   # module defaults (= values tuned on real fold 0)
            rows.append(pd.concat([ev[["gene_a", "gene_b", "fold", "part"]], F[FEATURES]], axis=1))
    E = pd.concat(rows, ignore_index=True)
    out = {"sims": sims.names, "known_answers": {
        "propensity_only_oracle": ka["auprc"]["propensity_only_oracle"],
        "prevalence": ka["auprc"]["prevalence_baseline"],
        "hard_ceiling": ka["recommended_bands_pre_stated"]["hard_ceiling_leak_flag_oracle_p99"]},
        "features": {}}
    for f in FEATURES:
        s = E[f].to_numpy(float)
        s = np.where(np.isfinite(s), s, np.nanmin(s) - 1 if np.isfinite(s).any() else 0.0)  # NaN -> bottom
        res = evaluate(E[["gene_a", "gene_b", "fold", "part"]].assign(score=s, seed=0), model=f"TOY_{f}",
                       runtime=T.as_dict(), n_boot=200, notes="toy verification of src/ryanfeat.py")
        m = res["metrics_mean"]
        out["features"][f] = {"primary": res["primary"], **{c: m[f"ap_{c}"] for c in ("C1", "C2", "C3")},
                              "prevalence": {c: m[f"pos_{c}"] / m[f"n_{c}"] for c in ("C1", "C2", "C3")}}
        print(f"{f:32s} primary {res['primary']:.4f}  C1 {m['ap_C1']:.4f}  C2 {m['ap_C2']:.4f}  "
              f"C3 {m['ap_C3']:.4f}")
    fe = out["features"]
    orc = out["known_answers"]["propensity_only_oracle"]
    hc = out["known_answers"]["hard_ceiling"]
    checks = {
        "b1_C1_ge_half_propensity_oracle": fe["b1_logit"]["C1"] >= 0.5 * orc["C1"],
        "modsim_beats_noisesim_C2": fe["nbr_MODSIM__hit_block"]["C2"] > fe["nbr_NOISESIM__hit_block"]["C2"],
        "modsim_beats_noisesim_C3": fe["nbr_MODSIM__hit_block"]["C3"] > fe["nbr_NOISESIM__hit_block"]["C3"],
        "noisesim_near_prevalence_C3": fe["nbr_NOISESIM__hit_block"]["C3"]
        < 1.5 * fe["nbr_NOISESIM__hit_block"]["prevalence"]["C3"] + 0.01,
        "no_class_above_hard_ceiling": all(v[c] <= hc[c] for v in fe.values() for c in ("C1", "C2", "C3")),
    }
    out["checks"] = checks
    out["runtime"] = T.as_dict()
    print(json.dumps(checks, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ryanfeat_toy_verification.json").write_text(json.dumps(out, indent=1, default=float))
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
