#!/usr/bin/env python
"""Known-answer check of the M2 gene-propensity (GP) model on the toy data (data/toy; latent truth in
data/toy/_truth, read ONLY here and in tests, never by pipeline code).

For each toy dev fold, src/models/m2.M2Builder is run on the fold's outer training pairs (drop_noncoding
applied) and the gene-level beta_hat is compared with the TRUE latent propensity `beta`:
  * unseen genes (the fold's held-out genes): Spearman(beta_hat, beta_true)
  * training genes: Spearman of the out-of-fold beta_hat and of the fitted B1 beta with beta_true
  * AUPRC of the pair feature gp_hat_sum on the C3 eval rows (labels joined only through src/evaluate.py)
Controls: (a) GP inputs = NOISE gene table only, no NBR inputs -> beta_hat must be uninformative;
          (b) GP inputs = INFB (planted noisy propensity) only -> informative.
Pre-stated pass criteria (fold mean): full GP unseen-gene Spearman >= 0.6; INFB-only >= 0.6;
NOISE-only |Spearman| < 0.2; full GP C3 AP of gp_hat_sum >= 3 x C3 prevalence.
Writes <results>/m2_toy_verification.json (aggregate numbers only).
Usage: .venv/bin/python scripts/verify_m2_toy.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "data" / "toy"
os.environ["PSL_DATA_ROOT"] = str(TOY)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import data, protocol  # noqa: E402
from src.models import m2  # noqa: E402
from src.ryanfeat import SimStore  # noqa: E402
from src.timing import set_threads  # noqa: E402


def main() -> int:
    set_threads(8)
    if not (TOY / "splits" / "dev_folds_seed20261028.tsv").exists():
        raise SystemExit("toy splits missing: PSL_DATA_ROOT=data/toy .venv/bin/python scripts/make_splits.py")
    truth = pd.read_csv(TOY / "_truth" / "gene_truth.tsv", sep="\t").set_index("gene").beta
    sims = SimStore()
    gt = m2.gene_table()
    variants = {"full": (gt, sims, {}),
                "INFB_only": (gt[[c for c in gt.columns if c.startswith("INFB")]], None, {"nbr_gene": False}),
                "NOISE_only": (gt[[c for c in gt.columns if c.startswith("NOISE")]], None, {"nbr_gene": False})}
    res = {v: {"folds": []} for v in variants}
    rows = []
    n = protocol.load()["dev_cv"]["n_folds"]
    for k in range(n):
        tr, ev, drop = m2.fold_pairs(None, k)
        held = set(data.fold_genes(None, k))
        for v, (tab, sm, gp) in variants.items():
            B = m2.make_builder(tab, gp)
            cfg = {"mf": False, "profile": False, "sims": None if sm is not None else []}
            b = B(tr, ev[["gene_a", "gene_b"]], sm, cfg)
            F = b.build()
            bh = b.info["gp"]["beta_hat"]
            beta = b.info["b1"]["beta"]
            tg = [g for g in beta.index if g in truth.index]
            ug = [g for g in bh.index if g in held and g in truth.index]
            r = {"fold": k, "n_unseen": len(ug), "n_train": len(tg),
                 "rho_unseen": float(spearmanr(bh[ug], truth[ug]).correlation),
                 "rho_train_oof": float(spearmanr(bh[tg], truth[tg]).correlation),
                 "rho_train_b1beta": float(spearmanr(beta[tg], truth[tg]).correlation),
                 "oof_r_vs_beta": b.info["gp"]["oof_r"]}
            res[v]["folds"].append(r)
            if v == "full":
                rows.append(ev[["gene_a", "gene_b", "fold", "part"]].assign(score=F.gp_hat_sum.to_numpy(), seed=0))
            print(v, r, flush=True)
    for v in variants:
        res[v]["mean"] = {key: float(np.mean([f[key] for f in res[v]["folds"]]))
                          for key in ("rho_unseen", "rho_train_oof", "rho_train_b1beta", "oof_r_vs_beta")}
    from src.evaluate import evaluate
    ev_res = evaluate(pd.concat(rows, ignore_index=True), model="TOY_gp_hat_sum", register=False, n_boot=200)
    m = ev_res["metrics_mean"]
    prev = {c: m[f"pos_{c}"] / m[f"n_{c}"] for c in ("C1", "C2", "C3")}
    res["gp_hat_sum_ap"] = {c: m[f"ap_{c}"] for c in ("C1", "C2", "C3")}
    res["prevalence"] = prev
    res["known_answers_propensity_only_oracle"] = json.loads((TOY / "known_answers.json").read_text())["auprc"][
        "propensity_only_oracle"]
    res["checks"] = {
        "full_unseen_rho_ge_0.6": res["full"]["mean"]["rho_unseen"] >= 0.6,
        "infb_unseen_rho_ge_0.6": res["INFB_only"]["mean"]["rho_unseen"] >= 0.6,
        "noise_unseen_abs_rho_lt_0.2": abs(res["NOISE_only"]["mean"]["rho_unseen"]) < 0.2,
        "full_C3_ap_ge_3x_prevalence": res["gp_hat_sum_ap"]["C3"] >= 3 * prev["C3"]}
    out = protocol.results_root() / "m2_toy_verification.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({v: res[v]["mean"] for v in variants}, indent=1))
    print("gp_hat_sum AP", res["gp_hat_sum_ap"], "prev", prev, "oracle", res["known_answers_propensity_only_oracle"])
    print("checks", res["checks"])
    return 0 if all(res["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
