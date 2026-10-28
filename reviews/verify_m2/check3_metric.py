"""Check 3: re-derive the M2 / M1 primary metric and the paired Δ from the saved OOF predictions with
independent AP + class-weighting + Poisson gene-bootstrap code (no src.evaluate). Prints aggregates only."""
import common  # noqa: F401  (env setup)
import json
import sys

import numpy as np
import pandas as pd
import yaml

R = common.REPO
P = yaml.safe_load(open(R / "config" / "protocol.yaml"))
CW = P["class_weights"]["unmeasured_universe"]
CW_ALL = P["class_weights"]["all_pairs"]
THR = P["label"]["threshold"]
RUNS = {"M2": "M2__20261028T183713__024416", "M1": "M1__20261028T185030__5ad666"}


def my_ap(y, s, w):
    """Weighted AP, step-wise: sum over distinct-score thresholds of (ΔRecall * precision)."""
    o = np.argsort(-s, kind="stable")
    y, s, w = y[o], s[o], w[o]
    last = np.r_[s[1:] != s[:-1], True]          # last row of each tie block
    tp = np.cumsum(w * y)[last]
    tot = np.cumsum(w)[last]
    prec = tp / tot
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def weights(part, cw):
    w = np.zeros(len(part))
    for c, v in cw.items():
        m = part == c
        w[m] = v / m.sum()
    return w


def primary(df, cw, gw=None):
    vals = []
    for k, d in df.groupby("fold"):
        part = d.part.to_numpy()
        base = np.ones(len(d)) if gw is None else gw[d.ia.to_numpy()] * gw[d.ib.to_numpy()]
        w = np.zeros(len(d))
        for c, v in cw.items():
            m = part == c
            if base[m].sum() > 0:
                w[m] = v * base[m] / base[m].sum()
        keep = w > 0
        vals.append(my_ap(d.y.to_numpy()[keep], d.score.to_numpy()[keep], w[keep]))
    return float(np.mean(vals)), vals


def main(nboot=200):
    lab = pd.read_parquet(common.FROZEN / "splits" / "labels_dev.parquet", columns=["gene_a", "gene_b", "s_score"])
    lab["y"] = (lab.s_score < THR).astype(np.int8)        # my own label from the raw S2 column
    lab = lab.drop(columns="s_score")
    reg = pd.read_csv(R / "results" / "registry.tsv", sep="\t", dtype=str).set_index("run_id")
    frames, out = {}, {}
    for name, rid in RUNS.items():
        p = pd.read_parquet(common.FROZEN / "preds" / f"{rid}.parquet")
        df = p.merge(lab, on=["gene_a", "gene_b"], how="left", validate="many_to_one")
        assert df.y.notna().all()
        frames[name] = df
        pm, per = primary(df, CW)
        cls = {c: my_ap(df.y.to_numpy()[df.part == c], df.score.to_numpy()[df.part == c],
                        np.ones((df.part == c).sum())) for c in ("C1", "C2", "C3")}
        # per-class AP per fold then averaged (as in evaluate 'ap_Cx' = mean over folds)
        clsf = {c: float(np.mean([my_ap(d.y.to_numpy()[d.part == c], d.score.to_numpy()[d.part == c],
                                        np.ones((d.part == c).sum())) for _, d in df.groupby("fold")]))
                for c in ("C1", "C2", "C3")}
        prev = float(np.mean([np.sum(weights(d.part.to_numpy(), CW) * d.y) for _, d in df.groupby("fold")]))
        out[name] = {"rows": len(df), "primary_mine": round(pm, 5), "fold_primary": [round(v, 4) for v in per],
                     "registry_primary": float(reg.loc[rid, "primary"]),
                     "classAP_fold_mean": {k: round(v, 4) for k, v in clsf.items()},
                     "registry_class": {c: round(float(reg.loc[rid, f"ap_{c}"]), 4) for c in ("C1", "C2", "C3")},
                     "all_pairs_mine": round(primary(df, CW_ALL)[0], 5),
                     "weighted_prevalence": round(prev, 5)}
    a, b = frames["M2"], frames["M1"]
    b = b.set_index(["gene_a", "gene_b", "fold"]).loc[pd.MultiIndex.from_frame(a[["gene_a", "gene_b", "fold"]])].reset_index()
    genes = pd.Index(sorted(set(a.gene_a) | set(a.gene_b)))
    for d in (a, b):
        d["ia"], d["ib"] = genes.get_indexer(d.gene_a), genes.get_indexer(d.gene_b)
    rng = np.random.default_rng(12345)                      # deliberately NOT the protocol seed
    deltas = []
    for r in range(nboot):
        gw = rng.poisson(1.0, len(genes)).astype(float)
        deltas.append(primary(a, CW, gw)[0] - primary(b, CW, gw)[0])
    deltas = np.array(deltas)
    out["delta_point"] = round(out["M2"]["primary_mine"] - out["M1"]["primary_mine"], 5)
    out["delta_boot"] = {"n": nboot, "mean": round(float(deltas.mean()), 4),
                         "ci95": [round(float(np.quantile(deltas, .025)), 4), round(float(np.quantile(deltas, .975)), 4)],
                         "frac_le_0": float((deltas <= 0).mean())}
    print(json.dumps(out, indent=1))
    (common.OUT / "check3_metric.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
