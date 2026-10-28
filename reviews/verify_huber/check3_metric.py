"""Check 3 (+ parts of 4/5): re-derive primary / per-class AP of the Huber runs and M2 from the saved OOF
predictions with my own AP, class weighting and Poisson(1) gene bootstrap (no src.evaluate). Also:
  - eval label recomputed from the raw S2 column with the protocol threshold (check 4);
  - registered prediction file == mean over the per-seed file, seeds == config seeds (check 5);
  - eval row sets / part labels identical between M2 and the Huber runs (check 4);
  - no lockbox gene in any prediction file.
Prints aggregates only (no pair-level values)."""
import common
import json

import numpy as np
import pandas as pd
import yaml

R = common.REPO
P = yaml.safe_load(open(R / "config" / "protocol.yaml"))
MP = yaml.safe_load(open(R / "config" / "m2plus.yaml"))
CW = P["class_weights"]["unmeasured_universe"]
CW_ALL = P["class_weights"]["all_pairs"]
THR = float(P["label"]["threshold"])
COL = P["label"]["column"]
RUNS = {"M2": "M2__20261028T183713__024416", "M2r": "M2r__20261028T194304__387281",
        "M2r_bag5_huber": "M2r_bag5_huber__20261028T202603__f8bde5", "M2r_huber": "M2r_huber__20261028T205545__7b5e9e",
        "M2r_bag5": "M2r_bag5__20261028T200015__02b9a0",
        "M2_split7": "M2_split7__20261028T185814__ac04a4",
        "M2r_bag5_huber_split7": "M2r_bag5_huber_split7__20261028T204628__38fb1c"}
EXPECT_SEEDS = {"M2r_bag5_huber": MP["seeds"]["bag"], "M2r_huber": MP["seeds"]["ladder"],
                "M2r_bag5_huber_split7": MP["seeds"]["bag"], "M2r_bag5": MP["seeds"]["bag"], "M2r": MP["seeds"]["ladder"]}


def ap(y, s, w):
    """Weighted AP = sum over tie blocks (descending score) of dRecall * precision (sklearn definition)."""
    o = np.argsort(-s, kind="stable")
    y, s, w = y[o], s[o], w[o]
    end = np.r_[s[1:] != s[:-1], True]
    tp = np.cumsum(w * y)[end]
    al = np.cumsum(w)[end]
    return float(np.sum(np.diff(np.r_[0.0, tp]) / tp[-1] * (tp / al)))


class Fold:
    """Fold-level precomputation so bootstrap replicates only recompute weights."""

    def __init__(self, d, genes):
        o = np.argsort(-d.score.to_numpy(), kind="stable")
        s = d.score.to_numpy()[o]
        self.end = np.r_[s[1:] != s[:-1], True]
        self.y = d.y.to_numpy()[o].astype(float)
        self.part = d.part.to_numpy()[o]
        self.ia = genes.get_indexer(d.gene_a.to_numpy()[o])
        self.ib = genes.get_indexer(d.gene_b.to_numpy()[o])
        assert (self.ia >= 0).all() and (self.ib >= 0).all()

    def ap(self, cw, gw=None):
        base = np.ones(len(self.y)) if gw is None else gw[self.ia] * gw[self.ib]
        w = np.zeros(len(self.y))
        for c, v in cw.items():
            m = self.part == c
            if base[m].sum() > 0:
                w[m] = v * base[m] / base[m].sum()
        tp = np.cumsum(w * self.y)[self.end]
        al = np.cumsum(w)[self.end]
        prec = np.divide(tp, al, out=np.zeros_like(tp), where=al > 0)
        return float(np.sum(np.diff(np.r_[0.0, tp]) * prec) / tp[-1])


def main(nboot=300):
    lab = pd.read_parquet(common.LIVE / "splits" / "labels_dev.parquet", columns=["gene_a", "gene_b", COL, "y"])
    out = {"threshold": THR, "label_column": COL,
           "labels_dev_y_equals_S2_lt_thr": bool(((lab[COL] < THR) == lab.y.astype(bool)).all())}
    lab["ymine"] = (lab[COL] < THR).astype(np.int8)
    lab = lab[["gene_a", "gene_b", "ymine"]].rename(columns={"ymine": "y"})
    lock = set(open(R / "config" / "lockbox_genes.txt").read().split())
    reg = pd.read_csv(R / "results" / "registry.tsv", sep="\t", dtype=str).set_index("run_id")
    frames = {}
    for name, rid in RUNS.items():
        p = pd.read_parquet(common.REG_PREDS / f"{rid}.parquet")
        df = p.merge(lab, on=["gene_a", "gene_b"], how="left", validate="many_to_one")
        assert df.y.notna().all()
        frames[name] = df
        genes = pd.Index(sorted(set(df.gene_a) | set(df.gene_b)))
        F = {k: Fold(d, genes) for k, d in df.groupby("fold")}
        fp = [F[k].ap(CW) for k in sorted(F)]
        cls = {c: float(np.mean([ap(d.y.to_numpy()[d.part == c].astype(float), d.score.to_numpy()[d.part == c],
                                    np.ones((d.part == c).sum())) for _, d in df.groupby("fold")]))
               for c in ("C1", "C2", "C3")}
        r = {"rows": len(df), "lockbox_genes_in_preds": len(lock & (set(df.gene_a) | set(df.gene_b))),
             "primary_mine": float(np.mean(fp)), "registry_primary": float(reg.loc[rid, "primary"]),
             "fold_primary": [round(v, 4) for v in fp],
             "class_mine": {c: round(v, 4) for c, v in cls.items()},
             "class_registry": {c: round(float(reg.loc[rid, f"ap_{c}"]), 4) for c in ("C1", "C2", "C3")},
             "all_pairs_mine": float(np.mean([F[k].ap(CW_ALL) for k in F])),
             "registry_all_pairs": float(reg.loc[rid, "pooled_all_pairs"]),
             "registry_seeds": reg.loc[rid, "seeds"], "registry_protocol_sha": reg.loc[rid, "protocol_sha256"][:12],
             "registry_splits_sha": reg.loc[rid, "splits_sha256"][:12], "git_dirty": reg.loc[rid, "git_dirty"],
             "data_root": reg.loc[rid, "data_root"]}
        ps = common.REG_PREDS / "m2plus" / f"{rid}_perseed.parquet"
        if ps.exists():
            q = pd.read_parquet(ps)
            seeds = sorted(int(s) for s in q.seed.unique())
            avg = q.groupby(["gene_a", "gene_b", "fold"]).score.mean().rename("avg").reset_index()
            m = p.merge(avg, on=["gene_a", "gene_b", "fold"], how="outer", indicator=True)
            r["perseed_seeds"] = seeds
            r["perseed_seeds_equal_config"] = seeds == [int(x) for x in EXPECT_SEEDS.get(name, seeds)]
            r["perseed_rows_per_seed"] = sorted(set(q.groupby("seed").size().tolist()))
            r["perseed_mean_matches_registered_maxabs"] = float(np.max(np.abs(m.score - m.avg))) \
                if (m._merge == "both").all() else "ROW MISMATCH"
            # per-seed primary (diagnostic: is any single seed driving it?)
            psd = []
            for s in seeds:
                qs = q[q.seed == s].merge(lab, on=["gene_a", "gene_b"], how="left")
                psd.append(round(float(np.mean([Fold(d, genes).ap(CW) for _, d in qs.groupby("fold")])), 4))
            r["per_seed_primary"] = psd
        out[name] = r
        print(name, json.dumps(r), flush=True)

    key = ["gene_a", "gene_b", "fold", "part"]

    def same_rows(a, b):
        A, B = frames[a][key + ["y"]], frames[b][key + ["y"]]
        m = A.merge(B, on=["gene_a", "gene_b", "fold"], how="outer", indicator=True, suffixes=("", "_b"))
        return {"n_a": len(A), "n_b": len(B), "only_one_side": int((m._merge != "both").sum()),
                "part_mismatch": int((m.part != m.part_b).sum()), "y_mismatch": int((m.y != m.y_b).sum())}
    out["eval_rows_M2_vs_bag5_huber"] = same_rows("M2", "M2r_bag5_huber")
    out["eval_rows_M2_vs_huber"] = same_rows("M2", "M2r_huber")
    out["eval_rows_M2split7_vs_bag5huber_split7"] = same_rows("M2_split7", "M2r_bag5_huber_split7")

    rng = np.random.default_rng(424242)        # deliberately not the protocol bootstrap seed
    res = {}
    for a, b in (("M2r_bag5_huber", "M2"), ("M2r_huber", "M2r"), ("M2r_bag5_huber", "M2r_huber")):
        A = frames[a]
        B = frames[b].set_index(["gene_a", "gene_b", "fold"]).loc[pd.MultiIndex.from_frame(A[["gene_a", "gene_b", "fold"]])].reset_index()
        genes = pd.Index(sorted(set(A.gene_a) | set(A.gene_b)))
        FA = {k: Fold(d, genes) for k, d in A.groupby("fold")}
        FB = {k: Fold(d, genes) for k, d in B.groupby("fold")}
        pt = np.mean([FA[k].ap(CW) for k in FA]) - np.mean([FB[k].ap(CW) for k in FB])
        d = []
        for _ in range(nboot):
            gw = rng.poisson(1.0, len(genes)).astype(float)
            d.append(np.mean([FA[k].ap(CW, gw) for k in FA]) - np.mean([FB[k].ap(CW, gw) for k in FB]))
        d = np.array(d)
        res[f"{a}-{b}"] = {"delta": round(float(pt), 5), "n_boot": nboot,
                           "ci95": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)],
                           "frac_le_0": float((d <= 0).mean()), "frac_lt_0.005": float((d < 0.005).mean())}
        print(f"{a}-{b}", res[f"{a}-{b}"], flush=True)
    out["paired"] = res
    out["seed7_delta_bag5huber_minus_M2"] = out["M2r_bag5_huber_split7"]["primary_mine"] - out["M2_split7"]["primary_mine"]
    print(json.dumps(out, indent=1, default=str))
    (common.OUT / "check3_metric.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
