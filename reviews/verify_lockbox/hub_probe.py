"""Hub-stratum probe (verifier): share of positives from hub-stratum genes and per-class AP with vs without
hub-gene pairs, lockbox (M2r_huber, B1) vs dev OOF (same models' registered dev runs; fold-mean per class).
Hub stratum = top 2% S2 genes by hit rate (scripts/draw_lockbox.py rule). Aggregates only."""
import json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.metrics import average_precision_score as aps

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
THR = yaml.safe_load(open(ROOT / "config/protocol.yaml"))["label"]["threshold"]
R = json.load(open(ROOT / "results/lockbox/lockbox_result_deec74e31a7c.json"))
LB = set((ROOT / "config/lockbox_genes.txt").read_text().split())
pairs = pd.read_parquet(ROOT / "data/processed/ryan_pairs.parquet", columns=["gene_a", "gene_b", "s_score", "in_s2"])
pairs = pairs[pairs.in_s2]
long = pd.concat([pairs[["gene_a", "s_score"]].rename(columns={"gene_a": "g"}), pairs[["gene_b", "s_score"]].rename(columns={"gene_b": "g"})])
hr = (long.s_score < THR).groupby(long.g).mean()
genes = pd.read_csv(ROOT / "data/processed/ryan_genes.tsv", sep="\t").dropna(subset=["pombase_id"])
hr = hr[hr.index.isin(set(genes.pombase_id))]
HUB = set(hr.index[hr >= hr.quantile(0.98)])
key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a)
L = pd.read_parquet(ROOT / "data/processed/splits/labels_lockbox.parquet", columns=["gene_a", "gene_b", "s_score"])
D = pd.read_parquet(ROOT / "data/processed/splits/labels_dev.parquet", columns=["gene_a", "gene_b", "s_score"])
lab = pd.concat([L, D]); lab = pd.Series(lab.s_score.values < THR, index=key(lab.gene_a.values, lab.gene_b.values))
out = {"n_hub": len(HUB), "n_hub_lockbox": len(HUB & LB)}

def per_class(df, by_fold):
    df = df.assign(hub=df.gene_a.isin(HUB) | df.gene_b.isin(HUB))
    r = {}
    for c in ("C1", "C2", "C3"):
        d = df[df.part == c]
        grp = list(d.groupby("fold")) if by_fold else [(0, d)]
        f = lambda x: float(np.mean([aps(g.y, g.score) for _, g in x if g.y.any()]))
        r[c] = {"ap_all": f(grp), "ap_nohub": f([(k, g[~g.hub]) for k, g in grp]),
                "hub_pair_frac": float(d.hub.mean()), "hub_pos_share": float(d.y[d.hub].sum() / d.y.sum())}
    return r
for m in ("M2r_huber", "M1", "B1_additive_logit"):
    p = pd.read_parquet(ROOT / f"data/processed/lockbox/preds_{m}.parquet")
    p["y"] = lab.loc[key(p.gene_a.values, p.gene_b.values)].values
    q = pd.read_parquet(ROOT / f"data/processed/preds/{R['models'][m]['dev_run']}.parquet")
    q = q.groupby(["gene_a", "gene_b", "fold", "part"], as_index=False).score.mean()   # mean over seeds
    q["y"] = lab.loc[key(q.gene_a.values, q.gene_b.values)].values
    out[m] = {"lockbox": per_class(p, False), "dev": per_class(q, True)}
    for c in ("C1", "C2", "C3"):
        a, b = out[m]["dev"][c], out[m]["lockbox"][c]
        print(m, c, "dev all/nohub %.4f/%.4f  lb all/nohub %.4f/%.4f  hub pos share dev %.3f lb %.3f  hub pair frac dev %.3f lb %.3f" % (
            a["ap_all"], a["ap_nohub"], b["ap_all"], b["ap_nohub"], a["hub_pos_share"], b["hub_pos_share"], a["hub_pair_frac"], b["hub_pair_frac"]))
json.dump(out, open(OUT / "hub_probe_result.json", "w"), indent=1)
