"""Draw the lockbox (held-out test) gene set ONCE, before any further analysis.

Design: DESIGN.md §2. 20% of Ryan S2 genes, seed 20261028, stratified by Ryan role
(query / array / both) x within-role hit-rate quintile, with the top 2% of genes by
hit rate as their own stratum. Hit rates use all S2 pairs and serve only for stratification.

Output: config/lockbox_genes.txt (one PomBase ID per line) + sha256 printed.
Refuses to overwrite an existing lockbox file.
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "lockbox_genes.txt"
SEED, FRAC, HUB_Q, THR = 20261028, 0.20, 0.98, -2.3   # moved to config/protocol.yaml at Phase A

if OUT.exists():
    sys.exit(f"{OUT} exists; the lockbox is drawn once. Delete only with a decision-log entry.")

pairs = pd.read_parquet(ROOT / "data/processed/ryan_pairs.parquet",
                        columns=["gene_a", "gene_b", "s_score", "in_s2"])
pairs = pairs[pairs.in_s2]
genes = pd.read_csv(ROOT / "data/processed/ryan_genes.tsv", sep="\t")
genes = genes.dropna(subset=["pombase_id"])
role = genes.groupby("pombase_id").agg(q=("is_query", "any"), a=("is_array", "any"))
role["role"] = np.where(role.q & role.a, "both", np.where(role.q, "query", "array"))

long = pd.concat([pairs[["gene_a", "s_score"]].rename(columns={"gene_a": "g"}),
                  pairs[["gene_b", "s_score"]].rename(columns={"gene_b": "g"})])
long["y"] = long.s_score < THR
hr = long.groupby("g").y.mean().rename("hit_rate")
df = role.join(hr, how="inner")            # S2 genes only
df["stratum"] = df.role
hub = df.hit_rate >= df.hit_rate.quantile(HUB_Q)
for r, sub in df[~hub].groupby("role"):
    q = pd.qcut(sub.hit_rate.rank(method="first"), 5, labels=False)
    df.loc[sub.index, "stratum"] = r + "_q" + q.astype(str)
df.loc[hub, "stratum"] = "hub"

rng = np.random.default_rng(SEED)
chosen = []
for s, sub in df.groupby("stratum", sort=True):
    ids = np.sort(sub.index.values)
    k = int(round(FRAC * len(ids)))
    chosen += list(rng.choice(ids, k, replace=False))
chosen = sorted(chosen)
OUT.write_text("\n".join(chosen) + "\n")
print(f"S2 genes: {len(df)}; lockbox genes: {len(chosen)}")
print(df.assign(lock=df.index.isin(chosen)).groupby("stratum").lock.agg(["size", "sum"]).to_string())
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
