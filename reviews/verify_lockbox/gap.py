"""Dev-vs-lockbox gap probes from existing artefacts (no retraining). Aggregates only."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P = yaml.safe_load(open(ROOT / "config/protocol.yaml"))
THR = P["label"]["threshold"]
R = json.load(open(ROOT / "results/lockbox/lockbox_result_deec74e31a7c.json"))
out = {}
# ---- 1. per-class dev vs lockbox, prevalence, lift, SE
rows = []
for m, r in R["models"].items():
    d = json.load(open(ROOT / f"results/runs/{r['dev_run']}.json"))
    mm, pf = d["metrics_mean"], d["metrics_per_fold"]
    row = {"model": m, "dev_primary": d["primary"], "lb_primary": r["primary"], "gap": r["primary"] - d["primary"],
           "dev_se": d["bootstrap"]["primary"]["se"], "lb_se": r["primary_se"]}
    row["gap_z"] = row["gap"] / np.hypot(row["dev_se"], row["lb_se"])
    row["dev_fold_primary_range"] = [min(pf[k]["pooled"] for k in pf), max(pf[k]["pooled"] for k in pf)]
    for c in ("C1", "C2", "C3"):
        dp = mm[f"pos_{c}"] / mm[f"n_{c}"]; lp = r[f"pos_{c}"] / r[f"n_{c}"]
        row[f"dev_{c}"] = mm[f"ap_{c}"]; row[f"lb_{c}"] = r[f"ap_{c}"]; row[f"gap_{c}"] = r[f"ap_{c}"] - mm[f"ap_{c}"]
        row[f"dev_prev_{c}"] = dp; row[f"lb_prev_{c}"] = lp
        row[f"dev_lift_{c}"] = mm[f"ap_{c}"] / dp; row[f"lb_lift_{c}"] = r[f"ap_{c}"] / lp
        row[f"dev_fold_{c}_range"] = [min(pf[k][f"ap_{c}"] for k in pf), max(pf[k][f"ap_{c}"] for k in pf)]
        row[f"lb_{c}_ci"] = r[f"ap_{c}_ci"]
    rows.append(row)
G = pd.DataFrame(rows).set_index("model")
out["per_model"] = G.to_dict(orient="index")
pd.set_option("display.width", 250)
print(G[["dev_primary", "lb_primary", "gap", "dev_se", "lb_se", "gap_z", "gap_C1", "gap_C2", "gap_C3"]].round(4))
print(G[[f"{a}_prev_{c}" for c in ("C1", "C2", "C3") for a in ("dev", "lb")]].iloc[:1].round(4))
print(G[[f"{a}_lift_{c}" for c in ("C1", "C2", "C3") for a in ("dev", "lb")]].round(2))
for f in sorted(glob.glob(str(ROOT / "results/runs/M2r_huber*.json"))):
    d = json.load(open(f)); print("dev run", Path(f).stem, round(d["primary"], 4), d.get("split_seed"))

# ---- 2. lockbox gene composition vs dev (draw_lockbox stratification re-derived)
LB = set((ROOT / "config/lockbox_genes.txt").read_text().split())
pairs = pd.read_parquet(ROOT / "data/processed/ryan_pairs.parquet", columns=["gene_a", "gene_b", "s_score", "in_s2"])
pairs = pairs[pairs.in_s2]
genes = pd.read_csv(ROOT / "data/processed/ryan_genes.tsv", sep="\t").dropna(subset=["pombase_id"])
role = genes.groupby("pombase_id").agg(q=("is_query", "any"), a=("is_array", "any"))
role["role"] = np.where(role.q & role.a, "both", np.where(role.q, "query", "array"))
long = pd.concat([pairs[["gene_a", "s_score"]].rename(columns={"gene_a": "g"}),
                  pairs[["gene_b", "s_score"]].rename(columns={"gene_b": "g"})])
long["y"] = long.s_score < THR
st = long.groupby("g").agg(hit_rate=("y", "mean"), degree=("y", "size"), n_pos=("y", "sum"))
df = role.join(st, how="inner")
hub = df.hit_rate >= df.hit_rate.quantile(0.98)
df["stratum"] = df.role
for r_, sub in df[~hub].groupby("role"):
    q = pd.qcut(sub.hit_rate.rank(method="first"), 5, labels=False)
    df.loc[sub.index, "stratum"] = r_ + "_q" + q.astype(str)
df.loc[hub, "stratum"] = "hub"
df["lock"] = df.index.isin(LB)
comp = df.groupby("stratum").lock.agg(n="size", n_lock="sum")
comp["frac_lock"] = comp.n_lock / comp.n
print(comp.round(3).to_string())
out["strata"] = comp.to_dict(orient="index")
out["lockbox_genes_in_S2_table"] = int(df.lock.sum())
summ = df.groupby("lock").agg(n=("role", "size"), hit_rate_mean=("hit_rate", "mean"), hit_rate_median=("hit_rate", "median"),
                              degree_median=("degree", "median"), frac_query=("q", "mean"), frac_array=("a", "mean"))
print(summ.round(4).to_string()); out["gene_summary_by_lock"] = summ.to_dict(orient="index")
# hubs: positives concentration
for lk, sub in df.groupby("lock"):
    s = sub.n_pos.sort_values(ascending=False)
    out[f"top10_share_of_gene_positives_lock={lk}"] = float(s.iloc[:10].sum() / s.sum())
    out[f"hub_count_lock={lk}"] = int((sub.stratum == "hub").sum())
print({k: v for k, v in out.items() if k.startswith(("top10", "hub_count"))})

# ---- 3. label-free feature coverage of lockbox vs dev genes (sim rows non-missing)
fg = [g for g in (ROOT / "data/processed/features/genes.txt").read_text().split() if g]
pos = pd.Series(np.arange(len(fg)), index=fg)
devg = [g for g in df.index[~df.lock] if g in pos.index]; lbg = [g for g in df.index[df.lock] if g in pos.index]
cov = {}
for f in sorted((ROOT / "data/processed/features/sim").glob("*.npy")):
    M = np.load(f, mmap_mode="r")
    fin = lambda gs: float(np.mean([np.isfinite(M[pos[g]]).any() for g in gs]))
    cov[f.stem] = {"dev": fin(devg), "lockbox": fin(lbg)}
print(pd.DataFrame(cov).T.round(3).to_string()); out["sim_row_coverage"] = cov

# ---- 4. concentration / influence of single lockbox genes on M2r_huber C2 AP
from sklearn.metrics import average_precision_score as aps
pr = pd.read_parquet(ROOT / "data/processed/lockbox/preds_M2r_huber.parquet")
L = pd.read_parquet(ROOT / "data/processed/splits/labels_lockbox.parquet", columns=["gene_a", "gene_b", "s_score"])
k = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a)
L.index = k(L.gene_a.values, L.gene_b.values)
pr["y"] = (L.loc[k(pr.gene_a.values, pr.gene_b.values), "s_score"].values < THR)
c2 = pr[pr.part == "C2"].copy()
c2["lg"] = np.where(c2.gene_a.isin(LB), c2.gene_a, c2.gene_b)
base = aps(c2.y, c2.score)
npos = c2.groupby("lg").y.sum().sort_values(ascending=False)
infl = {}
for g in npos.index[:15]:
    s = c2[c2.lg != g]; infl[g] = aps(s.y, s.score) - base
iv = pd.Series(infl)
out["C2_top15_lockbox_genes_share_of_C2_pos"] = float(npos.iloc[:15].sum() / npos.sum())
out["C2_drop1_influence_range"] = [float(iv.min()), float(iv.max())]
out["C2_drop_top5_influencers"] = float(aps(c2[~c2.lg.isin(iv.sort_values().index[:5])].y,
                                             c2[~c2.lg.isin(iv.sort_values().index[:5])].score))
print("C2 base", round(base, 4), {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in out.items() if k_.startswith("C2_")})
json.dump(out, open(OUT / "gap_result.json", "w"), indent=1, default=float)
