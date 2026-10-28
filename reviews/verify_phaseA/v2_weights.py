"""Independent class weights over unmeasured universe pairs (DESIGN §3). No src/ imports."""
import pandas as pd, numpy as np, yaml
R = "/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/"
P = yaml.safe_load(open(R + "config/protocol.yaml"))
u = pd.read_csv(R + "data/processed/universe_genes.tsv", sep="\t")
U = set(u.gene[u.in_universe])
p = pd.read_parquet(R + "data/processed/ryan_pairs.parquet", columns=["gene_a", "gene_b", "s_score"])
s2 = p[p.s_score.notna()]
print("S2 pairs", len(s2), "dup keys", s2.duplicated(["gene_a", "gene_b"]).sum(), "a<b", (s2.gene_a < s2.gene_b).all())
seen = (set(s2.gene_a) | set(s2.gene_b)) & U
n, r = len(U), len(seen)
inU = s2[s2.gene_a.isin(U) & s2.gene_b.isin(U)]
m = len(inU)
c = lambda k: k * (k - 1) // 2
allp = {"C1": c(r), "C2": r * (n - r), "C3": c(n - r)}
unm = dict(allp, C1=allp["C1"] - m)
# brute-force check of unmeasured C1 count via explicit enumeration of seen x seen minus measured
f = lambda d: {k: round(v / sum(d.values()), 4) for k, v in d.items()}
print("|U|", n, "|R|", r, "measured universe pairs", m)
print("unmeasured counts", unm, "->", f(unm), "| config", P["class_weights"]["unmeasured_universe"])
print("all-pairs counts", allp, "->", f(allp), "| config", P["class_weights"]["all_pairs"])
# Sensitivity: what if 'seen' excluded genes whose S2 pairs are all excluded from training (train_drop partners)?
drop = set(u.gene[u.drop_from_training])
s2t = s2[~(s2.gene_a.isin(drop) | s2.gene_b.isin(drop))]
seen_t = (set(s2t.gene_a) | set(s2t.gene_b)) & U
print("seen via training pairs only:", len(seen_t), "(diff", sorted(seen - seen_t), ")")
# Sensitivity: universe genes seen only in S2 pairs with non-universe partners
seen_inU = (set(inU.gene_a) | set(inU.gene_b))
print("universe genes seen only via non-universe partners:", len(seen - seen_inU))
