"""Independent per-pair DAmP-only determination from the RAW Ryan S1 file (DESIGN §1). No src/ imports.
Compares to labels_dev.parquet damp_only / eval_eligible (dev pairs only; lockbox file never read)."""
import re, numpy as np, pandas as pd
R = "/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/"
S1 = R + "data/raw/ryan2012/extracted/mmc4/Dataset S1 - Unaveraged E-MAP with 953 Alleles.txt"
s1 = pd.read_csv(S1, sep="\t", index_col=0)
LRE = re.compile(r"^([^\s(]+)(?:\(([^)]*)\))?(?:\s*-\s*(.*))?$")
def parse(l):
    m = LRE.match(str(l).strip()); return m.group(1).upper(), (m.group(3) or "").strip().upper()
g = pd.read_csv(R + "data/processed/ryan_genes.tsv", sep="\t")
idmap = dict(zip(g.ryan_id.str.upper(), g.pombase_id))
merged = set(g.pombase_id[g.merged_with_other_ryan_id].dropna())
qi = [parse(l) for l in s1.index]; ai = [parse(c)[0] for c in s1.columns]
M = s1.to_numpy(float)
qq, aa = np.nonzero(np.isfinite(M))
d = pd.DataFrame({"q": [idmap.get(qi[i][0]) for i in qq], "suf": [qi[i][1] for i in qq],
                  "a": [idmap.get(ai[j]) for j in aa]})
print("S1 measurements", len(d), "unmapped q/a", d.q.isna().sum(), d.a.isna().sum())
d = d.dropna(subset=["q", "a"]); d = d[d.q != d.a]
print("suffix value counts", d.suf.value_counts().to_dict())
MAIN = {"", "DAMP", "DELEITON"}
d["main"] = d.suf.isin(MAIN)
d["damp"] = d.suf.str.startswith("DAMP")
d["gene_a"] = np.where(d.q < d.a, d.q, d.a); d["gene_b"] = np.where(d.q < d.a, d.a, d.q)
mm = d[d.main]
agg = mm.groupby(["gene_a", "gene_b"]).damp.agg(["size", "sum"])
agg["damp_only"] = agg["sum"] == agg["size"]; agg["damp_mixed"] = (agg["sum"] > 0) & (agg["sum"] < agg["size"])
# broad hypomorph variant: all alleles, non-deletion suffix (anything other than '', DELEITON, DELETION_NL) = hypomorph
d["nondel"] = ~d.suf.isin({"", "DELEITON", "DELETION_NL"})
broad = d.groupby(["gene_a", "gene_b"]).nondel.all().rename("allnondel_any_allele")

p = pd.read_parquet(R + "data/processed/ryan_pairs.parquet", columns=["gene_a", "gene_b", "s_score"])
p = p[p.s_score.notna()].set_index(["gene_a", "gene_b"])
u = pd.read_csv(R + "data/processed/universe_genes.tsv", sep="\t"); U = set(u.gene[u.in_universe])
p = p.join(agg[["size", "damp_only", "damp_mixed"]]).join(broad)
p["damp_only"] = p.damp_only.fillna(False).astype(bool); p["damp_mixed"] = p.damp_mixed.fillna(False).astype(bool)
ga = p.index.get_level_values(0); gb = p.index.get_level_values(1)
p["inU"] = ga.isin(U) & gb.isin(U)
p["merged"] = ga.isin(merged) | gb.isin(merged)
print("S2 pairs with no main measurement:", int(p["size"].isna().sum()), " of which in universe:", int((p["size"].isna() & p.inU).sum()))
print("universe DAmP-only pairs (mine):", int((p.inU & p.damp_only).sum()), " (summary 16557)")
print("universe DAmP-mixed pairs (mine):", int((p.inU & p.damp_mixed).sum()), " (summary 16948)")
print("universe merged-ID pairs (mine):", int((p.inU & p.merged).sum()), " (summary 4862)")
print("universe pairs where every S1 measurement (any allele) is non-deletion:", int((p.inU & p.allnondel_any_allele.fillna(False)).sum()))
p["elig"] = p.inU & ~p.damp_only & ~p.merged
print("eval-eligible (all S2):", int(p.elig.sum()), " (summary 1078622)")
# genes whose main query allele(s) are DAmP only
qal = mm.groupby("q").damp.agg(["all", "any"])
dq = set(qal.index[qal["all"]]); print("query genes with DAmP-only main allele:", len(dq), "in universe:", len(dq & U))
genelevel = (ga.isin(dq) | gb.isin(dq)) & p.inU
print("gene-level (any pair touching a DAmP-only-query gene) universe pairs:", int(genelevel.sum()))
anyd = set(qal.index[qal["any"]])
print("universe pairs touching any DAmP-screened gene:", int(((ga.isin(anyd) | gb.isin(anyd)) & p.inU).sum()))
# compare to built dev labels
dev = pd.read_parquet(R + "data/processed/splits/labels_dev.parquet",
                      columns=["gene_a", "gene_b", "damp_only", "damp_mixed", "eval_eligible", "in_universe", "involves_merged_id", "hypomorph_a", "hypomorph_b"]).set_index(["gene_a", "gene_b"])
j = dev.join(p[["damp_only", "damp_mixed", "elig", "inU", "merged"]], rsuffix="_mine")
for a, b in [("damp_only", "damp_only_mine"), ("damp_mixed", "damp_mixed_mine"), ("eval_eligible", "elig"), ("in_universe", "inU"), ("involves_merged_id", "merged")]:
    print(f"dev {a}: built {int(j[a].sum())} mine {int(j[b].sum())} mismatches {int((j[a] != j[b]).sum())}")
# hypomorph flags: gene_a has a main DAmP measurement in this pair (as query)
dd = mm[mm.damp]
ha = set(zip(dd.gene_a[dd.q == dd.gene_a], dd.gene_b[dd.q == dd.gene_a])); hb = set(zip(dd.gene_a[dd.q == dd.gene_b], dd.gene_b[dd.q == dd.gene_b]))
idx = list(j.index)
myha = np.array([k in ha for k in idx]); myhb = np.array([k in hb for k in idx])
print("hypomorph_a mismatches", int((myha != j.hypomorph_a.astype(bool)).sum()), "hypomorph_b mismatches", int((myhb != j.hypomorph_b.astype(bool)).sum()))
# universe genes whose every S2 pair is DAmP-only (counted as 'seen' in class weights but no deletion data)
s = p[p.inU | True].reset_index()
long = pd.concat([s[["gene_a", "damp_only"]].rename(columns={"gene_a": "g"}), s[["gene_b", "damp_only"]].rename(columns={"gene_b": "g"})])
allD = long.groupby("g").damp_only.all()
print("universe genes 'seen' only through DAmP-only S2 pairs:", sorted(set(allD.index[allD]) & U))
# Universe genes screened only as DAmP query AND not array: what fraction of their pairs is DAmP-only?
