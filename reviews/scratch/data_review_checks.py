"""Read-only checks for the round-1 data review."""
import gzip
import sys
from pathlib import Path
import pandas as pd

R = Path(sys.argv[1])
P, RAW = R / "data/processed", R / "data/raw"
p = pd.read_parquet(P / "ryan_pairs.parquet")
g = pd.read_csv(P / "ryan_genes.tsv", sep="\t")
s2 = p[p.in_s2].copy()
s2["y"] = s2.s_score < -2.3
print("== viability classes of S2 pairs")
s2["vv"] = s2.viability_a.astype(str) + "|" + s2.viability_b.astype(str)
cd = s2[(s2.viability_a == "condition-dependent") | (s2.viability_b == "condition-dependent")]
cd_ok = cd[cd.viability_a.isin(["viable", "condition-dependent"]) & cd.viability_b.isin(["viable", "condition-dependent"])]
print("pairs cd x (viable|cd):", len(cd_ok), "pos", int(cd_ok.y.sum()))
unk = s2[(s2.viability_a.isna() | (s2.viability_a == "unknown") | s2.viability_b.isna() | (s2.viability_b == "unknown"))]
print("pairs with unknown/NA viability:", len(unk), "pos", int(unk.y.sum()))
vv = s2[(s2.viability_a == "viable") & (s2.viability_b == "viable")]
print("viable x viable:", len(vv), "pos", int(vv.y.sum()))
print("== viable/cd query genes whose S2 includes a DAmP strain")
dq = g[g.allele_classes.fillna("").str.contains("DAmP") & g.viability.isin(["viable", "condition-dependent"])]
print(dq[["ryan_id", "gene_name", "viability", "allele_classes"]].to_string())
dg = set(dq.pombase_id)
x = vv[vv.gene_a.isin(dg) | vv.gene_b.isin(dg)]
print("viable x viable pairs involving these genes:", len(x), "pos", int(x.y.sum()))
print("== merged IDs")
m = vv[vv.involves_merged_id]
print("viable x viable merged pairs:", len(m), "pos", int(m.y.sum()))
print(g[g.merged_with_other_ryan_id][["ryan_id", "pombase_id", "gene_name", "viability", "is_query", "is_array"]].to_string())
print("== orientation: positives by both_orientations in viable x viable")
print(vv.groupby("both_orientations").y.agg(["size", "mean"]))
print("== BioGRID pombe physical")
b = pd.read_csv(RAW / "biogrid/biogrid_5.0.262_spombe_physical.tab3.tsv.gz", sep="\t", low_memory=False)
print(b["Throughput"].value_counts().to_string())
print(b["Source Database"].value_counts().to_string())
b["a"] = b["Systematic Name Interactor A"]; b["b"] = b["Systematic Name Interactor B"]
b["ga"] = b[["a", "b"]].min(axis=1); b["gb"] = b[["a", "b"]].max(axis=1)
lt = b[b.Throughput.str.contains("Low")]
print("LTP rows", len(lt), "LTP PMIDs", lt["Publication Source"].nunique())
pos = vv[vv.y][["gene_a", "gene_b"]]
allp = vv[["gene_a", "gene_b", "y"]]
for name, sub in [("all", b), ("LTP", lt), ("HTP", b[~b.Throughput.str.contains("Low")])]:
    e = sub[["ga", "gb"]].drop_duplicates().rename(columns={"ga": "gene_a", "gb": "gene_b"})
    mm = allp.merge(e, on=["gene_a", "gene_b"])
    print(name, "Ryan vxv pairs with edge", len(mm), "pos", int(mm.y.sum()), "rate %.3f" % mm.y.mean() if len(mm) else "")
# year of LTP pubs
lt_y = lt["Publication Source"].str.extract(r"(\d{4})$")
print("== PomBase-source rows by throughput")
print(pd.crosstab(b["Source Database"], b["Throughput"]))
print("== PANTHER ID format")
d = pd.read_csv(RAW / "pombase_protein/protein_domains_and_features.tsv", sep="\t", low_memory=False)
print(d.columns.tolist())
print(d.head(3).to_string())
col = [c for c in d.columns if d[c].astype(str).str.startswith("PTHR").any()]
for c in col:
    v = d[c].astype(str); v = v[v.str.startswith("PTHR")]
    print(c, "PTHR ids", v.nunique(), "with :SF", v.str.contains(":SF").sum(), v.head(5).tolist())
print("== COXPRESdb titles mentioning double/genetic")
for f in ["Spo-r.c3-0.sample_metadata.tsv", "Spo-m.c4-0.sample_metadata.tsv"]:
    md = pd.read_csv(RAW / "coxpresdb_spo" / f, sep="\t")
    t = md.astype(str).agg(" ".join, axis=1)
    hit = t[t.str.contains(r"double|genetic interact|synthetic|epista", case=False)]
    print(f, len(md), "hits", len(hit)); print(hit.head(8).to_string())
