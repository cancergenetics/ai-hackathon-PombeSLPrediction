"""Read-only checks for the round-2 data/domain review.
Usage: .venv/bin/python -I reviews/scratch/round2_data_domain_checks.py <project dir>"""
import sys
from pathlib import Path
import pandas as pd

R = Path(sys.argv[1])
P, RAW = R / "data/processed", R / "data/raw"
g = pd.read_csv(P / "ryan_genes.tsv", sep="\t")
p = pd.read_parquet(P / "ryan_pairs.parquet")
m = pd.read_parquet(P / "ryan_measurements.parquet")
ids = pd.read_csv(RAW / "pombase/gene_IDs_names_products.tsv", sep="\t")
fy = pd.read_csv(RAW / "pombase/FYPOviability.tsv", sep="\t", header=None, names=["id", "via"])
pc = set(ids[ids.gene_type == "protein coding gene"].gene_systematic_id)
print("protein-coding genes:", len(pc))
fyd = dict(zip(fy.id, fy.via))
via = pd.Series({x: fyd.get(x, "NA") for x in pc})
print("viability of protein-coding:", via.value_counts().to_dict())

s2 = p[p.in_s2].copy(); s2["y"] = s2["neg_s2_lt_-2.3"]
print("\n== Ryan genes by viability x role x in_s2")
gg = g[g.gene_type == "protein coding gene"]
print(pd.crosstab([gg.viability.fillna("NA"), gg.is_query], gg.in_s2_rows))
unk = gg[gg.viability.isin(["unknown"]) | gg.viability.isna()]
print("unknown-viability Ryan protein-coding genes:", len(unk))
print(unk[["pombase_id", "gene_name", "is_query", "is_array", "in_s2_rows", "allele_classes"]].to_string())
uset = set(unk.pombase_id)
x = s2[s2.gene_a.isin(uset) | s2.gene_b.isin(uset)]
print("S2 pairs with an unknown gene:", len(x), "pos", int(x.y.sum()), "rate %.4f" % x.y.mean())
print("protein-coding unknown/NA genes NOT in Ryan:", len(set(via[via.isin(['unknown', 'NA'])].index) - set(gg.pombase_id)))

print("\n== DAmP: per-pair source of S2 (main alleles) for universe genes")
uni_via = {"viable", "condition-dependent", "unknown"}
mm = m[m.main_allele].copy()
mm["damp"] = mm.allele_class == "DAmP"
gid = dict(zip(g.ryan_id, g.pombase_id))
mm["qa"] = mm.query_ryan_id.map(gid); mm["ar"] = mm.array_ryan_id.map(gid)
mm = mm.dropna(subset=["qa", "ar"])
mm["a"] = mm[["qa", "ar"]].min(axis=1); mm["b"] = mm[["qa", "ar"]].max(axis=1)
dg = gg[gg.allele_classes.fillna("").str.contains("DAmP") & (gg.viability.isin(uni_via) | gg.viability.isna())]
print(dg[["pombase_id", "gene_name", "viability", "is_array", "allele_classes"]].to_string())
dset = set(dg.pombase_id)
# pairs involving these genes: does S2 rest only on DAmP-query measurements, or also on a deletion orientation?
sub = mm[mm.a.isin(dset) | mm.b.isin(dset)]
agg = sub.groupby(["a", "b"]).damp.agg(["all", "any", "size"])
agg = agg.reset_index().rename(columns={"a": "gene_a", "b": "gene_b"})
jj = s2.merge(agg, on=["gene_a", "gene_b"], how="inner")
uv = jj[jj.viability_a.isin(uni_via) & jj.viability_b.isin(uni_via)]
print("universe S2 pairs touching DAmP genes:", len(uv), "pos", int(uv.y.sum()))
print("  S2 rests ONLY on DAmP measurements:", int(uv["all"].sum()), "pos", int(uv[uv["all"]].y.sum()))
print("  mixed DAmP + deletion:", int((uv["any"] & ~uv["all"]).sum()), "pos", int(uv[uv["any"] & ~uv["all"]].y.sum()))
print("  deletion only (other orientation):", int((~uv["any"]).sum()), "pos", int(uv[~uv["any"]].y.sum()))
for gname, d in uv.assign(gene=uv.apply(lambda r: r.gene_a if r.gene_a in dset else r.gene_b, axis=1)).groupby("gene"):
    print("   ", gname, g.set_index("pombase_id").gene_name.get(gname), "n", len(d), "damp-only", int(d["all"].sum()),
          "pos", int(d.y.sum()), "pos_damp_only", int(d[d["all"]].y.sum()))

print("\n== Inviable genes with Ryan array deletion strains (in S2)")
inv = gg[(gg.viability == "inviable") & gg.is_array]
print(len(inv), inv.gene_name.tolist())

print("\n== BioGRID S. pombe physical")
b = pd.read_csv(RAW / "biogrid/biogrid_5.0.262_spombe_physical.tab3.tsv.gz", sep="\t", low_memory=False)
print("rows", len(b))
print(b["Experimental System"].value_counts().to_string())
both = b[(b["Organism ID Interactor A"] == 284812) & (b["Organism ID Interactor B"] == 284812)]
print("pombe-pombe rows", len(both), "PMIDs", both["Publication Source"].nunique())
pm = both.groupby("Publication Source").size().sort_values(ascending=False)
print("top PMIDs by rows:\n", pm.head(12).to_string())
print("fraction rows in top-10 PMIDs: %.3f" % (pm.head(10).sum() / pm.sum()))
print("Ryan 2012 PMID 22681890 in physical:", (both["Publication Source"] == "PUBMED:22681890").sum())

print("\n== domain file columns (GO/InterPro2GO check)")
d = pd.read_csv(RAW / "pombase_protein/protein_domains_and_features.tsv", sep="\t", low_memory=False)
print(d.columns.tolist())
print("any GO: tokens:", int(d.astype(str).apply(lambda c: c.str.contains("GO:")).values.sum()))
print(d.iloc[:, 1].astype(str).str.extract(r"^([A-Za-z]+)")[0].value_counts().head(15).to_dict() if d.shape[1] > 1 else "")

print("\n== Rodriguez-Lopez sheet names")
xl = pd.ExcelFile(RAW / "rodriguez_lopez_2023_phenomics/elife-88229-supp1-v1.xlsx")
print(xl.sheet_names)
