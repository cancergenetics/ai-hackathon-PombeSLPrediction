#!/usr/bin/env python
"""Quick coverage counts of the auxiliary datasets (no feature building).

Universe = protein-coding S. pombe genes with a peptide in PomBase peptide.fa.gz.
Run with:  .venv/bin/python -I scripts/aux_coverage.py
Writes results/aux_coverage.tsv
"""
import csv
import gzip
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SYS = re.compile(r"^SP[ABCN][A-Z0-9]*\.\d+c?$|^SP[A-Z]{2,3}\d*[A-Z0-9]*\.\d+c?$|^SP[A-Z]+\d*\.\d+c?$")

rows = []


def rec(ds, what, n, note=""):
    rows.append((ds, what, n, note))
    print(f"{ds:34s} {what:58s} {n:>9} {note}")


# universe
uni = set()
with gzip.open(RAW / "pombase_protein/peptide.fa.gz", "rt") as fh:
    for l in fh:
        if l.startswith(">"):
            g = l[1:].split()[0].split(".")[0] + "." + l[1:].split()[0].split(".")[1] if False else l[1:].split(":")[0]
            g = re.sub(r"\.\d+$", "", g)  # SPAC1002.01.1 -> SPAC1002.01
            uni.add(g)
rec("pombase_protein", "protein-coding genes with peptide (universe)", len(uni))

# Entrez -> systematic via UniProt idmapping if available
entrez2sys = {}
idm = RAW / "uniprot_idmapping_pombe/SCHPO_284812_idmapping.dat.gz"
if idm.exists():
    acc2sys, acc2ent = {}, {}
    with gzip.open(idm, "rt") as fh:
        for l in fh:
            a, k, v = l.rstrip("\n").split("\t")
            if k == "PomBase":
                acc2sys[a] = v
            elif k == "GeneID":
                acc2ent.setdefault(a, []).append(v)
    for a, es in acc2ent.items():
        if a in acc2sys:
            for e in es:
                entrez2sys[e] = acc2sys[a]
    rec("uniprot_idmapping_pombe", "Entrez GeneID -> systematic ID mappings", len(entrez2sys))
    rec("uniprot_idmapping_pombe", "UniProt accessions with PomBase ID", len(acc2sys))

# domains
d = pd.read_csv(RAW / "pombase_protein/protein_domains_and_features.tsv", sep="\t", dtype=str)
rec("pombase_protein", "genes with any domain/feature call", d.systematic_id.nunique())
rec("pombase_protein", "genes with Pfam domain", d[d.database == "Pfam"].systematic_id.nunique())
rec("pombase_protein", "genes with PANTHER family", d[d.database == "PANTHER"].systematic_id.nunique())
rec("pombase_protein", "distinct Pfam domains", d[d.database == "Pfam"].domain_id.nunique())

# quant expression
q = pd.read_csv(RAW / "pombase_quant_expression/htp_gene_expression_table.tsv", sep="\t", dtype=str)
for lab, sub in (("Marguerat2012 RNA", q[(q.first_author == "Marguerat") & (q.term_name == "RNA level")]),
                 ("Marguerat2012 protein", q[(q.first_author == "Marguerat") & (q.term_name == "protein level")]),
                 ("Carpy2014 protein", q[q.first_author == "Carpy"])):
    rec("pombase_quant_expression", f"genes with {lab}", sub.gene.nunique())

# biogrid
bp = pd.read_csv(RAW / "biogrid/biogrid_5.0.262_spombe_physical.tab3.tsv.gz", sep="\t", dtype=str)
a, b = bp["Systematic Name Interactor A"], bp["Systematic Name Interactor B"]
genes = set(a) | set(b)
rec("biogrid_pombe_physical", "physical rows", len(bp))
rec("biogrid_pombe_physical", "genes in universe with >=1 physical interaction", len(genes & uni), f"of {len(genes)} total ids")
pairs = {frozenset(x) for x in zip(a, b) if x[0] != x[1]}
rec("biogrid_pombe_physical", "unique gene pairs (no self)", len(pairs))
vo = bp[bp.Author.str.startswith("Vo TV (2016)")]
rec("biogrid_pombe_physical", "Vo2016 Y2H/PCA rows", len(vo))
vg = set(vo["Systematic Name Interactor A"]) | set(vo["Systematic Name Interactor B"])
rec("biogrid_pombe_physical", "genes with Vo2016 interaction", len(vg & uni))
ht = bp[bp.Throughput.str.contains("High")]
hg = set(ht["Systematic Name Interactor A"]) | set(ht["Systematic Name Interactor B"])
rec("biogrid_pombe_physical", "genes with >=1 HTP-flagged interaction", len(hg & uni))
bs = pd.read_csv(RAW / "biogrid/biogrid_5.0.262_scerevisiae_physical.tab3.tsv.gz", sep="\t", dtype=str)
sg = set(bs["Systematic Name Interactor A"]) | set(bs["Systematic Name Interactor B"])
rec("biogrid_scerevisiae_physical", "S. cerevisiae physical rows", len(bs))
rec("biogrid_scerevisiae_physical", "S. cerevisiae genes with physical interaction", len(sg))

# coxpresdb
for tag, lab in (("Spo-r.c3-0", "RNA-seq"), ("Spo-m.c4-0", "microarray")):
    z = zipfile.ZipFile(RAW / f"coxpresdb_spo/{tag}.expression.combat.txt.zip")
    x = pd.read_csv(z.open(z.namelist()[0]), sep="\t", index_col=0)
    ids = set(map(str, x.index))
    mapped = {entrez2sys[i] for i in ids if i in entrez2sys}
    rec("coxpresdb_spo", f"{lab}: genes x samples", f"{x.shape[0]}x{x.shape[1]}")
    rec("coxpresdb_spo", f"{lab}: genes mapped to PomBase ID in universe", len(mapped & uni))

# matsuyama
m = pd.read_csv(RAW / "matsuyama2006_localization/ORFeome_Localizome_data.csv", skiprows=1, dtype=str,
                encoding="latin1")
loc = m[m["Localization"].notna() & (m["Localization"].str.strip() != "")]
rec("matsuyama2006_localization", "genes with localisation call in universe", len(set(loc["Gene Name"]) & uni),
    f"{len(loc)} rows")

# rodriguez-lopez phenomics
xl = RAW / "rodriguez_lopez_2023_phenomics/elife-88229-supp1-v1.xlsx"
dd = pd.read_excel(xl, sheet_name="Discretized all conditions", dtype=str)
rec("rodriguez_lopez_2023_phenomics", "genes (deletion mutants) x conditions", f"{len(dd)}x{dd.shape[1]-1}")
rec("rodriguez_lopez_2023_phenomics", "mutant genes in universe", len(set(dd.gene_id) & uni))

# kim 2010
import xlrd  # noqa: E402
wb = xlrd.open_workbook(RAW / "kim2010_deletion_library/41587_2010_BFnbt1628_MOESM3_ESM.xls")
s = wb.sheet_by_index(0)
ids = {str(s.cell_value(i, 0)).strip() for i in range(1, s.nrows)}
rec("kim2010_deletion_library", "S-Table1 library genes in universe", len(ids & uni))
wb = xlrd.open_workbook(RAW / "kim2010_deletion_library/41587_2010_BFnbt1628_MOESM5_ESM.xls")
s = wb.sheet_by_index(0)
ids = {str(s.cell_value(i, 1)).strip() for i in range(3, s.nrows)} if False else {str(s.cell_value(i, 0)).strip() for i in range(3, s.nrows)}
rec("kim2010_deletion_library", "S-Table15 rows with SP id (col0) in universe", len(ids & uni))
wb = xlrd.open_workbook(RAW / "kim2010_deletion_library/41587_2010_BFnbt1628_MOESM4_ESM.xls")
s = wb.sheet_by_index(0)
ids = {str(s.cell_value(i, 0)).strip() for i in range(1, s.nrows)}
rec("kim2010_deletion_library", "S-Table11 1:1-ortholog table genes in universe", len(ids & uni))

# ozturk (stream the 710 MB member; protein groups x strains)
z = zipfile.ZipFile(RAW / "ozturk2022_ko_proteome/41467_2022_33814_MOESM3_ESM.zip")
mem = "20220918_SourceData/20220918_SourceData/Data_Figure_1b.txt"
strains, prots, n = set(), set(), 0
with io.TextIOWrapper(z.open(mem), encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    next(rd)
    for r in rd:
        strains.add(r[1]); prots.add(r[0]); n += 1
ko = {s.split("-", 1)[1] for s in strains if "-" in s}
rec("ozturk2022_ko_proteome", "long-format rows (strain x protein)", n)
rec("ozturk2022_ko_proteome", "distinct strains", len(strains))
rec("ozturk2022_ko_proteome", "KO genes in universe", len(ko & uni))
rec("ozturk2022_ko_proteome", "protein groups measured (in universe)", len({p for p in prots if p in uni}), f"{len(prots)} total")

out = ROOT / "results" / "aux_coverage.tsv"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as fh:
    fh.write("dataset\tmetric\tvalue\tnote\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
print("wrote", out)
