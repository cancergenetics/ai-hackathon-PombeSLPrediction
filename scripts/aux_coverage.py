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


ryan = set()
_rg = ROOT / "data" / "processed" / "ryan_genes.tsv"   # written by the core-data agent; read-only here
if _rg.exists():
    _g = pd.read_csv(_rg, sep="\t", dtype=str)
    ryan = set(_g.pombase_id.dropna())


def rec(ds, what, n, note="", gs=None):
    """gs = optional gene set; adds the count of Ryan-2012 genes covered."""
    nr = len(set(gs) & ryan) if (gs is not None and ryan) else ""
    rows.append((ds, what, n, nr, note))
    print(f"{ds:34s} {what:58s} {n:>9} {nr!s:>6} {note}")


# universe
uni = set()
with gzip.open(RAW / "pombase_protein/peptide.fa.gz", "rt") as fh:
    for l in fh:
        if l.startswith(">"):
            g = l[1:].split()[0].split(".")[0] + "." + l[1:].split()[0].split(".")[1] if False else l[1:].split(":")[0]
            g = re.sub(r"\.\d+$", "", g)  # SPAC1002.01.1 -> SPAC1002.01
            uni.add(g)
rec("pombase_protein", "protein-coding genes with peptide (universe)", len(uni), gs=uni)
rec("ryan2012", "Ryan genes (pombase_id) in universe [reference]", len(ryan & uni), gs=ryan)

# Entrez -> systematic via NCBI Datasets (preferred), plus UniProt idmapping as a cross-check
entrez2sys = {}
nc = RAW / "ncbi_gene_pombe/entrez_to_pombase.tsv"
if nc.exists():
    g = pd.read_csv(nc, sep="\t", dtype=str).dropna(subset=["pombase_id"])
    entrez2sys = dict(zip(g.entrez_id, g.pombase_id))
    rec("ncbi_gene_pombe", "Entrez GeneID -> PomBase ID (all gene types)", len(entrez2sys))
    rec("ncbi_gene_pombe", "protein-coding Entrez genes in universe", len(set(entrez2sys.values()) & uni))
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
    n_agree = n_conflict = 0
    for a, es in acc2ent.items():
        if a in acc2sys:
            for e in es:
                if e in entrez2sys:
                    if entrez2sys[e] == acc2sys[a]: n_agree += 1
                    else: n_conflict += 1
                else:
                    entrez2sys[e] = acc2sys[a]
    rec("uniprot_idmapping_pombe", "Entrez->sysID agreeing with NCBI / conflicting", f"{n_agree}/{n_conflict}")
    rec("uniprot_idmapping_pombe", "UniProt accessions with PomBase ID", len(acc2sys))

# domains
d = pd.read_csv(RAW / "pombase_protein/protein_domains_and_features.tsv", sep="\t", dtype=str)
rec("pombase_protein", "genes with any domain/feature call", d.systematic_id.nunique(), gs=set(d.systematic_id))
rec("pombase_protein", "genes with Pfam domain", d[d.database == "Pfam"].systematic_id.nunique(), gs=set(d[d.database == "Pfam"].systematic_id))
rec("pombase_protein", "genes with PANTHER family", d[d.database == "PANTHER"].systematic_id.nunique(), gs=set(d[d.database == "PANTHER"].systematic_id))
rec("pombase_protein", "distinct Pfam domains", d[d.database == "Pfam"].domain_id.nunique())

# quant expression
q = pd.read_csv(RAW / "pombase_quant_expression/htp_gene_expression_table.tsv", sep="\t", dtype=str)
for lab, sub in (("Marguerat2012 RNA", q[(q.first_author == "Marguerat") & (q.term_name == "RNA level")]),
                 ("Marguerat2012 protein", q[(q.first_author == "Marguerat") & (q.term_name == "protein level")]),
                 ("Carpy2014 protein", q[q.first_author == "Carpy"])):
    rec("pombase_quant_expression", f"universe genes with {lab}", len(set(sub.gene) & uni), gs=set(sub.gene))

# biogrid
bp = pd.read_csv(RAW / "biogrid/biogrid_5.0.262_spombe_physical.tab3.tsv.gz", sep="\t", dtype=str)
a, b = bp["Systematic Name Interactor A"], bp["Systematic Name Interactor B"]
genes = set(a) | set(b)
rec("biogrid_pombe_physical", "physical rows", len(bp))
rec("biogrid_pombe_physical", "genes in universe with >=1 physical interaction", len(genes & uni), f"of {len(genes)} total ids", gs=genes)
pairs = {frozenset(x) for x in zip(a, b) if x[0] != x[1]}
rec("biogrid_pombe_physical", "unique gene pairs (no self)", len(pairs))
vo = bp[bp.Author.str.startswith("Vo TV (2016)")]
rec("biogrid_pombe_physical", "Vo2016 Y2H/PCA rows", len(vo))
vg = set(vo["Systematic Name Interactor A"]) | set(vo["Systematic Name Interactor B"])
rec("biogrid_pombe_physical", "genes with Vo2016 interaction", len(vg & uni), gs=vg)
ht = bp[bp.Throughput.str.contains("High")]
hg = set(ht["Systematic Name Interactor A"]) | set(ht["Systematic Name Interactor B"])
rec("biogrid_pombe_physical", "genes with >=1 HTP-flagged interaction", len(hg & uni), gs=hg)
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
    rec("coxpresdb_spo", f"{lab}: genes mapped to PomBase ID in universe", len(mapped & uni), gs=mapped)

# matsuyama
m = pd.read_csv(RAW / "matsuyama2006_localization/ORFeome_Localizome_data.csv", skiprows=1, dtype=str,
                encoding="latin1")
loc = m[m["Localization"].notna() & (m["Localization"].str.strip() != "")]
rec("matsuyama2006_localization", "genes with localisation call in universe", len(set(loc["Gene Name"]) & uni),
    f"{len(loc)} rows", gs=set(loc["Gene Name"]))

# rodriguez-lopez phenomics
xl = RAW / "rodriguez_lopez_2023_phenomics/elife-88229-supp1-v1.xlsx"
dd = pd.read_excel(xl, sheet_name="Discretized all conditions", dtype=str)
rec("rodriguez_lopez_2023_phenomics", "genes (deletion mutants) x conditions", f"{len(dd)}x{dd.shape[1]-1}")
rec("rodriguez_lopez_2023_phenomics", "mutant genes in universe", len(set(dd.gene_id) & uni), gs=set(dd.gene_id))

# kim 2010
import xlrd  # noqa: E402
wb = xlrd.open_workbook(RAW / "kim2010_deletion_library/41587_2010_BFnbt1628_MOESM3_ESM.xls")
s = wb.sheet_by_index(0)
ids = {str(s.cell_value(i, 0)).strip() for i in range(1, s.nrows)}
rec("kim2010_deletion_library", "S-Table1 library genes in universe", len(ids & uni), gs=ids)
wb = xlrd.open_workbook(RAW / "kim2010_deletion_library/41587_2010_BFnbt1628_MOESM5_ESM.xls")
s = wb.sheet_by_index(0)
ids = {str(s.cell_value(i, 1)).strip() for i in range(3, s.nrows)} if False else {str(s.cell_value(i, 0)).strip() for i in range(3, s.nrows)}
rec("kim2010_deletion_library", "S-Table15 het-diploid fitness genes in universe", len(ids & uni), gs=ids)
wb = xlrd.open_workbook(RAW / "kim2010_deletion_library/41587_2010_BFnbt1628_MOESM4_ESM.xls")
s = wb.sheet_by_index(0)
ids = {str(s.cell_value(i, 0)).strip() for i in range(1, s.nrows)}
rec("kim2010_deletion_library", "S-Table11 (Sp-Sc paralog/ortholog) genes in universe", len(ids & uni), gs=ids)

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
rec("ozturk2022_ko_proteome", "KO genes in universe", len(ko & uni), gs=ko)
rec("ozturk2022_ko_proteome", "protein groups measured (in universe)", len({p for p in prots if p in uni}), f"{len(prots)} total", gs=prots)

# ensembl compara (pombe-filtered)
cf = RAW / "ensembl_fungi_compara/Compara.116.pombe_homologies.tsv.gz"
if cf.exists():
    cd = pd.read_csv(cf, sep="\t", dtype=str)
    P = "schizosaccharomyces_pombe"
    w = cd[(cd.species == P) & (cd.homology_species == P)]
    wg = set(w.gene_stable_id) | set(w.homology_gene_stable_id)
    rec("ensembl_fungi_compara", "within-pombe paralogue pairs (within_species + other_paralog)", len(w))
    rec("ensembl_fungi_compara", "genes in >=1 within-pombe paralogue pair", len(wg & uni), gs=wg)
    o = cd[cd.homology_type.str.startswith("ortholog")]
    og = (set(o[o.species == P].gene_stable_id) | set(o[o.homology_species == P].homology_gene_stable_id)) & uni
    rec("ensembl_fungi_compara", "genes with >=1 ortholog in another fungal genome", len(og), gs=og)
    partners = set(o[o.species == P].homology_species) | set(o[o.homology_species == P].species)
    rec("ensembl_fungi_compara", "partner genomes with pombe orthologs", len(partners - {P}))

out = ROOT / "results" / "aux_coverage.tsv"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as fh:
    fh.write("dataset\tmetric\tvalue\tn_in_ryan_genes\tnote\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
print("wrote", out)
