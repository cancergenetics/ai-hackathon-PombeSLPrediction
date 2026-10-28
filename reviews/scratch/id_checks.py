"""Read-only: how many aux-source IDs are not current PomBase sysIDs but are synonyms (old IDs)."""
import gzip, sys
from pathlib import Path
import pandas as pd
R = Path(sys.argv[1]); RAW = R / "data/raw"
pb = pd.read_csv(RAW / "pombase/gene_IDs_names_products.tsv", sep="\t")
cur = set(pb.gene_systematic_id)
syn = {}
for sid, s in zip(pb.gene_systematic_id, pb.synonyms.fillna("")):
    for x in s.split(","):
        x = x.strip()
        if x: syn.setdefault(x.upper(), set()).add(sid)
curU = {c.upper(): c for c in cur}
def report(name, ids):
    ids = pd.Series(list(set(i for i in ids if isinstance(i, str))))
    ex = ids.isin(cur); exu = ids.str.upper().isin(curU)
    rest = ids[~exu]
    sy = rest.str.upper().map(lambda x: x in syn)
    print(f"{name}: n={len(ids)} exact={ex.sum()} case-insens={exu.sum()} not-current={len(rest)} of which old-ID synonym={sy.sum()} e.g. {rest[sy].head(4).tolist()} / unmapped e.g. {rest[~sy].head(4).tolist()}")
m = pd.read_csv(RAW / "matsuyama2006_localization/ORFeome_Localizome_data.csv", skiprows=1, encoding="latin-1")
report("Matsuyama", m["Gene Name"].astype(str).str.strip())
c = pd.read_csv(RAW / "ensembl_fungi_compara/Compara.116.pombe_homologies.tsv.gz", sep="\t", usecols=["gene_stable_id","species","homology_gene_stable_id","homology_species"])
c = pd.concat([c.loc[c.species=="schizosaccharomyces_pombe","gene_stable_id"], c.loc[c.homology_species=="schizosaccharomyces_pombe","homology_gene_stable_id"]]).to_frame()
report("Compara", c.iloc[:, 0])
b = pd.read_csv(RAW / "biogrid/biogrid_5.0.262_spombe_physical.tab3.tsv.gz", sep="\t", low_memory=False)
report("BioGRID", pd.concat([b["Systematic Name Interactor A"], b["Systematic Name Interactor B"]]))
import xlrd
for f in sorted((RAW / "kim2010_deletion_library").glob("*MOESM3*.xls")):
    wb = xlrd.open_workbook(f); sh = wb.sheet_by_index(0)
    vals = [str(sh.cell_value(r, cc)).strip() for r in range(sh.nrows) for cc in range(min(3, sh.ncols))]
    vals = [v for v in vals if v.upper().startswith("SP")]
    report("Kim2010 " + f.name[-14:], vals)
