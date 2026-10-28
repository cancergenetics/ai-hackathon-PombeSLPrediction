#!/usr/bin/env python
"""Re-runnable downloader for AUXILIARY (non-core) feature datasets.

Core datasets (Ryan 2012, Costanzo 2016, PomBase IDs/orthologs/viability) are owned
by scripts for DATA.md and are NOT fetched here.

Usage:
    .venv/bin/python scripts/download_aux_data.py            # everything
    .venv/bin/python scripts/download_aux_data.py biogrid coxpresdb_spo
    .venv/bin/python scripts/download_aux_data.py --list
    .venv/bin/python scripts/download_aux_data.py --skip ensembl_compara alphafold

Each dataset goes to data/raw/<dataset>/ . Downloads are skipped if the file already
exists (delete to force). A SHA256SUMS file is written per dataset and a global
data/raw/aux_checksums.tsv is (re)generated from all dataset dirs handled here.

Circularity guard: BioGRID files are filtered to PHYSICAL experimental systems only
before anything is kept (the raw BioGRID zip contains S. pombe genetic interactions
incl. Ryan 2012, so the zip and unfiltered text are deleted after filtering).
Downloaded files are data only; nothing downloaded is ever executed.
"""
import argparse
import datetime
import gzip
import hashlib
import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = "Mozilla/5.0 (PombeSL aux-data downloader)"
POMBASE = "https://www.pombase.org/data"
TODAY = datetime.date.today().isoformat()


# ----------------------------------------------------------------------------- helpers
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(url, dest, min_bytes=1):
    """Download url -> dest with curl (retries); skip if present."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  [skip] {dest.relative_to(ROOT)}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-delay", "5", "-A", UA,
                    "-o", str(tmp), url], check=True)
    if tmp.stat().st_size < min_bytes:
        raise RuntimeError(f"{url}: too small ({tmp.stat().st_size} bytes)")
    tmp.rename(dest)
    return dest


def write_sums(ds_dir, files=None, extra_note=""):
    ds_dir = Path(ds_dir)
    files = files or sorted(p for p in ds_dir.iterdir()
                            if p.is_file() and p.name not in ("SHA256SUMS", "RETRIEVED.txt")
                            and not p.name.endswith(".part"))
    with open(ds_dir / "SHA256SUMS", "w") as fh:
        for p in files:
            fh.write(f"{sha256(p)}  {p.name}\n")
    (ds_dir / "RETRIEVED.txt").write_text(f"retrieved {TODAY}\n{extra_note}\n")


# ----------------------------------------------------------------------------- datasets
def pombase_protein():
    d = RAW / "pombase_protein"
    # peptide sequences + InterPro-member domain calls computed from sequence by PomBase.
    fetch(f"{POMBASE}/genome_sequence_and_features/feature_sequences/peptide.fa.gz", d / "peptide.fa.gz")
    fetch(f"{POMBASE}/Protein_data/protein_domains_and_features.tsv", d / "protein_domains_and_features.tsv")
    fetch(f"{POMBASE}/Protein_data/PeptideStats.tsv", d / "PeptideStats.tsv")
    fetch(f"{POMBASE}/Protein_data/aa_composition.tsv", d / "aa_composition.tsv")
    fetch(f"{POMBASE}/Protein_data/transmembrane_domain_coords_and_seqs.tsv",
          d / "transmembrane_domain_coords_and_seqs.tsv")
    write_sums(d, extra_note="PomBase /data/ current files (rolling release; see sha256)")


def pombase_quant_expression():
    d = RAW / "pombase_quant_expression"
    base = f"{POMBASE}/releases/latest/misc"
    fetch(f"{base}/htp_gene_expression_table.tsv", d / "htp_gene_expression_table.tsv")
    fetch(f"{base}/full_gene_expression_table.tsv", d / "full_gene_expression_table.tsv")
    write_sums(d, extra_note="PomBase release 'latest' (2026-10-01): Marguerat 2012 RNA+protein, Carpy 2014 protein")


def matsuyama2006_localization():
    d = RAW / "matsuyama2006_localization"
    fetch("https://dnaconda.riken.jp/DownLoad/ORFeome_Localizome_data.csv", d / "ORFeome_Localizome_data.csv")
    write_sums(d, extra_note="RIKEN BRC S. pombe ORFeome; raw YFP localisation calls (Matsuyama 2006 Nat Biotech 24:841)")


def rodriguez_lopez_2023_phenomics():
    d = RAW / "rodriguez_lopez_2023_phenomics"
    # Supplementary file 1 ONLY (phenomics fitness). Supp file 2 is NET-FF GO predictions -> EXCLUDED.
    fetch("https://cdn.elifesciences.org/articles/88229/elife-88229-supp1-v1.xlsx",
          d / "elife-88229-supp1-v1.xlsx", min_bytes=10**6)
    write_sums(d, extra_note="eLife 88229 (Rodriguez-Lopez 2023). Use ONLY sheets 'Fitness all conditions'/'Discretized all conditions'. "
                             "Ignore 'GO enrichments','GO slims','KEGG' sheets (GO-derived).")


def kim2010_deletion_library():
    d = RAW / "kim2010_deletion_library"
    b = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnbt.1628/MediaObjects/41587_2010_BFnbt1628_MOESM{n}_ESM.xls"
    for n in (3, 4, 5, 6, 7):  # S-Table1 library, S-Table11 orthologs/paralogs, S-Table15 fitness, haplo lists
        fetch(b.format(n=n), d / f"41587_2010_BFnbt1628_MOESM{n}_ESM.xls", min_bytes=10**5)
    write_sums(d, extra_note="Kim 2010 Nat Biotechnol 28:617 Supp Tables 1,11,15 + haploinsufficiency lists (.xls; read with xlrd)")


def ozturk2022_ko_proteome():
    d = RAW / "ozturk2022_ko_proteome"
    fetch("https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-33814-8/MediaObjects/41467_2022_33814_MOESM3_ESM.zip",
          d / "41467_2022_33814_MOESM3_ESM.zip", min_bytes=10**8)
    write_sums(d, extra_note="Ozturk 2022 Nat Commun 13:6232 Source Data zip (868 MB). Data_Figure_1b.txt = long-format "
                             "log2 intensity of every protein group in each of 3,308 KO strains. Read members with zipfile, "
                             "do not extract blindly.")


def coxpresdb_spo():
    d = RAW / "coxpresdb_spo"
    base = "https://coxpresdb.jp/download"
    fetch(f"{base}/Spo-r.c3-0/exp/Spo-r.c3-0.expression.combat.txt.zip", d / "Spo-r.c3-0.expression.combat.txt.zip")
    fetch(f"{base}/Spo-m.c4-0/exp/Spo-m.c4-0.expression.combat.txt.zip", d / "Spo-m.c4-0.expression.combat.txt.zip")
    fetch(f"{base}/Spo-r.c3-0/genechip/data.tsv", d / "Spo-r.c3-0.sample_metadata.tsv")
    fetch(f"{base}/Spo-m.c4-0/genechip/data.tsv", d / "Spo-m.c4-0.sample_metadata.tsv")
    write_sums(d, extra_note="COXPRESdb Spo-r.c3-0 (RNA-seq, 5512 genes x 556 runs, v21-10) and Spo-m.c4-0 (microarray, 4881 x 166, v21-06); "
                             "row ids = NCBI Entrez Gene IDs; we use the expression matrices only (NOT the GO/KEGG-scored coex tables)")


def uniprot_idmapping_pombe():
    d = RAW / "uniprot_idmapping_pombe"
    fetch("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/SCHPO_284812_idmapping.dat.gz",
          d / "SCHPO_284812_idmapping.dat.gz", min_bytes=10**5)
    write_sums(d, extra_note="UniProt release 2026_03-ish (see Last-Modified 2026-09-03). Maps UniProt acc <-> PomBase systematic ID, Entrez GeneID, etc. "
                             "ID mapping only; contains GO column which must be ignored.")


def alphafold_spombe():
    d = RAW / "alphafold_spombe"
    fetch("https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v6.tar",
          d / "UP000002485_284812_SCHPO_v6.tar", min_bytes=10**8)
    write_sums(d, extra_note="AlphaFold DB S. pombe reference proteome (UP000002485), CC-BY 4.0. Predicted structures; not yet parsed.")


def biogrid():
    """BioGRID physical interactions only (S. pombe 284812, S. cerevisiae 559292)."""
    ver = "5.0.262"
    d = RAW / "biogrid"
    d.mkdir(parents=True, exist_ok=True)
    out_p = d / f"biogrid_{ver}_spombe_physical.tab3.tsv.gz"
    out_s = d / f"biogrid_{ver}_scerevisiae_physical.tab3.tsv.gz"
    if out_p.exists() and out_s.exists():
        print("  [skip] filtered BioGRID files present")
    else:
        zpath = d / f"BIOGRID-ORGANISM-{ver}.tab3.zip"
        legacy = d / "BIOGRID-ORGANISM-LATEST.tab3.zip"   # earlier manual download of the same release
        if not zpath.exists() and legacy.exists():
            legacy.rename(zpath)
        fetch("https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/"
              f"BIOGRID-{ver}/BIOGRID-ORGANISM-{ver}.tab3.zip", zpath, min_bytes=10**7)
        zsha = sha256(zpath)
        import pandas as pd
        drop_prefix = "Ontology Term"
        with zipfile.ZipFile(zpath) as z:
            for tag, org, out in (("Schizosaccharomyces_pombe_972h", "284812", out_p),
                                  ("Saccharomyces_cerevisiae_S288c", "559292", out_s)):
                member = next(n for n in z.namelist() if tag in n)
                df = pd.read_csv(z.open(member), sep="\t", dtype=str, low_memory=False)
                n_all = len(df)
                df = df[(df["Experimental System Type"] == "physical")
                        & (df["Organism ID Interactor A"] == org) & (df["Organism ID Interactor B"] == org)]
                df = df[[c for c in df.columns if not c.startswith(drop_prefix)]]
                assert set(df["Experimental System Type"]) == {"physical"}
                df.to_csv(out, sep="\t", index=False, compression="gzip")
                print(f"  {tag}: {n_all} rows -> {len(df)} physical rows")
        (d / "SOURCE_ZIP_SHA256.txt").write_text(f"{zsha}  BIOGRID-ORGANISM-{ver}.tab3.zip\n")
        zpath.unlink()  # raw zip contains genetic interactions (incl. S. pombe) -> do not keep
        for p in d.glob("BIOGRID-ORGANISM-*.txt"):
            p.unlink()
    write_sums(d, files=sorted(p for p in d.iterdir() if p.name.endswith(".tsv.gz")),
               extra_note=f"BioGRID {ver}; PHYSICAL experimental systems only (genetic rows dropped, raw zip deleted; zip sha256 in SOURCE_ZIP_SHA256.txt)")


def ensembl_compara():
    """Stream the 7 GB Ensembl Fungi Compara 116 homology TSV and keep S. pombe rows only."""
    d = RAW / "ensembl_fungi_compara"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "Compara.116.pombe_homologies.tsv.gz"
    if out.exists():
        print("  [skip] filtered Compara file present")
    else:
        base = "https://ftp.ebi.ac.uk/ensemblgenomes/pub/current/fungi/tsv/ensembl-compara/homologies"
        want_md5 = subprocess.run(["curl", "-fsSL", f"{base}/MD5SUM"], check=True, capture_output=True,
                                  text=True).stdout.split()[0]
        print("  [stream] Compara.116.protein_default.homologies.tsv.gz (~7 GB; ~15-30 min)")
        p = subprocess.Popen(["curl", "-fsSL", "-A", UA, f"{base}/Compara.116.protein_default.homologies.tsv.gz"],
                             stdout=subprocess.PIPE)
        md5 = hashlib.md5()

        class Tee(io.RawIOBase):
            def readable(self): return True
            def readinto(self, b):
                chunk = p.stdout.read(len(b))
                md5.update(chunk)
                b[:len(chunk)] = chunk
                return len(chunk)

        n_in = n_out = 0
        tmp = out.with_suffix(".part")
        with gzip.open(io.BufferedReader(Tee(), 1 << 22), "rt") as fin, gzip.open(tmp, "wt") as fout:
            header = fin.readline()
            fout.write(header)
            for line in fin:
                n_in += 1
                if "schizosaccharomyces_pombe" in line:
                    f = line.split("\t", 8)
                    if f[2] == "schizosaccharomyces_pombe" or f[7] == "schizosaccharomyces_pombe":
                        fout.write(line)
                        n_out += 1
        p.wait()
        if p.returncode != 0 or md5.hexdigest() != want_md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Compara stream failed/md5 mismatch ({md5.hexdigest()} vs {want_md5})")
        tmp.rename(out)
        (d / "SOURCE_MD5.txt").write_text(f"{want_md5}  Compara.116.protein_default.homologies.tsv.gz (verified)\n"
                                          f"rows_in={n_in} rows_kept={n_out}\n")
    fetch("https://ftp.ebi.ac.uk/ensemblgenomes/pub/current/fungi/tsv/ensembl-compara/homologies/README.gene_trees.tsv_dumps.txt",
          d / "README.gene_trees.tsv_dumps.txt")
    write_sums(d, extra_note="Ensembl Fungi Compara 116 gene-tree homologies (sequence-based), rows involving S. pombe only")


DATASETS = {
    "pombase_protein": pombase_protein,
    "pombase_quant_expression": pombase_quant_expression,
    "biogrid": biogrid,
    "coxpresdb_spo": coxpresdb_spo,
    "ozturk2022_ko_proteome": ozturk2022_ko_proteome,
    "rodriguez_lopez_2023_phenomics": rodriguez_lopez_2023_phenomics,
    "kim2010_deletion_library": kim2010_deletion_library,
    "matsuyama2006_localization": matsuyama2006_localization,
    "uniprot_idmapping_pombe": uniprot_idmapping_pombe,
    "alphafold_spombe": alphafold_spombe,
    "ensembl_compara": ensembl_compara,
}
DIRNAME = {"ensembl_compara": "ensembl_fungi_compara"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="datasets to fetch (default: all)")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(DATASETS)); return
    names = a.names or list(DATASETS)
    bad = [n for n in names + a.skip if n not in DATASETS]
    if bad:
        sys.exit(f"unknown dataset(s): {bad}")
    for n in names:
        if n in a.skip:
            continue
        print(f"== {n}")
        DATASETS[n]()
    # global checksum table
    rows = []
    for n in DATASETS:
        dd = RAW / DIRNAME.get(n, n)
        s = dd / "SHA256SUMS"
        if s.exists():
            for line in s.read_text().splitlines():
                h, f = line.split("  ", 1)
                rows.append((dd.name, f, h, (dd / f).stat().st_size))
    with open(RAW / "aux_checksums.tsv", "w") as fh:
        fh.write("dataset\tfile\tsha256\tbytes\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {RAW / 'aux_checksums.tsv'} ({len(rows)} files)")


if __name__ == "__main__":
    main()
