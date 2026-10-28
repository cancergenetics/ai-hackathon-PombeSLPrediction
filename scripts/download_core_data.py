#!/usr/bin/env python
"""Download the CORE datasets for the project into data/raw/<dataset>/.

Datasets (see DATA.md for provenance/licences):
  ryan2012     Ryan et al. 2012 Mol Cell 46:691 supplementary files (Elsevier CDN, PII S1097276512004443)
  costanzo2016 Costanzo et al. 2016 Science global S. cerevisiae GI map, Data File S1 (Boone lab supplement site)
  pombase      PomBase monthly release (pinned) reference files: IDs/names/synonyms, viability, cerevisiae orthologs

Re-runnable: a file that already exists is not re-downloaded (use --force to refresh).
Every file's sha256 is written to data/raw/<dataset>/SHA256SUMS and to data/raw/core_manifest.tsv
(with URL, size and retrieval date). Zip archives are extracted (with path-traversal checks) into
data/raw/<dataset>/extracted/<archive-stem>/. Downloaded content is treated as untrusted data:
nothing from it is executed.

Usage:  .venv/bin/python -I scripts/download_core_data.py [--force] [--only ryan2012 pombase ...]
"""
import argparse
import datetime as dt
import hashlib
import sys
import urllib.parse
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

POMBASE_RELEASE = "pombase-2026-10-01"
_PB = f"https://www.pombase.org/data/releases/{POMBASE_RELEASE}"
_RYAN = "https://ars.els-cdn.com/content/image/1-s2.0-S1097276512004443-"
_COST = "https://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/data_files/"

# dataset -> list of (url, local filename, extract_zip)
SOURCES = {
    "ryan2012": [
        # Elsevier "mmc" numbering; mmc1-3 are .xls tables (strain list etc.), mmc4-7 zipped text
        # data files, mmc8 the supplementary methods PDF. Contents are documented in DATA.md.
        (_RYAN + "mmc1.xls", "mmc1.xls", False),
        (_RYAN + "mmc2.xls", "mmc2.xls", False),
        (_RYAN + "mmc3.xls", "mmc3.xls", False),
        (_RYAN + "mmc4.zip", "mmc4.zip", True),
        (_RYAN + "mmc5.zip", "mmc5.zip", True),
        (_RYAN + "mmc6.zip", "mmc6.zip", True),
        (_RYAN + "mmc7.zip", "mmc7.zip", True),
        (_RYAN + "mmc8.pdf", "mmc8_supplementary_methods.pdf", False),
    ],
    "costanzo2016": [
        (_COST + urllib.parse.quote("Data File S1_Raw genetic interaction datasets: Pair-wise interaction format.zip"),
         "Data_File_S1_pairwise.zip", True),
    ],
    "pombase": [
        (f"{_PB}/misc/gene_IDs_names_products.tsv", "gene_IDs_names_products.tsv", False),
        (f"{_PB}/misc/FYPOviability.tsv", "FYPOviability.tsv", False),
        (f"{_PB}/exports/pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz",
         "pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz", False),
        (f"{_PB}/pombe-embl/orthologs/cerevisiae_orthologs.tsv", "cerevisiae_orthologs_curation_table.tsv", False),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "PombeSL-data-fetch/1.0"}) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            raise RuntimeError(f"{url} returned HTML (challenge page?), refusing to save")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)


def safe_extract(zpath: Path, outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    root = outdir.resolve()
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            target = (outdir / info.filename).resolve()
            if root not in target.parents:
                raise RuntimeError(f"unsafe path in {zpath.name}: {info.filename}")
            if target.exists() and target.stat().st_size == info.file_size:
                n += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="restrict to these datasets")
    args = ap.parse_args()
    today = dt.date.today().isoformat()
    manifest_path = RAW / "core_manifest.tsv"
    manifest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines()[1:]:
            f = line.split("\t")
            manifest[f[1]] = f
    for ds, items in SOURCES.items():
        if args.only and ds not in args.only:
            continue
        ddir = RAW / ds
        ddir.mkdir(parents=True, exist_ok=True)
        sums = []
        for url, fname, extract in items:
            dest = ddir / fname
            rel = f"{ds}/{fname}"
            if dest.exists() and not args.force:
                print(f"[skip] {rel} exists")
                retrieved = manifest.get(rel, [None] * 6)[5] or today
            else:
                print(f"[get ] {url}")
                download(url, dest)
                retrieved = today
            digest = sha256(dest)
            sums.append(f"{digest}  {fname}")
            manifest[rel] = [ds, rel, url, str(dest.stat().st_size), digest, retrieved]
            if extract:
                n = safe_extract(dest, ddir / "extracted" / dest.stem)
                print(f"       extracted {n} files from {fname}")
        (ddir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with open(manifest_path, "w") as fh:
        fh.write("dataset\tpath\turl\tbytes\tsha256\tretrieved\n")
        for k in sorted(manifest):
            fh.write("\t".join(manifest[k]) + "\n")
    print(f"manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
