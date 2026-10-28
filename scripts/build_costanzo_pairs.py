#!/usr/bin/env python
"""Collapse Costanzo et al. 2016 (Data File S1) S. cerevisiae SGA scores to one row per ORF pair.

Input : data/raw/costanzo2016/extracted/Data_File_S1_pairwise/<folder>/SGA_{NxN,ExN,ExE,DAmP}.txt
        (hsp90_data.txt is a special HSP90 experiment and is NOT used.)
Output: data/processed/costanzo_measurements_summary.tsv  per-file row counts
        data/processed/costanzo_pairs.parquet  one row per unordered ORF pair:
            orf_a < orf_b, eps_mean (mean over all strain-level measurements: both orientations,
            multiple alleles/strains, 26/30 C arrays), eps_min, eps_max, n_meas, pval_min,
            pval_of_min_eps, sources (which files contributed)
        data/processed/costanzo_strains.tsv  strain -> ORF, allele, SMF (copied fields + strain class)
Self pairs (same ORF as query and array) are dropped.

Aggregation choice (mean over all measurements) mirrors the Ryan handling; it is a modelling input,
not a label, so the PI can revisit it later (per-file / per-allele values remain derivable from raw).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data/raw/costanzo2016/extracted/Data_File_S1_pairwise"
OUT = ROOT / "data/processed"
FILES = ["SGA_NxN.txt", "SGA_ExN.txt", "SGA_ExE.txt", "SGA_DAmP.txt"]
COLS = {"Query Strain ID": "qs", "Array Strain ID": "as", "Arraytype/Temp": "arraytype",
        "Genetic interaction score (ε)": "eps", "P-value": "p"}


def find(name: str) -> Path:
    hits = list(DIR.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {hits}")
    return hits[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts, summary = [], []
    for fname in FILES:
        df = pd.read_csv(find(fname), sep="\t", usecols=list(COLS), dtype={"P-value": "float64"},
                         encoding="utf-8").rename(columns=COLS)
        df["orf_q"] = df.qs.str.rsplit("_", n=1).str[0]
        df["orf_r"] = df["as"].str.rsplit("_", n=1).str[0]
        n0 = len(df)
        df = df[df.orf_q != df.orf_r]
        df["source"] = fname.replace("SGA_", "").replace(".txt", "")
        summary.append(dict(file=fname, rows=n0, self_pairs_dropped=n0 - len(df),
                            query_strains=df.qs.nunique(), array_strains=df["as"].nunique(),
                            eps_missing=int(df.eps.isna().sum())))
        a = np.where(df.orf_q < df.orf_r, df.orf_q, df.orf_r)
        b = np.where(df.orf_q < df.orf_r, df.orf_r, df.orf_q)
        parts.append(pd.DataFrame({"orf_a": a, "orf_b": b, "eps": df.eps.values, "p": df.p.values,
                                   "source": df.source.values}))
        print(f"read {fname}: {n0} rows")
    pd.DataFrame(summary).to_csv(OUT / "costanzo_measurements_summary.tsv", sep="\t", index=False)
    allm = pd.concat(parts, ignore_index=True)
    allm = allm.dropna(subset=["eps"])
    allm["source"] = allm.source.astype("category")
    g = allm.groupby(["orf_a", "orf_b"], sort=False)
    res = g.agg(eps_mean=("eps", "mean"), eps_min=("eps", "min"), eps_max=("eps", "max"),
                n_meas=("eps", "size"), pval_min=("p", "min"))
    idx = g.eps.idxmin()
    res["pval_of_min_eps"] = allm.p.loc[idx.values].values
    names = [f.replace("SGA_", "").replace(".txt", "") for f in FILES]
    allm["bit"] = allm.source.map({n: 1 << i for i, n in enumerate(names)}).astype("int64")
    bits = allm.drop_duplicates(["orf_a", "orf_b", "bit"]).groupby(["orf_a", "orf_b"], sort=False).bit.sum()
    decode = {m: ",".join(n for i, n in enumerate(names) if m >> i & 1) for m in range(1, 1 << len(names))}
    res["sources"] = bits.map(decode)
    res = res.reset_index()
    res.to_parquet(OUT / "costanzo_pairs.parquet", index=False)
    print(f"costanzo_pairs: {len(res)} ORF pairs from {len(allm)} measurements")

    st = pd.read_excel(find("strain_ids_and_single_mutant_fitness.xlsx"))
    st["strain_class"] = st["Strain ID"].str.rsplit("_", n=1).str[1].str.extract(r"^([a-z]+)", expand=False)
    st.to_csv(OUT / "costanzo_strains.tsv", sep="\t", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
