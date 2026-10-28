"""PomBase curated S. pombe -> S. cerevisiae orthologs (pombase-2026-10-01,
exports/pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz), pombe IDs passed through src.idmap."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from src.idmap import IDMapper

ROOT = Path(__file__).resolve().parent.parent
ORTH = ROOT / "data/raw/pombase/pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz"
SOURCE_FILE = "pombase:exports/pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz"


def load_orthologs(mapper: IDMapper, path: Path = ORTH) -> pd.DataFrame:
    """Return unique (pombe, sc) rows; pombe = current protein-coding systematic ID."""
    with gzip.open(path, "rt") as fh:
        rows = [l.rstrip("\n").split("\t") for l in fh if l.strip() and not l.startswith("#")]
    df = pd.DataFrame([r[:2] for r in rows], columns=["pombe_raw", "sc"])
    df["sc"] = df.sc.str.strip()
    m = mapper.map(df.pombe_raw, source="pombase_orthologs")
    df["pombe"] = df.pombe_raw.map(m)
    df = df.dropna(subset=["pombe"])
    df = df[df.sc.str.len() > 0].drop_duplicates(["pombe", "sc"])
    return df[["pombe", "sc"]].reset_index(drop=True)


def incidence(orth: pd.DataFrame, genes: list[str], orfs: list[str]) -> np.ndarray:
    """G x C float32 0/1 matrix: O[i, x] = 1 iff cerevisiae ORF x is a curated ortholog of gene i."""
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: j for j, c in enumerate(orfs)}
    O = np.zeros((len(genes), len(orfs)), dtype=np.float32)
    sub = orth[orth.pombe.isin(gi) & orth.sc.isin(ci)]
    O[sub.pombe.map(gi).values, sub.sc.map(ci).values] = 1.0
    return O
