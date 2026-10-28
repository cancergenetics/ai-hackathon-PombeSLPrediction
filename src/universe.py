"""Deliverable helpers (DESIGN.md §1 universe, §9 deliverable): gene universe, pair enumeration, gene
coordinates / linkage flag, class of a pair relative to a training set. Label-free; never reads labels.

    from src import universe as U
    genes, flags = U.universe_genes()              # sorted gene list + per-gene flags
    for ia, ib in U.pair_chunks(n, chunk): ...     # all i<j index pairs of 0..n-1, in row-major order
    coords = U.load_coords()                       # gene -> chrom, start, end (PomBase)
    same, gap = U.pair_distance(coords, ga, gb)    # same chromosome?, gap between gene bodies (bp)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import protocol

ROOT = protocol.ROOT
CFG_PATH = ROOT / "config" / "deliverable.yaml"


@lru_cache(maxsize=2)
def _cfg(path: str, mtime: float) -> dict:
    return yaml.safe_load(Path(path).read_text())


def deliverable_cfg(path: Path | None = None) -> dict:
    p = Path(path or CFG_PATH)
    return _cfg(str(p), p.stat().st_mtime)


def is_real_data() -> bool:
    """True iff the data root's splits are the real ones (data/processed/splits, also via symlinked stores
    such as data/frozen_m2)."""
    d = protocol.data_root() / protocol.load()["paths"]["splits_subdir"]
    real = ROOT / protocol.load()["paths"]["data_root_default"] / protocol.load()["paths"]["splits_subdir"]
    return d.exists() and real.exists() and d.resolve() == real.resolve()


# ============================================================================================ universe
def universe_genes(feature_genes=None, include_unknown: bool | None = None) -> tuple[list, pd.DataFrame]:
    """(sorted genes to score, per-gene frame [gene, in_universe, viability_unknown]).
    Real data: universe_genes.tsv `in_universe` genes, plus (if include_unknown) the `viability_unknown_flag`
    genes. Toy data (no universe_genes.tsv): every feature-store gene, none flagged.
    Genes absent from the feature store (`feature_genes`) are dropped (they cannot be featurised)."""
    if include_unknown is None:
        include_unknown = deliverable_cfg()["universe"]["include_viability_unknown"]
    f = protocol.data_root() / "universe_genes.tsv"
    if f.exists():
        u = pd.read_csv(f, sep="\t")
        unk = u.viability_unknown_flag.astype(bool) if "viability_unknown_flag" in u else pd.Series(False, u.index)
        keep = u.in_universe.astype(bool) | (unk & include_unknown)
        df = pd.DataFrame({"gene": u.gene[keep].values, "in_universe": u.in_universe[keep].astype(bool).values,
                           "viability_unknown": unk[keep].values})
    else:
        if feature_genes is None:
            raise FileNotFoundError(f"{f} missing and no feature-store genes given")
        df = pd.DataFrame({"gene": list(feature_genes), "in_universe": True, "viability_unknown": False})
    if feature_genes is not None:
        df = df[df.gene.isin(set(feature_genes))]
    df = df.sort_values("gene").drop_duplicates("gene").reset_index(drop=True)
    return df.gene.tolist(), df


def n_pairs(n: int) -> int:
    return n * (n - 1) // 2


def pair_chunks(n: int, chunk: int):
    """Yield (ia, ib) int64 arrays of all i<j pairs of range(n), row-major (i ascending, then j), about
    `chunk` pairs per block (whole rows; a row is never split). Concatenation = np.triu_indices(n, 1)."""
    A, B, m = [], [], 0
    for r in range(n - 1):
        k = n - r - 1
        A.append(np.full(k, r, dtype=np.int64))
        B.append(np.arange(r + 1, n, dtype=np.int64))
        m += k
        if m >= chunk:
            yield np.concatenate(A), np.concatenate(B)
            A, B, m = [], [], 0
    if m:
        yield np.concatenate(A), np.concatenate(B)


def covering_pairs(genes) -> pd.DataFrame:
    """A small pair set in which every gene of `genes` appears (consecutive genes paired, last gene with
    the first). Appended to every Ryan-feature query block so that all blocks share the same gene index
    (identical fitted state; see src/deliverable.py)."""
    g = list(genes)
    if len(g) < 2:
        raise ValueError("need >= 2 genes")
    a = np.array(g, dtype=object)
    b = np.roll(a, -1)
    lo, hi = np.where(a < b, a, b), np.where(a < b, b, a)
    out = pd.DataFrame({"gene_a": lo, "gene_b": hi}).drop_duplicates().reset_index(drop=True)
    return out


# ============================================================================================ genome
def coords_dir() -> Path:
    p = Path(deliverable_cfg()["genome"]["coords_dir"])
    return p if p.is_absolute() else ROOT / p


def download_coords(force: bool = False) -> Path:
    """Fetch PomBase gene coordinate files (pinned release) into coords_dir and verify sha256."""
    import requests
    G = deliverable_cfg()["genome"]
    d = coords_dir()
    d.mkdir(parents=True, exist_ok=True)
    for c in G["chromosomes"]:
        f = d / f"chromosome_{c}.gene.coords.tsv"
        if force or not f.exists():
            r = requests.get(G["url_template"].format(chrom=c), timeout=120)
            r.raise_for_status()
            f.write_bytes(r.content)
        want = G["sha256"].get(f.name)
        got = protocol.sha256_file(f)
        if want and got != want:
            raise RuntimeError(f"{f}: sha256 {got} != pinned {want}")
    (d / "SHA256SUMS").write_text("".join(f"{protocol.sha256_file(d / n)}  {n}\n" for n in sorted(G["sha256"])))
    return d


def load_coords(d: Path | None = None) -> pd.DataFrame:
    """gene -> chrom (int), start, end (bp, 1-based inclusive). Raises if the files are absent."""
    d = Path(d or coords_dir())
    G = deliverable_cfg()["genome"]
    parts = []
    for c in G["chromosomes"]:
        f = d / f"chromosome_{c}.gene.coords.tsv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run scripts/linkage_diagnostic.py --download")
        x = pd.read_csv(f, sep="\t", header=None, names=["gene", "start", "end", "strand"])
        x["chrom"] = int(c)
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    lo, hi = np.minimum(df.start, df.end), np.maximum(df.start, df.end)
    df["start"], df["end"] = lo, hi
    return df.drop_duplicates("gene").set_index("gene")[["chrom", "start", "end"]]


def gene_coord_arrays(coords: pd.DataFrame, genes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(chrom, start, end) for `genes`; chrom = 0 and start/end = -1 where a gene has no coordinates."""
    c = coords.reindex(list(genes))
    ok = c.chrom.notna().to_numpy()
    chrom = np.where(ok, c.chrom.fillna(0).to_numpy(), 0).astype(np.int8)
    start = np.where(ok, c.start.fillna(-1).to_numpy(), -1).astype(np.int64)
    end = np.where(ok, c.end.fillna(-1).to_numpy(), -1).astype(np.int64)
    return chrom, start, end


def gap_from_arrays(chrom, start, end, ia, ib) -> tuple[np.ndarray, np.ndarray]:
    """(same chromosome (bool), gap in bp between the two gene bodies; 0 if overlapping, -1 if not on the
    same chromosome or a gene lacks coordinates)."""
    same = (chrom[ia] == chrom[ib]) & (chrom[ia] > 0)
    gap = np.maximum(np.maximum(start[ia], start[ib]) - np.minimum(end[ia], end[ib]), 0)
    return same, np.where(same, gap, -1)


def pair_distance(coords: pd.DataFrame, ga, gb) -> tuple[np.ndarray, np.ndarray]:
    genes = pd.Index(pd.unique(np.r_[np.asarray(ga, object), np.asarray(gb, object)]))
    chrom, start, end = gene_coord_arrays(coords, genes)
    return gap_from_arrays(chrom, start, end, genes.get_indexer(ga), genes.get_indexer(gb))


def linked_window_bp() -> int:
    return int(round(float(deliverable_cfg()["linkage"]["window_kb"]) * 1000))


# ============================================================================================ OOF scores
def latest_run_id(model: str, split_seed: int | None = None, clean: bool = True) -> str | None:
    """Most recent registered run of `model` on `split_seed` (default: primary) in results/registry.tsv
    (any commit; clean runs only unless clean=False)."""
    f = protocol.results_root() / "registry.tsv"
    if not f.exists():
        return None
    reg = pd.read_csv(f, sep="\t", dtype=str)
    s = protocol.load()["dev_cv"]["split_seed"] if split_seed is None else int(split_seed)
    m = (reg.model == model) & (pd.to_numeric(reg.split_seed, errors="coerce") == s)
    if clean:
        m &= reg.git_dirty.astype(str) == "False"
    return reg.loc[m, "run_id"].iloc[-1] if m.any() else None


def oof_scores(run_id: str) -> pd.DataFrame:
    """Out-of-fold dev-CV scores of a registered run (preds/<run_id>.parquet: seed-averaged scores, no
    labels): one row per pair, score averaged over the folds in which the pair was an eval row (a C2 pair is
    evaluated in the folds of both genes), n_folds = that count."""
    p = pd.read_parquet(protocol.preds_dir() / f"{run_id}.parquet", columns=["gene_a", "gene_b", "score"])
    return p.groupby(["gene_a", "gene_b"], sort=False).score.agg(oof_score="mean", oof_n_folds="size").reset_index()


def lockbox_gene_list() -> list:
    """Lockbox gene IDs (no labels): config/lockbox_genes.txt (sha256-checked) on the real data, else the toy
    root's lockbox_genes.txt (drawn by scripts/make_splits.py)."""
    L = protocol.load()["lockbox"]
    if is_real_data():
        f = ROOT / L["genes_file"]
        if protocol.sha256_file(f) != L["genes_sha256"]:
            raise RuntimeError(f"{f}: sha256 mismatch with protocol.yaml")
    else:
        f = protocol.data_root() / "lockbox_genes.txt"
    return [g for g in f.read_text().split() if g]


# ============================================================================================ classes
def pair_class(seen_a: np.ndarray, seen_b: np.ndarray) -> np.ndarray:
    """C1 both genes seen in training, C2 exactly one, C3 neither (Park & Marcotte 2012)."""
    n = np.asarray(seen_a, bool).astype(np.int8) + np.asarray(seen_b, bool).astype(np.int8)
    return np.array(["C3", "C2", "C1"], dtype=object)[n]
