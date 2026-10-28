"""Feature-store helpers following docs/INTERFACES.md exactly.

Layout under <data root>/features/ (data root = $PSL_DATA_ROOT or data/processed):
  genes.txt                       global gene order (all PomBase protein-coding genes, sorted)
  gene/<GROUP>.parquet            index = systematic ID (subset of genes.txt), cols `<GROUP>__<name>`, float32
  sim/<source>.npy                float32 G x G, symmetric, NaN where either gene lacks data, NaN diagonal
  pair/<GROUP>__<name>.npy        float32 G x G, symmetric, NaN = missing (diagonal NaN as well)
  manifest.json                   {relative path: {group, sources, script, sha256, ...}}

All writers validate shape / symmetry / dtype before saving and update the manifest under a file lock.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    p = Path(os.environ.get("PSL_DATA_ROOT", "data/processed"))
    return p if p.is_absolute() else ROOT / p


def feat_dir() -> Path:
    return data_root() / "features"


def genes(path: Path | None = None) -> list[str]:
    path = path or feat_dir() / "genes.txt"
    return [g for g in Path(path).read_text().split("\n") if g]


def gene_index(gl: list[str] | None = None) -> dict[str, int]:
    gl = gl or genes()
    return {g: i for i, g in enumerate(gl)}


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
def update_manifest(key: str, entry: dict, fdir: Path | None = None) -> None:
    fdir = fdir or feat_dir()
    fdir.mkdir(parents=True, exist_ok=True)
    mpath = fdir / "manifest.json"
    with open(fdir / ".manifest.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        man = json.loads(mpath.read_text()) if mpath.exists() else {}
        man[key] = entry
        tmp = mpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(sorted(man.items())), indent=1) + "\n")
        tmp.replace(mpath)
        fcntl.flock(lk, fcntl.LOCK_UN)


def _entry(group, sources, script, path, **extra) -> dict:
    assert isinstance(sources, (list, tuple)) and sources, "sources must be a non-empty list"
    return {"group": group, "sources": sorted(set(sources)), "script": script, "sha256": sha256(path),
            "built": _dt.datetime.now().isoformat(timespec="seconds"), **extra}


def save_gene_table(group: str, df: pd.DataFrame, sources, script: str, column_sources: dict | None = None,
                    descriptions: dict | None = None, source_files: list | None = None) -> Path:
    """df: index = systematic IDs, columns = short names (prefixed here with `<GROUP>__`)."""
    gl = set(genes())
    bad = set(df.index) - gl
    assert not bad, f"{len(bad)} index IDs not in genes.txt, e.g. {sorted(bad)[:3]}"
    assert df.index.is_unique
    df = df.copy()
    df.columns = [c if c.startswith(f"{group}__") else f"{group}__{c}" for c in df.columns]
    df = df.astype("float32").sort_index()
    df.index.name = "gene"
    out = feat_dir() / "gene" / f"{group}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    rel = f"gene/{group}.parquet"
    cols = {c: {"n_nonmissing": int(df[c].notna().sum()),
                **({"sources": column_sources[c.split('__', 1)[1]]} if column_sources and
                   c.split('__', 1)[1] in column_sources else {}),
                **({"description": descriptions[c.split('__', 1)[1]]} if descriptions and
                   c.split('__', 1)[1] in descriptions else {})}
            for c in df.columns}
    update_manifest(rel, _entry(group, sources, script, out, kind="gene", columns=cols,
                                source_files=source_files or []))
    return out


def check_square(M: np.ndarray, G: int, name: str, sim: bool) -> None:
    assert M.shape == (G, G), f"{name}: shape {M.shape} != ({G},{G})"
    assert M.dtype == np.float32, f"{name}: dtype {M.dtype}"
    d = np.diagonal(M)
    assert np.isnan(d).all(), f"{name}: diagonal must be NaN"
    # symmetric incl. NaN pattern
    for s in range(0, G, 1024):
        a, b = M[s:s + 1024], M[:, s:s + 1024].T
        assert np.array_equal(a, b, equal_nan=True), f"{name}: not symmetric (rows {s}..)"
    if sim:
        v = M[np.isfinite(M)]
        assert v.size == 0 or (v.min() >= -1 - 1e-6 and v.max() <= 1 + 1e-6), f"{name}: sim outside [-1,1]"


def symmetrize_inplace(M: np.ndarray) -> np.ndarray:
    """Make exactly symmetric (upper triangle copied to lower) and set NaN diagonal."""
    iu = np.triu_indices(M.shape[0], 1)
    M[(iu[1], iu[0])] = M[iu]
    np.fill_diagonal(M, np.nan)
    return M


def save_matrix(kind: str, name: str, M: np.ndarray, group: str, sources, script: str,
                description: str = "", source_files: list | None = None, **extra) -> Path:
    """kind = 'pair' (name '<GROUP>__<feature>') or 'sim' (name = similarity source id)."""
    assert kind in ("pair", "sim")
    if kind == "pair":
        assert name.startswith(f"{group}__"), f"pair feature must be named {group}__<name>"
    G = len(genes())
    M = np.ascontiguousarray(M, dtype=np.float32)
    np.fill_diagonal(M, np.nan)
    check_square(M, G, name, sim=(kind == "sim"))
    out = feat_dir() / kind / f"{name}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, M)
    iu = np.triu_indices(G, 1)
    n_pairs = int(np.isfinite(M[iu]).sum())
    n_genes = int(np.isfinite(M).any(axis=1).sum())
    update_manifest(f"{kind}/{name}.npy", _entry(group, sources, script, out, kind=kind,
                                                  description=description, n_pairs_nonmissing=n_pairs,
                                                  n_genes_with_any=n_genes,
                                                  source_files=source_files or [], **extra))
    return out


def load_matrix(kind: str, name: str, mmap: bool = True) -> np.ndarray:
    return np.load(feat_dir() / kind / f"{name}.npy", mmap_mode="r" if mmap else None)


# ---------------------------------------------------------------------------------------------
class RuntimeLog:
    """with RuntimeLog('ppi') as rt: ... -> results/runtime/features_<name>.json"""

    def __init__(self, name: str, **info):
        self.name, self.info, self.steps = name, info, {}

    def __enter__(self):
        self.t0 = time.time()
        self.c0 = time.process_time()
        self._last = self.t0
        return self

    def step(self, label: str):
        now = time.time()
        self.steps[label] = round(now - self._last, 2)
        self._last = now

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / 1e6 if platform.system() == "Darwin" else rss / 1e3
        res_dir = Path(os.environ.get("PSL_RESULTS_ROOT", "results"))
        res_dir = res_dir if res_dir.is_absolute() else ROOT / res_dir
        out = res_dir / "runtime" / f"features_{self.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "task": f"features_{self.name}", "wall_s": round(time.time() - self.t0, 2),
            "cpu_s": round(time.process_time() - self.c0, 2), "peak_rss_mb": round(rss_mb, 1),
            "steps_s": self.steps, "hardware": {"machine": platform.machine(), "system": platform.system(),
                                                "cpu_count": os.cpu_count(), "python": platform.python_version()},
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "finished": _dt.datetime.now().isoformat(timespec="seconds"), **self.info}, indent=1) + "\n")
        return False
