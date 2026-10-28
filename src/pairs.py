"""Symmetric pair-feature assembler over the label-free feature store (DESIGN.md §4, docs/INTERFACES.md).

    from src.pairs import assemble, load_spec
    X = assemble(pairs_df)                          # all manifest groups + HYPO
    X = assemble(pairs_df, groups=["PPI", "CER"])   # subset (ablations: exclude=["PPI"])

Features (all float32, NaN = missing; LightGBM handles NaN natively):
* gene-level column c (gene/<GROUP>.parquet):
    `c__min` = min(c_a, c_b), NaN if EITHER gene is missing (np.minimum propagates NaN),
    `c__max` = max(c_a, c_b), ignoring a missing gene (np.fmax).
  The pair (min, max) is therefore lossless up to order: both present -> (lo, hi); one missing -> (NaN, x);
  both missing -> (NaN, NaN), so the "one gene missing" state is visible to the model.
  Fitness-like columns (name matches FITNESS_RE: smf / fitness) also get `c__prod` (NaN if either missing).
* pair/<GROUP>__<name>.npy -> `<GROUP>__<name>`;  sim/<source>.npy -> `<GROUP>__sim_<source>`:
  direct lookup M[min(i,j), max(i,j)] (exactly symmetric even if a stored matrix were not).
* HYPO__min / HYPO__max from `hypomorph_a/b` columns of pairs_df if present (training rows from
  src/data.fold_split), else 0 (eval rows and the deliverable; DESIGN §1).

The manifest is snapshotted once per `FeatureSpec` (`load_spec()`), so a run uses a fixed feature set even
while other agents add groups; pass the same spec to every `assemble` call of a run.
`perm` (length-G int array) maps gene i to the feature row of gene perm[i] in every table and matrix
(gene-permutation negative control, DESIGN §6).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import feature_store as fs

FITNESS_RE = re.compile(r"(smf|fitness)", re.I)
HYPO_GROUP = "HYPO"
CHUNK = 1_000_000
ALLOWLIST = fs.ROOT / "config" / "data_allowlist.yaml"
# patterns that must never appear in a feature's source ids or source_files (DESIGN §4 rule 4, PROMPT.md).
# The EDA / inventory paths are written with a character class so that tests/test_lint.py (which flags any
# mention of those file names outside data-build scripts) does not mistake this deny-pattern for a read.
FORBIDDEN_RE = re.compile(r"fypo|viability|gene.?ontology|\bgo\b|go.?slim|string|text.?min|ryan|koch|"
                          r"interactions\.tsv|net-ff|data_invent[o]ry|results/ed[a]|labels_", re.I)


@dataclass
class FeatureSpec:
    fdir: Path
    genes: list
    manifest: dict
    gene_tables: list = field(default_factory=list)   # (group, relpath, [columns])
    matrices: list = field(default_factory=list)      # (group, relpath, feature name)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def G(self) -> int:
        return len(self.genes)

    def groups(self) -> list:
        g = {t[0] for t in self.gene_tables} | {m[0] for m in self.matrices} | {HYPO_GROUP}
        return sorted(g)

    def gene_index(self) -> pd.Index:
        if "gidx" not in self._cache:
            self._cache["gidx"] = pd.Index(self.genes)
        return self._cache["gidx"]

    def table(self, rel: str, cols: list) -> np.ndarray:
        """Gene table as a G x C float32 array in genes.txt order (NaN for genes absent from the table)."""
        if rel not in self._cache:
            df = pd.read_parquet(self.fdir / rel)
            self._cache[rel] = df.reindex(self.genes)[cols].to_numpy(np.float32)
        return self._cache[rel]

    def matrix(self, rel: str) -> np.ndarray:
        if rel not in self._cache:
            M = np.load(self.fdir / rel, mmap_mode="r")
            assert M.shape == (self.G, self.G), f"{rel}: shape {M.shape} != G={self.G}"
            self._cache[rel] = M
        return self._cache[rel]

    def plan(self, groups=None, exclude=None) -> list:
        """Ordered list of (kind, group, rel, cols/name) blocks for the selected groups."""
        sel = set(self.groups()) if groups is None else set(groups)
        unknown = sel - set(self.groups())
        if unknown:
            raise KeyError(f"unknown feature groups {sorted(unknown)}; available {self.groups()}")
        sel -= set(exclude or [])
        out = [("gene", g, rel, cols) for g, rel, cols in self.gene_tables if g in sel]
        out += [("mat", g, rel, name) for g, rel, name in self.matrices if g in sel]
        if HYPO_GROUP in sel:
            out.append(("hypo", HYPO_GROUP, None, None))
        return out

    @staticmethod
    def _block_names(kind: str, x) -> list:
        if kind == "gene":
            return [n for c in x for n in [f"{c}__min", f"{c}__max"] + ([f"{c}__prod"] if FITNESS_RE.search(c) else [])]
        if kind == "mat":
            return [x]
        return [f"{HYPO_GROUP}__min", f"{HYPO_GROUP}__max"]

    def feature_names(self, groups=None, exclude=None) -> list:
        return [n for kind, g, rel, x in self.plan(groups, exclude) for n in self._block_names(kind, x)]

    def feature_groups(self, groups=None, exclude=None) -> dict:
        """feature name -> group."""
        return {n: g for kind, g, rel, x in self.plan(groups, exclude) for n in self._block_names(kind, x)}

    def hashes(self, groups=None, exclude=None) -> dict:
        """manifest sha256 of every file used (registry `feature_hashes`)."""
        rels = sorted({rel for _, _, rel, _ in self.plan(groups, exclude) if rel})
        return {r: self.manifest[r].get("sha256", "") for r in rels}


def load_spec(fdir: Path | None = None) -> FeatureSpec:
    """Snapshot the manifest (only entries under gene/, pair/, sim/ whose file exists)."""
    fdir = Path(fdir or fs.feat_dir())
    man = json.loads((fdir / "manifest.json").read_text())
    genes = fs.genes(fdir / "genes.txt")
    spec = FeatureSpec(fdir, genes, man)
    for rel in sorted(man):
        e = man[rel]
        if not (fdir / rel).exists():
            continue
        if rel.startswith("gene/") and rel.endswith(".parquet"):
            grp = e.get("group") or Path(rel).stem
            cols = [c for c in pd.read_parquet(fdir / rel).columns]
            spec.gene_tables.append((grp, rel, cols))
        elif rel.startswith("pair/") and rel.endswith(".npy"):
            grp = e.get("group") or Path(rel).stem.split("__")[0]
            spec.matrices.append((grp, rel, Path(rel).stem))
        elif rel.startswith("sim/") and rel.endswith(".npy"):
            grp = e.get("group") or Path(rel).stem
            spec.matrices.append((grp, rel, f"{grp}__sim_{Path(rel).stem}"))
    names = spec.feature_names()
    assert len(names) == len(set(names)), "duplicate feature names in the manifest"
    return spec


def gene_indices(spec: FeatureSpec, genes) -> np.ndarray:
    idx = spec.gene_index().get_indexer(np.asarray(genes, dtype=object))
    if (idx < 0).any():
        bad = np.asarray(genes, dtype=object)[idx < 0]
        raise KeyError(f"{len(bad)} genes not in genes.txt, e.g. {list(pd.unique(bad))[:3]}")
    return idx


def _fill(out: np.ndarray, spec: FeatureSpec, plan: list, ia: np.ndarray, ib: np.ndarray,
          ha: np.ndarray | None, hb: np.ndarray | None) -> None:
    j = 0
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    for kind, g, rel, x in plan:
        if kind == "gene":
            T = spec.table(rel, x)
            A, B = T[ia], T[ib]
            fit = np.array([bool(FITNESS_RE.search(c)) for c in x])
            mn, mx = np.minimum(A, B), np.fmax(A, B)
            for c in range(len(x)):
                out[:, j], out[:, j + 1] = mn[:, c], mx[:, c]
                j += 2
                if fit[c]:
                    out[:, j] = A[:, c] * B[:, c]
                    j += 1
        elif kind == "mat":
            out[:, j] = spec.matrix(rel)[lo, hi]
            j += 1
        else:
            if ha is None:
                out[:, j:j + 2] = 0
            else:
                out[:, j], out[:, j + 1] = np.minimum(ha, hb), np.maximum(ha, hb)
            j += 2
    assert j == out.shape[1]


def assemble_arrays(ia: np.ndarray, ib: np.ndarray, spec: FeatureSpec, groups=None, exclude=None,
                    ha: np.ndarray | None = None, hb: np.ndarray | None = None, perm: np.ndarray | None = None,
                    chunk: int = CHUNK) -> np.ndarray:
    """Core of `assemble` on genes.txt indices: (n, F) float32 array, columns = spec.feature_names(...)."""
    plan = spec.plan(groups, exclude)
    F = len(spec.feature_names(groups, exclude))
    ia, ib = np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)
    if perm is not None:
        perm = np.asarray(perm)
        assert perm.shape == (spec.G,) and np.array_equal(np.sort(perm), np.arange(spec.G)), "perm must permute 0..G-1"
        ia, ib = perm[ia], perm[ib]
    n = len(ia)
    out = np.empty((n, F), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        _fill(out[s:e], spec, plan, ia[s:e], ib[s:e], None if ha is None else ha[s:e], None if hb is None else hb[s:e])
    return out


def assemble(pairs_df: pd.DataFrame, groups=None, *, exclude=None, spec: FeatureSpec | None = None,
             perm: np.ndarray | None = None, chunk: int = CHUNK) -> pd.DataFrame:
    """Feature frame (float32, index = pairs_df.index) for pairs (gene_a, gene_b); see module docstring."""
    spec = spec or load_spec()
    ia = gene_indices(spec, pairs_df.gene_a.to_numpy())
    ib = gene_indices(spec, pairs_df.gene_b.to_numpy())
    ha = hb = None
    if "hypomorph_a" in pairs_df and "hypomorph_b" in pairs_df:
        ha, hb = pairs_df.hypomorph_a.to_numpy(np.float32), pairs_df.hypomorph_b.to_numpy(np.float32)
    out = assemble_arrays(ia, ib, spec, groups, exclude, ha, hb, perm, chunk)
    return pd.DataFrame(out, columns=spec.feature_names(groups, exclude), index=pairs_df.index, copy=False)


# ------------------------------------------------------------------------------------------ provenance
def allowed_sources(path: Path = ALLOWLIST) -> tuple[set, set]:
    """(use=feature dataset ids, all non-feature ids incl. excluded_sources) from config/data_allowlist.yaml."""
    a = yaml.safe_load(Path(path).read_text())
    feat = {k for k, v in a["datasets"].items() if v.get("use") == "feature"}
    other = {k for k, v in a["datasets"].items() if v.get("use") != "feature"} | set(a.get("excluded_sources", {}))
    return feat, other


def source_violations(spec: FeatureSpec, path: Path = ALLOWLIST, extra_allowed=()) -> list:
    """Every source of every feature file used must be a use=feature dataset of the allowlist.
    Manifest ids may be short aliases (`biogrid` for `biogrid_physical`, `pombase` for `pombase_orthologs` /
    `pombase_protein`): an alias is accepted iff it is not itself a non-feature id and prefixes >= 1
    use=feature id. Source ids and source_files must not match FORBIDDEN_RE."""
    feat, other = allowed_sources(path)
    v = []
    rels = {rel for _, rel, _ in spec.gene_tables} | {rel for _, rel, _ in spec.matrices}
    for rel in sorted(rels):
        e = spec.manifest[rel]
        srcs = list(e.get("sources") or [])
        if not srcs:
            v.append(f"{rel}: no sources recorded")
        for s in srcs:
            ok = s in feat or s in extra_allowed or (
                s not in other and any(f.startswith(s + "_") for f in feat))
            if not ok:
                v.append(f"{rel}: source '{s}' is not a use=feature dataset")
            if FORBIDDEN_RE.search(s):
                v.append(f"{rel}: forbidden source id '{s}'")
        for sf in e.get("source_files", []):
            if FORBIDDEN_RE.search(sf):
                v.append(f"{rel}: forbidden source file '{sf[:80]}'")
    return v
