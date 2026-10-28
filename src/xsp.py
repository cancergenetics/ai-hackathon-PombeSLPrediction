"""Cross-species transfer (XSP) helpers: species-agnostic gene and pair features, computed the same way for
S. cerevisiae (training, Costanzo 2016 labels) and S. pombe (application).  No file I/O, no labels here;
used by scripts/features/build_xsp.py and unit-tested in tests/features/test_xsp.py.

Species-agnostic inputs (both species): ESM-2 650M mean-pooled embeddings, protein sequence statistics,
physical PPI network topology, within-species paralogy measured by ESM similarity.

SINaTRA-style alignment (Jacunski et al. 2015): every gene-level feature is rank-normalised (percentile, ties
averaged) WITHIN its species; continuous pair features are mapped through the species' own empirical CDF over
random gene pairs (mid-rank for ties).  So a value means "position within this species' distribution".
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFVW")
DISORDER_PROMOTING = set("ARGQSPEK")
RESTART = 0.5
SP_CAP = 4
_QUIET = dict(over="ignore", divide="ignore", invalid="ignore")   # spurious Apple Accelerate matmul flags


# ------------------------------------------------------------------------------------------- sequences
def read_fasta(path: Path, id_fn) -> dict[str, str]:
    """Read a (gzipped) FASTA; id_fn(header_line_without_'>') -> gene ID (or None to skip)."""
    op = gzip.open if str(path).endswith(".gz") else open
    seqs: dict[str, list[str]] = {}
    cur = None
    with op(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                cur = id_fn(line[1:])
                if cur is not None:
                    assert cur not in seqs, f"duplicate FASTA id {cur}"
                    seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip())
    return {k: "".join(v).rstrip("*").upper() for k, v in seqs.items()}


def seq_stats(seqs: dict[str, str], genes: list[str]) -> pd.DataFrame:
    """Per-gene protein sequence statistics (NaN if the gene has no sequence)."""
    rows = {}
    for g in genes:
        s = seqs.get(g)
        if not s:
            continue
        s = s.replace("*", "")
        n = len(s)
        cnt = {a: s.count(a) for a in AA}
        tot = max(1, sum(cnt.values()))
        r = {"log_len": np.log10(n)}
        r.update({f"aa_{a}": cnt[a] / tot for a in AA})
        r["hydrophobic"] = sum(cnt[a] for a in HYDROPHOBIC) / tot
        r["disorder_prom"] = sum(cnt[a] for a in DISORDER_PROMOTING) / tot
        r["net_charge"] = (cnt["K"] + cnt["R"] - cnt["D"] - cnt["E"]) / tot
        r["charged"] = (cnt["K"] + cnt["R"] + cnt["D"] + cnt["E"]) / tot
        rows[g] = r
    return pd.DataFrame.from_dict(rows, orient="index").reindex(genes)


# ------------------------------------------------------------------------------------------- normalisation
def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank per column within this table (= within species); ties averaged; NaN kept."""
    return df.rank(method="average", pct=True, na_option="keep").astype(np.float32)


@dataclass
class ECDF:
    """Empirical CDF of a pair feature over random within-species gene pairs; mid-rank for ties."""
    sample: np.ndarray

    @classmethod
    def from_matrix(cls, M: np.ndarray, n: int = 2_000_000, seed: int = 0) -> "ECDF":
        G = M.shape[0]
        rng = np.random.default_rng(seed)
        i = rng.integers(0, G, n)
        j = rng.integers(0, G, n)
        ok = i != j
        v = np.asarray(M[i[ok], j[ok]], dtype=np.float64)
        v = np.sort(v[np.isfinite(v)])
        return cls(v)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full(x.shape, np.nan, dtype=np.float32)
        ok = np.isfinite(x)
        lo = np.searchsorted(self.sample, x[ok], side="left")
        hi = np.searchsorted(self.sample, x[ok], side="right")
        out[ok] = ((lo + hi) / (2.0 * len(self.sample))).astype(np.float32)
        return out


def ecdf_matrix(M: np.ndarray, seed: int = 0, chunk: int = 1024) -> np.ndarray:
    """Map a symmetric G x G matrix through its own within-species ECDF (keeps symmetry and NaNs)."""
    f = ECDF.from_matrix(M, seed=seed)
    out = np.empty(M.shape, dtype=np.float32)
    for s in range(0, M.shape[0], chunk):
        out[s:s + chunk] = f(M[s:s + chunk])
    np.fill_diagonal(out, np.nan)
    return out


# ------------------------------------------------------------------------------------------- ESM
def esm_joint_pca(E: dict[str, np.ndarray], n_pc: int = 32) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """E: species -> (n_s x d) embeddings.  Each species is centred on its OWN mean (removes the species
    offset), then one PCA basis is fitted on the stacked, equally weighted species.  Returns per-species PC
    scores and the explained-variance ratios."""
    C = {k: v.astype(np.float64) - v.astype(np.float64).mean(0) for k, v in E.items()}
    w = {k: 1.0 / np.sqrt(len(v)) for k, v in C.items()}            # equal total weight per species
    X = np.vstack([C[k] * w[k] for k in C])
    with np.errstate(**_QUIET):
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        evr = sv ** 2 / (sv ** 2).sum()
        return {k: (v @ Vt[:n_pc].T).astype(np.float32) for k, v in C.items()}, evr[:n_pc]


def centred_cosine(X: np.ndarray) -> np.ndarray:
    """Cosine similarity of species-centred embeddings, float32 n x n, NaN diagonal."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    V = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    with np.errstate(**_QUIET):
        S = np.clip(V @ V.T, -1, 1).astype(np.float32)
    S = (S + S.T) / 2
    np.fill_diagonal(S, np.nan)
    return S


def paralog_features(S: np.ndarray, thr: float = 0.5) -> pd.DataFrame:
    """Within-species paralogy from the ESM cosine matrix S (NaN diagonal): best hit, mean of top 5,
    # genes above thr."""
    A = np.where(np.isnan(S), -np.inf, S)
    top = -np.sort(-A, axis=1)[:, :5]
    return pd.DataFrame({"para_top1": top[:, 0], "para_top5": top.mean(1),
                         "para_n50": (A > thr).sum(1).astype(np.float64)})


def mutual_rank(S: np.ndarray) -> np.ndarray:
    """1 - log(MR)/log(n-1), MR = sqrt(rank_ij rank_ji) by descending S (float32, NaN diagonal)."""
    n = S.shape[0]
    A = np.where(np.isnan(S), -np.inf, S)
    np.fill_diagonal(A, -np.inf)
    order = np.argsort(-A, axis=1, kind="stable")
    R = np.empty((n, n), dtype=np.float32)
    R[np.arange(n)[:, None], order] = np.arange(1, n + 1, dtype=np.float32)[None, :]
    with np.errstate(**_QUIET):
        out = 1.0 - np.log(np.sqrt(R.astype(np.float64) * R.T)) / np.log(n - 1)
    out = np.clip(out, 0, 1).astype(np.float32)
    np.fill_diagonal(out, np.nan)
    return out


# ------------------------------------------------------------------------------------------- PPI
def rwr_sim(A: np.ndarray, restart: float = RESTART) -> np.ndarray:
    """Symmetric random-walk-with-restart similarity (as scripts/features/build_ppi.py); NaN for isolated
    genes and on the diagonal."""
    G = A.shape[0]
    deg = A.sum(1)
    nz = np.where(deg > 0)[0]
    S = np.full((G, G), np.nan, dtype=np.float32)
    d = deg[nz].astype(np.float64)
    with np.errstate(**_QUIET):
        K = np.linalg.inv(np.diag(d) - (1 - restart) * A[np.ix_(nz, nz)].astype(np.float64))
    K = (K + K.T) / 2
    sq = np.sqrt(d)
    S[np.ix_(nz, nz)] = np.clip(restart * sq[:, None] * K * sq[None, :], 0, 1).astype(np.float32)
    np.fill_diagonal(S, np.nan)
    return S


def ppi_gene_features(A: np.ndarray) -> pd.DataFrame:
    """degree, PageRank, clustering coefficient, k-core number, mean neighbour degree (0 for isolated)."""
    G = A.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(G))
    g.add_edges_from(map(tuple, np.argwhere(np.triu(A, 1) > 0)))
    deg = A.sum(1).astype(np.float64)
    pr = nx.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500)
    cl = nx.clustering(g)
    core = nx.core_number(g)
    with np.errstate(**_QUIET):
        nbr = np.where(deg > 0, (A @ deg) / np.maximum(deg, 1), 0.0)
    return pd.DataFrame({"ppi_degree": deg, "ppi_pagerank": [pr[i] * G for i in range(G)],
                         "ppi_clustering": [cl[i] for i in range(G)], "ppi_core": [core[i] for i in range(G)],
                         "ppi_nbr_degree": nbr})


def ppi_pair_matrices(A: np.ndarray) -> dict[str, np.ndarray]:
    """edge, shared neighbours, Jaccard (NaN if either isolated), capped shortest path (1..4, 5 = longer /
    disconnected, NaN if either isolated), RWR similarity.  All float32 G x G, NaN diagonal."""
    A = A.astype(np.float32)
    deg = A.sum(1)
    iso = deg == 0
    with np.errstate(**_QUIET):
        shared = A @ A
    union = deg[:, None] + deg[None, :] - shared
    with np.errstate(**_QUIET):
        jac = np.where(union > 0, shared / np.maximum(union, 1), 0).astype(np.float32)
    jac[iso, :] = np.nan
    jac[:, iso] = np.nan
    sp = np.full(A.shape, SP_CAP + 1, dtype=np.float32)
    reach = A > 0
    sp[reach] = 1
    Rk = reach.astype(np.float32)
    for k in range(2, SP_CAP + 1):
        with np.errstate(**_QUIET):
            Rk = ((Rk @ A) > 0).astype(np.float32)
        sp[(Rk > 0) & (sp > k)] = k
    sp[iso, :] = np.nan
    sp[:, iso] = np.nan
    out = {"edge": A.copy(), "shared": shared.astype(np.float32), "jaccard": jac, "sp": sp, "rwr": rwr_sim(A)}
    for M in out.values():
        np.fill_diagonal(M, np.nan)
    return out


def subsample_edges(A: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Keep each undirected edge independently with probability frac (seeded)."""
    iu = np.argwhere(np.triu(A, 1) > 0)
    keep = np.random.default_rng(seed).random(len(iu)) < frac
    B = np.zeros_like(A)
    e = iu[keep]
    B[e[:, 0], e[:, 1]] = 1
    B[e[:, 1], e[:, 0]] = 1
    return B


# ------------------------------------------------------------------------------------------- pair assembly
@dataclass
class Species:
    """All species-agnostic inputs of one species, already within-species normalised.
    gene: n x k DataFrame (index = gene IDs in matrix order, rank-normalised columns);
    pair: name -> n x n float32 matrix (ECDF-mapped or raw categorical), symmetric, NaN diagonal."""
    name: str
    genes: list[str]
    gene: pd.DataFrame
    pair: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genes)}


def pair_feature_names(gene_cols: list[str], pair_names: list[str]) -> list[str]:
    return [f"{c}__min" for c in gene_cols] + [f"{c}__max" for c in gene_cols] + [f"pair_{p}" for p in pair_names]


def assemble(sp: Species, i: np.ndarray, j: np.ndarray, gene_cols: list[str], pair_names: list[str],
             extra_gene: np.ndarray | None = None) -> np.ndarray:
    """Symmetric pair design matrix for index pairs (i, j): per gene column min and max (NaN-aware:
    min is NaN if either gene is missing, max ignores NaN), then direct lookups of pair matrices.
    extra_gene: optional n x e array appended to the gene columns (e.g. fold-specific propensity
    predictions); its names must be the trailing entries of gene_cols."""
    X = sp.gene[[c for c in gene_cols if c in sp.gene.columns]].to_numpy(np.float32)
    if extra_gene is not None:
        X = np.hstack([X, extra_gene.astype(np.float32)])
    assert X.shape[1] == len(gene_cols), (X.shape, len(gene_cols))
    a, b = X[i], X[j]
    mn = np.minimum(a, b)                                  # NaN propagates
    with np.errstate(**_QUIET):
        mx = np.fmax(a, b)                                 # NaN ignored
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    P = np.column_stack([sp.pair[p][lo, hi] for p in pair_names]) if pair_names else np.empty((len(i), 0))
    return np.hstack([mn, mx, P.astype(np.float32)]).astype(np.float32)


# ------------------------------------------------------------------------------------------- targets
def smoothed_rate(neg: np.ndarray, tested: np.ndarray, prior: float, strength: float = 20.0) -> np.ndarray:
    """(neg + strength*prior) / (tested + strength); NaN where tested == 0."""
    neg = np.asarray(neg, float)
    tested = np.asarray(tested, float)
    return np.where(tested > 0, (neg + strength * prior) / (tested + strength), np.nan)


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def gene_folds(genes: list[str], groups: list[tuple[str, str]], k: int = 5, seed: int = 0) -> np.ndarray:
    """Random gene fold ids (0..k-1); genes linked by `groups` pairs (e.g. ohnologs) share a fold."""
    idx = {g: n for n, g in enumerate(genes)}
    parent = list(range(len(genes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in groups:
        if a in idx and b in idx:
            ra, rb = find(idx[a]), find(idx[b])
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(x) for x in range(len(genes))])
    uniq = np.unique(roots)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_root = dict(zip(uniq, perm % k))
    return np.array([fold_of_root[r] for r in roots], dtype=np.int64)
