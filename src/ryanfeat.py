"""Ryan-derived (label-dependent) features for query pairs, computed ONLY from a given set of training
pairs (DESIGN.md §4 RYAN-fold + NBR, rules 2-3; §5 B1/B4).

    from src.ryanfeat import build_ryan_features, SimStore
    sims = SimStore()                                   # label-free similarity matrices (feature store)
    F = build_ryan_features(train_pairs, query_pairs, sims, cfg)   # one row per query pair, same order

`train_pairs`: gene_a, gene_b, y (bool label), s_score (Ryan S2) -- e.g. src.data.fold_split(...)[0] or
an inner-fold training set from src.data.inner_splits. `query_pairs`: gene_a, gene_b (any order; every
feature is symmetric in (a, b)). Nothing about the query pairs except their gene IDs is used, and every
statistic is a function of `train_pairs` (+ label-free similarities) only; the features of a query
pair do not depend on which other pairs are queried (tested in tests/test_ryanfeat.py).

Features (all rates / scores, never counts; NaN = undefined):
* seen_min, seen_max                  gene appears in a training pair (0/1)
* b1_logit, beta_min, beta_max        B1: L2-shrunk joint logistic fit y ~ c + beta_a + beta_b on sparse
                                      gene indicators; unseen genes beta = 0 (b1_logit = c + beta_a + beta_b)
* mf_score                            B4: symmetric logistic MF c + b_a + b_b + <u_a, u_b> (torch);
                                      unseen genes b = 0, u = 0
* prof_corr                           Pearson r of the training S profiles of a and b over shared measured
                                      partners (NaN if < cfg.min_overlap shared partners or a gene unseen)
* negs_mean_min/max                   gene mean of -S over its training pairs (NaN if unseen)
* nbr_<src>__<f>                      NBR similarity transfer per sim source (DESIGN §4 rule 3):
    W = rank-transformed similarity (global percentile of the source's off-diagonal finite values) where
    the raw similarity is > 0, else 0 (max(sim, 0)); NaN -> 0; self excluded. Each gene keeps its top-k
    neighbours among query-role TRAINING genes and, separately, its top-k among array-role TRAINING genes
    (union). Roles are Ryan design metadata (ryan_genes.tsv via src.data) restricted to the genes of the
    training pairs; used ONLY to choose neighbour pools, never as a feature (rule 1).
    With Y = training hit matrix (or -S), M = measured mask, lam, p0 = global training mean of Y:
      dir(a->b) = (Y W^T + lam p0)/(M W^T + lam) at (a, b)   -> hit_dmax/hit_dmean, negs_dmax/negs_dmean
                                                                (nan-max / nan-mean of the two directions)
      block     = W Y W^T / W M W^T                          -> hit_block, negs_block (NaN if 0/0)
      support   = M W^T / sum(W)  (per direction)            -> support_max, support_mean
      generate  = sum_c W[g,c] r_c / sum_c W[g,c] (r_c = training hit rate of neighbour c)
                                                             -> generate_min, generate_max
  plus nbr_mean__<f> = nan-mean over sources.
  -S is used for the S features so that, like the hit features, higher = more negative interaction.

The building blocks are methods of `RyanFeatureBuilder`; tests subclass it to plant leaky mutants.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import protocol

FEATURE_SET_VERSION = "ryanfeat-v1"
DEFAULT_CFG = {
    "k": None,              # NBR neighbours per role pool (None -> protocol nbr.k_default)
    "lam": None,            # NBR shrinkage (None -> protocol nbr.lambda)
    # B1 / MF values tuned once on outer fold 0 / inner fold 0 (results/baselines/tuning.json)
    "b1_C": 1.0,            # sklearn inverse L2 strength for B1
    "mf": True,
    "mf_k": 32,
    "mf_wd": 1e-2,
    "mf_epochs": 50,
    "mf_lr": 0.05,
    "mf_seed": 0,
    "min_overlap": 20,
    "profile": True,
    "sims": None,           # list of sim source names (None -> all in the SimStore)
    "roles": "ryan_genes",  # 'ryan_genes' (src.data roles) | None (one pool of all training genes)
}


def resolve_cfg(cfg: dict | None = None) -> dict:
    c = {**DEFAULT_CFG, **(cfg or {})}
    nb = protocol.load()["nbr"]
    if c["k"] is None:
        c["k"] = int(nb["k_default"])
    if c["lam"] is None:
        c["lam"] = float(nb["lambda"])
    return c


def cfg_hash(cfg: dict, sims: "SimStore | None" = None) -> str:
    c = resolve_cfg(cfg)
    payload = {"v": FEATURE_SET_VERSION, "cfg": {k: v for k, v in c.items() if k != "sims"},
               "sims": sims.hashes(c["sims"]) if sims is not None else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


# =========================================================================================== similarities
class SimStore:
    """Label-free similarity matrices (docs/INTERFACES.md sim/<source>.npy, float32 G x G, NaN = missing).

    Default: every sim/*.npy listed in the feature-store manifest (sha256 recorded) under the data root,
    minus names matching `exclude` (the HTP-only PPI network is a report-only ablation, DESIGN App. C).
    `SimStore.from_arrays(genes, {name: matrix})` builds one in memory (tests)."""

    def __init__(self, root: Path | None = None, names=None, exclude=("_htp",)):
        root = Path(root or protocol.data_root()) / "features"
        self.genes = [g for g in (root / "genes.txt").read_text().split() if g]
        self.pos = pd.Series(np.arange(len(self.genes)), index=self.genes)
        man = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
        files = {p.stem: p for p in sorted((root / "sim").glob("*.npy"))}
        if names is None:
            names = [n for n in files if not any(e in n for e in exclude)]
        self._files = {n: files[n] for n in names}
        self._sha = {n: man.get(f"sim/{n}.npy", {}).get("sha256") for n in names}
        self._arr = {}

    @classmethod
    def from_arrays(cls, genes, mats: dict) -> "SimStore":
        self = cls.__new__(cls)
        self.genes = list(genes)
        self.pos = pd.Series(np.arange(len(self.genes)), index=self.genes)
        self._files, self._arr = {}, {}
        self._sha = {}
        for n, m in mats.items():
            m = np.asarray(m, dtype=np.float32).copy()
            np.fill_diagonal(m, np.nan)
            self._arr[n] = m
            self._sha[n] = hashlib.sha256(m.tobytes()).hexdigest()
        return self

    @property
    def names(self) -> list:
        return sorted(set(self._files) | set(self._arr))

    def hashes(self, names=None) -> dict:
        out = {}
        for n in (names or self.names):
            if self._sha.get(n) is None:
                self._sha[n] = protocol.sha256_file(self._files[n])
            out[n] = self._sha[n]
        return out

    def matrix(self, name):
        if name not in self._arr:
            m = np.load(self._files[name], mmap_mode="r")
            assert m.shape == (len(self.genes), len(self.genes)), (name, m.shape)
            self._arr[name] = m
        return self._arr[name]

    def grid(self, name) -> np.ndarray:
        return _quantile_grid(id(self), name, self)

    def sub(self, name, genes) -> np.ndarray:
        """Raw similarity among `genes` (NaN for genes absent from genes.txt), float32."""
        p = self.pos.reindex(genes).to_numpy()
        ok = ~np.isnan(p)
        out = np.full((len(genes), len(genes)), np.nan, dtype=np.float32)
        idx = p[ok].astype(np.int64)
        o = np.flatnonzero(ok)
        if len(idx):
            m = self.matrix(name)
            order = np.argsort(idx)                    # sorted row reads from the memmap
            blk = np.asarray(m[idx[order]][:, idx[order]], dtype=np.float32)
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            out[np.ix_(o, o)] = blk[np.ix_(inv, inv)]
        return out

    def weights(self, name, genes) -> np.ndarray:
        """NBR weight among `genes`: global percentile of the similarity where the raw similarity is > 0,
        else 0; NaN -> 0; diagonal 0 (self excluded)."""
        s = self.sub(name, genes)
        g = self.grid(name)
        fin = np.isfinite(s)
        w = np.zeros_like(s)
        w[fin] = np.searchsorted(g, s[fin], side="right") / len(g)
        w[~fin | (s <= 0)] = 0.0
        np.fill_diagonal(w, 0.0)
        return w


_GRID_CACHE: dict = {}


def _quantile_grid(store_id, name, store, n_grid: int = 100_001) -> np.ndarray:
    """Quantiles of all finite off-diagonal values of the full matrix (label-free; global percentile)."""
    key = (store_id, name)
    if key not in _GRID_CACHE:
        m = store.matrix(name)
        vals = []
        for r0 in range(0, m.shape[0], 1024):
            blk = np.asarray(m[r0:r0 + 1024], dtype=np.float32)
            vals.append(blk[np.isfinite(blk)])
        v = np.sort(np.concatenate(vals))
        _GRID_CACHE[key] = v[np.linspace(0, len(v) - 1, min(n_grid, len(v))).round().astype(np.int64)]
    return _GRID_CACHE[key]


@lru_cache(maxsize=2)
def _design_roles(root: str) -> pd.Series | None:
    from src import data
    return data._roles(root)


# =========================================================================================== helpers
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _clean_train(train: pd.DataFrame) -> pd.DataFrame:
    t = train[["gene_a", "gene_b", "y", "s_score"]].copy()
    t["y"] = t.y.astype(bool)
    a, b = t.gene_a.to_numpy(object), t.gene_b.to_numpy(object)
    swap = a > b
    t["gene_a"], t["gene_b"] = np.where(swap, b, a), np.where(swap, a, b)
    return t.reset_index(drop=True)


def _dense_pairs(n, ia, ib, vals, dtype=np.float32) -> np.ndarray:
    Mx = np.zeros((n, n), dtype=dtype)
    Mx[ia, ib] = vals
    Mx[ib, ia] = vals
    return Mx


# =========================================================================================== B1 / MF
def fit_b1(train: pd.DataFrame, C: float) -> tuple[float, pd.Series]:
    """Joint L2-shrunk additive-logit propensity: logit P(y) = c + beta_a + beta_b (intercept unpenalised)."""
    from sklearn.linear_model import LogisticRegression
    genes = pd.Index(np.unique(np.r_[train.gene_a.to_numpy(object), train.gene_b.to_numpy(object)]))
    ia, ib = genes.get_indexer(train.gene_a), genes.get_indexer(train.gene_b)
    n = len(train)
    X = sp.csr_matrix((np.ones(2 * n), (np.r_[np.arange(n), np.arange(n)], np.r_[ia, ib])),
                      shape=(n, len(genes)))
    y = train.y.to_numpy(bool)
    if y.all() or not y.any():
        return float(_logit(y.mean())), pd.Series(0.0, index=genes)
    with np.errstate(all="ignore"):         # spurious FP flags from macOS Accelerate BLAS
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-6).fit(X, y)
    return float(m.intercept_[0]), pd.Series(m.coef_[0], index=genes)


def fit_mf(train: pd.DataFrame, k: int, wd: float, epochs: int, lr: float, seed: int,
           valid: tuple | None = None, eval_every: int = 25) -> dict:
    """Symmetric logistic MF with gene biases: logit = c + b_a + b_b + <u_a, u_b>. Full-batch Adam,
    loss = mean BCE + wd * mean_pairs(|u_a|^2 + |u_b|^2 + b_a^2 + b_b^2). Deterministic given `seed`.
    valid = (gene_a, gene_b, y) arrays -> also returns AP every `eval_every` epochs (tuning only)."""
    import torch
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    genes = pd.Index(np.unique(np.r_[train.gene_a.to_numpy(object), train.gene_b.to_numpy(object)]))
    ia = torch.as_tensor(genes.get_indexer(train.gene_a))
    ib = torch.as_tensor(genes.get_indexer(train.gene_b))
    y = torch.as_tensor(train.y.to_numpy(np.float32))
    g = torch.Generator().manual_seed(seed)
    U = (0.01 * torch.randn(len(genes), k, generator=g)).requires_grad_()
    b = torch.zeros(len(genes), requires_grad=True)
    c = torch.tensor(float(_logit(train.y.mean())), requires_grad=True)
    opt = torch.optim.Adam([U, b, c], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    hist = []
    if valid is not None:
        va, vb = genes.get_indexer(valid[0]), genes.get_indexer(valid[1])
        vok = (va >= 0) & (vb >= 0)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        ua, ub = U[ia], U[ib]
        z = c + b[ia] + b[ib] + (ua * ub).sum(1)
        reg = ((ua * ua).sum(1) + (ub * ub).sum(1) + b[ia] ** 2 + b[ib] ** 2).mean()
        loss = lossf(z, y) + wd * reg
        loss.backward()
        opt.step()
        if valid is not None and (ep % eval_every == 0 or ep == epochs):
            with torch.no_grad():
                P = _mf_params(genes, U, b, c)
                s = mf_score(P, valid[0][vok], valid[1][vok])
            hist.append((ep, float(average_precision_score(valid[2][vok], s))))
    P = _mf_params(genes, U, b, c)
    P["history"] = hist
    return P


def _mf_params(genes, U, b, c) -> dict:
    return {"genes": genes, "U": U.detach().numpy().copy(), "b": b.detach().numpy().copy(), "c": float(c.detach())}


def mf_score(P: dict, ga, gb) -> np.ndarray:
    """MF logit for pairs; unseen genes contribute bias 0 and embedding 0."""
    ia, ib = P["genes"].get_indexer(np.asarray(ga, object)), P["genes"].get_indexer(np.asarray(gb, object))
    U = np.vstack([P["U"], np.zeros((1, P["U"].shape[1]), np.float32)])
    bb = np.r_[P["b"], 0.0]
    ia, ib = np.where(ia < 0, len(bb) - 1, ia), np.where(ib < 0, len(bb) - 1, ib)
    return P["c"] + (bb[ia] + bb[ib]) + (U[ia] * U[ib]).sum(1)     # bracketed: exactly symmetric


# =========================================================================================== builder
class RyanFeatureBuilder:
    """Composes all Ryan-derived features for `query` pairs from `train` pairs only.
    Overridable hooks (used ONLY by leakage tests to plant mutants): `stat_pairs`, `pool_genes`,
    `exclude_self`, `gene_nbr_rate`."""

    exclude_self = True

    def __init__(self, train: pd.DataFrame, query: pd.DataFrame, sims: SimStore | None, cfg: dict | None = None):
        self.cfg = resolve_cfg(cfg)
        self.train = _clean_train(train)
        self.query = query[["gene_a", "gene_b"]].reset_index(drop=True)
        self.sims = sims
        self.info: dict = {}

    # ---------------------------------------------------------------- hooks
    def stat_pairs(self, component: str) -> pd.DataFrame:
        """Labelled pairs feeding `component` in {b1, mf, profile, nbr, rate}. Always the training pairs."""
        return self.train

    def pool_genes(self, role: str, train_genes: set) -> set:
        """Candidate neighbour genes for role in {query, array}: training genes with that role."""
        roles = self._roles()
        if roles is None:
            return set(train_genes)
        r = roles.reindex(sorted(train_genes)).fillna("both")
        ok = r.isin([role, "both"])
        return set(r.index[ok])

    def gene_nbr_rate(self, W: sp.csr_matrix, rate: np.ndarray, idx: pd.Index) -> np.ndarray:
        sw = np.asarray(W.sum(1)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sw > 0, (W @ rate) / sw, np.nan)

    # ---------------------------------------------------------------- internals
    def _roles(self):
        r = self.cfg.get("roles")
        if r is None:
            return None
        if isinstance(r, pd.Series):
            return r
        return _design_roles(str(protocol.data_root()))

    def build(self) -> pd.DataFrame:
        c = self.cfg
        q = self.query
        tr = self.train
        tg = set(tr.gene_a) | set(tr.gene_b)
        out = {}
        qa, qb = q.gene_a.to_numpy(object), q.gene_b.to_numpy(object)
        sa, sb = np.isin(qa, list(tg)).astype(np.float32), np.isin(qb, list(tg)).astype(np.float32)
        out["seen_min"], out["seen_max"] = np.minimum(sa, sb), np.maximum(sa, sb)

        # B1
        c0, beta = fit_b1(_clean_train(self.stat_pairs("b1")), c["b1_C"])
        ba, bb = beta.reindex(qa).fillna(0.0).to_numpy(), beta.reindex(qb).fillna(0.0).to_numpy()
        out["b1_logit"] = c0 + (ba + bb)                                 # exactly symmetric
        out["beta_min"], out["beta_max"] = np.minimum(ba, bb), np.maximum(ba, bb)
        self.info["b1"] = {"c": c0, "beta": beta}

        # MF
        if c["mf"]:
            P = fit_mf(_clean_train(self.stat_pairs("mf")), c["mf_k"], c["mf_wd"], c["mf_epochs"], c["mf_lr"],
                       c["mf_seed"])
            out["mf_score"] = mf_score(P, qa, qb)
            self.info["mf"] = P

        # gene index for matrix features: training genes + query genes
        idx = pd.Index(sorted(tg | set(qa) | set(qb)))
        n = len(idx)
        ia_q, ib_q = idx.get_indexer(qa), idx.get_indexer(qb)

        # gene mean -S
        st = _clean_train(self.stat_pairs("rate"))
        long = pd.concat([st[["gene_a", "s_score", "y"]].rename(columns={"gene_a": "g"}),
                          st[["gene_b", "s_score", "y"]].rename(columns={"gene_b": "g"})])
        gm = long.groupby("g").agg(negs=("s_score", lambda s: -s.mean()), rate=("y", "mean"))
        na_, nb_ = gm.negs.reindex(qa).to_numpy(), gm.negs.reindex(qb).to_numpy()
        out["negs_mean_min"], out["negs_mean_max"] = np.fmin(na_, nb_), np.fmax(na_, nb_)
        with np.errstate(invalid="ignore"):
            out["negs_mean_min"] = np.where(np.isnan(na_) | np.isnan(nb_), np.nan, out["negs_mean_min"])

        # profile correlation
        if c["profile"]:
            out["prof_corr"] = self._profile_corr(_clean_train(self.stat_pairs("profile")), idx, ia_q, ib_q)

        # NBR
        names = c["sims"] if c["sims"] is not None else (self.sims.names if self.sims is not None else [])
        if names:
            nt = _clean_train(self.stat_pairs("nbr"))
            ta, tb = idx.get_indexer(nt.gene_a), idx.get_indexer(nt.gene_b)
            assert (ta >= 0).all() and (tb >= 0).all()
            Y = _dense_pairs(n, ta, tb, nt.y.to_numpy(np.float32))
            NS = _dense_pairs(n, ta, tb, -nt.s_score.to_numpy(np.float32))
            M = _dense_pairs(n, ta, tb, np.float32(1))
            p0, p0s = float(nt.y.mean()), float(-nt.s_score.mean())
            rate = gm.rate.reindex(idx).fillna(0.0).to_numpy()
            pools = {r: idx.get_indexer(sorted(self.pool_genes(r, tg))) for r in ("query", "array")}
            per = {}
            for s in names:
                W = self._neighbour_matrix(self.sims.weights(s, list(idx)), pools)
                f = self._nbr(W, Y, NS, M, p0, p0s, rate, idx, ia_q, ib_q)
                self.info.setdefault("W", {})[s] = W
                for k_, v in f.items():
                    out[f"nbr_{s}__{k_}"] = v
                    per.setdefault(k_, []).append(v)
            with np.errstate(invalid="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for k_, vs in per.items():
                        out[f"nbr_mean__{k_}"] = np.nanmean(np.vstack(vs), axis=0)
            self.info["idx"] = idx
            self.info["pools"] = pools
        F = pd.DataFrame({k_: np.asarray(v, dtype=np.float32) for k_, v in out.items()})
        F.insert(0, "gene_b", qb)
        F.insert(0, "gene_a", qa)
        return F

    def _neighbour_matrix(self, Wd: np.ndarray, pools: dict) -> sp.csr_matrix:
        """Top-k per row within each role pool (union); self excluded unless a mutant says otherwise."""
        k = self.cfg["k"]
        n = Wd.shape[0]
        if self.exclude_self:
            np.fill_diagonal(Wd, 0.0)
        rows, cols, vals = [], [], []
        for pool in pools.values():
            if len(pool) == 0:
                continue
            sub = Wd[:, pool]
            kk = min(k, len(pool))
            top = np.argpartition(-sub, kk - 1, axis=1)[:, :kk]
            v = np.take_along_axis(sub, top, 1)
            r = np.repeat(np.arange(n), kk)
            m = v.ravel() > 0
            rows.append(r[m])
            cols.append(pool[top.ravel()[m]])
            vals.append(v.ravel()[m])
        W = sp.csr_matrix((np.concatenate(vals) if vals else [], (np.concatenate(rows) if rows else [],
                                                                     np.concatenate(cols) if cols else [])),
                          shape=(n, n), dtype=np.float64)
        W.sum_duplicates()
        W = W.tocoo()
        # a gene in both pools may be selected twice: keep its weight once (union, not sum)
        Wd_vals = Wd[W.row, W.col]
        return sp.csr_matrix((Wd_vals.astype(np.float64), (W.row, W.col)), shape=(n, n))

    def _nbr(self, W, Y, NS, M, p0, p0s, rate, idx, ia, ib) -> dict:
        lam = self.cfg["lam"]
        sw = np.asarray(W.sum(1)).ravel()
        out = {}
        MWt = np.asarray((W @ M).T)            # M symmetric: M W^T = (W M)^T
        WMWt = np.asarray(W @ MWt)             # W M W^T
        with np.errstate(invalid="ignore", divide="ignore"):
            for nm, Yx, pp in (("hit", Y, p0), ("negs", NS, p0s)):
                YWt = np.asarray((W @ Yx).T)
                D = (YWt + lam * pp) / (MWt + lam)
                dab, dba = D[ia, ib], D[ib, ia]
                out[f"{nm}_dmax"] = np.fmax(dab, dba)
                out[f"{nm}_dmean"] = np.nanmean(np.vstack([dab, dba]), axis=0)
                WYWt = np.asarray(W @ YWt)
                den = WMWt[ia, ib]
                out[f"{nm}_block"] = np.where(den > 0, WYWt[ia, ib] / np.where(den > 0, den, 1), np.nan)
            sab = np.where(sw[ib] > 0, MWt[ia, ib] / np.where(sw[ib] > 0, sw[ib], 1), np.nan)
            sba = np.where(sw[ia] > 0, MWt[ib, ia] / np.where(sw[ia] > 0, sw[ia], 1), np.nan)
            out["support_max"] = np.fmax(sab, sba)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["support_mean"] = np.nanmean(np.vstack([sab, sba]), axis=0)
            gr = self.gene_nbr_rate(W, rate, idx)
            out["generate_min"] = np.where(np.isnan(gr[ia]) | np.isnan(gr[ib]), np.nan, np.fmin(gr[ia], gr[ib]))
            out["generate_max"] = np.fmax(gr[ia], gr[ib])
        return out

    def _profile_corr(self, st: pd.DataFrame, idx: pd.Index, ia, ib) -> np.ndarray:
        """Pearson r of training S rows over shared measured partners (pairwise complete)."""
        g = pd.Index(sorted(set(st.gene_a) | set(st.gene_b)))
        ta, tb = g.get_indexer(st.gene_a), g.get_indexer(st.gene_b)
        m = len(g)
        S = _dense_pairs(m, ta, tb, st.s_score.to_numpy(np.float64), np.float64)
        Mk = _dense_pairs(m, ta, tb, 1.0, np.float64)
        with np.errstate(all="ignore"):     # spurious FP flags from macOS Accelerate matmul
            N = Mk @ Mk.T
            Sa = S @ Mk.T              # sum of a's S over partners shared with b
            Saa = (S * S) @ Mk.T
            Sab = S @ S.T
        qa_, qb_ = g.get_indexer(idx[ia]), g.get_indexer(idx[ib])
        ok = (qa_ >= 0) & (qb_ >= 0)
        r = np.full(len(ia), np.nan)
        x, z = qa_[ok], qb_[ok]
        n_ = N[x, z]
        num = n_ * Sab[x, z] - Sa[x, z] * Sa[z, x]
        den = (n_ * Saa[x, z] - Sa[x, z] ** 2) * (n_ * Saa[z, x] - Sa[z, x] ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = num / np.sqrt(den)
        rr[(n_ < self.cfg["min_overlap"]) | ~(den > 0)] = np.nan
        r[ok] = rr
        return r


def build_ryan_features(train_pairs_with_labels: pd.DataFrame, query_pairs: pd.DataFrame,
                        sims: SimStore | None, cfg: dict | None = None) -> pd.DataFrame:
    """Ryan-derived features for `query_pairs` computed only from `train_pairs_with_labels` (module doc)."""
    return RyanFeatureBuilder(train_pairs_with_labels, query_pairs, sims, cfg).build()


def feature_columns(F: pd.DataFrame) -> list:
    return [c for c in F.columns if c not in ("gene_a", "gene_b")]
