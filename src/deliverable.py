"""Full-training-set retrain + scoring of arbitrary pairs / the whole universe (DESIGN.md §2 "final models",
§9 deliverable). Shared by scripts/predict_all_pairs.py and scripts/lockbox_eval.py.

    from src import deliverable as DL
    spec = DL.model_spec("M2")                       # config/models.yaml (+ toy_overrides off the real data)
    m = DL.TrainedModel(spec, train_pairs).fit()     # train_pairs: gene_a, gene_b, y, s_score (+ flags)
    s = m.predict(pairs_df)                          # scores for any pairs (chunked)
    for ia, ib, s in m.score_universe(genes): ...    # all i<j pairs of the sorted gene list, streamed

M2 (kind m2_stack) is trained exactly as one outer fold of scripts/run_m2.py, with the outer training set
replaced by `train_pairs` (nested stacking, DESIGN §4 rule 2):
  * T = train_pairs -> data.drop_noncoding;  inner = data.inner_splits(T)  (5 inner gene folds + C1 hold-out);
  * stacker TRAINING rows = inner eval rows; their RYAN/NBR/GP features come from inner fold j's training
    pairs only (src.stacking.inner_rows with the M2 builder);
  * PREDICTION rows (universe / lockbox pairs) get the same features computed from ALL of T
    (the builder's fit on T: B1, MF, profiles, NBR neighbour matrices, GP regressor);
  * seed s: LightGBM with early stopping on inner fold j = s mod 5, fitted on the gene-grouped rows for j;
    the score is the mean raw score over seeds (as src/evaluate averages seeds).
Universe scale: the Ryan builder is a function of (T, query genes) whose per-pair output does not depend on
which other pairs are queried. Every query block is padded with `covering_pairs(cover genes)` so that all
blocks share one gene index (hence one identical fit, bit-identical features across blocks; checked at run
time on the padding rows), and label-free features are assembled in smaller sub-blocks.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import data, protocol
from src import pairs as PA
from src import universe as U

MODELS_PATH = protocol.ROOT / "config" / "models.yaml"
META = {"gene_a", "gene_b", "y", "s_score", "part", "inner_fold", "hypomorph_a", "hypomorph_b", "w", "fold"}


@contextlib.contextmanager
def deterministic_torch(threads: int):
    """Bit-reproducible torch for the MF fit inside the Ryan builder: with several threads the CPU backward of
    the embedding gather accumulates in a non-deterministic order (MF scores then differ by ~1e-6 between two
    identical fits). Restores the previous global settings on exit."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        yield
        return
    old_det, old_thr = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_det)
        torch.set_num_threads(old_thr)


# ============================================================================================ specs
def models_sha256(path: Path | None = None) -> str:
    return protocol.sha256_file(Path(path or MODELS_PATH))


def model_spec(model_id: str, path: Path | None = None, real: bool | None = None) -> dict:
    """Resolved spec of `model_id` from config/models.yaml; toy_overrides applied off the real data."""
    M = yaml.safe_load(Path(path or MODELS_PATH).read_text())["models"]
    if model_id not in M:
        raise KeyError(f"unknown model id {model_id!r}; known: {sorted(M)}")
    s = copy.deepcopy(M[model_id])
    real = U.is_real_data() if real is None else real
    if not real:
        s.update(s.get("toy_overrides") or {})
    s.pop("toy_overrides", None)
    s["id"] = model_id
    seeds = s.get("seeds", "ladder")
    s["seeds"] = [int(x) for x in (protocol.load()["seeds"][seeds] if isinstance(seeds, str) else seeds)]
    return s


def resolve_groups(spec: dict, lfspec: PA.FeatureSpec) -> list:
    g = spec.get("lf_groups", "all")
    if g == "all":
        return lfspec.groups()
    missing = set(g) - set(lfspec.groups())
    if missing:
        raise KeyError(f"model {spec['id']}: feature groups {sorted(missing)} not in the store {lfspec.groups()}")
    return list(g)


def train_hash(T: pd.DataFrame) -> str:
    cols = [c for c in ("gene_a", "gene_b", "y", "s_score", "hypomorph_a", "hypomorph_b", "eval_eligible") if c in T]
    return hashlib.sha256(pd.util.hash_pandas_object(T[cols], index=False).values.tobytes()).hexdigest()


# ============================================================================================ model
class TrainedModel:
    """One model spec retrained on one training pair set. `fit()` then `predict()` / `score_universe()`."""

    def __init__(self, spec: dict, train_pairs: pd.DataFrame, threads: int = 3, sims=None,
                 lfspec: PA.FeatureSpec | None = None, cover_genes=None, use_cache: bool = True, log=print):
        self.spec, self.threads, self.log = spec, int(threads), log
        self.lfspec = lfspec or PA.load_spec()
        self.kind = spec["kind"]
        T = train_pairs.reset_index(drop=True)
        self.noncoding = None
        if spec.get("drop_noncoding", True):
            T = data.drop_noncoding(T)
            self.noncoding = T.attrs.get("noncoding_dropped")
        self.T = T
        self.train_genes = set(T.gene_a) | set(T.gene_b)
        self.cover_genes = None if cover_genes is None else sorted(set(cover_genes))
        self.use_cache = use_cache
        self.info: dict = {"model": spec["id"], "kind": self.kind, "n_train_pairs": int(len(T)),
                           "n_train_positives": int(T.y.sum()), "n_train_genes": len(self.train_genes),
                           "noncoding_dropped": self.noncoding, "train_hash": train_hash(T), "fits": []}
        self.timings: dict = {}
        self._sims = sims
        self.lf_groups = resolve_groups(spec, self.lfspec) if self.kind in ("m2_stack", "lf_lgbm") else []
        self._cover_ref = None
        self.cover_max_abs_diff = 0.0

    # ---------------------------------------------------------------------------------- helpers
    def _t(self, name, t0):
        self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - t0, 2)

    @property
    def sims(self):
        if self._sims is None:
            from src.ryanfeat import SimStore
            names = self.spec.get("sims", "all")
            self._sims = SimStore() if names == "all" else SimStore(names=list(names))
        return self._sims

    def _ryan_cfg(self) -> dict:
        from src.ryanfeat import resolve_cfg
        return resolve_cfg(self.spec.get("ryan_cfg") or {})

    def _builder(self):
        from src.models import m2
        gt = self._gene_table()
        return m2.make_builder(gt, self.spec.get("gp_cfg") or {})

    def _gene_table(self) -> pd.DataFrame:
        """Label-free gene table restricted to the spec's feature groups (GP inputs). With all groups this is
        identical to src.models.m2.gene_table()."""
        sp = self.lfspec
        parts = [pd.DataFrame(sp.table(rel, cols), index=sp.genes, columns=cols)
                 for g, rel, cols in sp.gene_tables if g in self.lf_groups]
        if not parts:
            return None
        return pd.concat(parts, axis=1)

    def _cache_key(self) -> str:
        from src.models import m2
        from src.ryanfeat import cfg_hash
        sp = self.lfspec
        payload = {"v": "deliv-stack-v1", "train": self.info["train_hash"], "inner": protocol.load()["inner"],
                   "ryan": cfg_hash(self._ryan_cfg(), self.sims), "gp": m2.resolve_gp(self.spec.get("gp_cfg") or {}),
                   "m2v": m2.M2_FEATURE_VERSION,
                   "gene_tables": {rel: sp.manifest[rel].get("sha256", "") for g, rel, _ in sp.gene_tables
                                   if g in self.lf_groups}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ---------------------------------------------------------------------------------- fit
    def fit(self) -> "TrainedModel":
        from src.timing import set_threads
        set_threads(self.threads)
        t0 = time.perf_counter()
        with deterministic_torch(self.threads):
            getattr(self, f"_fit_{self.kind}")()
        self._t("fit_total", t0)
        return self

    def _fit_b1(self):
        from src.ryanfeat import fit_b1
        c = self._ryan_cfg()
        self.c0, self.beta = fit_b1(self.T, c["b1_C"])

    def _fit_b4(self):
        from src.ryanfeat import fit_mf
        self._fit_b1()
        c = self._ryan_cfg()
        self.mf = [fit_mf(self.T, c["mf_k"], c["mf_wd"], c["mf_epochs"], c["mf_lr"], s) for s in self.spec["seeds"]]

    def _fit_lf_lgbm(self):
        from src.models import lgbm
        T = self.T
        X = PA.assemble(T, groups=self.lf_groups, spec=self.lfspec).to_numpy()
        self.names = self.lfspec.feature_names(self.lf_groups)
        y = T.y.to_numpy(np.int8)
        params = lgbm.params_for(self.spec.get("lgbm_config", "default"), self.threads)
        nf = protocol.load()["dev_cv"]["n_folds"]
        self.boosters = []
        for s in self.spec["seeds"]:
            fm, vm = lgbm.gene_holdout(T, lgbm.VALID_GENE_FRAC, [protocol.load()["inner"]["seed"], nf, int(s)])
            b, inf = lgbm.fit(X, y, fm, vm, params, seed=int(s), feature_names=self.names)
            self.boosters.append(b)
            self.info["fits"].append({"seed": int(s), **inf})
            self.log(f"  [{self.spec['id']}] seed {s}: best_iter {inf['best_iteration']} valid AP {inf['valid_ap']:.4f}")
        del X

    def stacker_rows(self):
        """(stacker training rows, inner folds) with nested Ryan-derived features (cached)."""
        from src import stacking
        cfg = self._ryan_cfg()
        inner = data.inner_splits(self.T)
        builder = self._builder()
        key = self._cache_key()
        d = protocol.cache_root() / "deliverable" / key
        parts = []
        for f in inner:
            p = d / f"inner{f.j}.parquet"
            x = pd.read_parquet(p) if (self.use_cache and p.exists()) else None
            if x is None:
                t0 = time.perf_counter()
                x = stacking.inner_rows(f, self.sims, cfg, builder)
                self._t("stacker_features", t0)
                if self.use_cache:
                    d.mkdir(parents=True, exist_ok=True)
                    x.to_parquet(p)
            parts.append(x)
        self.info["stacker_cache"] = str(d)
        return pd.concat(parts, ignore_index=True), inner

    def _fit_m2_stack(self):
        import lightgbm as lgb
        from sklearn.metrics import average_precision_score
        from src.models import lgbm
        from src.models import m2
        from src.stacking import stacker_folds
        t0 = time.perf_counter()
        tr, inner = self.stacker_rows()
        self._t("stacker_rows", t0)
        t0 = time.perf_counter()
        Ltr = PA.assemble(tr, groups=self.lf_groups, spec=self.lfspec).to_numpy()
        self.rcols = [c for c in tr.columns if m2.family_of(c) != "LF" and c not in META]
        names = list(self.lfspec.feature_names(self.lf_groups)) + self.rcols
        fam = np.array([m2.family_of(n) for n in names])
        self.cols = np.flatnonzero(np.isin(fam, list(self.spec["families"])))
        self.names = list(np.array(names)[self.cols])
        self.n_lf = Ltr.shape[1]
        X = np.hstack([Ltr, tr[self.rcols].to_numpy(np.float32)])[:, self.cols]
        del Ltr
        y = tr.y.to_numpy(np.int8)
        w = tr.w.to_numpy(float) / tr.w.mean()
        init = tr.b1_logit.to_numpy(float) if self.spec.get("init_score") else None
        groups = {}
        for j, t, v in stacker_folds(tr, inner):
            fm = np.zeros(len(tr), bool)
            vm = np.zeros(len(tr), bool)
            fm[t.index.to_numpy()] = True
            vm[v.index.to_numpy()] = True
            groups[j] = (fm, vm)
        self._t("stacker_assemble", t0)
        params = lgbm.params_for(self.spec.get("lgbm_config", "default"), self.threads)
        if self.spec.get("rounds", "inner_early_stopping") != "inner_early_stopping":
            raise ValueError(f"unsupported rounds rule {self.spec.get('rounds')!r} for m2_stack")
        self.boosters = []
        safe = lgbm.safe_names(self.names)
        t0 = time.perf_counter()
        for s in self.spec["seeds"]:
            j = int(s) % len(inner)
            fm, vm = groups[j]
            p = {**params, "seed": int(s), "bagging_seed": int(s) + 1, "feature_fraction_seed": int(s) + 2,
                 "data_random_seed": int(s) + 3}
            dtr = lgb.Dataset(X[fm], label=y[fm], weight=w[fm], init_score=None if init is None else init[fm],
                              feature_name=safe, free_raw_data=True)
            dva = lgb.Dataset(X[vm], label=y[vm], weight=w[vm], init_score=None if init is None else init[vm],
                              reference=dtr, free_raw_data=True)
            b = lgb.train(p, dtr, num_boost_round=lgbm.MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"],
                          callbacks=[lgb.early_stopping(lgbm.EARLY_STOP, verbose=False)])
            pv = b.predict(X[vm], num_iteration=b.best_iteration, raw_score=True) + (0.0 if init is None else init[vm])
            inf = {"seed": int(s), "best_iteration": int(b.best_iteration), "inner_fold_valid": j,
                   "valid_ap_weighted": float(average_precision_score(y[vm], pv, sample_weight=w[vm])),
                   "n_fit": int(fm.sum()), "n_valid": int(vm.sum())}
            self.boosters.append(b)
            self.info["fits"].append(inf)
            self.log(f"  [{self.spec['id']}] seed {s}: best_iter {inf['best_iteration']} inner-valid wAP "
                     f"{inf['valid_ap_weighted']:.4f} ({inf['n_fit']} fit / {inf['n_valid']} valid)")
        self._t("stacker_train", t0)
        self.info["n_stacker_rows"] = int(len(tr))
        self.info["n_features"] = int(len(self.cols))
        del X

    # ---------------------------------------------------------------------------------- predict
    def _ryan_block(self, Q: pd.DataFrame) -> pd.DataFrame:
        """RYAN/NBR/GP features of Q computed from all of T (padded with covering pairs, see module doc)."""
        cov = U.covering_pairs(self.cover_genes) if self.cover_genes else None
        q = Q[["gene_a", "gene_b"]] if cov is None else pd.concat([Q[["gene_a", "gene_b"]], cov], ignore_index=True)
        t0 = time.perf_counter()
        with deterministic_torch(self.threads):
            F = self._builder()(self.T, q, self.sims, self._ryan_cfg()).build()
        self._t("ryan_features", t0)
        if cov is not None:
            C = F.iloc[len(Q):][self.rcols].to_numpy(np.float64)
            if self._cover_ref is None:
                self._cover_ref = C
            else:
                same = np.array_equal(C, self._cover_ref, equal_nan=True)
                if not same:
                    dd = np.abs(C - self._cover_ref)
                    d = np.nanmax(dd)
                    self.cover_max_abs_diff = max(self.cover_max_abs_diff, float(d))
                    bad = [c for c, x in zip(self.rcols, np.nanmax(dd, axis=0)) if x > 0]
                    self.cover_diff_cols = sorted(set(getattr(self, "cover_diff_cols", [])) | set(bad))
        return F.iloc[:len(Q)].reset_index(drop=True)

    def _predict_block(self, ia_g: np.ndarray, ib_g: np.ndarray, R: np.ndarray | None, init=None) -> np.ndarray:
        """Scores for pairs given as genes.txt indices; R = Ryan feature rows (rcols order) for m2_stack."""
        k = self.kind
        if k == "m2_stack":
            t0 = time.perf_counter()
            L = PA.assemble_arrays(ia_g, ib_g, self.lfspec, groups=self.lf_groups)
            X = np.hstack([L, R])[:, self.cols]
            del L
            self._t("lf_assemble", t0)
            t0 = time.perf_counter()
            s = np.mean([b.predict(X, num_iteration=b.best_iteration, raw_score=True, num_threads=self.threads)
                         for b in self.boosters], axis=0)
            if self.spec.get("init_score"):
                s = s + init
            self._t("predict", t0)
            return s
        if k == "lf_lgbm":
            from src.models import lgbm
            t0 = time.perf_counter()
            X = PA.assemble_arrays(ia_g, ib_g, self.lfspec, groups=self.lf_groups)
            self._t("lf_assemble", t0)
            t0 = time.perf_counter()
            s = np.mean([lgbm.predict(b, X, self.threads) for b in self.boosters], axis=0)
            self._t("predict", t0)
            return s
        G = np.asarray(self.lfspec.genes, dtype=object)
        ga, gb = G[ia_g], G[ib_g]
        b1 = self.c0 + (self.beta.reindex(ga).fillna(0.0).to_numpy() + self.beta.reindex(gb).fillna(0.0).to_numpy())
        if k == "b1":
            return b1
        if k == "b4":
            from src.ryanfeat import mf_score   # MF fitted in fit() under deterministic_torch
            both = np.isin(ga, list(self.train_genes)) & np.isin(gb, list(self.train_genes))
            return np.mean([np.where(both, mf_score(P, ga, gb), b1) for P in self.mf], axis=0)
        raise ValueError(k)

    def predict(self, pairs: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
        """Scores for arbitrary pairs (gene_a, gene_b); hypomorph flags are 0 (deletion x deletion)."""
        ia = PA.gene_indices(self.lfspec, pairs.gene_a.to_numpy(object))
        ib = PA.gene_indices(self.lfspec, pairs.gene_b.to_numpy(object))
        R = init = None
        if self.kind == "m2_stack":
            F = self._ryan_block(pairs.reset_index(drop=True))
            R = F[self.rcols].to_numpy(np.float32)
            init = F.b1_logit.to_numpy(float)
        out = np.empty(len(pairs))
        for s in range(0, len(pairs), chunk):
            e = min(len(pairs), s + chunk)
            out[s:e] = self._predict_block(ia[s:e], ib[s:e], None if R is None else R[s:e],
                                           None if init is None else init[s:e])
        return out

    def score_universe(self, genes: list, ryan_chunk: int, lf_chunk: int):
        """Yield (ia, ib, scores) over all i<j pairs of the sorted `genes` (indices into `genes`), row-major."""
        genes = list(genes)
        assert genes == sorted(genes) and len(set(genes)) == len(genes), "genes must be sorted and unique"
        G = np.asarray(genes, dtype=object)
        gidx = PA.gene_indices(self.lfspec, G)
        if self.cover_genes is None:
            self.cover_genes = genes
        for ia, ib in U.pair_chunks(len(genes), ryan_chunk):
            R = init = None
            if self.kind == "m2_stack":
                F = self._ryan_block(pd.DataFrame({"gene_a": G[ia], "gene_b": G[ib]}))
                R = F[self.rcols].to_numpy(np.float32)
                init = F.b1_logit.to_numpy(float)
                del F
            out = np.empty(len(ia))
            for s in range(0, len(ia), lf_chunk):
                e = min(len(ia), s + lf_chunk)
                out[s:e] = self._predict_block(gidx[ia[s:e]], gidx[ib[s:e]], None if R is None else R[s:e],
                                               None if init is None else init[s:e])
            yield ia, ib, out
