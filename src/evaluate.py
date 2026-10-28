"""The single metric entry point (DESIGN.md §3, §6, §8).

    from src.evaluate import evaluate, compare
    res = evaluate(preds, model="B0", runtime=timings.as_dict())   # preds per docs/INTERFACES.md
    cmp = compare(res["run_id"], other_run_id)

* `evaluate` validates that predictions cover exactly the eval pairs of all folds, averages scores over
  seeds, joins the dev labels (the only place eval labels are joined), computes the primary metric
  (per-fold pooled universe-weighted AUPRC, mean over folds) and the secondary metrics, runs the shared
  Poisson(1) gene bootstrap of the exact primary statistic, saves results/runs/<run_id>.json and
  <data root>/preds/<run_id>.parquet, and appends a row to results/registry.tsv.
* `compare(run_a, run_b)` = paired bootstrap of Δ = primary(a) − primary(b) on the same replicates, with
  the adoption decision of protocol.yaml; appended to results/comparisons.tsv.
* `pooled_metrics` / `bootstrap_replicates` are pure functions reused by scripts/lockbox_eval.py.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import protocol
from src.splits import PARTS

PRED_COLUMNS = ["gene_a", "gene_b", "fold", "part", "score", "seed"]
REGISTRY_COLUMNS = [
    "run_id", "timestamp", "model", "data_root", "split_seed", "seeds", "git_commit", "git_dirty", "git_diff_sha256",
    "protocol_sha256", "splits_sha256", "feature_hashes", "primary", "primary_se", "primary_ci_lo", "primary_ci_hi",
    "weighted_prevalence", "lift", "ap_C1", "ap_C2", "ap_C3", "mean_per_class", "pooled_all_pairs", "pooled_equal",
    "auroc", "precision_at_k", "n_eval_rows", "wall_s", "cpu_s", "peak_rss_mb", "threads", "hardware",
    "eval_wall_s", "notes"]
COMPARE_COLUMNS = ["timestamp", "run_a", "run_b", "model_a", "model_b", "delta", "ci_lo", "ci_hi",
                   "delta_all_pairs", "ci_all_pairs", "delta_equal", "ci_equal", "alt_delta", "clean",
                   "sign_flip_veto", "adopt", "reason"]


# =========================================================================================== AP core
def _sort_groups(score: np.ndarray):
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    return order, ends


def ap_weighted_matrix(y_sorted: np.ndarray, W: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Average precision for each column of W (rows sorted by descending score; `ends` = last index of
    each tie group). Identical definition to sklearn.average_precision_score with sample weights."""
    W = W if W.ndim == 2 else W[:, None]
    yv = y_sorted.astype(W.dtype)[:, None]
    tp = np.cumsum(W * yv, axis=0)[ends]
    fp = np.cumsum(W * (1 - yv), axis=0)[ends]
    P = tp[-1]
    den = tp + fp
    prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    dR = np.diff(np.vstack([np.zeros((1, tp.shape[1])), tp]), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (dR * prec).sum(axis=0) / P
    return np.where(P > 0, ap, np.nan)


def class_row_weights(part: np.ndarray, cw: dict) -> np.ndarray:
    """Row weights so that each class's total weight equals its frozen class weight."""
    w = np.zeros(len(part))
    for c in PARTS:
        m = part == c
        if m.any():
            w[m] = cw[c] / m.sum()
    return w


# =========================================================================================== metrics
def _fold_metrics(d: pd.DataFrame, W: dict, k_prec: int) -> dict:
    y, s, part = d.y.to_numpy(bool), d.score.to_numpy(float), d.part.to_numpy()
    out = {}
    for name, cw in W.items():
        w = class_row_weights(part, cw)
        key = "pooled" if name == "primary" else f"pooled_{name}"
        out[key] = float(average_precision_score(y, s, sample_weight=w))
        out[f"prevalence_{name}"] = float(np.sum(w * y) / np.sum(w))
    cw = W["primary"]
    w = class_row_weights(part, cw)
    out["lift"] = out["pooled"] / out["prevalence_primary"]
    out["auroc"] = float(roc_auc_score(y, s, sample_weight=w))
    top = np.argsort(-s, kind="mergesort")[:k_prec]
    out["precision_at_k"] = float(np.sum(w[top] * y[top]) / np.sum(w[top]))
    for c in PARTS:
        m = part == c
        out[f"n_{c}"], out[f"pos_{c}"] = int(m.sum()), int(y[m].sum())
        out[f"ap_{c}"] = float(average_precision_score(y[m], s[m])) if y[m].any() else float("nan")
        out[f"auroc_{c}"] = float(roc_auc_score(y[m], s[m])) if 0 < y[m].sum() < m.sum() else float("nan")
        mt = np.flatnonzero(m)[np.argsort(-s[m], kind="mergesort")[:k_prec]]
        out[f"precision_at_k_{c}"] = float(y[mt].mean()) if len(mt) else float("nan")
    out["mean_per_class"] = float(sum(cw[c] * out[f"ap_{c}"] for c in PARTS))
    # C1 reciprocal vs single-orientation (reporting only; never a feature)
    if "both_orientations" in d:
        for nm, m in (("reciprocal", (part == "C1") & d.both_orientations.to_numpy(bool)),
                      ("single", (part == "C1") & ~d.both_orientations.to_numpy(bool))):
            out[f"ap_C1_{nm}"] = float(average_precision_score(y[m], s[m])) if y[m].any() else float("nan")
            out[f"n_C1_{nm}"], out[f"prevalence_C1_{nm}"] = int(m.sum()), float(y[m].mean()) if m.any() else np.nan
    # secondary masks (report-only)
    for nm, col, keep in (("viable_viable", "viable_viable", True), ("excl_damp_mixed", "damp_mixed", False)):
        if col in d:
            m = d[col].to_numpy(bool) == keep
            if y[m].any():
                out[f"pooled_{nm}"] = float(average_precision_score(y[m], s[m],
                                                                    sample_weight=class_row_weights(part[m], cw)))
    return out


def pooled_metrics(df: pd.DataFrame, weightings: dict | None = None) -> dict:
    """df: one row per eval pair with fold, part, y, score (+ optional both_orientations, viable_viable,
    damp_mixed). Returns {'per_fold': {k: {...}}, 'mean': {...}, 'primary': fold-mean pooled AP}."""
    W = weightings or {k: v for k, v in protocol.weightings().items() if k != "primary_name"}
    k_prec = protocol.load()["eval"]["precision_at_k"]
    per = {int(k): _fold_metrics(d, W, k_prec) for k, d in df.groupby("fold", sort=True)}
    keys = sorted(set().union(*[set(v) for v in per.values()]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN metrics (e.g. empty strata) stay NaN
        mean = {k: float(np.nanmean([per[f].get(k, np.nan) for f in per])) for k in keys}
    return {"per_fold": per, "mean": mean, "primary": mean["pooled"]}


# =========================================================================================== bootstrap
def bootstrap_counts(genes, n_boot: int | None = None, seed: int | None = None) -> tuple[np.ndarray, pd.Index]:
    """Poisson(1) count per gene per replicate, shape (n_boot, n_genes); genes are sorted so the same
    replicates are reproduced for every run over the same gene set (paired comparisons)."""
    B = protocol.load()["bootstrap"]
    assert B["distribution"] == "poisson1"
    idx = pd.Index(sorted(set(genes)))
    rng = np.random.default_rng(B["seed"] if seed is None else seed)
    return rng.poisson(1.0, size=(B["n_replicates"] if n_boot is None else n_boot, len(idx))), idx


def _fold_plan(d: pd.DataFrame, gene_index: pd.Index) -> dict:
    """Precomputation for fast replicate APs on one fold. Only tie groups that contain a positive change
    recall, so AP needs cumulative positive / total weight at those group ends only, per class."""
    order, ends = _sort_groups(d.score.to_numpy(float))
    n = len(order)
    y = d.y.to_numpy(bool)[order]
    part = d.part.to_numpy()[order]
    ia = gene_index.get_indexer(d.gene_a.to_numpy()[order])
    ib = gene_index.get_indexer(d.gene_b.to_numpy()[order])
    assert (ia >= 0).all() and (ib >= 0).all(), "eval gene missing from bootstrap gene index"
    gid = np.zeros(n, dtype=np.int64)
    gid[ends[:-1] + 1] = 1
    gid = np.cumsum(gid)
    pos_groups = np.unique(gid[y])
    e = ends[pos_groups]                                            # global end index of positive groups
    cls = {}
    for c in PARTS:
        idx = np.flatnonzero(part == c)
        if not len(idx):
            continue
        pidx = idx[y[idx]]
        cls[c] = {"ia": ia[idx], "ib": ib[idx], "pa": ia[pidx], "pb": ib[pidx],
                  "n_upto": np.searchsorted(idx, e, side="right"),      # class rows with position <= e
                  "p_upto": np.searchsorted(pidx, e, side="right")}
    return {"cls": cls, "n_groups": len(e)}


def _cum_at(vals: np.ndarray, upto: np.ndarray) -> np.ndarray:
    """cumsum(vals)[upto-1] with 0 where upto == 0; vals: rows x r (int64, exact)."""
    cs = np.vstack([np.zeros((1, vals.shape[1]), dtype=vals.dtype), np.cumsum(vals, axis=0)])
    return cs[upto]


def bootstrap_replicates(df: pd.DataFrame, counts: np.ndarray, gene_index: pd.Index,
                         weightings: dict | None = None, chunk: int = 50) -> dict:
    """Replicate values of the fold-mean pooled AP for each weighting: {name: array (n_boot,)}.
    Pair weight = count_a * count_b * class weight, class totals rescaled to the frozen weights within
    each fold (DESIGN §3). With all counts = 1 this reproduces the point statistic exactly (tested)."""
    W = weightings or {k: v for k, v in protocol.weightings().items() if k != "primary_name"}
    R = counts.shape[0]
    res = {n: np.zeros(R) for n in W}
    folds = sorted(df.fold.unique())
    for k in folds:
        plan = _fold_plan(df[df.fold == k], gene_index)
        for r0 in range(0, R, chunk):
            cnt = counts[r0:r0 + chunk].T.astype(np.int64)              # genes x r
            r = cnt.shape[1]
            tp_c, all_c, tot_c = {}, {}, {}
            for c, q in plan["cls"].items():
                base = cnt[q["ia"]] * cnt[q["ib"]]
                tot_c[c] = base.sum(axis=0).astype(float)
                all_c[c] = _cum_at(base, q["n_upto"]).astype(float)
                tp_c[c] = _cum_at(cnt[q["pa"]] * cnt[q["pb"]], q["p_upto"]).astype(float)
            for n, cw in W.items():
                tp = np.zeros((plan["n_groups"], r))
                al = np.zeros((plan["n_groups"], r))
                for c in tp_c:
                    f = np.divide(cw[c], tot_c[c], out=np.zeros(r), where=tot_c[c] > 0)
                    tp += tp_c[c] * f
                    al += all_c[c] * f
                P = tp[-1] if len(tp) else np.zeros(r)
                prec = np.divide(tp, al, out=np.zeros_like(tp), where=al > 0)
                dR = np.diff(np.vstack([np.zeros((1, r)), tp]), axis=0)
                ap = np.divide((dR * prec).sum(axis=0), P, out=np.zeros(r), where=P > 0)
                res[n][r0:r0 + chunk] += ap
    return {n: v / len(folds) for n, v in res.items()}


def _ci(x: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


# =========================================================================================== provenance
def git_info() -> dict:
    root = protocol.ROOT

    def run(*a):
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout

    commit = run("rev-parse", "HEAD").strip() or "nogit"
    status = run("status", "--porcelain", "--untracked-files=all")
    h = hashlib.sha256((run("diff", "HEAD") + status).encode())
    for line in status.splitlines():
        if line.startswith("??"):
            p = root / line[3:].strip()
            if p.is_file() and p.stat().st_size < 5 * 2**20:
                h.update(p.read_bytes())
    dirty = bool(status.strip())
    return {"git_commit": commit, "git_dirty": dirty, "git_diff_sha256": h.hexdigest() if dirty else ""}


def splits_sha256(split_seed: int) -> str:
    d = protocol.splits_dir()
    h = hashlib.sha256()
    for f in ("labels_dev.parquet", f"dev_folds_seed{split_seed}.tsv", f"c1_holdout_seed{split_seed}.parquet"):
        h.update(protocol.sha256_file(d / f).encode())
    return h.hexdigest()


def _append_tsv(path: Path, row: dict, columns: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    vals = []
    for c in columns:
        v = row.get(c, "")
        v = "" if v is None else (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        vals.append(v.replace("\t", " ").replace("\n", " "))
    with open(path, "a") as fh:
        if new:
            fh.write("\t".join(columns) + "\n")
        fh.write("\t".join(vals) + "\n")


# =========================================================================================== evaluate
def _labels_for(split_seed: int) -> pd.DataFrame:
    from src import data  # local import: data.py is the label owner
    return data.labels_dev()[["gene_a", "gene_b", "y", "both_orientations", "viable_viable", "damp_mixed"]]


def prepare(preds: pd.DataFrame, split_seed: int | None = None) -> tuple[pd.DataFrame, list]:
    """Validate predictions against the expected eval pairs, average over seeds, join labels."""
    from src import data
    split_seed = data._seed(split_seed)
    miss = [c for c in PRED_COLUMNS if c not in preds]
    if miss:
        raise ValueError(f"predictions lack columns {miss} (docs/INTERFACES.md)")
    p = preds[PRED_COLUMNS]
    if not (p.gene_a < p.gene_b).all():
        raise ValueError("predictions must have gene_a < gene_b")
    if not np.isfinite(p.score.to_numpy(float)).all():
        raise ValueError("non-finite scores")
    seeds = sorted(int(s) for s in p.seed.unique())
    per_seed = p.groupby("seed").size()
    if per_seed.nunique() != 1:
        raise ValueError(f"seeds cover different numbers of rows: {per_seed.to_dict()}")
    avg = p.groupby(["gene_a", "gene_b", "fold", "part"], sort=False, observed=True).score.mean().reset_index()
    if len(avg) != per_seed.iloc[0]:
        raise ValueError("duplicate (pair, fold) rows within a seed, or seeds disagree on the part label")
    exp = data.all_eval_pairs(split_seed)[["gene_a", "gene_b", "fold", "part"]]
    m = exp.merge(avg, on=["gene_a", "gene_b", "fold"], how="outer", suffixes=("", "_pred"), indicator=True)
    if (m._merge != "both").any():
        raise ValueError(f"prediction rows do not match the eval pairs: {m._merge.value_counts().to_dict()}")
    if (m.part != m.part_pred).any():
        raise ValueError(f"{int((m.part != m.part_pred).sum())} rows carry the wrong part label")
    df = m.drop(columns=["part_pred", "_merge"]).merge(_labels_for(split_seed), on=["gene_a", "gene_b"], how="left")
    assert df.y.notna().all()
    return df, seeds


def evaluate(preds: pd.DataFrame, model: str, split_seed: int | None = None, run_id: str | None = None,
             runtime: dict | None = None, feature_hashes: dict | None = None, notes: str = "",
             register: bool = True, n_boot: int | None = None) -> dict:
    from src import data
    t0 = time.perf_counter()
    split_seed = data._seed(split_seed)
    df, seeds = prepare(preds, split_seed)
    met = pooled_metrics(df)
    counts, gidx = bootstrap_counts(data.dev_folds(split_seed).index, n_boot)
    reps = bootstrap_replicates(df, counts, gidx)
    lvl = protocol.load()["bootstrap"]["ci_level"]
    boot = {n: {"se": float(np.std(v, ddof=1)), "ci": _ci(v, lvl), "mean": float(np.mean(v))} for n, v in reps.items()}
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = run_id or f"{model}__{ts}__{os.urandom(3).hex()}"
    rt = runtime or {}
    res = {"run_id": run_id, "timestamp": ts, "model": model, "split_seed": split_seed, "seeds": seeds,
           "data_root": str(protocol.data_root()), **git_info(), "protocol_sha256": protocol.protocol_sha256(),
           "splits_sha256": splits_sha256(split_seed), "labels_dev_sha256": data.labels_hash(),
           "feature_hashes": feature_hashes or {}, "primary": met["primary"], "bootstrap": boot,
           "n_boot": int(counts.shape[0]), "metrics_mean": met["mean"], "metrics_per_fold": met["per_fold"],
           "weightings": protocol.weightings(), "runtime": rt, "notes": notes}
    res["eval_wall_s"] = round(time.perf_counter() - t0, 2)
    if register:
        rr = protocol.results_root()
        (rr / "runs").mkdir(parents=True, exist_ok=True)
        (rr / "runs" / f"{run_id}.json").write_text(json.dumps(res, indent=1, default=float))
        pd_ = protocol.preds_dir()
        pd_.mkdir(parents=True, exist_ok=True)
        df[["gene_a", "gene_b", "fold", "part", "score"]].to_parquet(pd_ / f"{run_id}.parquet", index=False)
        mm = met["mean"]
        _append_tsv(rr / "registry.tsv", {
            **res, "primary_se": boot["primary"]["se"], "primary_ci_lo": boot["primary"]["ci"][0],
            "primary_ci_hi": boot["primary"]["ci"][1], "weighted_prevalence": mm["prevalence_primary"],
            "lift": mm["lift"], "ap_C1": mm["ap_C1"], "ap_C2": mm["ap_C2"], "ap_C3": mm["ap_C3"],
            "mean_per_class": mm["mean_per_class"], "pooled_all_pairs": mm["pooled_all_pairs"],
            "pooled_equal": mm["pooled_equal"], "auroc": mm["auroc"], "precision_at_k": mm["precision_at_k"],
            "n_eval_rows": len(df), "wall_s": rt.get("wall_s"), "cpu_s": rt.get("cpu_s"),
            "peak_rss_mb": rt.get("peak_rss_mb"), "threads": rt.get("threads"), "hardware": rt.get("hardware")},
            REGISTRY_COLUMNS)
    return res


# =========================================================================================== compare
def load_run(run_id: str) -> dict:
    return json.loads((protocol.results_root() / "runs" / f"{run_id}.json").read_text())


def _run_frame(run: dict) -> pd.DataFrame:
    p = pd.read_parquet(protocol.preds_dir() / f"{run['run_id']}.parquet")
    return p.merge(_labels_for(run["split_seed"]), on=["gene_a", "gene_b"], how="left")


def compare(run_a: str, run_b: str, alt: tuple[str, str] | None = None, register: bool = True) -> dict:
    """Paired bootstrap Δ = primary(run_a) − primary(run_b) (a = candidate, b = incumbent).
    alt = (run_a', run_b') on the alternative split seed (required for main-ladder steps)."""
    from src import data
    A, B = load_run(run_a), load_run(run_b)
    for k in ("split_seed", "protocol_sha256", "splits_sha256"):
        if A[k] != B[k]:
            raise ValueError(f"runs differ in {k}; not comparable")
    da, db = _run_frame(A), _run_frame(B)
    key = ["gene_a", "gene_b", "fold"]
    db = db.set_index(key).loc[pd.MultiIndex.from_frame(da[key])].reset_index()
    if len(da) != len(db):
        raise ValueError("runs cover different eval rows")
    counts, gidx = bootstrap_counts(data.dev_folds(A["split_seed"]).index, A["n_boot"])
    ra, rb = bootstrap_replicates(da, counts, gidx), bootstrap_replicates(db, counts, gidx)
    ad = protocol.load()["adoption"]
    lvl = ad["ci_level"]
    pa, pb = pooled_metrics(da)["mean"], pooled_metrics(db)["mean"]
    out = {"run_a": run_a, "run_b": run_b, "model_a": A["model"], "model_b": B["model"],
           "delta": pa["pooled"] - pb["pooled"]}
    out["ci_lo"], out["ci_hi"] = _ci(ra["primary"] - rb["primary"], lvl)
    for n in ("all_pairs", "equal"):
        out[f"delta_{n}"] = pa[f"pooled_{n}"] - pb[f"pooled_{n}"]
        out[f"ci_{n}"] = list(_ci(ra[n] - rb[n], lvl))
    sgn = np.sign(out["delta"])
    out["sign_flip_veto"] = bool(any((sgn > 0 and out[f"ci_{n}"][1] < 0) or (sgn < 0 and out[f"ci_{n}"][0] > 0)
                                     for n in ("all_pairs", "equal")))
    out["clean"] = not (A["git_dirty"] or B["git_dirty"])
    reasons = []
    if not out["ci_lo"] > 0:
        reasons.append("CI includes/below 0")
    if not out["delta"] >= ad["min_delta"]:
        reasons.append(f"delta < {ad['min_delta']}")
    if ad["require_clean_git"] and not out["clean"]:
        reasons.append("dirty run(s)")
    if out["sign_flip_veto"]:
        reasons.append("opposite-direction CI under another weighting")
    out["alt_delta"] = None
    if alt is not None:
        out["alt_delta"] = load_run(alt[0])["primary"] - load_run(alt[1])["primary"]
        if ad["ladder_requires_alt_seed_positive"] and not out["alt_delta"] > 0:
            reasons.append("alt-seed delta not > 0")
    out["adopt"] = not reasons
    out["reason"] = "; ".join(reasons) or "adopted"
    out["timestamp"] = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if register:
        _append_tsv(protocol.results_root() / "comparisons.tsv", out, COMPARE_COLUMNS)
    return out
