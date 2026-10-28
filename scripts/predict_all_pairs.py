#!/usr/bin/env python
"""THE DELIVERABLE (DESIGN.md §9): train a model on all permitted Ryan data and score every pair of the
prediction universe.

    predict_all_pairs.py [--model M2r_huber] [--train-set dev|all] [--out PATH] [--oof-run RUN_ID|--no-oof]

--model           a key of config/models.yaml; default = its `default_model` (the incumbent, M2r_huber).
--train-set dev   all dev training pairs (src.data.dev_pairs: lockbox genes and lockbox-C1 pairs excluded).
                  The interim deliverable, usable before the lockbox evaluation.
--train-set all   dev + lockbox pairs: the FINAL deliverable. Refused until scripts/lockbox_eval.py has
                  completed (the lockbox labels are obtained only through lockbox_eval.unblinded_lockbox_pairs).
Training = src/deliverable.TrainedModel (config/models.yaml spec; for M2 the nested-stacking procedure of one
dev-CV outer fold, applied to the whole training set: stacker rows from inner folds, universe rows with
RYAN/NBR/GP features from the full training set). Scoring streams over row blocks of the sorted gene list.

Universe = universe_genes.tsv `in_universe` genes + the unknown-viability genes outside it (flagged), all
n(n-1)/2 pairs (toy data: every feature-store gene). Output (gitignored, data root):
  <data root>/universe/predictions_<model>_<train set>.parquet, sorted by rank, one row per pair:
    gene_a < gene_b, score (mean raw LightGBM score over seeds = log-odds scale, uncalibrated; higher = more
    likely S2 < threshold), rank (1 = highest), class (C1/C2/C3: both / one / neither gene in the training
    pairs), viability_unknown (either gene outside the protocol universe, unknown deletion viability),
    linked (same chromosome, gap < linkage.window_kb of config/deliverable.yaml: Ryan essentially never
    measured such pairs), same_chrom_gap_kb (NaN if different chromosomes / no coordinates),
    measured_in_ryan (True/False; <NA> = cannot be known without opening the lockbox, train-set dev only),
    in_training (label used to fit the model -> the score is IN-SAMPLE), oof_score / oof_n_folds / oof_source
    (out-of-fold score from the dev-CV run or, after the lockbox, from the lockbox evaluation).
Summary (aggregates only, committed): results/universe/summary_<model>_<train set>.json with runtime and
peak RSS, checks (row count = n(n-1)/2, no NaN, cross-block feature consistency), class / flag counts, and the
in-sample vs out-of-fold comparison on measured training pairs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, protocol  # noqa: E402
from src import deliverable as DL  # noqa: E402
from src import pairs as PA  # noqa: E402
from src import universe as U  # noqa: E402
from src.evaluate import git_info  # noqa: E402
from src.timing import Timings, hardware, peak_rss_mb, set_threads  # noqa: E402


def lockbox_module():
    s = importlib.util.spec_from_file_location("lockbox_eval", ROOT / "scripts" / "lockbox_eval.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def linear_keys(idx: pd.Index, ga, gb, n: int) -> np.ndarray:
    """i*n + j (i<j) for pairs whose genes are both in idx; -1 otherwise."""
    a, b = idx.get_indexer(np.asarray(ga, object)), idx.get_indexer(np.asarray(gb, object))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.where((lo >= 0) & (lo != hi), lo.astype(np.int64) * n + hi, -1)


def pos_of_key(key: np.ndarray, n: int) -> np.ndarray:
    """Row position of pair key i*n+j in the row-major i<j enumeration."""
    i, j = key // n, key % n
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def training_pairs(train_set: str, log=print) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(training pairs, all measured pairs incl. drop-from-training (for flags) or None)."""
    dev = data.dev_pairs()
    if train_set == "dev":
        return dev, data.dev_pairs(include_train_drop=True)
    LB = lockbox_module()
    lb = LB.unblinded_lockbox_pairs()
    log(f"train-set all: {len(dev)} dev + {len(lb)} lockbox training pairs")
    meas = pd.concat([data.dev_pairs(include_train_drop=True), LB.unblinded_lockbox_pairs(include_train_drop=True)],
                     ignore_index=True)
    return pd.concat([dev, lb], ignore_index=True), meas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score every universe pair")
    ap.add_argument("--model", default=None, help="config/models.yaml model id (default: its default_model)")
    ap.add_argument("--models-file", default=str(DL.MODELS_PATH))
    ap.add_argument("--train-set", default="dev", choices=["dev", "all"])
    ap.add_argument("--out", default=None, help="parquet path (default <data root>/universe/...)")
    ap.add_argument("--summary", default=None, help="summary json (default results/universe/...)")
    ap.add_argument("--oof-run", default=None)
    ap.add_argument("--oof-file", default=None, help="parquet (gene_a, gene_b, score) of out-of-fold scores, "
                    "instead of a registered run (testing / custom runs)")
    ap.add_argument("--no-oof", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ryan-chunk", type=int, default=None)
    ap.add_argument("--lf-chunk", type=int, default=None)
    ap.add_argument("--limit-genes", type=int, default=None, help="testing: first N universe genes only")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save-models", action="store_true", help="save LightGBM boosters next to the output")
    a = ap.parse_args(argv)
    C = U.deliverable_cfg()
    threads = a.threads or C["scoring"]["threads"]
    set_threads(threads)
    T = Timings()
    t_all = time.perf_counter()
    lfspec = PA.load_spec()
    a.model = a.model or DL.default_model(Path(a.models_file))
    spec = DL.model_spec(a.model, Path(a.models_file))
    tag = f"{a.model}_{a.train_set}" + (f"_first{a.limit_genes}" if a.limit_genes else "")
    print(f"data root {protocol.data_root()}; model {a.model} ({spec['kind']}); train set {a.train_set}; "
          f"threads {threads}", flush=True)

    with T.stage("load_training_pairs"):
        train, measured = training_pairs(a.train_set)
    genes, gflags = U.universe_genes(lfspec.genes)
    if a.limit_genes:
        genes, gflags = genes[:a.limit_genes], gflags.iloc[:a.limit_genes].reset_index(drop=True)
    n = len(genes)
    N = U.n_pairs(n)
    gi = pd.Index(genes)
    print(f"{n} genes ({int(gflags.viability_unknown.sum())} viability_unknown) -> {N:,} pairs", flush=True)

    with T.stage("fit"):
        model = DL.TrainedModel(spec, train, threads=threads, lfspec=lfspec, cover_genes=genes,
                                use_cache=not a.no_cache).fit()
    print(f"fit {model.timings}", flush=True)

    scores = np.full(N, np.nan, dtype=np.float64)
    pos = 0
    rc = a.ryan_chunk or C["scoring"]["ryan_chunk_pairs"]
    lc = a.lf_chunk or C["scoring"]["lf_chunk_pairs"]
    with T.stage("score_universe"):
        t0 = time.perf_counter()
        for ia, ib, s in model.score_universe(genes, rc, lc):
            assert ia[0] * n - ia[0] * (ia[0] + 1) // 2 + (ib[0] - ia[0] - 1) == pos
            scores[pos:pos + len(s)] = s
            pos += len(s)
            print(f"  scored {pos:,}/{N:,} pairs ({time.perf_counter() - t0:.0f} s, peak RSS {peak_rss_mb():.0f} MB)",
                  flush=True)
    assert pos == N, (pos, N)

    with T.stage("flags_and_write"):
        ia, ib = np.triu_indices(n, 1)
        seen = gi.isin(model.train_genes)
        cls = U.pair_class(seen[ia], seen[ib])
        vu = gflags.viability_unknown.to_numpy(bool)
        try:
            coords = U.load_coords()
            chrom, start, end = U.gene_coord_arrays(coords, genes)
            same, gap = U.gap_from_arrays(chrom, start, end, ia, ib)
            n_coords = int((chrom > 0).sum())
        except FileNotFoundError:
            same, gap, n_coords = np.zeros(N, bool), np.full(N, -1), 0
        gap_kb = np.where(same, gap / 1000.0, np.nan).astype(np.float32)
        linked = same & (gap < U.linked_window_bp())
        # in_training / measured
        tk = linear_keys(gi, model.T.gene_a, model.T.gene_b, n)
        in_tr = np.zeros(N, bool)
        in_tr[pos_of_key(tk[tk >= 0], n)] = True
        mk = linear_keys(gi, measured.gene_a, measured.gene_b, n)
        meas = np.zeros(N, bool)
        meas[pos_of_key(mk[mk >= 0], n)] = True
        meas_s = pd.array(meas, dtype="boolean")
        if a.train_set == "dev":
            # pairs among Ryan S2 genes that touch a lockbox gene, or dev x dev pairs absent from the dev labels
            # (lockbox-C1 or unmeasured): status unknown without the lockbox
            dev_g = set(measured.gene_a) | set(measured.gene_b)
            lbg = set(U.lockbox_gene_list())
            s2 = gi.isin(dev_g | lbg)
            unknown = s2[ia] & s2[ib] & ~meas
            meas_s[unknown] = pd.NA
        # OOF scores
        oof = np.full(N, np.nan, np.float32)
        oofn = np.zeros(N, np.int8)
        src = np.zeros(N, np.int8)              # 0 none, 1 dev_cv, 2 lockbox
        run = None if a.no_oof else (a.oof_run or DL.dev_run_id(spec))
        o = None
        if a.oof_file and not a.no_oof:
            o = pd.read_parquet(a.oof_file, columns=["gene_a", "gene_b", "score"]).groupby(
                ["gene_a", "gene_b"], sort=False).score.agg(oof_score="mean", oof_n_folds="size").reset_index()
            run = f"file:{a.oof_file}"
        elif run and (protocol.preds_dir() / f"{run}.parquet").exists():
            o = U.oof_scores(run)
        elif run and U.is_real_data():
            raise SystemExit(f"out-of-fold predictions of {run} not found in {protocol.preds_dir()} (use --no-oof "
                             "to ship without them)")
        if o is not None:
            k = linear_keys(gi, o.gene_a, o.gene_b, n)
            ok = k >= 0
            p = pos_of_key(k[ok], n)
            oof[p], oofn[p], src[p] = o.oof_score.to_numpy()[ok], o.oof_n_folds.to_numpy()[ok], 1
        else:
            run = None
        lb_oof = None
        if a.train_set == "all":
            f = lockbox_module().lockbox_preds_path(a.model)
            if f.exists():
                o = pd.read_parquet(f)
                k = linear_keys(gi, o.gene_a, o.gene_b, n)
                ok = k >= 0
                p = pos_of_key(k[ok], n)
                oof[p], oofn[p], src[p] = o.score.to_numpy()[ok], 1, 2
                lb_oof = str(f)
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(N, np.int64)
        rank[order] = np.arange(1, N + 1)
        G = pd.Categorical.from_codes(np.arange(n), categories=genes)
        df = pd.DataFrame({
            "gene_a": pd.Categorical.from_codes(ia[order], categories=genes),
            "gene_b": pd.Categorical.from_codes(ib[order], categories=genes),
            "score": scores[order].astype(np.float32), "rank": rank[order].astype(np.int32),
            "class": pd.Categorical(cls[order], categories=["C1", "C2", "C3"]),
            "viability_unknown": (vu[ia] | vu[ib])[order], "linked": linked[order],
            "same_chrom_gap_kb": gap_kb[order], "measured_in_ryan": meas_s[order], "in_training": in_tr[order],
            "oof_score": oof[order], "oof_n_folds": oofn[order],
            "oof_source": pd.Categorical.from_codes(src[order], categories=["", "dev_cv", "lockbox"])})
        del G
        out = Path(a.out) if a.out else protocol.data_root() / "universe" / f"predictions_{tag}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        if a.save_models and hasattr(model, "boosters"):
            md = out.parent / f"models_{tag}"
            md.mkdir(exist_ok=True)
            for s, b in zip(spec["seeds"], model.boosters):
                b.save_model(str(md / f"seed{s}.txt"), num_iteration=b.best_iteration)

    # ---------------------------------------------------------------- checks + summary (aggregates only)
    from sklearn.metrics import average_precision_score
    from scipy.stats import spearmanr
    checks = {"n_genes": n, "n_rows": int(len(df)), "expected_rows": N, "row_count_ok": bool(len(df) == N),
              "no_nan_scores": bool(np.isfinite(scores).all()), "pairs_unique_sorted": True,
              "cover_rows_identical_across_blocks": model.cover_max_abs_diff == 0.0,
              "cover_max_abs_diff": model.cover_max_abs_diff}
    assert checks["row_count_ok"] and checks["no_nan_scores"], checks
    # in-sample vs out-of-fold on measured training pairs with an OOF score
    cons = None
    m_ = in_tr & np.isfinite(oof) & (src == 1)
    if m_.any():
        lab = model.T[["gene_a", "gene_b", "y"]]
        k = linear_keys(gi, lab.gene_a, lab.gene_b, n)
        y = np.zeros(N, bool)
        y[pos_of_key(k[k >= 0], n)] = lab.y.to_numpy(bool)[k >= 0]
        cons = {"n_pairs": int(m_.sum()), "positives": int(y[m_].sum()),
                "ap_in_sample": float(average_precision_score(y[m_], scores[m_])),
                "ap_out_of_fold": float(average_precision_score(y[m_], oof[m_])),
                "spearman_in_sample_vs_oof": float(spearmanr(scores[m_], oof[m_]).statistic),
                "note": "in-sample scores are optimistic on training pairs; use oof_score for measured pairs"}
        by = {}
        for c in ("C1", "C2", "C3"):
            mc = m_ & (cls == c)
            if mc.sum() > 1 and y[mc].any():
                by[c] = {"n": int(mc.sum()), "ap_in_sample": float(average_precision_score(y[mc], scores[mc])),
                         "ap_out_of_fold": float(average_precision_score(y[mc], oof[mc]))}
        cons["by_class_of_pair_in_universe"] = by
    qs = [0.5, 0.9, 0.99, 0.999]
    top = {}
    for kk in (1000, 10000, 100000):
        if kk <= N:
            t = df.iloc[:kk]
            top[str(kk)] = {"class": t["class"].value_counts().to_dict(), "linked": int(t.linked.sum()),
                            "viability_unknown": int(t.viability_unknown.sum()), "in_training": int(t.in_training.sum())}
    rt = T.as_dict()
    summ = {
        "model": a.model, "spec": spec, "models_sha256": DL.models_sha256(Path(a.models_file)), "train_set": a.train_set,
        "data_root": str(protocol.data_root()), "real_data": U.is_real_data(), "output": str(out),
        "protocol_sha256": protocol.protocol_sha256(), "git": git_info(),
        "checks": checks, "fit_info": model.info, "model_timings_s": model.timings,
        "runtime": {"total_wall_s": round(time.perf_counter() - t_all, 1),
                    "stages_wall_s": {s["stage"]: s["wall_s"] for s in rt["stages"]},
                    "cpu_s": rt["cpu_s"], "peak_rss_mb": round(peak_rss_mb(), 1), "threads": threads,
                    "hardware": hardware(), "ryan_chunk": rc, "lf_chunk": lc},
        "counts": {"class": df["class"].value_counts().to_dict(), "viability_unknown_pairs": int(df.viability_unknown.sum()),
                   "linked_pairs": int(df.linked.sum()), "linked_window_kb": U.linked_window_bp() / 1000,
                   "genes_with_coords": n_coords, "in_training": int(df.in_training.sum()),
                   "measured_true": int((df.measured_in_ryan == True).sum()),  # noqa: E712
                   "measured_unknown": int(df.measured_in_ryan.isna().sum()),
                   "with_oof": int(np.isfinite(df.oof_score).sum())},
        "score_quantiles": {str(q): float(np.quantile(scores, q)) for q in qs},
        "score_mean_by_class": df.groupby("class", observed=True).score.mean().astype(float).to_dict(),
        "top_k_composition": top, "oof_run": run, "lockbox_oof": lb_oof, "in_sample_vs_oof": cons}
    sp = Path(a.summary) if a.summary else protocol.results_root() / "universe" / f"summary_{tag}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summ, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    print(json.dumps({k: summ[k] for k in ("checks", "runtime", "counts", "in_sample_vs_oof")}, indent=1, default=str))
    print(f"wrote {out} and {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
