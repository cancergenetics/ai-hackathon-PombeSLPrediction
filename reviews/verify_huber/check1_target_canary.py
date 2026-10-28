"""Check 1: target-leak canary for the Huber stacker (scripts/run_m2plus.py FoldDataP + fit_stacker_huber).

Outer fold K, variant M2r, the production path end to end with NO feature cache (private root):
  world A = clean labels; world B = labels_dev corrupted IN MEMORY for every pair touching a fold-K gene or in
  fold K's C1 hold-out (this covers every outer-eval pair): s_score / s_raw_* replaced by N(-1, 3) draws,
  y / y_raw recomputed. For each world: FoldDataP (nested features + stacker target D.s) -> fit_stacker_huber
  (seeds = config bag, inner fold s mod 5 early stopping) -> run_m2.predict. Torch 1 thread (MF bit-identity),
  LightGBM 2 threads. Pass: every stacker input (Xtr, Xev, y, s, w, inner masks) and every seed's eval
  prediction bit-identical between A and B.
Code-level assertions (world A): stacker rows touch no fold-K gene and no outer-eval pair; D.s equals the
  labels_dev S2 of the same (outer-training) pair; t = -min(S2, 0); early-stopping rows vm are inner-fold-j
  stacker rows disjoint from the fitting rows fm.
Positive controls (planted mutants, run on the clean feature matrices, which the canary shows are identical in
  both worlds, with the outer-eval labels taken from the world's label table):
  P_es     early stopping on the OUTER eval rows' binary AP (feval on eval labels) instead of inner fold j;
  P_target outer-eval rows appended to the stacker fit rows with target -min(S2_eval, 0).
  Each must change the fold's predictions between worlds A and B.
Also: clean world-A predictions vs the registered per-seed predictions of M2r_bag5_huber for fold K.
Prints aggregates only."""
import common
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from src.timing import Timings, set_threads
set_threads(common.THREADS)
import torch  # noqa: E402
torch.set_num_threads(1)

import run_m2plus as RP  # noqa: E402
from src import data, protocol  # noqa: E402
from src import pairs as PA  # noqa: E402
from src import splits as S  # noqa: E402
from src.models import lgbm, m2  # noqa: E402
from src.ryanfeat import SimStore  # noqa: E402

RP.THREADS = common.THREADS
RP.R.N_JOBS = common.THREADS
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEEDS = [int(s) for s in RP.CFG["seeds"]["bag"]]
THR = float(protocol.threshold())
LAB = protocol.label_column()
REG = "M2r_bag5_huber__20261028T202603__f8bde5"

ORIG_FT = m2.fold_tables_m2
m2.fold_tables_m2 = lambda *a, **kw: ORIG_FT(*a, **{**kw, "use_cache": False})   # never read a cache
data.cache_save = lambda *a, **kw: None                                         # never write one
CLEAN = data.labels_dev().copy()


def set_labels(df):
    data._labels_dev = lambda sd, _d=df: _d


def h(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ corrupted world
folds = data.dev_folds()
hold = pd.read_parquet(protocol.splits_dir() / f"c1_holdout_seed{data._seed(None)}.parquet")
hold = hold[hold.fold == K]
in_fold = (CLEAN.gene_a.map(folds) == K) | (CLEAN.gene_b.map(folds) == K)
in_hold = S.pair_keys(CLEAN.gene_a, CLEAN.gene_b).isin(S.pair_keys(hold.gene_a, hold.gene_b))
MASK = np.asarray(in_fold | in_hold)
BAD = CLEAN.copy()
rng = np.random.default_rng(777)
z = rng.normal(-1.0, 3.0, MASK.sum())
for c in ("s_score", "s_raw_mean", "s_raw_ab", "s_raw_ba"):
    BAD.loc[MASK, c] = z + (0 if c == "s_score" else rng.normal(0, 0.5, MASK.sum()))
BAD.loc[MASK, "y"] = BAD.loc[MASK, "s_score"] < THR
BAD.loc[MASK, "y_raw"] = BAD.loc[MASK, "s_raw_mean"] < THR
log = {"fold": K, "seeds": SEEDS, "corrupted_pairs": int(MASK.sum()),
       "corrupted_label_changes": int((BAD.y != CLEAN.y).sum())}

spec = PA.load_spec()
sims = SimStore(names=SimStore().names)
params = lgbm.params_for("default", common.THREADS)
t0 = time.time()


def eval_labels(D, world):
    lab = world[["gene_a", "gene_b", LAB, "y"]]
    e = D.ev[["gene_a", "gene_b"]].merge(lab, on=["gene_a", "gene_b"], how="left", validate="one_to_one")
    assert e[LAB].notna().all()
    return e[LAB].to_numpy(float), e.y.to_numpy(np.int8)


def mutant_fit(D, cols, seed, world, kind):
    """Planted leaks (positive controls)."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from src.stacking import class_weight_column
    fm, vm = D.groups[seed % D.n_inner]
    t = RP.etarget(D.s)
    s_ev, y_ev = eval_labels(D, world)
    w_ev = class_weight_column(D.ev.part.to_numpy())
    w_ev = w_ev / w_ev.mean()
    p = {**params, "objective": "huber", "alpha": RP.huber_delta(), "metric": "None", "seed": seed,
         "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2, "data_random_seed": seed + 3}
    Xf, tf, wf = D.Xtr[np.ix_(fm, cols)], t[fm], D.w[fm]
    if kind == "P_target":
        Xf = np.vstack([Xf, D.Xev[:, cols]])
        tf = np.r_[tf, RP.etarget(s_ev)]
        wf = np.r_[wf, w_ev]
        Xv, yv, wv, tv = D.Xtr[np.ix_(vm, cols)], D.y[vm], D.w[vm], t[vm]
    else:  # P_es: early stopping on outer eval labels
        Xv, yv, wv, tv = D.Xev[:, cols], y_ev, w_ev, RP.etarget(s_ev)
    dtr = lgb.Dataset(Xf, label=tf, weight=wf, free_raw_data=True)
    dva = lgb.Dataset(Xv, label=tv, weight=wv, reference=dtr, free_raw_data=True)
    fe = lambda pr, _d: ("wap", float(average_precision_score(yv, pr, sample_weight=wv)), True)  # noqa: E731
    b = lgb.train(p, dtr, num_boost_round=lgbm.MAX_ROUNDS, valid_sets=[dva], feval=fe,
                  callbacks=[lgb.early_stopping(lgbm.EARLY_STOP, verbose=False)])
    return b.predict(D.Xev[:, cols], num_iteration=b.best_iteration, raw_score=True), int(b.best_iteration)


res = {}
for wname, world in (("A_clean", CLEAN), ("B_corrupt", BAD)):
    set_labels(world)
    tr_pairs, _ = data.fold_split(None, K)
    D = RP.FoldDataP(None, K, spec, sims, Timings(), "M2r")
    print(f"{wname}: features built {time.time()-t0:.0f}s", flush=True)
    r = {"train_pairs_hash": h(S.pair_keys(tr_pairs.gene_a, tr_pairs.gene_b).to_numpy().astype(str)),
         "train_S2_hash": h(tr_pairs[LAB].to_numpy()),
         "Xtr": h(D.Xtr), "Xev": h(D.Xev), "y": h(D.y), "s": h(D.s), "w": h(D.w),
         "groups": h(np.stack([np.r_[D.groups[j][0], D.groups[j][1]] for j in sorted(D.groups)])),
         "ev_pairs": h(D.ev[["gene_a", "gene_b", "part"]].to_numpy().astype(str)),
         "n_stacker_rows": int(len(D.y)), "n_eval_rows": int(len(D.ev)), "names": h(np.array(D.names))}
    cols = D.cols(RP.FAMILIES)
    if wname == "A_clean":
        trm = D.tr_meta
        kf = set(folds.index[folds == K])
        sk = S.pair_keys(trm.gene_a, trm.gene_b)
        ek = S.pair_keys(D.ev.gene_a, D.ev.gene_b)
        lab = CLEAN.assign(k=S.pair_keys(CLEAN.gene_a, CLEAN.gene_b)).set_index("k")
        cc = {"stacker_rows_touching_fold_genes": int((trm.gene_a.isin(kf) | trm.gene_b.isin(kf)).sum()),
              "stacker_pairs_in_eval_pairs": int(sk.isin(ek).sum()),
              "stacker_pairs_in_corrupt_mask": int(sk.isin(S.pair_keys(CLEAN.gene_a[MASK], CLEAN.gene_b[MASK])).sum()),
              "stacker_pairs_not_in_outer_train": int((~sk.isin(S.pair_keys(tr_pairs.gene_a, tr_pairs.gene_b))).sum()),
              "D.s_equals_labels_dev_S2_of_same_pair": bool(np.array_equal(lab.loc[sk, LAB].to_numpy(float), D.s)),
              "D.y_equals_S2_lt_thr": bool(np.array_equal(D.y.astype(bool), D.s < THR)),
              "target_is_neg_min_s2_0": bool(np.array_equal(RP.etarget(D.s), np.maximum(-D.s, 0.0))),
              "huber_delta": RP.huber_delta(), "n_features_used": int(len(cols)),
              "n_XSP_features": int(sum(1 for n in D.names if "XSP" in n)), "s_column_in_features": LAB in D.names,
              "label_like_feature_names": [n for n in D.names if any(x in n.lower() for x in
                                                                     ("s_score", "s_raw", "y_raw"))]}
        for j, (fm, vm) in D.groups.items():
            cc[f"inner{j}"] = {"vm_all_inner_fold_j": bool((trm.inner_fold.to_numpy()[vm] == j).all()),
                               "fm_vm_disjoint": not bool((fm & vm).any()), "n_fm": int(fm.sum()), "n_vm": int(vm.sum())}
        log["code_level"] = cc
        print(json.dumps(cc, default=str), flush=True)
    preds, fits = [], []
    for s in SEEDS:
        b, inf = RP.fit_stacker_huber(D, cols, params, s, s % D.n_inner)
        preds.append(RP.R.predict(b, D, cols, False))
        fits.append({"seed": s, "best_iter": inf["best_iteration"], "inner_wAP": round(inf["valid_ap_weighted"], 5)})
        print(f"{wname} seed {s}: {fits[-1]} {time.time()-t0:.0f}s", flush=True)
    r["fits"] = fits
    r["pred_hashes"] = [h(p) for p in preds]
    res[wname] = {"meta": r, "preds": np.array(preds)}
    if wname == "A_clean":
        q = pd.read_parquet(common.REG_PREDS / "m2plus" / f"{REG}_perseed.parquet")
        q = q[q.fold == K]
        cmp = {}
        for i, s in enumerate(SEEDS):
            g = D.ev[["gene_a", "gene_b"]].assign(mine=preds[i]).merge(q[q.seed == s][["gene_a", "gene_b", "score"]],
                                                                       on=["gene_a", "gene_b"], validate="one_to_one")
            cmp[s] = {"n": len(g), "max_abs": float(np.max(np.abs(g.mine - g.score))),
                      "pearson": float(np.corrcoef(g.mine, g.score)[0, 1]), "bit_identical": bool((g.mine == g.score).all())}
        log["clean_vs_registered_perseed"] = cmp
        print(json.dumps(cmp), flush=True)
        pc = {}
        for kind in ("P_es", "P_target"):
            pa, ia = mutant_fit(D, cols, 0, CLEAN, kind)
            pb, ib = mutant_fit(D, cols, 0, BAD, kind)
            pc[kind] = {"best_iter_A": ia, "best_iter_B": ib, "preds_identical": bool(np.array_equal(pa, pb)),
                        "max_abs_diff": float(np.max(np.abs(pa - pb))), "detected": not np.array_equal(pa, pb)}
            print(kind, pc[kind], f"{time.time()-t0:.0f}s", flush=True)
        log["positive_controls"] = pc
    del D

A, B = res["A_clean"], res["B_corrupt"]
log["world_meta"] = {k: v["meta"] for k, v in res.items()}
log["inputs_identical"] = {k: A["meta"][k] == B["meta"][k] for k in
                           ("train_pairs_hash", "train_S2_hash", "Xtr", "Xev", "y", "s", "w", "groups", "ev_pairs", "names")}
log["preds_bit_identical_per_seed"] = [bool(np.array_equal(a, b)) for a, b in zip(A["preds"], B["preds"])]
log["preds_max_abs_diff_per_seed"] = [float(np.max(np.abs(a - b))) for a, b in zip(A["preds"], B["preds"])]
log["PASS"] = all(log["inputs_identical"].values()) and all(log["preds_bit_identical_per_seed"]) and \
    all(v["detected"] for v in log["positive_controls"].values())
log["wall_s"] = round(time.time() - t0)
print(json.dumps(log, indent=1, default=str))
(common.OUT / f"check1_target_canary_fold{K}.json").write_text(json.dumps(log, indent=1, default=str))
