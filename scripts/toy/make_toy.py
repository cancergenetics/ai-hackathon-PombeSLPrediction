#!/usr/bin/env python
"""Synthetic "toy" dataset with KNOWN answers for verifying the SL-prediction pipeline.

Mirrors data/processed (ryan_genes.tsv, ryan_pairs.parquet, features/...) under data/toy/.
Deterministic given --seed. See docs/INTERFACES.md for formats, DESIGN.md section 6 for purpose.

Generative model (all parameters in PARAMS below)
  latent per-gene propensity beta_g (log-normal bulk + a few hubs), module m_g (40 modules),
  module-relation matrix R (each module has one "between-pathway" partner module),
  latent pair covariate z_ab (symmetric N(0,1)).
  signal_ab = beta_a + beta_b + gamma_between*R[m_a,m_b] + gamma_within*1[m_a==m_b] + eta*z_ab
  S_ab      = -(signal_ab - mean(signal)) + sigma*eps          (more negative = stronger interaction)
  sigma is solved so that the expected prevalence P(S < -2.3) over measured pairs is PREV_TARGET.
  Pairs are measured only query x array (FILL fraction), one value per unordered pair.

Usage:
  .venv/bin/python scripts/toy/make_toy.py                    # -> data/toy
  .venv/bin/python scripts/toy/make_toy.py --permute-labels   # -> data/toy_permuted (negative control)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score as ap_score

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REL = "scripts/toy/make_toy.py"

PARAMS = dict(
    seed=20261028,
    n_query_only=180, n_both=120, n_array_only=500, n_universe_only=400,   # G = 1200
    n_modules=40, module_dirichlet_alpha=2.0,
    beta_scale=0.30, beta_log_sd=0.8, n_hubs=10, hub_range=(1.2, 2.2),
    gamma_between=1.8, gamma_within=0.5, eta=0.40,
    prev_target=0.02, threshold=-2.3, fill=0.80, both_orient_frac=0.5, raw_orient_sd=0.4,
    viability_probs=dict(viable=0.89, inviable=0.05, **{"condition-dependent": 0.03}, unknown=0.03),
    # feature noise levels (known => feature-informed oracle can be computed)
    infb_noise=(0.7, 1.5), infb_missing=0.05,
    modf_dim=8, modf_noise=0.6, modf_missing=0.10,
    noise_gene_features=6,
    pair_inf_noise=0.5, pair_missing=0.15,
    modsim_same=0.5, modsim_noise=0.2, sim_missing_genes=0.08,
    heldout_frac=0.20, heldout_seed=0,
    n_reps=100,                 # noise redraws for expected-AUPRC distributions
    # Pre-stated tolerance bands (fractions of the reference quantity); see known_answers.json notes
    bands=dict(floor_frac_of_feature_oracle=0.25,   # learner floor, fraction of feature-informed oracle
               soft_ceiling_mult_feature_oracle=1.15,  # features-only learner should not beat this by much
               good_frac_of_feature_oracle=0.50,
               prev_rel_tol=0.25),
)

SOURCE_ID = "toy_synthetic"


def rng_for(seed: int, name: str) -> np.random.Generator:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([seed, h])


def sym_noise(rng, G):
    A = rng.standard_normal((G, G)).astype(np.float32)
    S = (A + A.T) / np.float32(np.sqrt(2))
    return S


# --------------------------------------------------------------------------- world
def build_world(P: dict) -> dict:
    seed = P["seed"]
    nq, nb, na, nu = P["n_query_only"], P["n_both"], P["n_array_only"], P["n_universe_only"]
    G = nq + nb + na + nu
    ids = np.array([f"SPTOY{i:04d}" for i in range(G)])  # sorted == index order
    r = rng_for(seed, "roles")
    perm = r.permutation(G)
    role = np.empty(G, dtype=object)
    role[perm[:nq]] = "query"
    role[perm[nq:nq + nb]] = "both"
    role[perm[nq + nb:nq + nb + na]] = "array"
    role[perm[nq + nb + na:]] = "universe_only"
    is_query = np.isin(role, ["query", "both"])
    is_array = np.isin(role, ["array", "both"])
    measured = role != "universe_only"

    # modules, propensity
    r = rng_for(seed, "latent")
    M = P["n_modules"]
    w = r.dirichlet(np.full(M, P["module_dirichlet_alpha"]))
    module = r.choice(M, size=G, p=w)
    beta = P["beta_scale"] * np.exp(P["beta_log_sd"] * r.standard_normal(G))
    hubs = r.choice(G, size=P["n_hubs"], replace=False)
    beta[hubs] = r.uniform(*P["hub_range"], size=P["n_hubs"])
    R = np.zeros((M, M))
    partner = r.permutation(M)                      # module i -> partner[i] (!= i)
    for i in range(M):
        j = partner[i]
        if j == i:
            j = (i + 1) % M
        R[i, j] = R[j, i] = 1.0
    Zlat = sym_noise(r, G)                          # latent pair covariate

    # viability
    r = rng_for(seed, "viability")
    vk = list(P["viability_probs"].keys())
    viability = r.choice(vk, size=G, p=np.array(list(P["viability_probs"].values())) / sum(P["viability_probs"].values()))

    # candidate pairs: unordered {i<j} with (i query & j array) or (i array & j query)
    iu, ju = np.triu_indices(G, 1)
    cand = (is_query[iu] & is_array[ju]) | (is_array[iu] & is_query[ju])
    iu, ju = iu[cand], ju[cand]
    r = rng_for(seed, "fill")
    keep = r.random(len(iu)) < P["fill"]
    a, b = iu[keep], ju[keep]

    beta_sum = beta[a] + beta[b]
    signal = (beta_sum + P["gamma_between"] * R[module[a], module[b]]
              + P["gamma_within"] * (module[a] == module[b]) + P["eta"] * Zlat[a, b]).astype(np.float64)
    centered = signal - signal.mean()

    # solve sigma: mean_p Phi((thr + centered)/sigma) = prev_target   (S = -centered + sigma*eps)
    lo, hi = 0.05, 20.0
    for _ in range(200):          # prevalence is increasing in sigma (checked by test)
        mid = 0.5 * (lo + hi)
        prev = norm.cdf((P["threshold"] + centered) / mid).mean()
        lo, hi = (lo, mid) if prev > P["prev_target"] else (mid, hi)
    sigma = 0.5 * (lo + hi)
    eps = rng_for(seed, "noise").standard_normal(len(a))
    S = -centered + sigma * eps

    world = dict(P=P, G=G, ids=ids, role=role, is_query=is_query, is_array=is_array, measured=measured,
                 module=module, beta=beta, R=R, Zlat=Zlat, viability=viability, hubs=hubs,
                 a=a, b=b, signal=signal, centered=centered, sigma=sigma, S=S, beta_sum=beta_sum,
                 module_weights=w)
    build_features(world)
    return world


# --------------------------------------------------------------------------- features
def build_features(W: dict):
    P, G, seed = W["P"], W["G"], W["P"]["seed"]
    beta, module = W["beta"], W["module"]
    ids = W["ids"]
    zb = (beta - beta.mean()) / beta.std()
    W["zb_mean"], W["zb_sd"] = float(beta.mean()), float(beta.std())
    r = rng_for(seed, "feat_infb")
    gene = {}
    cols = {}
    for k, sd in enumerate(P["infb_noise"], start=1):
        v = zb + sd * r.standard_normal(G)
        v[r.random(G) < P["infb_missing"]] = np.nan
        cols[f"INFB__b{k}"] = v
    gene["INFB"] = pd.DataFrame(cols, index=pd.Index(ids, name="gene")).astype("float32")

    r = rng_for(seed, "feat_modf")
    D = P["modf_dim"]
    cent = r.standard_normal((P["n_modules"], D))
    X = cent[module] + P["modf_noise"] * r.standard_normal((G, D))
    miss = r.random(G) < P["modf_missing"]
    X[miss] = np.nan
    W["modf_centroids"], W["modf_missing"] = cent, miss
    gene["MODF"] = pd.DataFrame({f"MODF__x{k+1}": X[:, k] for k in range(D)},
                                index=pd.Index(ids, name="gene")).astype("float32")

    r = rng_for(seed, "feat_noise")
    gene["NOISE"] = pd.DataFrame({f"NOISE__n{k+1}": r.standard_normal(G) for k in range(P["noise_gene_features"])},
                                 index=pd.Index(ids, name="gene")).astype("float32")
    W["gene_feats"] = gene

    # similarity matrices
    r = rng_for(seed, "sim")
    same = (module[:, None] == module[None, :]).astype(np.float32)
    modsim = P["modsim_same"] * same + P["modsim_noise"] * sym_noise(r, G)
    nosim = sym_noise(r, G) * np.float32(0.3)
    sim_miss = r.random(G) < P["sim_missing_genes"]
    sim_miss2 = r.random(G) < P["sim_missing_genes"]
    for M_, mk in ((modsim, sim_miss), (nosim, sim_miss2)):
        M_[mk, :] = np.nan
        M_[:, mk] = np.nan
        np.fill_diagonal(M_, np.nan)
    W["sim"] = {"MODSIM": modsim.astype("float32"), "NOISESIM": nosim.astype("float32")}

    # pair features
    r = rng_for(seed, "pairfeat")
    pinf = W["Zlat"] + P["pair_inf_noise"] * sym_noise(r, G)
    pnoise = sym_noise(r, G)
    pm = r.random((G, G)) < P["pair_missing"]
    pm = np.triu(pm, 1)
    pm = pm | pm.T
    pinf = pinf.astype("float32")
    pinf[pm] = np.nan
    np.fill_diagonal(pinf, np.nan)
    np.fill_diagonal(pnoise, np.nan)
    W["pair"] = {"PAIRINF__z": pinf, "PAIRNOISE__n": pnoise.astype("float32")}


# --------------------------------------------------------------------------- tables
def gene_table(W) -> pd.DataFrame:
    P = W["P"]
    m = W["measured"]
    ids = W["ids"][m]
    idx = np.where(m)[0]
    in_pairs = np.zeros(W["G"], bool)
    in_pairs[W["a"]] = True
    in_pairs[W["b"]] = True
    isq = W["is_query"][idx]
    via = W["viability"][idx]
    damp = isq & (via == "inviable")
    df = pd.DataFrame(dict(
        ryan_id=ids, ryan_label_name=[f"toy{i}" for i in idx], pombase_id=ids, mapping="exact",
        label_name_consistent=True, gene_name=[f"toy{i}" for i in idx], gene_type="protein coding gene",
        viability=via, is_query=isq, is_array=W["is_array"][idx], in_s2_rows=in_pairs[idx],
        n_query_alleles=isq.astype("int64"),
        allele_classes=np.where(isq, np.where(damp, "DAmP", "unlabelled"), None),
        query_labels=[f"{g}(toy{i}YTOY)" if q else None for g, i, q in zip(ids, idx, isq)],
        merged_with_other_ryan_id=False))
    return df


def pair_table(W, S: np.ndarray) -> pd.DataFrame:
    P, a, b = W["P"], W["a"], W["b"]
    n = len(a)
    r = rng_for(P["seed"], "orient")
    q_a, q_b = W["is_query"][a], W["is_query"][b]
    both_possible = q_a & q_b
    u = r.random(n)
    pick_ab = r.random(n) < 0.5          # which single orientation when both possible
    n_ab = np.where(both_possible, np.where(u < P["both_orient_frac"], 1, pick_ab.astype(int)), q_a.astype(int))
    n_ba = np.where(both_possible, np.where(u < P["both_orient_frac"], 1, 1 - pick_ab.astype(int)), q_b.astype(int))
    n_ab = n_ab.astype("int64"); n_ba = n_ba.astype("int64")
    noise = P["raw_orient_sd"] * r.standard_normal((n, 2))
    raw_ab = np.where(n_ab > 0, S + noise[:, 0], np.nan)
    raw_ba = np.where(n_ba > 0, S + noise[:, 1], np.nan)
    stack = np.vstack([raw_ab, raw_ba])
    raw_mean = np.nanmean(stack, axis=0)
    raw_min, raw_max = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
    via = W["viability"]
    return pd.DataFrame(dict(
        gene_a=W["ids"][a], gene_b=W["ids"][b], s_score=S, s_raw_mean=raw_mean, s_raw_ab=raw_ab, s_raw_ba=raw_ba,
        n_meas=(n_ab + n_ba).astype("int64"), n_ab=n_ab, n_ba=n_ba, s_raw_mean_meas=raw_mean, s_raw_min=raw_min,
        s_raw_max=raw_max, s_raw_mean_all=raw_mean, n_meas_all=(n_ab + n_ba).astype("int64"),
        in_s2=True, both_orientations=(n_ab > 0) & (n_ba > 0), involves_merged_id=False,
        viability_a=via[a], viability_b=via[b],
        **{"neg_s2_lt_-2.3": S < P["threshold"], "neg_raw_lt_-2.3": raw_mean < P["threshold"]}))


# --------------------------------------------------------------------------- known answers
def ap(y, s):
    return float(ap_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def classes(W, heldout_mask):
    ha, hb = heldout_mask[W["a"]], heldout_mask[W["b"]]
    return {"C1": ~ha & ~hb, "C2": ha ^ hb, "C3": ha & hb, "all": np.ones(len(ha), bool)}


def posterior_scores(W):
    """Feature-informed Bayes-style oracle: uses ONLY the generated features + known generative
    noise levels (not the latent truth). Ceiling for a learner that sees only features, no labels."""
    P, G = W["P"], W["G"]
    gf = W["gene_feats"]
    zhat = np.zeros(G)
    prec = np.ones(G)                                  # prior N(0,1) on standardized beta
    for k, sd in enumerate(P["infb_noise"], start=1):
        f = gf["INFB"][f"INFB__b{k}"].to_numpy(dtype=float)
        ok = ~np.isnan(f)
        zhat[ok] += f[ok] / sd ** 2
        prec[ok] += 1 / sd ** 2
    zhat = zhat / prec
    bhat = W["zb_mean"] + W["zb_sd"] * zhat
    # module posterior
    X = gf["MODF"].to_numpy(dtype=float)
    C = W["modf_centroids"]
    logp = np.log(W["module_weights"])[None, :] - ((X[:, None, :] - C[None]) ** 2).sum(-1) / (2 * P["modf_noise"] ** 2)
    logp = np.where(np.isnan(logp), np.log(W["module_weights"])[None, :], logp)   # missing -> prior
    logp -= logp.max(1, keepdims=True)
    pm = np.exp(logp); pm /= pm.sum(1, keepdims=True)
    a, b = W["a"], W["b"]
    eR = np.einsum("nm,mk,nk->n", pm[a], W["R"], pm[b])
    eS = (pm[a] * pm[b]).sum(1)
    zp = W["pair"]["PAIRINF__z"][a, b].astype(float)
    zpost = np.where(np.isnan(zp), 0.0, zp / (1 + P["pair_inf_noise"] ** 2))
    return bhat[a] + bhat[b] + P["gamma_between"] * eR + P["gamma_within"] * eS + P["eta"] * zpost


def reference_learner(W, S, heldout_mask, seed=0):
    """Plain LightGBM on gene/pair/sim features + out-of-fold gene target encoding.
    Trained on C1-type pairs (dev x dev) only; evaluated per class. Calibration reference, not an oracle."""
    import lightgbm as lgb
    P, a, b = W["P"], W["a"], W["b"]
    y = (S < P["threshold"]).astype(int)
    cls = classes(W, heldout_mask)
    rng = np.random.default_rng(seed)
    train = cls["C1"] & (rng.random(len(a)) < 0.9)
    gf = W["gene_feats"]
    def gcols(grp): return gf[grp].to_numpy(dtype=np.float32)
    infb, noise, modf = gcols("INFB"), gcols("NOISE"), gcols("MODF")
    feats = {}
    for k in range(infb.shape[1]):
        feats[f"infb{k}_min"] = np.fmin(infb[a, k], infb[b, k]); feats[f"infb{k}_max"] = np.fmax(infb[a, k], infb[b, k])
    for k in range(noise.shape[1]):
        feats[f"noise{k}_min"] = np.fmin(noise[a, k], noise[b, k]); feats[f"noise{k}_max"] = np.fmax(noise[a, k], noise[b, k])
    for k in range(modf.shape[1]):
        feats[f"modf{k}_a"] = modf[a, k]; feats[f"modf{k}_b"] = modf[b, k]
    for nm, M_ in {**W["pair"], **W["sim"]}.items():
        feats[nm] = M_[a, b]
    X = pd.DataFrame(feats)
    # out-of-fold target encoding of each gene's hit rate (train pairs only)
    G = W["G"]
    fold = rng.integers(0, 5, len(a))
    def enc(mask_rows, src_mask):
        pos = np.zeros(G); cnt = np.zeros(G)
        for g_ in (a, b):
            np.add.at(pos, g_[src_mask], y[src_mask]); np.add.at(cnt, g_[src_mask], 1)
        return pos, cnt
    prior = y[train].mean(); m = 20.0
    te = np.full((len(a), 2), np.nan, dtype=np.float32)
    for f in range(5):
        src = train & (fold != f)
        pos, cnt = enc(None, src)
        rows = np.where(train & (fold == f))[0]
        for j, g_ in enumerate((a, b)):
            te[rows, j] = np.where(cnt[g_[rows]] > 0, (pos[g_[rows]] + m * prior) / (cnt[g_[rows]] + m), np.nan)
    pos, cnt = enc(None, train)
    rows = np.where(~train)[0]
    for j, g_ in enumerate((a, b)):
        te[rows, j] = np.where(cnt[g_[rows]] > 0, (pos[g_[rows]] + m * prior) / (cnt[g_[rows]] + m), np.nan)
    X["te_min"] = np.fmin(te[:, 0], te[:, 1]); X["te_max"] = np.fmax(te[:, 0], te[:, 1])
    X["te_sum"] = np.nansum(te, axis=1)
    mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                             n_jobs=4)
    mdl.fit(X[train], y[train])
    sc = mdl.predict_proba(X[~train])[:, 1]
    out = {}
    held = np.where(~train)[0]
    for k, msk in cls.items():
        sel = msk[held]
        out[k] = ap(y[held][sel], sc[sel])
    return out


def known_answers(W, S, heldout_mask, tag: str, with_reference=True, permuted=False) -> dict:
    P = W["P"]
    thr = P["threshold"]
    y = S < thr
    cls = classes(W, heldout_mask)
    sig_true = W["signal"]               # noiseless signal (higher = more negative)
    prop_only = W["beta_sum"]
    feat_or = posterior_scores(W)
    out = {"dataset": tag, "note_signal_direction": "scores: higher = more negative interaction",
           "n_genes_universe": int(W["G"]), "n_genes_measured": int(W["measured"].sum()),
           "n_pairs": int(len(y)), "sigma_noise": float(W["sigma"]),
           "prevalence_realized": float(y.mean()), "prevalence_target": P["prev_target"],
           "prevalence_expected_analytic": float(norm.cdf((thr + W["centered"]) / W["sigma"]).mean()),
           "heldout": dict(frac=P["heldout_frac"], seed=P["heldout_seed"],
                           n_genes=int(heldout_mask.sum()),
                           genes=[str(g) for g in W["ids"][heldout_mask]],
                           definition="rng=np.random.default_rng(0); sorted measured genes; choice(round(0.2*n)) w/o replacement"),
           "class_sizes": {k: dict(n=int(m.sum()), n_pos=int(y[m].sum())) for k, m in cls.items()}}
    scores = {"oracle": sig_true, "propensity_only_oracle": prop_only, "feature_informed_oracle": feat_or}
    out["auprc"] = {nm: {k: ap(y[m], s[m]) for k, m in cls.items()} for nm, s in scores.items()}
    out["auprc"]["prevalence_baseline"] = {k: float(y[m].mean()) for k, m in cls.items()}

    # expected distributions over noise redraws (same latent, same sigma), or label permutations
    rr = rng_for(P["seed"], "reps")
    nrep = P["n_reps"]
    centered = W["centered"]
    dist = {nm: {k: [] for k in cls} for nm in ("oracle", "propensity_only_oracle", "feature_informed_oracle", "random_score")}
    for _ in range(nrep):
        Sr = -centered + W["sigma"] * rr.standard_normal(len(y))
        yr = Sr < thr
        rs = rr.random(len(y))
        for k, m in cls.items():
            for nm, s in scores.items():
                dist[nm][k].append(ap(yr[m], s[m]))
            dist["random_score"][k].append(ap(yr[m], rs[m]))
    def summ(v):
        v = np.array(v); return dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                                     p01=float(np.quantile(v, .01)), p99=float(np.quantile(v, .99)))
    out["auprc_expected_over_noise_redraws"] = {nm: {k: summ(v) for k, v in d.items()} for nm, d in dist.items()}
    out["negative_control"] = {
        "expected_auprc_of_any_score_on_permuted_labels": "prevalence",
        "per_class": {k: dict(expected=float(y[m].mean()),
                              random_score_sd=out["auprc_expected_over_noise_redraws"]["random_score"][k]["sd"],
                              random_score_mean_over_redraws=out["auprc_expected_over_noise_redraws"]["random_score"][k]["mean"],
                              band_99=[out["auprc_expected_over_noise_redraws"]["random_score"][k]["p01"],
                                       out["auprc_expected_over_noise_redraws"]["random_score"][k]["p99"]])
                      for k, m in cls.items()},
        "rule": "pooled/all-pairs AUPRC of a permuted-label run must lie in prevalence +/- max(0.25*prevalence, 3*sd); "
                "per-class bands (esp. C3, ~100 positives) are wider: use band_99.",
    }
    fio, orc, ex = out["auprc"]["feature_informed_oracle"], out["auprc"]["oracle"], out["auprc_expected_over_noise_redraws"]
    B = P["bands"]
    out["recommended_bands_pre_stated"] = {
        "reasoning": ("The latent signal is beta_a+beta_b (+module-pair terms, +pair covariate). A C3 pair has no labelled gene, so a "
                      "learner can only use features; its ceiling is the feature-informed oracle (posterior-mean beta, module "
                      "posterior, pair covariate, computed with the TRUE noise levels but no labels). A realistic learner (noisy "
                      "estimation of weights, 40 modules to learn, ~1e2 C3 positives) should recover a quarter of that ceiling "
                      "(floor) and a good one half. Fractions fixed before looking at pipeline results; the LightGBM reference "
                      "learner below is a calibration that a naive pipeline clears the C3 floor by a small margin."),
        "positive_control_floor": {k: B["floor_frac_of_feature_oracle"] * fio[k] for k in cls},
        "positive_control_good": {k: B["good_frac_of_feature_oracle"] * fio[k] for k in cls},
        "soft_ceiling_features_only": {k: B["soft_ceiling_mult_feature_oracle"] * fio[k] for k in cls},
        "hard_ceiling_leak_flag_oracle_p99": {k: ex["oracle"][k]["p99"] for k in cls},
        "rules": [
            "C3 AUPRC >= positive_control_floor['C3'] (pipeline using informative features; also >= 4x prevalence).",
            "pooled/all-pairs AUPRC >= positive_control_floor['all'].",
            "No class AUPRC may exceed hard_ceiling_leak_flag_oracle_p99 (oracle upper 99th pct over noise redraws): exceeding it means label leakage.",
            "A features-only model (no label-derived features) exceeding soft_ceiling_features_only['C3'] warrants a leakage check.",
            "Propensity-only model on C3 (gene features -> beta) should beat prevalence by >= 3x but stay below the oracle.",
            "Models using training labels (target encoding / NBR): expect C1 >= C3.",
            "Noise-only features (NOISE group, NOISESIM, PAIRNOISE): importance/ablation gain must be within noise; AUPRC with only these ~ prevalence band."],
        "tolerance_note": "per-class sampling sd over noise redraws is in auprc_expected_over_noise_redraws; C3 has only ~1e2 positives, so allow +/-2 sd.",
    }
    if with_reference:
        out["reference_learner_lightgbm_trained_on_C1_pairs"] = reference_learner(W, S, heldout_mask)
    if permuted:
        out["oracle_auprc_on_permuted_labels_should_equal_prevalence"] = out["auprc"]["oracle"]
    return out


def heldout_genes(W, P):
    meas = np.where(W["measured"])[0]
    r = np.random.default_rng(P["heldout_seed"])
    k = int(round(P["heldout_frac"] * len(meas)))
    ch = r.choice(meas, size=k, replace=False)
    m = np.zeros(W["G"], bool); m[ch] = True
    return m


# --------------------------------------------------------------------------- IO
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write_all(W, S, out: Path, permuted: bool):
    out.mkdir(parents=True, exist_ok=True)
    feat = out / "features"
    for d in ("gene", "sim", "pair"):
        (feat / d).mkdir(parents=True, exist_ok=True)
    (feat / "genes.txt").write_text("\n".join(W["ids"]) + "\n")
    gene_table(W).to_csv(out / "ryan_genes.tsv", sep="\t", index=False)
    pair_table(W, S).to_parquet(out / "ryan_pairs.parquet", index=False)
    man = {}
    for grp, df in W["gene_feats"].items():
        p = feat / "gene" / f"{grp}.parquet"
        df.to_parquet(p)
        man[f"gene/{grp}.parquet"] = dict(group=grp, sources=[SOURCE_ID], script=SCRIPT_REL, sha256=sha256(p))
    for nm, M_ in W["sim"].items():
        p = feat / "sim" / f"{nm}.npy"; np.save(p, M_)
        man[f"sim/{nm}.npy"] = dict(group=nm, sources=[SOURCE_ID], script=SCRIPT_REL, sha256=sha256(p))
    for nm, M_ in W["pair"].items():
        p = feat / "pair" / f"{nm}.npy"; np.save(p, M_)
        man[f"pair/{nm}.npy"] = dict(group=nm.split("__")[0], sources=[SOURCE_ID], script=SCRIPT_REL, sha256=sha256(p))
    (feat / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    # latent truth (for tests/known answers ONLY; pipelines must never read this)
    td = out / "_truth"; td.mkdir(exist_ok=True)
    pd.DataFrame(dict(gene=W["ids"], role=W["role"], module=W["module"], beta=W["beta"])).to_csv(
        td / "gene_truth.tsv", sep="\t", index=False)
    pd.DataFrame(dict(gene_a=W["ids"][W["a"]], gene_b=W["ids"][W["b"]], signal=W["signal"],
                      beta_sum=W["beta_sum"])).to_parquet(td / "pair_signal.parquet", index=False)
    (td / "README.txt").write_text("Latent truth of the toy generator. For tests only; never read from pipeline code.\n")


def main(argv=None):
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--permute-labels", action="store_true")
    ap_.add_argument("--seed", type=int, default=PARAMS["seed"])
    ap_.add_argument("--out", type=Path, default=None)
    ap_.add_argument("--no-reference", action="store_true", help="skip the LightGBM reference learner")
    ap_.add_argument("--reps", type=int, default=PARAMS["n_reps"])
    args = ap_.parse_args(argv)
    P = dict(PARAMS, seed=args.seed, n_reps=args.reps)
    out = args.out or ROOT / "data" / ("toy_permuted" if args.permute_labels else "toy")
    W = build_world(P)
    S = W["S"].copy()
    if args.permute_labels:
        S = rng_for(P["seed"], "permute").permutation(S)
    write_all(W, S, out, args.permute_labels)
    held = heldout_genes(W, P)
    ka = known_answers(W, S, held, "toy_permuted" if args.permute_labels else "toy", with_reference=not args.no_reference, permuted=args.permute_labels)
    ka["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}
    (out / "known_answers.json").write_text(json.dumps(ka, indent=2))
    print(f"wrote {out}  prevalence={ka['prevalence_realized']:.4f}  n_pairs={ka['n_pairs']}")
    print(json.dumps(ka["auprc"], indent=1))
    return ka


if __name__ == "__main__":
    main()
