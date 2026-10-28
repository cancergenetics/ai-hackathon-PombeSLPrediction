#!/usr/bin/env python
"""XSP: cross-species transfer of Costanzo 2016 genetic-interaction knowledge to S. pombe through
SPECIES-AGNOSTIC features (DESIGN.md §5 M4, "full transfer model restricted to features available in both
species").  Label-free with respect to S. pombe: no Ryan / pombe GI data is read.

Sources (config/data_allowlist.yaml dataset ids)
  costanzo2016      Data File S1 SGA_{NxN,ExN,ExE,DAmP}.txt + strain classes (allele collapse and
                    essentiality exactly as scripts/features/build_cer.py; its functions are imported)
  biogrid_physical  BioGRID 5.0.262 physical.  pombe: the tier-1 network (pair/PPI__edge.npy, system filter +
                    pombe-GI PMID blocklist).  cerevisiae: same protein-protein system filter
                    (drop Affinity Capture-RNA, Protein-RNA, Co-localization), no PMID blocklist.
  esm2_weights      ESM-2 650M mean-pooled embeddings (data/processed/esm2_650M_{pombe,cerevisiae}.parquet)
  pombase_protein   peptide.fa.gz (pombe sequences);  sgd_protein  orf_trans_all.fasta.gz (R64-5-1)
  ygob              Pillars.tab ohnolog pairs: ONLY to keep ohnologs in the same cerevisiae CV gene fold
  pombase_orthologs ONLY for the label-free transfer diagnostic (never a model input)

Species universes: pombe = genes.txt (5,126 protein-coding); cerevisiae = SGD Verified + Uncharacterized ORFs
with an ESM embedding (Dubious ORFs, pseudogenes and transposable-element genes excluded so both proteomes
are comparable).  Every gene feature is rank-normalised WITHIN species; continuous pair features are mapped
through the species' own ECDF (src/xsp.py).

Gene level   targets on cerevisiae Costanzo genes: prop = logit smoothed negative-interaction rate over all
             tested partners; prop_nn = same over non-essential partners only; ess = Costanzo-essential.
             LightGBM; gene-held-out 5-fold CV (ohnologs share a fold).  Applied to pombe ->
             XSP__prop_hat, XSP__prop_nn_hat, XSP__ess_hat.
Pair level   label = collapsed Costanzo negative (eps < -0.08 & p < 0.05) AND eps < T, T chosen so that the
             prevalence over measured universe pairs is 2%.  LightGBM on N_PAIRS sampled measured pairs,
             symmetric features (gene min/max, ESM cosine/mutual rank, PPI edge/shared/Jaccard/sp/RWR, gene
             transfer predictions as nested out-of-fold values).  Gene-held-out CV: C3-like (both genes
             held out) and C2-like (one held out).  Final model on all sampled pairs -> pombe G x G
             XSP__pair_hat (symmetric, NaN diagonal).
Diagnostic   ortholog-held-out transfer proxy: the fold-k models applied to POMBE features of genes whose
             1:1 curated cerevisiae ortholog is in held-out fold k, scored against the ortholog's Costanzo
             value -> how much accuracy survives the species switch (no pombe GI data involved).

Outputs go to the STAGING store data/processed/features_staging/ (gene/XSP.parquet, pair/XSP__pair_hat.npy,
manifest_xsp.json); checks -> results/features/xsp_checks.json; runtime -> results/runtime/features_xsp.json.

Run: .venv/bin/python -I scripts/features/build_xsp.py [--quick] [--ppi-variant full|sub]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import xsp  # noqa: E402
from src.idmap import IDMapper  # noqa: E402
from src.orthologs import load_orthologs  # noqa: E402

SCRIPT = "scripts/features/build_xsp.py"
STAGE = fs.data_root() / "features_staging"
CACHE = STAGE / "cache"
ESM_P = ROOT / "data/processed/esm2_650M_pombe.parquet"
ESM_C = ROOT / "data/processed/esm2_650M_cerevisiae.parquet"
PEP_P = ROOT / "data/raw/pombase_protein/peptide.fa.gz"
PEP_C = ROOT / "data/raw/sgd_protein/orf_trans_all.fasta.gz"
BIOGRID_C = ROOT / "data/raw/biogrid/biogrid_5.0.262_scerevisiae_physical.tab3.tsv.gz"
DROP_SYSTEMS = {"Affinity Capture-RNA", "Protein-RNA", "Co-localization"}
CER_KEEP_QUALIFIERS = ("Verified ORF", "Uncharacterized ORF")
CHECKS = ROOT / "results/features/xsp_checks.json"
SOURCES = ["costanzo2016", "biogrid_physical", "esm2_weights", "pombase_protein", "sgd_protein", "ygob"]
SEED = 0
K_FOLDS = 5
N_PC = 32
N_PC_PAIR = 16
PREVALENCE = 0.02
N_PAIRS = 2_500_000
N_JOBS = 6
PAIR_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   metric="average_precision", verbose=-1, seed=SEED, deterministic=True, force_row_wise=True,
                   num_threads=N_JOBS)
GENE_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED,
                   deterministic=True, force_row_wise=True, num_threads=N_JOBS)
GENE_ROUNDS = 400
BASE_PAIR_GENE_COLS = ["ppi_degree", "ppi_pagerank", "ppi_clustering", "ppi_core", "ppi_nbr_degree",
                       "para_top1", "para_top5", "para_n50", "log_len", "hydrophobic", "disorder_prom",
                       "net_charge", "charged"]
PAIR_GENE_COLS = BASE_PAIR_GENE_COLS + [f"esm_pc{k + 1}" for k in range(N_PC_PAIR)]
HAT = ["prop_hat", "prop_nn_hat", "ess_hat"]
PAIR_MATS = ["esm_cos", "esm_mr", "edge", "shared", "jaccard", "sp", "rwr"]


def load_build_cer():
    spec = importlib.util.spec_from_file_location("build_cer", ROOT / "scripts/features/build_cer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- inputs
def costanzo_tables() -> tuple[pd.DataFrame, dict, list]:
    """Allele-collapsed ORF pairs (orf_a < orf_b, eps, neg) + ORF essentiality (cached in staging)."""
    pq, js = CACHE / "costanzo_collapsed.parquet", CACHE / "costanzo_essential.json"
    if pq.exists() and js.exists():
        meta = json.loads(js.read_text())
        return pd.read_parquet(pq), meta["essential"], meta["ohnologs"]
    bc = load_build_cer()
    strains, q, a, eps, p = bc.read_measurements()
    orfs, op, info = bc.collapse(strains, q, a, eps, p)
    st = pd.read_csv(bc.STRAINS, sep="\t")
    cls = pd.concat([st[["Strain ID"]], pd.DataFrame({"Strain ID": strains})]).drop_duplicates()
    cls["orf"] = cls["Strain ID"].str.rsplit("_", n=1).str[0]
    cls["cls"] = cls["Strain ID"].str.rsplit("_", n=1).str[1].str.extract(r"^([a-z]+)", expand=False)
    byorf = cls.groupby("orf").cls.agg(lambda s: set(s.dropna()))
    essential = {o: bool(c & bc.ESS_CLASSES) and not (c & bc.DEL_CLASSES) for o, c in byorf.items()}
    bc.fetch_ygob()
    pil = pd.read_csv(bc.YGOB, sep="\t", header=None, dtype=str)
    orf_re = r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$"
    bt = pil[(pil[11] != "---") & (pil[21] != "---")]
    bt = bt[bt[11].str.match(orf_re) & bt[21].str.match(orf_re)]
    oh = [list(x) for x in bt[[11, 21]].values]
    o = np.array(orfs)
    df = pd.DataFrame({"orf_a": o[op.i.values], "orf_b": o[op.j.values], "eps": op.eps.values.astype(np.float32),
                       "neg": op.neg.values.astype(bool)})
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq)
    js.write_text(json.dumps({"essential": essential, "ohnologs": oh, "collapse_info": info}))
    return df, essential, oh


def cer_universe(esm_index) -> list[str]:
    keep = []
    with gzip.open(PEP_C, "rt") as fh:
        for line in fh:
            if line.startswith(">") and any(q in line for q in CER_KEEP_QUALIFIERS):
                keep.append(line[1:].split()[0])
    s = set(esm_index)
    return sorted(g for g in keep if g in s)


def cer_network(genes: list[str]) -> tuple[np.ndarray, dict]:
    c = pd.read_csv(BIOGRID_C, sep="\t", dtype=str, usecols=[
        "Systematic Name Interactor A", "Systematic Name Interactor B", "Experimental System",
        "Experimental System Type", "Organism ID Interactor A", "Organism ID Interactor B"])
    assert set(c["Experimental System Type"]) == {"physical"}
    n0 = len(c)
    c = c[(c["Organism ID Interactor A"] == "559292") & (c["Organism ID Interactor B"] == "559292")]
    c = c[~c["Experimental System"].isin(DROP_SYSTEMS)]
    n1 = len(c)
    gi = {g: i for i, g in enumerate(genes)}
    a = c["Systematic Name Interactor A"].map(gi)
    b = c["Systematic Name Interactor B"].map(gi)
    ok = a.notna() & b.notna() & (a != b)
    e = np.unique(np.sort(np.column_stack([a[ok].astype(int), b[ok].astype(int)]), axis=1), axis=0)
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    A[e[:, 0], e[:, 1]] = 1
    A[e[:, 1], e[:, 0]] = 1
    return A, {"rows": n0, "rows_after_system_filter": n1, "rows_both_in_universe_non_self": int(ok.sum()),
               "edges": int(len(e))}


def build_species(name, genes, emb_pcs, esm_raw, seqs, A, seed) -> xsp.Species:
    t = time.time()
    S = xsp.centred_cosine(esm_raw)
    raw = pd.concat([xsp.seq_stats(seqs, genes).reset_index(drop=True),
                     pd.DataFrame(emb_pcs, columns=[f"esm_pc{k + 1}" for k in range(emb_pcs.shape[1])]),
                     xsp.paralog_features(S), xsp.ppi_gene_features(A)], axis=1)
    raw.index = genes
    gene = xsp.rank_normalise(raw)
    pm = xsp.ppi_pair_matrices(A)
    pair = {"esm_cos": xsp.ecdf_matrix(S, seed=seed), "esm_mr": xsp.mutual_rank(S), "edge": pm["edge"],
            "shared": xsp.ecdf_matrix(pm["shared"], seed=seed), "jaccard": xsp.ecdf_matrix(pm["jaccard"], seed=seed),
            "sp": pm["sp"], "rwr": xsp.ecdf_matrix(pm["rwr"], seed=seed)}
    del S, pm
    print(f"[{name}] species features built in {time.time() - t:.1f}s; n={len(genes)}, gene cols={gene.shape[1]}")
    sp = xsp.Species(name, genes, gene, pair)
    sp.raw = raw
    return sp


# ------------------------------------------------------------------------------------------- gene models
def gene_targets(NEG, MEAS, ess, partner_mask, prior=None):
    """Per-gene smoothed logit negative rates over partners in partner_mask (all / non-essential)."""
    pm = partner_mask.astype(np.float32)
    nn = pm * (~ess).astype(np.float32)
    t_all, n_all = MEAS @ pm, NEG @ pm
    t_nn, n_nn = MEAS @ nn, NEG @ nn
    p0 = prior if prior is not None else float(n_all.sum() / max(t_all.sum(), 1))
    p0n = float(n_nn.sum() / max(t_nn.sum(), 1)) if prior is None else prior
    return {"prop": xsp.logit(xsp.smoothed_rate(n_all, t_all, p0)), "prop_nn": xsp.logit(xsp.smoothed_rate(n_nn, t_nn, p0n)),
            "rate": np.where(t_all > 0, n_all / np.maximum(t_all, 1), np.nan),
            "rate_nn": np.where(t_nn > 0, n_nn / np.maximum(t_nn, 1), np.nan), "tested": t_all, "tested_nn": t_nn}


def fit_gene_models(X, y_prop, y_nn, y_ess, idx):
    out = {}
    for key, y, obj in (("prop_hat", y_prop, "regression"), ("prop_nn_hat", y_nn, "regression"),
                        ("ess_hat", y_ess, "binary")):
        r = idx[np.isfinite(y[idx])]
        out[key] = lgb.train({**GENE_PARAMS, "objective": obj}, lgb.Dataset(X[r], y[r]), GENE_ROUNDS)
    return out


def predict_gene(models, X):
    return np.column_stack([models[k].predict(X) for k in HAT]).astype(np.float32)


def oof_gene(X, tg, y_ess, idx, folds):
    """Out-of-fold gene predictions for genes idx (fold ids `folds` aligned with idx)."""
    out = np.full((X.shape[0], 3), np.nan, dtype=np.float32)
    for f in np.unique(folds):
        tr, te = idx[folds != f], idx[folds == f]
        out[te] = predict_gene(fit_gene_models(X, tg["prop"], tg["prop_nn"], y_ess, tr), X[te])
    return out


# ------------------------------------------------------------------------------------------- pair model
def train_pair(Xtr, ytr, Xva, yva, rounds=None):
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    if rounds is not None:
        return lgb.train(PAIR_PARAMS, dtr, rounds), rounds
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train(PAIR_PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, int(m.best_iteration)


def metrics(y, s):
    y = np.asarray(y, bool)
    if y.sum() == 0 or y.all():
        return {"n": int(len(y)), "n_pos": int(y.sum())}
    return {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
            "auprc": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s)),
            "lift": float(average_precision_score(y, s) / y.mean())}


# ------------------------------------------------------------------------------------------- staging I/O
def stage_manifest(key, entry):
    """Staging manifest (same entry format as the live manifest.json); single writer, atomic replace."""
    man_x = STAGE / "manifest_xsp.json"
    m = json.loads(man_x.read_text()) if man_x.exists() else {}
    m[key] = entry
    tmp = man_x.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(sorted(m.items())), indent=1) + "\n")
    tmp.replace(man_x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="variant check: 1M pairs, folds 0-1, no outputs")
    ap.add_argument("--ppi-variant", choices=["full", "sub"], default="full")
    ap.add_argument("--pair-esm-pcs", type=int, default=N_PC_PAIR, help="# ESM PCs entering the pair model (min/max)")
    args = ap.parse_args()
    global PAIR_GENE_COLS
    PAIR_GENE_COLS = BASE_PAIR_GENE_COLS + [f"esm_pc{k + 1}" for k in range(args.pair_esm_pcs)]
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    rng = np.random.default_rng(SEED)
    n_pairs = 1_000_000 if args.quick else N_PAIRS
    run_folds = [0, 1] if args.quick else list(range(K_FOLDS))
    checks: dict = {"variant": args.ppi_variant, "pair_esm_pcs": args.pair_esm_pcs, "quick": args.quick, "n_pairs_sampled": n_pairs, "seed": SEED,
                    "pair_params": PAIR_PARAMS, "gene_params": GENE_PARAMS, "gene_rounds": GENE_ROUNDS}
    tag = "xsp" if not args.quick else f"xsp_quick_{args.ppi_variant}_pc{args.pair_esm_pcs}"
    with fs.RuntimeLog(tag, variant=args.ppi_variant, quick=args.quick, n_jobs=N_JOBS) as rt:
        # ---------------- inputs
        pg = fs.genes()
        G = len(pg)
        Ep = pd.read_parquet(ESM_P).reindex(pg)
        assert Ep.notna().all().all()
        Ec_all = pd.read_parquet(ESM_C)
        cg = cer_universe(Ec_all.index)
        Ec = Ec_all.reindex(cg)
        del Ec_all
        seq_p = xsp.read_fasta(PEP_P, lambda h: h.split()[0].rsplit(":", 1)[0].rsplit(".", 1)[0])
        seq_c = xsp.read_fasta(PEP_C, lambda h: h.split()[0])
        Ap = np.array(fs.load_matrix("pair", "PPI__edge"), dtype=np.float32)
        np.fill_diagonal(Ap, 0)
        Ac, netinfo = cer_network(cg)
        e_p, e_c = Ap.sum() / 2, Ac.sum() / 2
        frac = float((e_p / G) / (e_c / len(cg)))
        checks["networks"] = {"pombe_edges": int(e_p), "pombe_mean_degree": float(2 * e_p / G),
                              "pombe_frac_isolated": float((Ap.sum(1) == 0).mean()),
                              "cer": netinfo, "cer_mean_degree_full": float(2 * e_c / len(cg)),
                              "cer_frac_isolated_full": float((Ac.sum(1) == 0).mean())}
        if args.ppi_variant == "sub":
            Ac = xsp.subsample_edges(Ac, frac, seed=SEED)
            checks["networks"]["sub_keep_frac"] = frac
            checks["networks"]["cer_edges_sub"] = int(Ac.sum() / 2)
            checks["networks"]["cer_frac_isolated_sub"] = float((Ac.sum(1) == 0).mean())
        checks["n_genes"] = {"pombe": G, "cerevisiae_universe": len(cg),
                             "pombe_with_seq": int(sum(g in seq_p for g in pg)),
                             "cer_with_seq": int(sum(g in seq_c for g in cg))}
        rt.step("inputs")

        pcs, evr = xsp.esm_joint_pca({"p": Ep.values, "c": Ec.values}, N_PC)
        checks["esm_joint_pca_evr_cum"] = float(evr.sum())
        SP = build_species("pombe", pg, pcs["p"], Ep.values, seq_p, Ap, SEED)
        SC = build_species("cerevisiae", cg, pcs["c"], Ec.values, seq_c, Ac, SEED)
        del Ep, Ec, Ap, Ac
        gcols = list(SC.gene.columns)
        assert gcols == list(SP.gene.columns)
        Xg_c, Xg_p = SC.gene.to_numpy(np.float32), SP.gene.to_numpy(np.float32)
        # domain-shift diagnostic: raw medians per species for a few features
        checks["raw_feature_medians"] = {c: {"pombe": float(np.nanmedian(SP.raw[c])),
                                             "cerevisiae": float(np.nanmedian(SC.raw[c]))}
                                         for c in ["log_len", "para_top1", "ppi_degree", "ppi_clustering"]}
        rt.step("species_features")

        # ---------------- Costanzo labels on the cerevisiae universe
        cp, essential, ohn = costanzo_tables()
        ci = SC.index
        n = len(cg)
        a = cp.orf_a.map(ci)
        b = cp.orf_b.map(ci)
        ok = a.notna() & b.notna()
        cp = cp[ok]
        ia, ib = a[ok].astype(int).values, b[ok].astype(int).values
        negs = cp.neg.values
        n_neg_needed = int(round(PREVALENCE * len(cp)))
        eps_neg = np.sort(cp.eps.values[negs])
        T = float(eps_neg[n_neg_needed]) if n_neg_needed < len(eps_neg) else 0.0
        lab = negs & (cp.eps.values < T)
        checks["labels"] = {"n_measured_universe_pairs": int(len(cp)), "frac_neg_costanzo": float(negs.mean()),
                            "eps_threshold_T": T, "prevalence": float(lab.mean()),
                            "n_costanzo_orf_pairs_dropped_not_in_universe": int((~ok).sum())}
        L = np.full((n, n), -1, dtype=np.int8)
        L[ia, ib] = lab
        L[ib, ia] = lab
        NEG = np.zeros((n, n), dtype=np.float32)
        NEG[ia, ib] = negs
        NEG[ib, ia] = negs
        MEAS = (L >= 0).astype(np.float32)
        ess = np.array([essential.get(g, False) for g in cg])
        in_cost = MEAS.sum(1) > 0
        cidx = np.where(in_cost)[0]
        y_ess = np.where(in_cost, ess.astype(float), np.nan)
        checks["labels"]["n_costanzo_genes_in_universe"] = int(in_cost.sum())
        checks["labels"]["n_essential"] = int(ess[in_cost].sum())
        folds_all = np.full(n, -1)
        folds_all[cidx] = xsp.gene_folds([cg[i] for i in cidx], [tuple(x) for x in ohn], K_FOLDS, SEED)
        rt.step("labels")

        # ---------------- gene level CV (targets over all partners)
        tg_all = gene_targets(NEG, MEAS, ess, in_cost)
        oof = oof_gene(Xg_c, tg_all, y_ess, cidx, folds_all[cidx])
        g_cv = {}
        for k, key, rkey in ((0, "prop_hat", "rate"), (1, "prop_nn_hat", "rate_nn")):
            r = cidx[np.isfinite(tg_all[rkey][cidx])]
            g_cv[key] = {"spearman_vs_rate": float(spearmanr(oof[r, k], tg_all[rkey][r]).statistic), "n": int(len(r))}
            nn = r[~ess[r]]
            g_cv[key]["spearman_nonessential_genes"] = float(spearmanr(oof[nn, k], tg_all[rkey][nn]).statistic)
            g_cv[key]["per_fold"] = [float(spearmanr(oof[r[folds_all[r] == f], k], tg_all[rkey][r[folds_all[r] == f]]).statistic)
                                     for f in range(K_FOLDS)]
        g_cv["ess_hat"] = metrics(ess[cidx], oof[cidx, 2])
        # single-feature references (rank-normalised raw features)
        ref = {}
        for c in ["ppi_degree", "ppi_pagerank", "log_len", "para_top1", "esm_pc1"]:
            ref[c] = float(spearmanr(SC.gene[c].values[cidx], tg_all["rate"][cidx], nan_policy="omit").statistic)
        g_cv["single_feature_spearman_vs_rate"] = ref
        g_cv["corr_prop_vs_prop_nn_target"] = float(spearmanr(tg_all["rate"][cidx], tg_all["rate_nn"][cidx],
                                                              nan_policy="omit").statistic)
        checks["gene_cv"] = g_cv
        print(json.dumps(g_cv, indent=1))
        rt.step("gene_cv")

        # ---------------- 1:1 orthologs (diagnostic only)
        orth = load_orthologs(IDMapper())
        orth = orth[orth.pombe.isin(SP.index) & orth.sc.isin(ci)]
        orth = orth[~orth.pombe.duplicated(keep=False) & ~orth.sc.duplicated(keep=False)]
        o_p = orth.pombe.map(SP.index).values
        o_c = orth.sc.map(ci).values
        o_ok = in_cost[o_c]
        o_p, o_c = o_p[o_ok], o_c[o_ok]
        checks["orthologs_1to1_costanzo"] = int(len(o_p))

        # ---------------- pair sample
        iu_a, iu_b = ia, ib
        sel = rng.choice(len(iu_a), size=min(n_pairs, len(iu_a)), replace=False)
        sel.sort()
        pa, pb = iu_a[sel], iu_b[sel]
        py = L[pa, pb].astype(np.int8)
        checks["pair_sample"] = {"n": int(len(pa)), "prevalence": float(py.mean())}
        fa, fb = folds_all[pa], folds_all[pb]
        rt.step("pair_sample")

        # ---------------- pair-level gene-held-out CV
        fnames = xsp.pair_feature_names(PAIR_GENE_COLS + HAT, PAIR_MATS)
        cv, best_iters, gproxy = [], [], []
        prox_p_s, prox_p_y, prox_c_s = [], [], []
        c3_s, c3_y, c3_nn, c3_b, prox_b = [], [], [], [], []
        for f in run_folds:
            t0 = time.time()
            Tm = in_cost & (folds_all != f)
            tr_idx = np.where(Tm)[0]
            te_idx = np.where(folds_all == f)[0]
            tg = gene_targets(NEG, MEAS, ess, Tm)                  # rates over training partners only
            hat = np.full((n, 3), np.nan, dtype=np.float32)
            inner = xsp.gene_folds([cg[i] for i in tr_idx], [tuple(x) for x in ohn], K_FOLDS, SEED + 1 + f)
            hat[tr_idx] = oof_gene(Xg_c, tg, y_ess, tr_idx, inner)[tr_idx]
            gm = fit_gene_models(Xg_c, tg["prop"], tg["prop_nn"], y_ess, tr_idx)
            hat[te_idx] = predict_gene(gm, Xg_c[te_idx])
            hat_p = predict_gene(gm, Xg_p)
            # gene-level ortholog-held-out transfer proxy (pombe features -> ortholog's Costanzo rate)
            sel_o = folds_all[o_c] == f
            gproxy.append(pd.DataFrame({"p_hat": hat_p[o_p[sel_o], 0], "c_hat": hat[o_c[sel_o], 0],
                                        "rate": tg_all["rate"][o_c[sel_o]]}))
            both_tr = (fa != f) & (fb != f)
            both_te = (fa == f) & (fb == f)
            one_te = (fa == f) ^ (fb == f)
            tr_rows = np.where(both_tr)[0]
            vg = rng.random(n) < 0.15                              # validation genes (early stopping)
            va = vg[pa[tr_rows]] | vg[pb[tr_rows]]
            Xtr = xsp.assemble(SC, pa[tr_rows], pb[tr_rows], PAIR_GENE_COLS + HAT, PAIR_MATS, hat)
            m, bi = train_pair(Xtr[~va], py[tr_rows][~va], Xtr[va], py[tr_rows][va])
            del Xtr
            best_iters.append(bi)
            res = {"fold": f, "best_iter": bi, "n_train": int((~va).sum()), "n_valid": int(va.sum())}
            for nm, rows in (("C3_both_heldout", np.where(both_te)[0]), ("C2_one_heldout", np.where(one_te)[0])):
                X = xsp.assemble(SC, pa[rows], pb[rows], PAIR_GENE_COLS + HAT, PAIR_MATS, hat)
                s = m.predict(X)
                res[nm] = metrics(py[rows], s)
                res[nm + "_baseline_prop_sum"] = metrics(py[rows], hat[pa[rows], 0] + hat[pb[rows], 0])
                if nm.startswith("C3"):
                    c3_s.append(s)
                    c3_y.append(py[rows])
                    c3_nn.append(~ess[pa[rows]] & ~ess[pb[rows]])
                    c3_b.append(hat[pa[rows], 0] + hat[pb[rows], 0])
                    nnm = ~ess[pa[rows]] & ~ess[pb[rows]]
                    res["C3_nxn_only"] = metrics(py[rows][nnm], s[nnm])
            # pair-level ortholog transfer proxy: pombe pairs whose 1:1 orthologs are both in held-out fold f
            op_ = o_p[sel_o]
            oc_ = o_c[sel_o]
            ii, jj = np.triu_indices(len(op_), 1)
            yy = L[oc_[ii], oc_[jj]]
            mm = yy >= 0
            ii, jj, yy = ii[mm], jj[mm], yy[mm]
            Xp = xsp.assemble(SP, op_[ii], op_[jj], PAIR_GENE_COLS + HAT, PAIR_MATS, hat_p)
            Xc = xsp.assemble(SC, oc_[ii], oc_[jj], PAIR_GENE_COLS + HAT, PAIR_MATS, hat)
            sp_, sc_ = m.predict(Xp), m.predict(Xc)
            res["ortholog_proxy_pombe_features"] = metrics(yy, sp_)
            res["ortholog_proxy_cer_features_same_pairs"] = metrics(yy, sc_)
            res["ortholog_proxy_spearman_pombe_vs_cer_score"] = float(spearmanr(sp_, sc_).statistic)
            prox_p_s.append(sp_)
            prox_p_y.append(yy)
            prox_c_s.append(sc_)
            prox_b.append(hat_p[op_[ii], 0] + hat_p[op_[jj], 0])
            if f == run_folds[0]:
                imp = pd.Series(m.feature_importance("gain"), index=fnames)
                checks["pair_gain_importance_fold0_top25"] = (imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_dict()
            res["wall_s"] = round(time.time() - t0, 1)
            cv.append(res)
            print(json.dumps(res, indent=1))
            del m
        gp = pd.concat(gproxy)
        checks["gene_ortholog_proxy"] = {
            "n": int(len(gp)),
            "spearman_pombe_features_vs_ortholog_rate": float(spearmanr(gp.p_hat, gp.rate, nan_policy="omit").statistic),
            "spearman_cer_features_vs_rate_same_genes": float(spearmanr(gp.c_hat, gp.rate, nan_policy="omit").statistic),
            "spearman_pombe_hat_vs_cer_hat": float(spearmanr(gp.p_hat, gp.c_hat).statistic)}
        checks["pair_cv"] = cv
        checks["pair_cv_pooled"] = {
            "C3_both_heldout": metrics(np.concatenate(c3_y), np.concatenate(c3_s)),
            "C3_nxn_only": metrics(np.concatenate(c3_y)[np.concatenate(c3_nn)], np.concatenate(c3_s)[np.concatenate(c3_nn)]),
            "C3_baseline_prop_sum": metrics(np.concatenate(c3_y), np.concatenate(c3_b)),
            "ortholog_proxy_pombe_baseline_prop_sum": metrics(np.concatenate(prox_p_y), np.concatenate(prox_b)),
            "C3_mean_fold_auprc": float(np.mean([r["C3_both_heldout"]["auprc"] for r in cv])),
            "C2_mean_fold_auprc": float(np.mean([r["C2_one_heldout"]["auprc"] for r in cv])),
            "ortholog_proxy_pombe_features": metrics(np.concatenate(prox_p_y), np.concatenate(prox_p_s)),
            "ortholog_proxy_cer_features": metrics(np.concatenate(prox_p_y), np.concatenate(prox_c_s)),
            "best_iters": best_iters}
        print(json.dumps({k: checks[k] for k in ("gene_ortholog_proxy", "pair_cv_pooled")}, indent=1))
        rt.step("pair_cv")

        if args.quick:
            out = CHECKS.with_name(f"{tag}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(checks, indent=1, default=str) + "\n")
            return 0

        # ---------------- final models
        gm_all = fit_gene_models(Xg_c, tg_all["prop"], tg_all["prop_nn"], y_ess, cidx)
        hat_p = predict_gene(gm_all, Xg_p)
        hat_c = oof  # 5-fold OOF over all Costanzo genes, targets over all partners
        rounds = int(np.median(best_iters) * 1.1)
        X = xsp.assemble(SC, pa, pb, PAIR_GENE_COLS + HAT, PAIR_MATS, hat_c)
        m_final, _ = train_pair(X, py, None, None, rounds=rounds)
        del X
        checks["final"] = {"rounds": rounds, "n_train_pairs": int(len(pa))}
        mdir = STAGE / "models"
        mdir.mkdir(parents=True, exist_ok=True)
        m_final.save_model(str(mdir / "xsp_pair_lgbm.txt"))
        for k, mod in gm_all.items():
            mod.save_model(str(mdir / f"xsp_gene_{k}.txt"))
        rt.step("final_fit")

        # ---------------- apply to all pombe pairs
        P = np.full((G, G), np.nan, dtype=np.float32)
        rows_per = 200
        for s in range(0, G, rows_per):
            ii, jj = [], []
            for r in range(s, min(s + rows_per, G)):
                ii.append(np.full(G - r - 1, r))
                jj.append(np.arange(r + 1, G))
            ii, jj = np.concatenate(ii), np.concatenate(jj)
            if len(ii) == 0:
                continue
            sc = m_final.predict(xsp.assemble(SP, ii, jj, PAIR_GENE_COLS + HAT, PAIR_MATS, hat_p)).astype(np.float32)
            P[ii, jj] = sc
            P[jj, ii] = sc
        np.fill_diagonal(P, np.nan)
        # symmetry spot check: assemble(b, a) == assemble(a, b)
        tri = np.random.default_rng(1).integers(0, G, (20000, 2))
        tri = tri[tri[:, 0] != tri[:, 1]]
        Xab = xsp.assemble(SP, tri[:, 0], tri[:, 1], PAIR_GENE_COLS + HAT, PAIR_MATS, hat_p)
        Xba = xsp.assemble(SP, tri[:, 1], tri[:, 0], PAIR_GENE_COLS + HAT, PAIR_MATS, hat_p)
        assert np.array_equal(Xab, Xba, equal_nan=True)
        checks["pombe_apply"] = {"pair_hat_quantiles": {q: float(np.nanquantile(P[np.triu_indices(G, 1)], q))
                                                        for q in (0.01, 0.5, 0.9, 0.99, 0.999)},
                                 "symmetric_features_checked": int(len(tri))}
        rt.step("apply_pombe")

        # ---------------- write staging store
        gdf = pd.DataFrame(hat_p, index=pg, columns=[f"XSP__{h}" for h in HAT]).astype(np.float32)
        gdf.index.name = "gene"
        gpath = STAGE / "gene" / "XSP.parquet"
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(gpath)
        desc = {"XSP__prop_hat": "cerevisiae-trained prediction (logit) of a gene's Costanzo negative-GI rate over all "
                                 "partners from species-agnostic features (ESM, sequence, PPI topology, ESM paralogy)",
                "XSP__prop_nn_hat": "same, rate over non-essential partners only",
                "XSP__ess_hat": "cerevisiae-trained probability of Costanzo essentiality from the same features"}
        built = dt.datetime.now().isoformat(timespec="seconds")
        sfiles = ["costanzo2016:Data File S1 SGA_*.txt + strain classes (allele collapse of build_cer.py)",
                  f"biogrid_physical:{BIOGRID_C.name} (systems not in {sorted(DROP_SYSTEMS)}); pombe network = "
                  f"pair/PPI__edge.npy (tier-1 incl. PMID blocklist)",
                  "esm2_weights:data/processed/esm2_650M_{pombe,cerevisiae}.parquet",
                  "pombase_protein:peptide.fa.gz", "sgd_protein:orf_trans_all.fasta.gz (Verified+Uncharacterized ORFs)",
                  "ygob:Pillars.tab (ohnolog pairs share a CV fold only)"]
        stage_manifest("gene/XSP.parquet", {
            "group": "XSP", "sources": SOURCES, "script": SCRIPT, "sha256": fs.sha256(gpath), "built": built,
            "kind": "gene", "ppi_variant": args.ppi_variant, "source_files": sfiles,
            "columns": {c: {"n_nonmissing": int(gdf[c].notna().sum()), "description": desc[c]} for c in gdf.columns}})
        fs.check_square(P, G, "XSP__pair_hat", sim=False)
        ppath = STAGE / "pair" / "XSP__pair_hat.npy"
        ppath.parent.mkdir(parents=True, exist_ok=True)
        np.save(ppath, P)
        iu = np.triu_indices(G, 1)
        stage_manifest("pair/XSP__pair_hat.npy", {
            "group": "XSP", "sources": SOURCES, "script": SCRIPT, "sha256": fs.sha256(ppath), "built": built,
            "kind": "pair", "ppi_variant": args.ppi_variant, "source_files": sfiles,
            "description": f"cerevisiae-trained LightGBM P(strong negative eps; Costanzo eps<-0.08 & p<0.05 & eps<{T:.4f}, "
                           f"prevalence 2%) applied to pombe species-agnostic pair features",
            "n_pairs_nonmissing": int(np.isfinite(P[iu]).sum()), "n_genes_with_any": G})
        rt.step("write")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
