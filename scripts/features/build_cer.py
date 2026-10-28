#!/usr/bin/env python
"""Tier-1 CER feature group: Costanzo 2016 S. cerevisiae GI map transferred through PomBase curated
orthologs (DESIGN.md v3 §4, group CER).

Sources (DATA.md dataset ids): costanzo2016 (Data File S1 pair-wise SGA_{NxN,ExN,ExE,DAmP}.txt: columns
Query/Array Strain ID, Genetic interaction score (eps), P-value; strain_ids_and_single_mutant_fitness.xlsx
via data/processed/costanzo_strains.tsv: strain class + SMF 30C), pombase (curated orthologs),
ygob (YGOB v7-Aug2012 Pillars.tab: S. cerevisiae WGD ohnolog pairs).  hsp90_data.txt is not used.

Allele collapse (DESIGN: "collapse alleles first, then aggregate over ortholog pairs").  NOTE: the core
table data/processed/costanzo_pairs.parquet averages all *measurements* per ORF pair, so it weights
alleles by their number of measurements and carries no per-measurement significance; it is therefore not
used.  Instead, from the raw files:
  stage 1  unordered strain pair: eps_s = mean eps over its measurements (both orientations, DMA26/30
           arrays); neg_s = 1 if >= half of its measurements satisfy eps < -0.08 and p < 0.05
           (Costanzo 2016 intermediate cut-off);
  stage 2  unordered ORF pair (alleles collapsed): eps = mean of eps_s over strain pairs (each allele
           pair weighted equally); neg = 1 if >= half of its strain pairs are neg_s.
ORF essentiality: an ORF is essential iff it has a TS (tsq/tsa) or DAmP strain and no deletion strain
(sn/dma) -- Costanzo strain classes, not SGD annotations (review DA10).

Gene-level (gene/CER.parquet; pooled over a gene's curated orthologs that have Costanzo data):
  has_ortholog, n_orthologs (curated, any), n_orthologs_costanzo,
  any_ortholog_essential (NaN if no ortholog has a Costanzo strain),
  ortholog_smf (mean 30C SMF of deletion strains of non-essential orthologs; NaN if none),
  neg_rate_nxn = # negative partner ORFs / # tested partner ORFs over NxN ORF pairs (both non-essential),
  neg_rate_e   = same over ORF pairs involving >= 1 essential ORF (ExN, NxE, ExE, DAmP screens),
  emb1..emb20  = PCA of ORF eps profiles (missing eps = 0), averaged over orthologs,
  has_ohnolog_orthologs (gene's orthologs include both members of a YGOB ohnolog pair),
  ohnolog_orthologs_neg (... and that ohnolog pair is a negative Costanzo interaction).
Pair-level (pair/CER__*.npy), over ortholog pairs (x in orth(a), y in orth(b), x != y, measured):
  eps_mean, eps_min, eps_kmin (= eps_min - (m_k - m_1), m_k = expected minimum of k iid draws from the
  empirical ORF-pair eps distribution, k = n_pairs; removes the downward bias of a min over k pairs),
  frac_neg, n_pairs, measured (n_pairs > 0),
  ohnolog_sl: 1 iff some ortholog pair (x, y), x != y, is a YGOB ohnolog pair whose collapsed Costanzo
  interaction is negative (neg = 1 above); 0 if both genes have curated orthologs but no such pair; NaN if
  either gene has no curated ortholog,
  shared_ortholog: 1 iff a and b share a curated cerevisiae ortholog (NaN if either has none).
  sim/cer_profile_corr.npy: mean Pearson r of ortholog eps profiles over ortholog combos x != y.
Missing: eps_* NaN when n_pairs == 0; n_pairs / measured are 0 there.

Run: .venv/bin/python -I scripts/features/build_cer.py     (~3-5 min, ~10 GB peak)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src.idmap import IDMapper, log_unmatched  # noqa: E402
from src.orthologs import SOURCE_FILE as ORTH_FILE, incidence, load_orthologs  # noqa: E402

SCRIPT = "scripts/features/build_cer.py"
CDIR = ROOT / "data/raw/costanzo2016/extracted/Data_File_S1_pairwise"
FILES = ["SGA_NxN.txt", "SGA_ExN.txt", "SGA_ExE.txt", "SGA_DAmP.txt"]
STRAINS = ROOT / "data/processed/costanzo_strains.tsv"
YGOB = ROOT / "data/raw/ygob/Pillars.tab"
YGOB_URL = "http://ygob.ucd.ie/ygob/data/v7-Aug2012/Pillars.tab"
YGOB_SHA256 = "162c1919b762f34006d0be8ea00b090531ea6c67be43d63b6bf683c4bc48212b"   # retrieved 2026-10-28
EPS_CUT, P_CUT = -0.08, 0.05
N_EMB = 20
MIN_PROFILE_PARTNERS = 50
SEED = 0
CHECKS = ROOT / "results/features/cer_checks.json"
SOURCES = ["costanzo2016", "pombase_orthologs"]
ESS_CLASSES, DEL_CLASSES = {"tsq", "tsa", "damp"}, {"sn", "dma"}


def find(name: str) -> Path:
    hits = list(CDIR.rglob(name))
    assert len(hits) == 1, (name, hits)
    return hits[0]


def read_measurements():
    parts = []
    for f in FILES:
        df = pd.read_csv(find(f), sep="\t", usecols=["Query Strain ID", "Array Strain ID",
                                                    "Genetic interaction score (ε)", "P-value"],
                         dtype={"Query Strain ID": "category", "Array Strain ID": "category"}, encoding="utf-8")
        df.columns = ["q", "a", "eps", "p"]
        parts.append(df)
        print(f"read {f}: {len(df)}")
    strains = sorted(set().union(*[set(d.q.cat.categories) | set(d.a.cat.categories) for d in parts]))
    si = {s: i for i, s in enumerate(strains)}
    qs, as_, eps, p = [], [], [], []
    for d in parts:
        qs.append(d.q.cat.categories.map(si).values.astype(np.int64)[d.q.cat.codes.values])
        as_.append(d.a.cat.categories.map(si).values.astype(np.int64)[d.a.cat.codes.values])
        eps.append(d.eps.values.astype(np.float64))
        p.append(d.p.values.astype(np.float64))
    return strains, np.concatenate(qs), np.concatenate(as_), np.concatenate(eps), np.concatenate(p)


def collapse(strains, q, a, eps, p):
    ok = np.isfinite(eps)
    q, a, eps, p = q[ok], a[ok], eps[ok], p[ok]
    orf_of = np.array([s.rsplit("_", 1)[0] for s in strains])
    orfs = sorted(set(orf_of))
    oi = {o: i for i, o in enumerate(orfs)}
    s2o = np.array([oi[o] for o in orf_of])
    keep = s2o[q] != s2o[a]
    q, a, eps, p = q[keep], a[keep], eps[keep], p[keep]
    sig = ((eps < EPS_CUT) & (p < P_CUT)).astype(np.float64)
    NS = len(strains)
    lo, hi = np.minimum(q, a), np.maximum(q, a)
    df = pd.DataFrame({"k": lo * NS + hi, "eps": eps, "sig": sig})
    st = df.groupby("k", sort=False).agg(eps=("eps", "mean"), sig=("sig", "mean"), n=("eps", "size"))
    k = st.index.values
    s_lo, s_hi = k // NS, k % NS
    o1, o2 = s2o[s_lo], s2o[s_hi]
    NO = len(orfs)
    st2 = pd.DataFrame({"k": np.minimum(o1, o2) * NO + np.maximum(o1, o2), "eps": st.eps.values,
                        "neg": (st.sig.values >= 0.5).astype(np.float64)})
    op = st2.groupby("k", sort=False).agg(eps=("eps", "mean"), neg=("neg", "mean"), n_strain_pairs=("eps", "size"))
    kk = op.index.values
    out = pd.DataFrame({"i": kk // NO, "j": kk % NO, "eps": op.eps.values,
                        "neg": (op.neg.values >= 0.5), "n_strain_pairs": op.n_strain_pairs.values})
    info = {"n_measurements": int(len(eps)), "n_strain_pairs": int(len(st)), "n_orf_pairs": int(len(out)),
            "n_orfs": NO, "frac_orf_pairs_neg": float(out.neg.mean()),
            "frac_orf_pairs_multi_allele": float((out.n_strain_pairs > 1).mean())}
    return orfs, out, info


def fetch_ygob() -> None:
    """Download YGOB v7 Pillars.tab into its own directory if absent; verify sha256."""
    if not YGOB.exists():
        import subprocess
        YGOB.parent.mkdir(parents=True, exist_ok=True)
        tmp = YGOB.with_suffix(".part")
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(tmp), YGOB_URL], check=True)
        tmp.rename(YGOB)
        (YGOB.parent / "RETRIEVED.txt").write_text(f"{YGOB_URL}\nretrieved by {SCRIPT}\n")
    got = fs.sha256(YGOB)
    if got != YGOB_SHA256:
        raise RuntimeError(f"YGOB Pillars.tab sha256 {got} != {YGOB_SHA256}")


def expected_min(sorted_vals: np.ndarray, kmax: int) -> np.ndarray:
    n = len(sorted_vals)
    i = np.arange(1, n + 1)
    m = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        w = (1 - (i - 1) / n) ** k - (1 - i / n) ** k
        m[k] = float((w * sorted_vals).sum())
    return m


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")  # spurious Accelerate matmul flags
    checks = {}
    with fs.RuntimeLog("cer", eps_cut=EPS_CUT, p_cut=P_CUT, n_emb=N_EMB) as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        mapper = IDMapper()
        orth = load_orthologs(mapper)

        strains, q, a, eps, p = read_measurements()
        rt.step("read")
        orfs, op, info = collapse(strains, q, a, eps, p)
        del q, a, eps, p
        checks["collapse"] = info
        C = len(orfs)
        oi = {o: i for i, o in enumerate(orfs)}
        rt.step("collapse")

        # --- strain classes / essentiality / SMF
        st = pd.read_csv(STRAINS, sep="\t")
        cls = pd.concat([st[["Strain ID"]], pd.DataFrame({"Strain ID": strains})]).drop_duplicates()
        cls["orf"] = cls["Strain ID"].str.rsplit("_", n=1).str[0]
        cls["cls"] = cls["Strain ID"].str.rsplit("_", n=1).str[1].str.extract(r"^([a-z]+)", expand=False)
        byorf = cls.groupby("orf").cls.agg(lambda s: set(s.dropna()))
        essential = {o: bool(c & ESS_CLASSES) and not (c & DEL_CLASSES) for o, c in byorf.items()}
        checks["essentiality"] = {
            "n_orfs_with_strains": int(len(byorf)), "n_essential": int(sum(essential.values())),
            "n_orfs_with_both_deletion_and_ts_or_damp": int(sum(bool(c & ESS_CLASSES) and bool(c & DEL_CLASSES)
                                                               for c in byorf.values)),
            "class_counts": cls.cls.value_counts().to_dict()}
        dels = st[st["Strain ID"].str.rsplit("_", n=1).str[1].str.extract(r"^([a-z]+)", expand=False).isin(DEL_CLASSES)]
        smf_orf = dels.groupby(dels["Strain ID"].str.rsplit("_", n=1).str[0])["Single mutant fitness (30°)"].mean()
        ess_vec = np.array([essential.get(o, False) for o in orfs])

        # --- ORF x ORF matrices
        E = np.full((C, C), np.nan, dtype=np.float32)
        E[op.i.values, op.j.values] = op.eps.values
        E[op.j.values, op.i.values] = op.eps.values
        Mmask = np.isfinite(E).astype(np.float32)
        NEG = np.zeros((C, C), dtype=np.float32)
        NEG[op.i.values, op.j.values] = op.neg.values
        NEG[op.j.values, op.i.values] = op.neg.values
        E0 = np.nan_to_num(E, nan=0.0)
        rt.step("orf_matrices")

        # --- per-ORF negative rates (NxN vs E-involving)
        nn = (~ess_vec[:, None]) & (~ess_vec[None, :])
        tested_nn = (Mmask * nn).sum(1)
        neg_nn = (NEG * nn).sum(1)
        tested_e = (Mmask * ~nn).sum(1)
        neg_e = (NEG * ~nn).sum(1)

        # --- profiles: PCA embedding + correlation
        n_partners = Mmask.sum(1)
        prof = np.where(n_partners >= MIN_PROFILE_PARTNERS)[0]
        Xp = E0[prof].astype(np.float64)
        pca = PCA(n_components=N_EMB, random_state=SEED, svd_solver="randomized")
        emb_orf = pca.fit_transform(Xp)
        checks["profile_pca"] = {"n_orfs_profiled": int(len(prof)),
                                 "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_]}
        Vc = Xp - Xp.mean(1, keepdims=True)
        Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
        Rorf = (Vc @ Vc.T).astype(np.float32)
        np.fill_diagonal(Rorf, 0)
        del Xp, Vc
        rt.step("profiles")

        # --- ortholog incidence
        miss = sorted(set(orth.sc) - set(oi))
        log_unmatched("costanzo2016_ortholog_orfs_absent", miss, "curated ortholog ORF absent from Costanzo 2016 SGA data")
        O_all = incidence(orth, genes, sorted(set(orth.sc)))        # any curated ortholog
        has_orth = O_all.sum(1) > 0
        O = incidence(orth, genes, orfs)                              # orthologs with Costanzo data
        n_oc = O.sum(1)
        Op = O[:, prof]                                               # orthologs with a profile

        # --- gene-level
        gene = pd.DataFrame(index=genes)
        gene["has_ortholog"] = has_orth.astype(float)
        gene["n_orthologs"] = O_all.sum(1)
        gene["n_orthologs_costanzo"] = n_oc
        orf_has_strain = np.array([o in byorf.index for o in orfs], dtype=np.float32)
        n_with_strain = O @ orf_has_strain
        gene["any_ortholog_essential"] = np.where(n_with_strain > 0, ((O @ ess_vec.astype(np.float32)) > 0).astype(float), np.nan)
        smf_vec = np.array([smf_orf.get(o, np.nan) if not essential.get(o, False) else np.nan for o in orfs])
        has_smf = np.isfinite(smf_vec).astype(np.float32)
        n_smf = O @ has_smf
        gene["ortholog_smf"] = np.where(n_smf > 0, (O @ np.nan_to_num(smf_vec)) / np.maximum(n_smf, 1), np.nan)
        t_nn, t_e = O @ tested_nn, O @ tested_e
        gene["neg_rate_nxn"] = np.where(t_nn > 0, (O @ neg_nn) / np.maximum(t_nn, 1), np.nan)
        gene["neg_rate_e"] = np.where(t_e > 0, (O @ neg_e) / np.maximum(t_e, 1), np.nan)
        n_p = Op.sum(1)
        emb = (Op @ emb_orf) / np.maximum(n_p, 1)[:, None]
        emb[n_p == 0] = np.nan
        for k in range(N_EMB):
            gene[f"emb{k + 1}"] = emb[:, k]

        # ohnologs (YGOB pillars: col 12 = S. cerevisiae track A, col 22 = S. cerevisiae track B)
        fetch_ygob()
        pil = pd.read_csv(YGOB, sep="\t", header=None, dtype=str)
        assert pil.shape[1] == 33
        orf_re = r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$"   # nuclear ORFs only (drops Scer_CEN*, Scer_YGOB_* entries)
        both_tracks = pil[(pil[11] != "---") & (pil[21] != "---")]
        ohp = both_tracks[both_tracks[11].str.match(orf_re) & both_tracks[21].str.match(orf_re)]
        oh = ohp[[11, 21]].values
        checks["ygob"] = {"n_pillars_both_tracks": int(len(both_tracks)), "n_ohnolog_pairs": int(len(oh)),
                          "n_pairs_both_in_costanzo": int(sum(x in oi and y in oi for x, y in oh))}
        log_unmatched("ygob_ohnolog_orfs_absent_from_costanzo",
                      [x for pr in oh for x in pr if x not in oi], "YGOB ohnolog ORF absent from Costanzo data")
        allsc = sorted(set(orth.sc))
        ai = {o: i for i, o in enumerate(allsc)}
        H_all = np.zeros((len(allsc), len(allsc)), dtype=np.float32)   # ohnolog pairs among curated orthologs
        Hn = np.zeros((C, C), dtype=np.float32)                         # ohnolog pairs with negative eps
        for x, y in oh:
            if x in ai and y in ai:
                H_all[ai[x], ai[y]] = H_all[ai[y], ai[x]] = 1
            if x in oi and y in oi and NEG[oi[x], oi[y]] > 0:
                Hn[oi[x], oi[y]] = Hn[oi[y], oi[x]] = 1
        checks["ygob"]["n_ohnolog_pairs_negative"] = int(Hn.sum() / 2)
        within_all = ((O_all @ H_all) * O_all).sum(1) / 2
        within_neg = ((O @ Hn) * O).sum(1) / 2
        gene["has_ohnolog_orthologs"] = np.where(has_orth, (within_all > 0).astype(float), np.nan)
        gene["ohnolog_orthologs_neg"] = np.where(has_orth, (within_neg > 0).astype(float), np.nan)
        colsrc = {c: SOURCES for c in gene.columns}
        for c in ("has_ohnolog_orthologs", "ohnolog_orthologs_neg"):
            colsrc[c] = SOURCES + ["ygob"]
        for c in ("has_ortholog", "n_orthologs"):
            colsrc[c] = ["pombase_orthologs"]
        sfiles = ["costanzo2016:Data File S1 SGA_NxN/ExN/ExE/DAmP.txt (strain IDs, eps, P-value)",
                  "costanzo2016:strain_ids_and_single_mutant_fitness.xlsx (via data/processed/costanzo_strains.tsv: "
                  "strain class, 'Single mutant fitness (30°)')", ORTH_FILE,
                  "ygob:Pillars.tab (v7-Aug2012) columns 12 and 22 (S. cerevisiae ohnolog pairs)"]
        fs.save_gene_table("CER", gene[gene.index.isin(gi)], SOURCES + ["ygob"], SCRIPT, column_sources=colsrc,
                           source_files=sfiles)
        checks["coverage_genes"] = {c: int(gene[c].notna().sum()) for c in gene.columns}
        checks["n_genes_ortholog_with_costanzo"] = int((n_oc > 0).sum())
        rt.step("gene_level")

        # --- pair-level aggregates over ortholog pairs
        npairs = O @ Mmask @ O.T
        s_eps = O @ E0 @ O.T
        s_neg = O @ NEG @ O.T
        meas = npairs > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            eps_mean = np.where(meas, s_eps / npairs, np.nan).astype(np.float32)
            frac_neg = np.where(meas, s_neg / npairs, np.nan).astype(np.float32)
        # min over ortholog pairs via slot gathering (genes have <= Kmax Costanzo orthologs)
        lists = [np.where(O[i] > 0)[0] for i in range(G)]
        Kmax = max(len(l) for l in lists)
        eps_min = np.full((G, G), np.nan, dtype=np.float32)
        for s in range(Kmax):
            rs = np.array([i for i in range(G) if len(lists[i]) > s])
            xs = np.array([lists[i][s] for i in rs])
            for t in range(Kmax):
                cs = np.array([i for i in range(G) if len(lists[i]) > t])
                ys = np.array([lists[i][t] for i in cs])
                sub = E[np.ix_(xs, ys)]
                blk = eps_min[np.ix_(rs, cs)]
                eps_min[np.ix_(rs, cs)] = np.fmin(blk, sub)
        nint = np.rint(npairs).astype(np.int64)
        kmax_pairs = int(nint.max())
        vals = np.sort(op.eps.values)
        mk = expected_min(vals, kmax_pairs)
        corr = (mk - mk[1]).astype(np.float32)
        corr[0] = np.nan
        eps_kmin = (eps_min - corr[nint]).astype(np.float32)
        checks["kmin"] = {"kmax": kmax_pairs, "m_k_minus_m_1": {int(k): round(float(corr[k]), 4)
                                                                for k in range(1, min(kmax_pairs, 10) + 1)}}
        ohn = (O @ Hn @ O.T > 0).astype(np.float32)
        ohn[~has_orth, :] = np.nan
        ohn[:, ~has_orth] = np.nan
        shared = ((O_all @ O_all.T) > 0).astype(np.float32)
        shared[~has_orth, :] = np.nan
        shared[:, ~has_orth] = np.nan
        offd = 1 - np.eye(len(prof), dtype=np.float32)
        cnt = Op @ offd @ Op.T
        with np.errstate(invalid="ignore", divide="ignore"):
            pc = np.where(cnt > 0, (Op @ Rorf @ Op.T) / cnt, np.nan).astype(np.float32)
        pc = np.clip(pc, -1, 1)
        rt.step("pair_aggregates")

        pairs = {"eps_mean": eps_mean, "eps_min": eps_min, "eps_kmin": eps_kmin, "frac_neg": frac_neg,
                 "n_pairs": npairs.astype(np.float32), "measured": meas.astype(np.float32),
                 "ohnolog_sl": ohn, "shared_ortholog": shared}
        desc = {"eps_mean": "mean allele-collapsed Costanzo eps over measured ortholog ORF pairs",
                "eps_min": "min allele-collapsed eps over measured ortholog ORF pairs",
                "eps_kmin": "eps_min - (E[min of k draws] - E[1 draw]), k = n_pairs",
                "frac_neg": "fraction of measured ortholog ORF pairs that are negative (eps<-0.08, p<0.05)",
                "n_pairs": "# measured ortholog ORF pairs (x != y)", "measured": "n_pairs > 0",
                "ohnolog_sl": "ortholog pair includes a YGOB ohnolog pair with negative eps",
                "shared_ortholog": "a and b share a curated cerevisiae ortholog"}
        for name, M in pairs.items():
            M = fs.symmetrize_inplace(np.ascontiguousarray(M, dtype=np.float32))
            src = SOURCES + (["ygob"] if name == "ohnolog_sl" else [])
            if name == "shared_ortholog":
                src = ["pombase_orthologs"]
            fs.save_matrix("pair", f"CER__{name}", M, "CER", src, SCRIPT, description=desc[name], source_files=sfiles)
        pc = fs.symmetrize_inplace(pc)
        fs.save_matrix("sim", "cer_profile_corr", pc, "CER", SOURCES, SCRIPT, source_files=sfiles[:1] + [ORTH_FILE],
                       description="mean Pearson r of allele-collapsed Costanzo eps profiles over ortholog combos x != y")
        iu = np.triu_indices(G, 1)
        checks["pairs"] = {"n_measured_pairs": int(meas[iu].sum()),
                           "n_pairs_with_profile_corr": int(np.isfinite(pc[iu]).sum()),
                           "n_ohnolog_sl_pairs": int(np.nansum(ohn[iu])),
                           "n_shared_ortholog_pairs": int(np.nansum(shared[iu])),
                           "eps_mean_quantiles": {qq: float(np.nanquantile(eps_mean[iu][meas[iu]], qq))
                                                  for qq in (0.01, 0.05, 0.5, 0.95)}}
        # consistency with the core table (mean over all measurements)
        core = pd.read_parquet(ROOT / "data/processed/costanzo_pairs.parquet", columns=["orf_a", "orf_b", "eps_mean"])
        o_a, o_b = np.array(orfs)[op.i.values], np.array(orfs)[op.j.values]
        assert (o_a < o_b).all()  # orfs sorted and i < j by construction
        mine = pd.DataFrame({"orf_a": o_a, "orf_b": o_b, "eps2": op.eps.values})
        jj = core.merge(mine, on=["orf_a", "orf_b"], how="outer", indicator=True)
        both = jj[jj._merge == "both"]
        checks["vs_core_costanzo_pairs"] = {"n_core": int(len(core)), "n_mine": int(len(mine)),
                                            "n_both": int(len(both)),
                                            "pearson_eps": float(np.corrcoef(both.eps_mean, both.eps2)[0, 1]),
                                            "frac_identical_1e-6": float((abs(both.eps_mean - both.eps2) < 1e-6).mean())}
        rt.step("save")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1, default=str) + "\n")
    print(json.dumps(checks, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
