#!/usr/bin/env python
"""Tier-2 KOPROT feature group (DESIGN.md v3.1 §4, group KOPROT).  Label-free.

Source (config/data_allowlist.yaml dataset id) and the ONLY parts read:
  ozturk2022_ko_proteome  41467_2022_33814_MOESM3_ESM.zip, read with zipfile (never extracted), members
      20220918_SourceData/20220918_SourceData/Data_Figure_1b.txt  (proteinGroup, Strain, log2(intensity),
          strainType; 11,013,018 rows).  strainType must be exactly {'3,308 knockout strains', '461 wt'}
          (asserted): only KO and WT strains exist in this member.
      20220918_SourceData/20220918_SourceData/Data_Figure_2a.txt  (Strain, Regulated_pFDR_05: the authors'
          per-KO count of regulated proteins; used as one gene feature and to validate our threshold).
  Gene IDs: KO gene = Strain suffix after '-' (e.g. Sp01A02-SPAC1002.20); WT strains end in '-BIO'.
  Decoy protein groups (REV__) are dropped; multi-gene protein groups ('a;b') are kept as profile columns
  but not used for the gene-resolved directed features.

Normalisation of log2 intensities (0 = not quantified -> missing)
  1. Strain loading: subtract each strain's median over "core" proteins (quantified in >= 95% of strains).
  2. Plate reference: for each of the 41 plates (strain prefix SpNN) and protein, subtract the median of the
     KO strains on that plate (>= 10 observations; otherwise missing).  The KO population, not the WT
     controls, is the reference: the 461 WT control strains differ systematically from the deletion library
     (with the same scale they show a median of ~180 'changed' proteins vs ~9 for KO strains; see checks),
     so a WT reference would add one shared offset to every KO profile and inflate all KO-KO correlations.
     Most deletions leave most proteins unchanged, so the plate KO median estimates the unperturbed level.
  3. Robust per-protein scale: sd_p = 1.4826 * MAD over KO strains of the residuals (>= 30 obs), floored at
     the 5th percentile of sd_p.   z = residual / sd_p ;  log2FC = residual.
  The KO'd gene's own protein in its own strain is set to missing (trivially absent).
"Significantly changed" (documented threshold): |z| >= 3 AND |log2FC| >= log2(1.5).  This gives a median
  of ~9 changed proteins per KO and Spearman ~0.89 with the authors' Regulated_pFDR_05 counts (checks).

Features
  gene/KOPROT.parquet  (index = KO gene for ko_*, = protein gene for protein_*)
    ko_n_changed           # proteins significantly changed in the gene's KO strain
    ko_frac_changed        ko_n_changed / # proteins quantified in that strain
    ko_rms_z               root-mean-square z over quantified proteins (perturbation magnitude)
    ko_n_regulated_paper   authors' Regulated_pFDR_05 (Data_Figure_2a)
    protein_frac_kos_changed  fraction of KO strains in which this gene's protein is significantly changed
    ko_pc1..ko_pc5         PC scores of the KO-response profile matrix (global response axes, e.g. growth)
  sim/koprot_corr.npy   Pearson r between KO-response profiles: z clipped to [-10, 10], missing -> 0, over the
                        protein groups quantified in >= 50% of KO strains, column-centred, with the top 10
                        principal components removed.  The PCs carry shared technical/global axes (same
                        plate-row strains correlate at r = 0.07 before removal); n_pc = 10 was chosen among
                        {0,1,3,5,10,20,40} by a LABEL-FREE control, the BioGRID physical-edge rate among the
                        10,000 most correlated KO pairs (0.0009 -> 0.0033 vs base rate 0.0005; AUROC 0.51 -> 0.53).
  pair/KOPROT__a_changes_b.npy   max(|z(KO a -> protein b)|, |z(KO b -> protein a)|); NaN if neither direction
                        is measurable (a not deleted or b not quantified, and vice versa)
  pair/KOPROT__min_z.npy         min of the two signed z (most negative: KO destabilises the partner)
  pair/KOPROT__changed.npy       1 if either direction passes the threshold, 0 if measurable but not, else NaN
  directed/KOPROT__z_ko_row_protein_col.npy  ASYMMETRIC G x G float32, [i, j] = z of protein j in KO i
                        (interface extension: not a pair/ or sim/ file; never fed to models directly)

Run: .venv/bin/python -I scripts/features/build_koprot.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import sim_utils as su  # noqa: E402
from src.idmap import IDMapper  # noqa: E402

SCRIPT = "scripts/features/build_koprot.py"
ZIP = ROOT / "data/raw/ozturk2022_ko_proteome/41467_2022_33814_MOESM3_ESM.zip"
PFX = "20220918_SourceData/20220918_SourceData/"
M1B, M2A = PFX + "Data_Figure_1b.txt", PFX + "Data_Figure_2a.txt"
SRC = "ozturk2022_ko_proteome"
STYPES = {"3,308 knockout strains": "KO", "461 wt": "WT"}
CORE_OBS, PLATE_MIN, SD_MIN_OBS, PROFILE_OBS = 0.95, 10, 30, 0.5
Z_THR, FC_THR = 3.0, float(np.log2(1.5))
Z_CLIP = 10.0
N_PC_REMOVE, N_PC_FEAT = 10, 5
CHECKS = ROOT / "results/features/koprot_checks.json"


def read_member(name: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(ZIP) as z:
        assert name in z.namelist(), name
        with z.open(name) as fh:
            return pd.read_csv(fh, sep="\t", **kw)


def sig_mask(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (np.abs(Z) >= Z_THR) & (np.abs(R) >= FC_THR)


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    checks: dict = {"thresholds": {"abs_z": Z_THR, "abs_log2fc": FC_THR}}
    with fs.RuntimeLog("koprot") as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        mapper = IDMapper()

        d = read_member(M1B, dtype={"proteinGroup": str, "Strain": str, "strainType": str})
        d.columns = ["protein", "strain", "log2i", "stype"]
        assert set(d.stype.unique()) == set(STYPES), set(d.stype.unique())
        d["stype"] = d.stype.map(STYPES)
        checks["input"] = {"rows": int(len(d)), "strains": d.groupby("stype").strain.nunique().to_dict(),
                           "protein_groups": int(d.protein.nunique()),
                           "decoy_groups": int(d.protein.str.contains("REV__").groupby(d.protein).first().sum())}
        d = d[~d.protein.str.contains("REV__")]
        rt.step("read")

        L = d.pivot(index="strain", columns="protein", values="log2i")
        stype = d.drop_duplicates("strain").set_index("strain").stype.reindex(L.index)
        del d
        L = L.where(L > 0)                                        # 0 = not quantified
        is_ko = (stype == "KO").values
        assert L.index.str.match(r"^Sp\d\d(WT_|[A-H]\d\d-)").all()
        plate = L.index.str[:4].values                            # SpNN = 96-well plate
        assert len(set(plate)) == 41, len(set(plate))
        obs = L.notna().mean()
        core = obs >= CORE_OBS
        L = L.sub(L.loc[:, core].median(1), axis=0)              # 1. loading
        R = L.copy()
        for p in np.unique(plate):                                # 2. plate KO-median reference
            m = plate == p
            ko = L[m & is_ko]
            ref = ko.median().where(ko.notna().sum() >= PLATE_MIN)
            R.loc[m] = L[m] - ref
        sd = 1.4826 * R[is_ko].abs().median()                      # 3. robust scale over KO strains
        sd = sd.where(R[is_ko].notna().sum() >= SD_MIN_OBS)
        sd_floor = float(sd.quantile(0.05))
        sd = sd.clip(lower=sd_floor)
        Z = R / sd
        checks["normalisation"] = {"n_core_proteins": int(core.sum()), "sd_floor": sd_floor,
                                   "sd_median": float(sd.median()), "n_proteins_with_scale": int(sd.notna().sum())}
        rt.step("normalise")

        # ---- IDs: KO strains -> gene, protein groups -> gene (single-gene groups only)
        ko_ids = pd.Series(L.index[is_ko].str.split("-", n=1).str[1], index=L.index[is_ko])
        km = mapper.map(ko_ids, source="ozturk2022_ko_strains")
        ko_gene = ko_ids.map(km)
        # two strains for one current gene (merged/obsolete IDs): keep the strain named by the current ID
        dup = ko_gene.notna() & ko_gene.duplicated(keep=False) & (ko_ids != ko_gene)
        checks["ko_strains_dropped_duplicate_gene"] = {s: f"{ko_ids[s]} -> {ko_gene[s]}" for s in ko_gene[dup].index}
        ko_gene[dup] = np.nan
        single = [p for p in L.columns if ";" not in p]
        pm = mapper.map(single, source="ozturk2022_protein_groups")
        prot_gene = pd.Series({p: pm[p] for p in single}).dropna()
        assert prot_gene.is_unique, "two protein groups map to one gene"
        # KO'd protein in its own strain -> missing
        n_self = 0
        for s, g in ko_gene.dropna().items():
            hit = prot_gene.index[prot_gene == g]
            if len(hit):
                n_self += int(Z.loc[s, hit[0]] == Z.loc[s, hit[0]])
                Z.loc[s, hit[0]] = np.nan
                R.loc[s, hit[0]] = np.nan
        checks["ids"] = {"ko_strains": int(is_ko.sum()), "ko_strains_mapped": int(ko_gene.notna().sum()),
                         "ko_genes_unique": bool(ko_gene.dropna().is_unique),
                         "single_gene_protein_groups": len(single), "protein_groups_mapped": int(len(prot_gene)),
                         "self_protein_values_masked": n_self}
        assert ko_gene.dropna().is_unique

        # ---- KO-response profiles: technical-axis removal, PCs, correlation
        prof_cols = Z.columns[Z[is_ko].notna().mean() >= PROFILE_OBS]
        P = Z.loc[is_ko, prof_cols]
        P = P[ko_gene.reindex(P.index).notna().values]
        pg = ko_gene.reindex(P.index).values
        keep = np.array([g in gi for g in pg])
        P, pg = P[keep], pg[keep]
        X = P.clip(-Z_CLIP, Z_CLIP).fillna(0.0).values
        X = X - X.mean(0)
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = pd.DataFrame(U[:, :N_PC_FEAT] * sv[:N_PC_FEAT], index=pg,
                           columns=[f"ko_pc{i + 1}" for i in range(N_PC_FEAT)])
        Xr = X - (U[:, :N_PC_REMOVE] * sv[:N_PC_REMOVE]) @ Vt[:N_PC_REMOVE]
        evr = sv ** 2 / (sv ** 2).sum()
        Sim, ok = su.row_pearson(Xr)
        idx = np.array([gi[g] for g in pg[ok]])
        checks["profile"] = {"n_protein_groups": int(len(prof_cols)), "n_ko_genes": int(len(idx)),
                             "frac_missing_cells": float(P.isna().mean().mean()),
                             "n_pc_removed": N_PC_REMOVE,
                             "explained_variance_ratio_top_pcs": [round(float(v), 4) for v in evr[:N_PC_REMOVE]]}
        rt.step("profile")

        # ---- significance and gene-level features
        Zv, Rv = Z.values, R.values
        S = sig_mask(Zv, Rv)
        n_meas = np.isfinite(Zv).sum(1)
        n_sig = S.sum(1)
        rms = np.sqrt(np.nanmean(np.clip(Zv, -Z_CLIP, Z_CLIP) ** 2, axis=1))
        per = pd.DataFrame({"n": n_sig, "frac": n_sig / n_meas, "rms": rms, "is_ko": is_ko}, index=Z.index)
        paper = read_member(M2A, usecols=["Strain", "Regulated_pFDR_05"]).set_index("Strain").Regulated_pFDR_05
        kop = per[per.is_ko]
        checks["significance"] = {
            "ko_n_changed_median": float(kop.n.median()), "ko_n_changed_mean": float(kop.n.mean()),
            "wt_n_changed_median": float(per[~per.is_ko].n.median()),
            "spearman_vs_paper_Regulated_pFDR_05": float(spearmanr(kop.n, paper.reindex(kop.index)).correlation),
            "paper_Regulated_pFDR_05_median": float(paper.median()),
            "note": "WT controls differ systematically from the KO library; they are not the reference"}
        gene_ko = kop.assign(gene=ko_gene.reindex(kop.index)).dropna(subset=["gene"]).set_index("gene")
        frac_prot = pd.Series(S[is_ko].sum(0) / np.isfinite(Zv[is_ko]).sum(0), index=Z.columns)
        gt = pd.DataFrame({"ko_n_changed": gene_ko.n, "ko_frac_changed": gene_ko.frac, "ko_rms_z": gene_ko.rms})
        pp = paper.reindex(kop.index)
        pp.index = ko_gene.reindex(kop.index).values
        gt["ko_n_regulated_paper"] = pp[pp.index.notna()]
        pf = frac_prot.reindex(prot_gene.index)
        pf.index = prot_gene.values
        gt = gt.join(pf.rename("protein_frac_kos_changed"), how="outer").join(pcs, how="outer")
        gt = gt[gt.index.isin(gi)]
        files = [f"{SRC}:{M1B} (proteinGroup, Strain, log2(intensity), strainType in KO/WT)",
                 f"{SRC}:{M2A} (Strain, Regulated_pFDR_05)"]
        desc = {"ko_n_changed": f"# proteins with |z|>={Z_THR:g} and |log2FC|>=log2(1.5) in the gene's KO strain",
                "ko_frac_changed": "ko_n_changed / # proteins quantified in the KO strain",
                "ko_rms_z": "RMS z over quantified proteins in the KO strain",
                "ko_n_regulated_paper": "Ozturk 2022 Regulated_pFDR_05 count (Data_Figure_2a)",
                "protein_frac_kos_changed": "fraction of KO strains in which this protein is significantly changed",
                **{f"ko_pc{i + 1}": f"PC{i + 1} score of the KO-response profile matrix" for i in range(N_PC_FEAT)}}
        fs.save_gene_table("KOPROT", gt, [SRC], SCRIPT, descriptions=desc, source_files=files)
        checks["coverage_genes"] = {c: int(gt[c].notna().sum()) for c in gt.columns}
        rt.step("gene_table")

        M = su.scatter(Sim, idx, G)
        fs.save_matrix("sim", "koprot_corr", M, "KOPROT", [SRC], SCRIPT, source_files=files[:1],
                       description=f"Pearson r of KO-response z-profiles over {len(prof_cols)} protein groups, "
                                   f"top {N_PC_REMOVE} PCs removed")
        checks["sim_distribution"] = {"koprot_corr": su.dist_summary(M)}
        del M, Sim

        # ---- directed and symmetric "KO of a changes protein b"
        D = np.full((G, G), np.nan, dtype=np.float32)
        Dsig = np.zeros((G, G), dtype=bool)
        rows_s = [s for s in kop.index if pd.notna(ko_gene.get(s)) and ko_gene[s] in gi]
        ri = np.array([gi[ko_gene[s]] for s in rows_s])
        cols_p = [p for p in prot_gene.index if prot_gene[p] in gi]
        ci = np.array([gi[prot_gene[p]] for p in cols_p])
        D[np.ix_(ri, ci)] = Z.loc[rows_s, cols_p].values.astype(np.float32)
        Dsig[np.ix_(ri, ci)] = sig_mask(Z.loc[rows_s, cols_p].values, R.loc[rows_s, cols_p].values)
        np.fill_diagonal(D, np.nan)
        ddir = fs.feat_dir() / "directed"
        ddir.mkdir(parents=True, exist_ok=True)
        dpath = ddir / "KOPROT__z_ko_row_protein_col.npy"
        np.save(dpath, D)
        fs.update_manifest("directed/KOPROT__z_ko_row_protein_col.npy", fs._entry(
            "KOPROT", [SRC], SCRIPT, dpath, kind="directed", source_files=files[:1],
            description="ASYMMETRIC: [i, j] = robust z of protein j in the KO strain of gene i (NaN = not measured)",
            n_pairs_nonmissing=int(np.isfinite(D).sum())))
        A = np.abs(D)
        with np.errstate(invalid="ignore"):
            amax = np.fmax(A, A.T)
            zmin = np.fmin(D, D.T)
        meas = np.isfinite(D) | np.isfinite(D.T)
        chg = np.where(meas, (Dsig | Dsig.T).astype(np.float32), np.nan).astype(np.float32)
        del A
        for name, M, desc_ in (("KOPROT__a_changes_b", amax, "max(|z(KO a->protein b)|, |z(KO b->protein a)|)"),
                               ("KOPROT__min_z", zmin, "min of the two signed directed z (most negative)"),
                               ("KOPROT__changed", chg, "1 if either direction is significantly changed "
                                                        f"(|z|>={Z_THR:g}, |log2FC|>=log2 1.5), 0 if measured")):
            fs.save_matrix("pair", name, M, "KOPROT", [SRC], SCRIPT, source_files=files[:1], description=desc_)
        iu = np.triu_indices(G, 1)
        checks["directed"] = {"n_ko_rows": int(len(ri)), "n_protein_cols": int(len(ci)),
                              "n_directed_measured": int(np.isfinite(D).sum()),
                              "n_directed_significant": int(Dsig.sum()),
                              "n_pairs_measured_either_direction": int(meas[iu].sum()),
                              "n_pairs_changed": int(np.nansum(chg[iu]))}
        rt.step("directed")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
