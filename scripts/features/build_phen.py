#!/usr/bin/env python
"""Tier-1 SMF/PHEN feature group (DESIGN.md v3 §4, group PHEN).

Sources (DATA.md dataset ids) and the ONLY parts read:
  rodriguez_lopez_2023_phenomics  elife-88229-supp1-v1.xlsx sheets
      'Fitness standard conditions' (gene_id, condition in {YES_32, EMM_32}, fitness_median),
      'Fitness all conditions'      (gene_id, condition, fitness_median),
      'Discretized all conditions'  (gene x 129 conditions, -1 sensitive / 0 / +1 resistant),
      'Key to conditions'           (not needed: the medium is the condition prefix YES_/EMM_; checked).
      NOT read: GO enrichments, GO slims, KEGG, K-medoids clusters, Data for clustering, Oxidative stress,
      Standard conditions, Hits per mutant/condition (recomputed from the discretized sheet).
  kim2010_deletion_library  41587_2010_BFnbt1628_MOESM5_ESM.xls (Supplementary Table 15) columns
      'Systematic name', 'Average relative fitness (Het_AV)', 'Elimination of strains (filtering)'.
      Strains the authors eliminated (non-empty filtering column) are set to NaN.

Features
  gene/PHEN.parquet
    smf_yes32, smf_emm32   haploid deletion-mutant fitness (colony size relative to WT; column
                           fitness_median) in the two standard conditions YES 32C / EMM 32C
    pc1..pc10              PCA of the condition-specific profile (below); explained variance in checks
    n_sensitive, n_resistant, n_pheno   # of the 129 discretized conditions with phenotype -1 / +1 / either
    frac_pheno             n_pheno / # discretized conditions
    het_diploid_fitness_kim2010   HETEROZYGOUS-DIPLOID relative fitness (Kim 2010 Het_AV; covers
                           essential genes; NOT a haploid SMF)
  sim/phen_corr.npy        Pearson r between condition-specific profiles
  Pair features min/max/product of SMF are simple functions of smf_* and are left to the pair assembler.

Condition-specific profile: for each of the 129 non-control conditions c of medium m(c) (prefix YES/EMM),
  x[g, c] = log2(fitness_median[g, c]) - log2(fitness_median[g, m(c)_32]),  fitness clipped to >= 0.01,
then z-scored per condition across genes, clipped to [-5, 5], missing -> 0 (= no condition effect).
Genes with < 50% of conditions observed (or no control) get NaN PCs / NaN similarity.

Missingness: inviable genes have no deletion mutant, so haploid PHEN features are NaN for them (DESIGN §1
hypomorph note: a `hypomorph` indicator, always 0 at inference, prevents this from acting as an
essentiality proxy in training).

Run: .venv/bin/python -I scripts/features/build_phen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src.idmap import IDMapper  # noqa: E402

SCRIPT = "scripts/features/build_phen.py"
RL = ROOT / "data/raw/rodriguez_lopez_2023_phenomics/elife-88229-supp1-v1.xlsx"
KIM = ROOT / "data/raw/kim2010_deletion_library/41587_2010_BFnbt1628_MOESM5_ESM.xls"
N_PC = 10
MIN_OBS = 0.5
SEED = 0
CHECKS = ROOT / "results/features/phen_checks.json"
RL_SRC = "rodriguez_lopez_2023_phenomics"
KIM_SRC = "kim2010_deletion_library"


def map_index(df: pd.DataFrame, col: str, mapper: IDMapper, source: str) -> pd.DataFrame:
    m = mapper.map(df[col], source=source)
    df = df.assign(gene=df[col].map(m)).dropna(subset=["gene"])
    return df


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")  # spurious Accelerate matmul flags
    checks = {}
    with fs.RuntimeLog("phen", n_pc=N_PC) as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        mapper = IDMapper()

        # --- standard-condition fitness
        std = pd.read_excel(RL, sheet_name="Fitness standard conditions")
        assert set(std.condition) == {"YES_32", "EMM_32"}, set(std.condition)
        std = map_index(std, "gene_id", mapper, "rodriguez_lopez_2023_standard")
        smf = std.pivot_table(index="gene", columns="condition", values="fitness_median", aggfunc="mean")
        rt.step("read_standard")

        # --- all conditions -> condition-specific profile
        allc = pd.read_excel(RL, sheet_name="Fitness all conditions", usecols=["gene_id", "condition",
                                                                              "fitness_median"])
        allc = map_index(allc, "gene_id", mapper, "rodriguez_lopez_2023_all_conditions")
        F = allc.pivot_table(index="gene", columns="condition", values="fitness_median", aggfunc="mean")
        controls = ["YES_32", "EMM_32"]
        conds = sorted(c for c in F.columns if c not in controls)
        assert len(conds) == 129, len(conds)
        media = {c: c.split("_")[0] for c in conds}
        assert set(media.values()) <= {"YES", "EMM"}
        L = np.log2(F.clip(lower=0.01))
        X = pd.DataFrame({c: L[c] - L[f"{media[c]}_32"] for c in conds})
        obs = X.notna().mean(1)
        keep = obs >= MIN_OBS
        Z = (X - X.mean()) / X.std(ddof=0)
        Z = Z.clip(-5, 5).fillna(0.0)[keep]
        checks["profile"] = {"n_genes_mapped": int(len(X)), "n_genes_kept": int(keep.sum()),
                             "n_conditions": len(conds),
                             "frac_missing_cells_kept_genes": float(X[keep].isna().mean().mean())}
        pca = PCA(n_components=N_PC, random_state=SEED, svd_solver="full")
        pcs = pd.DataFrame(pca.fit_transform(Z.values), index=Z.index,
                           columns=[f"pc{i + 1}" for i in range(N_PC)])
        checks["pca_explained_variance_ratio"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        rt.step("profile_pca")

        # --- discretized counts
        disc = pd.read_excel(RL, sheet_name="Discretized all conditions")
        disc = map_index(disc, "gene_id", mapper, "rodriguez_lopez_2023_discretized")
        D = disc.drop(columns=["gene_id"]).groupby("gene").mean()  # unique after mapping (checked below)
        assert disc.gene.is_unique, "duplicate genes after ID mapping in discretized sheet"
        vals = set(np.unique(D.values[~np.isnan(D.values)]))
        assert vals <= {-1, 0, 1}, vals
        cnt = pd.DataFrame({"n_sensitive": (D == -1).sum(1), "n_resistant": (D == 1).sum(1),
                            "n_pheno": (D != 0).sum(1) - D.isna().sum(1)})
        n_obs = D.notna().sum(1)
        cnt["frac_pheno"] = cnt.n_pheno / n_obs
        cnt.loc[n_obs == 0, :] = np.nan          # genes whose discretized row is entirely empty
        checks["discretized"] = {"n_genes": int(len(D)), "n_conditions": int(D.shape[1]),
                                 "n_pheno_median": float(cnt.n_pheno.median())}
        rt.step("discretized")

        # --- Kim 2010 heterozygous diploid fitness
        sh = xlrd.open_workbook(KIM).sheet_by_index(0)
        hdr = sh.row_values(3)
        ci, cf, ce = hdr.index("Systematic name"), hdr.index("Average relative fitness (Het_AV)"), \
            hdr.index("Elimination of strains (filtering)")
        rows = [(sh.cell_value(r, ci), sh.cell_value(r, cf), str(sh.cell_value(r, ce)).strip())
                for r in range(4, sh.nrows)]
        kim = pd.DataFrame(rows, columns=["sysid", "het", "elim"])
        kim["het"] = pd.to_numeric(kim.het, errors="coerce")
        checks["kim2010"] = {"rows": int(len(kim)), "eliminated_by_authors": int((kim.elim != "").sum())}
        kim.loc[kim.elim != "", "het"] = np.nan
        kim = map_index(kim, "sysid", mapper, "kim2010_s15")
        het = kim.groupby("gene").het.mean()
        rt.step("kim")

        # --- gene table
        gt = pd.DataFrame(index=sorted(set(smf.index) | set(pcs.index) | set(cnt.index) | set(het.dropna().index)))
        gt["smf_yes32"] = smf.get("YES_32")
        gt["smf_emm32"] = smf.get("EMM_32")
        gt = gt.join(pcs).join(cnt)
        gt["het_diploid_fitness_kim2010"] = het
        gt = gt[gt.index.isin(gi)]
        rl_files = [f"{RL_SRC}:elife-88229-supp1-v1.xlsx sheet 'Fitness standard conditions' col fitness_median",
                    f"{RL_SRC}:elife-88229-supp1-v1.xlsx sheet 'Fitness all conditions' col fitness_median",
                    f"{RL_SRC}:elife-88229-supp1-v1.xlsx sheet 'Discretized all conditions'"]
        kim_files = [f"{KIM_SRC}:41587_2010_BFnbt1628_MOESM5_ESM.xls (Supp. Table 15) col "
                     "'Average relative fitness (Het_AV)', eliminated strains removed"]
        colsrc = {c: [KIM_SRC] if c.endswith("kim2010") else [RL_SRC] for c in gt.columns}
        desc = {"smf_yes32": "haploid deletion fitness, YES 32C (fitness_median, rel. to WT)",
                "smf_emm32": "haploid deletion fitness, EMM 32C (fitness_median, rel. to WT)",
                "n_sensitive": "# of 129 conditions discretized sensitive (-1)",
                "n_resistant": "# of 129 conditions discretized resistant (+1)",
                "n_pheno": "# of 129 conditions with any phenotype", "frac_pheno": "n_pheno / # conditions",
                "het_diploid_fitness_kim2010": "HETEROZYGOUS DIPLOID relative fitness (Kim 2010 Het_AV)",
                **{f"pc{i + 1}": f"PC{i + 1} of the condition-specific profile" for i in range(N_PC)}}
        fs.save_gene_table("PHEN", gt, [RL_SRC, KIM_SRC], SCRIPT, column_sources=colsrc, descriptions=desc,
                           source_files=rl_files + kim_files)
        checks["coverage_genes"] = {c: int(gt[c].notna().sum()) for c in gt.columns}
        both = gt[["smf_yes32", "smf_emm32"]].dropna()
        checks["corr_smf_yes32_vs_emm32"] = float(both.corr(method="spearman").iloc[0, 1])
        kh = gt[["smf_yes32", "het_diploid_fitness_kim2010"]].dropna()
        checks["corr_smf_yes32_vs_kim_het"] = {"n": int(len(kh)),
                                               "spearman": float(kh.corr(method="spearman").iloc[0, 1])}
        rt.step("gene_table")

        # --- similarity: Pearson r of z-scored profiles
        Zg = Z[Z.index.isin(gi)]
        V = Zg.values - Zg.values.mean(1, keepdims=True)
        nrm = np.linalg.norm(V, axis=1)
        ok = nrm > 0
        V = V[ok] / nrm[ok, None]
        idx = np.array([gi[g] for g in Zg.index[ok]])
        S = np.full((G, G), np.nan, dtype=np.float32)
        S[np.ix_(idx, idx)] = np.clip(V @ V.T, -1, 1).astype(np.float32)
        S = fs.symmetrize_inplace(S)
        fs.save_matrix("sim", "phen_corr", S, "PHEN", [RL_SRC], SCRIPT, source_files=rl_files[1:2],
                       description="Pearson r of condition-specific phenomics profiles (129 conditions)")
        iu = np.triu_indices(len(idx), 1)
        vals = (V @ V.T)[iu]
        checks["phen_corr"] = {"n_genes": int(len(idx)), "mean": float(vals.mean()),
                               "quantiles": {q: float(np.quantile(vals, q)) for q in (0.01, 0.5, 0.99)}}
        rt.step("similarity")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
