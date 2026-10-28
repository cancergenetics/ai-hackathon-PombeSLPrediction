#!/usr/bin/env python
"""Tier-2 EXPR feature group (DESIGN.md v3.1 §4, group EXPR).  Label-free.

Sources (config/data_allowlist.yaml dataset ids) and the ONLY parts read:
  coxpresdb_spo             Spo-r.c3-0.expression.combat.txt.zip (RNA-seq, 556 runs, 32 studies) and
                            Spo-m.c4-0.expression.combat.txt.zip (microarray, 166 arrays, 9 studies): the
                            expression matrices only (rows = NCBI Entrez gene IDs, values = ComBat-corrected
                            log-scale expression).  The COXPRESdb coex tables are not used (not downloaded).
  pombase_quant_expression  htp_gene_expression_table.tsv columns reference, gene, term_name (RNA/protein
                            level), during_term_id (used ONLY as the condition key: GO:0072690 vegetative
                            growth, PBQ:0000003 quiescence, GO:0000080/84/85/87 = G1/S/G2/M phase),
                            average_copies_per_cell.  Marguerat 2012 (PMID:23101633), Carpy 2014 (PMID:24763107).
  ID mapping only (not a feature source): ncbi_gene_pombe entrez_to_pombase.tsv (entrez_id, pombase_id), then
  src/idmap.py.  Recorded in the manifest as `id_mapping`.

Normalisation (per platform)
  1. Each sample (column) is median-centred over all genes (removes residual library/scale offsets after
     ComBat; range of sample medians before centring is in results/features/expr_checks.json).
  2. Several Entrez IDs mapping to one PomBase gene: rows averaged (logged in checks).
  3. Co-expression = Pearson r across samples of the centred log values (== Pearson of per-gene z-scores).
  4. Rank normalisation: mutual-rank score 1 - log(MR)/log(n-1), MR = sqrt(rank_ab * rank_ba) of r within
     each gene's partner list (removes gene-specific baselines; 1 = mutual best partner).

Features
  gene/EXPR.parquet
    rnaseq_mean, array_mean      mean log expression (before sample centring) per platform
    rnaseq_sd, array_sd          sd of centred log expression across samples (variability)
    mrna_prolif_log10            Marguerat RNA copies/cell, proliferating (log10(x + 0.01))
    mrna_quiesc_log10            Marguerat RNA copies/cell, quiescent (log10(x + 0.01))
    mrna_quiesc_vs_prolif_log2   log2((q + 0.01) / (p + 0.01))
    protein_prolif_log10         Marguerat protein copies/cell, proliferating (log10(x + 1))
    protein_quiesc_log10         Marguerat protein copies/cell, quiescent (log10(x + 1))
    protein_per_mrna_log10       protein_prolif_log10 - mrna_prolif_log10 (translation-efficiency proxy)
    carpy_protein_log10          Carpy protein copies/cell, vegetative growth (log10(x + 1))
    carpy_cellcycle_sd_log2      sd over G1/S/G2/M of log2(Carpy copies + 1) (needs all 4 phases)
  sim/expr_rnaseq.npy, sim/expr_array.npy        Pearson r (NBR input)
  pair/EXPR__mr_rnaseq.npy, pair/EXPR__mr_array.npy   mutual-rank score in [0, 1]
Missingness: NaN when a gene is absent from a platform/study.

Run: .venv/bin/python -I scripts/features/build_expr.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import sim_utils as su  # noqa: E402
from src.idmap import IDMapper  # noqa: E402

SCRIPT = "scripts/features/build_expr.py"
CX = ROOT / "data/raw/coxpresdb_spo"
PLATFORMS = {"rnaseq": "Spo-r.c3-0", "array": "Spo-m.c4-0"}
ENTREZ = ROOT / "data/raw/ncbi_gene_pombe/entrez_to_pombase.tsv"
QUANT = ROOT / "data/raw/pombase_quant_expression/htp_gene_expression_table.tsv"
CX_SRC, Q_SRC = "coxpresdb_spo", "pombase_quant_expression"
ID_MAP = ["ncbi_gene_pombe:entrez_to_pombase.tsv (entrez_id, pombase_id)", "src/idmap.py"]
CHECKS = ROOT / "results/features/expr_checks.json"
PSEUDO_RNA, PSEUDO_PROT = 0.01, 1.0
COND = {"GO:0072690": "prolif", "PBQ:0000003": "quiesc", "GO:0000080": "G1", "GO:0000084": "S",
        "GO:0000085": "G2", "GO:0000087": "M"}


def read_coxpresdb(prefix: str) -> pd.DataFrame:
    z = zipfile.ZipFile(CX / f"{prefix}.expression.combat.txt.zip")
    members = [i for i in z.infolist() if i.filename == f"{prefix}.expression.combat.txt"]
    assert len(members) == 1, [i.filename for i in z.infolist()]
    with z.open(members[0]) as fh:
        X = pd.read_csv(fh, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    assert X.index.is_unique and np.isfinite(X.values).all()
    return X


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    checks: dict = {"platforms": {}, "sim_distribution": {}}
    with fs.RuntimeLog("expr") as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        mapper = IDMapper()
        ez = pd.read_csv(ENTREZ, sep="\t", dtype=str, usecols=["entrez_id", "pombase_id"]).dropna()
        assert ez.entrez_id.is_unique
        ez = ez.set_index("entrez_id").pombase_id

        gene_cols: dict[str, pd.Series] = {}
        for plat, prefix in PLATFORMS.items():
            X = read_coxpresdb(prefix)
            n_raw = len(X)
            pb = X.index.map(ez)                                   # Entrez -> PomBase (NaN if unknown)
            no_pb = int(pd.isna(pb).sum())
            X = X[pd.notna(pb)]
            pb = pb[pd.notna(pb)]
            m = mapper.map(pb, source=f"coxpresdb_{plat}")
            sid = pd.Series(pb, index=X.index).map(m)
            X = X[sid.notna().values]
            sid = sid.dropna()
            n_multi = int(sid.duplicated(keep=False).sum())
            level = X.mean(1).groupby(sid.values).mean()
            Xc = X - X.median(0)                                   # per-sample median centring
            Xc = Xc.groupby(sid.values).mean()                     # collapse many Entrez -> one gene
            Xc = Xc[Xc.index.isin(gi)]
            gene_cols[f"{plat}_mean"] = level
            gene_cols[f"{plat}_sd"] = Xc.std(1, ddof=1)
            S, ok = su.row_pearson(Xc.values)
            ids = Xc.index[ok]
            idx = np.array([gi[g] for g in ids])
            files = [f"{CX_SRC}:{prefix}.expression.combat.txt.zip member {prefix}.expression.combat.txt"]
            M = su.scatter(S, idx, G)
            fs.save_matrix("sim", f"expr_{plat}", M, "EXPR", [CX_SRC], SCRIPT, source_files=files,
                           id_mapping=ID_MAP,
                           description=f"Pearson r across {X.shape[1]} {plat} samples (COXPRESdb {prefix}, "
                                       "per-sample median-centred log expression)")
            checks["sim_distribution"][f"expr_{plat}"] = su.dist_summary(M)
            del M
            MR = su.mutual_rank_score(S)
            M = su.scatter(MR, idx, G)
            fs.save_matrix("pair", f"EXPR__mr_{plat}", M, "EXPR", [CX_SRC], SCRIPT, source_files=files,
                           id_mapping=ID_MAP,
                           description=f"mutual-rank co-expression score 1-log(MR)/log(n-1), {plat}")
            checks["sim_distribution"][f"EXPR__mr_{plat}"] = su.dist_summary(M)
            del M, MR, S
            checks["platforms"][plat] = {"prefix": prefix, "n_rows_raw": n_raw, "n_samples": int(X.shape[1]),
                                         "rows_without_pombase_id": no_pb,
                                         "rows_sharing_a_gene_averaged": n_multi,
                                         "n_genes_protein_coding": int(len(Xc)), "n_genes_sim": int(len(idx)),
                                         "sample_median_range_before_centring":
                                             [float(X.median(0).min()), float(X.median(0).max())]}
            rt.step(plat)

        # ---- PomBase quantitative expression (Marguerat 2012, Carpy 2014)
        q = pd.read_csv(QUANT, sep="\t", dtype=str,
                        usecols=["reference", "gene", "term_name", "during_term_id", "average_copies_per_cell"])
        q["v"] = pd.to_numeric(q.average_copies_per_cell, errors="coerce")
        assert q.v.notna().all() and (q.v >= 0).all()
        q["cond"] = q.during_term_id.map(COND)
        assert q.cond.notna().all(), set(q.during_term_id) - set(COND)
        q["study"] = q.reference.map({"PMID:23101633": "marguerat", "PMID:24763107": "carpy"})
        assert q.study.notna().all()
        q["mol"] = q.term_name.map({"RNA level": "rna", "protein level": "protein"})
        assert q.mol.notna().all()
        m = mapper.map(q.gene, source="pombase_quant_expression")
        q["sid"] = q.gene.map(m)
        q = q.dropna(subset=["sid"])
        key = q.study + "_" + q.mol + "_" + q.cond
        W = q.assign(key=key).pivot_table(index="sid", columns="key", values="v", aggfunc="mean")
        checks["quant"] = {"n_rows": int(len(q)), "columns": {c: int(W[c].notna().sum()) for c in W.columns},
                           "n_zero_values": {c: int((W[c] == 0).sum()) for c in W.columns}}
        lr = lambda c: np.log10(W[c] + PSEUDO_RNA)  # noqa: E731
        lp = lambda c: np.log10(W[c] + PSEUDO_PROT)  # noqa: E731
        gene_cols["mrna_prolif_log10"] = lr("marguerat_rna_prolif")
        gene_cols["mrna_quiesc_log10"] = lr("marguerat_rna_quiesc")
        gene_cols["mrna_quiesc_vs_prolif_log2"] = np.log2((W["marguerat_rna_quiesc"] + PSEUDO_RNA)
                                                          / (W["marguerat_rna_prolif"] + PSEUDO_RNA))
        gene_cols["protein_prolif_log10"] = lp("marguerat_protein_prolif")
        gene_cols["protein_quiesc_log10"] = lp("marguerat_protein_quiesc")
        gene_cols["protein_per_mrna_log10"] = gene_cols["protein_prolif_log10"] - gene_cols["mrna_prolif_log10"]
        gene_cols["carpy_protein_log10"] = lp("carpy_protein_prolif")
        cc = np.log2(W[[f"carpy_protein_{p}" for p in ("G1", "S", "G2", "M")]] + PSEUDO_PROT)
        gene_cols["carpy_cellcycle_sd_log2"] = cc.std(1, ddof=1).where(cc.notna().all(1))
        rt.step("quant")

        gt = pd.DataFrame(gene_cols)
        gt = gt[gt.index.isin(gi)]
        quant_cols = [c for c in gt.columns if not c.startswith(("rnaseq", "array"))]
        colsrc = {c: [Q_SRC] if c in quant_cols else [CX_SRC] for c in gt.columns}
        desc = {"rnaseq_mean": "mean log expression, COXPRESdb RNA-seq (Spo-r.c3-0)",
                "array_mean": "mean log expression, COXPRESdb microarray (Spo-m.c4-0)",
                "rnaseq_sd": "sd across samples of sample-centred log expression, RNA-seq",
                "array_sd": "sd across samples of sample-centred log expression, microarray",
                "mrna_prolif_log10": "log10(Marguerat RNA copies/cell + 0.01), proliferating",
                "mrna_quiesc_log10": "log10(Marguerat RNA copies/cell + 0.01), quiescent",
                "mrna_quiesc_vs_prolif_log2": "log2 quiescent/proliferating RNA (+0.01)",
                "protein_prolif_log10": "log10(Marguerat protein copies/cell + 1), proliferating",
                "protein_quiesc_log10": "log10(Marguerat protein copies/cell + 1), quiescent",
                "protein_per_mrna_log10": "protein_prolif_log10 - mrna_prolif_log10",
                "carpy_protein_log10": "log10(Carpy protein copies/cell + 1), vegetative growth",
                "carpy_cellcycle_sd_log2": "sd over G1/S/G2/M of log2(Carpy copies + 1)"}
        files = [f"{CX_SRC}:{p}.expression.combat.txt.zip" for p in PLATFORMS.values()] + \
                [f"{Q_SRC}:htp_gene_expression_table.tsv cols gene, term_name, average_copies_per_cell "
                 "(during_term_id as condition key only)"]
        fs.save_gene_table("EXPR", gt, [CX_SRC, Q_SRC], SCRIPT, column_sources=colsrc, descriptions=desc,
                           source_files=files)
        both = gt[["rnaseq_mean", "mrna_prolif_log10"]].dropna()
        checks["consistency"] = {
            "spearman_rnaseq_mean_vs_marguerat_mrna": float(both.corr(method="spearman").iloc[0, 1]),
            "spearman_array_mean_vs_marguerat_mrna":
                float(gt[["array_mean", "mrna_prolif_log10"]].dropna().corr(method="spearman").iloc[0, 1]),
            "spearman_marguerat_vs_carpy_protein":
                float(gt[["protein_prolif_log10", "carpy_protein_log10"]].dropna().corr(method="spearman").iloc[0, 1])}
        checks["coverage_genes"] = {c: int(gt[c].notna().sum()) for c in gt.columns}
        # cross-platform agreement of co-expression on genes in both
        A = fs.load_matrix("sim", "expr_rnaseq")
        B = fs.load_matrix("sim", "expr_array")
        rng = np.random.default_rng(0)
        r_i, r_j = rng.integers(0, G, 400_000), rng.integers(0, G, 400_000)
        a, b = np.asarray(A[r_i, r_j]), np.asarray(B[r_i, r_j])
        f = np.isfinite(a) & np.isfinite(b)
        checks["consistency"]["pearson_rnaseq_vs_array_coexp_random_pairs"] = {
            "n": int(f.sum()), "r": float(np.corrcoef(a[f], b[f])[0, 1])}
        rt.step("gene_table")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps({k: v for k, v in checks.items() if k != "sim_distribution"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
