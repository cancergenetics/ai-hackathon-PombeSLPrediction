#!/usr/bin/env python
"""Coverage + sanity report for the tier-1 label-free features -> docs/features_tier1.md
(also results/features/tier1_coverage.tsv).

Reads only the feature store, the build checks (results/features/*_checks.json), results/idmap/summary.tsv,
results/runtime/features_*.json, and -- for coverage denominators only -- the universe gene list
(data/processed/universe_genes.tsv: gene, in_universe) and the Ryan gene list
(data/processed/ryan_genes.tsv: pombase_id).  No labels are read.
Run: .venv/bin/python -I scripts/features/report_tier1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402

OUT = ROOT / "docs/features_tier1.md"
RES = ROOT / "results/features"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and abs(v) >= 1):
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in r.values) + " |")
    return "\n".join(lines)


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    man = json.loads((fs.feat_dir() / "manifest.json").read_text())
    genes = fs.genes()
    G = len(genes)
    gi = {g: i for i, g in enumerate(genes)}
    uni = pd.read_csv(ROOT / "data/processed/universe_genes.tsv", sep="\t", usecols=["gene", "in_universe"])
    U = np.array(sorted(gi[g] for g in uni.loc[uni.in_universe.astype(bool), "gene"] if g in gi))
    ry = pd.read_csv(ROOT / "data/processed/ryan_genes.tsv", sep="\t", usecols=["pombase_id"]).pombase_id.dropna()
    R = np.array(sorted({gi[g] for g in ry if g in gi}))
    nU, nR = len(U), len(R)
    pairs_all, pairs_U = G * (G - 1) // 2, nU * (nU - 1) // 2

    # ---- gene-level coverage
    grows = []
    for k in sorted(k for k in man if k.startswith("gene/")):
        df = pd.read_parquet(fs.feat_dir() / k).reindex(genes)
        for c in df.columns:
            v = df[c].values
            grows.append({"feature": c, "file": k, "sources": ",".join(man[k]["columns"][c].get("sources", man[k]["sources"])),
                          "genes_all": int(np.isfinite(v).sum()), "genes_universe": int(np.isfinite(v[U]).sum()),
                          "genes_ryan": int(np.isfinite(v[R]).sum()),
                          "median": float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan})
    gcov = pd.DataFrame(grows)

    # ---- pair / sim coverage
    prows = []
    iuU = np.triu_indices(nU, 1)
    for k in sorted(k for k in man if k.startswith(("pair/", "sim/"))):
        M = np.load(fs.feat_dir() / k, mmap_mode="r")
        sub = np.asarray(M[np.ix_(U, U)])[iuU]
        fin = np.isfinite(sub)
        binary = fin.any() and np.isin(sub[fin][:100000], [0, 1]).all()
        prows.append({"feature": Path(k).stem, "kind": k.split("/")[0], "sources": ",".join(man[k]["sources"]),
                      "pairs_all": man[k].get("n_pairs_nonmissing"), "pairs_universe": int(fin.sum()),
                      "frac_universe": float(fin.sum() / pairs_U),
                      "genes_universe_any": int(np.isfinite(np.asarray(M[np.ix_(U, U)])).any(1).sum()),
                      "n_ones_universe": int((sub[fin] == 1).sum()) if binary else "",
                      "median_universe": float(np.median(sub[fin])) if fin.any() else np.nan})
    pcov = pd.DataFrame(prows)
    RES.mkdir(parents=True, exist_ok=True)
    pd.concat([gcov.assign(kind="gene"), pcov]).to_csv(RES / "tier1_coverage.tsv", sep="\t", index=False)

    # ---- cross-source consistency (label-free): do PPI edges have more similar profiles?
    E = np.load(fs.feat_dir() / "pair/PPI__edge.npy", mmap_mode="r")
    Esub = np.asarray(E[np.ix_(U, U)])[iuU]
    cons = []
    for name, kind in (("cer_profile_corr", "sim"), ("phen_corr", "sim"), ("CER__frac_neg", "pair"),
                       ("CER__eps_mean", "pair"), ("ppi_cer_rwr", "sim")):
        M = np.asarray(np.load(fs.feat_dir() / f"{kind}/{name}.npy", mmap_mode="r")[np.ix_(U, U)])[iuU]
        f = np.isfinite(M)
        e1, e0 = f & (Esub == 1), f & (Esub == 0)
        cons.append({"feature": name, "n_edge_pairs": int(e1.sum()), "mean_on_pombe_PPI_edges": float(M[e1].mean()),
                     "mean_on_non_edges": float(M[e0].mean())})
    cons = pd.DataFrame(cons)

    ppi = json.loads((RES / "ppi_checks.json").read_text())
    phen = json.loads((RES / "phen_checks.json").read_text())
    cer = json.loads((RES / "cer_checks.json").read_text())
    idm = pd.read_csv(ROOT / "results/idmap/summary.tsv", sep="\t")
    rt = []
    for n in ("ppi", "phen", "cer"):
        p = ROOT / f"results/runtime/features_{n}.json"
        if p.exists():
            j = json.loads(p.read_text())
            rt.append({"script": f"build_{n}.py", "wall_s": j["wall_s"], "peak_rss_mb": j["peak_rss_mb"]})
    filt = pd.DataFrame([{k: v for k, v in r.items() if k in ("network", "step", "rows", "edges")}
                         for r in ppi["filters"]])
    marg = [r for r in ppi["filters"] if r["step"].startswith("blocklist marginal")]

    # ---- sim matrix checks (full matrices)
    simrows = []
    for k in sorted(k for k in man if k.startswith("sim/")):
        M = np.load(fs.feat_dir() / k)
        v = M[np.isfinite(M)]
        simrows.append({"sim": Path(k).stem, "symmetric": bool(np.array_equal(M, M.T, equal_nan=True)),
                        "diag_all_nan": bool(np.isnan(np.diagonal(M)).all()), "min": float(v.min()),
                        "max": float(v.max()), "genes_with_data": int(np.isfinite(M).any(1).sum())})
        del M
    simdf = pd.DataFrame(simrows)

    pc = ppi["positive_control_cer_vs_pombe_edges"]
    d = ppi["network_default"]
    dh = ppi["network_htp"]
    txt = f"""# Tier-1 label-free features (PPI, SMF/PHEN, CER): coverage and sanity checks

Generated by `scripts/features/report_tier1.py` (do not edit by hand). Feature store layout:
docs/INTERFACES.md. No Ryan labels were read by any tier-1 builder (enforced by
`tests/features/test_tier1_features.py::test_feature_scripts_are_label_free`).

Rebuild (≈2 min total):
```bash
.venv/bin/python -I scripts/features/build_genes.py   # genes.txt ({G} protein-coding genes)
.venv/bin/python -I scripts/features/build_ppi.py     # PPI (+ BioGRID PMID blocklist re-fetch on first run)
.venv/bin/python -I scripts/features/build_phen.py    # SMF/PHEN
.venv/bin/python -I scripts/features/build_cer.py     # CER
.venv/bin/python -I scripts/features/report_tier1.py  # this file
.venv/bin/python -m pytest tests/features -q
```
Denominators: all = {G} protein-coding genes ({pairs_all:,} pairs); universe = {nU} non-essential
universe genes from `universe_genes.tsv` ({pairs_U:,} pairs); Ryan = {nR} Ryan genes (gene list only).

## Runtime
{md_table(pd.DataFrame(rt))}

## ID mapping (src/idmap.py; per-ID logs in results/idmap/<source>.tsv)
{md_table(idm)}

## PPI: BioGRID filters (rows / unique undirected edges after each step)
{md_table(filt)}

**PMID blocklist** ({ppi['n_blocklist_publications']:,} publications with ≥1 S. pombe genetic row in BioGRID
{"5.0.262"}; list + counts in `data/processed/features/biogrid_genetic_pmids.txt`, sha256 in manifest).
On the pombe network (after the system filter) it removes **{marg[0]['rows']:,} rows and {marg[0]['edges']:,}
edges** from {marg[0]['n_publications_hit']} publications ({marg[0]['htp_rows_removed']:,} of those rows are
HTP-flagged); top publications: {', '.join(f"{k} ({v})" for k, v in list(marg[0]['top_publications'].items())[:6])}.
On cerevisiae it removes {marg[1]['rows']:,} rows / {marg[1]['edges']:,} edges ({marg[1]['n_publications_hit']} publications).
Vo et al. 2016 (PMID 26771498) is not in the blocklist. Gene coverage cost (development check): universe
genes with ≥1 edge fall from 2,466 (system filter only) to 1,954 with the blocklist (HTP-only: 1,975 → 1,453).

Networks: default {d['n_edges']:,} edges on {d['n_genes_with_edge']:,} genes (degree median
{d['degree_quantiles']['0.5']:.0f}, 90th pct {d['degree_quantiles']['0.9']:.0f}, 99th pct
{d['degree_quantiles']['0.99']:.0f}, max {d['degree_max']:.0f}; top hubs {', '.join(d['top_hubs'][:5])});
HTP-only {dh['n_edges']:,} edges on {dh['n_genes_with_edge']:,} genes (median degree
{dh['degree_quantiles']['0.5']:.0f}, max {dh['degree_max']:.0f}). Capped shortest-path counts (default, all pairs
of connected genes): {d['sp_counts']}.

**Positive control** (genes with a pombe edge and a cerevisiae ortholog in the cerevisiae network,
n = {pc['n_genes']:,}): P(pombe edge | cerevisiae-ortholog edge) = {pc['P(pombe edge | cer edge)']:.4f} vs
{pc['P(pombe edge | no cer edge)']:.5f} without; odds ratio **{pc['odds_ratio']:.1f}**
({pc['p1c1']:,} shared edges).

## SMF/PHEN
- Haploid SMF = Rodríguez-López 2023 sheet *Fitness standard conditions*, column `fitness_median`
  (colony size relative to WT), conditions YES_32 (`smf_yes32`) and EMM_32 (`smf_emm32`); Spearman
  between them {phen['corr_smf_yes32_vs_emm32']:.2f}.
- Condition profile: 129 non-control conditions of *Fitness all conditions*,
  log2(fitness_median / fitness_median in the same medium's 32 °C control), z-scored per condition;
  {phen['profile']['n_genes_kept']:,} genes kept (≥50% conditions observed). PCA (10 comps) explained variance:
  {phen['pca_explained_variance_ratio']}.
- `n_sensitive/n_resistant/n_pheno` recomputed from *Discretized all conditions*; identical to the authors'
  *Hits per mutant* totals (checked during development).
- `het_diploid_fitness_kim2010`: Kim 2010 Supp. Table 15 `Het_AV` — **heterozygous diploid** fitness
  (dosage sensitivity, covers essential genes), {phen['kim2010']['eliminated_by_authors']} strains eliminated by
  the authors set to NaN. Spearman with haploid SMF (YES): {phen['corr_smf_yes32_vs_kim_het']['spearman']:.3f}
  (n = {phen['corr_smf_yes32_vs_kim_het']['n']:,}) — essentially unrelated phenotypes, as expected.
- `sim/phen_corr`: Pearson r of z-scored profiles; median {phen['phen_corr']['quantiles']['0.5']:.3f},
  1st/99th pct {phen['phen_corr']['quantiles']['0.01']:.2f}/{phen['phen_corr']['quantiles']['0.99']:.2f}.
- Pair min/max/product SMF are simple functions of the gene columns and are left to the pair assembler.

## CER
- Allele collapse from raw Data File S1 (not `costanzo_pairs.parquet`, which averages measurements, not
  alleles): {cer['collapse']['n_measurements']:,} measurements → {cer['collapse']['n_strain_pairs']:,} strain
  pairs → {cer['collapse']['n_orf_pairs']:,} ORF pairs ({cer['collapse']['n_orfs']:,} ORFs;
  {100 * cer['collapse']['frac_orf_pairs_multi_allele']:.0f}% have >1 strain pair; negative rate
  {100 * cer['collapse']['frac_orf_pairs_neg']:.1f}%). Same ORF-pair set as the core table; Pearson with its
  measurement mean {cer['vs_core_costanzo_pairs']['pearson_eps']:.5f}
  ({100 * cer['vs_core_costanzo_pairs']['frac_identical_1e-6']:.1f}% identical).
- Essential ORFs (TS/DAmP strain and no deletion strain): {cer['essentiality']['n_essential']:,};
  {cer['essentiality']['n_orfs_with_both_deletion_and_ts_or_damp']} ORFs with both deletion and TS/DAmP
  strains are treated as non-essential.
- Pombe genes with a curated ortholog that has Costanzo data: {cer['n_genes_ortholog_with_costanzo']:,}.
- k-corrected min: eps_min − (E[min of k] − E[min of 1]) under the empirical ORF-pair ε distribution;
  corrections for k = 2..5: {[cer['kmin']['m_k_minus_m_1'][str(k)] for k in (2, 3, 4, 5)]} (max k = {cer['kmin']['kmax']}).
- Ohnologs (YGOB v7 pillars, both S. cerevisiae tracks, ORF names only): {cer['ygob']['n_ohnolog_pairs']} pairs,
  {cer['ygob']['n_pairs_both_in_costanzo']} measured in Costanzo, {cer['ygob']['n_ohnolog_pairs_negative']}
  negative. `CER__ohnolog_sl` = 1 for {cer['pairs']['n_ohnolog_sl_pairs']} pombe pairs.
- Profile PCA (20 comps on the ORF × ORF ε matrix, missing = 0) explains
  {100 * sum(cer['profile_pca']['explained_variance_ratio']):.1f}% of variance (GI profiles are high-dimensional/noisy).

## Coverage: gene-level features
{md_table(gcov)}

## Coverage: pair-level features and similarity matrices (universe × universe pairs)
{md_table(pcov)}

## Similarity-matrix checks (full G × G)
{md_table(simdf)}

## Cross-source consistency (label-free; universe pairs)
Physically interacting pombe pairs should have more similar profiles in independent sources.
{md_table(cons)}

## Conventions and missingness
- PPI: degree/pagerank/edge/shared-neighbour counts are defined for every gene (0 = no recorded
  interaction); `jaccard`, `sp` and `sim/ppi_rwr*` are NaN if either gene has no edge in that network;
  `PPI__edge_cer`, `cer_ortholog_degree` NaN if a gene has no curated cerevisiae ortholog. `sp` = 1..4, 5 = longer
  or disconnected. RWR: restart 0.5, S_ij = sqrt(R_ij R_ji) ∈ [0, 1].
- HTP network = rows with `Throughput == 'High Throughput'` exactly (25 mixed-flag rows excluded).
- CER: `eps_*`, `frac_neg` NaN when no measured ortholog pair; `n_pairs`/`measured` 0 there; `ohnolog_sl`,
  `shared_ortholog` NaN when either gene lacks a curated ortholog.
- PHEN: haploid deletion features are NaN for inviable genes (no deletion mutant) — see DESIGN §1 hypomorph
  indicator; Kim het-diploid fitness does cover essential genes.
- Diagonals of all pair matrices are NaN.
"""
    OUT.write_text(txt)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
