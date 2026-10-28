#!/usr/bin/env python
"""Coverage + sanity report for the tier-2 label-free features (EXPR, KOPROT, LOC, SEQ, CONS)
-> docs/features_tier2.md (also results/features/tier2_coverage.tsv).

Reads only the feature store, the build checks (results/features/{expr,koprot,loc,seq,cons}_checks.json),
results/idmap/summary.tsv, results/runtime/features_*.json and -- for coverage denominators only -- the
universe gene list (data/processed/universe_genes.tsv: gene, in_universe) and the Ryan gene list
(data/processed/ryan_genes.tsv: pombase_id).  No labels are read.  The tier-1 PPI edge matrix and SEQ paralogue
flags serve as label-free positive controls.
Run: .venv/bin/python -I scripts/features/report_tier2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/features"))
from src import feature_store as fs  # noqa: E402
from src import sim_utils as su  # noqa: E402
from report_tier1 import md_table  # noqa: E402

OUT = ROOT / "docs/features_tier2.md"
RES = ROOT / "results/features"
GROUPS = ("EXPR", "KOPROT", "LOC", "SEQ", "CONS")
SCRIPTS = {"EXPR": "expr", "KOPROT": "koprot", "LOC": "loc", "SEQ": "seq", "CONS": "cons"}
IDMAP_SOURCES = ("coxpresdb_rnaseq", "coxpresdb_array", "pombase_quant_expression", "ozturk2022_ko_strains",
                 "ozturk2022_protein_groups", "matsuyama2006", "pombase_peptidestats", "pombase_protein_domains",
                 "pombase_tm", "compara_pombe_paralogs", "compara_pombe_orthologs", "esm2_650M_pombe")


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
    pairs_U = nU * (nU - 1) // 2
    mine = {k: e for k, e in man.items() if e.get("group") in GROUPS}

    # ---- gene-level coverage
    grows = []
    for k in sorted(k for k in mine if k.startswith("gene/")):
        df = pd.read_parquet(fs.feat_dir() / k).reindex(genes)
        for c in df.columns:
            v = df[c].values
            grows.append({"feature": c, "sources": ",".join(mine[k]["columns"][c].get("sources", mine[k]["sources"])),
                          "genes_all": int(np.isfinite(v).sum()), "genes_universe": int(np.isfinite(v[U]).sum()),
                          "genes_ryan": int(np.isfinite(v[R]).sum()),
                          "median": float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan})
    gcov = pd.DataFrame(grows)

    # ---- pair / sim coverage + distribution on universe x universe pairs
    iuU = np.triu_indices(nU, 1)
    E = np.asarray(np.load(fs.feat_dir() / "pair/PPI__edge.npy", mmap_mode="r")[np.ix_(U, U)])[iuU]
    PAR = np.asarray(np.load(fs.feat_dir() / "pair/SEQ__paralog.npy", mmap_mode="r")[np.ix_(U, U)])[iuU]
    prows, drows, crows = [], [], []
    for k in sorted(k for k in mine if k.startswith(("pair/", "sim/"))):
        M = np.load(fs.feat_dir() / k, mmap_mode="r")
        subm = np.asarray(M[np.ix_(U, U)])
        sub = subm[iuU]
        fin = np.isfinite(sub)
        v = sub[fin]
        binary = v.size > 0 and np.isin(v[:200000], [0, 1]).all() and np.isin(v, [0, 1]).all()
        name = Path(k).stem
        prows.append({"feature": name, "kind": k.split("/")[0], "sources": ",".join(mine[k]["sources"]),
                      "pairs_all": mine[k].get("n_pairs_nonmissing"), "pairs_universe": int(fin.sum()),
                      "frac_universe": float(fin.sum() / pairs_U),
                      "genes_universe_any": int(np.isfinite(subm).any(1).sum()),
                      "n_ones_universe": int((v == 1).sum()) if binary else ""})
        if binary:
            drows.append({"feature": name, "type": "binary", "mean": float(v.mean())})
        else:
            ds = su.dist_summary(subm)
            q = ds["quantiles"]
            drows.append({"feature": name, "type": "continuous", "mean": ds["mean"], "min": q["min"], "q01": q["q01"],
                          "q25": q["q25"], "median": q["median"], "q75": q["q75"], "q99": q["q99"], "max": q["max"],
                          "skewness": ds["skewness"], "frac_at_min": ds["frac_at_min"],
                          "heavily_skewed": ds["heavily_skewed"]})
        e1, e0 = fin & (E == 1), fin & (E == 0)
        p1, p0 = fin & (PAR == 1), fin & (PAR == 0)
        crows.append({"feature": name, "n_ppi_edge_pairs": int(e1.sum()),
                      "mean_ppi_edges": float(sub[e1].mean()) if e1.any() else np.nan,
                      "mean_non_edges": float(sub[e0].mean()) if e0.any() else np.nan,
                      "n_paralog_pairs": int(p1.sum()),
                      "mean_paralogs": float(sub[p1].mean()) if p1.any() else np.nan,
                      "mean_non_paralogs": float(sub[p0].mean()) if p0.any() else np.nan})
        del M, subm, sub
    pcov, dist, cons = pd.DataFrame(prows), pd.DataFrame(drows), pd.DataFrame(crows)
    RES.mkdir(parents=True, exist_ok=True)
    pd.concat([gcov.assign(kind="gene"), pcov]).to_csv(RES / "tier2_coverage.tsv", sep="\t", index=False)

    # ---- full-matrix structural checks for sim/
    simrows = []
    for k in sorted(k for k in mine if k.startswith("sim/")):
        M = np.load(fs.feat_dir() / k)
        v = M[np.isfinite(M)]
        simrows.append({"sim": Path(k).stem, "symmetric": bool(np.array_equal(M, M.T, equal_nan=True)),
                        "diag_all_nan": bool(np.isnan(np.diagonal(M)).all()), "min": float(v.min()),
                        "max": float(v.max()), "genes_with_data": int(np.isfinite(M).any(1).sum())})
        del M
    simdf = pd.DataFrame(simrows)

    ck = {g: json.loads((RES / f"{SCRIPTS[g]}_checks.json").read_text()) for g in GROUPS}
    rt = []
    for g in GROUPS:
        p = ROOT / f"results/runtime/features_{SCRIPTS[g]}.json"
        if p.exists():
            j = json.loads(p.read_text())
            rt.append({"script": f"build_{SCRIPTS[g]}.py", "wall_s": j["wall_s"], "peak_rss_mb": j["peak_rss_mb"]})
    idm = pd.read_csv(ROOT / "results/idmap/summary.tsv", sep="\t")
    idm = idm[idm.source.isin(IDMAP_SOURCES)]
    ex, ko, lo, sq, co = (ck[g] for g in GROUPS)
    skewed = [r["feature"] for r in drows if r.get("heavily_skewed")]
    esm_txt = (f"ESM-2 650M embeddings ({sq['esm']['n_genes']:,} genes × {sq['esm']['dim']}): PCA of centred "
               f"embeddings, {sq['esm']['n_pc']} PCs explain {100 * sq['esm']['explained_variance_ratio_cum']:.1f}% of "
               "variance; `sim/esm_cos` = cosine of centred embeddings (centring removes the large shared component "
               "that otherwise puts raw cosines near 1: median 0.94, 1st percentile 0.78 on random pairs, development check)."
               if isinstance(sq["esm"], dict) else f"**ESM not built**: {sq['esm']}")
    sig = ko["significance"]
    txt = f"""# Tier-2 label-free features (EXPR, KOPROT, LOC, SEQ, CONS): coverage and sanity checks

Generated by `scripts/features/report_tier2.py` (do not edit by hand). Feature store layout:
docs/INTERFACES.md. No Ryan labels were read by any tier-2 builder (enforced by
`tests/features/test_tier2_features.py::test_tier2_scripts_are_label_free`). Every manifest source is an
allowlisted `use: feature` dataset id from `config/data_allowlist.yaml` (tested).

Rebuild (≈2 min total; tier-1 `genes.txt` and PPI must exist for the report's positive controls):
```bash
.venv/bin/python -I scripts/features/build_expr.py
.venv/bin/python -I scripts/features/build_koprot.py
.venv/bin/python -I scripts/features/build_loc.py
.venv/bin/python -I scripts/features/build_seq.py     # ESM part needs data/processed/esm2_650M_pombe.parquet
.venv/bin/python -I scripts/features/build_cons.py
.venv/bin/python -I scripts/features/report_tier2.py  # this file
.venv/bin/python -m pytest tests/features -q
```
Denominators: all = {G} protein-coding genes; universe = {nU} non-essential universe genes
({pairs_U:,} pairs); Ryan = {nR} Ryan genes (gene list only).

## Runtime
{md_table(pd.DataFrame(rt))}

## ID mapping (src/idmap.py; per-ID logs in results/idmap/<source>.tsv)
{md_table(idm)}
COXPRESdb rows are NCBI Entrez IDs: mapped Entrez → PomBase with `ncbi_gene_pombe` (an ID-mapping-only,
`universe-only` dataset; recorded as `id_mapping` in the manifest, not as a feature source), then src/idmap.py.

## EXPR
- COXPRESdb v8.1 matrices (ComBat-corrected log expression): RNA-seq Spo-r.c3-0
  {ex['platforms']['rnaseq']['n_samples']} runs → {ex['platforms']['rnaseq']['n_genes_sim']:,} genes; microarray Spo-m.c4-0
  {ex['platforms']['array']['n_samples']} arrays → {ex['platforms']['array']['n_genes_sim']:,} genes.
- **Normalisation**: each sample median-centred over genes (RNA-seq sample medians spanned
  {ex['platforms']['rnaseq']['sample_median_range_before_centring'][0]:.1f}–{ex['platforms']['rnaseq']['sample_median_range_before_centring'][1]:.1f}
  before centring), then Pearson r across samples (`sim/expr_rnaseq`, `sim/expr_array`). **Rank normalisation**: mutual-rank score
  1 − log(MR)/log(n−1), MR = √(rank_ab·rank_ba) (`pair/EXPR__mr_*`). Gene level = mean log expression (before
  centring); variability = sd of the centred log values.
- Study titles were screened (32 RNA-seq + 9 array studies): mutant/stress/cell-cycle expression profiling,
  some of double mutants, but no genetic-interaction screen data.
- Checks: Spearman of COXPRESdb mean level with Marguerat mRNA copies: RNA-seq {ex['consistency']['spearman_rnaseq_mean_vs_marguerat_mrna']:.2f},
  array {ex['consistency']['spearman_array_mean_vs_marguerat_mrna']:.2f}; Marguerat vs Carpy protein {ex['consistency']['spearman_marguerat_vs_carpy_protein']:.2f};
  RNA-seq vs array co-expression on random pairs r = {ex['consistency']['pearson_rnaseq_vs_array_coexp_random_pairs']['r']:.2f}.
- Marguerat/Carpy copies per cell: log10(x + 0.01) for RNA, log10(x + 1) for protein (pseudocounts
  because {ex['quant']['n_zero_values']['marguerat_rna_prolif']} genes have 0 RNA copies). Condition keys
  (GO:0072690, PBQ:0000003, GO:0000080/84/85/87) are used only to pick the condition column.

## KOPROT
- Öztürk 2022 `Data_Figure_1b.txt`: strainType values verified to be exactly `3,308 knockout strains`
  and `461 wt`; {ko['input']['protein_groups']:,} protein groups ({ko['input']['decoy_groups']} REV__ decoys dropped).
  KO strains mapped: {ko['ids']['ko_strains_mapped']:,}/{ko['ids']['ko_strains']:,}
  (dropped duplicate: {', '.join(f'{k} ({v})' for k, v in ko.get('ko_strains_dropped_duplicate_gene', {}).items()) or 'none'}).
- **Normalisation**: log2 intensity (0 = missing) − strain median over {ko['normalisation']['n_core_proteins']}
  core proteins − per-plate median of KO strains (41 plates); z = residual / (1.4826·MAD over KO strains), sd floored
  at the 5th percentile ({ko['normalisation']['sd_floor']:.3f}). Reference is the KO population, not WT: with the
  same scale, WT controls show a median of {sig['wt_n_changed_median']:.0f} "changed" proteins vs
  {sig['ko_n_changed_median']:.0f} for KO strains, i.e. the WT controls differ systematically from the library.
  The KO'd gene's own protein is masked in its own strain ({ko['ids']['self_protein_values_masked']} values).
- **Threshold** for "significantly changed": |z| ≥ {ko['thresholds']['abs_z']:g} and |log2FC| ≥ log2 1.5. Median
  {sig['ko_n_changed_median']:.0f} (mean {sig['ko_n_changed_mean']:.1f}) per KO; Spearman with the authors'
  `Regulated_pFDR_05` counts = {sig['spearman_vs_paper_Regulated_pFDR_05']:.2f} (authors' median
  {sig['paper_Regulated_pFDR_05_median']:.0f}). The authors' count is also kept as `ko_n_regulated_paper`.
- `sim/koprot_corr`: Pearson r of KO z-profiles over {ko['profile']['n_protein_groups']:,} protein groups quantified in
  ≥50% of KOs (missing → 0, clipped ±10, column-centred), after removing the top {ko['profile']['n_pc_removed']} PCs
  (explained variance {ko['profile']['explained_variance_ratio_top_pcs'][:3]} …). Without removal, strains in
  the same plate row correlate at r ≈ 0.07 (technical) and the BioGRID-edge rate in the top-10,000 KO pairs is 0.0009
  (base 0.0005); with 10 PCs removed it is 0.0033 (n_pc chosen among 0–40 on this label-free control only).
  The first 5 PC scores are gene features (`ko_pc1..5`).
- Directed: `directed/KOPROT__z_ko_row_protein_col.npy` (ASYMMETRIC, [i, j] = z of protein j in KO i;
  {ko['directed']['n_directed_measured']:,} measured, {ko['directed']['n_directed_significant']:,} significant) is an
  **interface extension** (not pair/ or sim/). Model features are the symmetric `KOPROT__a_changes_b` (max |z|
  of the two directions), `KOPROT__min_z` (most negative signed z) and `KOPROT__changed` (either direction
  significant; {ko['directed']['n_pairs_changed']:,} of {ko['directed']['n_pairs_measured_either_direction']:,} measurable pairs).

## LOC
- Matsuyama 2006: {lo['input']['genes']:,} genes mapped; status {lo['input']['status_counts']}.
  The `Localization` grammar (';' segments, '>'/'>>' ranks, '=' ties) is parsed into 13 canonical compartments by
  keyword rules (`COMPARTMENTS` in build_loc.py); primary = first-ranked (tied) compartments of the first segment.
  Unmatched tokens ({lo['input']['n_unmatched_token_occurrences']} occurrences, e.g. 'ambiguous structure',
  'filamentous structure') count as observed without a canonical compartment.
- Compartment counts: {lo['compartment_counts']}.
- `no apparent signal` → one-hot all 0 and `no_signal = 1`; pair features are NaN when either gene has no
  canonical compartment. `nmt1_protein_level_log10` is the **nmt1-driven (overexpression)** YFP-fusion level.

## SEQ
- Domain/family IDs as opaque IDs from {', '.join(sq['id_databases'])}. Not used (not on the PI-confirmed
  sequence-computed list): {', '.join(sq['id_databases_not_used'])}. TM: PomBase TM file
  ({sq['tm_agreement']['genes_tm_file']} genes) and DeepTMHMM ({sq['tm_agreement']['genes_deeptmhmm']}; both {sq['tm_agreement']['both']}).
- Paralogues: Compara within-pombe rows {sq['paralogs']['types']} → {sq['paralogs']['n_pairs']:,} pairs on
  {sq['paralogs']['n_genes_with_paralog']:,} genes. `SEQ__paralog_identity` = max of the two directional % identities;
  median identity: other_paralog {sq['paralogs']['median_identity_by_type']['other_paralog']:.1f}%,
  within_species_paralog {sq['paralogs']['median_identity_by_type']['within_species_paralog']:.1f}%. **Recent** =
  identity ≥ {sq['recent_identity']:g}% ({sq['paralogs']['n_pairs_recent']} pairs, the top {100 * sq['paralogs']['n_pairs_recent'] / sq['paralogs']['n_pairs']:.0f}% of paralogue pairs and
  above the within-species-paralog median); **ancient** = paralogue below it.
- {esm_txt}

## CONS
- Ensembl Compara 116: {co['input']['ortholog_rows']:,} ortholog rows to {co['input']['n_genomes']} genomes;
  {co['genes']['n_with_any_ortholog']:,} genes with ≥1 ortholog; breadth quantiles {co['genes']['n_species_quantiles']}.
- S. japonicus is an Ensembl Fungi genome: `has_sjaponicus` = 1 for {co['genes']['n_has_sjaponicus']:,} genes
  (`n_schizo` also counts S. octosporus / S. cryophilus). `has_scerevisiae` (Compara) for {co['genes']['n_has_scerevisiae']:,}.
- `sim/cons_phylo` = Pearson (phi) of binary presence profiles; NaN for genes without any ortholog (constant
  profile). Phylogenetically redundant genomes are not down-weighted. Profile PCA explained variance:
  {co['genes']['phylo_pca_explained_variance'][:5]} … (PC1 ≈ breadth).

## Coverage: gene-level features
{md_table(gcov)}

## Coverage: pair-level features and similarity matrices (universe × universe pairs)
{md_table(pcov)}

## Value distributions (universe × universe pairs)
Heavily skewed (|skewness| > 2 or > 50% of values at the minimum) continuous features:
**{', '.join(skewed) if skewed else 'none'}** — NBR should rank-transform these (as for tier-1 RWR). The
similarity matrices (`sim/`) are all roughly symmetric around their median and need no special handling.
{md_table(dist)}

## Similarity-matrix checks (full G × G)
{md_table(simdf)}

## Label-free positive controls (universe pairs)
Pairs with a BioGRID physical edge (tier-1 `PPI__edge`) or Compara paralogues (`SEQ__paralog`) should score
higher than other pairs on functional/sequence similarity.
{md_table(cons)}

## Signal notes (label-free; for the PI)
- Strongest PPI-edge enrichment: co-expression (both platforms), ESM cosine, LOC, domain sharing, paralogy.
- `sim/koprot_corr` is weak (mean on PPI edges {cons.set_index('feature').loc['koprot_corr', 'mean_ppi_edges']:.4f} vs
  {cons.set_index('feature').loc['koprot_corr', 'mean_non_edges']:.4f}) even after technical-PC removal; the directed
  `KOPROT__changed` is the more PPI-enriched KOPROT pair feature.
- `sim/cons_phylo` shows no PPI-edge enrichment (dominated by conservation breadth, PC1 = 65% of variance) but is
  enriched for paralogues; phylogenetic redundancy of the 367 genomes is not corrected.

## Conventions and missingness
- NaN = no data for either gene in that source; diagonals of all pair/sim matrices are NaN.
- Count/flag features defined for every gene with a peptide (0 = none): SEQ domain counts, TM counts,
  paralogue features (Compara covers the whole proteome), `SEQ__shared_pfam`, `SEQ__shared_panther`; CONS breadth
  and presence flags. `SEQ__domain_jaccard` is NaN only if neither gene has any domain ID.
- KOPROT gene features: `ko_*` are defined for deleted genes (non-essential, in the Bioneer library);
  `protein_frac_kos_changed` for genes whose protein was quantified.
"""
    OUT.write_text(txt)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
