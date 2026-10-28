# DATA.md additions from the tier-1 feature build (for the PI to merge)

Written by the feature engineer, 2026-10-28. Scripts: `scripts/features/build_{genes,ppi,phen,cer}.py`.
Coverage and checks: `docs/features_tier1.md`.

## 1. New dataset: YGOB ohnolog pillars (`ygob`)

- **Citation:** Byrne KP, Wolfe KH. *The Yeast Gene Order Browser: combining curated homology and syntenic
  context reveals gene fate in polyploid species.* Genome Res 15:1456–1461 (2005).
- **Source:** `http://ygob.ucd.ie/ygob/data/v7-Aug2012/Pillars.tab`. Version: YGOB v7, August 2012 (server
  Last-Modified 2012-08-27). Retrieved 2026-10-28; `build_cer.py` downloads it again if it is missing.
- **sha256:** `162c1919b762f34006d0be8ea00b090531ea6c67be43d63b6bf683c4bc48212b`. Size: 2,778,211 bytes, 14,101
  pillars × 33 columns. Stored in `data/raw/ygob/` with `SHA256SUMS` and `RETRIEVED.txt`.
- **Licence:** none stated on the site. Treat as academic use: cite, do not redistribute (`data/` is
  git-ignored).
- **Content used:** columns 12 and 22, which hold the S. cerevisiae gene in YGOB tracks A and B. A pillar
  with an ORF in both columns is a whole-genome-duplication ohnolog pair.
  - 633 pillars have entries in both columns. Restricting to nuclear ORF names gives 545 pairs; this drops
    the `Scer_CEN*` centromere pillars and the `Scer_YGOB_*` placeholders.
  - 477 of the 545 pairs are measured in Costanzo 2016, and 131 of those are negative.
- **Circularity:** ALLOWED. Ohnolog calls come from synteny plus sequence homology and were manually curated
  on gene order. They involve no genetic interactions, no GO and no text mining.
- **use = feature:** CER group only (`CER__ohnolog_sl`, `CER__has_ohnolog_orthologs`,
  `CER__ohnolog_orthologs_neg`).

## 2. Derived file: BioGRID genetic-interaction PMID blocklist

- `data/processed/features/biogrid_genetic_pmids.txt`, sha256
  `1dbdf2b083320d29d703bf686582fd5ecfc6598b03f4a1197cc2f54eb0404e75`. Contents: 2,079 publication IDs plus
  the number of genetic rows per publication. No interaction pairs are stored.
- **How it was made:**
  - `build_ppi.py` re-downloaded `BIOGRID-ORGANISM-5.0.262.tab3.zip`. Its sha256 was
    `096a9e4d871ad5d698bd503b2b9a2409297b88f5b55bc479c710f0c700429820`, which equals the earlier download
    recorded in `data/raw/biogrid/SOURCE_ZIP_SHA256.txt`.
  - The script streamed only the S. pombe member. That member has 83,000 rows, of which 62,842 have
    `Experimental System Type == genetic`.
  - It kept only the `Publication Source` values of those genetic rows, then deleted the zip and its cache
    directory.
- **Use:** physical rows whose publication is on this list are removed, from both the S. pombe and the
  S. cerevisiae networks. Blocking is by publication, never by pair.
- **Effect** (S. pombe physical rows, after dropping non-protein-protein systems):
  - 7,343 of 18,507 rows (40%) and 5,201 of 13,431 edges (39%) are removed, from 723 publications.
    3,485 of the removed rows are HTP-flagged.
  - Vo et al. 2016 (PMID 26771498) is **not** on the list.
  - S. cerevisiae: 591 rows and 338 edges are removed (16 publications).

## 3. Proposed `use = feature` allowlist entries (granular: file / sheet / column)

| dataset id | file | sheet / columns used | NOT used |
|---|---|---|---|
| `biogrid` | `biogrid_5.0.262_spombe_physical.tab3.tsv.gz`, `biogrid_5.0.262_scerevisiae_physical.tab3.tsv.gz` | Systematic Name A/B, Official Symbol A/B (ID rescue only), Experimental System, Publication Source, Throughput | Affinity Capture-RNA, Protein-RNA and Co-localization rows; rows from blocklisted publications; Score/Qualifications/Tags |
| `biogrid` (blocklist) | `biogrid_genetic_pmids.txt` | publication IDs only | any genetic interaction pair |
| `pombase` | `pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz` | pombe ID, cerevisiae ORF | — |
| `pombase` (IDs) | `gene_IDs_names_products.tsv` | systematic ID, name, synonyms, gene_type (ID mapping and `genes.txt` only) | gene_product (not a feature) |
| `costanzo2016` | Data File S1 `SGA_{NxN,ExN,ExE,DAmP}.txt` | Query/Array Strain ID, Genetic interaction score (ε), P-value | `hsp90_data.txt`; SMF/DMF columns of the pair files |
| `costanzo2016` | `strain_ids_and_single_mutant_fitness.xlsx` (via `costanzo_strains.tsv`) | Strain ID (→ class sn/dma/tsq/tsa/damp), Single mutant fitness (30°) | — |
| `rodriguez_lopez_2023_phenomics` | `elife-88229-supp1-v1.xlsx` | sheets *Fitness standard conditions* (fitness_median), *Fitness all conditions* (fitness_median), *Discretized all conditions* | GO enrichments, GO slims, KEGG, K-medoids clusters, Data for clustering, Oxidative stress, Standard conditions, Hits sheets |
| `kim2010_deletion_library` | `41587_2010_BFnbt1628_MOESM5_ESM.xls` (Supp. Table 15) | Systematic name, Average relative fitness (Het_AV), Elimination of strains (filtering) | Supp. Table 1 *Published phenotype*; S-Table 11 dispensability columns |
| `ygob` | `Pillars.tab` | columns 12, 22 | all other columns |

`FYPOviability.tsv` and `universe_genes.tsv` are **not** feature sources. `report_tier1.py` reads them only
as coverage denominators.

## 4. ID-mapping note (corrects the aux inventory's "no mapping needed")

`src/idmap.py` maps every source. It tries the exact systematic ID first, then a unique PomBase synonym,
then a unique gene name. Per-ID logs are in `results/idmap/<source>.tsv`; the per-source summary is in
`results/idmap/summary.tsv`. Losses in the tier-1 sources:

| source | IDs dropped as non-coding | unmapped | rescued as synonyms |
|---|---|---|---|
| BioGRID pombe | 39 | 1 | 0 |
| Kim 2010 S15 | 28 | 16 | 1 |
| Rodríguez-López | 3 | 4 | 3 |
| PomBase orthologs | 38 | 0 | 0 |

The PomBase ortholog losses are non-coding pombe entries. Curated ortholog ORFs that are absent from
Costanzo, and from the filtered cerevisiae BioGRID, are listed in `results/idmap/costanzo2016_ortholog_orfs_absent.tsv`
and `results/idmap/biogrid_scerevisiae_physical_ortholog_orfs_absent.tsv`.
