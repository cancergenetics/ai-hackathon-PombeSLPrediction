# DATA.md — datasets, provenance, licences

All raw data live in `data/raw/<dataset>/` and processed tables in `data/processed/`; both are git-ignored
and are **never committed**. Everything is re-created by scripts:

```bash
.venv/bin/python -I scripts/download_core_data.py     # download + sha256 -> data/raw/core_manifest.tsv, data/raw/<ds>/SHA256SUMS
.venv/bin/python -I scripts/build_ryan_pairs.py       # Ryan pair table (~10 s)
.venv/bin/python -I scripts/build_costanzo_pairs.py   # Costanzo pair table (~2 min, ~11 GB RAM peak)
.venv/bin/python -I scripts/eda_core_data.py          # -> docs/data_inventory.md
.venv/bin/python -I scripts/verify_core_data.py       # checksums, row counts, paper facts, controls
```

Downloaded files are treated as untrusted data: nothing in them is executed; Python is run with `-I`;
zip archives are extracted with a path-traversal check. Retrieval date for all core files: **2026-10-28**.

Machine-readable counterpart: **`config/data_allowlist.yaml`** (dataset id → use, files with sha256, allowed
archive members / sheets / columns, rolling-URL flags, excluded sources). The two are generated from the same
information; if they disagree, fix both. Loaders are default-deny: anything not listed there must not be read.

## Dataset register and `use` categories

Every dataset carries a `use` field:

| `use` | meaning |
|---|---|
| `labels` | the training / evaluation target (Ryan 2012 S1/S2 only) |
| `feature` | allowed model input |
| `universe-only` | defines the gene universe, ID mapping or non-essential set; never a model input (e.g. FYPO viability) |
| `excluded` | forbidden (circularity under the PROMPT.md rules, or licence); must not be loaded; a test should grep feature tables for these sources |

Per-file overrides exist (e.g. Ryan Datasets S3/S4 are `excluded` although Ryan is `labels`). `status: provisional`
marks files whose use was inferred by the documentation pass rather than named in the PI's allowlist instruction
(sequence-computed PomBase files other than peptides/Pfam/PANTHER, AlphaFold, other InterPro member databases).

**Licence rules.** Raw and processed data are never committed. **Costanzo 2016, Kim 2010 and Matsuyama 2006: licence
unclear — never redistribute (including derived pair-level tables).** Ryan 2012 (Elsevier user licence: no
redistribution or adaptation) is treated the same way. PomBase's CC BY 4.0 is known only from secondary sources.

| id | dataset | use | version / release | retrieved | licence (status) | rolling URL? |
|---|---|---|---|---|---|---|
| `ryan2012` | Ryan et al. 2012 S. pombe E-MAP | **labels** | as published 2012 (no revisions) | 2026-10-28 | Elsevier user licence 1.0 (limited) | no |
| `costanzo2016` | Costanzo et al. 2016 S. cerevisiae global genetic interaction map | **feature** | published dataset; server Last-Modified 2016-11-28 (no versioning) | 2026-10-28 | UNCLEAR: no licence stated on the supplement site or TheCellMap (unclear) | no |
| `pombase_ids_viability` | PomBase release 2026-10-01: gene IDs and FYPO viability | **universe-only** | pombase-2026-10-01 (pinned; Chado dump 2026-09-30) | 2026-10-28 | CC BY 4.0 per secondary sources (secondary-source only) | no |
| `pombase_orthologs` | PomBase manually curated S. pombe - S. cerevisiae orthologs | **feature** | pombase-2026-10-01 (pinned) | 2026-10-28 | as PomBase above (secondary-source only) | no |
| `pombase_protein` | PomBase protein sequences and sequence-computed domain calls | **feature** | PomBase rolling /data/ files at retrieval; nearest release 2026-10-01  | 2026-10-28 | CC BY 4.0 per secondary sources (secondary-source only) | **yes** |
| `sgd_protein` | SGD S. cerevisiae ORF translations | **feature** | SGD genome release R64-5-1 (FASTA headers); server Last-Modified 2024- | 2026-10-28 | SGD: freely available, CC BY 4.0 per SGD terms (stated, not re-verified) | **yes** |
| `esm2_weights` | ESM-2 weights | **feature** | fair-esm 2.0.0 release weights | 2026-10-28 | MIT (clear) | no |
| `pombase_quant_expression` | PomBase quantitative gene expression: Marguerat 2012 RNA/protein, Carpy 2014 protein | **feature** | release 2026-10-01 (latest at retrieval) | 2026-10-28 | CC BY 4.0 per secondary sources (secondary-source only) | **yes** |
| `biogrid_physical` | BioGRID 5.0.262 physical interactions | **feature** | BioGRID release 5.0.262 (pinned archive URL) | 2026-10-28 | MIT (clear) | no |
| `coxpresdb_spo` | COXPRESdb S. pombe expression matrices | **feature** | v8.1 (release 2022-07): Spo-r.c3-0 (v21-10), Spo-m.c4-0 (v21-06); vers | 2026-10-28 | CC BY 4.0 (clear) | no |
| `ozturk2022_ko_proteome` | Öztürk et al. 2022 KO proteomes | **feature** | as published 2022 (fixed file) | 2026-10-28 | CC BY 4.0 (clear) | no |
| `rodriguez_lopez_2023_phenomics` | Rodríguez-López et al. 2023 deletion-mutant phenomics | **feature** | as published (v1 file) | 2026-10-28 | CC BY 4.0 (clear) | no |
| `kim2010_deletion_library` | Kim et al. 2010 Bioneer deletion library | **feature** | as published 2010 (fixed files) | 2026-10-28 | UNCLEAR: Springer Nature supplementary data, no explicit open licence (unclear) | no |
| `matsuyama2006_localization` | Matsuyama et al. 2006 ORFeome YFP localisation | **feature** | RIKEN BRC file as deposited 2007-07-02 (unversioned URL) | 2026-10-28 | UNCLEAR: no licence stated (unclear) | no |
| `ensembl_fungi_compara` | Ensembl Fungi Compara 116 homologies | **feature** | Compara 116 (Ensembl Genomes release 63) | 2026-10-28 | Ensembl Genomes data free of restrictions (clear) | **yes** |
| `alphafold_spombe` | AlphaFold DB S. pombe reference proteome UP000002485 | **feature** | AFDB v6 (in file name); remote Last-Modified 2025-08-24, size unchange | 2026-10-28 | CC BY 4.0 (clear) | **yes** |
| `uniprot_idmapping_pombe` | UniProt ID mapping for S. pombe | **universe-only** | UniProt release 2026_03 (02-Sep-2026); file Last-Modified 2026-09-03 | 2026-10-28 | CC BY 4.0 (clear) | **yes** |
| `ncbi_gene_pombe` | NCBI Datasets gene report, S. pombe | **universe-only** | API snapshot 2026-10-28 (unversioned) | 2026-10-28 | Public domain (clear) | **yes** |
| `fypo_phenotype_annotations` | PomBase FYPO phenotype annotations (phenotype_annotations.pombase.phaf.gz, canto_fypo_anno | **excluded** | not retrieved | - | - | - |
| `biogrid_genetic` | BioGRID genetic interaction rows (S. pombe and others) | **excluded** | not retrieved | - | - | - |
| `pombase_interactions` | PomBase interactions.tsv / interactions_with_comments.tsv | **excluded** | not retrieved | - | - | - |
| `string` | STRING (any channel) and text-mining-derived networks (GeneMANIA, HumanNet) | **excluded** | not retrieved | - | - | - |
| `gene_ontology` | Gene Ontology: GAF, GO slims, GOCAM, PomBase Complex_annotation.tsv, pombase-go-physical-i | **excluded** | not retrieved | - | - | - |
| `net_ff_go_predictions` | Rodriguez-Lopez 2023 Supplementary file 2 (NET-FF GO predictions) and the phenomics GO enr | **excluded** | not retrieved | - | - | - |
| `koch2012_additional_file2` | Koch et al. 2012 Additional file 2 (S. pombe SGA control-screen fitness, S. pombe GI degre | **excluded** | not retrieved | - | - | - |
| `other_pombe_gi_screens` | Roguev 2008, Dixon 2008, Frost 2012 and any other fission-yeast GI screen, from any source | **excluded** | not retrieved | - | - | - |
| `coxpresdb_coex_tables` | COXPRESdb coex / Spo-u union tables | **excluded** | not retrieved | - | - | - |
| `ryan_derived_tables` | Ryan Table S2 modules, Dataset S3 similarity, Dataset S4 (see ryan2012 files) | **excluded** | not retrieved | - | - | - |

**Rolling ("latest"/"current") URLs.** Datasets fetched from a rolling URL are flagged `yes` above and described in
their section. Summary of the check made on 2026-10-28 (nothing large was re-downloaded):

| dataset | rolling URL used | pinned equivalent | status |
|---|---|---|---|
| `pombase_quant_expression` | `releases/latest/misc/` | `releases/pombase-2026-10-01/misc/` (same file names) | exists; **byte-identical** (sha256 checked) |
| `pombase_protein` | `/data/...` (unversioned) | `releases/pombase-2026-10-01/fasta/feature_sequences/peptide.fa.gz`; `.../misc/{PeptideStats,aa_composition,transmembrane_domain_coords_and_seqs}.tsv`; `.../misc/ProteinFeatures.tsv` (= `protein_domains_and_features.tsv`) | exists; local files **not byte-identical** except `aa_composition.tsv`; peptide sequences identical (5,126 genes) so ESM-2 embeddings are unaffected; pinned sha256s are in the allowlist |
| `ensembl_fungi_compara` | `pub/current/` (Compara 116) | `pub/release-63/fungi/tsv/ensembl-compara/homologies/Compara.116.protein_default.homologies.tsv.gz` | exists (7.07 GB), not downloaded; source md5 was verified |
| `uniprot_idmapping_pombe` | `current_release` (2026_03) | none (`previous_releases/release-2026_03` returns 404 until the next release) | file size and Last-Modified re-checked; sha256 is the pin |
| `alphafold_spombe` | `.../alphafold/latest/` | none verified; file name carries `v6`, remote Last-Modified 2025-08-24 unchanged | low risk |
| `sgd_protein` | `S288C_reference/` directory | none checked; headers state R64-5-1 | low risk |
| `ncbi_gene_pombe` | NCBI Datasets API | none (API) | retrieval date + sha256 only |

Recommended follow-up (not done here; needs a code change): point `scripts/download_aux_data.py` at the pinned URLs for
the PomBase protein files, quant-expression tables and Compara.

## Core datasets

### 1. Ryan et al. 2012 — S. pombe genome-scale E-MAP (training labels)

- **Citation:** Ryan CJ, Roguev A, Patrick K, … Krogan NJ. *Hierarchical modularity and the evolution of genetic
  interactomes across species.* Mol Cell 46(5):691–704 (2012). doi:10.1016/j.molcel.2012.05.028; PMC3380636.
- **Source (authoritative):** the journal's supplementary files on the Elsevier CDN (PII S1097276512004443):
  `https://ars.els-cdn.com/content/image/1-s2.0-S1097276512004443-mmc{1..8}.*` (server Last-Modified 2012-08-02).
  The same files exist on PMC as `NIHMS380848-supplement-02..08`, but PMC now serves a proof-of-work
  browser challenge to scripts, so the Elsevier copy is used. BioGRID was **not** used.
- **Version:** as published (2012); no later revisions exist.
- **Licence / terms:** article in Elsevier's open archive under the *Elsevier user license 1.0*
  (https://www.elsevier.com/open-access/userlicense/1.0/): non-commercial access, download, copying and
  text/data mining allowed; **no redistribution or adaptation** → do not commit or redistribute the files;
  cite the article. The licence does not mention supplementary data explicitly (treat the same way).

| file (data/raw/ryan2012/) | content | bytes | sha256 |
|---|---|---|---|
| mmc1.xls | Table S1: 953 NAT query strain labels, 1955 G418 array strain labels, species distribution, process annotations (pombe/cerevisiae) | 1093120 | 7cb13179e55cf3c2f61687963bf7f82afd6e21d12678fb01f8e07a7270bfb208 |
| mmc2.xls | Table S2: functional modules, conserved modules | 177664 | 49691b1cd750f8d22fa6bc609783e8a5bd20eeb15c5a407e44c7316b23777523 |
| mmc3.xls | Table S3: between-process enrichment | 37376 | a6d8e8d3c4d8d4a53856400e369735537b88cb3642149e7b926c91a3120810eb |
| mmc4.zip | **Dataset S1 – Unaveraged E-MAP with 953 alleles** (953 query alleles × 1955 arrays, raw S-scores) | 6005873 | 76e1cc1f0fcfc34448d749a1888dcc89f850b6101d51ee4b67ba4d361dd73e4c |
| mmc5.zip | **Dataset S2 – Averaged E-MAP, one allele per gene** (862 × 1955, symmetric, pseudo-averaged S-scores; CR line endings) | 7662684 | 34be523bcc2a65c0fb966dbb7ac94c1bb9dc843f62ae8f751cf8d0661708efcb |
| mmc6.zip | Dataset S3 – S. pombe profile similarity scores (derived from the GI data) | 6864192 | 4115a44ee29e37a1af8343bcd8cb3172a95c58ea6031f5b8d966b840dee91a04 |
| mmc7.zip | Dataset S4 – S. cerevisiae merged/scaled data (Costanzo 2010 SGA + E-MAPs) | 19104161 | b157b9a185ce9845f2aad97718833c9a67b580f9f94dbeac1e3bb83a04162326 |
| mmc8_supplementary_methods.pdf | Supplemental Information (methods, Fig. S1–S7) | 2479925 | b85475829072161cf6ec464e9431851c14297752c8d6c79bfb020d31557c6bce |

- **Per-mutant metadata available:** query/array identity and role, allele type (from label suffix: deletion,
  DAmP, point mutants, ts alleles, extra deletion strains …), S. pombe gene name and S. cerevisiae ortholog
  names embedded in labels (pombe names have every `0` stripped, e.g. `hsp9` = hsp90). **Not available:**
  single-mutant fitness / colony sizes, batch IDs, replicate-level scores, p-values (not in the supplement).
- **`use`: `labels`** (Datasets S1/S2 only; Tables S1–S3 and Datasets S3/S4 are `excluded`). Sole source of training/evaluation labels (S < -2.3). Datasets S1/S2 are used.
  **Not used as model input:** mmc1 "Process Annotations" (GO-derived → excluded by project rules),
  mmc2 modules and mmc6 similarity scores (derived from the pombe GI data itself → label leakage if used as
  features across splits), mmc7 (we use Costanzo 2016 directly instead).
- **Processed:** `data/processed/ryan_pairs.parquet` (1,359,493 gene pairs; 1,337,033 with S2),
  `ryan_measurements.parquet` (1,554,025 S1 measurements), `ryan_genes.tsv` (ID mapping + metadata),
  `ryan_s2_transform.tsv`. Column definitions in `scripts/build_ryan_pairs.py`; EDA in `docs/data_inventory.md`.

### 2. Costanzo et al. 2016 — global S. cerevisiae genetic interaction map (input features)

- **Citation:** Costanzo M, VanderSluis B, Koch EN, … Boone C. *A global genetic interaction network maps a
  wiring diagram of cellular function.* Science 353(6306):aaf1420 (2016). doi:10.1126/science.aaf1420.
- **Source:** Boone lab supplement site, Data File S1 (pair-wise format), linked from TheCellMap:
  `https://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/data_files/Data%20File%20S1_Raw%20genetic%20interaction%20datasets%3A%20Pair-wise%20interaction%20format.zip`
  (mirror: `https://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip`).
- **Version:** server Last-Modified 2016-11-28 (published dataset; no versioning).
- **Licence / terms:** **unclear** — no licence stated on the supplement site or TheCellMap; published as Science
  supplementary material. Treat as academic, non-redistributable; cite Costanzo et al. 2016 (and Usaj et al.
  2017 G3 for TheCellMap).

| file (data/raw/costanzo2016/) | bytes | sha256 |
|---|---|---|
| Data_File_S1_pairwise.zip | 431580254 | 2134d990e66f754605e936d1bab70f874e264d1467c9a08534b054570fbe3775 |

Extracted (`extracted/Data_File_S1_pairwise/…/`): `SGA_NxN.txt` (11,295,484 rows), `SGA_ExN.txt` (3,881,456;
contains both ExN and NxE), `SGA_ExE.txt` (601,690), `SGA_DAmP.txt` (1,370,104), `hsp90_data.txt` (special
HSP90 screen, not used), `strain_ids_and_single_mutant_fitness.xlsx` (10,859 strains: ORF, allele, SMF at
26/30 °C). Columns: query/array strain ID and allele, array type/temperature, ε, p-value, query/array SMF,
DMF, DMF sd.

- **`use`: `feature`** — **licence unclear: never redistribute (including derived pair-level tables).** Allowed archive members: `SGA_NxN.txt`, `SGA_ExN.txt`, `SGA_ExE.txt`, `SGA_DAmP.txt`, `strain_ids_and_single_mutant_fitness.xlsx`; `hsp90_data.txt` is `excluded`. Cross-species feature source (ortholog-mapped ε, profiles, SMF). Processed into
  `data/processed/costanzo_pairs.parquet` (11,430,205 unordered ORF pairs; mean/min/max ε over all
  strain-level measurements, min p, sources) and `costanzo_strains.tsv`.

### 3. PomBase reference files (IDs, viability, orthologs)

- **`use`:** `gene_IDs_names_products.tsv` and `FYPOviability.tsv` = `universe-only` (viability = single-deletion class, never a feature); the curated ortholog export = `feature`; `cerevisiae_orthologs_curation_table.tsv` = `excluded` (provenance only, not loaded). Pinned release URL (no rolling-URL issue). FYPO phenotype *annotations* are `excluded`.

- **Source:** PomBase monthly release **pombase-2026-10-01** (pinned; Chado dump of 2026-09-30),
  `https://www.pombase.org/data/releases/pombase-2026-10-01/`.
- **Licence / terms:** PomBase curated data are distributed under **CC BY 4.0** according to secondary sources
  (Database Commons, Wikipedia infobox); PomBase's own terms page is JavaScript-rendered and could not be read
  by script — **verify**. Cite PomBase (Rutherford et al. 2024, Genetics).

| file (data/raw/pombase/) | URL path (under the release) | bytes | sha256 | use |
|---|---|---|---|---|
| gene_IDs_names_products.tsv | misc/gene_IDs_names_products.tsv | 1334868 | 7f2f228fe66e81588ff708b209050cdc00804f9876fb44bbfb641bce4e5542fb | current systematic IDs, names, synonyms (incl. old sysIDs), gene_type (protein-coding list: 5,126) |
| FYPOviability.tsv | misc/FYPOviability.tsv | 261142 | 3c9c55d45d8ed98f7fe56ca3cf43c964e0e4882ffb108fec12e3d80876c6cac5 | deletion viability (viable / inviable / condition-dependent / unknown) → non-essential universe. Single-mutant data, allowed. |
| pombe-cerevisiae-orthologs-with-systematic-ids.txt.gz | exports/… | 29223 | 55d15c874a467b60535e6d5cd91738625c1571f7d4ed6213f8508b8837efdc6b | manually curated pombe↔cerevisiae orthologs (pombe sysID → cerevisiae ORF; 5,334 rows) |
| cerevisiae_orthologs_curation_table.tsv | pombe-embl/orthologs/cerevisiae_orthologs.tsv | 95621 | 3d9ee37537c3c456da142285cc1a9d67a40177371f4aa67c4f6b7d912916554e | raw curation table (with N/C-term qualifiers); provenance only |

No separate SGD mapping was needed: Costanzo files and the PomBase export both use S. cerevisiae systematic ORF
names; all ortholog ORFs used are matched directly.

## Decisions flagged for the PI (current defaults in brackets)

1. **Label scale [S2 `s_score` < -2.3].** Dataset S2 is the authors' *pseudo-averaged* score (Collins et al.
   2006): for reciprocally measured pairs (62k) S2 ≈ mean of the two raw scores; for single-orientation pairs
   (1.28M, 95%) the raw score is shrunk by a global monotone map, so S2 = -2.3 ⇔ raw ≈ -3.76. Positive rate:
   1.88% (S2) vs 4.31% (raw mean < -2.3). We default to S2 because it is the authors' one-value-per-pair
   dataset, on the scale where the paper applies -2.3 (and S > 1.8). Verified exactly by
   `verify_core_data.py`. Raw values are kept (`s_raw_mean`, `s_raw_ab`, `s_raw_ba`).
2. **Duplicate measurements [use S2 as is].** Options considered: (a) authors' S2 (mean over orientations and
   over the gene's main allele(s), then pseudo-averaged) — *chosen*; (b) plain mean of raw S1 measurements;
   (c) most-negative measurement (inflates positives); (d) keep each measurement as a separate training row
   with gene-grouped splits; (e) require agreement across orientations. Reciprocal-orientation agreement is
   low (r = 0.40; only 32% of raw S<-2.3 calls replicate in the other orientation), so labels are noisy;
   (d) or a noise-aware loss could be revisited.
3. **Alleles [main alleles only].** 953 query alleles / 876 genes; S2 uses, per gene, the unlabelled strain
   plus DAmP strain(s) (and the single "DELEITON" strain of php5). Point mutants, ts alleles, extra deletion
   and "KLP" strains (46 multi-allele genes; 14 query genes have only such alleles and are absent from S2) are excluded from `s_score`;
   22,460 pairs exist only via such alleles (`in_s2 == False`). Pairs involving DAmP/hypomorphic alleles of
   essential genes fall outside the non-essential universe anyway.
4. **Non-essential universe [PomBase FYPO `viable` protein-coding genes: 3,580 genes, 6.41M pairs].**
   Alternative: also include `condition-dependent` (3,757 genes, 7.06M pairs). 166 protein-coding genes have
   no viability annotation and are excluded. 12 Ryan array genes are now annotated inviable.
5. **ID merges.** 4 current genes each absorb two Ryan IDs (e.g. SPAPB1A11.05 → SPAC31G5.01/sap49); their
   values are averaged and flagged (`involves_merged_id`, 7,105 pairs). 3 array IDs could not be mapped
   (SPAC1F12.03C, SPCC417.04 unmapped; SPAC1834.12 ambiguous) and are dropped; SPAC12D12.09 (rev7, a typo
   for SPBC12D12.09) is mapped via its label gene name.

## Excluded sources and why

| source | reason |
|---|---|
| BioGRID S. pombe genetic interactions (incl. its copy of Ryan 2012), Roguev 2008, Frost 2012, Dixon 2008, any other fission-yeast GI screen | Project rule: no fission yeast genetic interactions other than Ryan 2012, from any source (circularity / test contamination). Ryan data taken from the journal supplement, not BioGRID. |
| PomBase `interactions.tsv` / phenotype annotations of double mutants | contain fission-yeast genetic interactions (same rule). |
| STRING (any channel) and other text-mining-derived resources | Project rule: text mining / derived resources excluded. |
| Gene Ontology (incl. PomBase GO annotations, GO slims) and Ryan Table S1 "Process Annotations" | Project rule: GO annotations may derive from interaction experiments. |
| Ryan Table S2 modules, Dataset S3 similarity scores | derived from the pombe GI matrix itself → leakage across train/test splits. |
| Ryan Dataset S4 (merged S. cerevisiae data) | superseded by Costanzo 2016 used directly. |
| Costanzo 2016 `hsp90_data.txt` | special-purpose HSP90 screen, not part of the global map. |
| PomBase FYPO phenotype annotations (`phaf`, `canto_fypo_annotations*`, `public_api_genotypes_phenotypes`) | multi-allele / double-mutant phenotypes include fission-yeast GI results. Not downloaded. Only the FYPO *viability* table is used (`universe-only`). |
| Koch et al. 2012 Additional file 2 (S. pombe degree, SMF/fitness, GO multifunctionality columns) | PI decision: degree and SMF columns excluded (degree derives from fission-yeast GI screens; multifunctionality is GO). Not downloaded. |
| Rodríguez-López 2023 Supplementary file 2 (NET-FF) and the GO enrichments / GO slims / KEGG sheets | GO-derived; NET-FF integrates STRING-type networks. Not retained. |
| PomBase `Complex_annotation.tsv`, `go-physical-interactions`, `interactions.tsv`; COXPRESdb `coex` / `Spo-u` tables; UniProt ID-mapping GO rows; InterPro2GO | GO-derived or contain curated GI; never read. |
| Costanzo/Kim/Matsuyama as *redistributed* data | licence unclear: usable as inputs, never committed or shared (incl. derived pair-level tables). |

## Auxiliary datasets

Merged from `docs/data_aux_inventory.md` (retrieved 2026-10-28 by `scripts/download_aux_data.py`; per-directory
`SHA256SUMS` + `RETRIEVED.txt`; collected in `data/raw/aux_checksums.tsv`). Coverage numbers, suggested features and
the rejected/deferred list remain in that document. Checksums below are read from the `SHA256SUMS` files.
Corrections to that inventory: Öztürk 2022 is Nat Commun **13:6153** (not 6232); the Kim 2010 `Gene function` column is
additionally forbidden. Not downloaded and therefore not in the register: Vo 2016 supplement (its edges are inside
BioGRID), ProtT5 embeddings, eggNOG/OrthoDB/OMA, GEO series, Rhind 2011/Jeffares 2015 VCFs.

### pombase_protein — PomBase protein sequences and sequence-computed domain calls

- **use:** `feature` (confirmed (peptides, Pfam, PANTHER); provisional (other files/databases))
- **Source:** https://www.pombase.org/data/{genome_sequence_and_features/feature_sequences/peptide.fa.gz, Protein_data/*.tsv} (unversioned current files, dated 2026-10-28)
- **Version:** PomBase rolling /data/ files at retrieval; nearest release 2026-10-01 (not byte-identical, see rolling flag); retrieved 2026-10-28
- **Licence:** CC BY 4.0 per secondary sources (not confirmed at pombase.org) [status: secondary-source only]
- **Allowed use:** Peptide sequences (ESM-2 input) and Pfam / PANTHER family IDs for shared-domain / family features. InterPro-to-GO mappings must never be used.
- **ROLLING-URL FLAG:** ROLLING: fetched from unversioned /data/ paths. Pinned release-2026-10-01 equivalents exist (checked 2026-10-28, not re-downloaded; hashes below were computed by streaming): fasta/feature_sequences/peptide.fa.gz (sha256 76c1eb8212c193ed24143ab2693e19ec7ca9a2f2e9127069098b7409cc9ff2d6), misc/PeptideStats.tsv (e2fce7bd5f706f8cbf6ff48fa7f757dde24639a290b7af14b782bbae4f7f57d4), misc/aa_composition.tsv (a52b79d80569ad72f19918702ed4516b5dc5bef95905a4363d219d72d0d68e13, IDENTICAL to local), misc/transmembrane_domain_coords_and_seqs.tsv (ec5c7211243e10fd4676abebd2062a72b2cc4fde1637a37003997b39499c7e8e), misc/ProteinFeatures.tsv (cbcc05a13f338384b573abac1c1d04a0c5ae62ebe0289dec0202d790c27cee1f; same header as local protein_domains_and_features.tsv, renamed). Local files differ in bytes from the pinned ones except aa_composition.tsv. peptide.fa.gz: same 5,126 gene IDs and identical sequences (only FASTA header text differs), so ESM-2 embeddings are unaffected. The other three differing files were not diffed.
- **Allowlist detail:** `{"protein_domains_and_features.tsv": {"database_allowed": ["Pfam", "PANTHER"], "database_pending_pi_decision": ["CATH-Gene3D", "SUPERFAMILY", "SMART", "PROSITE", "CDD", "PRINTS", "DeepTMHMM", "MobiDB", "COILS"]}}`

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `peptide.fa.gz` | feature | 1620552 | `803b4146fe270338b740459cf26567b24f9d3dae7d9484203d27e5350108f19b` | 5,126 protein-coding peptides |
| `protein_domains_and_features.tsv` | feature | 4167771 | `139a4f97d9e5f2fd0705cd9bd9d50d1779c0f6f6d49df308ef14a135c623bd3b` | allowed databases: Pfam, PANTHER (family/subfamily IDs as opaque IDs). Other databases in the file (CATH-Gene3D, SUPERFAMILY, SMART, PROSITE, CDD, PRINTS, DeepTMHMM, MobiDB, coiled coils) are sequence-computed but NOT on the PI allowlist: provisional, PI to confirm |
| `PeptideStats.tsv` | feature | 194436 | `62c6fb25558c4bf3e3a4746e60f479cec58413b47eb6c333b6f1b822ca704426` | provisional: sequence-computed (length, mass, pI...) |
| `aa_composition.tsv` | feature | 340748 | `a52b79d80569ad72f19918702ed4516b5dc5bef95905a4363d219d72d0d68e13` | provisional: sequence-computed |
| `transmembrane_domain_coords_and_seqs.tsv` | feature | 576268 | `a747e6736a2378171326b39317a99c1cf868a6b79a156923be7563d3bbaccd6e` | provisional: sequence-computed |

### sgd_protein — SGD S. cerevisiae ORF translations (S288C R64-5-1)

- **use:** `feature` (confirmed)
- **Source:** https://downloads.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans_all.fasta.gz
- **Version:** SGD genome release R64-5-1 (FASTA headers); server Last-Modified 2024-06-13; retrieved 2026-10-28
- **Licence:** SGD: freely available, CC BY 4.0 per SGD terms (not independently re-verified); cite Engel et al. Genetics 2022 [status: stated, not re-verified]
- **Allowed use:** Input to ESM-2 embeddings for S. cerevisiae orthologs.
- **ROLLING-URL FLAG:** ROLLING DIRECTORY (S288C_reference/...), but the release is identifiable from the FASTA headers (R64-5-1) and the Last-Modified date; no pinned archive URL checked.

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `orf_trans_all.fasta.gz` | feature | 2689634 | `17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b` |  |

### esm2_weights — ESM-2 weights (esm2_t33_650M_UR50D primary; t30_150M backup unused)

- **use:** `feature` (confirmed)
- **Source:** https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt (+ regression/*-contact-regression.pt)
- **Version:** fair-esm 2.0.0 release weights; retrieved 2026-10-28
- **Licence:** MIT (weights and code); Lin et al. Science 379:1123 (2023) [status: clear]
- **Allowed use:** Pretrained protein language model (trained on UniRef50 sequences only; not circular with GI data). Embeddings from peptide/ORF sequences.

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `esm2_t33_650M_UR50D.pt` | feature | 2604537549 | `ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c` |  |
| `esm2_t30_150M_UR50D.pt` | feature | 592774773 | `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62` | backup, not used |
| `esm2_t33_650M_UR50D-contact-regression.pt` | feature | 3687 | `8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575` | loader requirement, unused |
| `esm2_t30_150M_UR50D-contact-regression.pt` | feature | 3431 | `6a604b96722ed052eef8a094ad90b275ba2e987d406315dbed0bdc6b3c4238a7` | loader requirement, unused |

### pombase_quant_expression — PomBase quantitative gene expression: Marguerat 2012 RNA/protein, Carpy 2014 protein

- **use:** `feature` (confirmed)
- **Source:** https://www.pombase.org/data/releases/latest/misc/{htp,full}_gene_expression_table.tsv
- **Version:** release 2026-10-01 (latest at retrieval); retrieved 2026-10-28
- **Licence:** CC BY 4.0 per secondary sources (PomBase); primary papers Marguerat et al. Cell 151:671 (2012), Carpy et al. MCP 13:1925 (2014) [status: secondary-source only]
- **Allowed use:** Gene-level abundance features. The during(GO:...)/during(PBQ:...) strings are experiment-condition keys, never GO features.
- **ROLLING-URL FLAG:** ROLLING ('latest' URL). Pinned URL https://www.pombase.org/data/releases/pombase-2026-10-01/misc/{htp,full}_gene_expression_table.tsv exists and was verified 2026-10-28 to be BYTE-IDENTICAL (same sha256) to the local files. Use the pinned URL in scripts.
- **Allowlist detail:** `{"columns": "gene, study, RNA/protein, copies per cell (+ condition as key only)", "condition_strings": "GO:/PBQ: strings used only as condition keys"}`

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `htp_gene_expression_table.tsv` | feature | 4137911 | `bc512e72e636f5c8600a4f589d272dbc1d873d86283cd83c90858ff6453901fa` |  |
| `full_gene_expression_table.tsv` | feature | 5181827 | `0bb00425e7fa9121f6b8513451d13ed4c77e03c7402d864a4960ac6313ebecf1` |  |

### biogrid_physical — BioGRID 5.0.262 physical interactions (S. pombe, S. cerevisiae)

- **use:** `feature` (confirmed)
- **Source:** https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.262/BIOGRID-ORGANISM-5.0.262.tab3.zip (file dated 2026-09-29)
- **Version:** BioGRID release 5.0.262 (pinned archive URL); retrieved 2026-10-28
- **Licence:** MIT (BioGRID terms); cite Oughtred et al. Protein Sci 30:187 (2021) [status: clear]
- **Allowed use:** Physical PPI features only (Experimental System Type == physical; drop Affinity Capture-RNA, Protein-RNA, Co-localization for PPI features). The unfiltered source zip (sha256 096a9e4d871ad5d698bd503b2b9a2409297b88f5b55bc479c710f0c700429820) contained 62,842 S. pombe GENETIC rows (incl. Ryan 2012) and was deleted after filtering; BioGRID genetic data are excluded.
- **Allowlist detail:** `{"filter": "Experimental System Type == physical", "drop_columns": "Ontology Term*", "drop_systems_for_ppi": ["Affinity Capture-RNA", "Protein-RNA", "Co-localization"]}`
- **Notes:** Optional ablations: BioGRID-HTP-only; exclude Source Database == POMBASE.

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `biogrid_5.0.262_spombe_physical.tab3.tsv.gz` | feature | 712829 | `e2a6d6ebf5f7aa60ab78fa757f7066bacd9cbffb055dab6e16448c042d8a349f` |  |
| `biogrid_5.0.262_scerevisiae_physical.tab3.tsv.gz` | feature | 12912917 | `51fa43118aed4cffc410848da93ff07a37e95c926f8d861625a5c7c11f7af939` |  |

### coxpresdb_spo — COXPRESdb S. pombe expression matrices (Spo-r.c3-0 RNA-seq, Spo-m.c4-0 microarray)

- **use:** `feature` (confirmed)
- **Source:** https://coxpresdb.jp/download/ ; Zenodo record 6861444 (same release)
- **Version:** v8.1 (release 2022-07): Spo-r.c3-0 (v21-10), Spo-m.c4-0 (v21-06); versioned filenames; retrieved 2026-10-28
- **Licence:** CC BY 4.0 (Zenodo record); cite Obayashi et al. NAR 2022 [status: clear]
- **Allowed use:** Expression matrices only; we recompute correlations ourselves. The coex correlation tables (GO/KEGG-scored) and Spo-u union tables are excluded (not downloaded).
- **Notes:** md5 of the Spo-r zip matches COXPRESdb; Spo-m not checked. Open check: confirm no included study is GI-derived (titles screened, none seen).

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `Spo-r.c3-0.expression.combat.txt.zip` | feature | 8534664 | `1db80fe4b29d7017626cbee296b95710e8df12f6ca9f9bd42d120784c964bf7d` |  |
| `Spo-m.c4-0.expression.combat.txt.zip` | feature | 2243827 | `a983f5d7ad4a6f212aa4fc8f217d81ada920fcb0dd35618bdfe83c4705bb4de6` |  |
| `Spo-r.c3-0.sample_metadata.tsv` | feature | 80704 | `ab2635e35feacd7ee08a5517f8bf487c8b34fb8af76910931a1cb2494db71b6e` | study metadata |
| `Spo-m.c4-0.sample_metadata.tsv` | feature | 23863 | `bc2af80da3b588860ad7005219fb2b2c50ef9ea488db9f46f0d3c5060ae348a4` | study metadata |

### ozturk2022_ko_proteome — Öztürk et al. 2022 KO proteomes (Nat Commun 13:6153, doi 10.1038/s41467-022-33814-8)

- **use:** `feature` (confirmed)
- **Source:** https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-33814-8/MediaObjects/41467_2022_33814_MOESM3_ESM.zip (Source Data)
- **Version:** as published 2022 (fixed file); retrieved 2026-10-28
- **Licence:** CC BY 4.0 (open-access article) [status: clear]
- **Allowed use:** KO-vs-WT proteome response vectors from raw intensities (KO-WT). Read zip members with zipfile, never extract blindly. Paper uses GO for interpretation; we use intensities only.
- **Allowlist detail:** `{"archive_members_allowed": ["20220918_SourceData/20220918_SourceData/Data_Figure_1b.txt", ".../Data_Figure_2a.txt", ".../Data_Figure_3b.txt"], "archive_members_other": "all other figure-panel members: not on the allowlist (default deny)"}`
- **Notes:** Note: docs/data_aux_inventory.md cites Nat Commun 13:6232; the correct page is 6153 (Europe PMC).

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `41467_2022_33814_MOESM3_ESM.zip` | feature | 867639064 | `8722f55172e9ffed7152edf56b97e35a834f91ba4d659e5a1d7acbdeb45b1d03` | allowed members only |

### rodriguez_lopez_2023_phenomics — Rodríguez-López et al. 2023 deletion-mutant phenomics (eLife 12:RP88229, doi 10.7554/eLife.88229)

- **use:** `feature` (confirmed)
- **Source:** https://cdn.elifesciences.org/articles/88229/elife-88229-supp1-v1.xlsx (Supplementary file 1)
- **Version:** as published (v1 file); retrieved 2026-10-28
- **Licence:** CC BY 4.0 (eLife) [status: clear]
- **Allowed use:** Raw colony-growth fitness sheets only (129 usable conditions; the key lists 131). GO/KEGG/cluster sheets are not used. Supplementary file 2 (NET-FF GO predictions) was never retained: GO-derived, network inputs may include STRING.
- **Allowlist detail:** `{"sheets_allowed": ["Key to conditions", "Fitness standard conditions", "Fitness all conditions", "Discretized all conditions", "Hits per mutant"], "sheets_excluded": ["GO enrichments", "GO slims", "KEGG", "K-medoids clusters", "Data for clustering", "Description of data (documentation)", "Standard conditions", "Hits per condition", "Oxidative stress"]}`
- **Notes:** Workbook sheets: Description of data, Key to conditions, Standard conditions, Fitness standard conditions, Fitness all conditions, Discretized all conditions, Hits per mutant, Hits per condition, Oxidative stress, Data for clustering, K-medoids clusters, GO enrichments, GO slims, KEGG. Only the five listed are on the allowlist (default deny for the rest).

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `elife-88229-supp1-v1.xlsx` | feature | 39092649 | `e34cde18090839de8af4a85b284a4e3474435d6ba330ae40321fc0552b6add0b` | allowed sheets only |

### kim2010_deletion_library — Kim et al. 2010 Bioneer deletion library (Nat Biotechnol 28:617, doi 10.1038/nbt.1628)

- **use:** `feature` (confirmed)
- **Source:** https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnbt.1628/MediaObjects/41587_2010_BFnbt1628_MOESM{3..7}_ESM.xls
- **Version:** as published 2010 (fixed files); retrieved 2026-10-28
- **Licence:** UNCLEAR: Springer Nature supplementary data, no explicit open licence (copyright NPG). Weakest licence of all datasets. [status: unclear]
- **Allowed use:** Heterozygous-deletion fitness / haploinsufficiency (single-mutant fitness covariate), library gene list, paralogue flag. Licence unclear: never redistribute (including derived pair-level tables).
- **LICENCE UNCLEAR — never redistribute (including derived pair-level tables).**
- **Allowlist detail:** `{"Supplementary Table 1": {"columns_forbidden": ["Published phenotype", "Gene function"]}, "note": "Published phenotype is literature-curated; Gene function is free-text annotation (added here as a precaution: may embed GI-derived statements). Primer/tag sequence columns are not features."}`
- **Notes:** Read with xlrd (installed in .venv, not in requirements.txt).

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `41587_2010_BFnbt1628_MOESM3_ESM.xls` | feature | 2373120 | `f4bdca2b845dafaf35984453d8be5100d743373de7bc7aca23eb3f884acdcb94` | Supplementary Table 1 (library genes, KO%, tags) |
| `41587_2010_BFnbt1628_MOESM4_ESM.xls` | feature | 586240 | `3d010b4325148620c8154961d50af2c75912ee2d85f8595535fa6ba83b7f766e` | Supp Table 11 (orthologues, paralogues, dispensability) |
| `41587_2010_BFnbt1628_MOESM5_ESM.xls` | feature | 1379840 | `7ca9d36bb781a4f2967a5a9cf139a6652a73b5f03216bdc981eeedeead180e77` | Supp Table 15 (heterozygote fitness) |
| `41587_2010_BFnbt1628_MOESM6_ESM.xls` | feature | 266752 | `9652ac8b2dec762b0678b6129ba0af97bb182d15b8a2f720d0c7acdb94fae123` | low-fitness lists (RF<0.97) |
| `41587_2010_BFnbt1628_MOESM7_ESM.xls` | feature | 222208 | `45805b7c836fef1500aeee9ec6a5086ad2a09940aed0dcd8cba8d9efdef2dfa6` | haploinsufficient / haploproficient lists |

### matsuyama2006_localization — Matsuyama et al. 2006 ORFeome YFP localisation (Nat Biotechnol 24:841, doi 10.1038/nbt1222)

- **use:** `feature` (confirmed)
- **Source:** https://dnaconda.riken.jp/DownLoad/ORFeome_Localizome_data.csv (RIKEN BRC DNA Bank; depositor file dated 2007-07-02)
- **Version:** RIKEN BRC file as deposited 2007-07-02 (unversioned URL); retrieved 2026-10-28
- **Licence:** UNCLEAR: no licence stated; RIKEN BRC asks that Matsuyama et al. 2006 is cited. [status: unclear]
- **Allowed use:** Raw localisation calls and nmt1-driven protein level as gene features. Licence unclear: never redistribute (including derived pair-level tables). PomBase GO CC annotations derived from this paper are NOT used.
- **LICENCE UNCLEAR — never redistribute (including derived pair-level tables).**
- **Allowlist detail:** `{"columns": ["Gene Name (holds SP* systematic IDs)", "Protein levels (driven by the nmt1 promoter)", "Localization"], "columns_not_features": "cloning/primer/sequence-status columns"}`

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `ORFeome_Localizome_data.csv` | feature | 1188426 | `2d3fa2eae71dde401433517e755c7533d71d1247f161caac87ccd3ca00034938` |  |

### ensembl_fungi_compara — Ensembl Fungi Compara 116 homologies (rows involving S. pombe only)

- **use:** `feature` (confirmed)
- **Source:** https://ftp.ebi.ac.uk/ensemblgenomes/pub/current/fungi/tsv/ensembl-compara/homologies/Compara.116.protein_default.homologies.tsv.gz (6.6 GiB; filtered while streaming)
- **Version:** Compara 116 (Ensembl Genomes release 63); retrieved 2026-10-28
- **Licence:** Ensembl Genomes data free of restrictions (EMBL-EBI terms); cite Ensembl Fungi [status: clear]
- **Allowed use:** Sequence-based orthology/paralogy (gene-tree homologies), phylogenetic profiles, within-pombe paralogues. Local file is a derived subset (1,163,614 of 273,618,195 rows).
- **ROLLING-URL FLAG:** ROLLING ('current'). A pinned path exists (checked 2026-10-28, not downloaded): https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-63/fungi/tsv/ensembl-compara/homologies/Compara.116.protein_default.homologies.tsv.gz (7,073,684,424 bytes, Last-Modified 2026-02-28). The source md5 (655b68c945e9325a4e5950121d999a2d) was verified against Ensembl MD5SUM during streaming and is stored in SOURCE_MD5.txt. Use the release-63 URL in scripts.

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `Compara.116.pombe_homologies.tsv.gz` | feature | 31012574 | `2db200272f32dbb2e9eccb74d9baf784363b278f38e23ac4b6aad03b1b0effee` | derived subset |
| `README.gene_trees.tsv_dumps.txt` | feature | 3776 | `d133cd293f64db66d2c7b91ad7ab0eccea8545492fa48c79970bbb3f0548bb9e` | documentation |
| `SOURCE_MD5.txt` | feature | 127 | `74a4f41a056d72f62b7abcf7b29694c59dcb9b6a53a709b92aa21e8751cf2c3e` | source md5 record |

### alphafold_spombe — AlphaFold DB S. pombe reference proteome UP000002485 (v6)

- **use:** `feature` (provisional (optional, low priority))
- **Source:** https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v6.tar
- **Version:** AFDB v6 (in file name); remote Last-Modified 2025-08-24, size unchanged on 2026-10-28; retrieved 2026-10-28
- **Licence:** CC BY 4.0 [status: clear]
- **Allowed use:** Predicted-structure features (pLDDT, disorder fraction, optional Foldseek). Not yet unpacked. Unpack with Python tar/zipfile only; treat as data.
- **ROLLING-URL FLAG:** 'latest' directory, but the file name carries v6 and the remote Last-Modified/size are unchanged since 2025-08-24; no release-pinned URL verified.

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `UP000002485_284812_SCHPO_v6.tar` | feature | 840982016 | `e7a3a4dff1a7fdc4b7a8ec0d4e7f556f17a6a6d56d16eb6df3f1bdd851011663` |  |

### uniprot_idmapping_pombe — UniProt ID mapping for S. pombe (taxon 284812)

- **use:** `universe-only` (confirmed)
- **Source:** https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/SCHPO_284812_idmapping.dat.gz
- **Version:** UniProt release 2026_03 (02-Sep-2026); file Last-Modified 2026-09-03; retrieved 2026-10-28
- **Licence:** CC BY 4.0 (UniProt) [status: clear]
- **Allowed use:** ID mapping only (PomBase, GeneID, UniProtKB-ID, UniProtKB-AC rows). The file also contains GO rows: never read them.
- **ROLLING-URL FLAG:** ROLLING ('current_release'). File size (864,651 B) and Last-Modified (2026-09-03) were re-checked on 2026-10-28 and match local. No pinned previous_releases URL exists yet (release-2026_03 path returned 404), so sha256 is the only pin.
- **Allowlist detail:** `{"id_types_allowed": ["PomBase", "GeneID", "UniProtKB-ID", "UniProtKB-AC"], "id_types_forbidden": ["GO (all GO rows)"]}`

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `SCHPO_284812_idmapping.dat.gz` | universe-only | 864651 | `3122478c973f8242566e9c10ae4d1e4946b5b23b4e2467eeed77490cc70804a2` |  |

### ncbi_gene_pombe — NCBI Datasets gene report, S. pombe (taxon 4896), ID fields

- **use:** `universe-only` (confirmed)
- **Source:** https://api.ncbi.nlm.nih.gov/datasets/v2/gene/taxon/4896/dataset_report
- **Version:** API snapshot 2026-10-28 (unversioned); retrieved 2026-10-28
- **Licence:** Public domain (NCBI) [status: clear]
- **Allowed use:** Entrez ID to PomBase ID mapping (to map COXPRESdb). ID fields only; the API response embeds GO terms and the script drops them.
- **ROLLING-URL FLAG:** ROLLING API (no version); the retrieval date and sha256 are the only pin. No pinned alternative.
- **Allowlist detail:** `{"columns": ["entrez_id", "symbol", "type", "pombase_id"], "columns_not_features": ["description"]}`
- **Notes:** On 3,200 shared Entrez IDs NCBI and UniProt disagree for 13; NCBI is used.

| file | use | bytes | sha256 | note |
|---|---|---|---|---|
| `entrez_to_pombase.tsv` | universe-only | 735549 | `b96382447b31a6417c88db8a933221a3faae7604d86ecdf7e44b1146a0f7ffca` |  |

### Notes from the embedding task: sequences and ESM-2 embeddings

**S. pombe proteins** — `data/raw/pombase_protein/peptide.fa.gz` (PomBase release 2026-10-01 /data/ files, retrieved
2026-10-28; checksum in that dir's `SHA256SUMS`: `803b4146…bd3b`; CC BY 4.0, see PomBase entry above). FASTA headers
are `>SPAC1002.01.1:pep name|product`; the gene systematic ID is obtained by dropping `:pep` and the trailing transcript
suffix `.<n>` (5,126 records → 5,126 unique genes, no multi-transcript genes; if one occurred the longest protein is kept).

**S. cerevisiae proteins (SGD)** — `data/raw/sgd_protein/orf_trans_all.fasta.gz`
- Source: https://downloads.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans_all.fasta.gz
  (S288C reference, SGD genome release R64-5-1 as stated in FASTA headers; server Last-Modified 2024-06-13;
  retrieved 2026-10-28; HTTP headers in `headers.txt`).
- sha256: `17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b` (also `data/raw/sgd_protein/SHA256SUMS`).
- Content: 6,722 ORF translations (Verified, Uncharacterized and Dubious ORFs); ID = first header token = systematic
  ORF name (e.g. YAL001C); terminal `*` stripped by the embedding script.
- Licence: SGD data are freely available; SGD states its data may be used without restriction with attribution
  (CC BY 4.0 per SGD's terms page — not independently re-verified here; cite Engel et al., Genetics 2022 / SGD).

**ESM-2 model weights** — `data/raw/esm2_weights/` (git-ignored)
- Source: Meta AI FAIR `fair-esm` release, https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
  (primary model, 33 layers, 650M params, 1280-d) and `esm2_t30_150M_UR50D.pt` (backup, not used); plus the
  `…-contact-regression.pt` files from `/fair-esm/regression/` (needed by the loader, unused for embeddings).
  Retrieved 2026-10-28. Paper: Lin et al., Science 379:1123 (2023). Code: `fair-esm==2.0.0` (PyPI).
- sha256: `esm2_t33_650M_UR50D.pt` `ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c`;
  `esm2_t30_150M_UR50D.pt` `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62` (all in `SHA256SUMS`).
- Licence: MIT (weights and code, github.com/facebookresearch/esm).
- Derived (processed, not committed): `data/processed/esm2_650M_{pombe,cerevisiae}.parquet` (mean-pooled final-layer
  residue embeddings, float32, 1280 cols, index = gene systematic ID), produced by `scripts/compute_esm_embeddings.py`;
  run logs `results/runtime/esm_{pombe,cerevisiae}.json`; checks by `scripts/verify_esm_embeddings.py`
  (`results/verify_esm_embeddings.json`). Proteins >1022 aa use 1022-aa windows with 256 overlap; overlapping residue
  embeddings are averaged before mean pooling. Pretrained-model use is not circular with the GI data (ESM-2 is trained on
  UniRef50 sequences only).
