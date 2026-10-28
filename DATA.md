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
- **Use:** the only source of training/evaluation labels (S < -2.3). Datasets S1/S2 are used.
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

- **Use:** cross-species feature source (ortholog-mapped ε, profiles, SMF). Processed into
  `data/processed/costanzo_pairs.parquet` (11,430,205 unordered ORF pairs; mean/min/max ε over all
  strain-level measurements, min p, sources) and `costanzo_strains.tsv`.

### 3. PomBase reference files (IDs, viability, orthologs)

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

## Auxiliary datasets

(Added separately by the auxiliary-data acquisition; see `scripts/download_aux_data.py`.)
