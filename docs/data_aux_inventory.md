# Auxiliary feature datasets: inventory

Retrieved 2026-10-28 by `scripts/download_aux_data.py` (re-runnable; skips existing files; one directory per
dataset under `data/raw/`, which is git-ignored). Per-dataset `SHA256SUMS` + `RETRIEVED.txt` are written in each
directory and collected in `data/raw/aux_checksums.tsv` (full table in Appendix A).
Coverage numbers come from `scripts/aux_coverage.py` -> `results/aux_coverage.tsv`
(`.venv/bin/python -I scripts/aux_coverage.py`).

Core datasets (Ryan 2012, Costanzo 2016, PomBase IDs/orthologs/viability) are NOT handled here (see DATA.md).

**Coverage conventions.** "Universe" = 5,126 protein-coding S. pombe genes with a PomBase peptide.
"Ryan" = the 2,416 genes in `data/processed/ryan_genes.tsv` (2,400 are protein-coding, in the universe).
All gene IDs below are PomBase systematic IDs (SPAC...) unless stated; no dataset was found to need symbol-based mapping.

Circularity key: **ALLOWED** = raw measurement or sequence-derived, no fission-yeast GI/GO/text mining.
**ALLOWED-with-caveat** = allowed after a specific filter or use restriction (given). **RISKY / EXCLUDED** = see section 3.

---

## 1. Downloaded datasets

### 1.1 `biogrid` - BioGRID physical interactions (S. pombe and S. cerevisiae)
- Source: https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.262/BIOGRID-ORGANISM-5.0.262.tab3.zip (release 5.0.262, file dated 2026-09-29).
- License: MIT (BioGRID terms; free for all use). Cite Oughtred et al.
- Kept files: `biogrid_5.0.262_spombe_physical.tab3.tsv.gz` (713 KB), `biogrid_5.0.262_scerevisiae_physical.tab3.tsv.gz` (12.9 MB). The 189 MB source zip (sha256 in `SOURCE_ZIP_SHA256.txt`) was DELETED after filtering because it contains 62,842 S. pombe *genetic* interaction rows (Negative/Positive Genetic, Synthetic Lethality ... including Ryan 2012). The filter keeps `Experimental System Type == physical` and both interactors in the organism; `Ontology Term*` columns dropped. Script asserts only `physical` rows remain.
- Content: S. pombe 19,416 physical rows (3,700 gene IDs, 14,023 unique non-self pairs); systems: Affinity Capture-MS 8,806, Two-hybrid 3,777, Affinity Capture-Western 2,620, Reconstituted Complex 852, PCA 754, Biochemical Activity 588, Co-purification 489, Affinity Capture-RNA 450, Co-localization 407, Proximity Label-MS 303, others <120. 9,666 rows flagged High Throughput. **Vo et al. 2016 (Cell) systematic Y2H is present: 2,347 rows (2,278 Two-hybrid + 67 PCA + 2 AC-Western), 1,305 genes.** S. cerevisiae: 283,038 physical rows, 6,074 genes.
- Coverage (universe 5,126 / Ryan 2,416): genes with >=1 physical interaction 3,660 / 1,733; with Vo 2016 interaction 1,305 / 634; with >=1 HTP-flagged interaction 3,081 / 1,422.
- IDs: `Systematic Name Interactor A/B` is PomBase systematic ID for S. pombe (100% match `SP[ABC]...`) and SGD systematic ID (YAL001C) for S. cerevisiae. Map S. cerevisiae -> pombe through the PomBase ortholog file (core data).
- Circularity: **ALLOWED-with-caveat.** Literature-curated by humans (not text-mined). Keep physical only (done at download). Recommended additional filter for PPI features: drop `Affinity Capture-RNA`, `Protein-RNA`, `Co-localization` (not protein-protein binding). 2,627 rows have `Source Database = POMBASE` (PomBase-curated physical interactions from papers); these are literature-curated and independent of GI data, but PI may prefer a "BioGRID-only" ablation. Per-paper leakage check: no physical row comes from a GI-screen paper (Ryan 2012, Roguev 2008, Frost 2012 absent from the physical subset).
- Suggested features: per-gene degree (log), HTP-only degree; pair features: direct-edge indicator (physical interaction A-B, also by system: Y2H vs AP-MS), shared-neighbour count / Jaccard / Adamic-Adar over the physical graph, shortest-path length (cap at 4), random-walk-with-restart proximity. Budding yeast: same graph features for orthologous pairs (transfer via 1:1 ortholog map). Beware self-interactions and the AP-MS "spoke vs matrix" expansion (BioGRID is bait-prey: treat as undirected).

### 1.2 `coxpresdb_spo` - fission yeast expression matrices (COXPRESdb v8.1)
- Source: https://coxpresdb.jp/download/ (Spo-r.c3-0 RNA-seq, Spo-m.c4-0 microarray); Zenodo record 6861444 (same release). Release 2022-07; underlying data = public GEO/SRA/ArrayExpress runs, re-processed by COXPRESdb (ComBat batch-corrected, log-scale).
- License: CC BY 4.0 (Zenodo record). Cite Obayashi et al. NAR 2022 (COXPRESdb).
- Files: `Spo-r.c3-0.expression.combat.txt.zip` (8.5 MB; 5,512 genes x 556 runs from 32 SRA studies), `Spo-m.c4-0.expression.combat.txt.zip` (2.2 MB; 4,881 genes x 166 arrays from 9 studies), plus `*.sample_metadata.tsv` (study id, run id, title). md5 of the Spo-r zip matches the md5 published by COXPRESdb.
- IDs: row names are NCBI Entrez Gene IDs -> mapped to PomBase IDs with `ncbi_gene_pombe` (NCBI Datasets; supplemented by UniProt idmapping which agrees on 3,187 / 3,200 entries). Mapped genes in universe: RNA-seq 4,703 (Ryan 2,216); microarray 4,845 (Ryan 2,363).
- Circularity: **ALLOWED.** We use only the expression matrices (raw-derived measurements). We do NOT download the `coex` correlation tables, which carry GO/KEGG-based quality scores. Deletion-mutant and stress studies in the compendium are single-mutant expression, not GI. Check on first use: confirm no study in the sample metadata is a GI-derived dataset (none seen in titles: studies are transcription profiling of mutants/stress/cell cycle).
- Suggested features: gene-by-gene Pearson (or Spearman) and mutual rank across 556 RNA-seq runs (z-scored per gene; also run separately on microarray and take max / mean), top-k neighbour overlap; per-gene mean, variance, max-fold (expression level / variability). Also keep per-gene 556-d vectors as input for learned embeddings (SVD).

### 1.3 `pombase_quant_expression` - absolute RNA and protein abundance (Marguerat 2012, Carpy 2014)
- Source: https://www.pombase.org/data/releases/latest/misc/{htp,full}_gene_expression_table.tsv (PomBase release 2026-10-01).
- License: PomBase data CC BY 4.0 (cite PomBase 2022 and the two primary papers: Marguerat 2012 Cell PMID 23101633; Carpy 2014 MCP PMID 24763107).
- Content: `htp_...` 33,574 rows (gene, study, RNA/protein, condition term, `average_copies_per_cell`; per-replicate copies column is 'ND'); `full_...` 33,661 rows with per-measurement `copies_per_cell`, `range`, `condition`. Marguerat: RNA (proliferating glucose MM; quiescent after N-starvation) and protein (population-wide). Carpy: mass-spec protein copies per cell in G1, S, G2 (M) phases of the cell cycle.
- Coverage universe / Ryan: Marguerat RNA 5,117 / 2,416; Marguerat protein 3,619 / 1,648; Carpy protein 3,175 / 1,435.
- Circularity: **ALLOWED.** The `during(GO:...)` / `during(PBQ:...)` strings are *condition* (cell-cycle phase / quiescence) labels of the experiment, not gene-function annotations; do not use them as GO features, only as condition keys (GO:0072690 = vegetative growth, PBQ:0000003 = quiescence, GO:0000080/84/87/90 = G1/S/G2/M phases).
- Suggested features: log10 mRNA and protein copies per cell (proliferating), quiescent/proliferating RNA ratio, protein-to-mRNA ratio (translation efficiency proxy), Carpy cell-cycle protein profile (4-vector) and per-pair Pearson; per-pair |difference| and product of abundances (abundance-matching; abundance is a known confounder of GI detection power).

### 1.4 `ozturk2022_ko_proteome` - proteome of 3,308 single-gene knockout strains (Ozturk 2022)
- Source: Nat Commun 13:6232 (doi 10.1038/s41467-022-33814-8), Source Data zip `41467_2022_33814_MOESM3_ESM.zip` (868 MB) from static-content.springer.com. Raw data also in ProteomeXchange PXD024332 (not downloaded; far larger).
- License: CC BY 4.0 (open access article).
- Key member: `Data_Figure_1b.txt` (710 MB, 11,013,018 long-format rows: `proteinGroup, Strain, log2(intensity), strainType`); `Data_Figure_2a.txt` (per-strain n proteins identified/regulated); `Data_Figure_3b.txt` (z-score/FDR/FC for a subset); other members are figure panels. The zip is kept compressed; read members with `zipfile` (do not extract blindly).
- Coverage: 3,769 distinct strains (3,301 KO genes in universe; Ryan 2,179), 2,922 protein groups measured (2,880 in universe; Ryan 1,286). Strain labels look like `Sp01A02-SPAC1002.20` (plate position - KO gene id; wild-type / control strains also present).
- Circularity: **ALLOWED** (raw single-mutant proteomics). Note the paper itself used GO for interpretation; we use only intensities.
- Suggested features: for each KO gene, a proteome-response vector (z-score vs WT distribution across proteins); pair feature = Pearson correlation of KO-response vectors (functionally related genes give similar proteome shifts), number of significantly regulated proteins (global perturbation size), whether KO of A changes abundance of B (a directed, dosage-style feature: protein B level in strain KO-A), per-gene "proteome sensitivity" (how often protein B changes when other genes are deleted; ~"hub-ness").

### 1.5 `rodriguez_lopez_2023_phenomics` - deletion-mutant fitness over 131 conditions
- Source: eLife 2023;12:RP88229 (doi 10.7554/eLife.88229), Supplementary file 1 `elife-88229-supp1-v1.xlsx` (https://cdn.elifesciences.org/articles/88229/elife-88229-supp1-v1.xlsx, 39 MB). License CC BY 4.0.
- Sheets used: `Fitness all conditions` (450,843 rows: gene, condition, fitness mean/median, effect size, p, FDR), `Fitness standard conditions`, `Discretized all conditions` (3,509 genes x 129 conditions; -1/0/+1 sensitive/none/resistant), `Hits per mutant`, `Key to conditions` (131 conditions: media, drug, dose, unit, temperature).
- Coverage: 3,509 deletion mutants (3,499 in universe; Ryan 2,162).
- Circularity: **ALLOWED for the phenomics sheets** (colony-growth fitness of single haploid deletion mutants; non-essential genes only). **IGNORE** sheets `GO enrichments`, `GO slims`, `KEGG` and the K-medoid / Cytoscape cluster sheets are fine as derived-from-phenotypes only but offer nothing beyond the raw data (and GO-enriched labels must not be used). **Supplementary file 2 (NET-FF GO predictions, GOGO similarity) was deliberately NOT retained** (GO-derived, and NET-FF integrates STRING-type networks); the downloaded copy was deleted.
- Suggested features: per-pair Pearson/cosine across 129 conditions on the fitness (or effect size) matrix; discretized agreement (shared sensitive conditions, Jaccard over sensitive sets); per-gene n_sensitive, mean fitness in YES_32 (a gene with a strong single-mutant fitness defect has a bounded GI range - important covariate); condition-specific sensitivity to DNA damage / drugs.

### 1.6 `kim2010_deletion_library` - Bioneer deletion library characterisation (Kim 2010)
- Source: Nat Biotechnol 28:617 (doi 10.1038/nbt.1628) Supplementary Tables 1, 11, 15 and the haploinsufficiency lists (MOESM3-7, `.xls`, legacy Excel; read with `xlrd`, installed in `.venv` but not in `requirements.txt` - add if used). Raw CEL files (MOESM12) and primer tables were not retained.
- License: Springer Nature supplementary data, no explicit open license (copyright NPG). Used for research under fair use; do not redistribute (data/ is git-ignored). **License is the weakest of all datasets here.**
- Content: S-Table 1 (4,914 library genes, KO%, tags); S-Table 11 (2,436 genes with S. cerevisiae ortholog, **paralogue column**, Sp|Sc copy-number relationship, dispensability in each yeast); S-Table 15 (4,287 genes: relative growth fitness of pooled heterozygous deletion strains by tag microarray, incl. essential genes; P-values, haploinsufficiency flags); MOESM6/7 lists of 138 haploinsufficient / 136 haploproficient fission yeast genes.
- Coverage universe / Ryan: library 4,906 / 2,379; fitness table 4,287 / 2,202; ortholog table 2,436 / 1,023.
- Circularity: **ALLOWED-with-caveat.** Raw HTP fitness is fine. Do NOT use the `Published phenotype` column (literature-curated) in S-Table 1. The dispensability columns in S-Table 11 duplicate viability (core data).
- Suggested features: heterozygous-diploid fitness (dosage sensitivity / haploinsufficiency flag), per-gene fitness as single-mutant growth covariate; "has paralogue" flag from S-Table 11 (coarse - prefer the Ensembl Compara / domain-based paralogue definitions below).

### 1.7 `matsuyama2006_localization` - ORFeome YFP localisation (Matsuyama 2006)
- Source: RIKEN BRC DNA Bank, `ORFeome_Localizome_data.csv` (https://dnaconda.riken.jp/DownLoad/ORFeome_Localizome_data.csv; provided by the depositor 2007-07-02; derived from Nat Biotechnol 24:841 Supp. Table 6).
- License: none stated beyond RIKEN BRC "must cite Matsuyama et al. 2006"; PI should treat as research-use, cite, do not redistribute (data/ git-ignored).
- Content: 4,954 rows (one per cloned ORF); columns incl. systematic ID (`Gene Name` column holds SPxx IDs), nmt1-driven protein level, and `Localization` free-text with a compact grammar: `>`/`>>` ranks compartments (`nucleus>cytosol`), `=` ties (`cytosol=nucleus`), plus `no apparent signal`, `cytoplasmic dots`, `periphery`, `nucleolus>nucleus`, ... 
- Coverage: 4,691 genes with a localisation call in universe (Ryan 2,276).
- Circularity: **ALLOWED** (raw microscopy calls; the PomBase GO CC annotations derived from this paper are *not* used).
- Suggested features: parse into a compartment x gene multi-hot (nucleus, cytosol, mitochondrion, ER, nucleolus, periphery, cytoplasmic dots, ...) using the dominant (first-ranked) plus tied compartments; pair features = shared dominant compartment, Jaccard of compartment sets; per-gene nmt1 expression level.

### 1.8 `pombase_protein` - sequences and sequence-computed domain calls
- Source: PomBase /data (rolling; files dated 2026-10-28): `genome_sequence_and_features/feature_sequences/peptide.fa.gz` (5,126 protein-coding genes; one peptide per gene id after stripping `.1:pep`), `Protein_data/protein_domains_and_features.tsv` (InterPro-consortium member-database matches computed on sequence: Pfam, PANTHER, CATH-Gene3D, SUPERFAMILY, SMART, PROSITE, CDD, PRINTS, DeepTMHMM, MobiDB, coiled coils ...), `PeptideStats.tsv`, `aa_composition.tsv`, `transmembrane_domain_coords_and_seqs.tsv`.
- License: CC BY 4.0 (PomBase).
- Coverage: 5,126 peptides; 4,925 genes with any domain/feature call (Ryan 2,331); Pfam 4,603 (Ryan 2,173) with 3,812 distinct Pfam IDs; PANTHER 4,489 (Ryan 2,107).
- Circularity: **ALLOWED.** Domain/family calls are HMM/profile matches on sequence. The InterPro-to-GO mapping is *not* used and must not be. (The file contains no GO columns; verified.)
- Suggested features: shared-Pfam count and IDF-weighted Jaccard between A and B; same PANTHER family/subfamily (coarse paralogue/family flag); Pfam bag-of-domains SVD embedding; per-gene length, #TM helices, disorder fraction (MobiDB), amino-acid composition; pairwise sequence identity from an all-vs-all alignment (to compute later with e.g. `parasail`/DIAMOND; not yet run) as the paralogue feature (paralogue pairs are the strongest known driver of SL).

### 1.9 `ensembl_fungi_compara` - sequence-based orthology / paralogy across fungi (phylogenetic profiles)
- Source: Ensembl Fungi Compara 116 gene-tree homologies, https://ftp.ebi.ac.uk/ensemblgenomes/pub/current/fungi/tsv/ensembl-compara/homologies/Compara.116.protein_default.homologies.tsv.gz (6.6 GiB; md5 655b68c9...; verified while streaming). Only rows involving *schizosaccharomyces_pombe* were kept (`Compara.116.pombe_homologies.tsv.gz`).
- License: Ensembl Genomes data free of restrictions (EMBL-EBI terms); cite Ensembl Fungi.
- Content (filtered file, 31 MB, 1,163,614 rows kept of 273,618,195; columns as in `README.gene_trees.tsv_dumps.txt`, includes % identity both directions, `is_high_confidence`): ortholog_one2one 755,185; one2many 297,314; many2many 105,397; within-pombe paralogue pairs 5,718 (1,434 `within_species_paralog` + 4,284 `other_paralog`, all with both genes in S. pombe). 367 partner fungal genomes (Ascomycota, Basidiomycota, early-diverging fungi); 3,600 rows to S. cerevisiae.
- Coverage (universe / Ryan): genes with >=1 ortholog in another fungus 4,833 / 2,287 (3,622 genes have orthologs in >=50 genomes, 2,913 in >=200); genes in a within-pombe paralogue pair 2,452 / 1,203.
- Circularity: **ALLOWED.** Homologies are inferred from gene trees built on protein sequence (HMM-based); no GO or text mining involved. Note: includes within-pombe paralogues (`within_species_paralog`).
- Suggested features: (a) paralogue flag/identity (`within_species_paralog` pairs, with % identity and `is_high_confidence`), number of paralogues per gene; (b) phylogenetic profile = presence/absence (or best identity) of an ortholog in each fungal genome -> per-gene vector; pair feature = profile correlation/Hamming (co-evolution), plus profile length = conservation breadth; (c) Sc ortholog IDs (cross-check against PomBase ortholog file in core data; **prefer the PomBase file as primary** for S. cerevisiae mapping).

### 1.10 `alphafold_spombe` - predicted structures
- Source: AlphaFold DB S. pombe reference proteome UP000002485, https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v6.tar (841 MB). License CC BY 4.0.
- Content: per-UniProt-accession PDB/mmCIF + PAE JSON (not yet unpacked or parsed). UniProt -> PomBase ID via `uniprot_idmapping_pombe` (5,130 accessions with PomBase ID).
- Circularity: **ALLOWED.** Optional / low priority.
- Suggested features: mean pLDDT, fraction disordered (pLDDT<50), radius of gyration, optional Foldseek structural-similarity between pairs (paralogue/fold-sharing proxy). Unpack with Python tar/`zipfile` only; treat as data.

### 1.11 ID-mapping helpers (no biological signal)
- `uniprot_idmapping_pombe`: `SCHPO_284812_idmapping.dat.gz` from UniProt (release as of 2026-09-03). CC BY 4.0. Contains a GO column among many ID types: **use only the `PomBase`, `GeneID`, `UniProtKB-ID` rows**.
- `ncbi_gene_pombe`: `entrez_to_pombase.tsv` (Entrez ID, symbol, type, PomBase ID, description) from the NCBI Datasets gene API, taxon 4896 (12,645 genes, 12,403 with PomBase IDs; 5,106 protein-coding in universe). Public domain. The API response also embeds GO terms; the script writes only ID fields. Used to map COXPRESdb. On the 3,200 Entrez IDs in both sources, NCBI and UniProt agree for 3,187; the 13 conflicts are resolved in favour of NCBI (check if COXPRESdb genes in conflict matter).

---

## 2. Coverage summary (Ryan genes = 2,416)

| dataset | feature type | genes (universe 5,126) | Ryan genes covered |
|---|---|---|---|
| pombase_protein | sequence / domains | 5,126 / 4,925 any domain | 2,400 / 2,331 |
| Marguerat RNA | abundance | 5,117 | 2,416 |
| Marguerat protein | abundance | 3,619 | 1,648 |
| Carpy protein | abundance by cell-cycle phase | 3,175 | 1,435 |
| COXPRESdb RNA-seq | co-expression (556 runs) | 4,703 | 2,216 |
| COXPRESdb microarray | co-expression (166 arrays) | 4,845 | 2,363 |
| BioGRID physical (pombe) | PPI | 3,660 | 1,733 |
| Vo 2016 Y2H (within BioGRID) | PPI (systematic) | 1,305 | 634 |
| Matsuyama localisation | localisation | 4,691 | 2,276 |
| Rodriguez-Lopez phenomics | single-mutant fitness, 129 cond. | 3,499 | 2,162 |
| Kim 2010 het-diploid fitness | fitness | 4,287 | 2,202 |
| Ozturk KO proteome | KO-response vectors | 3,301 KO genes | 2,179 |
| Ensembl Compara orthologs (367 fungal genomes) | phylogenetic profile | 4,833 | 2,287 |
| Ensembl Compara within-pombe paralogues | paralogy | 2,452 | 1,203 |

---

## 3. Rejected, excluded, deferred

**Excluded for circularity (rules):**
- STRING / any text-mining-derived network; Gene Ontology (GAF, GO slim files `bp/cc_goslim_pombe*`, GOCAM nodes); PomBase `Complex_annotation.tsv` (GO-derived complexes); PomBase `pombase-go-physical-interactions` and `-substrates` (GO-evidence derived).
- PomBase `interactions.tsv` / `interactions_with_comments.tsv` (contain curated genetic interactions) and the BioGRID *genetic* rows (deleted).
- Rodriguez-Lopez 2023 Supplementary file 2 (NET-FF GO predictions; GOGO semantic similarity) - GO-derived; and the NET-FF paper's STRING-type inputs.
- COXPRESdb `coex` tables (GO/KEGG scored; we recompute correlations ourselves) and the `Spo-u` union tables (redundant with the expression matrices).

**RISKY (PI judgement needed):**
- **PomBase FYPO phenotype annotations** (`phenotype_annotations.pombase.phaf.gz`, `canto_fypo_annotations_with_comments`, `public_api_genotypes_phenotypes.parquet`): ontology-annotated, curated from papers that include double-mutant / GI studies; cannot cleanly separate single-mutant HTP from GI-derived annotations -> not downloaded. Raw HTP single-mutant sets (Rodriguez-Lopez, Kim, Ozturk) are used instead. (FYPO viability is core data owned by the data agent.)
- BioGRID physical rows with `Source Database = POMBASE` or from small-scale papers (see 1.1) - recommended to keep, but offer a BioGRID-HTP-only ablation.
- Kim 2010 and Matsuyama 2006 licensing (not open-access); Kim `Published phenotype` column.
- COXPRESdb studies: re-processed and ComBat/PCA-corrected; we should spot-check that no pombe GI-derived data are among the 41 studies (titles screened, none seen).

**Rejected as impractical / redundant / not found:**
- Vo 2016 Cell supplementary tables from the publisher: not fetched; BioGRID already contains the Y2H/PCA edges (2,347 rows from the paper). The Y2H scoring/confidence column in the supplement is not retained (BioGRID has no score).
- Chen 2003 stress, Rustici 2004 cell-cycle, Mata 2002 meiosis microarrays: only in heterogeneous Sanger FTP archives (`ftp.sanger.ac.uk/pub/postgenomics/s_pombe`) / ArrayExpress E-MEXP-54..64 with no uniform gene IDs; the GEO/ArrayExpress-derived compendium in COXPRESdb already integrates the GEO-hosted pombe studies. Individual GEO series (e.g. GSE4283/4284 HU time courses, GSE47544 kinase-deletion N-starvation, 148 arrays) were screened and not downloaded (partly S. cerevisiae series matched by search; heterogeneous array platforms).
- Marguerat 2012 raw BAM/bigWig (tens of GB) and Mata 2013 / Wilhelm 2008 tracks in PomBase `external_datasets/`: not needed; quantified values are in `pombase_quant_expression`.
- Rhind 2011 / Jeffares 2015 (natural-isolate SNP/indel VCFs, 73 MB; could yield per-gene diversity, dN/dS): deferred; optional idea.
- ProtT5 per-protein embeddings (UniProt `embeddings/uniprot_sprot/per-protein.h5`, 1.3 GB): covers only the Swiss-Prot-reviewed subset of S. pombe proteins (S. pombe has no dedicated embeddings proteome there), and HDF5 reader `h5py` is not in `requirements.txt`. Deferred; prefer computing ESM-2 embeddings locally from `peptide.fa.gz` (torch is available) if protein-language-model features are wanted.
- eggNOG / OrthoDB / OMA orthology: large; eggNOG annotations carry GO/text-derived functional descriptions. Ensembl Compara (above) is sequence-only and sufficient for presence/absence profiles. Also available in PomBase `orthologs/` (core-data owner): pombe-japonicus orthologs (`pombe-japonicus-orthologs*.tsv`) - a cheap, GO-free conservation feature (S. japonicus) worth requesting from the core-data owner.
- COXPRESdb S. cerevisiae (`Sce-u`) co-expression and SPELL/yMGV: not fetched; S. cerevisiae expression only matters for ortholog-transfer features, optional later.
- Hayles 2013 cell cycle/size screen and Carpy proteome supplements: data exist only as FYPO annotations (risky) / already in the PomBase quantitative table.

---

## 4. Verification notes
- `scripts/download_aux_data.py` re-run is idempotent (tested: second run skips all existing files).
- BioGRID filter asserted: only `physical` rows kept; no S. pombe genetic row remains on disk (checked by re-reading the output: 19,416 rows, systems listed above; raw zip and unfiltered text deleted).
- COXPRESdb md5 of the Spo-r.c3-0 zip equals the md5 published by COXPRESdb (Spo-m not checked).
- Compara stream md5 compared with Ensembl MD5SUM; mismatch aborts and removes output.
- Not yet done (feature-building phase): confirm ID mappings do not drop Ryan genes silently (n_in_ryan_genes column in `results/aux_coverage.tsv` is the baseline), screen COXPRESdb study titles, parse Matsuyama localisation grammar, check Vo 2016 edge direction (bait->prey) handling.

---

## Appendix A. File checksums (sha256)
(generated from `data/raw/aux_checksums.tsv`; the BioGRID source zip sha256 is `096a9e4d871ad5d698bd503b2b9a2409297b88f5b55bc479c710f0c700429820`, Compara source md5 `655b68c945e9325a4e5950121d999a2d`)


| dataset | file | bytes | sha256 |
|---|---|---|---|
| pombase_protein | PeptideStats.tsv | 194,436 | `62c6fb25558c4bf3e3a4746e60f479cec58413b47eb6c333b6f1b822ca704426` |
| pombase_protein | aa_composition.tsv | 340,748 | `a52b79d80569ad72f19918702ed4516b5dc5bef95905a4363d219d72d0d68e13` |
| pombase_protein | peptide.fa.gz | 1,620,552 | `803b4146fe270338b740459cf26567b24f9d3dae7d9484203d27e5350108f19b` |
| pombase_protein | protein_domains_and_features.tsv | 4,167,771 | `139a4f97d9e5f2fd0705cd9bd9d50d1779c0f6f6d49df308ef14a135c623bd3b` |
| pombase_protein | transmembrane_domain_coords_and_seqs.tsv | 576,268 | `a747e6736a2378171326b39317a99c1cf868a6b79a156923be7563d3bbaccd6e` |
| pombase_quant_expression | full_gene_expression_table.tsv | 5,181,827 | `0bb00425e7fa9121f6b8513451d13ed4c77e03c7402d864a4960ac6313ebecf1` |
| pombase_quant_expression | htp_gene_expression_table.tsv | 4,137,911 | `bc512e72e636f5c8600a4f589d272dbc1d873d86283cd83c90858ff6453901fa` |
| biogrid | biogrid_5.0.262_scerevisiae_physical.tab3.tsv.gz | 12,912,917 | `51fa43118aed4cffc410848da93ff07a37e95c926f8d861625a5c7c11f7af939` |
| biogrid | biogrid_5.0.262_spombe_physical.tab3.tsv.gz | 712,829 | `e2a6d6ebf5f7aa60ab78fa757f7066bacd9cbffb055dab6e16448c042d8a349f` |
| coxpresdb_spo | Spo-m.c4-0.expression.combat.txt.zip | 2,243,827 | `a983f5d7ad4a6f212aa4fc8f217d81ada920fcb0dd35618bdfe83c4705bb4de6` |
| coxpresdb_spo | Spo-m.c4-0.sample_metadata.tsv | 23,863 | `bc2af80da3b588860ad7005219fb2b2c50ef9ea488db9f46f0d3c5060ae348a4` |
| coxpresdb_spo | Spo-r.c3-0.expression.combat.txt.zip | 8,534,664 | `1db80fe4b29d7017626cbee296b95710e8df12f6ca9f9bd42d120784c964bf7d` |
| coxpresdb_spo | Spo-r.c3-0.sample_metadata.tsv | 80,704 | `ab2635e35feacd7ee08a5517f8bf487c8b34fb8af76910931a1cb2494db71b6e` |
| ozturk2022_ko_proteome | 41467_2022_33814_MOESM3_ESM.zip | 867,639,064 | `8722f55172e9ffed7152edf56b97e35a834f91ba4d659e5a1d7acbdeb45b1d03` |
| rodriguez_lopez_2023_phenomics | elife-88229-supp1-v1.xlsx | 39,092,649 | `e34cde18090839de8af4a85b284a4e3474435d6ba330ae40321fc0552b6add0b` |
| kim2010_deletion_library | 41587_2010_BFnbt1628_MOESM3_ESM.xls | 2,373,120 | `f4bdca2b845dafaf35984453d8be5100d743373de7bc7aca23eb3f884acdcb94` |
| kim2010_deletion_library | 41587_2010_BFnbt1628_MOESM4_ESM.xls | 586,240 | `3d010b4325148620c8154961d50af2c75912ee2d85f8595535fa6ba83b7f766e` |
| kim2010_deletion_library | 41587_2010_BFnbt1628_MOESM5_ESM.xls | 1,379,840 | `7ca9d36bb781a4f2967a5a9cf139a6652a73b5f03216bdc981eeedeead180e77` |
| kim2010_deletion_library | 41587_2010_BFnbt1628_MOESM6_ESM.xls | 266,752 | `9652ac8b2dec762b0678b6129ba0af97bb182d15b8a2f720d0c7acdb94fae123` |
| kim2010_deletion_library | 41587_2010_BFnbt1628_MOESM7_ESM.xls | 222,208 | `45805b7c836fef1500aeee9ec6a5086ad2a09940aed0dcd8cba8d9efdef2dfa6` |
| matsuyama2006_localization | ORFeome_Localizome_data.csv | 1,188,426 | `2d3fa2eae71dde401433517e755c7533d71d1247f161caac87ccd3ca00034938` |
| uniprot_idmapping_pombe | SCHPO_284812_idmapping.dat.gz | 864,651 | `3122478c973f8242566e9c10ae4d1e4946b5b23b4e2467eeed77490cc70804a2` |
| ncbi_gene_pombe | entrez_to_pombase.tsv | 735,549 | `b96382447b31a6417c88db8a933221a3faae7604d86ecdf7e44b1146a0f7ffca` |
| alphafold_spombe | UP000002485_284812_SCHPO_v6.tar | 840,982,016 | `e7a3a4dff1a7fdc4b7a8ec0d4e7f556f17a6a6d56d16eb6df3f1bdd851011663` |
| ensembl_fungi_compara | Compara.116.pombe_homologies.tsv.gz | 31,012,574 | `2db200272f32dbb2e9eccb74d9baf784363b278f38e23ac4b6aad03b1b0effee` |
| ensembl_fungi_compara | README.gene_trees.tsv_dumps.txt | 3,776 | `d133cd293f64db66d2c7b91ad7ab0eccea8545492fa48c79970bbb3f0548bb9e` |
| ensembl_fungi_compara | SOURCE_MD5.txt | 127 | `74a4f41a056d72f62b7abcf7b29694c59dcb9b6a53a709b92aa21e8751cf2c3e` |
