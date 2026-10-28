# DESIGN.md — Predicting synthetic-lethal (strong negative) genetic interactions in *S. pombe*

Version: **v3.1** (after review rounds 1–3; round-3 fixes in Appendix C) · 2026-10-28 · Owner: PI agent
Inputs: PROMPT.md (fixed goal), LITERATURE.md, DATA.md, docs/data_inventory.md, docs/data_aux_inventory.md, reviews/round1_*.md
Change log vs v1 and the disposition of every reviewer proposal: Appendix A.

## 0. Fixed goal (from PROMPT.md, not negotiable)
- Tool that scores **every pair of non-essential S. pombe genes** for a strong negative genetic interaction.
- Training data: Ryan et al. 2012 E-MAP. Positive = **S-score < −2.3**. Metric: **AUPRC**.
- Must work for genes **unseen or rarely seen** in Ryan → splits must reflect that.
- Allowed: Costanzo 2016 (fully) and other non-circular data. Forbidden: any other fission-yeast GI data; text mining / text-mining-derived resources (STRING …); Gene Ontology in any form.
- Report runtime alongside accuracy.

**Protocol freeze.** Every value that defines the evaluation — label column and threshold, universe/eval masks, split seed and fractions, class weights, seeds list, bootstrap settings, adoption rule — lives in `config/protocol.yaml`. Its sha256 is recorded in RESEARCH_TREE.md when frozen (end of Phase A). Code reads these values only from the config; a lint test fails if the threshold or weights are typed as literals elsewhere in `src/` or `scripts/`. Changing the frozen protocol requires a decision-log entry, and results under the old protocol stay reported.

## 1. Labels, universe, evaluation set
- **Label:** Ryan Dataset S2 (authors' per-pair averaged S; 1,337,033 pairs). `y = 1[S2 < −2.3]`. Rationale: the brief speaks of one quantitative value per pair, and S2 is the authors' canonical per-pair matrix. **Sensitivity (report-only):** the final shortlist is also scored against `raw_mean < −2.3`.
- **Training targets (experiment E-TARGET, evaluation always on the fixed label):** S2 binary (default); S2 continuous (Huber, clipped); raw-mean binary; soft/multi-threshold.
- **Prediction universe (deliverable):** protein-coding genes that are not essential, i.e. PomBase deletion viability ∈ {`viable`, `condition-dependent`} (3,757 genes), plus the 22 unknown-viability genes that have a main (deletion) strain in Ryan (sap10 excluded: DAmP-only).
  - The other ≈143 unknown-viability genes are scored in the deliverable but flagged `viability_unknown`; they are not part of the core universe.
  - The 12 PomBase-inviable genes with a Ryan "deletion" array strain are dropped from training by default (likely wrong or suppressed strains).
  - FYPO viability (`FYPOviability.tsv`) is a single-mutant phenotype. It is used **only** to define the universe and the evaluation masks and for stratification, **never as a feature**.
- **Primary evaluation pairs:** S2 pairs where both genes are in the universe. Exclusions, determined per pair and orientation by script:
  - **pairs** (not genes) whose S2 derives only from measurements involving a DAmP/hypomorph allele of either gene (16,557 universe pairs; verified independently);
  - pairs involving merged Ryan IDs (train-only).

  Mixed DAmP+deletion pairs (≈17k) are kept, with a report-only sensitivity check that excludes them. **Secondary:** viable×viable only.
- **Training pairs:** all S2 pairs except those involving the 12 PomBase-inviable genes with a Ryan "deletion" array strain; inviable/DAmP genes are otherwise included. Experiment E-ALLELE (default vs universe-only training) tests this.
- **Hypomorph indicator:** `hypomorph_a/b` = 1 if the gene's Ryan data come from a DAmP/hypomorph allele, else 0. It is the one permitted exception to §4 rule 1: it describes the allele the label was measured on, not the screen layout. It is always 0 on eval rows and in the deliverable, and a test checks this. It stops missing deletion-based features from acting as an essentiality proxy.
- Pseudogenes, ncRNAs and unmapped IDs are dropped.
- **Linkage:** E-MAP filters linked pairs. A diagnostic plots hit rate vs chromosomal distance, using PomBase coordinates. Distance is **never** a deliverable feature. Universe pairs within the linkage window are flagged "not E-MAP-testable" in the deliverable.

## 2. Splits (gene-disjoint; lockbox physically separated)
Classes (Park & Marcotte 2012): **C1** both genes seen in training, **C2** exactly one seen, **C3** neither seen.
- **Lockbox genes (DRAWN 2026-10-28):** 20% of Ryan S2 genes (480/2,404), split seed `20261028`, via `scripts/draw_lockbox.py`. Stored in `config/lockbox_genes.txt` (sha256 77fb02e1…). Stratified by Ryan role (query / array / both) × within-role hit-rate quintile, with the top 2% of hubs as their own stratum. Expected viable×viable sizes: C2 ≈ 331k pairs / 6.5k positives; C3 ≈ 41k pairs / ≈805 ± 53 positives, from ≈141 query genes. A script reports the actual counts.
- **Lockbox C1 pairs:** 10% of dev×dev pairs (random, label-stratified).
- **Physical separation:**
  - `scripts/make_splits.py` writes `labels_dev.parquet` and `labels_lockbox.parquet`.
  - A lint test blocks reads of **every** full-label file outside the data-build scripts: `ryan_pairs.parquet`, `ryan_measurements.parquet`, `results/eda/*` and `docs/data_inventory.md`.
  - The only way models get labels is the fold-scoped API `src/data.fold_labels(split, fold, part)`. Feature caches are keyed by (split, fold, label-file hash).
  - Only `scripts/lockbox_eval.py` reads the lockbox. It requires a committed shortlist file and hashes of the prediction files, appends to `results/lockbox_access.log`, and refuses a second run without a `--reason`.
  - A test asserts that no lockbox C1 pair is in the dev labels file.
- **Development CV:** dev genes are split into 5 gene folds with the same stratification. For fold k: C3_k = pairs with both genes in fold k; C2_k = exactly one gene in fold k; C1_k = a 10% random pair hold-out among the other folds' genes. Training = the remaining pairs with neither gene in fold k. Early stopping and hyperparameter selection always use an **inner** split of the training genes, never the reported fold.
- **Split-sensitivity:** an alternative seed (`7`) **re-partitions the dev folds only**; the lockbox never changes.
- **"Rarely seen" (E-FEWSHOT):** for held-out genes, reveal k ∈ {5, 20, 100} labelled pairs in training and evaluate on the rest, bootstrapped by gene.
- **Distribution-shift diagnostics:**
  - **(a) C1-arr:** mask the query-role rows of a random subset of query+array genes. Evaluate pairs among them, which mimics the array×array pairs that make up 81% of unmeasured C1 universe pairs.
  - **(b)** C2 stratified by the seen partner's role, plus a version reweighted to the universe mix (33% query).
  - **(c)** Covariate shift: a classifier on gene features separating Ryan genes from never-measured universe genes. If AUROC > 0.65, report importance-weighted C2/C3 (report-only).
  - **(d) Homology:** C2/C3 stratified by maximum sequence identity to training genes. One sensitivity split keeps paralog families in the same fold.
- Final models are retrained on all dev pairs for the lockbox evaluation. The deliverable model is then retrained on all Ryan pairs.

## 3. Metrics and statistics
- **Primary selection metric:** the **pooled, universe-weighted AUPRC**. Within each dev fold, C1_k, C2_k and C3_k predictions are pooled into one ranking. `average_precision_score` is computed with sample weights that scale each class's total weight to the fractions of **unmeasured universe pairs** (≈0.25/0.57/0.18, computed by script and frozen in protocol.yaml). The score is averaged over the 5 folds. Why pooled: the deliverable is one ranked list, so scores must be comparable across classes.
- **C1 composition:** the C1 term consists of query×array pairs, the only C1 pairs Ryan measures. C1-arr (simulated array×array; §2) is reported next to the headline.
- **Secondary:** per-class AUPRC; the weighted mean of per-class AUPRCs; the pooled score under all-pairs weights (0.37/0.48/0.15) and equal weights. Decisions whose sign flips across weightings are flagged. Also lift (AUPRC/prevalence), AUROC, precision@1k, and C1 split into reciprocal vs single-orientation pairs. Per-gene macro AUPRC is a diagnostic only.
- **Uncertainty (one shared gene bootstrap):**
  - Each of 1,000 replicates draws one Poisson(1) count per gene, shared across all classes and folds.
  - Pair weight = count_a × count_b × class weight. Class totals are rescaled to the frozen weights within each fold.
  - The replicate statistic is the fold mean of the pooled AUPRC, i.e. exactly the point statistic.
  - Model comparisons use the **same** replicates (paired) and seed-averaged predictions.
  - The procedure is validated on the toy data.
- **Adoption rule:** a change is adopted only if the paired 95% CI on Δ(primary) excludes 0 **and** Δ ≥ 0.005.
  - Main ladder steps must also show Δ > 0 on the seed-7 dev partition.
  - If the paired CI under the all-pairs or equal weighting **excludes 0 in the opposite direction**, keep the simpler model.
  - Feature-group ablations are run one group at a time.
  - Every comparison goes into the registry, adopted or not.
  - Expect about one false adoption per 30–40 null comparisons; the lockbox is the guard against this.
- **Tuning fairness and caps:**
  - Shortlist candidates get the same random-search budget (≤30 configs), tuned on inner gene folds.
  - Ablations reuse the parent model's configuration.
  - Ladder runs use seeds 0–2; the shortlist uses all 5 seeds.
- **Lockbox:**
  - Pre-registered in `config/lockbox_prereg.yaml` (committed before opening):
    - the bootstrap p-value definition and Holm correction;
    - how boosting rounds are fixed for the all-dev retrain;
    - that the headline model is reported and shipped whatever the lockbox ranking shows;
    - the shortlist: one headline model, ≤4 others, all baselines, and the contrasts to test.
  - Reported alongside: the dev-minus-lockbox gap and per-class CIs.
- **Noise reference (not a ceiling):** among reciprocal pairs, predict orientation B's single-orientation label (raw < −3.76) from orientation A's raw score. AP ≈ 0.35 (lift ≈16) is a lower bound on achievable replicate agreement.

## 4. Features
All features are computed for every universe gene. Pair features are symmetric in (a, b). Each group below is one ablation unit.

**Sources and the features derived from them**

| Group | Source | Gene-level features | Pair-level features |
|---|---|---|---|
| SMF/PHEN | Rodríguez-López 2023 (standard-condition fitness + 129 conditions); Kim 2010 (heterozygous-diploid fitness, labelled as such) | haploid SMF; phenotype-vector PCA; # conditions with phenotype | min/max/product SMF; phenotype-profile correlation |
| CER | Costanzo 2016 via PomBase curated orthologs | ortholog degree as a **rate**, separately for NxN and ExN/ExE; ortholog SMF; any-ortholog-essential (Costanzo strain class); n-orthologs; has-ortholog; **Costanzo-profile embedding** (PCA of the ortholog's GI profile) | ε aggregated after collapsing alleles: mean, min, k-corrected min, n_pairs; ohnolog-pair-SL flag; ortholog GI-profile Pearson r; **module×module negative density** (our own clustering of Costanzo profiles, never the GO-derived SAFE labels; fractional membership for 1:many orthologs); measured flag |
| PPI | BioGRID physical, protein–protein systems only (Affinity Capture-RNA, Protein-RNA and Co-localization dropped), with a **PMID blocklist**: re-download BIOGRID-ORGANISM-5.0.262, stream the pombe member, keep only the PMIDs of genetic rows (sha256 recorded), delete the zip, block by PMID. Plus an HTP-only ablation; Vo 2016 Y2H; cerevisiae BioGRID physical via orthologs | degree, PageRank | direct edge; # shared neighbours; Jaccard; "shared neighbours but no edge"; capped shortest path; inductive diffusion (random-walk-with-restart) similarity |
| EXPR | COXPRESdb RNA-seq and microarray matrices; Marguerat/Carpy abundance | level, variability | rank-normalised co-expression per platform |
| KOPROT | Öztürk 2022 (KO and WT strains only) | # proteins changed in KO | KO-profile correlation; **directed** "KO of a changes b's abundance" (and vice versa) |
| LOC | Matsuyama 2006 | one-hot localisation | same compartment; Jaccard |
| SEQ | PomBase peptides, Pfam/PANTHER family IDs, Ensembl paralogues, ESM-2 650M embeddings | length, # domains, ESM PCA | paralog flag (recent vs ancient by % identity); shared family; ESM cosine |
| CONS | Ensembl Compara (367 fungi), S. japonicus presence | conservation breadth; phylogenetic profile | profile correlation |
| **NBR** (similarity transfer; key C2/C3 lever) | For each non-GI similarity space (phenomics, KO-proteome, co-expression, PPI-diffusion, ESM, Costanzo-profile): kNN among **training** genes | similarity-weighted mean training hit rate of a gene's k nearest training neighbours | for (a, b): weighted mean training S2 / hit rate of (a, c) over b's training neighbours c (and symmetric); for C3, block mean over (c, d) neighbour pairs |
| **RYAN-fold** (C1/C2; includes the Ryan module×module density, computed nested) | Ryan training pairs only | out-of-fold additive-logit propensity β (shrunk); training mean S | out-of-fold logistic-MF score; training-profile correlation; `seen_a`/`seen_b` flags |

**Leakage rules for features**
1. **No design features.** No feature encodes Ryan design (query/array status, measurement count, reciprocal flag).
2. **Nested stacking (replaces "masked mix").**
   - **Inner split.** Within each outer training set, re-run the C1/C2/C3 split generator: 5 inner gene folds T_j and a 10% inner pair hold-out H_j.
   - **Inner statistics.** For each j, all Ryan statistics come only from pairs with neither gene in T_j and not in H_j: the B1 β (`init_score`), the MF score, NBR rates and profile correlations.
   - **Training rows** are the inner eval rows:
     - C3-like: both genes in T_j.
     - C2-like: one gene in T_j, with `seen=0`. Each such pair appears in two inner folds; one seeded copy is kept.
     - C1-like: H_j.
   - Class weights are 0.25/0.57/0.18. All features are rates, not counts.
   - **Outer eval rows** get features computed exactly the same way, from all outer-training pairs.
   - Inner folds also serve tuning and early stopping. **Gene-grouped validation:** when validating on the inner-fold-j rows, drop every training row that involves a T_j gene, so the stacker never validates on its own training rows.
3. **NBR specification.**
   - **Similarity.** W per source (non-GI only), with weights max(sim, 0) and self excluded.
   - **Neighbours.** Each gene's k = 20 nearest training genes, chosen separately among query-role and array-role training genes. k ∈ {10, 20, 50} is tuned once.
   - **Features**, with Y = training hit (or S) matrix and M = measured mask:
     - directional: `dir(a→b) = (Y Wᵀ + λp₀)/(M Wᵀ + λ)` at (a, b), with λ = 1 and p₀ = global training prevalence (never a's own rate);
     - block: `W Y Wᵀ / W M Wᵀ`;
     - support: `M Wᵀ / ΣW`, i.e. the normalised fraction of neighbour weight with measurements, so it does not scale with training-set size.
   - NaN similarities get weight 0.
   - Symmetrised as nan-max and nan-mean, for both hit and S, per source, plus a cross-source mean.
4. **Allowlist** (granular to file, sheet and column). Every feature table records its sources in a manifest. A test checks that every source is on DATA.md's `use=feature` allowlist. Not allowed: FYPO, `results/eda/*`, `docs/data_inventory.md`, Koch 2012 columns, NET-FF, and anything else not on the list.
5. **One ID mapper.** All sources go through a single mapper (`src/idmap.py`: exact, then PomBase synonyms/obsolete IDs), which logs unmapped IDs. No silent drops.

## 5. Model ladder (simplest first)
- **B0** Random (= prevalence).
- **B1** Additive-logit gene propensity `c + β_a + β_b`, L2-shrunk, fitted jointly on training pairs so the partner mix is accounted for. Unseen genes get β = 0, so this is the pure degree baseline. Feature-predicted propensity lives inside M1/M2.
- **B2** Costanzo ortholog ε. **B3** Costanzo ortholog profile similarity.
- **B4** Symmetric logistic matrix factorisation with gene biases (k ∈ 8–32). Covers C1 and few-shot fold-in; falls back to B1 for unseen genes.
- **Order of work:** B0 → B1 → B4 → B2/B3 → build the full M2 table once → M2, with M1 and M1N as ablations → M4-as-feature → M3 (one attempt) → M5 (average of ≤3 models). E-TARGET compares binary vs Huber only.
- **M1** LightGBM on non-Ryan G+P features. The inner-fold B1 logit is passed as `init_score`; with and without this is an ablation. Symmetric (min/max plus propensity-ordered raw features); a test asserts `score(a,b) == score(b,a)`.
- **M1N** M1 + NBR features.
- **M2** M1N + RYAN-fold features, built by nested stacking (§4 rule 2).
- **M3** Inductive neural pair model: gene encoder → e; score = MLP([e_a+e_b, e_a⊙e_b, P]). BCE + Huber S2 auxiliary loss; feature dropout for cold-start.
- **M5** Ensemble / stacking: nested out-of-fold, isotonic calibration per class if pooled ranking requires it.
- **M4** Cross-species pre-training on Costanzo. First as a **feature**: a Costanzo-trained model's score on ortholog pairs. A full transfer model follows only if that helps, and is restricted to features available in both species (ESM, PPI topology, sequence/conservation). More cerevisiae data is acquired only if M4 shows promise.
- **Stopping rule:** keep an idea backlog in RESEARCH_TREE.md. Stop when 3 consecutive rungs or ideas fail the adoption rule and no supported idea remains, or when near the time limit.

## 6. Verification, controls, review
- **Toy generator** (written by an agent other than the pipeline author): planted propensities + modules + noise at Ryan-like sparsity and prevalence. Expected AUPRC is stated in advance. Two checks:
  - Positive control: the pipeline recovers the planted signal within tolerance.
  - Negative control: with labels permuted, AUPRC ≈ prevalence.
- **Each test has a deliberately broken variant it must catch:** a split that leaks a gene; a target encoding using eval labels; a non-symmetric scorer; a metric with flipped labels.
- **Out-of-fold leakage tests:**
  - Flipping eval labels leaves eval features and predictions bit-identical.
  - Flipping one training row's label leaves that row's own features unchanged.
  - A random-label canary scores ≈ prevalence.
  - Five planted leaky mutants must each be caught (e.g. NBR using the eval gene's own rate, β fitted in-sample).
- **Gene-permutation test**, pass criterion stated in advance: M1 primary ≤ B1 primary + 0.005.
- **Split tests:** no C2/C3 eval gene in that fold's training pairs; no pair in both train and eval; counts reported.
- **Leakage tests:**
  - Scrambling eval labels leaves predictions unchanged.
  - Permuting gene identities in the features drops M1 to ≤ B1.
- **Metric tests:** our AP vs sklearn; bootstrap code vs a toy analytic case.
- **Independent review:** key numbers are re-derived by a different agent **with its own minimal code** (not importing `src/`).
- **Registry:**
  - `src/evaluate.py` itself appends one row per run to `results/registry.tsv`: run_id, git commit (plus a diff hash if the tree is dirty), protocol/split/feature hashes, metrics and runtime.
  - Dirty runs are recorded but barred from adoption.
  - Parallel implementation agents work in separate git worktrees.
  - Numbers in REPORT.md cite run_ids. `scripts/check_claims.py` and the fresh-clone rerun are needed only before Phase F.

## 7. Compute and runtime
- Hardware: 10-core Apple Silicon, 32 GB, MPS. Measured: LightGBM 1M rows × 150 features, 300 rounds in 12 s; predicting 1M rows takes 2.5 s; peak 2.4 GB.
- Feature store: gene-level matrices plus G×G similarity matrices (float32 memmap, ~5.1k² ≈ 100 MB each), built once. Pair features are assembled in 1M-row chunks; universe scoring is chunked.
- **Runtime reporting** goes through one timing harness with fixed threads (`n_jobs=8`) and records the hardware. It reports feature build (one-off), fold features, training, prediction for 7M pairs, and peak RSS.

## 8. AI-agent failure modes → mechanisms
Protocol freeze plus lint (§0); a single evaluation entry point, `src/evaluate.py`, the only code that joins labels to predictions and computes metrics; physical lockbox separation plus access log (§2); registry plus claim checking (§6); broken-variant tests (§6); independent re-derivation (§6); allowlist manifest (§4).

**Task template for agents:**
- one hypothesis and input hashes;
- forbidden actions (edit protocol/splits/tests, read the lockbox);
- deliverables and a wall-clock cap;
- a summary of ≤300 words citing run_ids and stating what was not verified.

`run_pipeline.sh` re-runs everything from a fresh clone at each milestone.

## 9. Phases and deliverables
- **Vertical slice first:** minimal Phase A → tier-1 features (SMF/PHEN, CER, PPI, ESM) → B0/B1/B4 → NBR → M2. Remaining feature groups are added afterwards, one at a time, as ablations.
- **A. Infrastructure:** protocol.yaml, ID mapper, splits, evaluate.py, registry, toy generator, tests. Then the **protocol freeze**.
- **B. Features:** gene matrices, similarity matrices, PPI with PMID blocklist, NBR/RYAN-fold builders (out-of-fold), linkage diagnostic, covariate-shift check.
- **C. Baselines:** B0–B4, plus an oracle-propensity diagnostic (true held-out-gene hit rates; diagnostic only).
- **D.** M1 → M1N → M2, ablations, E-TARGET, E-ALLELE.
- **E.** M3, M5, M4, E-FEWSHOT.
- **F.** Lockbox (once), deliverable predictions, REPORT.md.

Deliverable: `scripts/predict_all_pairs.py` writes a ranked score for every universe pair, with flags (class relative to Ryan, linkage-untestable). For Ryan-measured pairs it also provides an out-of-fold score. The file is not committed if large; a summary is.

---
## Appendix A — Round-1 review disposition
Reviewers: external PI (PI), prior work (PW), statistics (ST), data (DA), algorithms/compute (AL), AI failure modes (AI), yeast genetics (GE). Full texts in reviews/round1_*.md.

**Accepted**
- Out-of-fold Ryan features with masked-mix training (PI1, AL3, PW1, ST11) → §4 rule 2.
- Similarity-transfer/NBR features (PI4, AL4, GE4, PW2) → §4 NBR.
- Pooled universe-weighted AUPRC as the primary metric (PI2, ST1); weights computed over unmeasured pairs (ST2) → §3.
- Universe = not essential, including condition-dependent genes (PI3, GE11, DA7) → §1.
- Two-way gene bootstrap and paired comparisons (ST3) → §3.
- Stricter adoption rule (ST4, PI11) → §3.
- Registry and pre-registered shortlist with Holm correction (ST5, AI5) → §3, §6.
- Equal tuning budget and inner-split HP/early stopping (ST6, AL12) → §2, §3.
- Seed 7 re-partitions dev only (ST7, AI4) → §2.
- Stratification with a hub stratum (ST8) → §2.
- Noise reference fixed (ST9) → §3.
- Shift diagnostics: C1-arr, partner role, covariate shift, homology (PI5, PI8, ST10, PW5, PW6, GE10) → §2.
- Protocol.yaml freeze and lint (AI1); single evaluate entry point (AI2); lockbox separation (AI3) → §0, §2, §8.
- EDA files forbidden as inputs (AI6) → §4 rule 4.
- Explicit train ids / nested supervised steps (AI7) → §4, §5.
- Broken-variant tests and toy generator by another agent (AI8) → §6.
- Independent re-derivation with own code (AI9) → §6.
- No pair-level Ryan in git (AI10) → already policy.
- Task template, stopping rule and timing harness (AI12–14) → §5, §7, §8.
- run_pipeline.sh (AI15) → §8.
- Additive-logit B1 (AL1, PW7, GE15); B4 logistic MF (AL2); `init_score` (AL5); symmetry tests (AL6); BCE + Huber loss and calibration (AL7) → §5.
- ESM-2 650M (AL8) → §4.
- Inductive PPI features only, no GNN on the Ryan graph (AL9) → §4.
- Ladder reorder and M4-as-feature first (AL10, PI7, PW4 partly) → §5.
- Memmap feature store (AL11) and runtime components (AL13) → §7.
- Linkage diagnostic, never a feature (GE1) → §1.
- SMF first-class; Kim relabelled as heterozygous-diploid fitness (GE2) → §4.
- Hypomorph indicator (GE3, DA12) → §1.
- Costanzo modules/embedding (GE5, PW3) → §4.
- ε aggregation fix (GE6) → §4.
- Within-complex vs between-pathway via "shared neighbours, no edge" (GE7) → §4.
- Paralog recent/ancient split; directed KO-proteome (GE8) → §4.
- Ortholog-class stratum and S. japonicus flag (GE9) → §3, §4.
- Phenomics PCA (GE12) → §4.
- DATA.md completeness and pinned releases (DA1) → Phase A task.
- FYPO carve-out and allowlist (DA2) → §1, §4.
- BioGRID PMID blocklist (DA3) → §4.
- Single ID mapper (DA4) → §4.
- DAmP-only and merged-ID exclusions (DA5, DA6) → §1.
- Reciprocal/single-orientation C1 strata (DA8, PI10) → §3.
- Essential-in-cerevisiae from Costanzo strain classes (DA10) → §4.
- Fix LITERATURE.md errors (PW11) → Phase A task.
- E-TARGET (PI6) → §1.
- Oracle-propensity diagnostic (PI7) → Phase C.
- Out-of-fold scores for measured pairs in the deliverable (PI9) → §9.
- Per-gene macro AUPRC as a diagnostic only (PI12) → §3.

**Deferred** (backlog, revisited only if the relevant stratum underperforms)
- Extra cerevisiae datasets for full M4 (DA9, PW4).
- Compara-based CER features (DA11).
- Soft orthologs and AlphaFold disorder (PW8).
- EMF/XSMF comparator (PW9).
- Kuzmin trigenic data (GE16).
- Sporulation expression (GE13).
- Selection-drug hub check (GE12, second half).
- DAmP-mixed S2 sensitivity (GE14): done as a report-only check.

**Rejected**
- Koch 2012 SMF column (GE2, part). It derives from E-MAP colony measurements, which is a circularity risk; Rodríguez-López standard-condition fitness is used instead.

## Appendix B — Round-2 review disposition
Reviewers: external PI (PI), statistics (ST), AI failure modes (AI), data+domain (DD), algorithms+prior work (AP). Full texts in reviews/round2_*.md.

**Accepted**

| Change | Reviewer refs |
|---|---|
| Shared Poisson gene bootstrap of the exact statistic used for selection | PI1, ST1 |
| Lockbox drawn immediately | PI2, AI2 |
| protocol.yaml includes seeds, inner folds and the NBR k grid | PI3, AI10 |
| Build a vertical slice first | PI4 |
| B1 sets β = 0 for unseen genes | PI5 |
| Tuning caps and seed counts | PI6, AP5 |
| Shift diagnostics are report-only and run once | PI7 |
| Separate per-class models dropped | PI9 |
| Block all full-label files; fold-scoped label API | AI1, AI8 |
| −2.3 read from config, including in the build scripts | AI3 |
| evaluate.py writes the registry; dirty runs recorded but barred from adoption | AI5, AI6 |
| Lighter process overhead | AI7 |
| Out-of-fold leakage tests with planted mutants | AI9, ST2 |
| Report feature shift between training and eval rows | AI11 |
| Pass criterion for the permutation test | AI12 |
| DAmP exclusion decided per pair | DD1 |
| Rule for unknown-viability genes | DD2 |
| BioGRID re-fetch for the PMID blocklist; HTP-only ablation | DD3 |
| Our own Costanzo clustering | DD4 |
| Granular allowlist | DD5 |
| Complete DATA.md; pin releases | DD6 |
| Nested stacking replaces the masked mix | ST2, AP2 |
| Seed-7 confirmation and sign-flip rule | ST3 |
| Lockbox pre-registration file | ST4 |
| NBR specification | AP1 |
| Inner-fold init_score | AP3 |
| Build the M2 table once; ladder order | AP4 |
| M3 gets one attempt; M5 averages ≤3 models; E-TARGET is binary vs Huber | AP6 |
| Report the NBR-ESM gain on the paralog split | AP9 |
| Fix LITERATURE.md | AP10 |

**Settled**
- PI8: the primary C1 term is the standard C1 hold-out, whose pairs contain query genes. C1-arr stays a diagnostic, because Ryan has no array×array labels; they can only be simulated by masking, which shrinks training data.

**Deferred**
- Costanzo smoothed ε (AP7).
- Rank-normalising M4 inputs and a zero-shot cerevisiae baseline (AP8). Revisit if M4-as-feature helps.
- Ortholog agreement table (DD7).

**Folded in**
- R1 AI11 (literal gene-ID lint) becomes part of the Phase A protocol lint.

## Appendix C — Round-3 (final) review disposition
Reviewer: fresh senior reviewer; full text in reviews/round3_final.md. Verdict: GO-with-fixes.

**Applied:** fixes 1–7.
1. Contradictions removed:
   - "masked mix" and per-class models dropped from M2;
   - one training-pairs rule;
   - rule numbering fixed;
   - duplicate bullet removed.
2. Gene-grouped validation in the stacker.
3. Ryan module density moved to RYAN-fold.
4. NBR support normalised.
5. Hypomorph indicator defined, with a test.
6. Sign-flip veto applies only when the opposite-direction CI excludes 0.
7. C1 composition stated; C1-arr reported next to the headline.

**Applied in docs/INTERFACES.md:** fix 8.

**Partly accepted:** fix 9 (cuts).
- The HTP-only PPI ablation and importance weighting become **report-only, run once**.
- FEWSHOT is limited to k ∈ {5, 20}.
- Full M4 stays gated on M4-as-feature helping.

**Implementation detail:** fix 10 (definitions). The implementing scripts document it:
- −3.76 = the raw-score equivalent of S2 = −2.3 for single-orientation pairs (docs/data_inventory.md);
- the linkage window is derived empirically in the diagnostic;
- "k-corrected min" and the ExN/ExE definitions are documented in build_cer.py.

**Design frozen** for implementation as of v3.1. Further changes go in the decision log in RESEARCH_TREE.md.
