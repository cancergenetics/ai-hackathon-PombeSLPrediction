# DESIGN.md — Predicting synthetic-lethal (strong negative) genetic interactions in *S. pombe*

Version: **v2** (after review round 1) · 2026-10-28 · Owner: PI agent
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
- **Prediction universe (deliverable):** protein-coding genes that are not essential, i.e. PomBase deletion viability ∈ {`viable`, `condition-dependent`} (3,757 genes, ≈7.06M pairs), plus unknown-viability genes that exist as deletion strains in Ryan. FYPO viability (`FYPOviability.tsv`) is a single-mutant phenotype. It is used **only** to define the universe and the evaluation masks and for stratification, **never as a feature**.
- **Primary evaluation pairs:** S2 pairs where both genes are in the universe. Exclusions: pairs involving a gene whose S2 is DAmP-only (e.g. rap1, sep1, ten1; list built by script); pairs involving merged Ryan IDs (train-only). **Secondary:** viable×viable only.
- **Training pairs:** all S2 pairs, including inviable/DAmP genes, are allowed. Experiment E-ALLELE (all vs universe-only training) decides this. A `hypomorph` indicator (always 0 at inference) prevents missingness of deletion-based features from acting as an essentiality proxy.
- Pseudogenes, ncRNAs and unmapped IDs are dropped.
- **Linkage:** E-MAP filters linked pairs. A diagnostic plots hit rate vs chromosomal distance, using PomBase coordinates. Distance is **never** a deliverable feature. Universe pairs within the linkage window are flagged "not E-MAP-testable" in the deliverable.

## 2. Splits (gene-disjoint; lockbox physically separated)
Classes (Park & Marcotte 2012): **C1** both genes seen in training, **C2** exactly one seen, **C3** neither seen.
- **Lockbox genes:** 20% of Ryan genes, split seed `20261028`. Stratified by Ryan role (query / array / both) × within-role hit-rate quintile, with the top 2% of hubs as their own stratum. Expected viable×viable sizes: C2 ≈ 331k pairs / 6.5k positives; C3 ≈ 41k pairs / ≈805 ± 53 positives, from ≈141 query genes. A script reports the actual counts.
- **Lockbox C1 pairs:** 10% of dev×dev pairs (random, label-stratified).
- **Physical separation:** `scripts/make_splits.py` writes `labels_dev.parquet` and `labels_lockbox.parquet`. `src/data.load_dev()` is the only loader of labels during development, and a lint test blocks reading the lockbox file anywhere except `scripts/lockbox_eval.py`. That script requires a committed shortlist file and hashes of the prediction files. It appends to `results/lockbox_access.log` and refuses a second run without a `--reason`.
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
- **Secondary:** per-class AUPRC; the weighted mean of per-class AUPRCs; the pooled score under all-pairs weights (0.37/0.48/0.15) and equal weights. Decisions whose sign flips across weightings are flagged. Also lift (AUPRC/prevalence), AUROC, precision@1k, and C1 split into reciprocal vs single-orientation pairs. Per-gene macro AUPRC is a diagnostic only.
- **Uncertainty:** a gene-block bootstrap (1,000 resamples) on the concatenated fold predictions.
  - C2: resample the held-out genes.
  - C1/C3: two-way bootstrap, resampling query-side and array-side genes independently and weighting each pair by the product of multiplicities.
  - Model comparisons use the **same** resamples (paired) and seed-averaged predictions.
- **Adoption rule:** a change is adopted only if the paired 95% CI on Δ(primary) excludes 0 **and** Δ ≥ 0.005. Otherwise the simpler model is kept. Every comparison goes into the registry, adopted or not.
- **Tuning fairness:** each compared model gets the same random-search budget (≤30 configs), tuned on inner splits.
- **Lockbox:**
  - Before opening it, a shortlist is pre-registered: one headline model, ≤4 others, all baselines, and the contrasts to test (Holm-corrected).
  - Reported alongside: the dev-minus-lockbox gap and per-class CIs.
- **Noise reference (not a ceiling):** among reciprocal pairs, predict orientation B's single-orientation label (raw < −3.76) from orientation A's raw score. AP ≈ 0.35 (lift ≈16) is a lower bound on achievable replicate agreement.

## 4. Features
All features are computed for every universe gene. Pair features are symmetric in (a, b). Each group below is one ablation unit.

**Sources and the features derived from them**

| Group | Source | Gene-level features | Pair-level features |
|---|---|---|---|
| SMF/PHEN | Rodríguez-López 2023 (standard-condition fitness + 129 conditions); Kim 2010 (heterozygous-diploid fitness, labelled as such) | haploid SMF; phenotype-vector PCA; # conditions with phenotype | min/max/product SMF; phenotype-profile correlation |
| CER | Costanzo 2016 via PomBase curated orthologs | ortholog degree as a **rate**, separately for NxN and ExN/ExE; ortholog SMF; any-ortholog-essential (Costanzo strain class); n-orthologs; has-ortholog; **Costanzo-profile embedding** (PCA of the ortholog's GI profile) | ε aggregated after collapsing alleles: mean, min, k-corrected min, n_pairs; ohnolog-pair-SL flag; ortholog GI-profile Pearson r; **module×module negative density** (Costanzo-profile clusters); measured flag |
| PPI | BioGRID physical, protein–protein systems only, with a **PMID blocklist**: drop rows from any publication that has S. pombe genetic rows in BioGRID; Vo 2016 Y2H; cerevisiae BioGRID physical via orthologs | degree, PageRank | direct edge; # shared neighbours; Jaccard; "shared neighbours but no edge"; capped shortest path; inductive diffusion (random-walk-with-restart) similarity |
| EXPR | COXPRESdb RNA-seq and microarray matrices; Marguerat/Carpy abundance | level, variability | rank-normalised co-expression per platform |
| KOPROT | Öztürk 2022 (KO and WT strains only) | # proteins changed in KO | KO-profile correlation; **directed** "KO of a changes b's abundance" (and vice versa) |
| LOC | Matsuyama 2006 | one-hot localisation | same compartment; Jaccard |
| SEQ | PomBase peptides, Pfam/PANTHER family IDs, Ensembl paralogues, ESM-2 650M embeddings | length, # domains, ESM PCA | paralog flag (recent vs ancient by % identity); shared family; ESM cosine |
| CONS | Ensembl Compara (367 fungi), S. japonicus presence | conservation breadth; phylogenetic profile | profile correlation |
| **NBR** (similarity transfer; key C2/C3 lever) | For each non-GI similarity space (phenomics, KO-proteome, co-expression, PPI-diffusion, ESM, Costanzo-profile): kNN among **training** genes | similarity-weighted mean training hit rate of a gene's k nearest training neighbours | for (a, b): weighted mean training S2 / hit rate of (a, c) over b's training neighbours c (and symmetric); for C3, block mean over (c, d) neighbour pairs |
| **RYAN-fold** (C1/C2) | Ryan training pairs only | out-of-fold additive-logit propensity β (shrunk); training mean S | out-of-fold logistic-MF score; training-profile correlation; `seen_a`/`seen_b` flags |

**Leakage rules for features**
1. **No design features.** No feature encodes Ryan design (query/array status, measurement count, reciprocal flag).
2. **Out-of-fold Ryan features.** RYAN-fold and NBR features for **training rows** are computed out-of-fold over **inner gene folds**, so a row's own label never feeds its features. Training includes rows whose Ryan features are masked at a rate matching the C1/C2/C3 mix: entire genes are masked, and `seen` flags are set accordingly. Eval rows use features from the full fold-training set.
3. **Allowlist.** Every feature table records its sources in a manifest. A test checks that every source is on DATA.md's `use=feature` allowlist. Not allowed: FYPO, `results/eda/*`, `docs/data_inventory.md`, Koch 2012 columns, NET-FF, and anything else not on the list.
4. **One ID mapper.** All sources go through a single mapper (`src/idmap.py`: exact, then PomBase synonyms/obsolete IDs), which logs unmapped IDs. No silent drops.

## 5. Model ladder (simplest first)
- **B0** Random (= prevalence).
- **B1** Additive-logit gene propensity `c + β_a + β_b`, L2-shrunk. Fit on training pairs (joint fit, so the partner mix is accounted for). For unseen genes, β is **predicted from gene features**, including NBR gene features.
- **B2** Costanzo ortholog ε. **B3** Costanzo ortholog profile similarity.
- **B4** Symmetric logistic matrix factorisation with gene biases (k ∈ 8–32). Covers C1 and few-shot fold-in; falls back to B1 for unseen genes.
- **M1** LightGBM on non-Ryan G+P features. The B1 logit is passed as `init_score`; with and without this is an ablation. Symmetric (min/max plus propensity-ordered raw features); a test asserts `score(a,b) == score(b,a)`.
- **M1N** M1 + NBR features.
- **M2** M1N + RYAN-fold features (out-of-fold, masked mix). Also compared with separate per-class models.
- **M3** Inductive neural pair model: gene encoder → e; score = MLP([e_a+e_b, e_a⊙e_b, P]). BCE + Huber S2 auxiliary loss; feature dropout for cold-start.
- **M5** Ensemble / stacking: nested out-of-fold, isotonic calibration per class if pooled ranking requires it.
- **M4** Cross-species pre-training on Costanzo. First as a **feature**: a Costanzo-trained model's score on ortholog pairs. A full transfer model follows only if that helps, and is restricted to features available in both species (ESM, PPI topology, sequence/conservation). More cerevisiae data is acquired only if M4 shows promise.
- **Stopping rule:** keep an idea backlog in RESEARCH_TREE.md. Stop when 3 consecutive rungs or ideas fail the adoption rule and no supported idea remains, or when near the time limit.

## 6. Verification, controls, review
- **Toy generator** (written by an agent other than the pipeline author): planted propensities + modules + noise at Ryan-like sparsity and prevalence. Expected AUPRC is stated in advance. Two checks:
  - Positive control: the pipeline recovers the planted signal within tolerance.
  - Negative control: with labels permuted, AUPRC ≈ prevalence.
- **Each test has a deliberately broken variant it must catch:** a split that leaks a gene; a target encoding using eval labels; a non-symmetric scorer; a metric with flipped labels.
- **Split tests:** no C2/C3 eval gene in that fold's training pairs; no pair in both train and eval; counts reported.
- **Leakage tests:**
  - Scrambling eval labels leaves predictions unchanged.
  - Permuting gene identities in the features drops M1 to ≤ B1.
- **Metric tests:** our AP vs sklearn; bootstrap code vs a toy analytic case.
- **Independent review:** key numbers are re-derived by a different agent **with its own minimal code** (not importing `src/`).
- **Registry:** `results/registry.tsv` is append-only, one row per run. Each row records run_id, git commit (refuses a dirty tree), protocol/split/feature hashes, metrics and runtime. Numbers in REPORT.md cite run_ids, and `scripts/check_claims.py` verifies them.

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
- EDA files forbidden as inputs (AI6) → §4 rule 3.
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
