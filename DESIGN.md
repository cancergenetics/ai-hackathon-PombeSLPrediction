# DESIGN.md — Predicting synthetic-lethal (strong negative) genetic interactions in *S. pombe*

Version: v1 (draft for review round 1) · 2026-10-28 · Owner: PI agent
Inputs: PROMPT.md (fixed goal), LITERATURE.md, DATA.md, docs/data_inventory.md, docs/data_aux_inventory.md

## 0. Fixed goal (from PROMPT.md, not negotiable)
- Tool that scores **every pair of non-essential S. pombe genes** for a strong negative genetic interaction.
- Training data: Ryan et al. 2012 E-MAP. Positive = **S-score < −2.3**. Metric: **AUPRC**.
- Must work for genes **unseen or rarely seen** in Ryan → splits must reflect that.
- Allowed inputs: Costanzo 2016 (fully), other non-circular data. Forbidden: any other fission-yeast GI data, text mining / text-mining-derived resources (STRING etc.), Gene Ontology in any form.
- Report runtime alongside accuracy.

## 1. Labels and prediction universe
- **Label source:** Ryan Dataset S2 (authors' per-gene-pair averaged S-score; one value per unordered pair; 1,337,033 pairs). `y = 1[S2 < −2.3]`. Rationale: the brief speaks of "quantitative measurements for each pair" and a single per-pair S threshold; S2 is the authors' canonical per-pair matrix. S2 shrinks single-orientation scores (pseudo-averaging), so prevalence is 1.88% (2.0% in viable×viable) vs 4.3% on raw means. **Sensitivity analysis:** final models are also evaluated against `raw_mean < −2.3` labels (never used for selection).
- **Regression target (internal only):** S2 continuous value may be used as an auxiliary training target (e.g. Huber/quantile loss, or classification at several thresholds). Evaluation is always the binary label.
- **Prediction universe:** protein-coding genes with PomBase deletion viability = `viable` (3,580 genes; 6.41M pairs). Condition-dependent genes (177) are scored too but flagged; inviable genes are excluded from the deliverable.
- **Evaluation pairs:** only pairs where **both genes are `viable`** (1,036,718 S2 pairs, 2.0% positive). Pairs involving inviable genes (DAmP/hypomorph queries) are **allowed as training data** (decision to be tested by ablation E-ALLELE: train on all S2 pairs vs viable×viable only), never as evaluation.
- Pseudogenes / ncRNA / unmapped IDs in Ryan (≈20 genes) are dropped.

## 2. Splits (leakage-safe, gene-disjoint)
Test classes (Park & Marcotte 2012): **C1** both genes seen in training; **C2** exactly one seen; **C3** neither seen.

- **Lockbox test genes (G_test):** 20% of the 2,4xx Ryan genes, drawn once with seed 20261028, **stratified** by (Ryan role: query / array / both) × (negative-degree quintile) × (viability). Gives ≈ 333k C2 pairs (≈6.9k positives) and ≈ 42k C3 pairs (≈960 positives), viable×viable.
- **Lockbox C1 pairs:** 10% of pairs among dev genes, drawn at random (stratified by label), held out of all training.
- The lockbox is **touched once**, at the end, for a pre-declared shortlist (≤5 models + all baselines). It is never used for selection or tuning.
- **Development:** dev genes partitioned into **5 gene folds** (same stratification). For fold k: validation-C3 = pairs with both genes in fold k; validation-C2 = exactly one gene in fold k; validation-C1 = a 10% random pair hold-out among the other folds' genes; training = remaining pairs with neither gene in fold k (and not in the C1 hold-out).
- **"Rarely seen" genes (secondary, E-FEWSHOT):** for test-fold genes, reveal k ∈ {5, 20, 100} random labelled pairs per gene in training; evaluate on the rest. Shows how quickly per-gene data helps.
- **Why gene-level, not allele-level:** alleles of one gene and both orientations of a pair are merged in S2 → one row per gene pair, so gene-disjoint splitting is complete.
- **Seeds:** fixed list `SEEDS = [0, 1, 2, 3, 4]` (model randomness); split seed fixed (`20261028`) and one alternative split seed (`7`) to check split sensitivity of the final shortlist.

## 3. Metrics and statistics
- **Primary:** AUPRC (average precision, sklearn `average_precision_score`) per class C1, C2, C3, on viable×viable pairs.
- **Selection score:** universe-weighted mean `0.37·AUPRC_C1 + 0.48·AUPRC_C2 + 0.15·AUPRC_C3`. The weights are the fractions of the 6.41M universe pairs whose genes are both / one / neither in Ryan (from docs/data_inventory.md §8; recomputed by script). Rationale: reflects what the deliverable will be used on.
- Also report: AUPRC / prevalence (lift), AUROC, precision@top-1k per class, per-gene (macro) AUPRC for genes with ≥5 positives (guards against the "rank whole hubs" shortcut).
- **Uncertainty:** gene-block bootstrap (resample test genes with replacement, 1,000×; pairs follow genes) for CIs; **paired** bootstrap for model-vs-model differences. Across-fold mean ± sd in dev. Seed variance from the 5 seeds.
- **Decision rule for adopting a change:** improves the dev selection score across folds with paired-bootstrap 95% CI excluding 0 **or** consistent improvement in ≥4/5 folds; otherwise keep the simpler model.
- **Noise ceiling:** estimated from the 62k reciprocal pairs (orientation r = 0.40): AUPRC of predicting one orientation's label from the other orientation's score, as a rough upper reference.

## 4. Features (all computed for every universe gene; pair features symmetric in (a, b))
Gene-level (G) and pair-level (P) feature groups; each group is an ablation unit.

| Group | Source | Gene-level | Pair-level |
|---|---|---|---|
| CER | Costanzo 2016 via PomBase curated orthologs | ortholog GI degree (neg ε<−0.08,p<.05), ortholog SMF, essential-in-cerevisiae, has-ortholog, n-orthologs | max/min/mean ε over ortholog pairs, cerevisiae negative call, **ortholog GI-profile Pearson r** (computed from Data File S1), measured flag |
| PPI | BioGRID physical (pombe; protein–protein systems only), Vo 2016 Y2H; cerevisiae BioGRID physical via orthologs | degree, betweenness/pagerank | direct edge, # shared neighbours, Jaccard, shortest-path length (capped), network-embedding cosine (node2vec/spectral, transductive on PPI only) |
| EXPR | COXPRESdb RNA-seq + microarray matrices; Marguerat/Carpy abundance | mean level, variability, abundance | Pearson / Spearman co-expression (per platform), rank-normalised |
| PHEN | Rodríguez-López 2023 (129 conditions), Kim 2010 fitness | fitness, # conditions with phenotype | phenotype-profile correlation |
| KOPROT | Öztürk 2022 KO proteomes | # proteins changed in KO | KO-proteome profile correlation; does KO of a change abundance of b (and vice versa) |
| LOC | Matsuyama 2006 | one-hot localisation | same compartment, Jaccard of compartments |
| SEQ | PomBase peptides, Pfam/PANTHER domains, Ensembl paralogues, ESM-2 embeddings (computed locally) | length, # domains, ESM embedding (PCA-reduced) | paralog flag, % identity, shared Pfam/PANTHER family, ESM cosine |
| CONS | Ensembl Compara across 367 fungi | conservation breadth, phylogenetic profile | phylogenetic-profile correlation |
| RYAN-fold (C1/C2 only) | **Ryan training pairs in the current fold only** | training negative degree / hit rate (smoothed target encoding), training mean S | training-profile correlation between a and b; neighbour-based score (mean S of a with b's top-k profile neighbours) |

Rules: (i) no feature encodes Ryan **design** (query/array status, measurement count, reciprocal flag) because the universe has no such design; (ii) RYAN-fold features are recomputed inside every fold from training pairs only, and are missing (NaN) for unseen genes; (iii) every feature table records its source in a manifest; a test fails if a forbidden source (BioGRID genetic systems, GO, STRING, FYPO, NET-FF) appears.

## 5. Model ladder (simplest first; every rung compared with paired statistics)
- **B0** random (= prevalence).
- **B1 gene propensity:** P(pos) ≈ f(rate_a) + f(rate_b) from smoothed training hit rates; unseen genes get a hit rate **predicted from gene features** (regressor trained on training genes). This is the mandatory "degree-only" baseline.
- **B2** Costanzo ortholog ε (min over ortholog pairs; missing → median).
- **B3** Costanzo ortholog profile similarity.
- **M1** LightGBM on all G (symmetrised: min, max) + P features, binary objective. Main workhorse.
- **M2** M1 + RYAN-fold features (C1/C2).
- **M3** Inductive neural pair model: gene encoder (MLP over gene features + ESM) → embeddings; score = MLP([e_a+e_b, e_a⊙e_b, P features]); trained with BCE + auxiliary S2 regression; symmetric by construction.
- **M4** Cross-species pre-training: build the same feature set for *S. cerevisiae* genes where available (PPI, sequence/ESM, conservation, CER-intrinsic features excluded) and pre-train M3/M1 on Costanzo labels (ε < threshold matched to Ryan prevalence), then fine-tune on Ryan. Tests whether 11M cerevisiae pairs teach transferable rules.
- **M5** Ensemble (rank average / stacking on dev folds) of the best members.
- Hyper-parameter search: small, fixed budget per model (≤30 configs, random search) on dev folds, inner selection by the dev selection score.

Ablations: feature-group leave-one-out on M1; E-ALLELE (training-set composition); E-FEWSHOT; label threshold sensitivity (raw vs S2) for the final model.

## 6. Verification, controls, and review
- **Toy data with known answers:** synthetic GI matrix generated from a planted model (gene propensities + modules + noise) with matching sparsity/prevalence; the full pipeline must recover planted signal (positive control) and give AUPRC ≈ prevalence when labels are permuted (negative control).
- **Split tests:** assert no gene of a C2/C3 evaluation gene set occurs in that fold's training pairs; no pair occurs in both train and eval; C-class counts match expectation.
- **Leakage tests:** RYAN-fold features recomputed with eval labels scrambled must leave predictions unchanged; gene-label shuffle (permute features across genes) drops M1 to ≈B1 or below.
- **Metric tests:** our AUPRC vs sklearn on toy vectors; bootstrap code vs analytic toy case.
- **Independent review:** each milestone result is re-checked by an agent that did not produce it (re-run script from a clean checkout, re-derive key numbers).
- Every result file is JSON/TSV with git commit hash, seed, split id, runtime (wall clock, CPU), written by scripts only.

## 7. Compute plan
- Hardware: 10-core Apple Silicon, 32 GB RAM, MPS GPU. LightGBM on ~1M rows × ~150 features: minutes per fold. Universe scoring: 6.4M pairs in chunks. ESM-2 (650M) embeddings for 5,126 proteins on MPS: ~1 h one-off (fallback: 150M model). Costanzo profile correlations: 5.5k × 5.5k matrix, cheap.
- Pair feature store: compute gene-level matrices once; compute pair features on the fly per pair batch (vectorised), cached as parquet for Ryan pairs.

## 8. AI-agent failure modes and mitigations
- Silent leakage (fold-unsafe features, wrong subset) → asserts + leakage tests above + independent review.
- Lockbox peeking / metric shopping → lockbox evaluated once by a script that logs every invocation; metrics fixed here.
- Hallucinated data facts → numbers in docs come from scripts; reviewers re-derive.
- Over-long agent runs / context loss → each agent gets a narrow scope, writes a summary + result files; PI reads summaries; RESEARCH_TREE.md is the memory.
- Over-claiming → every claim in REPORT.md links to a results file.

## 9. Deliverables
- `src/` package + `scripts/predict_all_pairs.py` → ranked scores for all universe pairs (gzipped TSV, not committed if large; summary committed).
- REPORT.md (approach, best methods, lockbox results per class with CIs, runtime, limitations, follow-ups).
