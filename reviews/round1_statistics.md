# Round 1 review — Statistics (splits, metrics, inference)

Statistics reviewer · 2026-10-28 · Checks: `reviews/scratch/stats_review_checks.py` (viable×viable S2, n = 1,036,718, prev 1.99%).

## 1. Verdict

**Sound skeleton, revise before implementation.** Gene-disjoint splits, a once-only lockbox and gene-block bootstrap are right. Problems: (a) the per-class score never tests whether scores are **comparable across classes**, though the deliverable is one ranked list; (b) the bootstrap is under-specified for C1/C3, where pairs belong to two crossed genes; (c) the adoption rule is too lenient for dozens of decisions on the same folds; (d) the noise ceiling as written can be circular; (e) nothing measures the shift from measured Ryan pairs to the universe.

Checked (20 simulated stratified 20% lockbox draws, 435 viable genes): C2 ≈ 331k pairs / 6.5k positives; C3 ≈ 41k pairs / **805 ± 53 positives from only 141 query genes**. On C3, AUPRC for a single fixed predictor varies by **sd 0.030 between lockbox draws** (C2: 0.010). For one draw, the gene-block bootstrap SE is **2.3× (C2) and 1.9× (C3)** the pair-bootstrap SE (C2 0.011 vs 0.0046; C3 0.028 vs 0.014). A per-gene propensity score alone (leaky; variance demo only) reaches AUPRC ≈ 0.20 (10× prevalence): hubs dominate. Universe weights 0.370/0.477/0.153 reproduce; over **unmeasured** universe pairs they are 0.248/0.569/0.183, and **81% of unmeasured C1 pairs are array-only×array-only**, a pair type Ryan never measures. Universe C2 seen partners are queries only 33% of the time. Prevalence: query×query 3.1%, query×non-query 1.8%.

## 2. Changes to DESIGN.md

1. **Add a pooled, universe-weighted AUPRC as the primary selection score (must).**
   - *What:* Pool out-of-fold predictions from C1, C2 and C3 into one ranking. Compute `average_precision_score` with sample weights that scale each class's total weight to the universe fractions. Keep the weighted mean of per-class AUPRCs as a secondary score.
   - *Why:* Any external AUPRC is computed on one ranked list. A model can do well within each class yet rank C3 pairs badly against C1 pairs (miscalibration across missing-feature regimes); the per-class mean cannot detect that.

2. **Choose the weights over the pairs that actually need predicting, and test robustness (must).**
   - *What:* Base the weights on unmeasured universe pairs (≈0.25/0.57/0.18), since measured pairs have known labels. Also report comparisons under all-pairs (0.37/0.48/0.15) and equal weights; flag decisions whose sign flips.
   - *Why:* The weights are a choice; the brief stresses unseen genes.

3. **Specify the bootstrap exactly (must).**
   - *What:* Resample **out-of-fold predictions pooled across all 5 folds**, not per fold.
     - C2: one-way bootstrap over the held-out gene.
     - C3 and C1: two-way (pigeonhole, Owen 2007) bootstrap. Resample query-side and array-side genes independently and weight each pair by the product of the two multiplicities. Use `sample_weight` in AP.
     - Paired comparisons: the same resample indices for both models, ≥1,000 replicates, percentile CIs.
   - *Why:* "Pairs follow genes" is undefined when both genes are resampled; per-fold C3 (≈110 query genes, ≈500 positives) is too noisy alone; pair-level CIs are ~2× too narrow.

4. **Tighten the adoption rule (must).**
   - *What:* Drop "or ≥4/5 folds" (sign-test p ≈ 0.19; folds share training data). Adopt a change only if both hold:
     - the paired gene-bootstrap 95% CI of Δ(pooled score) excludes 0, and
     - Δ ≥ a pre-declared minimum effect δ = 0.005 absolute.

     Compare seed-averaged predictions (mean over `SEEDS`) so seed noise does not drive decisions. Ties go to the simpler model.
   - *Why:* Dozens of decisions on the same folds; a lenient rule admits noise the lockbox later exposes as optimism.

5. **Keep a decision registry and measure optimism (should).**
   - *What:* Log every dev comparison (`results/decisions.tsv`: change, Δ, CI, adopted?), whether or not it was adopted. Before opening the lockbox, pre-register one **headline model**, the shortlist, and the contrasts of interest (headline vs B1; headline vs M1). Apply Holm correction across the shortlist contrasts. Report the dev-minus-lockbox gap as measured optimism.
   - *Why:* Makes the forking paths auditable; keeps the lockbox confirmatory.

6. **Give every rung the same tuning budget (should).**
   - *What:* Give every model being compared the same hyperparameter budget, including B1 smoothing and the B2/B3 missing-value handling. Alternatively, do inner tuning on a gene split inside each training fold.
   - *Why:* Best-of-30 on the comparison folds inflates tuned models vs untuned baselines.

7. **Fix how alternative split seeds are used (must).**
   - *What:* Split seed 7 re-partitions **only the dev folds**, and the lockbox genes stay fixed. State that lockbox and deliverable models are retrained on all dev pairs, and then on all Ryan pairs, with the hyperparameters chosen in CV.
   - *Why:* Re-drawing the lockbox would put dev-used genes into it (leak).

8. **Refine stratification (should).**
   - *What:* Stratify on **hit rate within role**, not raw degree, because queries are tested about 2× more often than arrays. Give the top roughly 2% of hubs (≈50 genes, ≥100 hits) their own stratum, spread round-robin across lockbox and folds. Report realised per-fold counts of C3 positives and query genes.
   - *Why:* Top 5% of genes carry 35% of positive endpoints; the 10 largest C3 genes carry ~20% of C3 positive endpoints.

9. **Use a non-circular noise reference (must).**
   - *What:* On the 62k reciprocal pairs, predict the orientation-B **single-orientation label** (raw < −3.76, the S2-equivalent threshold) from orientation A's raw score. This gives **AP = 0.345 (lift 16; prevalence 2.1%)**. Report it as a *replicate-agreement reference*, which is a lower bound on the noise-free ceiling, not an upper bound. If the S2 label is used instead, AP = 0.74, because the label averages in the predictor itself, so it is circular. Optionally simulate a latent-score + heavy-tailed-noise model for an upper bound.
   - *Why:* The current wording is ambiguous and one reading is circular.

10. **Add proxies and diagnostics for distribution shift (should).**
    - (a) **Role-masked C1:** for the 409 query+array genes, drop their query rows from training and evaluate their held-out pairs. Both genes are then "seen only as arrays", which mimics the 81% array×array share of unmeasured C1.
    - (b) Report C2 separately by seen-partner role (query vs array-only), and reweight it to the universe mix (33% query).
    - (c) Covariate-shift check: train a classifier on gene features to separate Ryan genes from the 1,406 unseen universe genes. If AUROC > 0.65, add importance-weighted C2/C3 scores (weights based on the held-out gene's propensity) as a sensitivity analysis.
    - (d) Report lift and AUPRC standardised to a common prevalence (2.0%) next to raw AUPRC, because class and pair-type prevalences differ (3.1% vs 1.8%) and raw AUPRC depends on prevalence.
    - *Why:* Otherwise lockbox numbers describe query-involving measured pairs, not the deliverable.

11. **Use out-of-fold target encoding inside training (should).**
    - *What:* Compute RYAN-fold gene rates and profiles for each *training* pair with an inner K-fold over training pairs, so a pair's own label never enters its features. During training, randomly mask RYAN features for whole genes so the model sees the C2/C3 missing-feature pattern.
    - *Why:* In-sample encoding inflates C1 and miscalibrates C2/C3 vs C1 (change 1); the scramble test does not catch it.

**No change needed:** labels and raw-label sensitivity; lockbox size (C3 gene-bootstrap SE ≈ 0.03 — fine for paired contrasts; state it); single lockbox touch; per-gene macro AUPRC; E-FEWSHOT (bootstrap by gene).
