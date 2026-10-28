# Round 2 review — Statistics

2026-10-28 · Checks: `reviews/scratch/stats_review_r2_checks*.py`. They use real S2 universe labels (1.08M pairs) with synthetic scorers and measure variance only.

## 1. Were the round-1 points addressed?
Points 1, 2, 4–9 and 10(a–c) are addressed. Point 10(d) can be dropped. Partly addressed:
- **Point 3 (bootstrap).** The scheme differs by class, and "query-side vs array-side" is undefined for C2 and for dual-role genes. The CI uses concatenated predictions, but the point estimate is a fold mean.
- **Point 11 (out-of-fold Ryan features).** As written, it leaks or yields no C1 rows.

## 2. Changes

**1. Use one shared gene bootstrap that matches the point estimate (must).**
- Each replicate draws one multiplicity m_g ~ Poisson(1) per dev gene, used for all classes and all folds. Pair weight = m_a·m_b.
- Within each fold, rescale the class totals back to 0.25/0.57/0.18.
- The replicate statistic is the **fold mean of the within-fold pooled AP**, the same as the point estimate.
- The same draws serve every model (paired).
- *Evidence:* resampling classes independently, as §3 reads, gives SE **0.0040 instead of 0.0080**. The CIs are 2× too narrow, since each gene feeds all three classes.

**2. Replace "masked-mix" with nested stacking that reuses the split code (must).**
- *Problem:*
  - If features are computed out-of-fold over inner *gene* folds, every training row is C2/C3-like, so there are no clean C1 rows.
  - If a masked gene's partner gets β/MF/NBR features from all training pairs, the masked row's label leaks into them.
  - The same applies to the NBR rates of its neighbours.
- *Fix:*
  - Within each outer training set, re-run `make_cv_split`: 5 inner gene folds plus a 20% C1 pair hold-out.
  - For each inner split, build the RYAN-fold features, the NBR features and the B1 `init_score` exactly as at evaluation, for the inner C1/C2/C3 rows.
  - Stack these rows (≈1.5M), tagged by class and weighted to the target mix.
  - Do HP selection and early stopping on the stacked rows with **gene-grouped** splits.
- *Broken-variant test:* on toy gene-propensity data, a leaky-partner variant must show training C2 AP above evaluation C2 AP. The correct pipeline must not.

**3. The adoption rule is workable; close three gaps (should).**
- Paired SE of Δ:
  - ≈0.0006–0.0014 for systematic changes, where Δ ≥ 0.005 binds.
  - ≈0.0028 for a *null* change (same skill, different gene-level error). One such realised Δ was 0.0053, with its CI just touching 0.
  - So expect about 1 false adoption per 30–40 null comparisons. This is acceptable.
- Gaps to close:
  - (a) Run feature-group ablations **sequentially**: drop one group, re-fit, then re-test. Never drop two redundant groups based on separate leave-one-out tests.
  - (b) The bootstrap ignores training-partition variance. For rung-level adoptions (M1N, M2, M3, M4, M5), also require Δ > 0 on the seed-7 dev partition.
  - (c) If the sign flips across weightings, keep the simpler model.

**4. Make the lockbox confirmatory in practice (should).** Pre-register in a committed file:
- **p-value:** one-sided paired bootstrap, p = (1 + #{Δ* ≤ 0}) / (B + 1), with B = 10,000. Holm at α = 0.05, over the listed contrasts only.
- **Headline:** reported and shipped **whatever the lockbox ranking**; no switching.
- **Bootstrap:** follows change 1, resampling lockbox genes and dev genes (C2 seen partners and C1 genes).
- **All-dev retrain:** HPs and the number of boosting rounds are fixed from CV, e.g. median inner-best rounds × 1.25.
- **Test:** no lockbox C1 pair appears in `labels_dev.parquet`.
- **Expected precision:** paired-contrast SE ≈0.001–0.003, so Δ ≳ 0.01 is detectable after Holm.

**No change needed** elsewhere.
