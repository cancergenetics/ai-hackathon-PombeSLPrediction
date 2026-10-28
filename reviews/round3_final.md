# Review round 3 (final): go/no-go

Reviewer: independent PI (comp. biology/statistics) · 2026-10-28 · Reviewed: PROMPT.md, DESIGN.md v3 (incl. App. A/B), docs/INTERFACES.md, reviews/round2_external_pi.md, round2_statistics.md

## Verdict: **GO-with-fixes**
The design is sound and honest. The class weights are consistent: with about 2.3k seen and 1.5k unseen universe genes, a back-of-envelope count reproduces 0.37/0.48/0.15 and about 0.25/0.57/0.18. The lockbox sizes are plausible, and the lockbox sha256 matches the file. Fixes 1–4 are text-level and should be made **before the protocol freeze**. None of them needs re-review.

## Fixes (priority)
1. **[must] Remove contradictions.**
   - §5 M2 still says "masked mix"; it should say "nested stacking (§4 rule 2)".
   - §5 M2 still says "Also compared with separate per-class models", which Appendix B dropped. Delete it.
   - §1 says "all S2 pairs allowed" for training, but also that 12 inviable-deletion genes are dropped by default. State one rule.
   - Appendix A cites "§4 rule 3" for the EDA ban; it is rule 4.
   - §3 Lockbox lists the shortlist pre-registration twice. Remove one.
2. **[must] Gene-grouped tuning and early stopping for the stacker.** The inner eval rows are the stacker's training rows, so "inner folds serve early stopping" means validating on training rows. When validating on inner fold j, drop every row that touches a T_j gene from training. C2-like rows span two inner folds, so they need this. Without it, early stopping overfits and the predictor gets worse.
3. **[must] Move the "Ryan-density" module variant out of CER and into RYAN-fold.** It uses Ryan labels. As written, M1 ("non-Ryan features") and the CER ablation would contain Ryan information.
4. **[must] Normalise NBR `support = M Wᵀ`.** It is a count, which contradicts "all features are rates". It is also systematically about 1.25× larger on outer eval rows (100% of outer-training genes) than on stacked training rows (about 80%). Use support / Σ W, and include it in the train/eval feature-shift report.
5. **[should] Define the hypomorph indicator precisely.**
   - It is 1 iff the pair's label derives only from DAmP measurements.
   - Declare it an explicit exception to leakage rule 1.
   - Add a test that it is 0 on every dev, lockbox and deliverable eval row.
6. **[should] Make the sign-flip rule noise-robust.** Apply "keep the simpler model" only when the paired CI of Δ under the alternative weighting excludes 0 in the opposite direction. Under equal weights, C3 (about 700 positives per fold) will flip the sign of small real gains by chance and veto them.
7. **[should] State in §3 that the primary C1 term covers query×array pairs only.** REPORT.md must show the C1-arr diagnostic next to the headline, and state that the headline is not a direct estimate for array×array deliverable pairs.
8. **[should] Update INTERFACES.md.**
   - Add the path and keying for fold-scoped (NBR/RYAN-fold) feature caches.
   - Specify that a NaN similarity means weight 0 in NBR.
   - Specify that `lockbox_eval.py` calls `src/evaluate.py`, which stays the only label↔prediction join.
9. **[could] Cut overhead.**
   - Merge the duplicate §6 tests: "scrambling eval labels" is the same as the bit-identical flip test, and the gene-permutation test is stated twice.
   - Drop the importance-weighted C2/C3 and keep only the covariate-shift AUROC.
   - Drop the HTP-only PPI ablation.
   - Drop E-FEWSHOT k = 100.
   - Gate the full M4 transfer on M4-as-feature passing the adoption rule.
10. **[could] Define the remaining terms.**
    - −3.76: point to DATA.md (the S2 = −2.3 ⇔ raw map).
    - The linkage window: derive it from the diagnostic and freeze it before Phase F.
    - "k-corrected min" and ExN/ExE.

No other changes are needed. The labels, universe, gene-disjoint C1/C2/C3 splits, lockbox separation, shared Poisson bootstrap, adoption threshold, ladder order and deliverable format are fine as written.
