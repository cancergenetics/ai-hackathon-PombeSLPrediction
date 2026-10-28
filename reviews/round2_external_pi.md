# Review round 2 — External PI

Reviewer: external PI agent · 2026-10-28 · Reviewed: DESIGN.md v2, Appendix A, reviews/round1_external_pi.md

## 1. Were my round-1 points addressed?
Yes, with two partial exceptions.
- Fully addressed:
  - PI1 (out-of-fold Ryan features with masked-mix training) → §4 rule 2.
  - PI2 (pooled AUPRC), with a better weighting: unmeasured-universe weights.
  - PI3 (universe), PI4 (NBR), PI6 (E-TARGET), PI8 (covariate shift), PI9 (out-of-fold deliverable), PI10–12.
- **PI5 (array×array) is only partly addressed.** C1-arr is a diagnostic, but the primary metric's C1 term (weight 0.25) still comes entirely from query×array hold-outs. Array×array pairs are 81% of unmeasured C1.
- **PI7 (staging) is partly lost.** The oracle diagnostic was added. But §9 now builds every feature group, including ESM, COXPRESdb, Compara and KOPROT, in Phase B, before any baseline runs. That is the opposite of the staging I proposed.

## 2. Assessment
v2 is scientifically sound and mostly consistent. Its risk has moved from leakage to throughput. Phase A has about 12 infrastructure components, and several diagnostics will cost as much as model work while changing no decisions. One statistical inconsistency must be fixed before the protocol freeze.

## 3. Changes

1. **[must] Bootstrap the statistic you select on.**
   - *Problem:* the primary is the mean over folds of each fold's weighted pooled AP. But §3 bootstraps the *concatenated* fold predictions, which is a different statistic: it mixes scores from 5 separately trained models. It also prescribes one-way resampling for C2 and two-way for C1/C3, and those cannot be combined into a single pooled AP.
   - *Fix:* use one scheme for all classes.
     - Draw multiplicities for all dev genes, with query-side and array-side draws separate if you keep two-way resampling.
     - Weight each pair by class weight × its genes' multiplicities.
     - Recompute each fold's pooled AP and average over folds.
   - Each dev gene is held out in exactly one fold, so one draw serves all folds. Test the scheme on the toy generator.
2. **[must] Draw the lockbox now**, before any round-2 probes or Phase-B EDA (LESSONS #4). It is cheap and removes the exposure risk logged in the decision log.
3. **[must] Complete protocol.yaml before the freeze:**
   - the seeds list (for example 3 seeds) and how many seeds the adoption comparisons average over;
   - the masked-mix rates, explicitly equal to the frozen class weights;
   - the inner-fold count.

   The design says these values are frozen, but it does not give them.
4. **[should] Re-sequence as a vertical slice.**
   - Order: A-minimal (protocol, idmap, splits, evaluate.py, registry, toy positive and negative controls) → B0/B1 → M1 on tier-1 features (SMF/PHEN, CER, PPI) → NBR → M2.
   - Build EXPR, KOPROT, LOC, SEQ/ESM and CONS afterwards, each as an ablation unit with a measured gain.
   - Defer the claim checker, the literal-lint test and the access-log `--reason` mechanics to Phase F.
5. **[should] Simplify B1.**
   - Give unseen genes β = 0 (the shrinkage prior) and let M1 learn feature-predicted propensity.
   - Predicting β from features inside B1 turns the baseline into a nested stacked model that needs its own out-of-fold fitting before it can serve as `init_score`. It duplicates M1 and adds a leakage path.
6. **[should] Make the tuning budget proportionate.**
   - Ablations and feature-group tests reuse one configuration, tuned once per phase on inner splits.
   - The equal ≤30-config search applies only to candidates entering the pre-registered shortlist.
   - Otherwise about 20 ablations × 30 configs × 5 folds × inner folds will dominate the schedule.
7. **[should] Demote to report-only, run once in Phase F:**
   - covariate-shift importance weighting (§2c);
   - the paralog-family sensitivity split (§2d; I said in round 1 that free paralog splitting mirrors the universe);
   - the universe-mix reweighting of C2 (§2b);
   - E-FEWSHOT k = 100 (keep 5 and 20).

   Keep C1-arr: it is cheap and decision-relevant.
8. **[could] Settle the C1 term's composition before the freeze.**
   - Either pool the C1-arr pairs into C1_k, weighted to the 81/19 array×array vs query×array mix, or state in §3 that the primary's C1 term covers only query×array.
   - Decide now. Changing it after the freeze would need a protocol amendment.
9. **[could] Drop "separate per-class models" as an M2 sub-experiment.** It is admissible only with calibration onto a shared scale, which is what M5 already does.

**No change needed:** labels and universe, the gene-disjoint C1/C2/C3 scheme, NBR design, leakage rules 1–4, the adoption threshold, the ladder order M1→M1N→M2→M3/M5/M4, and the deliverable format.
