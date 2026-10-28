# Independent verification of the Huber incumbent (2026-10-28)

The verifier did not build the model. Verdict on M2r_huber and M2r_bag5_huber: **all checks PASS; no leak, no bug.** The scripts and their aggregate outputs are in this directory.

1. **Target leak — PASS** (`check1_target_canary.py`, outer fold 0, 1 torch thread)
   - The Huber target is −min(S2,0) with δ = 2.3. It comes only from inner-eval rows of the outer training set.
   - Early stopping uses only the inner validation rows of fold j.
   - Corruption test: S2 was randomised on all 325,360 pairs that touch held-out genes or sit in the C1 hold-out, flipping 109,805 labels. Features, targets, masks, best_iter and the predictions of all 5 seeds stayed bit-identical.
   - Positive controls, both detected:
     - early stopping on outer-eval AP: best_iter 233→9;
     - outer-eval rows added to the target: best_iter 424→188.
2. **Label permutation under Huber — PASS** (`check2_permutation.py`)
   - Folds 0 and 1, seeds 0–2, full pipeline rebuilt.
   - Primary 0.0208 and 0.0235, against a prevalence of 0.0211.
3. **Re-derivation — PASS** (`check3_metric.py`, the verifier's own AP, weights and bootstrap)
   - Scores match the registry:

     | Run | Primary | C1 / C2 / C3 |
     |---|---|---|
     | M2r_bag5_huber | 0.18138 | .2773 / .1681 / .0947 |
     | M2r_huber | 0.17915 | .2721 / .1668 / .0936 |

   - Paired Δ (300 replicates):
     - huber − M2r: +0.0091 [0.0047, 0.0131]
     - bag5_huber − M2: +0.0114 [0.0069, 0.0154]
     - bag5_huber − huber: +0.0022 [0.0011, 0.0035]
   - Seed-7 Δ: +0.0076.
   - One model per seed: Huber scores 0.170–0.173 against binary 0.157–0.159.
4. **Eval label and rows unchanged — PASS**
   - Eval rows, labels and the protocol hash are identical across M2 and the Huber runs.
   - The evaluation, protocol, splits and stacking code is unchanged since 3a9bf62.
5. **Seeds and rules pre-stated — PASS**
   - Seeds come from config/m2plus.yaml (committed at 19:23, before the first Huber run at 20:26).
   - Unregistered trial runs cannot be ruled out.

**Notes:**
- Rebuilding mf_score with a different torch thread count moves it by up to 5.6e-4, not ~1e-6. This shifts Huber early stopping and moves a single-fold primary by about 0.001.
- LightGBM Huber predictions differ across LightGBM thread counts.
- So single runs carry about ±0.001 of rebuild noise. Paired Δs computed on shared cached features are unaffected.
