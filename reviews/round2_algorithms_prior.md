# Review round 2: algorithms/compute + prior work (DESIGN.md v2)

Reviewer: combined agent · 2026-10-28

## Verdict
- **Round 1:** the AL and PW points were adopted correctly. Gaps: PW3(ii) is missing, and PW11 is pending (LITERATURE.md still says 0.04 and 131 conditions).
- **NBR and masked mix:** right, but not yet unambiguous. They ignore the query×array structure, and k, the weights and the inner-fold row construction are undefined.
- **Compute:** feasible. W·Y·Wᵀ on a 5.2k² matrix takes 1.0 s.

## Changes

**1. Specify NBR. (MUST)**
- **Inputs:**
  - Y is the G×G training S2 matrix (also kept as a 0/1 hit matrix), set to 0 where unmeasured. M is its measured mask.
  - For each source, W keeps each gene's top-k training neighbours **separately among query genes and among array genes**. Use k=20 for each role, weight max(sim, 0), and exclude self.
- **Why split by role:** array-only genes are measured only against queries, and 81% of unmeasured C1 pairs are array×array. A global top-k leaves about 5 usable neighbours.
- **Features:**
  - `dir(a→b) = ((YWᵀ)[a,b] + λp₀) / ((MWᵀ)[a,b] + λ)`, with p₀ the global prevalence and λ=1. Never shrink toward a's own rate, because that leaks.
  - `block = (WYWᵀ)/(WMWᵀ)`.
  - `support = MWᵀ`.
  - Symmetrise as the nan-max and nan-mean of the two directions, plus block and support. Compute this for S and for hit, per source, plus a mean across sources.
- **Tuning:** tune k ∈ {10, 20, 50} once, on inner folds.

**2. Specify the masked mix. (MUST)**
- **Inner folds:** split the outer-training genes T into 5 stratified inner gene folds Tⱼ.
- **Ryan statistics:** for each j, compute all Ryan statistics (β, MF, NBR, profile correlation) from pairs with neither gene in Tⱼ, minus a 10% pair hold-out Hⱼ.
- **Rows emitted:**
  - C3-like: both genes in Tⱼ.
  - C2-like: one gene in Tⱼ. Set its `seen` flag to 0 and its features to NaN. Keep one copy of each duplicate, chosen with a seed.
  - C1-like: the pairs in Hⱼ.
- **Weights:** set class totals to the frozen universe weights (0.25/0.57/0.18). Training then targets the pooled metric. Per-class isotonic calibration is needed only for the per-class-model alternative.
- **Eval rows:** use features from all outer-training pairs, excluding C1ₖ.
- **Scale:** use rates, not counts, so that having 80% vs 100% of the data does not shift the scale.

**3. Treat B1 `init_score` as a Ryan-fold feature. (MUST)**
Training offsets come from the inner-fold β; in-sample β leaks labels into the "non-Ryan" M1.

**4. Collapse M1 → M1N → M2. (SHOULD)**
Build the M2 table once and run M1/M1N as group ablations, which is about two rungs faster. Order: B0, B1, B4, M2, ablations, the M4 feature, M3/M5. B2/B3 become free single-feature AUPRCs.

**5. Cap tuning cost. (SHOULD)**
- **Problem:** 30 configs × 5 folds × ~20 ablations × seeds ≈ 15–20 h.
- **Fix:** full search only for ladder rungs; ablations reuse the parent's configuration with inner early stopping; tune MF once per outer fold; seeds 3 (ladder) and 5 (shortlist).

**6. Cut or gate. (SHOULD)**
M3: one attempt, after M2. M5: an average of ≤3 models, since LightGBM is already the stacker. E-TARGET: binary vs Huber only. Per-class models: one comparison.

**7. (COULD)** Add Costanzo smoothed ε (PW3(ii)) as one W·E·Wᵀ product.

**8. (COULD)** M4: rank-normalise inputs within each species, and add the zero-shot cerevisiae score as a baseline row.

**9. (SHOULD)** Report the NBR-ESM gain on the paralog-family split too, since homology leakage inflates neighbour transfer most.

**10. (SHOULD)** Apply the LITERATURE.md fixes before Phase B. Agents anchor on 0.04.

## Compute
Store ≈2 GB; ~2 min of matrix products per outer fold; bootstrap ≈2 min per model. Nothing infeasible.
