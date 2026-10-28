# Review round 2: algorithms/compute + prior work (DESIGN.md v2)

Reviewer: combined agent · 2026-10-28

## Verdict
The round-1 AL and PW points were adopted correctly, with two gaps:
- PW3(ii), the smoothed ε feature, is missing.
- PW11 is still pending. LITERATURE.md still says prevalence ≈0.04 and 131 conditions.

NBR and the masked mix are right but **not yet implementable unambiguously**:
- They ignore Ryan's query×array structure.
- k, the weights and the way inner folds produce C1/C2/C3-like rows are all undefined.

Compute is feasible: W·Y·Wᵀ on a 5.2k² float32 matrix takes 1.0 s here.

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
- Training offsets come from the inner-fold β.
- In-sample β leaks labels into a supposedly "non-Ryan" M1.

**4. Collapse M1 → M1N → M2. (SHOULD)**
- Build the M2 table once and run M1 and M1N as group ablations. This reaches the best model about two rungs sooner.
- New order: B0, B1, B4, M2, then ablations, the M4 feature, and M3/M5.
- B2/B3 become free single-feature AUPRCs.

**5. Cap tuning cost. (SHOULD)**
- **Problem:** 30 configs × 5 folds × ~20 ablations × seeds comes to roughly 15–20 h.
- **Fix:**
  - Run the full random search only for ladder rungs. Ablations reuse the parent's configuration, with inner-split early stopping.
  - Tune MF once per outer fold, on inner fold 1.
  - Freeze the seed count: 3 for the ladder, 5 for the shortlist.

**6. Cut or gate. (SHOULD)**
- M3: one budgeted attempt, after M2.
- M5: a rank/logit average of ≤3 models, since LightGBM is already the stacker.
- E-TARGET: two variants only (binary vs Huber).
- The per-class-model comparison: run it once.

**7. Add Costanzo neighbourhood-smoothed ε (PW3(ii)). (COULD)**
- One W·E·Wᵀ product over the ortholog profile neighbours.

**8. M4. (COULD)**
- Rank-normalise inputs within each species (SINaTRA).
- Add the zero-shot cerevisiae-trained score as a baseline row.

**9. Homology. (SHOULD)**
- Also report the NBR-ESM gain on the paralog-family split. Neighbour transfer is what homology leakage inflates most.

**10. Apply the LITERATURE.md fixes before Phase B. (SHOULD)**
- Agents anchor on the 0.04 prevalence.

## Compute
- Memmap store: about 2 GB.
- Per outer fold: 6 feature builds × ~20 matrix products × ~1 s, plus MF.
- Bootstrap: 1,000 resamples over 1M rows take about 2 min per model.
- Nothing is infeasible on 10 cores and 32 GB.
