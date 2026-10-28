# Round 1 review — ML algorithms and compute

Reviewer: algorithms/compute agent · 2026-10-28 ·

## 1. Verdict

**Approve with changes.** The ladder order is right: baselines, then GBDT on engineered features, then deep or cross-species add-ons (LITERATURE §4.4–4.5). Compute is not a constraint. There are three algorithmic gaps:

- (a) B1 combines gene propensities additively.
- (b) There is no collaborative-filtering rung for C1.
- (c) M2 has a train/test mismatch. Its Ryan features are never missing in training but are always missing (NaN) for unseen genes at test. They also include the row's own label.

The biggest missing C2/C3 idea is to let unseen genes **borrow the Ryan profiles of their nearest seen neighbours** in non-GI feature space.

**Probe.** This was exploratory and used no design split; nothing was selected from it.

- Setup: random 10% pair hold-out of viable×viable S2 pairs (1.04M pairs, prevalence 1.97%), seed 0.
- Script (reviewer scratchpad, not committed): `/private/tmp/claude-503/-Users-colmclaude-ClaudeExperiments-PombeSL-prediction/b2f61750-ea96-4fbb-9f26-9936c21c7a69/scratchpad/c1_probe.py`.
- The probe touched pairs that may later fall in the lockbox. The PI should log this in the research tree.

| Model | Held-out AUPRC |
|---|---|
| Smoothed hit-rate **sum** (B1 as written) | 0.155 |
| Hit-rate product / logit-sum | 0.195 |
| Logistic gene-bias model (MF k = 0) | 0.196 |
| Logistic MF + biases, k = 16, untuned | **0.270** (2 s, CPU) |

**Benchmark.** LightGBM on synthetic 1M × 150 float32, 300 rounds, 63 leaves, 10 threads:

- Training: 12 s.
- Prediction: 2.5 s per 1M rows.
- Peak RSS: 2.4 GB.

**Environment.** Apple M4, MPS available. `fair-esm` and `transformers` are **not installed**.

## 2. Concrete changes

**1. B1 in logit space. (must)**
- **What:** Change B1 to `logit P = c + β_a + β_b`, a logistic gene-bias model fitted on training pairs. For unseen genes, predict β with a regressor on gene features trained on the fitted β values.
- **Why:** Under hub structure, P(neg) is roughly multiplicative in the two propensities. The additive form loses about 20% AUPRC (0.155 vs 0.195). A weak degree-only control would flatter every later rung.

**2. New rung B4: symmetric logistic matrix factorisation for C1. (must)**
- **What:** `logit = c + β_a + β_b + ⟨e_a, e_b⟩`, with k ∈ {8, 16, 32} and tuned L2. Use a BCE loss, optionally with an auxiliary S2 regression head. Fit it per fold.
- **Why:** In the probe it adds +0.07 AUPRC over propensity at negligible cost.
- **Also:** E-FEWSHOT can use MF fold-in. The MF score feeds M2, computed out-of-fold (change 3).

**3. Nested gene-level out-of-fold construction for all Ryan-derived features. (must)**
- **What:** Split each outer training set into 5 inner gene folds.
  - Compute target encoding, profile correlation, MF score and neighbour scores for each training pair from inner-training pairs only.
  - Pairs touching an inner-held gene then carry NaN for that gene's features, so training rows look C1-, C2- and C3-like in realistic proportions.
  - Add `seen_a`/`seen_b` indicators.
- **Why:** As written, M2 over-trusts features that contain the row's own label. It also never sees the missing pattern that defines C2/C3 test rows.
- **Alternative:** Separate per-class models. Compare the two.

**4. Neighbour-borrowing features for C2/C3. (must; largest expected C2/C3 gain)**
- **What:** For each gene, find its kNN among *training* genes, separately in each non-GI similarity: phenomics, KO-proteome, Costanzo ortholog GI profile, co-expression, PPI diffusion and ESM.
  - **C2:** For (a seen, b unseen), the feature is the similarity-weighted mean S2 (and hit rate) of (a, c) over c ∈ kNN(b).
  - **C3:** The mean over (c, d), with c ∈ kNN(a) and d ∈ kNN(b).
- **Why:** Propensity says how often a hub interacts, not with *whom*. This feature transfers the hub's pair-specific profile to unseen genes.
- **Cost:** One sparse matrix product per source.
- **Rules:** One feature per source, so each is an ablation unit. Build them with the nesting from change 3.

**5. Explicit decomposition. (should)**
- **What:** Pass the B1 logit as LightGBM `init_score`, so the trees model residual relatedness. Compare against using propensity as an ordinary feature.
- **Why:** The top 5% of genes hold 35% of hit endpoints. With the offset, the "beyond-degree" gain becomes directly measurable.

**6. Symmetry. (should)**
- **What:** Keep min/max/mean of gene features. Also add canonically ordered raw features, with gene 1 being the gene of higher predicted propensity, so that hub × partner-type cues survive.
- **Test:** Assert `score(a,b) == score(b,a)` on 10k pairs for every model.

**7. Loss, imbalance and calibration. (should)**
- **Losses:**
  - Keep BCE as the primary loss.
  - Add one LightGBM Huber regression on clipped S2 ([−6, 3]), because near-threshold pairs carry information.
  - Do not use lambdarank grouped by gene: it optimises within-gene ranking, not global AUPRC.
- **Imbalance:** Do not reweight Ryan. AUPRC is rank-based, so weighting only adds variance. Weighted negative subsampling is fine for Costanzo.
- **Calibration:** The deliverable ranks C1, C2 and C3 pairs together. Calibrate each class's scores with isotonic regression on out-of-fold predictions. Report the pooled, universe-weighted AUPRC as a secondary metric.

**8. Concrete ESM plan. (should)**
- **Model:** ESM-2 650M (`esm2_t33_650M_UR50D`), mean-pooled, fp32 on MPS. About 20–45 min for 5,126 proteins. Proteins over 1,022 aa use averaged overlapping windows.
- **Setup:** Pin `fair-esm` or `transformers` in requirements. Record the weight checksum and licence in DATA.md.
- **3B model (11 GB, 2–4 h):** Only a "could", and only if the 650M ablation shows that ESM matters.
- **Paralog leakage:** Put paralog families in the same gene fold. Otherwise ESM similarity leaks across "unseen" genes.

**9. Inductive graph features only. (should)**
- **Do:** Use closed-form diffusion-kernel / random-walk-with-restart similarities on PPI, computed with scipy. This avoids a node2vec dependency.
- **Don't:** Use GNNs on the Ryan graph. They are transductive and reach chance in C3 (Feng 2024).

**10. Reorder and gate. (should)**
- **Order:** B0 → B1(logit) → B2/B3 → **B4 MF** → M1(offset) → **M1+neighbour-borrowing** → M2(nested) → M3 → M5 → M4.
- **M4 first as a feature:**
  - Train LightGBM on Costanzo with rank-normalised, species-agnostic features.
  - Use its prediction on pombe pairs as a feature.
  - Fine-tune only if that helps.
- **M4 memory:** 11.4M × 60 float32 is 2.7 GB. Subsample negatives 1:10.
- **Stop rule:** Stop M3/M4 after one budgeted attempt if they fail the §3 decision rule.

**11. Compute plan and feature store. (should)**
- **Training cost:** LightGBM is seconds per fold, not minutes.
- **Precompute:** Build each pair similarity once as a dense G×G float32 memory-mapped `.npy`. Each is 105 MB, about 1.6 GB for 15 matrices. Compute shortest paths with scipy `csgraph`.
- **Assembly:** Gather pair features by index, in 1M-row chunks. Never hold the 6.4M × 150 table in pandas.

**12. No early stopping or HP selection on the reported fold. (must)**
- **What:** Use an inner gene-disjoint split of the training genes.
- **Why:** Otherwise the dev scores that drive the §3 decision rule are inflated.

**13. Fair runtime reporting. (should)**
- **Log separately to the result JSON:**
  - one-off feature build;
  - per-fold Ryan-feature construction;
  - per-fold training;
  - 6.4M-universe prediction;
  - peak RSS.
- **Conditions:** Use fixed `num_threads=10` and record the device, the median over the 5 seeds, and the hardware string.
