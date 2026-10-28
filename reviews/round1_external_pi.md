# Review round 1 — External PI (overall scientific strategy)

Reviewer: external PI agent, independent of the team · 2026-10-28 · Reviewed: DESIGN.md v1 (plus PROMPT.md, DATA.md, docs/data_inventory.md, docs/data_aux_inventory.md, LITERATURE.md §4–7)

## 1. Overall verdict

The design is sound and unusually careful. It has gene-disjoint C1/C2/C3 splits, a once-only lockbox, a gene-propensity baseline, toy-data controls and a forbidden-source manifest. I would keep its skeleton. It has three strategic gaps, though, each of which could cost more AUPRC or more honesty than any model change could recover:

1. **Training-time leakage and a missingness mismatch in the Ryan-derived features.** These features are computed in-sample for training rows and are always present in training, but they are absent for C2/C3 pairs.
2. **A misaligned selection target.** The tool will be judged on one AUPRC over one ranked list, but the design selects on a weighted average of per-class AUPRCs. That ignores whether scores are comparable across classes, and it never validates the array×array pairs that make up most of the deliverable's "C1".
3. **A "non-essential" definition that is too narrow.** It drops condition-dependent hubs such as fft3 and prw1, two of the top 10 genes by degree, from the universe and from evaluation.

Priorities are roughly right: trees on engineered features first, then deep and transfer models. But the ladder should be explicitly staged around where the signal lies, which is gene propensity plus inductive neighbour transfer. ESM, the neural pair model and cross-species pre-training should be later bets, not parallel work.

## 2. Proposed changes

**1. Make RYAN-fold features out-of-fold for training rows, and train under C2/C3-like missingness. Priority: must.**
- *Change (§4 rule ii, §5 M2):* within each outer fold, split the training genes into inner gene folds. Compute every RYAN-fold feature for a training row (hit rate, mean S, profile correlation, neighbour score) only from inner-training pairs. The inner held-out genes then supply training rows where one or both genes' Ryan features are NaN, in proportions matched to the target C-class mix.
- *Why:* smoothed target encoding and profile correlation on in-sample rows include each row's own label. The model will over-trust these features. And because training rows always have both genes "seen", the model never learns the one-side-NaN regime it must handle in C2 (48% of the universe). This is the most likely silent failure in M2.

**2. Make pooled AUPRC the selection score; keep per-class AUPRC as diagnostics. Priority: must.**
- *Change (§3):* the primary selection metric becomes AUPRC over the pooled lockbox/dev set C1∪C2∪C3. Each pair gets a sample weight so the class mix matches the universe (0.37/0.48/0.15, recomputed), passed as `sample_weight` to `average_precision_score`. Report the per-class AUPRCs and the current weighted mean alongside it.
- *Why:* the assessor ranks one list. A model can win in every class and still lose pooled AUPRC if its C1 scores (with Ryan features) and C3 scores (without) are on different scales. Forbid per-class models unless they are calibrated onto one scale, and test that calibration.

**3. Define "non-essential" as "not inviable". Priority: must.**
- *Change (§1):* the universe becomes viable + condition-dependent + unknown-viability genes that exist as deletion strains in Ryan. Primary evaluation pairs are both-genes-non-inviable pairs measured with deletion alleles. Report viable×viable as a subset.
- *Why:*
  - fft3 (285 hits) and prw1 (260 hits) are condition-dependent and among the strongest hubs.
  - The array library is a haploid deletion collection, so array genes are viable by construction.
  - An assessor's "non-essential" almost certainly means "deletion is viable". Excluding these genes removes real positives from both training and evaluation.

**4. Add inductive neighbour-transfer features usable for unseen genes. Priority: must.**
- *Change (§4 RYAN-fold row):* define gene b's neighbours by non-Ryan similarity: phenomics, KO-proteome, co-expression, PPI proximity, Costanzo ortholog profile and sequence/ESM. Then compute:
  - for C2 (a seen, b unseen): the mean training S / hit rate of a with b's top-k neighbours;
  - for C3: the mean over training pairs between a's neighbours and b's neighbours;
  - the gene-level analogue: the hit rate of b's neighbours (a propensity estimate for b).
- *Why:* the design's neighbour score needs b's Ryan profile, so it is unavailable for 63% of universe pairs. Guilt-by-association transfer through independent similarity is the main inductive route to pair-specific signal (LITERATURE §7.1 items 1, 4 and 5).

**5. Validate the array×array pair type. Priority: should.**
- *Change (§2):* take a random subset of the 409 query+array genes. Remove their query-role rows from training, so they are seen only as arrays, and evaluate pairs among them. Report this as stratum "C1-AA".
- *Why:* most universe "C1" pairs whose genes are both array-only were never measured in Ryan (arrays were only crossed to queries). The current C1 holdout contains only query×array pairs, so it says nothing about 37% of the deliverable.

**6. Treat the training target as a first-class early experiment. Priority: should.**
- *Change (§1, §5):* compare these training targets, always evaluating on the fixed S2 < −2.3 label:
  - S2 binary;
  - raw-mean < −2.3 (4.3% prevalence);
  - continuous S2 / raw regression (Huber);
  - soft labels weighted by number of orientations.
- *Why:* only 32% of raw negatives replicate in the reciprocal orientation. A denser or softer target is a cheap, legitimate way to gain AUPRC on a noisy label. The evaluation label stays unchanged.

**7. Stage the ladder and add an oracle headroom diagnostic. Priority: should.**
- *Change (§5):*
  - Phase 1: data, splits and controls, then B0–B3, then M1 on gene-propensity features (Costanzo ortholog degree/SMF/ortho-essentiality, phenomics fitness, Kim het-fitness, PPI degree, abundance, conservation) plus cheap pair features (paralogy, PPI, co-expression, phenotype correlation).
  - Phase 2: M2 (change 1) plus neighbour transfer (change 4) and the target experiment (change 6).
  - Phase 3: M4, M3, ESM and AlphaFold, only if Phase 2 plateaus.
  - Add an "oracle propensity" diagnostic: B1 using the test genes' true hit rates.
- *Why:* the literature and the hub concentration (top 5% of genes hold 35% of positive endpoints) say gene propensity dominates. The oracle measures how much AUPRC is available at gene level versus pair level, which tells us where to invest.

**8. Account for covariate shift to the 1,406 never-screened genes. Priority: should.**
- *Change (§3):*
  - Train a gene-level classifier for "in Ryan vs not" on features.
  - Report how feature coverage and missingness differ between the groups.
  - Report lockbox C2/C3 AUPRC both unweighted and importance-weighted towards never-screened genes.
  - Stratify results by feature coverage (has ortholog, has phenomics, has PPI).
- *Why:* Ryan queries were chosen (conserved and nuclear-enriched), so held-out Ryan genes are not representative of unscreened ones. The honest unseen-gene estimate needs this.

**9. Separate measured values from predictions in the deliverable. Priority: should.**
- *Change (§9):* for Ryan-measured pairs, ship cross-fitted (out-of-fold) predictions. Put the observed S2 in a separate column, and never blend it in silently.
- *Why:* returning training labels as "predictions" would inflate any later assessment on Ryan pairs and would misrepresent the model.

**10. Report a reciprocal-measurement stratum. Priority: could.**
- *Change (§3):* report a stratum for reciprocally measured pairs, and compute the noise ceiling on the S2 scale.
- *Why:* the S2 label depends on the measurement design. Prevalence is 3.58% for reciprocal pairs vs 1.79% for single-orientation pairs.

**11. Tighten the adoption rule. Priority: could.**
- *Change (§3):* replace "CI excludes 0 OR ≥4/5 folds" with "paired-bootstrap CI on the pooled score excludes 0".
- *Why:* about 20 ablations are planned, and the OR rule is lenient. Ties should go to the simpler model.

**12. State that per-gene macro AUPRC is diagnostic only. Priority: could.**
- *Why:* ranking hubs highly is legitimate under the stated pooled-AUPRC metric. Penalising it would optimise the wrong objective.

**No change needed** for:
- gene-level (not allele-level) disjoint splits and stratified lockbox;
- the once-only lockbox with an invocation log;
- the forbidden-source manifest test;
- the toy/permutation controls;
- the fixed seeds;
- excluding Ryan design features from inputs;
- keeping paralogues split freely across folds (this mirrors the universe).
