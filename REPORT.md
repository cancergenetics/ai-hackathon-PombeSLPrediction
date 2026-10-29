# REPORT — predicting synthetic lethality across all non-essential *S. pombe* gene pairs

Date 2026-10-29. The full record is in RESEARCH_TREE.md (decision log, research tree), DESIGN.md v3.1 (frozen protocol), DATA.md (sources, versions, checksums, licences) and docs/*.md (one write-up per rung). Every number below comes from a committed results file at a clean git commit, and each one an adoption decision rested on was re-derived by an agent that did not produce it (reviews/verify_*).

## 1. Summary

- **Task.** Rank every pair of non-essential *S. pombe* genes by how likely it is to be a strong negative genetic interaction, defined as Ryan et al. 2012 S-score < −2.3.
  - The universe has 3,923 genes: viable and condition-dependent protein-coding genes, plus Ryan deletion strains of unknown viability. That gives **7,693,003 pairs**. The shipped model trains and scores them in 24 min on a laptop CPU (4 threads).
  - Training data: Ryan 2012 only (Dataset S2, the authors' averaged S-scores; 1.88% positives).
  - Inputs: label-free gene and pair data, plus the full budding-yeast map (Costanzo 2016).
  - No fission-yeast GI data from any other source, no text-mining-derived networks, and no GO.
- **Final model, `M2r_huber`.** A nested-stacked LightGBM, built in two layers:
  - **Label-free features (358).** Phenomics, budding-yeast ortholog interaction profiles, PPI, expression, KO proteome, localisation, sequence/ESM-2 and conservation.
  - **Features derived from the training labels:**
    - an additive gene-propensity logit;
    - matrix factorisation;
    - similarity-transfer ("NBR") rates from the measured partners of each gene's nearest neighbours under 9 label-free similarity sources;
    - an explicit gene-propensity model that predicts how interaction-prone an unseen gene is from its label-free features.

    These features are built inside nested inner folds, so no row ever sees its own label.
  - **Stacker.** Trained with a Huber loss on the continuous score −min(S2, 0) rather than the binary label, averaged over 3 seeds.
- **Held-out test (lockbox).** 480 genes were removed before any modelling. The shortlist was preregistered and the evaluation run exactly once.

  | model | primary AUPRC [95% CI] | C1 (both genes seen) | C2 (one gene unseen) | C3 (both unseen) |
  |---|---|---|---|---|
  | **M2r_huber (shipped)** | **0.211 [0.179, 0.247]** | 0.285 | 0.210 | 0.125 [0.084, 0.184] |
  | M2 (binary stacker) | 0.201 [0.172, 0.233] | 0.266 | 0.199 | 0.127 |
  | M1 (label-free LightGBM) | 0.162 [0.138, 0.189] | 0.214 | 0.154 | 0.122 |
  | B4 logistic matrix factorisation | 0.110 [0.089, 0.132] | 0.247 | 0.080 | 0.023 |
  | B1 additive gene-propensity logit | 0.106 [0.086, 0.128] | 0.223 | 0.080 | 0.023 |
  | random | 0.021 (= prevalence) | | | |

  - The primary metric is pooled AUPRC, with classes weighted to the mix of *unmeasured* universe pairs: C1/C2/C3 = 0.24/0.57/0.19. That is the mix a user of the tool will face.
  - All four preregistered Holm-adjusted contrasts of M2r_huber reject:
    - vs B1 +0.105 [0.079, 0.135]
    - vs B4 +0.101 [0.075, 0.131]
    - vs M1 +0.049 [0.033, 0.067]
    - vs M2 +0.010 [0.003, 0.017]
  - Other metrics: AUROC 0.86. Class-weighted precision among the top 1,000 lockbox pairs is 0.55.
- **Main lesson.** Most of the predictable signal is gene-level: how "sick" or interaction-prone each gene is.
  - An oracle that knows each held-out gene's true hit rate scores 0.22 on dev.
  - Degree-only baselines score 0.10 and are at chance for C3.
  - For unseen genes the model recovers only part of this. On dev, C3 is 0.094 against the oracle's 0.235 (≈40%). On the lockbox, C3 is 0.125 against 0.023 for the degree baselines.
  - Almost all of the C3 signal comes from label-free gene features: label-free-only M1 already reaches 0.122 on lockbox C3. The Ryan-derived stacking helps mainly C1 and C2.
  - Pair-specific signal (which partner of a hub) is much weaker.

## 2. Approach

### 2.1 Data and labels (DATA.md, DESIGN.md §1)
- **Labels.** Ryan Dataset S2, y = S2 < −2.3, read from config/protocol.yaml (frozen; sha256 73edd56d…).
  - Raw single-orientation scores are used only for a sensitivity check. On the lockbox, the headline model scores 0.235 on the raw-mean label, but that label has ~4.3% prevalence, so the number is not comparable with 0.211.
  - DAmP-only pairs are excluded from evaluation, decided per pair.
  - Non-coding genes and inviable strains are dropped from training.
- **Universe.** PomBase 2026-10-01 viability (used for the universe only, never as a feature): viable + condition-dependent protein-coding genes, plus 22 unknown-viability genes that have a Ryan deletion strain. A further 144 unknown-viability genes are scored and flagged `viability_unknown`.
- **Inputs.** 20 registered datasets (15 feature sources, 4 universe-only, 1 label source) (DATA.md; machine-readable config/data_allowlist.yaml), each with source, version, sha256 and licence. Circularity rules, enforced by tests:
  - PPI from BioGRID physical only, with a PMID blocklist that removes 40% of pombe rows coming from papers that also report genetic interactions;
  - no FYPO multi-allele phenotypes, no STRING/GeneMANIA, no GO or InterPro2GO, no other pombe GI screens.
- **Restricted data.** Never committed: data/ is git-ignored, and only aggregates and model outputs are in git.

### 2.2 Splits and metric (DESIGN.md §3)
- **Gene-disjoint evaluation**, because the tool must work for genes absent or rare in Ryan. 39% of universe genes and 63% of universe pairs involve a gene never measured by Ryan.
  - **C1:** both genes seen in training (a 10% held-out pair set).
  - **C2:** one gene held out.
  - **C3:** both genes held out.
- **Lockbox.** 480 of 2,404 Ryan genes (stratified by role × hit-rate quintile + hubs; seed 20261028), plus a lockbox C1 set. The lockbox was drawn early, but not before every label-touching analysis: during design review round 1, two reviewer agents ran exploratory probes on all Ryan pairs. No model or feature decision was made from those probes, and this is logged in the decision log.
- **Dev CV.** 5 gene folds, plus a second partition (seed 7) as a robustness check.
- **Statistics.**
  - A shared Poisson(1) gene bootstrap (1,000 replicates) gives paired CIs.
  - Adoption rule: the paired 95% CI excludes 0, Δ ≥ 0.005, and Δ > 0 on the seed-7 partition. One step at a time: a bundle may not carry a component that fails on its own.

### 2.3 Model ladder (dev CV, primary AUPRC; prevalence 0.021)

| rung | model | dev primary | C1/C2/C3 | dev-CV wall time (5 folds, registry) | verdict |
|---|---|---|---|---|---|
| B0 | random | 0.021 | | – | control |
| B1 | additive gene-propensity logit | 0.103 | .207/.080/.022 | 4 s | baseline |
| B4 | logistic MF | 0.108 | .232/–/– | 166 s | baseline |
| B2/B3 | Costanzo ε / profile similarity of orthologs | 0.026 / 0.025 | | < 1 s | weak alone |
| diag | NBR cross-source mean | 0.125 | C2 .117 | 892 s | lever for unseen genes |
| diag | oracle gene propensity | 0.220 | C3 .235 | – | upper bound for a gene-level model |
| M1 | LightGBM, 358 label-free features | 0.145 | .214/.134/.088 | 352 s | adopted (+0.041 vs B1) |
| M2 | + nested RYAN/NBR/GP stacking | 0.170 | .251/.159/.095 | 948 s (incl. nested features) | adopted (+0.025 vs M1; seed-7 +0.031) |
| **M2r_huber** | M2 with a Huber stacker on −min(S2, 0) | **0.179** | .272/.167/.094 | 529 s (features cached) | **adopted** (+0.009 [0.005, 0.013]; seed-7 +0.006, clean rerun M2r_huber_split7__…55fdbe vs M2_split7) |
| | + 5-seed bagging | 0.181 | | 869 s | not adopted (+0.002 < bar) |
| | + cross-species XSP features | 0.170 (on M2) | | 1,056 s (incl. 629 s feature build) | null (+0.0002) |
| | + C3-focused logistic blend | 0.181 | | ~120 s | null (−0.0001) |
| | + continuous-target GP/NBR (E-CONT) | 0.179 | | 1,812–2,102 s (incl. add-on features) | null (−0.0001 to −0.0007) |
| M3 | two-tower MLP (symmetric) | 0.170 | .263/.154/.095 | 1,075 s | null (−0.010) |
| M5 | rank ensembles | 0.145–0.181 | | 1–2 s + members | null (best +0.0015, n.s.) |

- **Ablations of M2** (docs/m2.md):
  - NBR is the largest Ryan-derived family, +0.019.
  - RYAN + GP together add +0.007.
  - GP alone adds +0.003, which is not significant; it substitutes for β/MF/NBR.
  - GP's toy check **failed** its pre-stated bar: unseen-gene Spearman 0.563 < 0.6, because the B1 β target itself correlates only 0.55 with the truth. GP was kept because it belongs to the RYAN block, which passed as a whole (M2 vs M1N +0.007). Its own contribution is unproven.
- **M1 ablations:**
  - Of the label-free groups only phenomics (+0.022) and budding-yeast ortholog features (+0.004) mattered.
  - A gene-permuted control collapses C3 to prevalence.

### 2.4 Verification (reviews/)
- **Phase A infrastructure:** all numbers were re-derived. 4 of 11 planted mutants escaped the original tests, which led to test hardening; all are now caught.
- **M2:**
  - A leak canary corrupting every held-out label left all Ryan-derived features bit-identical, except `mf_score`, which jitters with torch thread count. Two planted leaks were detected.
  - Label permutation gives 0.023, at prevalence.
  - The metric was re-derived exactly.
- **Huber incumbent:**
  - The target-leak canary was bit-identical, and two planted leaks were detected.
  - Permutation gives 0.022.
  - The paired Δ was re-derived.
- **Lockbox:** all numbers re-derived to 1e-14 with independent code; eval/training hygiene had 0 mismatches; prereg compliance confirmed; no leakage (reviews/verify_lockbox/REPORT.md).
- **Tests:** 249 passed, 1 skipped (results/tests/pytest_final.txt; pytest, including mutation tests, toy data with known answers, and lints that restrict label access and hardcoded thresholds).

## 3. Results on the held-out test set

### 3.1 Lockbox
- **Setup.** Prereg `config/lockbox_prereg.yaml` (sha256 deec74e3…):
  - headline M2r_huber, shipped regardless of ranking;
  - shortlist M2r_huber, M2, M1 plus baselines B1 and B4;
  - seeds exactly as adopted;
  - 4 contrasts with Holm correction;
  - boosting rounds fixed by the dev-CV inner early-stopping rule.
- **Training and scoring.** All models were retrained on all 775,918 dev training pairs (1,904 genes) and scored once on 456,107 eligible lockbox pairs (2,238 genes). This took 772 s on 4 threads, with peak RSS 14 GB.
- **Access log** (results/lockbox/access_log.tsv): one refused attempt (a path bug in the commit check, fixed and regression-tested; no lockbox data read), then one completed run.
- **Lockbox vs dev.** The lockbox scores are higher than dev for the learned models (M2r_huber 0.211 vs 0.179; M1 0.162 vs 0.145) but not for the degree baselines (0.106 vs 0.103). An independent verifier (reviews/verify_lockbox/REPORT.md) re-derived every number exactly and found no leakage. Their reading is that the gap is mostly sampling noise plus a small training-data effect:
  - **Not significant.** z ≈ 1.5 for M2r_huber, with the dev and lockbox SEs combined.
  - **Lockbox genes look like dev genes.** Prevalence, stratum composition, hit rate, degree and feature coverage all match.
  - **Driven by a few genes.** The gap is concentrated in C2/C3, and dropping just 5 lockbox genes moves C2 from 0.210 to 0.183.
  - **More training data explains only a little.** An M1 learning-curve probe shows a small part of the gap comes from training on all dev genes instead of 80%.
  - **Why the baselines don't move.** Their C2/C3 scores ignore the unseen gene.

  Treat about 0.18–0.21 as the realistic range for primary AUPRC on new genes.

### 3.2 Rarely-seen genes (docs/rarely_seen.md; report-only)
- **No gene in Ryan is rarely *measured*.** Every seen gene has ≥106 training pairs. The practical "rarely seen" case is genes with few *hits*.
  - Grouping one-gene-unseen (C2) pairs by how many training hits their seen gene has (0, 1–2, 3–9, 10+):

    | seen gene's training hits | 0 | 1–2 | 3–9 | 10+ |
    |---|---|---|---|---|
    | AUPRC | 0.008 | 0.022 | 0.055 | 0.194 |
    | prevalence | 0.25% | 0.42% | 0.93% | 4.3% |

  - That is 3–6× lift over prevalence in every bin.
- **Few-shot test.** Revealing k measured pairs of each held-out gene (pooled C2+C3 AUPRC):

  | k revealed pairs | 0 | 5 | 20 |
  |---|---|---|---|
  | pooled C2+C3 AUPRC | 0.149 | 0.143 | 0.170 |

  - **k = 5 does not help.** The stacker never saw sparsely measured genes. This does not affect predictions from Ryan data, but it matters for a user who adds a few new measurements. See §5.

### 3.3 Linkage (docs/deliverable.md)
- Ryan measured almost no pairs closer than 300 kb on the same chromosome: coverage ≤6%, against ~77% elsewhere. Such pairs are flagged `linked` and not penalised.
- Same-chromosome pairs 300–1,500 kb apart show a small excess hit rate. At 300–1,000 kb it is 2.15% against 1.82% between chromosomes (OR 1.18, p 3e-8); at 1,000–1,500 kb it is 2.11% (OR 1.16, p 1.5e-5). The M2 out-of-fold scores (the diagnostic predates M2r_huber) do not track this excess.

## 4. The deliverable

- **Command.** `scripts/predict_all_pairs.py` (model spec: config/models.yaml, default `M2r_huber`).
- **Output.** One row per universe pair (data/processed/universe/, git-ignored), with these columns:
  - `score` and `rank`;
  - `class`: C1/C2/C3 relative to Ryan;
  - flags `viability_unknown`, `linked` (with the gap in kb), `measured_in_ryan` and `in_training`;
  - for measured pairs, the **out-of-fold / lockbox score**, which users should prefer over the in-sample score. On measured dev pairs the in-sample AP is 0.34 against 0.21 out of fold.
- **Final model, trained on all Ryan data (dev + lockbox).** Shipped file: `predictions_M2r_huber_all.parquet`; summary in results/universe/summary_M2r_huber_all.json.
  - **Training data:** 1,311,979 pairs from 2,376 genes, 24,729 of them positive.
  - **Output:** all 7,693,003 pairs (3,923 genes) were scored, and every output check passed: row count, no NaN, unique pairs, and padding rows identical across blocks.
  - **Class mix relative to all Ryan:**

    | class | pairs |
    |---|---|
    | C1 (both genes seen) | 2.53M |
    | C2 (one gene unseen) | 3.76M |
    | C3 (both unseen) | 1.40M |

  - **Runtime:** 1,459 s wall on 4 threads (fit 659 s, scoring 795 s), peak RSS 18.3 GB, on an Apple M4 with 10 cores and 32 GB.
  - **Measured pairs.** 1.10M pairs are flagged `measured_in_ryan`, and 1.08M of them carry an out-of-sample score: dev out-of-fold, or the lockbox prediction from the dev-trained model.
  - **Run provenance.** The run is flagged dirty only because REPORT.md was untracked at the time. There is no code diff between start (605d2e5) and end (8fdd33a) under src/, scripts/, config/ or tests/.
  - **Top-ranked pairs:** the top 1,000 are 891 C1 + 109 C2. The top 100,000 are 52% C1, 43% C2 and 5% C3.
- **Runtime summary** (10-core laptop CPU, no GPU):

  | step | time |
  |---|---|
  | dev CV of the incumbent (5 folds × 3 seeds, incl. nested features) | ~9 min wall (486 s training) |
  | lockbox retrain + scoring of 5 models | 13 min |
  | final model: retrain on all Ryan + score 7.69M pairs | 24 min on 4 threads (fit 11 min, scoring 13 min), peak RSS 18.3 GB |
  | one-off feature building (ESM-2 650M) | 43 min (pombe) + 64 min (cerevisiae) on MPS |

- **How to read scores.**
  - Scores are comparable *within* a class but compressed for C3. The top-ranked pairs are almost all C1/C2: the top 1,000 has 0 C3 pairs, and the top 100k has 5% C3.
  - A user interested in uncharacterised genes should rank within class C3.

## 5. Limitations

1. **Gene-level signal dominates.** The model is good at saying which genes are interaction-prone, and much weaker at saying *which* partner of such a gene.
   - The C3 ceiling seems set by the label-free gene features: the oracle reaches 0.235 on dev and we reach 0.095 on dev and 0.125 on the lockbox.
   - Four separate attempts did not move dev C3: explicit propensity transfer, cross-species transfer, a C3 blend and continuous targets.
2. **Label noise.** Replicate S-score correlation in E-MAPs is about 0.4 (LITERATURE.md). Many "negatives" are untested or noisy, so absolute AUPRC is bounded well below 1. One orientation's raw score predicts the other orientation's label with AP 0.348 among reciprocal pairs (scripts/noise_reference.py). This is a reference point for measurement noise, not a ceiling.
3. **Coverage and design of Ryan.** Ryan is a query × array design with essentially no linked pairs (<300 kb) and no sparsely measured genes. Predictions for linked pairs and the few-shot regime are extrapolations.
4. **S2 shrinkage.** The S2 averaged score shrinks single-orientation scores (S2 −2.3 ≈ raw −3.76), so "strong negative" here is stricter than a raw threshold would be.
5. **Reproducibility.**
   - The Huber stacker and the torch MF feature are not bit-reproducible across thread counts: single runs vary by about ±0.001 primary. Paired comparisons share cached features and are unaffected.
   - The lockbox and final runs used a fixed 4 threads.
6. **Feature-source caveats.**
   - **Ryan 2012 is under the Elsevier user licence, with no redistribution or adaptation.** The shipped prediction file contains Ryan-derived pair-level scores and flags (`measured_in_ryan`, out-of-fold scores). It must therefore stay local and git-ignored; only aggregates are committed. Sharing predictions publicly would need a version without the Ryan-derived columns, plus a licence check.
   - Some sources have unclear licences: Costanzo, Kim 2010, Matsuyama.
   - Some sources were fetched from rolling URLs; pinned equivalents are documented in DATA.md.
   - The data themselves are not redistributed.
7. **Lockbox reuse.** The 480 lockbox genes are now spent. Any future model change has no untouched test set left inside Ryan.

## 6. Follow-ups and remaining ideas

- **Few-shot robustness.** Train the stacker with sparse-reveal rows, e.g. inner-fold genes with only k revealed pairs, or treat genes with fewer than about 20 partners as unseen. This makes the tool useful when users add a handful of new measurements.
- **M3 as a stacker feature.** The MLP is less correlated with the incumbent on C3 (ρ 0.80). As a rank-average it lifted C3 to 0.099, though not significantly. A nested M3 feature costs about 25 extra fits.
- **Better gene-level inputs for unseen genes.** This is the binding constraint. Candidates:
  - condition-specific fitness screens, if circularity-safe;
  - protein-complex membership from structure-based predictions (e.g. AlphaFold-multimer screens), not from curation;
  - richer pombe phenomics.
- **Linkage-aware modelling.** The 300–1,500 kb excess hit rate is unexplained. It could be a mild linkage artefact in E-MAP S-scores.
- **External validation.** Not allowed here, by design: no other pombe GI data. A future independent pombe screen would be the real test.
- **Ideas tried and closed:**
  - XSP: cerevisiae gene-propensity transfer gave CV ρ 0.39 in cerevisiae and 0.25 transferred, but added nothing on top of the ortholog features.
  - Seed bagging; C3-only blends; rank ensembles with weaker members.

## 7. Where things are

| what | where |
|---|---|
| brief, plan and record | PROMPT.md, DESIGN.md, RESEARCH_TREE.md, LESSONS.md, LITERATURE.md, DATA.md |
| per-rung write-ups | docs/m1.md, m2.md, m2plus.md, m2cont.md, m5m3.md, rarely_seen.md, deliverable.md, baselines.md, features_*.md |
| run registry / comparisons | results/registry.tsv, results/comparisons.tsv, results/runs/*.json |
| lockbox | config/lockbox_prereg.yaml, results/lockbox/ |
| verification reports | reviews/verify_phaseA, verify_m2, verify_huber, verify_lockbox |
| reproduce | scripts/setup_env.sh → download/build scripts (DATA.md) → scripts/make_splits.py → scripts/run_m2plus.py → scripts/lockbox_eval.py → scripts/predict_all_pairs.py |
