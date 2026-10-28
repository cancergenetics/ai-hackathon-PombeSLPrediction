# Research tree — S. pombe synthetic-lethality prediction

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (latest first)

### 2026-10-28 — M2: baselines and first model
Infrastructure is built, independently verified (all numbers re-derived; test gaps closed, 11/11 planted mutants caught) and frozen (protocol sha256 73edd56d…). Label-free features cover 9 groups (PPI with PMID blocklist, phenomics/SMF, Costanzo orthologs, expression, KO proteome, localisation, sequence/ESM-2, conservation). Dev results (pooled universe-weighted AUPRC; prevalence 0.021): B1 degree 0.103, B4 MF 0.108, NBR mean 0.125, **M1 LightGBM 0.145** (C3 0.088 vs ≈0.02 for degree baselines). An oracle diagnostic (true held-out-gene propensity) reaches 0.220, so predicting how "hub-like" an unseen gene is matters most.

Next: M2 (nested stacking adding Ryan-derived, NBR and an explicit gene-propensity model) with clean reruns and paired comparisons; then E-TARGET, M3/M5, M4-as-feature. All runs so far were flagged dirty (fixed); adoption decisions wait for clean reruns.

### 2026-10-28 — M1: design complete (3 review rounds), implementation started
Literature, data acquisition and design are done. DESIGN.md v3 is the result of 12 independent reviews in two rounds plus a final go/no-go check. Key decisions: labels = Ryan S2 < −2.3 (1.88% positives); universe = non-essential (viable + condition-dependent) protein-coding genes; gene-disjoint splits with a lockbox of 480 genes drawn once; primary metric = pooled AUPRC weighted to the class mix of unmeasured universe pairs (C1/C2/C3); Ryan-derived features only via nested stacking; similarity-transfer (NBR) features as the main lever for unseen genes.

Implementation runs as a vertical slice: Phase A infrastructure (protocol, splits, evaluate, registry, tests), tier-1 label-free features, a toy generator written by a separate agent, and docs fixes are all in progress in parallel. Next: baselines B0/B1/B4, then NBR and the M2 model.

### 2026-10-28 — M0: project kickoff
Project initialised: git repo, Python 3.10 venv (`scripts/setup_env.sh`, `requirements.txt`; LightGBM/XGBoost need a libomp rpath fix on this Mac), PROMPT.md saved verbatim. Literature review and data discovery launched in parallel as independent agents.

Next: integrate LITERATURE.md + data inventory, draft DESIGN.md, run up to three rounds of multi-reviewer critique, then build the data pipeline and leakage-safe splits.

## Tree
- [x] 1. Literature review → LITERATURE.md (8k words; key: replicate r≈0.4 noise ceiling; edge-level cross-species transfer weak, module/profile-level stronger; Feng 2024: SL graph methods ≈ chance on unseen genes)
- [x] 2. Data discovery & acquisition → DATA.md, docs/data_inventory.md, docs/data_aux_inventory.md
  - [x] 2a. Core: Ryan S1/S2 (953 alleles×1955 arrays; 1.337M S2 pairs; 2,416 genes), Costanzo 2016 (11.4M ORF pairs), PomBase 2026-10-01
  - [x] 2b. Aux: BioGRID physical, COXPRESdb, Marguerat/Carpy abundance, Öztürk KO proteome, Rodríguez-López phenomics, Kim 2010, Matsuyama loc, Pfam/PANTHER, Compara, AlphaFold
  - [~] 2c. ESM-2 embeddings (pombe + cerevisiae)
  - [!] 2d. Label scale matters: S2 (authors' averaged) <−2.3 → 1.88% positives; raw mean → 4.3%. 39% of universe genes never measured; 63% of universe pairs involve an unseen gene.
- [~] 3. Design → DESIGN.md (≤3 review rounds)
  - [x] 3a. v1 draft
  - [x] 3b. Review round 1 (7 reviewers) → reviews/round1_*.md; disposition in DESIGN.md App. A
  - [!] 3c. Consensus of 4–7 independent reviewers: Ryan-derived features must be out-of-fold with masked-mix training; similarity-transfer (NBR) features are the main lever for unseen genes; select on pooled universe-weighted AUPRC.
  - [x] 3d. DESIGN v2
  - [x] 3e. Review round 2 (5 reviewers) → DESIGN v3 (nested stacking replaces masked-mix; precise NBR spec; shared Poisson gene bootstrap; per-pair DAmP exclusion; vertical slice)
  - [~] 3f. Review round 3 (final go/no-go, 1 fresh reviewer)
- [x] 4. Phase A infrastructure: protocol.yaml, splits, evaluate.py, registry, tests (57 pass), B0 smoke test (primary 0.0209 = prevalence)
  - [x] 4a. Toy generator with known answers (oracle AUPRC all .66; feature-informed oracle .38; propensity-only .22)
  - [x] 4b. LITERATURE.md fixes, DATA.md complete register + config/data_allowlist.yaml
  - [x] 4c. Independent review of Phase A → reviews/verify_phaseA/REPORT.md: all numbers re-derived exactly; 4/11 planted mutants uncaught → test hardening
- [x] 5a. Tier-1 label-free features: idmap, PPI (PMID blocklist removes 40% of pombe physical rows), SMF/PHEN, CER (+YGOB ohnologs)
- [x] 5b. Tier-2 features: EXPR, KOPROT, LOC, SEQ (+ESM 650M PCs/cosine), CONS
- [x] 6. Baselines (docs/baselines.md; dirty-tree runs, to be re-run clean)
  - [x] B0 random 0.0209 · B1 additive logit 0.103 (C1 .207/C2 .080/C3 .022) · B4 logistic MF 0.108 (C1 .232) · B2 Costanzo ε 0.026 · B3 Costanzo profile 0.025
  - [!] DIAG oracle propensity (true held-out hit rates) = 0.220 (C3 .235) → predicting gene propensity for unseen genes is the dominant lever
  - [x] DIAG NBR cross-source mean = 0.125 (C2 .117 > B1 .080); C3 still ≈ chance for all single scores
- [x] 7. M1 LightGBM on label-free features (src/pairs.py; 358 features/9 groups): primary 0.145 ± 0.011 (C1 .214/C2 .134/C3 .088); M1−B1 +0.041 [0.031, 0.051]; gene-permuted control 0.071 (C3 → prevalence) ✓; ablation: PHEN Δ+0.022, CER +0.004, others ≈0; universe scoring 7.14M pairs in 80 s
  - [!] M1 signal is mostly gene-level "sickness" (phenotype breadth, cerevisiae-ortholog negative rate, ortholog fitness); pair-similarity features add little
- [x] 8. M2 (docs/m2.md): nested stacking = label-free + RYAN-fold + NBR + gene-propensity (GP) model. **Primary 0.1700 ± 0.012** (C1 .251/C2 .159/C3 .095); M2−M1 +0.025 [0.019, 0.033], seed-7 +0.031 → **adopted as incumbent**. Clean reruns at 3a9bf62 reproduce B1 .1033 / B4 .1079 / M1 .1447 exactly. Runtime: nested features ~130 s/fold, train 276 s (15 models), predict 10 s.
  - [!] NBR is the largest Ryan-derived family (+0.019 [0.013, 0.026]); RYAN+GP +0.007; GP alone +0.003 (n.s.; substitutes for β/MF/NBR); init_score 0 → not used
  - [-] GP toy check failed its pre-stated bar (unseen-gene Spearman 0.563 < 0.6; B1 β itself only 0.55 vs truth) — reported, bar not moved
  - [~] Independent verification of M2 (nesting leak canary, label-permutation control, re-derived metric)
- [~] 10. M2+ rung (agent, branch m2plus): XSP merge + ablation, 5-seed bagging, E-TARGET (binary vs continuous S2), C3-focused blend
- [~] 11. Deliverable tooling (agent, branch deliverable): predict_all_pairs.py (universe scoring, OOF scores for measured pairs, flags), lockbox_eval.py + prereg template (one-shot, access log), linkage diagnostic
- [~] 9. XSP cross-species gene/pair transfer (docs/features_xsp.md; staged): cer gene-propensity CV ρ 0.391, transfer ρ 0.246; merge + ablation on M2 pending
- [ ] 6. Feature-based models
- [ ] 7. Advanced models / cross-species transfer
- [ ] 8. Final evaluation on held-out test → REPORT.md

## Idea backlog (ranked; stop rule: 3 consecutive failed rungs and no supported idea left)
1. [x] Explicit gene-propensity model β̂ (in M2; own increment n.s.)
2. [ ] Seed bagging (5+ seeds) of best model — cheap, +0.016 seen for M1
3. [~] Cross-species gene-propensity transfer: learn cerevisiae gene GI degree from cerevisiae ESM/PPI (Costanzo labels), apply to pombe genes (esp. without orthologs) → gene feature (M4-as-feature)
4. [ ] C3-focused low-capacity model (monotone constraints on sickness/propensity features)
5. [ ] Rank-average ensemble (M5) of M2 + MF + NBR-mean; E-TARGET (Huber on S2)
6. [ ] M3 neural inductive pair model (one attempt)

## Decision log
- 2026-10-28: Milestone M2 retrospective adopted: stop adding low-signal feature groups and per-source diagnostics; prefer bagging over tuning; cap reviews to one targeted reviewer per milestone (LESSONS.md pruned to 12).
- 2026-10-28: B7 decision — the 8 universe genes with only DAmP data in Ryan (pot1, sod1, bud32, rhb1, sep1, hsk1, ten1, cwf20) stay "seen" for class assignment (their Ryan data exist; weight impact ≤0.0016; protocol stays frozen). Their DAmP-only pairs are excluded from eval as designed. DESIGN numbers corrected in-place: 22 unknown-viability genes; 16,557 DAmP-only universe pairs.
- 2026-10-28: **Protocol FROZEN**: config/protocol.yaml sha256 73edd56db38e6d82007b2ece6e66beabae0fb6bce126c6e26f834cf5686bc230. Class weights (unmeasured universe) C1/C2/C3 = 0.2365/0.5698/0.1937. B0 primary = 0.0209 (= weighted prevalence). Accepted Phase A deviations: 22 (not 21) unknown-viability genes; C1 hold-outs drawn from eval-eligible pairs only; comparisons logged in results/comparisons.tsv.
- 2026-10-28: **Lockbox drawn** (scripts/draw_lockbox.py; 480/2404 S2 genes; seed 20261028; config/lockbox_genes.txt sha256 77fb02e1a5d688884e0a282c49c9f100840ee0f1988b74247abee150228d3457). From now on all analyses use dev genes only. Why now: R2 external PI — prevent further exposure.
- 2026-10-28: Lockbox exposure note — during R1 review two reviewers ran exploratory probes on ALL Ryan pairs (random 10% pair hold-out; per-gene propensity variance demo). No model/feature selection was made from them; lockbox genes not yet drawn. Logged for transparency.
- 2026-10-28: DESIGN v2 — universe = viable + condition-dependent (+ Ryan deletion strains of unknown viability); primary metric = pooled AUPRC weighted to unmeasured-universe class mix; Koch 2012 SMF rejected (E-MAP-derived).
- 2026-10-28: Labels = Ryan Dataset S2 (authors' per-pair averaged S) < −2.3; raw-scale labels only as sensitivity analysis. Why: brief specifies one quantitative value per pair; S2 is the canonical per-pair matrix.
- 2026-10-28: Circularity calls — BioGRID physical protein–protein systems kept (incl. PomBase-curated LTP rows); RNA/co-localisation systems dropped; FYPO excluded; Rodríguez-López NET-FF/GO sheets excluded; COXPRESdb expression matrices only.
- 2026-10-28: Commit policy — aggregate statistics and model outputs only; never Ryan pair-level scores (Ryan licence: no redistribution).
- 2026-10-28: Use a project-local venv; pin with `pip freeze`. Rationale: bare system Python, no conda/brew.
