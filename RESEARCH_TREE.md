# Research tree — S. pombe synthetic-lethality prediction

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (latest first)

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
  - [~] 3e. Review round 2 (5 reviewers)
- [ ] 4. Data pipeline, gene ID mapping, labels, splits (+ verification)
- [ ] 5. Baselines
- [ ] 6. Feature-based models
- [ ] 7. Advanced models / cross-species transfer
- [ ] 8. Final evaluation on held-out test → REPORT.md

## Decision log
- 2026-10-28: Lockbox exposure note — during R1 review two reviewers ran exploratory probes on ALL Ryan pairs (random 10% pair hold-out; per-gene propensity variance demo). No model/feature selection was made from them; lockbox genes not yet drawn. Logged for transparency.
- 2026-10-28: DESIGN v2 — universe = viable + condition-dependent (+ Ryan deletion strains of unknown viability); primary metric = pooled AUPRC weighted to unmeasured-universe class mix; Koch 2012 SMF rejected (E-MAP-derived).
- 2026-10-28: Labels = Ryan Dataset S2 (authors' per-pair averaged S) < −2.3; raw-scale labels only as sensitivity analysis. Why: brief specifies one quantitative value per pair; S2 is the canonical per-pair matrix.
- 2026-10-28: Circularity calls — BioGRID physical protein–protein systems kept (incl. PomBase-curated LTP rows); RNA/co-localisation systems dropped; FYPO excluded; Rodríguez-López NET-FF/GO sheets excluded; COXPRESdb expression matrices only.
- 2026-10-28: Commit policy — aggregate statistics and model outputs only; never Ryan pair-level scores (Ryan licence: no redistribution).
- 2026-10-28: Use a project-local venv; pin with `pip freeze`. Rationale: bare system Python, no conda/brew.
