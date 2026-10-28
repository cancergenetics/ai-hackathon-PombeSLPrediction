# LESSONS — process rules for running this project efficiently

One line per rule, each with evidence. Pruned at each milestone.

**Last pruned:** 2026-10-28, milestone M2, after a fresh-agent retrospective ("stop doing / worked well").

1. **Diagnose before modelling.** Run cheap oracle and permutation diagnostics first.
   - *Evidence:* the propensity oracle scored 0.22 against B1's 0.10, which redirected M2 to gene propensity. The gene-permuted M1 fell to 0.071, with C3 at prevalence.
2. **Draw the lockbox before any analysis touches labels.**
   - *Evidence:* two round-1 reviewer probes used all pairs before the lockbox existed.
3. **Independent verifiers write their own code and inject mutants.**
   - *Evidence:* the re-derivation matched, but 4 of 11 planted mutants escaped the self-written tests.
4. **Commit before any run you intend to adopt.**
   - *Evidence:* all first B1/B4/NBR/M1 runs were flagged dirty and had to be re-run.
5. **Scope dirty and provenance checks to code, not outputs.**
   - *Evidence:* every run writes `results/`, so whole-tree checks made consecutive clean runs impossible.
6. **Keep design review proportionate.**
   - *Evidence:* rounds 1–2 changed the design substantially, but round 3 gave text-only fixes. About 1h of the first 3h went to design.
   - *Rule:* after the design phase, use one targeted reviewer per milestone, aimed at code and results.
7. **Cache expensive nested feature builds and reuse them across diagnostics and ablations.**
   - *Evidence:* 10 per-source NBR diagnostic runs took about 15 min each, and only the cross-source mean mattered.
8. **Ablate jointly and only where a group could matter; stop adding low-signal feature groups.**
   - *Evidence:* 9 single-group ablations cost 36 min, and only PHEN had a CI above 0. Tier-2 groups added about 0.01 in total.
9. **Prefer seed bagging over hyperparameter tuning for GBDTs here.**
   - *Evidence:* the config spread was 0.0007, while 3-seed averaging gained +0.016.
10. **Give parallel agents disjoint file ownership and a shared interface spec.**
    - *Evidence:* five concurrent agents produced zero merge conflicts (docs/INTERFACES.md).
11. **Ask subagents to return report text in their hand-back.**
    - *Evidence:* some subagents cannot write report files (the Phase A verifier).
12. **Environment pins.**
    - LightGBM/XGBoost need the libomp rpath fix in `scripts/setup_env.sh`.
    - Pin `pytest==8.3.5`; 9.2.0 crashes with an INTERNALERROR.
