# LESSONS — process rules for running this project efficiently

One line per rule, each with evidence. Pruned at each milestone.

**Last pruned:** 2026-10-29, milestone M4 (final), after a fresh-agent fact-check of REPORT.md and a retrospective.

1. **Diagnose before modelling.** Run cheap oracle and permutation diagnostics first.
   - *Evidence:* the propensity oracle scored 0.22 against B1's 0.10, which redirected M2 to gene propensity. Permutation controls (M1 0.071, M2 0.023) confirmed the signal is real.
2. **Probe before building a feature family.** Run a cheap 1-fold ablation or importance check on cached or staged features before writing a builder, docs and a merge for a new family.
   - *Evidence:* XSP cost a pipeline, a 61-entry manifest fix and a default-group leak, for +0.0002. The tier-2 groups added about 0.01 in total.
3. **Change the target before the features.** For noisy quantitative labels, try a continuous or robust training objective early.
   - *Evidence:* the Huber objective on S2 gave +0.009, the only gain of the M2+ rung. XSP, bagging and the C3 blend gave ≤0.003.
4. **Adopt one step at a time.** Each change must pass the adoption bar alone; never let a bundled Δ carry a component that fails.
   - *Evidence:* "bag5 + Huber" +0.011 hid bagging's +0.002.
5. **Draw the lockbox before any analysis touches labels.**
   - *Evidence:* two round-1 reviewer probes used all pairs before the lockbox existed.
6. **Independent verifiers write their own code: a leak canary with a planted positive control, mutants, and a metric re-derivation.**
   - *Evidence:* 4 of 11 mutants escaped the self-written tests. The M2 canary plus planted leaks gave a clear PASS.
7. **Verify the incumbent before shipping it.**
   - *Evidence:* the deliverable was built and run on M2, then the incumbent changed and the universe run had to be redone.
8. **Commit before adoptable runs, and run them from a worktree plus a frozen data snapshot. Scope dirty checks to code.**
   - *Evidence:* the first B1/B4/M1 runs were all dirty. The worktree + data/frozen_m2 runs reproduced bit-for-bit across agents.
9. **Pin model inputs explicitly in config. Never let a default "all groups in the store" decide them.**
   - *Evidence:* merging XSP silently changed the defaults of run_m1/run_m2.
10. **Cache nested builds and ablate jointly. Stop a line of ideas once the stop rule fires.**
    - *Evidence:*
      - 10 per-source NBR diagnostics took ~15 min each, and 9 single-group ablations took 36 min.
      - C3 stayed at 0.095 across 4 ideas.
11. **Keep the research tree current before launching the next rung. Before writing the report, sweep the decision log** for exposures, failed pre-stated checks and licence limits, and carry them into Limitations.
    - *Evidence:*
      - The M3 retrospective found a stale "Next: M2" summary.
      - The first REPORT draft omitted the round-1 lockbox exposure, the failed GP toy check and Ryan's no-redistribution licence.
12. **Take report numbers from the shipped run's summary JSON, and write the report after the final run, not before.**
    - *Evidence:* the drafted REPORT carried the dev-trained run's numbers (17% C3 in the top 100k; 31 min) instead of the shipped run's (5%; 24 min).
13. **Dry-run one-shot scripts against the committed config in the real repo layout, using a toy data root.**
    - *Evidence:* the toy tests put the prereg at the repo root. The real prereg in config/ hit a path bug, causing one refused lockbox attempt and a prereg re-commit.
14. **Comparator runs must be clean too.**
    - *Evidence:* both seed-7 M2r_huber comparator runs were dirty, because of untracked files of a parallel agent. A clean rerun was needed before the report.
15. **Give parallel agents disjoint file ownership, a shared interface spec, and their own worktree. Ask for report text in the hand-back.**
    - *Evidence:* there were zero merge conflicts over 4 parallel branches. Some subagents can't write report files.
16. **Keep design review proportionate:** after the design phase, use one targeted reviewer per milestone.
    - *Evidence:* round 3 gave only text fixes. About 1h of the first 3h went to design.
17. **Environment pins.**
    - LightGBM needs the libomp rpath fix in `scripts/setup_env.sh`.
    - Pin `pytest==8.3.5`.
    - torch MF is only bit-reproducible at 1 thread.
