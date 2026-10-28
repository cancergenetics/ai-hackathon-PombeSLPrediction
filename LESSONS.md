# LESSONS — process rules for running this project efficiently

(One line per rule, with evidence. Pruned at each milestone.)

1. On this Mac, LightGBM/XGBoost wheels fail to import without libomp; fix by copying sklearn's bundled libomp + `install_name_tool -add_rpath @loader_path` (scripts/setup_env.sh). Evidence: M0 import errors, ~5 min lost.
2. Run design reviewers as independent parallel lenses: 4–7 of 7 converged unprompted on the same top issue (out-of-fold Ryan features / similarity transfer), which gives high confidence cheaply (~5 min wall clock). Evidence: round-1 reviews.
3. Ask reviewers to quantify claims with read-only scripts; this caught a wrong number in the design (C3 lockbox positives 805, not 960) and a circular noise ceiling. Evidence: reviews/round1_statistics.md.
4. Reviewers' exploratory probes must avoid touching future lockbox data — draw the lockbox first, before anyone runs analyses. Evidence: two R1 probes used all pairs.
5. pytest 9.2.0 + pluggy 1.7.0 crashes with INTERNALERROR NodeId._str_cache; pin pytest==8.3.5. Evidence: toy agent lost time running tests by hand.
6. Scope provenance/dirty checks to code, not outputs: a whole-tree `git status` made consecutive clean runs impossible because every run writes results/. Evidence: all baseline/M1 runs flagged dirty; fixed in src/evaluate.git_info.
7. Run a cheap oracle diagnostic before building models: the true-propensity oracle (0.22 vs B1 0.10) showed gene-level propensity for unseen genes is the dominant lever, and redirected M2 toward an explicit gene-propensity model. Evidence: docs/baselines.md.
8. Independent verifiers must write their own code: re-derivation matched every number, but injecting mutants showed 4/11 test blind spots that self-written tests missed. Evidence: reviews/verify_phaseA/REPORT.md.
9. Some subagents cannot write report files (harness block); ask for the report content in the hand-back so the PI can save it. Evidence: Phase A verifier.
10. Give each parallel agent disjoint file ownership and a shared interface spec (docs/INTERFACES.md); 5 agents ran concurrently with zero merge conflicts. Evidence: Phase A/B.
