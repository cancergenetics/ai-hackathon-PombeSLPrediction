# LESSONS — process rules for running this project efficiently

(One line per rule, with evidence. Pruned at each milestone.)

1. On this Mac, LightGBM/XGBoost wheels fail to import without libomp; fix by copying sklearn's bundled libomp + `install_name_tool -add_rpath @loader_path` (scripts/setup_env.sh). Evidence: M0 import errors, ~5 min lost.
2. Run design reviewers as independent parallel lenses: 4–7 of 7 converged unprompted on the same top issue (out-of-fold Ryan features / similarity transfer), which gives high confidence cheaply (~5 min wall clock). Evidence: round-1 reviews.
3. Ask reviewers to quantify claims with read-only scripts; this caught a wrong number in the design (C3 lockbox positives 805, not 960) and a circular noise ceiling. Evidence: reviews/round1_statistics.md.
4. Reviewers' exploratory probes must avoid touching future lockbox data — draw the lockbox first, before anyone runs analyses. Evidence: two R1 probes used all pairs.
5. pytest 9.2.0 + pluggy 1.7.0 crashes with INTERNALERROR NodeId._str_cache; pin pytest==8.3.5. Evidence: toy agent lost time running tests by hand.
