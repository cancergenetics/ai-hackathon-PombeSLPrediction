# Review round 1 — AI-agent failure modes (DESIGN.md v1)

## 1. Verdict

**Sound scientifically, but the safeguards are intentions, not mechanisms.** §8 lists the right risks, but each depends on agents behaving well. Nothing enforces them by construction or detects when they are broken. Problems that already exist:

- −2.3 is typed into three scripts: `build_ryan_pairs.py`, `eda_core_data.py` and `verify_core_data.py`.
- The committed file `results/eda/ryan_degree_s2_neg.tsv` holds per-gene hit rates for **all** genes, including future lockbox genes. It is a ready-made, leaky propensity feature.
- The §2 split-seed-7 check would re-draw the lockbox.

Revise before implementation. Items 1–9 are must-fix.

## 2. Concrete changes

**1. Frozen protocol file with checksum — MUST.**
- *What:* add `config/protocol.yaml` with:
  - the label column and threshold;
  - the evaluation subset and universe;
  - the C1/C2/C3 weights (computed once by script, then frozen as literals);
  - the metric, seeds, split seed, bootstrap count and decision rule.
- Record its sha256 in DESIGN §0. All code reads these values only from this file, and a lint test fails on the literals `-2.3` or `0.37/0.48/0.15` anywhere else.
- A protocol change requires a DESIGN version bump, a decision-log entry and re-run baselines. Registry rows made under different hashes are never compared.
- *Why:* PROMPT forbids changing metrics or labels. Agents "fix" thresholds locally without noticing, and the checksum makes that visible.

**2. Single evaluation entry point — MUST.**
- *What:* `src/evaluate.py` is the only code that joins predictions to labels and computes AUPRC per class, the bootstrap and the selection score.
- Models write predictions for a pair-index file and receive labels for training pairs only.
- *Why:* otherwise each agent re-implements the metric slightly differently (AUROC vs AP, inviable pairs left in, class assignment from the wrong fold), and runs become incomparable.

**3. Physical lockbox separation plus pre-registration — MUST.**
- *What:* `scripts/make_splits.py` writes `splits/split_v1.json` (gene lists, C1 hold-out pair IDs, a hash).
- `src/data.load_dev()` returns labels with lockbox genes and pairs removed. It is the only sanctioned label access, and a lint test fails if any other module reads `ryan_pairs.parquet`.
- `scripts/lockbox_eval.py` is run by the PI only. It requires a committed `lockbox/shortlist.yaml` holding model IDs and the sha256 of each prediction file, all produced before labels are joined.
- It appends to `lockbox/access_log.tsv`, and a re-run needs `--reason`, which REPORT.md discloses.
- *Why:* every agent can currently read the full parquet, so the v1 "logging script" is optional. Separation prevents accidental peeking; hashes and the log detect deliberate peeking.

**4. Fix the split-sensitivity check — MUST.**
- *What:* run the seed-7 check by re-partitioning **dev genes only**, or run it after the one lockbox evaluation, labelled post-hoc.
- *Why:* a new seed-7 test set trains on old lockbox genes, which contaminates the lockbox.

**5. Results registry; claims must cite it — MUST.**
- *What:* an append-only `results/registry.tsv`. Each row records:
  - run_id and git commit (runs refuse to start on a dirty tree);
  - protocol, split and feature-manifest hashes;
  - seed and fold;
  - per-class AUPRC and selection score;
  - wall/CPU time and thread count;
  - task ID and agent.
- Every attempt is logged, failures included, so the number of comparisons tried is visible.
- Every number in RESEARCH_TREE.md or REPORT.md cites a `[run_id]`, and `scripts/check_claims.py` verifies it against the registry.
- *Why:* agents misremember or carry over stale numbers across context resets, and logging only successes hides forking paths.

**6. Quarantine existing full-data label derivatives — MUST.**
- *What:* add `results/eda/*` and `docs/data_inventory.md` to the forbidden-feature-input manifest test (§4 iii).
- Future label-using EDA runs on dev genes only.
- Disclose that some choices (the B2 cut-off, feature priorities) saw all-pair statistics.
- *Why:* a committed table of per-gene hit rates that covers the test genes is exactly the shortcut an agent building B1 or target encodings will grab.

**7. Fold-scoped API for every supervised step — MUST.**
- *What:* every function that learns from labels takes explicit `train_pair_ids` and asserts they are disjoint from the evaluation IDs. This covers:
  - RYAN-fold features;
  - B1's unseen-gene hit-rate regressor;
  - target-encoding priors;
  - M4's prevalence-matched Costanzo threshold;
  - the HP search;
  - M5 weights and the stacker (which need nested out-of-fold predictions).
- C1 hold-out pairs are excluded from training degree.
- *Why:* the likeliest silent leak is a "global" statistic or stacking layer fitted on all dev data, not the split itself.

**8. Tests must be shown to fail — MUST.**
- *What:* each split or leakage test ships with a **mutant** it must catch:
  - a feature builder that peeks at evaluation labels;
  - a split with one overlapping gene;
  - a metric with swapped arguments.
- CI asserts that every mutant fails.
- The toy positive control pre-states its expected AUPRC (an oracle on the planted probabilities) and a tolerance. A different agent from the model author writes the toy generator.
- *Why:* "scrambled eval labels leave predictions unchanged" passes trivially if the builder never receives labels, even when it reads them from disk. `verify_core_data.py` already tests the parser it imports from the code under test.

**9. Independent re-derivation, not re-running — MUST.**
- *What:* for the key numbers, the reviewer writes a minimal independent implementation from raw files and the spec, without importing project code, and compares. The key numbers are class prevalences, split class counts, the best dev score and the lockbox AUPRCs.
- A clean-checkout re-run checks reproducibility, not correctness.
- *Why:* re-running a script reproduces its bugs.

**10. Never commit pair-level Ryan data — SHOULD.**
- *What:* per-pair prediction or evaluation files that contain S-scores or labels stay git-ignored. A pre-commit check enforces this, and only aggregates are committed.
- *Why:* the Elsevier licence forbids redistribution, and agents commit result folders wholesale.

**11. No hardcoded data or hand-typed numbers — SHOULD.**
- *What:* lint for literal systematic IDs (`SP[ABC]…`) outside test fixtures.
- Replace "2,4xx Ryan genes" and "≈333k C2 pairs" with a script-generated `docs/split_summary.md`.

**12. Task template — SHOULD.**
- *What:* each task is a committed `tasks/<id>.md` containing:
  - one hypothesis;
  - inputs with their hashes;
  - forbidden actions (lockbox, protocol edits, direct label reads);
  - deliverables (script, registry rows, a summary of 300 words or fewer citing run_ids, deviations, and "what I did not verify");
  - acceptance tests;
  - a wall-clock cap.

**13. Stopping rule and anti-tinkering — SHOULD.**
- *What:* keep an idea backlog in RESEARCH_TREE.md with the expected gain and cost of each idea. Branches are time-boxed and closed with run_ids.
- Stop when three consecutive ladder steps fail the §3 decision rule and no backlog item has literature or EDA support, or at the time limit.
- The HP budget counts across re-launches of the same idea.
- Keep a cumulative compute ledger in the tree.

**14. Runtime protocol — SHOULD.**
- *What:* a timing harness records feature build (one-off ESM-2 cost listed separately), per-fold training, and full 6.41M-pair scoring, with a fixed thread count. REPORT.md takes its runtimes from the registry.

**15. One-command rebuild — COULD.**
- *What:* `scripts/run_pipeline.sh` runs from raw data to the registry. A reviewer runs it on a fresh clone at each milestone and diffs the registry.
- Parallel agents use separate output directories.
