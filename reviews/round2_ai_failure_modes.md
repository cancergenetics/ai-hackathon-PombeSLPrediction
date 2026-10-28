# Review round 2 — AI-agent failure modes (DESIGN v2)

**Verdict:** round 1 was mostly adopted correctly, with three gaps:
- two leaks are only half closed;
- the safeguards are heavier than agents will sustain;
- the new nested out-of-fold (OOF) / masked-mix builder has no functional test.

P1 before Phase B; P2 in Phase A; P3 optional.

## A. Round-1 gaps
1. **P1. Lint all full-label reads.** §2 lints only `labels_lockbox.parquet`, but `data/processed/ryan_pairs.parquet` holds every lockbox label.
   - Allowed readers: `make_splits.py`, `lockbox_eval.py`, `verify_core_data.py`.
   - Also ban `results/eda/` and `docs/data_inventory.md` paths in code. The §4 allowlist checks only *declared* sources; an undeclared read of the committed `ryan_degree_s2_neg.tsv` is the realistic leak.
2. **P1. Draw the lockbox first.** Phase A order: ID mapper → `make_splits.py` → commit the split hash → only then any other label reads (LESSONS #4).
   - Regenerate the EDA on dev genes.
   - Restore the AI6 disclosure: the B2 cut-off and feature priorities saw all-pair statistics.
3. **P2. Fix the existing −2.3 literals** (`build_ryan_pairs.py`, `verify_core_data.py`): read the threshold from config, or whitelist these files by name. Otherwise the lint fails on day one and gets disabled.
4. **P3.** AI11 is missing from Appendix A; record its disposition.

## B. Proportionality: minimal set
5. **P1. Logging as a side effect.**
   - `evaluate.py` loads `protocol.yaml`, stamps the protocol, split and feature hashes, and appends the registry row itself.
   - It is the only route to a metric.
6. **P2. Replace "refuse a dirty tree"** with recording the commit plus `sha256(git diff)`.
   - Dirty runs are flagged and cannot be adopted or shortlisted.
   - Each parallel coding agent gets its own git worktree.
   - Why: in a shared checkout, hard refusal gets bypassed.
7. **P2. Lighten three items:**
   - the task template becomes a prompt checklist, not committed files;
   - `check_claims.py` becomes a single regex check (run_id exists, value matches);
   - the fresh-clone `run_pipeline.sh` runs only before Phase F.

   Keep the rest.

## C. New failure modes in v2
8. **P1. `FoldContext` API.**
   - Label-using builders get labels only via `ctx.train_labels()`, which excludes C1_k and lockbox C1 pairs.
   - Feature caches are keyed by (split hash, outer fold, inner fold, builder version, label hash). `evaluate.py` rejects mismatched keys.
   - Why: mis-keyed caches are the likeliest real leak.
9. **P1. Functional leakage tests.** Run them on toy data in CI, and once on real outer fold 0.
   - a. *Eval-label invariance:* flip all labels of pairs touching fold k. Eval-row features and predictions must stay bit-identical.
   - b. *Own-label invariance:* flip some training labels. Those rows' own features must not change.
   - c. *Null canary:* with random labels, eval AUPRC must be within the CI of prevalence for B1, M2 and M5. Also, train-row feature→label AP must not exceed eval-row AP (non-OOF encodings inflate the train side).
   - d. *Mutants that must be caught:*
     - a non-OOF target encoding;
     - NBR using the eval gene's own labels;
     - C1_k left in the MF fit;
     - a stacker fit on outer-fold predictions;
     - fold j's cache reused for fold k.
10. **P2. Freeze the OOF structure.** Put the inner-fold count, mask rates and NBR k grid in `protocol.yaml` and stamp them in the registry. Under compute pressure, agents silently shrink or skip OOF.
11. **P2. Check train/eval feature shift.** Report per-class KS statistics between training-row features (OOF, masked) and eval-row features. A large shift means a mis-specified mask mix.
12. **P3.** Give the "permute gene identities → M1 ≤ B1" test a pre-stated pass criterion, or drop it.
