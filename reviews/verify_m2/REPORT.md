# Independent verification of M2 (2026-10-28)

The verifier did not build M2. Verdict: **all checks PASS; no leak, no bug.** The scripts and their aggregate outputs are in this directory.

1. **Nesting leak canary — PASS** (`check1_canary.py`, outer fold 0 and inner folds 0 and 3).
   - Method: every label/S on pairs touching held-out genes or in the C1 hold-out was corrupted.
   - Result: all 467 RYAN/NBR/GP columns on the eval rows and stacker rows stayed bit-identical, except `mf_score`.
   - Why `mf_score` moves (`check1b`): torch is non-deterministic on CPU with more than one thread, by ~1e-6 within a thread count. With 1 thread the outputs are bit-identical.
   - Positive controls: both planted leaks (C1 eval pairs in train; H_j in the inner train) were detected, changing 95 columns.
2. **Label-permutation control — PASS** (`check2_permutation.py`).
   - Method: within-train row shuffle, 4 fold×seed runs.
   - Result: primary 0.0199–0.0252, mean 0.0227, against a weighted prevalence of 0.0211.
   - Clean fold 0 was reproduced from cached features: 0.1660, same as the registered value.
3. **Metric re-derivation — PASS** (`check3_metric.py`, the verifier's own AP and weights).
   - M2 0.17001 and M1 0.14466 match the registry.
   - Δ +0.0254. The verifier's own 200-replicate bootstrap gives CI [0.017, 0.033].
4. **Split hygiene — PASS** (`check4_hygiene.py`).
   - No lockbox genes anywhere.
   - Class assignment and the DAmP-only flag were re-derived independently: 0 mismatches.
   - y equals S2 < protocol threshold on all 790,237 dev rows.
5. **Notes (not bugs):**
   - `mf_score` is not bit-reproducible with more than one torch thread.
   - Stacker early stopping is slightly optimistic, because inner-fold validation rows can sit in other inner folds' training sets. DESIGN allows this, and it affects only the stopping round and the inner validation AP, not the outer metric.
   - `run_m2.py` skips the allowlist check for non-default data roots. The verifier ran it on the frozen store: 0 violations.
