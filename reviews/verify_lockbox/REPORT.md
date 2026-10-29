# Independent verification of the lockbox evaluation (2026-10-29)

The verifier did not run the lockbox. Verdict: **all checks PASS; no leakage; no blocking bugs.**

1. **Re-derivation — PASS** (`rederive.py`)
   - Every point estimate was reproduced to within 3e-15: primary and per-class AP for 5 models, and 4 Δs.
   - Own bootstrap, 1,000 replicates with a different seed:
     - M2r_huber 0.2113 [0.182, 0.245].
     - Contrasts, all of which reject after the verifier's own Holm adjustment:
       - vs B1 +0.105 [0.080, 0.133]
       - vs B4 +0.101 [0.075, 0.129]
       - vs M1 +0.049 [0.032, 0.068]
       - vs M2 +0.010 [0.003, 0.017], p = 0.002
2. **Eval-set hygiene — PASS** (`hygiene.py`)
   - The 456,107 eval pairs are exactly the eligible lockbox pairs: C1 69,169 / C2 344,186 / C3 42,752.
   - 61,445 DAmP-only pairs were excluded.
   - The training set has 0 lockbox genes, 0 lockbox-C1 pairs and 0 eval pairs.
   - Training hashes were re-implemented and match.
3. **Prereg compliance — PASS**
   - No code diff between code_commit and the run commit.
   - models sha, seeds [0,1,2] and threads 4 all match.
   - The access log has exactly one started and one completed run.
   - The earlier refused attempt raised before the YAML load and before any read of the lockbox labels.
4. **Lockbox > dev gap** (`gap.py`, `hub_probe.py`, `m1_gene_curve.py`)
   - The gap is mostly sampling noise plus a small training-data effect; there is no sign of leakage.
     - The gap is not significant: z ≈ 1.5 for M2r_huber.
     - Prevalence, stratum composition, mean hit rate, degree and feature coverage of the lockbox genes all match dev.
     - The gap sits in C2/C3.
     - Removing 5 lockbox genes moves C2 from 0.210 to 0.183.
     - An M1 learning-curve probe explains only a small part of the gap through extra training data.
   - The baselines don't move because their C2/C3 scores ignore the unseen gene.

**Minor issues** (none affect the numbers):
- p = 0.0 should be reported as < 1/n_boot.
- B4 records no seeds in the result JSON.
- The prereg does not pin the thread count.
