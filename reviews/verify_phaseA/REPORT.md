# Phase A independent verification (saved by PI from the verifier's hand-back; verifier could not write files)

Verdict: no correctness bugs. All re-derived numbers match (own code, no src/ imports except public evaluate() for comparison).

| # | Check | Result |
|---|---|---|
| 1 | Universe gene set | PASS — 3,779 = 3,779 identical (22 unknown-viability genes, not 21) |
| 2 | Class weights | PASS — 0.2365/0.5698/0.1937 (all-pairs 0.3541/0.4820/0.1639) |
| 3 | Per-pair DAmP-only exclusion | PASS — 16,557 universe DAmP-only pairs, 0 mismatches |
| 4 | Fold split (20261028 and 7) | PASS — all counts equal splits_summary; lockbox disjoint |
| 5 | Primary metric | PASS — B0 0.020882855364 identical; degree score 0.0906 identical |
| 6 | Bootstrap | PASS — shared Poisson(1) gene bootstrap of exact statistic, identical to 1e-10 |
| 7 | Test tautology | PARTIAL — 7/11 planted mutants caught |
| 8 | Inner splits | PASS |

Gaps: B1 eval-set completeness untested (dropping half of C3 or moving C1 hold-out into train passes); B2 DAmP-only exclusion check circular; B3 label-access lint is substring-based and evadable (evaluate.prepare/_labels_for give labels); B4 class weights not pinned by a test; B5 random-baseline tolerance too loose; B6 DESIGN numbers 21→22 genes, ≈18.3k→16,557 pairs; B7 8 universe genes with DAmP-only data count as "seen" (weights change ≤0.0016).
Verifier scripts: reviews/verify_phaseA/v*.py, mutplug.py, mut/.
