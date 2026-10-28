# M2+ rung: XSP merge, M2 ± XSP, seed bagging, E-TARGET (Huber), C3 blend

Status 2026-10-28. Branch `m2plus` (worktree `../psl_wt_m2plus`, `data/` = real dir of symlinks into the
main tree's `data/`). Code: `scripts/run_m2plus.py`, `scripts/features/merge_staged.py`, additive options in
`src/models/m2.py`. Rules for every step were pre-stated in `config/m2plus.yaml` and committed (bf7e469)
before any outer-fold result of this rung existed. `config/protocol.yaml` is untouched.

## Summary
- **New incumbent candidate: `M2_huber_bag5`** (registered as `M2r_bag5_huber`): M2 with the stacker
  objective switched from binary to **Huber regression on t = −min(S2, 0)** (δ = |label threshold|), 5 stacker
  seeds. **Primary 0.1814 ± 0.0128** (C1 0.277 / C2 0.168 / C3 0.095).
  - vs incumbent M2: **+0.0114 [+0.0069, +0.0153]**; all-pairs +0.0135, equal +0.0118. Seed-7: 0.1779 vs
    0.1703, **+0.0076** → **ADOPT** (DESIGN §3, main-ladder rule).
  - Attribution: Huber at the ladder seeds (`M2r_huber`) vs M2r **+0.0091 [+0.0048, +0.0133]** (passes alone).
    Bagging 3→5 seeds adds +0.002–0.003 with CI > 0 but below the 0.005 bar, both for binary and for Huber.
- **XSP (cross-species transfer features): no effect.** M2_XSP − M2r = +0.0002 [−0.0027, +0.0029]; C3 drops
  0.095 → 0.092. XSP carries ~1% of gain importance. Not adopted; permutation control not run (pre-stated: only
  if XSP helps).
- **C3-only low-capacity blend: no effect** (−0.0001 [−0.0007, +0.0005]; C3 0.0947 → 0.0945).
- **M2r reproduces the incumbent bit-for-bit** (Δ = 0.0000, CI [0, 0]) at the new commit with 6 instead of 8
  LightGBM threads, after the XSP merge (XSP excluded by group). The incumbent's feature cache is reused.
- The Huber gain is in **C1 (+0.026) and C2 (+0.009)**; C3 is unchanged (0.0948 → 0.0947). C3 remains the
  weak class. Neither cross-species propensity (XSP) nor a C3-specialised blend moved it.

## Step 1 — XSP merge, exact source ids
- `scripts/features/merge_staged.py`:
  - checks each staged file's sha256 against `manifest_xsp.json` and that every source is an exact
    use=feature id;
  - copies via temp file + atomic rename and re-hashes the copy. An existing destination with a different
    sha256 aborts the merge;
  - merges the entries under the store's manifest lock.
  - `--fix-aliases` rewrites short aliases in the live manifest (61 entry/column-level changes):
    `biogrid`→`biogrid_physical`, `pombase`→`pombase_orthologs` (build_cer/build_ppi) and
    `pombase_ids_viability` (genes.txt, META). It writes a timestamped manifest backup and changes no file or
    sha256. Record: `results/features/merge_staged_20261028T191630.json`.
- Builders (`build_cer.py`, `build_ppi.py`, `build_genes.py`) now write the exact ids.
- `tests/features/test_tier1_features.py`:
  - `ALLOWED_SOURCES` is read from `config/data_allowlist.yaml` (all use=feature ids) instead of being
    hardcoded. The PI's request is covered: the shared manifest's `biogrid_genetic_pmids.txt` → `biogrid_physical`
    passes.
  - META (genes.txt) may also cite universe-only ids.
  - Column-level sources are checked too.
  - The `biogrid_genetic_pmids` FORBIDDEN exemption is unchanged and still passes on the live store.
  - `src/pairs.source_violations` still *tolerates* aliases, because `data/frozen_m2` (verifier snapshot) keeps
    the old manifest.
- Registry: `src/pairs.load_spec` picks XSP up automatically (group `XSP`, 6 gene-pair + 1 pair column, family
  LF).
  - New additive option: `m2.gene_table(exclude_groups)`, `gene_table_hashes(exclude_groups)`,
    `m2_cache_name(..., exclude_groups, extra)`, `fold_tables_m2(..., exclude_groups, gt_transform, extra_key)`.
    With `exclude_groups=("XSP",)` the cache name equals the incumbent's (`m2feat_3d35f8c314e9`).
  - Tests: `tests/features/test_merge_staged.py` (merge, checksum/source/conflict rejection, alias fix,
    idempotence, live registry) and `tests/test_m2plus.py`.
- **Drift hazard:** any code that calls `PA.load_spec()` / `m2.gene_table()` without a group list now includes
  XSP. `config/models.yaml` (deliverable) already pins the 9 incumbent groups, so it is unaffected.
  `scripts/run_m1.py` / `run_m2.py` defaults are affected.

## Runs (dev CV, split seed 20261028 unless noted; all clean)
| model (registered name) | run_id | primary ± SE | C1 | C2 | C3 | all-pairs | equal | seeds | commit | train s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M2 (incumbent, docs/m2.md) | `M2__20261028T183713__024416` | 0.1700 ± 0.0119 | 0.2514 | 0.1592 | 0.0948 | 0.1831 | 0.1706 | 0–2 | 3a9bf62 | 276 (8 thr) | 948 |
| M2r (re-run, XSP excluded) | `M2r__20261028T194304__387281` | 0.1700 ± 0.0119 | 0.2514 | 0.1592 | 0.0948 | 0.1831 | 0.1706 | 0–2 | bf7e469 | 370 | 408 |
| M2_XSP | `M2_XSP__20261028T193546__f679df` | 0.1702 ± 0.0119 | 0.2522 | 0.1593 | 0.0918 | 0.1835 | 0.1707 | 0–2 | bf7e469 | 389 | 1056* |
| M2_XSP_LF (diagnostic) | `M2_XSP_LF__20261028T194928__91a8c7` | 0.1696 ± 0.0118 | 0.2500 | 0.1588 | 0.0942 | 0.1826 | 0.1702 | 0–2 | bf7e469 | 323 | 358 |
| M2r_bag5 | `M2r_bag5__20261028T200015__02b9a0` | 0.1728 ± 0.0121 | 0.2581 | 0.1604 | 0.0960 | 0.1866 | 0.1738 | 0–4 | bf7e469 | 564 | 610 |
| M2r_bag5_split7 | `M2r_bag5_split7__20261028T201104__948aa0` | 0.1727 ± 0.0118 | 0.2515 | 0.1634 | 0.0880 | 0.1855 | 0.1726 | 0–4 | bf7e469 | 576 | 613 |
| M2r_huber | `M2r_huber__20261028T205545__7b5e9e` | 0.1791 ± 0.0126 | 0.2721 | 0.1668 | 0.0936 | 0.1940 | 0.1798 | 0–2 | 592bdf1 | 486 | 529 |
| **M2r_bag5_huber** | `M2r_bag5_huber__20261028T202603__f8bde5` | **0.1814 ± 0.0128** | 0.2773 | 0.1681 | 0.0947 | 0.1967 | 0.1824 | 0–4 | bf7e469 | 812 | 869 |
| M2r_bag5_huber_split7 | `M2r_bag5_huber_split7__20261028T204628__38fb1c` | 0.1779 ± 0.0122 | 0.2626 | 0.1693 | 0.0877 | 0.1914 | 0.1773 | 0–4 | 592bdf1 | 793 | 841 |
| C3 blend of M2r_bag5_huber | `M2r_bag5_huber__20261028T202603__f8bde5_c3blend__20261028T205619__28b131` | 0.1813 | 0.2773 | 0.1681 | 0.0945 | | | 0–4 | 592bdf1 | — | ~120 |

- **Runtime.** Wall seconds on an Apple M4 with 6 LightGBM threads, other agents running.
  - Training = sum over 5 folds × seeds; prediction is 12–33 s.
  - \*M2_XSP total includes its one-off nested feature build: 629 s summed over folds, about 125 s/fold at
    2 threads, 3 folds in parallel, so ~260 s wall.
  - Huber training is ~1.4× binary, because it runs more rounds (median best_iter 234 vs 80) and uses a Python
    AP early-stopping metric.
  - The incumbent's split-7 M2 is `M2_split7__20261028T185814__ac04a4` (0.1703).
- The C3 blend's registered name contains the base run id, because it was invoked with a run id.

## Paired comparisons (`results/m2plus/comparisons.json`, also in results/comparisons.tsv)
| comparison (a − b) | Δ | 95% CI | all-pairs | equal | seed-7 Δ | verdict |
|---|---|---|---|---|---|---|
| M2r − M2 (reproducibility) | +0.0000 | [+0.0000, +0.0000] | 0 | 0 | | identical |
| M2_XSP − M2r | +0.0002 | [−0.0027, +0.0029] | +0.0003 | +0.0001 | | not adopted |
| M2_XSP_LF − M2r | −0.0004 | [−0.0035, +0.0023] | −0.0005 | −0.0004 | | not adopted |
| M2r_bag5 − M2r | +0.0028 | [+0.0011, +0.0045] | +0.0034 | +0.0032 | +0.0024 | not adopted (Δ < 0.005) |
| M2r_bag5_huber − M2r_bag5 (E-TARGET at 5 seeds) | +0.0086 | [+0.0049, +0.0121] | +0.0101 | +0.0086 | +0.0052† | ADOPT |
| M2r_huber − M2r (E-TARGET at ladder seeds) | +0.0091 | [+0.0048, +0.0133] | +0.0109 | +0.0092 | | ADOPT |
| **M2r_bag5_huber − M2 (incumbent)** | **+0.0114** | **[+0.0069, +0.0153]** | +0.0135 | +0.0118 | **+0.0076** | **ADOPT** |
| M2r_bag5_huber − M2r_huber (bagging under Huber) | +0.0022 | [+0.0009, +0.0034] | +0.0027 | +0.0026 | | not adopted (Δ < 0.005) |
| C3 blend − M2r_bag5_huber | −0.0001 | [−0.0007, +0.0005] | −0.0000 | −0.0001 | | not adopted |

† Seed-7 Huber vs binary at 5 seeds, as its own paired comparison: +0.0052 [+0.0015, +0.0088].

## Step 2 — XSP ablation
- Candidate (pre-stated): **M2_XSP** = XSP in the LF pair features *and* the GP model's gene inputs. M2_XSP_LF
  (pair features only) is a diagnostic.
- XSP takes 0.9–1.0% of gain importance in either variant. M2_XSP shifts importance from LF to GP (GP 0.37 →
  0.41), i.e. XSP enters mostly via β̂, but the pooled metric does not move. C3 falls slightly in both variants.
- Reading: on pombe, XSP's gene propensity (transfer ρ 0.25 on cerevisiae orthologs) is redundant with
  PHEN/CER/NBR/GP. For the ~1,100 ortholog-less genes it is too weak to change C3 ranking.
- Gene-permuted control: not run. The pre-stated condition was "only if XSP helps". The code path
  (`M2_XSPperm`) exists and is tested.

## Step 3 — seed bagging
- 5 seeds (protocol `seeds.all`; seed s early-stops on inner fold s mod 5) vs 3.
- Δ +0.0028 [+0.0011, +0.0045]; seed-7 +0.0024. The CI excludes 0 but the gain is below 0.005, so it is not
  adopted on its own.
- This is much smaller than M1's +0.016, because M2 already averages 3 differently-validated models.

## Step 4 — E-TARGET
- Pre-stated single candidate (DESIGN §5 allows "binary vs Huber only"):
  - LightGBM `objective=huber`, `alpha` = |protocol threshold|;
  - target −min(S2, 0) (positive S2 clipped to 0);
  - the same class-weighted rows, features, seeds and gene-grouped inner validation;
  - early stopping on the class-weighted binary AP of the protocol label (custom metric);
  - eval label and metric unchanged.
- Passes at 3 seeds (+0.009) and at 5 seeds (+0.009), on both partitions, under all three weightings. The
  inner-validation weighted AP rises the same way (0.149 → 0.161 mean), so the outer result agrees with the
  inner one.
- Where the gain is: C1 +0.026, C2 +0.009, C3 ≈ 0.
  - The regression target uses the graded strength of the negative tail, which helps rank the top pairs of
    seen genes.
  - It does not help unseen pairs, where the signal is gene-level propensity.
- Importance shifts from GP (0.37 → 0.27) to NBR (0.16 → 0.23).

## Step 5 — C3-focused low-capacity model (pre-stated in config/m2plus.yaml `c3`)
- Model: L2 logistic regression (C = 0.1) on 17 pair min/max gene features (gp_hat, XSP prop/ess, PHEN
  n_pheno/pc1/smf, CER neg-rate/ortholog-essential). It is fitted on the ~91k nested C3-like stacker rows per
  fold.
- Blend: C3 rows only, ranks averaged 50/50. The C3 rows are then given M2's own C3 scores in the blended
  order, so the pooled position of C3 is unchanged. Nothing was tuned.
- Inner check (fit on inner folds ≠ 0, score C3-like rows of inner fold 0): LR AP 0.056–0.105 vs prevalence
  ≈ 0.02 (folds 0–4: .081/.077/.056/.075/.105).
- Outer result: C3 0.0947 → 0.0945, primary −0.0001. The LR ranks no better within C3 than M2 does, and
  averaging two rankings that are largely the same signal adds nothing.

## Surprises / notes for the PI
1. The biggest gain of the rung came from the **training target**, not from features: Huber on the clipped
   S2 is +0.009, while every feature or C3 idea was ≈ 0.
2. XSP was null despite its decent cerevisiae-side CV. The tree's C3 lever (unseen-gene propensity) did not move
   with either cross-species features or a specialised C3 model. C3 seems capped near 0.095 by the current
   gene features.
3. M2 reproduced exactly with 6 vs 8 threads (LightGBM force_col_wise), which makes paired comparisons across
   machines and loads trustworthy.
4. **Deliverable impact:** `src/deliverable.py` / `config/models.yaml` have no Huber objective. Shipping the new
   incumbent needs an `objective: huber` (target −min(S2, 0), δ = |threshold|, AP early stopping) and
   `seeds: all` option there. That file is owned by the deliverable agent; the implementation to mirror is
   `scripts/run_m2plus.py::fit_stacker_huber`.
5. `results/registry.tsv` / `comparisons.tsv` on this branch have rows appended after main's; the merge of main
   into m2plus was conflict-free.

## Reproduce
```bash
python scripts/features/merge_staged.py --fix-aliases                      # step 1 (idempotent)
python scripts/run_m2plus.py --stages features,run --variants M2_XSP,M2r,M2_XSP_LF
python scripts/run_m2plus.py --stages run --variants M2r --seeds bag [--split-seed 7]
python scripts/run_m2plus.py --stages run --variants M2r --seeds bag --objective huber [--split-seed 7]
python scripts/run_m2plus.py --stages run --variants M2r --objective huber
python scripts/run_m2plus.py --stages c3blend --variants M2r --base-model <M2r_bag5_huber run id>
python scripts/run_m2plus.py --stages compare --pairs "A:B[:A7:B7],..."     # run ids or model names
```
