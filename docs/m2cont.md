# E-CONT rung: gene propensity and NBR rates from the continuous S2

Status 2026-10-28. Branch `m2cont` (worktree `../psl_wt_cont`, `data/` = symlinks into the main tree's `data/`).
Code: `scripts/run_m2cont.py`, additive options in `src/ryanfeat.py` (`cont_nbr`, `cont_perm_seed`) and
`src/models/m2.py` (`gp_cfg["cont"]`, `fit_additive_cont`, families GPC / NBRC), tests `tests/test_m2cont.py`.
The rules were pre-stated in `config/m2cont.yaml` and committed (5e1b7a5) before any result. `config/protocol.yaml`
is untouched.

## Summary
- **All three variants are null.** None is adopted. Δ primary against the incumbent M2r_huber (3 seeds, Huber
  stacker) is −0.0001 to −0.0007, and every CI spans 0.
  - Per the pre-stated rule, no variant seed-7 runs and no permutation control were made (each needed a passing
    rule 1).
- **The hypothesis' premise holds; its consequence does not.**
  - The continuous target is less noisy. Toy: Spearman with the true propensity is 0.812 for θ vs 0.551 for B1
    β. Real data: split-half reliability is 0.893 for θ vs 0.782 for β.
  - The stacker ranks no better with it. On real data θ and β correlate at ρ 0.91. The stacker already had
    the continuous information through the Huber target and the unclipped −S `negs` NBR/RYAN columns
    (`nbr_mean__negs_dmean` is the incumbent's #3 feature).
- **C3 is still ≈ 0.094.**
  - M2c_nbr C3: +0.0039 [−0.0008, +0.0093] (paired per-class diagnostic), offset by C1 −0.0021.
  - This is the closest thing to a C3 signal of any idea so far, but it is not significant and not a candidate.

## Variants (config/m2cont.yaml)
- t = −min(S2, 0), the same target as the Huber stacker. New columns are built with exactly the incumbent's
  nesting: `stacking.inner_rows` / `outer_rows` with `M2Builder`. They are cached as add-on tables
  `m2cont_9b32ca0a7e6d` and joined row-aligned onto the incumbent cache `m2feat_3d35f8c314e9`, with alignment
  asserted.
  - The binary RYAN/NBR/GP columns recomputed in the add-on build equal the incumbent cache exactly: max |diff|
    0, with 0 NaN mismatches in all 30 tables.
- **(a) M2c_gp:** the GP target is the continuous additive propensity θ instead of B1 β.
  - θ comes from a ridge additive model t_ab = c + θ_a + θ_b, with an empirical-Bayes moment rule for α
    (α ≈ 7 on toy).
  - Inputs, regressor and cross-fitting are unchanged.
  - The `gpc_*` columns **replace** the binary `gp_*` columns.
- **(b) M2c_nbr:** NBR on t, using the same neighbour matrices: directional dmax/dmean, block, and neighbour
  gene mean of t (gen_min/max). That is 9 sources + mean = 50 columns, **added** to the binary NBR columns.
- **(c) M2c_both:** (a) + (b).
- Stacker: the incumbent's `run_m2plus.fit_stacker_huber`, imported unchanged. Main now wraps this in
  `src/models/stacker.fit_huber` (bit-identical at a fixed thread count; merged at cde86c6).
- All runs used **4 LightGBM threads**. The registered incumbent used 6, so it was re-run at 4 threads
  (amendment 2, logged in the config). The two agree: Δ primary ≈ 1e-8, per-class Δ 0.0000.

## Checks before running
- **Toy known-answer check** (`results/m2cont/toy_check.json`). Pre-stated bar: ρ_θ ≥ ρ_B1 + 0.05 (fold mean).
  - **PASS:** ρ_θ = 0.812 vs ρ_B1 = 0.551.
  - Secondary: unseen-gene GP β̂ ρ is 0.628 with the continuous target vs 0.563 with the B1 target (PASS).
  - Shrunken gene mean of t: 0.812 (diagnostic).
  - Caveat: the toy S is Gaussian and additive, which favours continuous targets.
- **Canary** (`results/m2cont/canary_toy.json` + `tests/test_m2cont.py`).
  - Pipeline canary, toy data, outer fold 0: S2 of the 18,691 pairs held out of inner fold 0's training set was
    corrupted, with labels kept. Inner fold 0's stacker rows kept bit-identical new columns. The other inner
    folds and the eval rows moved (positive control). **PASS.**
  - Unit canaries, synthetic data:
    - corrupting held-out (eval) S2 leaves the new columns bit-identical, and a planted eval-S2 mutant is
      caught;
    - corrupting a stacker row's own S2 leaves its nested new columns unchanged, and a non-nested mutant is
      caught.
  - An outer-level label-file canary would need to name the dev label file, which `tests/test_lint.py` forbids
    outside its whitelist. Its outer-level role is covered by the unit canaries, because the builder only ever
    receives training pairs.

## Runs (primary split 20261028 unless noted; 3 seeds; 4 threads; all clean)
| model | run_id | primary ± SE | C1 | C2 | C3 | all-pairs | equal | inner wAP | train s | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| M2r_huber (incumbent, 6 thr) | `M2r_huber__20261028T205545__7b5e9e` | 0.1791 ± 0.0126 | 0.2721 | 0.1668 | 0.0936 | 0.1940 | 0.1798 | — | 486 | 592bdf1 |
| M2r_huber (re-run, 4 thr) | `M2r_huber__20261028T225718__1bfff1` | 0.1791 ± 0.0126 | 0.2721 | 0.1668 | 0.0936 | 0.1940 | 0.1798 | 0.1599 | 632 | d0483c8 |
| M2c_gp | `M2c_gp__20261028T215430__d5782a` | 0.1791 ± 0.0127 | 0.2731 | 0.1663 | 0.0931 | 0.1941 | 0.1800 | 0.1591 | 1023 | 6016810 |
| M2c_nbr | `M2c_nbr__20261028T221120__f7dd74` | 0.1785 ± 0.0126 | 0.2700 | 0.1663 | 0.0975 | 0.1930 | 0.1792 | 0.1604 | 888 | 6016810 |
| M2c_both | `M2c_both__20261028T222521__304655` | 0.1790 ± 0.0126 | 0.2718 | 0.1660 | 0.0959 | 0.1938 | 0.1800 | 0.1602 | 741 | 6016810 |
| M2r_huber_split7 (seed-7 comparator) | `M2r_huber_split7__20261028T224059__43f04c` | 0.1763 ± 0.0122 | 0.2591 | 0.1679 | 0.0870 | 0.1895 | 0.1757 | 0.1612 | 671 | 6016810 |

- Runtime, wall seconds, Apple M4 with other agents running:
  - The add-on nested build is one-off for all three variants: 1,015 s summed over folds (2 folds in parallel ×
    2 threads, ~9 min wall). Per fold: 25–41 s per inner build and 41–55 s outer; MF/profile off.
  - Variant run totals (train + predict + load): 1,144 / 1,009 / 841 s. Train time varies mostly with machine
    load.
  - Peak RSS 12.4 GB during the run stage.
- Importance:
  - GPC takes over GP's share almost exactly (0.25 vs 0.27); `gpc_best_sum` replaces `gp_best_sum`.
  - NBRC takes 0.12, mostly from NBR (0.22 → 0.15). `nbrc_mean__dmean` substitutes for `nbr_mean__negs_dmean`.

## Paired comparisons (`results/m2cont/comparisons.json`, `perclass.json`; also results/comparisons.tsv)
| a − b (b = 4-thread incumbent re-run) | Δ primary [95% CI] | all-pairs | equal | ΔC1 [CI] | ΔC2 [CI] | ΔC3 [CI] | verdict |
|---|---|---|---|---|---|---|---|
| M2c_gp | −0.0001 [−0.0018, +0.0020] | +0.0001 | +0.0002 | +0.0010 [−0.0032, +0.0057] | −0.0006 [−0.0025, +0.0015] | −0.0005 [−0.0056, +0.0049] | not adopted |
| M2c_nbr | −0.0007 [−0.0021, +0.0007] | −0.0010 | −0.0006 | −0.0021 [−0.0058, +0.0019] | −0.0006 [−0.0022, +0.0012] | +0.0039 [−0.0008, +0.0093] | not adopted |
| M2c_both | −0.0002 [−0.0020, +0.0018] | −0.0002 | +0.0002 | −0.0002 [−0.0046, +0.0046] | −0.0008 [−0.0029, +0.0015] | +0.0023 [−0.0026, +0.0078] | not adopted |

- Against the registered 6-thread incumbent run (the pre-stated comparator), the deltas and CIs are identical to
  4 dp.
- No weighting gives an opposite-direction CI (no sign-flip veto). Seed-7 Δ: not applicable, because no variant
  passed rule 1.
- The per-class CIs come from the shared gene bootstrap with one-class weightings (diagnostic only; the
  adoption rule uses the pooled primary).

## Surprises / notes for the PI
1. **Less noisy target ≠ better ranking.** The continuous θ is clearly more reliable (split-half 0.89 vs 0.78
   on real fold-0 training pairs; `results/m2cont/target_noise_real.json`). But GP β̂ for unseen genes is
   limited by the label-free gene inputs, not by target noise. On real data θ ≈ β (ρ 0.91).
2. **The binary-only premise was partly wrong.** RYAN `negs_mean_*` and NBR `*_negs_*` were already built from
   continuous (unclipped) −S. Clipping to −min(S2, 0) mostly re-expresses them.
3. **Thread count changes per-row scores.** The 4-thread split-7 run differs from seeds 0–2 of the 6-thread
   bag5 split-7 run by up to 0.33 per row (0.2% of rows identical), more than the PI's 0.027. Early-stopping
   iterations probably shift. Pooled primary is unaffected (~1e-8 on the primary split). Compare only at equal
   thread counts.
4. **Lint failures inherited from main.** `tests/test_lint.py` fails on main (1fa0520) itself:
   `scripts/run_fewshot.py` mentions the dev label file, and `scripts/rarely_seen.py` accesses
   `evaluate._labels_for`. These are not E-CONT files and are left untouched. Otherwise the suite passes
   (239 passed after the merge).
5. **Stacker unification.** `run_m2cont.py` imports `fit_stacker_huber` / `FoldDataP` / `assemble_lf` from
   `scripts/run_m2plus.py`. After the merge these are wrappers over `src/models/stacker.py`, so nothing needs
   unifying beyond that.

## Reproduce
```bash
PSL_DATA_ROOT=data/toy python scripts/run_m2cont.py --stages toycheck,canary
python scripts/run_m2cont.py --stages features                    # add-on tables, primary split
python scripts/run_m2cont.py --stages run --variants M2c_gp,M2c_nbr,M2c_both,M2r_huber
python scripts/run_m2cont.py --stages run --variants M2r_huber --split-seed 7
python scripts/run_m2cont.py --stages compare,perclass --pairs M2c_gp:<inc>,M2c_nbr:<inc>,M2c_both:<inc>
python scripts/run_m2cont.py --stages noise
```
