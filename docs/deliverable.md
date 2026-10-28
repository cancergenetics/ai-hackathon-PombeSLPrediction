# Deliverable tooling: universe predictions, lockbox evaluation, linkage diagnostic

Status 2026-10-28. Owner: deliverable engineering agent (branches `deliverable`, then `ship`).

**Shipped model: `M2r_huber`** (PI decision 2026-10-28: M2 features with XSP excluded, Huber stacker on −min(S2, 0),
stacker seeds 0–2; dev run `M2r_huber__20261028T205545__7b5e9e`, primary 0.1791). It is `default_model` in
`config/models.yaml`. `M2r_bag5_huber` (5 seeds) and the previous binary `M2` stay selectable with `--model`.

| file | role |
|---|---|
| `scripts/predict_all_pairs.py` | **the deliverable**: retrain a model spec on all permitted Ryan pairs, score every universe pair |
| `scripts/lockbox_eval.py` + `config/lockbox_prereg.yaml` (template) | one-shot lockbox evaluation of the pre-registered shortlist |
| `scripts/linkage_diagnostic.py` | report-only: hit rate / coverage vs chromosomal distance (dev genes only) |
| `src/deliverable.py` | shared retrain + scoring code (`TrainedModel`) used by both scripts |
| `src/models/stacker.py` | the stacker fits (`fit_binary`, `fit_huber`, `fit_stacker`), shared with dev CV (`run_m2.fit_stacker`, `run_m2plus.fit_stacker_huber` are thin wrappers) |
| `src/model_inputs.py` | pinned default inputs (`config/models.yaml` `default_inputs`): label-free groups and similarity sources |
| `src/universe.py` | universe genes, pair enumeration, PomBase coordinates, `linked` flag, OOF-score lookup |
| `config/models.yaml` | `default_model`, `default_inputs`, and model specs (M2r_huber, M2r_bag5_huber, M2, M2_init, M1N, M1, B1, B4) for full-training-set retrains |
| `config/deliverable.yaml` | linkage window, coordinate files (sha256-pinned), chunk sizes, threads |
| `src/data.py: dev_pairs()` | additive helper: all dev training pairs (lockbox excluded by construction) |
| `tests/test_deliverable.py`, `tests/test_stacker.py`, `tests/test_model_inputs.py` | 24 + 6 + 7 tests (unit + toy end-to-end + live-store input pinning) |

## 1. `predict_all_pairs.py`

```bash
# interim deliverable (dev genes only; the lockbox stays closed)
.venv/bin/python scripts/predict_all_pairs.py --train-set dev            # default model M2r_huber
.venv/bin/python scripts/predict_all_pairs.py --model M2 --train-set dev # previous incumbent, for comparison
# FINAL deliverable, only after scripts/lockbox_eval.py has completed (refused before)
.venv/bin/python scripts/predict_all_pairs.py --model <headline> --train-set all
# toy
PSL_DATA_ROOT=data/toy PSL_RESULTS_ROOT=/tmp/r .venv/bin/python scripts/predict_all_pairs.py --limit-genes 300
```

**Training set.**
- `dev`: `src.data.dev_pairs()`, i.e. every dev S2 pair except drop-from-training pairs. Lockbox genes and
  lockbox-C1 pairs are absent from the dev label file. Non-coding genes are then dropped (as in M2).
- `all`: dev plus lockbox pairs. The lockbox labels come only through `lockbox_eval.unblinded_lockbox_pairs()`.
  That function raises unless `results/lockbox/access_log.tsv` records a completed lockbox run, so
  `lockbox_eval.py` stays the only reader of the lockbox file.

**Universe.** The universe is the 3,779 `in_universe` genes plus the 144 unknown-viability genes outside it,
which are flagged. That gives 3,923 genes and 7,693,003 pairs, all n(n−1)/2 of them. On toy data, every
feature-store gene is scored.

### M2 / M2r_huber at universe scale (nested stacking)

`src/deliverable.TrainedModel` runs **exactly one dev-CV outer fold of `scripts/run_m2.py`**, with the outer
training set replaced by the full training set T:

1. **Inner split.** `data.inner_splits(T)` creates 5 inner gene folds and a 10% inner C1 hold-out.
2. **Stacker training rows.** These are the inner eval rows. Their RYAN, NBR and GP features come from inner
   fold j's training pairs only (`stacking.inner_rows`, M2 builder). They are cached under
   `<data root>/cache/deliverable/<key>/`.
3. **Seeds and objective.** For seed s, early stopping runs on inner fold j = s mod 5, and the model is fitted on
   the gene-grouped rows for j (`src/models/stacker.fit_stacker`). The spec's `objective` selects the stacker:
   - `binary` (M2): LightGBM binary on the protocol label, early stopping on class-weighted AP.
   - `huber` (M2r_huber): Huber regression on t = −min(S2, 0), δ = |protocol threshold|, early stopping on the
     class-weighted binary AP of the protocol label (custom metric) — `scripts/run_m2plus.py --objective huber`.
   The score is the mean raw score over the spec's seeds (0–2 for M2r_huber and M2).
4. **Universe rows.** These get RYAN, NBR and GP features from all of T: the B1, MF, profile, NBR and GP fits
   on T. Label-free features come from `pairs.assemble_arrays`, with HYPO = 0 (deletion × deletion).

The universe pairs are streamed in row blocks of the sorted gene list: 2M pairs per Ryan-feature build and
500k per label-free block. Two properties make this safe:

- **Blocks share one fit.** The builder's per-pair output does not depend on the other pairs in the query
  (tested in `tests/test_ryanfeat.py`). Every block is padded with the same covering pairs, so all blocks have
  the same gene index and therefore the same fit.
- **The fit is checked at run time.** The padding rows must be bit-identical across blocks
  (`checks.cover_rows_identical_across_blocks`).

Torch MF runs under `torch.use_deterministic_algorithms(True)`, scoped to the deliverable. Without it, two
identical MF fits differ by about 1e-6, because the multi-threaded embedding-gather backward is
non-deterministic. This also applies to the dev-CV M2 runs, where it is harmless; see the notes for the PI.

**Equivalence with dev CV.** Given an outer fold's training pairs, `TrainedModel` reproduces
`run_m2.FoldData` + `fit_stacker` + `predict` (`tests/test_deliverable.py::test_m2_path_reproduces_dev_cv_fold`):

- Toy fold 0, before deterministic MF: predictions were **bit-identical**.
- Since deterministic MF, the test allows <0.1% of rows to differ through MF ulps.

### Output

The predictions file is `<data root>/universe/predictions_<model>_<train set>.parquet`. It is gitignored,
about 115 MB for M2, and sorted by rank. Columns:

| column | meaning |
|---|---|
| `gene_a < gene_b` | PomBase systematic IDs (categorical) |
| `score` | Mean raw LightGBM score over seeds, **uncalibrated**, higher = more likely S2 below the protocol threshold. For M2r_huber it is the predicted −min(S2, 0) (class-weighted Huber fit: a ranking score on the S-score scale, not an S-score estimate and not a probability); for M2 it is a log-odds margin. |
| `rank` | 1 = highest score |
| `class` | C1 / C2 / C3: both, one or neither gene in the training pairs |
| `viability_unknown` | Either gene is outside the protocol universe because its deletion viability is unknown |
| `linked` | Same chromosome and gap between the gene bodies < 300 kb (see §3). Ryan essentially never measured these pairs, and their double mutants are hard to make. |
| `same_chrom_gap_kb` | The gap itself, so users can apply their own window. NaN means different chromosomes or no coordinates (18 genes lack coordinates). |
| `measured_in_ryan` | True / False. `<NA>` means the status cannot be known without opening the lockbox (train-set `dev` only): the pair is among Ryan S2 genes and touches a lockbox gene, or it is dev×dev but absent from the dev labels. |
| `in_training` | The pair's label was used to fit the model, so **its score is in-sample** |
| `oof_score`, `oof_n_folds`, `oof_source` | Out-of-fold score for measured pairs: `dev_cv` is the registered dev-CV run (a C2 pair is evaluated in two folds and averaged); `lockbox` is the lockbox evaluation's all-dev model (train-set `all` only). **For measured pairs, use `oof_score`, not `score`.** |

The summary is `results/universe/summary_<model>_<train set>.json`. It holds aggregates only: checks, runtime,
peak RSS, class and flag counts, top-k composition, and the in-sample vs out-of-fold comparison.

### Real run: M2, train-set dev (`results/universe/summary_M2_dev.json`)

Clean run at commit `a269a5f`, `--no-cache`. Data root `data/frozen_m2`: the feature store of the adopted M2
run. Apple M4, **3 threads**.

| quantity | value |
|---|---|
| pairs scored | 7,693,003 = 3,923·3,922/2; no NaN; row count check passed; padding rows bit-identical across the 4 Ryan blocks |
| classes | C1 1,620,900 · C2 3,821,722 · C3 2,250,381 |
| flags | `linked` 353,615 · `viability_unknown` 554,472 · `in_training` 637,564 · `measured_in_ryan` unknown 1,892,561 · with OOF 622,515 |
| **wall time** | **985 s (16.4 min) total**: fit 289 s (stacker nested features 124 s, stacker LightGBM ×3 152 s), universe scoring 690 s (Ryan features from the full training set 598 s in 4 blocks, LF assembly 49 s, prediction 40 s), flags/write 4 s |
| **peak RSS** | **12.4 GB** (`/usr/bin/time` maxrss 13.0 GB). This includes about 4.1 GB of memory-mapped feature-store pages (42 G×G matrices). The macOS peak physical footprint was **6.3 GB**. |
| stacker | 975k stacker rows, 475 features; best_iter 110 / 53 / 27 (seeds 0/1/2) |

A shakedown run before the deterministic-MF fix took 1,053 s with peak RSS 11.5 GB. Its padding rows differed
by 2.9e-6 because of MF jitter.

The cost is dominated by refitting the Ryan builder (B1, MF, GP, NBR on a 4.1k-gene index) once per 2M-pair
block, about 150 s each. Larger blocks are faster but use more memory: the block DataFrames are about 1 GB per
1M pairs.

**Toy, full universe** (`results/universe/toy_summary_M2_dev.json`): 1,200 genes, 719,400 pairs, all checks
pass, 56 s, peak RSS 2.4 GB.

**In-sample vs out-of-fold.** On the 622,515 measured dev pairs with a dev-CV OOF score (12,992 positives):

| score | AP |
|---|---|
| in-sample deliverable `score` | 0.307 |
| out-of-fold (`M2__20261028T183713__024416`) | 0.201 |

The Spearman correlation between the two is 0.87. In-sample scores are optimistic, as expected, which is why
the OOF columns exist. 522 of the top-1,000 pairs are in-sample training pairs.

**Top-k composition.**
- Top 1,000: 822 C1, 178 C2, 0 C3.
- Top 100,000: 32% C1, 54% C2, 14% C3. C3 is 29% of the universe.

## 2. `lockbox_eval.py` + `config/lockbox_prereg.yaml`

**Not run on real data.** The committed prereg is a template with TODOs, and the script refuses while any TODO
remains.

**Refusals.** Exit code 2; every refusal is logged.
- The prereg is untracked, or modified relative to HEAD.
- A TODO remains, or the prereg is invalid: the headline model is not in the shortlist, there are more than 5
  shortlisted models, or a contrast falls outside shortlist + baselines.
- A sha256 differs from the prereg: protocol, lockbox genes, or `config/models.yaml`.
- HEAD differs from the prereg's `code_commit` in `src/`, `scripts/` or the config files.
- On real data, the code tree is dirty.
- The access log already has a `completed` run for this prereg sha256 (**one-shot**). Override with
  `--i-understand-second-access --reason "..."`; the reason is required and the override is logged.

**Access log.** Every invocation appends rows to `results/lockbox/access_log.tsv`: timestamp, event
(refused / started / completed / failed), git commit and dirty flag, prereg path and sha256, user, data root,
override, reason, detail.

**Run.**
1. Each shortlisted model and baseline is retrained on all dev training pairs through the same `TrainedModel`
   code as the deliverable. Seeds come from the prereg (default `all` = 0–4, per DESIGN §3).
2. Each model is scored once on the eligible lockbox C1/C2/C3 pairs. These are the `labels_lockbox` parts,
   including the lockbox-C1 pair set.

**Metrics.**
- Primary: pooled universe-weighted AUPRC (`evaluate.pooled_metrics`, one "fold").
- Per-class AP, plus all-pairs and equal weightings, AUROC, P@1000, and the raw-mean-label sensitivity.
- Shared Poisson(1) gene bootstrap (`evaluate.bootstrap_replicates`) for the primary and per-class CIs. The
  per-class CIs come from class-indicator weightings on the same replicates.
- Paired two-sided bootstrap p-values for the prereg contrasts, Holm-adjusted.
- Dev-minus-lockbox gap, taken from the model's latest clean dev run.

**Outputs.**
- `results/lockbox/lockbox_result_<sha12>.json` (aggregates).
- `<data root>/lockbox/preds_<model>.parquet` (scores only). These become the out-of-sample scores of lockbox
  pairs in the `--train-set all` deliverable.

**Toy test** (`test_lockbox_sequence`, toy lockbox, M2 and B1 with seed 0, 50 bootstrap replicates):
1. Refused: untracked prereg, modified prereg, TODO left, and `--train-set all` before any completed run.
2. First run completes, M2 > B1, and the Holm p is ≥ the raw p.
3. A second access is refused, an override without a reason is refused, and an override with a reason
   completes and is logged as `override=True`.
4. `--train-set all` then works: lockbox pairs become training pairs, and their OOF source is `lockbox`.

### Prereg fields (the PI fills the TODOs)

| field | meaning |
|---|---|
| `models_sha256`, `code_commit` | Exact model configurations and code |
| `shortlist`, `headline_model` | ≤5 models. The headline model is **shipped regardless of the lockbox ranking.** |
| `baselines` | B1, B4 |
| `seeds` | `all` |
| `boosting_rounds` | `dev_cv_inner_early_stopping`: the per-fold dev-CV rule applied to the all-dev training set, with no lockbox information |
| `metric` | Primary/secondary definitions, bootstrap and p-value definitions |
| `holm_family` | `alpha` and `contrasts` |

## 3. Linkage diagnostic (`results/linkage/`)

`scripts/linkage_diagnostic.py [--download]` uses dev genes only and never reads the lockbox.

**Coordinates.** They come from PomBase release 2026-10-01, `misc/chromosome_{1,2,3}.gene.coords.tsv`. These
were **not** among the downloaded files, so they were fetched to `data/raw/pombase_coords/` and
sha256-pinned in `config/deliverable.yaml`. They are label-free and used for flags and reports only; distance
is never a feature.

**Method.**
- **Distance** = the gap between gene bodies on the same chromosome.
- **Coverage** = measured dev pairs divided by design-testable dev×dev pairs, where design-testable means one
  gene is a Ryan query and the other a query or array gene. Lockbox-C1 pairs (a random 10%) lower every bin
  alike.
- **OOF column** = the fraction of pairs in the top-q of M2 OOF scores, with q = the overall hit rate.

| gap (kb) | measured pairs | coverage | hit rate | M2 OOF top-q rate |
|---|---|---|---|---|
| 0–100 | 52 | 0.3% | 3.8% (2/52) | 0 |
| 100–300 | 1,053 | 1–6% | 2.6% | ≈0.1% |
| 300–500 | 9,605 | 13–47% | 1.84% | 1.3% |
| 500–1,000 | 52,471 | 62–77% | 2.20% | 1.8% |
| 1,000–1,500 | 47,484 | 78% | 2.11% | 2.1% |
| ≥ 1,500 | 145,075 | 77% | 1.75% | 2.0% |
| different chromosome | 534,497 | — | 1.82% | 2.1% |

Finding: Ryan **effectively excluded linked pairs**.
- Coverage is ≤6% below 300 kb, against a ~77% plateau beyond 800 kb. Coverage ramps between 300 and 800 kb, a
  graded filter rather than a sharp cut.
- Among the 1,105 measured pairs below 300 kb, the hit rate is 2.6% (OR 1.45 vs different chromosome,
  p = 0.055).
- Same-chromosome pairs at 300–1,500 kb show a small, highly significant excess:
  - 300–1,000 kb: 2.15% (OR 1.18, p = 3e-8);
  - 1,000–1,500 kb: 2.11% (OR 1.16, p = 2e-5).
  - Pairs ≥1.5 Mb (1.75%) look like pairs on different chromosomes (1.82%).
  - M2's OOF scores do not predict this excess. Among OOF pairs at 300–1,000 kb, the hit rate is 2.43% and the
    top-q rate 1.79%; on different chromosomes the two are 2.05% and 2.14%. So the excess is not explained by
    gene propensity. It is consistent with a residual linkage effect on S-scores of moderately linked pairs,
    though chromosomal-neighbourhood biology cannot be excluded.

**Decision implemented.**
- Universe pairs on the same chromosome with gap **< 300 kb** are flagged `linked`: 353,615 pairs, 4.6%. These
  are pairs Ryan essentially never measured, so their scores are unvalidated extrapolations, and the double
  mutants are hard to construct.
- The gap itself is in the output for stricter use.
- **No score penalty.** The model never sees distance and is not trained on such pairs. The observed excess
  is in the labels (+0.3–0.4 percentage points at 300–1,500 kb), not something the model reproduces.

## 4. Decisions / notes for the PI

1. **`linked` window.** 300 kb is implemented, based on coverage. The alternatives are about 500 kb (coverage
   ≤47%) or 1 Mb (covers the label excess as well). A sensitivity re-evaluation of the dev metric that excludes
   300–1,500 kb same-chromosome pairs would show whether the excess matters for model selection. Not done here.
2. **Prereg.** Fill `shortlist`, `headline_model`, `contrasts`, `models_sha256` and `code_commit`, then commit.
   - Models can only come from `config/models.yaml`. An M2+ or XSP variant needs a spec entry there.
   - The m2_stack kind reads `families`, `lf_groups` and `sims`, so a new feature group only needs a new list.
3. **`--train-set all`** is gated on the access log, and it reads lockbox labels through
   `lockbox_eval.unblinded_lockbox_pairs()`. That keeps the single-reader rule literally.
4. **Non-deterministic MF in dev CV.** `src/ryanfeat.fit_mf` under multi-threaded torch gives about 1e-6
   run-to-run jitter. M2 dev scores are reproducible only up to that; in toy tests, scores were unchanged. The
   deliverable fixes this locally. A one-line `torch.use_deterministic_algorithms(True)` in run_m2/ryanfeat
   would fix it everywhere; I did not touch those files.
5. **Pre-existing test failure, not from this branch.** `tests/features/test_tier1_features.py::test_manifest_sources_allowlisted`
   fails on main too: `biogrid_genetic_pmids.txt` lists source `biogrid_physical`. The data-store manifest was
   modified at 19:16, probably by the XSP agent.
6. **Docs to update (PI-owned).**
   - DATA.md: add the `pombase_coords` register entry (use = report/flag only).
   - docs/INTERFACES.md: `data.dev_pairs`, `src/universe.py`, `src/deliverable.py`, the lockbox access log path
     (`results/lockbox/access_log.tsv`; DESIGN said `results/lockbox_access.log`).
