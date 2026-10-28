# Phase A — minimal infrastructure (vertical slice)

## What exists
- `config/protocol.yaml`: all evaluation-defining values (label, universe, masks, lockbox sha256, seeds, fractions, bootstrap, adoption rule, class weights). Derived blocks are script-written. Read only via `src/protocol.py`.
- `scripts/build_universe.py` → `universe_genes.tsv`: 3,779 universe genes (3,580 viable + 177 condition-dependent + 22 unknown-viability genes with a Ryan deletion strain; sap10 excluded). Also 12 drop-from-training genes.
- `scripts/compute_class_weights.py` → weights C1/C2/C3 = 0.2365/0.5698/0.1937 over unmeasured universe pairs (primary); 0.3541/0.4820/0.1639 over all universe pairs.
- `scripts/make_splits.py` → `data/processed/splits/` (`labels_dev`, `labels_lockbox`, `dev_folds_seed{20261028,7}.tsv`, `c1_holdout_seed*.parquet`, `manifest.json`) and `results/splits_summary.json`.
- `src/data.py`: `fold_split`, `fold_labels`, `inner_splits`, `gene_grouped_validation`, `cache_path/save/load`. `src/splits.py` holds the pure generators and checkers.
- `src/evaluate.py`: `evaluate()` (validate, seed-average, join labels, metrics, gene bootstrap, registry) and `compare()` (paired Δ, adoption).
- `src/timing.py`: `Timer`, `Timings`, `set_threads`.
- `scripts/run_b0.py` (random baseline) and `scripts/noise_reference.py`.

## How to run
```
.venv/bin/python scripts/build_universe.py
.venv/bin/python scripts/compute_class_weights.py
.venv/bin/python scripts/make_splits.py
.venv/bin/python scripts/run_b0.py [--compare-to RUN_ID]
.venv/bin/python scripts/noise_reference.py
.venv/bin/python -m pytest -q tests/
```
Toy data: set `PSL_DATA_ROOT` / `PSL_RESULTS_ROOT`.

## Tests (57)
- `test_metrics`: AP vs sklearn (ties, weights); hand example 49/60; flipped-label metric caught; bootstrap fast path equals brute force.
- `test_bootstrap`: Poisson counts; analytic SE for a pair-weighted mean and an all-ties AP; perfect ranking gives SE 0.
- `test_splits`: no C2/C3 gene in training; no train/eval overlap; no lockbox gene or lockbox-C1 pair in dev; eval masks; hypomorph = 0 on eval rows; per-pair DAmP flags. Each check also catches a broken variant. Runs on synthetic and real data.
- `test_inner`: no T_j gene or H_j pair in inner training; each C2 pair kept once; gene-grouped stacker CV. Broken variants are caught.
- `test_evaluate`: random ≈ prevalence; oracle = 1; registry, compare, input rejection, timer.
- `test_lint`: no threshold or weight literals; restricted-file whitelist. Broken snippets are caught.

## Numbers (`results/splits_summary.json`)
- 1,337,033 S2 pairs. Dev: 1,924 genes, 790,237 pairs; 622,515 eval-eligible pairs with 12,992 positives.
- Excluded universe pairs: DAmP-only 16,557; merged-ID 4,862. Training drop: 9,800 pairs.
- Lockbox (eligible): C1 69,169 pairs / 1,444 positives; C2 344,186 / 7,247; C3 42,752 / 983.
- Per fold (seed 20261028): C1 ≈ 40k pairs / ≈833 positives; C2 ≈ 199k / ≈4.1k; C3 ≈ 24.8k / ≈533; training ≈ 462k / ≈8.5k.
- B0: primary 0.0209 (SE 0.0015) against a weighted prevalence of 0.0209.
- Noise reference AP: 0.348 (lift ≈ 16) at a single-orientation raw threshold of −3.76.
