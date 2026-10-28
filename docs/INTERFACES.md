# Shared interfaces (PI-owned; agents must follow, propose changes to PI)

## Gene universe / ordering
- `data/processed/features/genes.txt`: global gene order = all PomBase protein-coding genes (one systematic ID per line, sorted). Index i of every G×G matrix refers to line i.

## Feature store (`data/processed/features/`, git-ignored; built by scripts in `scripts/features/`)
- Gene-level: `gene/<GROUP>.parquet` — index = PomBase systematic ID (subset of genes.txt), columns named `<GROUP>__<name>`, float32, NaN = missing.
- Similarity (for NBR and pair features): `sim/<source>.npy` — float32 G×G (G = len(genes.txt)), symmetric, NaN where either gene lacks data, diagonal NaN.
- Pair-level scalar features: `pair/<GROUP>__<name>.npy` — float32 G×G symmetric, NaN = missing.
- `manifest.json`: {feature_or_file: {"group":..., "sources": [DATA.md source ids], "script":..., "sha256":...}}. Sources must be on the DATA.md `use=feature` allowlist.

## Labels and splits (owned by Phase A; see DESIGN.md §2)
- Raw label file `data/processed/ryan_pairs.parquet` is read ONLY by data-build/split scripts.
- Models obtain labels only via `src/data.py` fold-scoped API.

## Predictions passed to evaluation
- DataFrame/parquet with columns: `gene_a`, `gene_b` (sorted so gene_a < gene_b), `fold`, `part` (C1/C2/C3), `score` (higher = more likely negative interaction), `seed`.

## Toy data
- `data/toy/` mirrors `data/processed/` layout (ryan_pairs.parquet, ryan_genes.tsv, features/...) so the whole pipeline can run on it by switching the data root (`PSL_DATA_ROOT` env var, default `data/processed`).

## Additions after round-3 review
- Fold-scoped Ryan-derived feature cache: `data/processed/cache/<split_seed>/fold<k>/<name>.parquet` (outer) and `.../fold<k>/inner<j>/<name>.parquet` (nested); cache key includes the label-file hash.
- NaN similarity → weight 0 in all NBR computations.
- Lockbox evaluation: `scripts/lockbox_eval.py` is the ONLY reader of `labels_lockbox.parquet`; it calls `src/evaluate.py` metric functions on lockbox predictions.

## Additions after the M2+ rung (ship branch, 2026-10-28)
- **Default model inputs**: `config/models.yaml` `default_inputs` = `{lf_groups: [...], sims: [...]}` is the ONE pinned list of
  label-free groups / similarity sources of the default models (= the adopted M2 snapshot, `data/frozen_m2`).
  Read through `src/model_inputs.py`: `default_spec()` (FeatureSpec restricted to the groups), `default_lf_groups()`,
  `default_sims()` / `default_sim_store()`, `default_gene_table_exclude()` (store gene-table groups outside the
  default = the `exclude_groups` of `m2.gene_table` / `fold_tables_m2` / `m2_cache_name`), `restrict(spec, groups)`.
  `pairs.load_spec()`, `m2.gene_table()`, `ryanfeat.SimStore()` remain raw "everything in the store" accessors
  (ablations such as XSP); model scripts must not use them for default inputs (tests/test_model_inputs.py).
  On toy / synthetic stores (not the real splits) the default is every group of that store.
- **Stacker fits**: `src/models/stacker.py` — `fit_binary(...)` (M2), `fit_huber(...)` (E-TARGET: Huber on
  `etarget(S2) = -min(S2, 0)`, delta `huber_delta()` = |protocol threshold|, early stopping on class-weighted AP of
  the protocol label), `fit_stacker(objective, X, y, s, w, fit_rows, valid_rows, params, seed, names, init=None,
  etarget_cfg=None)` on boolean row masks, `predict(booster, X, objective, threads, init=None)`, `seed_params`.
  Used by `scripts/run_m2.py::fit_stacker`, `scripts/run_m2plus.py::fit_stacker_huber` (thin wrappers, signatures
  unchanged) and `src/deliverable.py`.
- **`src/data.py: dev_pairs(include_train_drop=False)`**: all dev training pairs (`data.TRAIN_COLUMNS`); lockbox genes
  and lockbox-C1 pairs are absent by construction. `include_train_drop=True` adds drop-from-training pairs (flags only).
- **`src/universe.py`**: `universe_genes(feature_genes)` -> (sorted genes, flags incl. `viability_unknown`),
  `n_pairs`, `pair_chunks(n, chunk)` (row-major i<j blocks), `covering_pairs(genes)`, `pair_class(seen_a, seen_b)`,
  PomBase coordinates (`load_coords`, `gene_coord_arrays`, `gap_from_arrays`, `pair_distance`, `linked_window_bp`),
  `latest_run_id(model)`, `oof_scores(run_id)` (seed- and fold-averaged dev-CV OOF scores), `lockbox_gene_list()`,
  `is_real_data()`, `deliverable_cfg()` (config/deliverable.yaml).
- **`src/deliverable.py`**: `model_spec(id)` (config/models.yaml + toy_overrides), `default_model()`
  (`default_model` key = the model predict_all_pairs.py ships by default, **M2r_huber**), `dev_run_id(spec)` (pinned
  `dev_run` or latest clean `dev_run_model` run: the source of `oof_score`), `TrainedModel(spec, train_pairs, threads,
  ...).fit()` then `.predict(pairs)` / `.score_universe(genes, ryan_chunk, lf_chunk)`; spec `objective: binary|huber`.
- **Universe output** (gitignored): `<data root>/universe/predictions_<model>_<train set>.parquet`; summary (aggregates
  only, committed): `results/universe/summary_<model>_<train set>.json`.
- **Lockbox access log**: `results/lockbox/access_log.tsv` (DESIGN's `results/lockbox_access.log` is superseded):
  one row per `scripts/lockbox_eval.py` invocation (refused | started | completed | failed). The prereg
  `config/lockbox_prereg.yaml` may set `seeds: model` (each model's own config/models.yaml seeds).
