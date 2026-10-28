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
