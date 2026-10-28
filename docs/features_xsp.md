# XSP — cross-species transfer features (Costanzo → S. pombe via species-agnostic features)

Status 2026-10-28. Builder `scripts/features/build_xsp.py`, helpers in `src/xsp.py`, tests in
`tests/features/test_xsp.py` (14 pass). DESIGN.md §5 M4 lists a "full transfer model restricted to features
available in both species". This is that model, built as a feature group. **No S. pombe GI data was read:**
no Ryan files, splits or preds (the builder is grep-tested for this). The only labels are Costanzo 2016 labels
on *S. cerevisiae*.

**Outputs are STAGED, not live.** Another agent was modelling on the live store, so everything is in
`data/processed/features_staging/`:

| file | content |
|---|---|
| `gene/XSP.parquet` | `XSP__prop_hat`, `XSP__prop_nn_hat`, `XSP__ess_hat` for all 5,126 genes in genes.txt (no NaN) |
| `pair/XSP__pair_hat.npy` | float32 5,126 × 5,126, genes.txt order, symmetric, NaN diagonal, no other NaN |
| `manifest_xsp.json` | entries in the live manifest format; sources are exact allowlist ids |
| `models/`, `cache/` | final LightGBM models; allele-collapsed Costanzo cache |

**To adopt (PI):**
1. Copy `gene/XSP.parquet` and `pair/XSP__pair_hat.npy` into `data/processed/features/`.
2. Merge the two `manifest_xsp.json` entries into `manifest.json`.
3. Run `pytest tests/features`.

Checks are in `results/features/xsp_checks.json`. Quick variant runs are in `results/features/xsp_quick_*.json`.
Runtime is in `results/runtime/features_xsp.json`.

## Method
- **Universes**
  - Pombe: the 5,126 genes in genes.txt.
  - Cerevisiae: the 5,930 SGD R64-5-1 ORFs that are *Verified* or *Uncharacterized* and have an ESM embedding.
    Dubious ORFs, pseudogenes and transposable elements are excluded so the two proteomes are comparable.
- **Species-agnostic gene features (65).** Every one is rank-normalised **within its species** (percentile,
  ties averaged; SINaTRA-style).
  - Sequence statistics (25): log length, 20 amino-acid fractions, hydrophobic fraction,
    disorder-promoting fraction, net charge, charged fraction.
  - ESM-2 650M, 32 PCs. Each species is centred on its own mean, then one PCA basis is fitted on both species
    with equal weight. The 32 PCs explain 79% of the variance.
  - ESM paralogy within each species (3): top-1 and top-5 cosine of the centred embeddings, and the number of
    genes with cosine > 0.5. Compara paralogues were not used because no cerevisiae equivalent is in the
    allowlist.
  - PPI topology (5): degree, PageRank, clustering, k-core and mean neighbour degree.
- **PPI networks**
  - Pombe: the tier-1 `pair/PPI__edge.npy` network (system filter + PMID blocklist): 8,230 edges, mean degree
    3.2, 44% of genes isolated.
  - Cerevisiae: BioGRID 5.0.262 physical with the same system filter (Affinity Capture-RNA, Protein-RNA and
    Co-localization dropped) and no blocklist: 162,497 edges, mean degree 54.8, 7% of genes isolated.
- **Pair features (71), symmetric.**
  - Min and max of 13 gene features + ESM PC1–16 + the 3 gene-level transfer predictions.
  - ESM cosine (ECDF-mapped within species).
  - ESM mutual rank.
  - PPI edge.
  - Shared neighbours, Jaccard and RWR (restart 0.5), each ECDF-mapped within species.
  - Capped shortest path.
  - Symmetry: assembling (b, a) equals assembling (a, b) exactly. This is tested, and was also checked on 20k
    pombe pairs during the build.
- **Costanzo labels**
  - Allele collapse and essentiality come from `build_cer.py` (its functions are imported).
  - This is the same ORF-pair set as `costanzo_pairs.parquet` (Pearson 0.99994). The collapsed table was used
    instead because the parquet averages measurements rather than alleles and has no per-pair significance flag.
  - 10.2M measured universe pairs and 5,203 Costanzo genes (954 essential: TS/DAmP strain and no deletion strain).
  - **Pair label** = collapsed negative (ε < −0.08 & p < 0.05) **and ε < −0.0897**. The second threshold was
    chosen so that prevalence is **2.00%**; the plain Costanzo-negative rate is 3.35%.
- **Gene level**
  - LightGBM, 400 rounds, learning rate 0.03, 15 leaves.
  - Targets:
    - `prop`: logit of the smoothed negative rate over all tested partners (prior strength 20).
    - `prop_nn`: the same over non-essential partners only.
    - `ess`: Costanzo-essential.
  - Gene-held-out 5-fold CV; YGOB ohnolog pairs are kept in the same fold.
- **Pair level**
  - LightGBM (binary; learning rate 0.05, 63 leaves, min leaf 200, feature fraction 0.7) on **2.5M sampled
    measured pairs**.
  - Gene-held-out CV with the same 5 folds.
  - Within each outer fold:
    - Gene targets are recomputed over training partners only.
    - `*_hat` for training genes are inner 5-fold OOF values; for held-out genes they come from the model fitted
      on the training genes. The test (C3) labels therefore never touch any feature.
    - Early stopping uses 15% validation genes.
  - Final model: all 2.5M pairs, 762 rounds (1.1 × median best_iter), with OOF `*_hat`. Applied to all
    13.1M pombe pairs, with pombe `*_hat` from gene models fitted on all 5,203 Costanzo genes.

## Cerevisiae CV (gene-held-out)
| quantity | value |
|---|---|
| prop_hat Spearman vs negative rate (all 5,203 genes) | **0.391** (folds 0.36–0.42); non-essential genes only 0.321 |
| prop_nn_hat Spearman vs non-essential-partner rate | 0.381 |
| reference: rank PPI degree alone | 0.300 |
| ess_hat | AUROC **0.856**, AUPRC 0.605 (prevalence 0.183) |
| pair C3 (both genes held out), pooled 500,728 pairs / 10,068 positives | **AUPRC 0.0414** (lift 2.06), AUROC 0.629; fold mean 0.0422 |
| pair C3 baseline: prop_hat_a + prop_hat_b | AUPRC 0.0334 (lift 1.66) |
| pair C3, NxN only (both non-essential) | AUPRC 0.0250 at prevalence 0.0155 (lift 1.62) |
| pair C2 (one gene held out), fold mean | AUPRC 0.0616 (baseline prop-sum 0.034) |

**Transfer diagnostic** (label-free for pombe). Each fold-k model is applied to POMBE features of genes whose
1:1 curated ortholog is in held-out fold k (2,395 such genes). The score is compared with that ortholog's
Costanzo value. Running the same model on the cerevisiae features of the same genes or pairs isolates the cost
of the species switch.

| level | pombe features | cerevisiae features (same items) |
|---|---|---|
| gene: Spearman with ortholog negative rate | **0.246** | 0.290 |
| pair: AUPRC on 420,793 ortholog pairs (prevalence 0.0335; label = ortholog Costanzo label) | **0.0474** (lift 1.42; prop-sum baseline 0.0448) | 0.0640 (lift 1.91) |

Gene-level transfer keeps most of its accuracy across species (0.25 vs 0.29). Pair-level transfer keeps about
half of the lift. The pair-specific part on top of gene propensity is small in pombe: 0.0474 vs 0.0448 for the
prop-sum baseline.

**PPI variant choice.** The rule was stated before the comparison: maximise the pombe-feature pair proxy
AUPRC; gaps < 0.002 count as ties and go to the simpler option. Quick runs used 1M pairs and folds 0–1
(`results/features/xsp_quick_*`):

| variant | proxy pombe | proxy cer | C3 |
|---|---|---|---|
| full network, 16 ESM PCs | **0.0448** | 0.0589 | 0.0390 |
| full network, 0 ESM PCs | 0.0418 | 0.0593 | 0.0383 |
| cerevisiae edges subsampled to pombe mean degree (5.9%), 16 ESM PCs | 0.0436 | 0.0424 | 0.0300 |
| subsampled, 0 ESM PCs | 0.0431 | 0.0409 | 0.0307 |

**Chosen: full network, 16 PCs.** Subsampling the cerevisiae network removes the species gap (pombe ≈ cer) but
also loses cerevisiae signal, so the net effect is not positive.

## Pombe-side sanity checks (label-free; live features read-only)
- `XSP__prop_hat` vs pombe-native sickness signals, which never entered XSP:
  - PHEN__n_pheno ρ = 0.29 (n = 3,500).
  - PHEN__smf_yes32 ρ = −0.22.
- vs ortholog Costanzo features:
  - CER__neg_rate_e ρ = 0.32; CER__neg_rate_nxn 0.26.
  - CER__any_ortholog_essential 0.36 (ess_hat: 0.47).
- `XSP__prop_hat` vs `XSP__prop_nn_hat` ρ = 0.955, so they are nearly redundant. Keep one, or let the model
  choose. `ess_hat` vs `prop_hat`: 0.78.
- **Coverage:** every pombe gene and pair has a value, including the **1,113 genes with no curated cerevisiae
  ortholog**. This coverage is the point of the group: CER features are NaN for those genes.
- `XSP__pair_hat` on pombe pairs that have Costanzo ortholog data (1.18M in a random 3M-pair sample): AUPRC 0.069
  for the ortholog's frac_neg ≥ 0.5 (prevalence 0.048, lift 1.43). This is in-sample with respect to
  cerevisiae.
- pair_hat distribution: median 0.015; 99th percentile 0.065; 99.9th percentile 0.118.

## Caveats
- Transfer is modest. The value for Ryan AUPRC must be measured by an M1/M2 ablation (+XSP, with the
  gene-permuted control). It may be largely redundant with CER and PHEN for genes that have orthologs. Its main
  hope is the 1,113 ortholog-less genes and C3.
- The PPI density shift (mean degree 55 vs 3) is the main domain gap. Rank/ECDF normalisation aligns the
  marginals but not what a missing edge means.
- Minor nested-stacking caveat: inner-OOF `*_hat` for a training gene comes from a model whose targets include
  that gene's pairs with other training genes. This affects at most about 1 pair in ~4,000 per gene; C3 test
  labels are never used.
- Pair rows sample all screens (NxN, ExN, ExE, DAmP); essential context enters through `ess_hat`.
- LightGBM runs with deterministic=True and seed 0. Thread count was 6 while other jobs were running.

## Runtime (Apple M4, 10 cores; n_jobs 6; other agents running)
| step | wall s |
|---|---|
| species features (ESM PCA/cosine/mutual rank, PPI matrices, ECDFs) | 96 |
| labels + gene CV | 20 |
| pair CV (5 folds × ~1.6M training rows) | 322 |
| final fit (2.5M × 71, 762 rounds) | 62 |
| apply to 13.1M pombe pairs | 78 |
| **total** | **581 s; peak RSS 11.6 GB** (+ ~50 s on first run for the Costanzo collapse, now cached) |

## Reproduce
```bash
.venv/bin/python -I scripts/features/build_xsp.py --quick --ppi-variant {full,sub} --pair-esm-pcs {16,0}  # variant check
.venv/bin/python -I scripts/features/build_xsp.py --ppi-variant full --pair-esm-pcs 16                    # staged outputs
.venv/bin/python -m pytest -q tests/features/test_xsp.py
```
