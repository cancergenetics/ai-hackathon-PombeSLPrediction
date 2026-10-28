# Rarely-seen genes: degree-stratified C1/C2 and E-FEWSHOT

Status 2026-10-28. Branch `fewshot` (worktree `../psl_wt_fewshot`, data via `PSL_DATA_ROOT` = main tree).
Report-only: no model selection, protocol untouched. Pre-statement: `config/fewshot.yaml` (commit 8e98fa4, before any
few-shot run). Code: `scripts/rarely_seen.py`, `scripts/run_fewshot.py`. Aggregates: `results/rarely_seen/`
(`degree_bins.{json,tsv}`, `fewshot.json`, `fewshot_runtime.json`, `rarely_seen.png`). All CIs are 95%
Poisson(1) gene bootstraps (1,000 replicates, the protocol seed) built from `src/evaluate.py` functions.

Incumbent per PI decision: **M2r_huber** (`M2r_huber__20261028T205545__7b5e9e`, 3 seeds, Huber stacker).
Extra columns: M2r_bag5_huber, M2 (`M2__20261028T183713__024416`), B1 (`B1_additive_logit__20261028T183903__da755f`),
B4 (`B4_logistic_MF__20261028T183958__e1e7b0`).

## Key caveat: the dev CV contains no rarely-seen gene
Every seen gene in dev has at least **106 training pairs** (median 423 for C1's rarer gene, 826 for C2's seen gene;
max about 1,300). Ryan measured every gene against hundreds of partners, so "few training pairs" cannot be
sampled from the dev folds. The degree analysis therefore shows sensitivity within the well-measured range
(106-1,300 pairs); it does **not** extrapolate to genes with 5-20 measurements. Only E-FEWSHOT speaks to that.

## 1. AUPRC vs seen-gene degree (no retraining; per-fold AP, mean over 5 folds)
Seen-gene degree = number of training pairs of the seen gene in that outer fold (C1: the smaller of the two genes).
Bins are 4 quantile bins of the pooled seen-gene degree (edges 419 / 691 / 1022), fixed in the config.
Positive-count bins: training positives of the seen gene, [0], [1-2], [3-9], [10+]. AUPRC [95% CI]; prevalence in bin.

**C2 (one gene unseen), by seen gene's training-pair count**
| bin | prevalence | B1 = B4 | M2 | M2r_huber (incumbent) | M2r_bag5_huber |
|---|---|---|---|---|---|
| < 419 (n 203k, 4.5k pos) | 0.022 | 0.069 [0.060, 0.079] | 0.144 [0.120, 0.175] | **0.152 [0.126, 0.184]** | 0.153 |
| 419-690 (228k, 3.7k) | 0.016 | 0.062 [0.046, 0.080] | 0.122 [0.100, 0.156] | **0.128 [0.103, 0.164]** | 0.129 |
| 691-1021 (276k, 7.1k) | 0.026 | 0.104 [0.085, 0.120] | 0.188 [0.161, 0.220] | **0.195 [0.167, 0.230]** | 0.196 |
| >= 1022 (290k, 5.3k) | 0.018 | 0.064 [0.050, 0.077] | 0.165 [0.133, 0.198] | **0.174 [0.140, 0.209]** | 0.176 |

**C1 (both seen), by the rarer gene's training-pair count**
| bin | prevalence | B1 | B4 | M2 | M2r_huber (incumbent) |
|---|---|---|---|---|---|
| < 419 (n 84k, 1.9k pos) | 0.022 | 0.202 [0.170, 0.252] | 0.224 [0.188, 0.279] | 0.225 [0.192, 0.280] | **0.236 [0.201, 0.291]** |
| 419-690 (84k, 1.3k) | 0.015 | 0.190 [0.149, 0.251] | 0.221 [0.171, 0.283] | 0.219 [0.175, 0.278] | **0.242 [0.190, 0.307]** |
| 691-1021 (23k, 0.8k) | 0.034 | 0.255 [0.190, 0.337] | 0.288 [0.222, 0.372] | 0.345 [0.272, 0.435] | **0.366 [0.291, 0.458]** |
| >= 1022 (9k, 0.3k) | 0.029 | 0.267 [0.201, 0.396] | 0.294 [0.215, 0.429] | 0.368 [0.280, 0.497] | **0.411 [0.303, 0.545]** |

**By the seen gene's number of training positives (hit count)**
| bin | C2 prevalence | C2 B1 | C2 M2 | C2 incumbent | C1 prevalence | C1 B4 | C1 incumbent |
|---|---|---|---|---|---|---|---|
| 0 | 0.0025 | 0.003 | 0.006 | 0.008 [0.005, 0.023] | 0.0028 | 0.020 | 0.027 [0.012, 0.118] |
| 1-2 | 0.0042 | 0.009 | 0.022 | 0.022 [0.015, 0.042] | 0.0052 | 0.053 | 0.056 [0.039, 0.109] |
| 3-9 | 0.0093 | 0.021 | 0.050 | 0.055 [0.043, 0.079] | 0.0130 | 0.111 | 0.130 [0.098, 0.193] |
| >= 10 | 0.0431 | 0.091 | 0.185 | 0.194 [0.168, 0.220] | 0.0538 | 0.285 | 0.335 [0.288, 0.386] |

Reading:
- **By pair count, nothing monotone**: C2 AUPRC is 0.128-0.195 across bins with overlapping CIs, and tracks the
  bin's prevalence (non-monotone too: 0.022, 0.016, 0.026, 0.018), i.e. it is not a smooth degree effect.
  Within 106-1,300 measured pairs the number of measurements is not what limits accuracy. C1 AUPRC rises with
  degree (0.24 to 0.41) but the CIs overlap and the highest bin is small (9k pairs, 300 positives).
- **By hit count, strongly monotone**: lift over bin prevalence for C2 is about 3x (0 hits), 5x (1-2),
  6x (3-9) and 4.5x (10+). The seen gene's hit count is the dominant determinant of absolute AUPRC; it
  mostly moves the prevalence (0.25% to 4.3%). For a seen gene with no positive the incumbent reaches only
  AUPRC 0.008 in C2, but that is still 3x prevalence.
- The incumbent beats B1/B4 in every bin; M2r_huber beats M2 by 0.003-0.04, all within CI. B1 and B4 are
  identical in C2 by construction (an unseen gene contributes nothing; both rank by the seen gene's beta).

## 2. E-FEWSHOT (k = 0, 5, 20)
Design (pre-stated in `config/fewshot.yaml`): for each outer fold, each held-out gene's eval pairs (C2 and
C3) get a fixed random order; reveal(k) = the first k pairs of every held-out gene (so reveal(5) is contained in
reveal(20); a C3 pair is revealed if either gene draws it). Revealed pairs (labels from `labels_dev`) are added
to the outer training set as ordinary rows and the **whole nested incumbent pipeline is re-run** (nested RYAN/NBR/GP
features, inner splits, Huber stacker, seeds 0-2, 2 threads). Revealed pairs are removed from evaluation. The
evaluation set E (C2+C3 pairs minus reveal(20)) is identical for k = 0, 5, 20, and k = 0 is the incumbent's
existing OOF predictions on E (same seeds 0-2, averaged). **Full 5 folds x 3 seeds** were run (the pre-stated
plan; the 2-fold fallback was not needed), one reveal draw per fold. Revealed per gene: k = 5 gives mean 5.5
pairs and 0.13 positives (12% of genes have >= 1 positive); k = 20 gives 22 pairs and 0.44 positives (28% of genes).

| k | C2 AUPRC | C3 AUPRC | pooled C2+C3 (protocol class weights, C1 absent) |
|---|---|---|---|
| 0 | 0.1675 [0.145, 0.192] | 0.0937 [0.078, 0.122] | 0.1490 [0.128, 0.171] |
| 5 | 0.1602 [0.137, 0.184] | 0.0981 [0.079, 0.127] | 0.1427 [0.122, 0.164] |
| 20 | 0.1824 [0.157, 0.208] | 0.1362 [0.114, 0.169] | 0.1698 [0.145, 0.193] |

Prevalence 0.021 (pooled). n = 193.6k C2 pairs (4.0k pos) and 23.4k C3 pairs (500 pos) per fold on average.

Paired gene-bootstrap differences (same rows, same replicates):
| contrast | C2 | C3 | pooled |
|---|---|---|---|
| k5 - k0 | -0.0073 [-0.016, +0.002] | +0.0044 [-0.013, +0.023] | -0.0062 [-0.015, +0.003] |
| k20 - k0 | +0.0149 [+0.005, +0.025] | +0.0425 [+0.021, +0.066] | +0.0208 [+0.010, +0.031] |
| k20 - k5 | | | +0.0270 [+0.019, +0.036] |

Reading:
- **k = 20 helps, notably C3 (+0.043, +45% relative)**, i.e. for a gene with a few dozen measurements the
  pipeline does use them. C2 gains +0.015 (+9%).
- **k = 5 does not help and slightly hurts C2** (-0.007, CI includes 0). The per-seed point estimates agree in sign
  (pooled -0.004 / -0.008 / -0.007 for seeds 0/1/2; the seed-to-seed spread of a single level is about 0.002),
  and the fold-level effect is mixed (pooled k5 - k0: +0.005, -0.016, +0.004, +0.008, -0.032 for folds 0-4; the largest drop is fold 4, 0.154 to 0.122). Caveat: one reveal draw per fold, so reveal-sampling noise is not separated from the effect.
- Hypothesis (not tested): the stacker is trained only on inner folds where held-out genes have **zero** pairs
  or hundreds of pairs, never a handful. With 5 pairs the gene counts as "seen" and the features that use a seen
  gene's own beta (`gp_beta_*`, `gp_best_*`, B1/NBR rates) take a very noisy estimate (92% of k = 5 genes have no revealed
  positive, so beta is pushed down) in place of the regularised GP prediction. Consistent with this, at k = 20 C2 pairs
  whose held-out gene got >= 1 revealed positive gain (0.214 to 0.224) while those with none lose slightly
  (0.116 to 0.109); at k = 5 both groups are lower (0.234 to 0.217; 0.147 to 0.139). Differences are within CI.
- **Practical reading for the brief**: a newly measured gene is predicted no better than an unmeasured one until
  roughly 20 measurements, with the current pipeline. A cheap follow-up (not run, would need selection discipline):
  treat genes with fewer than about 20 measured partners as unseen (use the k = 0 path), or shrink beta by pair
  count / train the stacker with a simulated k-pair inner state.

Runtime (2 threads, other agents running): 4,471 s summed for 10 fold-k runs (about 340-540 s each: feature build
120-190 s, 3 stacker fits 190-270 s); peak RSS 11.2 GB. Selftest of the reveal sets (nesting, disjointness,
row counts) in `python scripts/run_fewshot.py --selftest`. Per-seed pair-level predictions stay in the git-ignored
`<data root>/preds/fewshot/`.

## Limits
- Not the headline metric: these are C2/C3-only pooled APs on the remaining pairs of held-out genes, so absolute
  values are not comparable with the primary 0.1700/0.1791 (which includes C1).
- k = 0 here is the incumbent's OOF predictions, i.e. trained on the training set without any revealed pairs; k > 0
  models are separate fits, so the contrast contains fit-to-fit variation (seed spread about 0.002).
- Revealed pairs are drawn from the held-out gene's *measured* pairs at random (mostly query x array partners as in
  Ryan); no choice of which pairs to measure is modelled. k = 100 was cut in DESIGN.
- Degree bins cover 106-1,300 training pairs only (see caveat).
