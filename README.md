# Predicting synthetic lethality across all non-essential *S. pombe* gene pairs

> **Disclaimer**
>
> This is an exercise in assessing the bioinformatics capabilities of agentic AI. This exercise was performed by the Ryan lab at University College Dublin, collaborating as part of a hackathon run by the Beltrao Lab at ETH Zurich, where we used Claude to run bioinformatics projects. All results in this project were generated entirely by AI with minimal human intervention. This is not intended to be viewed as a traditional publication and bioinformatics tool.
>
> The goal of this specific project is to develop an approach to predict synthetic lethal (strong negative genetic) interactions between all pairs of non-essential genes in fission yeast (*Schizosaccharomyces pombe*), trained on the Ryan et al. 2012 genetic interaction map.
>
> - **Agents:** a Claude Opus 5.5 agent acting as principal investigator in Claude Code, which delegated work to 41 subagents (33 Claude Opus 5.5 and 8 Claude Sonnet 5.5). The subagents did literature review, data collection, design critique, implementation, independent verification and report writing.
> - **Time:** about 11.5 hours of agent work: about 10.5 hours of research on 28–29 October 2026, plus about 1 hour for the write-up.
> - **Machine:** a laptop with an Apple M4 chip (10 CPU cores, 32 GB RAM, macOS). All model training ran on the CPU; only the ESM-2 protein embeddings used the Apple GPU.
>
> Currently none of the results have been verified by a human. We plan to verify the agent's methodology and benchmarking, and we will update this section accordingly.
>
> The rest of this repository is entirely AI generated. It includes code for implementation of the predictive model and a AI-written report in the style of a journal article, detailing the final method, the path followed to reach it, and benchmarking against other methods.

## What is here

| Start with | What it is |
| --- | --- |
| [`docs/briefing/BRIEFING.md`](docs/briefing/BRIEFING.md) | The journal-style write-up for a general scientific audience, with figures. |
| [`REPORT.md`](REPORT.md) | The technical final report: approach, results, runtime, limitations and follow-ups. |
| [`PROMPT.md`](PROMPT.md) | The brief the agents were given, verbatim. |
| [`RESEARCH_TREE.md`](RESEARCH_TREE.md) | The research record: every idea tried, its outcome, and a dated decision log. |
| [`DESIGN.md`](DESIGN.md) | The frozen study design, after three rounds of review by agents (reviews are in `reviews/`). |
| [`LITERATURE.md`](LITERATURE.md) | The literature review the design was built on. |
| [`DATA.md`](DATA.md) | Every dataset used or excluded, with source, version, checksum and licence. |
| [`LESSONS.md`](LESSONS.md) | Process rules the agent team learned while working. |

## Headline result

The test set is a held-out "lockbox" of 480 genes, set aside before modelling and evaluated once under a preregistered plan. Scores are pooled AUPRC on strong negative interactions (Ryan S-score < −2.3), with classes weighted to the mix of unmeasured pairs. The pair classes are:

- **C1:** both genes seen in training.
- **C2:** one gene unseen.
- **C3:** both genes unseen.

| Model | AUPRC [95% CI] | C1 | C2 | C3 |
| --- | --- | --- | --- | --- |
| **Final model (M2r_huber)** | **0.211 [0.179, 0.247]** | 0.285 | 0.210 | 0.125 |
| M2 (binary stacker) | 0.201 | 0.266 | 0.199 | 0.127 |
| M1 (label-free features only) | 0.162 | 0.214 | 0.154 | 0.122 |
| B4 matrix factorisation | 0.110 | 0.247 | 0.080 | 0.023 |
| B1 gene-propensity baseline | 0.106 | 0.223 | 0.080 | 0.023 |
| Random | 0.021 | | | |

In development cross-validation the final model scored 0.179, so a realistic expectation for new genes is about 0.18–0.21.

**Model.** The final model is a nested-stacked LightGBM model. Its inputs are:
- label-free gene and pair features: phenomics, budding-yeast orthologs from Costanzo et al. 2016, protein interactions, expression, proteomes, localisation, sequence;
- cross-fitted features derived from the training labels.

Its stacker is trained with a Huber loss on the continuous S-score.

**Caveats.** Most of the predictive signal is gene-level, i.e. which genes interact a lot. See the limitations in `REPORT.md` and the briefing.

## Repository layout

| Path | What it holds |
| --- | --- |
| `src/` | Library code: splits, evaluation, features, models, deliverable. |
| `scripts/` | Data download and processing, feature builders, model runs, the lockbox evaluation, all-pair prediction. |
| `config/` | Protocol and settings: `protocol.yaml` is the frozen evaluation protocol, plus the data allowlist, model specifications and the lockbox preregistration. |
| `tests/` | pytest suite, including toy data with known answers, mutation tests and lints that restrict access to labels. |
| `results/` | Aggregate results only: run registry, comparisons, per-run JSON records, lockbox result and access log. |
| `reviews/` | Design reviews and independent verification reports, each with its scripts. |
| `docs/` | Per-step write-ups, interface notes and the briefing. |

## Data are not included

No data files are in this repository: no raw data, no processed data, no pair-level predictions.

- `data/` is git-ignored.
- `DATA.md` and `config/data_allowlist.yaml` list every source with its URL, version and sha256 checksum.
- `scripts/download_core_data.py` and `scripts/download_aux_data.py` re-create `data/raw/`.

The Ryan et al. 2012 data are distributed under a publisher licence that does not allow redistribution. For that reason this repository holds only aggregate statistics derived from them. The final predictions for all 7,693,003 pairs can be regenerated with `scripts/predict_all_pairs.py`.

## Reproducing

```bash
bash scripts/setup_env.sh      # Python 3.10 venv; includes a libomp fix needed for LightGBM on macOS
# Data: see DATA.md for the full ordered list of download, build and verification scripts.
# Features: scripts/features/build_*.py. Splits: scripts/make_splits.py.
python scripts/run_m2plus.py --stages run --variants M2r --seeds ladder --objective huber   # final model, dev CV
python scripts/predict_all_pairs.py --train-set dev                                         # score all pairs
python -m pytest -q                                                                          # test suite
```

The lockbox evaluation (`scripts/lockbox_eval.py`) has already been run once. By design it refuses a second run without an explicit, logged override.

## Note on the git history

This history is a filtered copy of the agents' working repository. Early in the project, a per-gene table derived from the Ryan data was committed and later deleted. It was removed from the published history, so commit hashes differ from those cited in the documents and run records, including the lockbox preregistration. [`docs/COMMIT_MAP.tsv`](docs/COMMIT_MAP.tsv) maps every original commit hash to its published equivalent. The final file tree is identical to the original.

Run records also contain absolute paths from the machine the agents ran on.
