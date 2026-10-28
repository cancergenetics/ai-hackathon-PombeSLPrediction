"""Shared fixtures: a small synthetic data root with known structure, split by scripts/make_splits.py.

The synthetic root is used for split/inner/evaluate tests so they run fast and never touch real
results. Tests that need the real data (data/processed/splits) skip when it is absent.
"""
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import protocol  # noqa: E402

REAL_SPLITS = ROOT / "data" / "processed" / "splits"
N_GENES = 160


def make_synthetic_root(root: Path, seed: int = 1) -> dict:
    """Ryan-like toy: 50 query+array, 30 query-only, 80 array-only genes; every query x gene pair
    measured (S2). Propensity-driven labels (~5% positives). Known design facts are returned."""
    rng = np.random.default_rng(seed)
    genes = [f"SPTOY{i:03d}" for i in range(N_GENES)]
    role = ["both"] * 50 + ["query"] * 30 + ["array"] * 80
    queries = genes[:80]
    prop = rng.normal(0, 1, N_GENES)
    pairs = {}
    for q in range(80):
        for g in range(N_GENES):
            if g == q:
                continue
            a, b = sorted((genes[q], genes[g]))
            k = (a, b)
            if k in pairs:
                pairs[k]["both"] = True
                continue
            pairs[k] = {"both": False, "s": 0.2 * rng.normal() - 0.9 * (prop[q] + prop[g]) / 2 + 0.6 * rng.normal()}
    df = pd.DataFrame([{"gene_a": a, "gene_b": b, "s_score": v["s"] * 2.2, "both_orientations": v["both"]}
                       for (a, b), v in pairs.items()])
    df["s_raw_mean"] = df.s_score * 1.3
    df["s_raw_ab"] = df.s_raw_mean + rng.normal(0, 0.5, len(df))
    df["s_raw_ba"] = np.where(df.both_orientations, df.s_raw_mean + rng.normal(0, 0.5, len(df)), np.nan)
    merged = genes[150]
    df["involves_merged_id"] = (df.gene_a == merged) | (df.gene_b == merged)
    df["in_s2"] = True
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "ryan_pairs.parquet", index=False)

    viab = np.array(["viable"] * N_GENES, dtype=object)
    viab[[3, 4, 5]] = "inviable"         # inviable query genes (DAmP)
    viab[[100, 101]] = "inviable"        # inviable array genes -> drop_from_training
    viab[[60, 61]] = "condition-dependent"
    g = pd.DataFrame({"ryan_id": genes, "pombase_id": genes, "viability": viab,
                      "is_query": [r in ("both", "query") for r in role],
                      "is_array": [r in ("both", "array") for r in role]})
    g.to_csv(root / "ryan_genes.tsv", sep="\t", index=False)
    u = pd.DataFrame({"gene": genes, "viability": viab,
                      "in_universe": [v in ("viable", "condition-dependent") for v in viab],
                      "drop_from_training": [v == "inviable" and r in ("both", "array") for v, r in zip(viab, role)]})
    u.to_csv(root / "universe_genes.tsv", sep="\t", index=False)

    # measurements: query genes 3,4,5 (inviable) and 70 (viable) screened as DAmP only; 71 has DAmP + deletion
    rows = []
    for q in range(80):
        for gi in range(N_GENES):
            if gi == q:
                continue
            classes = ["DAmP"] if q in (3, 4, 5, 70) else (["DAmP", "unlabelled"] if q == 71 else ["unlabelled"])
            for c in classes:
                rows.append((genes[q], genes[gi], c, True, False))
    m = pd.DataFrame(rows, columns=["query_gene", "array_gene", "allele_class", "main_allele", "self_pair"])
    m.to_parquet(root / "ryan_measurements.parquet", index=False)
    return {"genes": genes, "role": dict(zip(genes, role)), "damp_only_query": [genes[i] for i in (3, 4, 5, 70)],
            "damp_mixed_query": genes[71], "merged": merged, "drop": [genes[100], genes[101]]}


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthroot")
    info = make_synthetic_root(root / "data")
    old = {k: os.environ.get(k) for k in ("PSL_DATA_ROOT", "PSL_RESULTS_ROOT")}
    os.environ["PSL_DATA_ROOT"] = str(root / "data")
    os.environ["PSL_RESULTS_ROOT"] = str(root / "results")
    argv = sys.argv
    sys.argv = ["make_splits.py"]
    try:
        runpy.run_path(str(ROOT / "scripts" / "make_splits.py"), run_name="__main__")
    except SystemExit as e:
        assert not e.code, e.code
    finally:
        sys.argv = argv
    info["root"] = root
    yield info
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture()
def synth_env(synth, monkeypatch):
    monkeypatch.setenv("PSL_DATA_ROOT", str(synth["root"] / "data"))
    monkeypatch.setenv("PSL_RESULTS_ROOT", str(synth["root"] / "results"))
    return synth


@pytest.fixture()
def real_env(monkeypatch):
    if not (REAL_SPLITS / "labels_dev.parquet").exists():
        pytest.skip("real splits not built (run scripts/make_splits.py)")
    monkeypatch.delenv("PSL_DATA_ROOT", raising=False)
    monkeypatch.delenv("PSL_RESULTS_ROOT", raising=False)
    return protocol.splits_dir()
