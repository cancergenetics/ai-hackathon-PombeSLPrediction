"""Tests for scripts/toy/make_toy.py (seeded toy generator with known answers)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_toy", ROOT / "scripts" / "toy" / "make_toy.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

# Schema copied from data/processed/ryan_genes.tsv and ryan_pairs.parquet (column names / dtypes only)
GENES_COLS = ["ryan_id", "ryan_label_name", "pombase_id", "mapping", "label_name_consistent", "gene_name", "gene_type",
              "viability", "is_query", "is_array", "in_s2_rows", "n_query_alleles", "allele_classes", "query_labels",
              "merged_with_other_ryan_id"]
PAIRS_DTYPES = {
    "gene_a": "object", "gene_b": "object", "s_score": "float64", "s_raw_mean": "float64", "s_raw_ab": "float64",
    "s_raw_ba": "float64", "n_meas": "int64", "n_ab": "int64", "n_ba": "int64", "s_raw_mean_meas": "float64",
    "s_raw_min": "float64", "s_raw_max": "float64", "s_raw_mean_all": "float64", "n_meas_all": "int64",
    "in_s2": "bool", "both_orientations": "bool", "involves_merged_id": "bool", "viability_a": "object",
    "viability_b": "object", "neg_s2_lt_-2.3": "bool", "neg_raw_lt_-2.3": "bool"}


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    base = tmp_path_factory.mktemp("toy")
    args = ["--no-reference", "--reps", "20"]
    mt.main(args + ["--out", str(base / "a")])
    mt.main(args + ["--out", str(base / "b")])
    mt.main(args + ["--out", str(base / "perm"), "--permute-labels"])
    mt.main(args + ["--out", str(base / "seed2"), "--seed", "7"])
    return base


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_deterministic(toy):
    files = [p.relative_to(toy / "a") for p in (toy / "a").rglob("*") if p.is_file()]
    assert files
    for f in files:
        assert h(toy / "a" / f) == h(toy / "b" / f), f
    assert h(toy / "a/ryan_pairs.parquet") != h(toy / "seed2/ryan_pairs.parquet")


def test_schema_matches_real(toy):
    g = pd.read_csv(toy / "a/ryan_genes.tsv", sep="\t")
    assert list(g.columns) == GENES_COLS
    assert g.is_query.dtype == bool and g.n_query_alleles.dtype == np.int64
    p = pd.read_parquet(toy / "a/ryan_pairs.parquet")
    assert list(p.columns) == list(PAIRS_DTYPES)
    assert {c: str(t) for c, t in p.dtypes.items()} == PAIRS_DTYPES
    assert (p.gene_a < p.gene_b).all()
    assert not p[["gene_a", "gene_b"]].duplicated().any()
    assert set(p.gene_a) | set(p.gene_b) <= set(g.pombase_id)
    assert (p["neg_s2_lt_-2.3"] == (p.s_score < -2.3)).all()
    assert set(g.viability) <= {"viable", "inviable", "condition-dependent", "unknown"}
    assert (g.viability == "viable").mean() > 0.8 and (g.viability == "inviable").sum() > 0


def test_roles_and_fill(toy):
    g = pd.read_csv(toy / "a/ryan_genes.tsv", sep="\t").set_index("pombase_id")
    p = pd.read_parquet(toy / "a/ryan_pairs.parquet")
    qa, qb = g.is_query[p.gene_a].to_numpy(), g.is_query[p.gene_b].to_numpy()
    aa, ab = g.is_array[p.gene_a].to_numpy(), g.is_array[p.gene_b].to_numpy()
    assert ((qa & ab) | (qb & aa)).all()                     # only query x array
    assert ((p.n_ab > 0) == qa).sum() > 0 and (p.n_meas >= 1).all()
    nq, na, nboth = g.is_query.sum(), g.is_array.sum(), (g.is_query & g.is_array).sum()
    assert nq == 300 and nboth == 120 and na == 620
    # unordered pairs {i,j}, i!=j, one query and the other array: nq*na minus ordered both-role self-pairs,
    # minus double counting of both-role pairs (counted twice among nboth*(nboth-1) ordered combos -> /2)
    n_cand = nq * na - nboth - nboth * (nboth - 1) // 2
    allp = len(p) / 0.80
    assert 0.9 * n_cand < allp < 1.1 * n_cand
    genes = (toy / "a/features/genes.txt").read_text().split()
    assert len(genes) == 1200 and genes == sorted(genes) and set(g.index) < set(genes)
    assert len(set(genes) - set(g.index)) == 400      # universe-only genes never measured


def test_feature_store_format(toy):
    f = toy / "a/features"
    genes = (f / "genes.txt").read_text().split()
    G = len(genes)
    man = json.loads((f / "manifest.json").read_text())
    for p in (f / "gene").glob("*.parquet"):
        df = pd.read_parquet(p)
        assert set(df.index) <= set(genes)
        assert all(c.startswith(p.stem + "__") for c in df.columns)
        assert all(t == np.float32 for t in df.dtypes)
        assert man[f"gene/{p.name}"]["sha256"] == h(p)
    for sub in ("sim", "pair"):
        for p in (f / sub).glob("*.npy"):
            M = np.load(p)
            assert M.shape == (G, G) and M.dtype == np.float32
            assert np.isnan(np.diag(M)).all()
            assert np.array_equal(M, M.T, equal_nan=True)
            assert np.isnan(M).mean() < 0.4
            assert man[f"{sub}/{p.name}"]["sha256"] == h(p)
    for k, v in man.items():
        assert set(v) == {"group", "sources", "script", "sha256"}


def test_prevalence_in_band(toy):
    p = pd.read_parquet(toy / "a/ryan_pairs.parquet")
    assert 0.016 < p["neg_s2_lt_-2.3"].mean() < 0.024
    ka = json.loads((toy / "a/known_answers.json").read_text())
    assert abs(ka["prevalence_expected_analytic"] - 0.02) < 1e-3     # sigma solved exactly


def test_known_answers_ordering(toy):
    ka = json.loads((toy / "a/known_answers.json").read_text())
    A = ka["auprc"]
    for cls in ("all", "C1", "C2", "C3"):
        assert A["oracle"][cls] > A["feature_informed_oracle"][cls] > A["propensity_only_oracle"][cls] \
               > 1.5 * A["prevalence_baseline"][cls], cls
    # the oracle must not be a perfect score (noise matters) nor trivially weak
    assert 0.3 < A["oracle"]["all"] < 0.95
    # expected-over-redraws agree with realized within a few sd (C3 is noisy)
    E = ka["auprc_expected_over_noise_redraws"]["oracle"]
    for cls in ("all", "C1", "C2", "C3"):
        assert abs(E[cls]["mean"] - A["oracle"][cls]) < 5 * E[cls]["sd"] + 0.02
    held = ka["heldout"]
    assert held["n_genes"] == round(0.2 * 800)
    sizes = ka["class_sizes"]
    assert sizes["C1"]["n"] + sizes["C2"]["n"] + sizes["C3"]["n"] == sizes["all"]["n"]


def test_permuted_variant(toy):
    a = pd.read_parquet(toy / "a/ryan_pairs.parquet")
    pm = pd.read_parquet(toy / "perm/ryan_pairs.parquet")
    assert (a[["gene_a", "gene_b"]].to_numpy() == pm[["gene_a", "gene_b"]].to_numpy()).all()
    assert np.allclose(np.sort(a.s_score), np.sort(pm.s_score))          # same multiset
    assert not np.allclose(a.s_score, pm.s_score)
    assert a["neg_s2_lt_-2.3"].sum() == pm["neg_s2_lt_-2.3"].sum()
    for f in (toy / "a/features").rglob("*"):
        if f.is_file() and f.name != "manifest.json":
            assert h(f) == h(toy / "perm/features" / f.relative_to(toy / "a/features"))
    # truth-based scores carry no information on permuted labels
    ka = json.loads((toy / "perm/known_answers.json").read_text())
    nc = ka["negative_control"]["per_class"]
    for cls in ("all", "C1", "C2"):
        lo, hi = nc[cls]["band_99"]
        assert lo - 0.002 <= ka["auprc"]["oracle"][cls] <= hi + 0.002, cls
    assert ka["auprc"]["oracle"]["all"] < 0.05


def test_oracle_recomputable_from_truth(toy):
    """Independent recomputation of the oracle AUPRC from the saved truth + pairs."""
    from sklearn.metrics import average_precision_score
    p = pd.read_parquet(toy / "a/ryan_pairs.parquet")
    t = pd.read_parquet(toy / "a/_truth/pair_signal.parquet")
    assert (p.gene_a.to_numpy() == t.gene_a.to_numpy()).all()
    ap = average_precision_score(p["neg_s2_lt_-2.3"], t.signal)
    ka = json.loads((toy / "a/known_answers.json").read_text())
    assert ap == pytest.approx(ka["auprc"]["oracle"]["all"])
