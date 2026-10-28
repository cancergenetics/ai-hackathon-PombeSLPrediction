"""(b) Bootstrap tests on analytic toy cases."""
import numpy as np
import pandas as pd
import pytest

from src import evaluate as E

ONE = {"primary": {"C1": 0.0, "C2": 1.0, "C3": 0.0}}


def test_poisson_counts_shape_moments_and_reproducible():
    genes = [f"g{i}" for i in range(500)]
    c1, idx = E.bootstrap_counts(genes, n_boot=400, seed=11)
    c2, _ = E.bootstrap_counts(list(reversed(genes)), n_boot=400, seed=11)
    assert c1.shape == (400, 500) and list(idx) == sorted(genes)
    assert np.array_equal(c1, c2)                      # same replicates regardless of input order
    assert c1.mean() == pytest.approx(1, abs=0.02) and c1.var() == pytest.approx(1, abs=0.03)


def test_gene_bootstrap_se_of_pairwise_mean_matches_analytic():
    """Pair-weighted mean of v_ab = u_a + u_b over all pairs, weights c_a*c_b. By the delta method this is
    ~2*sum(c u)/sum(c), so SE ~ 2*sd(u)/sqrt(n)."""
    n = 300
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1, n)
    ia, ib = np.triu_indices(n, 1)
    counts, _ = E.bootstrap_counts([f"g{i:04d}" for i in range(n)], n_boot=2000, seed=5)
    v = u[ia] + u[ib]
    stats = []
    for c in counts:
        w = c[ia] * c[ib]
        stats.append(np.sum(w * v) / np.sum(w))
    assert np.std(stats, ddof=1) == pytest.approx(2 * u.std() / np.sqrt(n), rel=0.1)


def test_ap_bootstrap_se_analytic_all_ties():
    """All scores tied -> AP = weighted prevalence p_hat = sum(w y)/sum(w). Disjoint pairs (each gene in one
    pair), w = c_a c_b with E[w]=1, Var(w)=E[c^2]^2-1=3; labels fixed -> by the delta method
    SE(p_hat) ~ sqrt(Var(w) p (1-p) / n)."""
    n, p = 3000, 0.1
    genes = [f"g{i:05d}" for i in range(2 * n)]
    df = pd.DataFrame({"gene_a": genes[0::2], "gene_b": genes[1::2], "fold": 0, "part": "C2",
                       "score": 0.0, "y": np.arange(n) < p * n})
    counts, idx = E.bootstrap_counts(genes, n_boot=2000, seed=3)
    reps = E.bootstrap_replicates(df, counts, idx, ONE)["primary"]
    assert np.mean(reps) == pytest.approx(p, abs=0.003)
    assert np.std(reps, ddof=1) == pytest.approx(np.sqrt(3 * p * (1 - p) / n), rel=0.1)


def test_ap_bootstrap_perfect_ranking_has_zero_se():
    n = 400
    genes = [f"g{i:04d}" for i in range(2 * n)]
    y = np.arange(n) < 40
    df = pd.DataFrame({"gene_a": genes[0::2], "gene_b": genes[1::2], "fold": 0, "part": "C2",
                       "score": y.astype(float), "y": y})
    counts, idx = E.bootstrap_counts(genes, n_boot=200, seed=1)
    reps = E.bootstrap_replicates(df, counts, idx, ONE)["primary"]
    ok = reps[reps > 0]           # replicates where every positive got count 0 are 0 (no positives)
    assert np.allclose(ok, 1.0)
