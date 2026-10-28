"""(a) Metric tests: our AP vs sklearn; pooled class-weighted AP on a hand-computable example;
broken variant (flipped labels) must be caught; fast bootstrap AP == brute force sklearn."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import evaluate as E

W3 = {"primary": {"C1": 0.25, "C2": 0.5, "C3": 0.25}}


def hand_example():
    # C1: (0.9,+), (0.2,-)  C2: (0.8,-), (0.7,+), (0.1,-), (0.05,-)  C3: (0.6,+)
    return pd.DataFrame({
        "gene_a": list("abcdefg"), "gene_b": list("hijklmn"), "fold": 0,
        "part": ["C1", "C1", "C2", "C2", "C2", "C2", "C3"],
        "score": [0.9, 0.2, 0.8, 0.7, 0.1, 0.05, 0.6],
        "y": [True, False, False, True, False, False, True]})


# Hand computation with class weights 0.25/0.5/0.25 -> row weights C1 .125, C2 .125, C3 .25:
# ranking .9(+.125) .8(-.125) .7(+.125) .6(+.25) ...; total positive weight .5
# AP = .25*1 + .25*(.25/.375) + .5*(.5/.625) = .25 + 1/6 + .4 = 49/60
HAND_AP = 49 / 60


def pooled_ap(df, cw):
    return average_precision_score(df.y, df.score, sample_weight=E.class_row_weights(df.part.to_numpy(), cw))


def pooled_ap_flipped(df, cw):   # deliberately broken: labels flipped
    return average_precision_score(~df.y, df.score, sample_weight=E.class_row_weights(df.part.to_numpy(), cw))


def check_metric(fn) -> bool:
    return abs(fn(hand_example(), W3["primary"]) - HAND_AP) < 1e-12


def test_pooled_weighted_ap_hand_example():
    assert check_metric(pooled_ap)
    m = E.pooled_metrics(hand_example(), weightings=W3)
    assert m["primary"] == pytest.approx(HAND_AP, abs=1e-12)
    assert m["mean"]["prevalence_primary"] == pytest.approx(0.5)


def test_broken_metric_flipped_labels_is_caught():
    assert not check_metric(pooled_ap_flipped)


def test_class_row_weights_sum_to_class_weights():
    part = np.array(["C1"] * 3 + ["C2"] * 10 + ["C3"] * 2)
    w = E.class_row_weights(part, W3["primary"])
    for c, v in W3["primary"].items():
        assert w[part == c].sum() == pytest.approx(v)


@pytest.mark.parametrize("seed", range(5))
def test_ap_matrix_matches_sklearn_with_ties_and_weights(seed):
    rng = np.random.default_rng(seed)
    n = 500
    y = rng.random(n) < 0.1
    s = np.round(rng.random(n), 1)            # heavy ties
    w = rng.random((n, 3)) * (rng.random((n, 3)) > 0.2)   # some zero weights
    order, ends = E._sort_groups(s)
    got = E.ap_weighted_matrix(y[order], w[order], ends)
    for j in range(3):
        assert got[j] == pytest.approx(average_precision_score(y, s, sample_weight=w[:, j]), abs=1e-12)


def _toy_folds(seed=0, n_genes=50, n=3000):
    rng = np.random.default_rng(seed)
    genes = [f"g{i:03d}" for i in range(n_genes)]
    rows = []
    for k in range(2):
        for _ in range(n):
            a, b = sorted(rng.choice(genes, 2, replace=False))
            rows.append((a, b, k, rng.choice(["C1", "C2", "C3"], p=[.3, .5, .2]), rng.random() < .1,
                         np.round(rng.random(), 2)))
    df = pd.DataFrame(rows, columns=["gene_a", "gene_b", "fold", "part", "y", "score"])
    return df.drop_duplicates(["gene_a", "gene_b", "fold"]).reset_index(drop=True), pd.Index(genes)


def test_bootstrap_unit_counts_reproduce_point_statistic():
    df, idx = _toy_folds()
    reps = E.bootstrap_replicates(df, np.ones((2, len(idx)), int), idx, W3)
    assert np.allclose(reps["primary"], E.pooled_metrics(df, W3)["primary"], atol=1e-12)


def test_bootstrap_replicate_matches_bruteforce_sklearn():
    df, idx = _toy_folds(seed=3)
    cnt = np.random.default_rng(9).poisson(1, (4, len(idx)))
    reps = E.bootstrap_replicates(df, cnt, idx, W3)["primary"]
    for r in range(4):
        vals = []
        for _, d in df.groupby("fold"):
            base = cnt[r][idx.get_indexer(d.gene_a)] * cnt[r][idx.get_indexer(d.gene_b)]
            p = d.part.to_numpy()
            w = np.zeros(len(d))
            for c, v in W3["primary"].items():
                m = p == c
                w[m] = base[m] * v / base[m].sum()
            vals.append(average_precision_score(d.y, d.score, sample_weight=w))
        assert reps[r] == pytest.approx(np.mean(vals), abs=1e-10)
