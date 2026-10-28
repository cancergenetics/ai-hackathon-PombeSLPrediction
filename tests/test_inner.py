"""(e) Nested inner split tests: inner-training pairs never include a T_j gene or an H_j pair; C2-like
pairs kept once; gene-grouped stacker validation. Each checker must flag a broken variant."""
import pandas as pd
import pytest

from src import data
from src import splits as S


def broken_inner_keeps_holdout(inner):
    f = inner[0]
    return [S.InnerFold(f.j, f.genes, pd.concat([f.train, f.holdout]), f.eval, f.holdout)] + inner[1:]


def broken_inner_keeps_fold_gene(inner):
    f = inner[0]
    leak = f.eval[f.eval.part == "C3"].head(1)
    return [S.InnerFold(f.j, f.genes, pd.concat([f.train, leak]), f.eval, f.holdout)] + inner[1:]


def broken_inner_duplicates_c2(inner):
    f, g = inner[0], inner[1]
    dup = g.eval[g.eval.part == "C2"].head(1).assign(inner_fold=f.j)
    return [S.InnerFold(f.j, f.genes, f.train, pd.concat([f.eval, dup]), f.holdout)] + inner[1:]


def broken_stacker_no_gene_grouping(inner):
    rows = pd.concat([f.eval for f in inner], ignore_index=True)
    return [(f.j, rows[rows.inner_fold != f.j], rows[rows.inner_fold == f.j]) for f in inner]


def _check(train):
    inner = data.inner_splits(train)
    assert S.check_inner_split(inner) == []
    for f in inner:
        T = set(f.genes)
        n_in = f.eval.gene_a.isin(T).astype(int) + f.eval.gene_b.isin(T).astype(int)
        assert (n_in[f.eval.part == "C3"] == 2).all()
        assert (n_in[f.eval.part == "C2"] == 1).all()
        assert (n_in[f.eval.part == "C1"] == 0).all()
        assert not (f.train.gene_a.isin(T) | f.train.gene_b.isin(T)).any()
    # every training gene in exactly one inner fold
    allg = [g for f in inner for g in f.genes]
    assert len(allg) == len(set(allg)) == len(set(train.gene_a) | set(train.gene_b))
    # C2-like pairs: every cross-fold pair of the training set is kept exactly once
    gf = {g: f.j for f in inner for g in f.genes}
    cross = train[train.gene_a.map(gf) != train.gene_b.map(gf)]
    c2 = pd.concat([f.eval[f.eval.part == "C2"] for f in inner])
    assert len(c2) == len(cross)
    groups = data.gene_grouped_validation(inner)
    assert S.check_gene_grouped(groups, inner) == []
    # broken variants are caught
    assert S.check_inner_split(broken_inner_keeps_holdout(inner))
    assert S.check_inner_split(broken_inner_keeps_fold_gene(inner))
    assert S.check_inner_split(broken_inner_duplicates_c2(inner))
    assert S.check_gene_grouped(broken_stacker_no_gene_grouping(inner), inner)
    # reproducible
    again = data.inner_splits(train)
    assert all(a.eval.equals(b.eval) for a, b in zip(inner, again))


def test_inner_synthetic(synth_env):
    tr, _ = data.fold_split(None, 0)
    _check(tr)


def test_inner_real(real_env):
    tr, _ = data.fold_split(None, 1)
    _check(tr)


def test_inner_eval_rows_carry_true_hypomorph(synth_env):
    tr, _ = data.fold_split(None, 0)
    rows = pd.concat([f.eval for f in data.inner_splits(tr)])
    assert rows.hypomorph_a.sum() + rows.hypomorph_b.sum() > 0
    with pytest.raises(PermissionError):
        data._read(data.protocol.splits_dir() / "labels_lockbox.parquet")
