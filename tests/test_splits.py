"""(c) Outer split tests, on the synthetic root (always) and the real splits (when built).
Every checker is shown to flag a deliberately broken variant."""
import numpy as np
import pandas as pd
import pytest

from src import data, protocol
from src import splits as S


def lockbox_frames(splits_dir):
    lock = pd.read_parquet(splits_dir / "labels_lockbox.parquet")    # test-only read (whitelisted: tests/)
    genes = set(lock.gene_a[lock.part == "C3"]) | set(lock.gene_b[lock.part == "C3"])
    c2 = lock[lock.part == "C2"]
    dev = data.labels_dev()
    devg = set(dev.gene_a) | set(dev.gene_b)
    genes |= (set(c2.gene_a) | set(c2.gene_b)) - devg
    return lock, genes


# ----------------------------------------------------------------------------------- broken variants
def broken_split_leaks_gene(train, ev, fold_genes):
    """Puts back one C2 pair into training (leaks a held-out gene)."""
    leak = ev[ev.part == "C2"].head(1)
    return pd.concat([train, leak[["gene_a", "gene_b"]]], ignore_index=True), ev


def broken_split_pair_overlap(train, ev, fold_genes):
    """Adds a training pair to the eval set as C1 (pair in both train and eval)."""
    extra = train.head(1)[["gene_a", "gene_b"]].assign(part="C1")
    return train, pd.concat([ev, extra], ignore_index=True)


def broken_dev_with_lockbox_gene(dev, lock):
    return pd.concat([dev, lock[lock.part != "C1"].head(3)], ignore_index=True)


def broken_dev_with_lockbox_c1(dev, lock):
    return pd.concat([dev, lock[lock.part == "C1"].head(3)], ignore_index=True)


# ----------------------------------------------------------------------------------- shared checks
def _check_all_folds(seed):
    n = protocol.load()["dev_cv"]["n_folds"]
    for k in range(n):
        tr, ev = data.fold_split(seed, k)
        assert S.check_fold_split(tr, ev, data.fold_genes(seed, k)) == [], (seed, k)
        # broken variants must be caught
        assert S.check_fold_split(*broken_split_leaks_gene(tr, ev, None), data.fold_genes(seed, k))
        assert S.check_fold_split(*broken_split_pair_overlap(tr, ev, None), data.fold_genes(seed, k))


def _check_dev_vs_lockbox():
    sd = protocol.splits_dir()
    lock, lgenes = lockbox_frames(sd)
    dev = data.labels_dev()
    c1 = lock[lock.part == "C1"]
    assert len(c1) > 0
    assert S.check_dev_labels(dev, lgenes, c1) == []
    assert S.check_dev_labels(broken_dev_with_lockbox_gene(dev, lock), lgenes, c1)
    assert S.check_dev_labels(broken_dev_with_lockbox_c1(dev, lock), lgenes, c1)


def _check_masks_and_hypomorph(seed=None):
    dev = data.labels_dev()
    key = S.pair_keys(dev.gene_a, dev.gene_b)
    for k in range(protocol.load()["dev_cv"]["n_folds"]):
        tr, ev = data.fold_split(seed, k)
        assert (ev.hypomorph_a == 0).all() and (ev.hypomorph_b == 0).all()      # DESIGN §1 v3.1
        assert "y" not in ev and "s_score" not in ev                            # no eval labels exposed
        m = dev.set_index(key).loc[S.pair_keys(ev.gene_a, ev.gene_b)]
        assert m.eval_eligible.all()
        t = dev.set_index(key).loc[S.pair_keys(tr.gene_a, tr.gene_b)]
        assert not t.train_drop.any()
        assert set(ev.part) <= set(S.PARTS)


# ----------------------------------------------------------------------------------- synthetic
def test_synthetic_fold_splits(synth_env):
    for seed in data.split_seeds():
        _check_all_folds(seed)


def test_synthetic_dev_has_no_lockbox(synth_env):
    _check_dev_vs_lockbox()


def test_synthetic_masks_hypomorph(synth_env):
    _check_masks_and_hypomorph()
    tr, _ = data.fold_split(None, 0)
    assert tr.hypomorph_a.sum() + tr.hypomorph_b.sum() > 0       # true flags present on training rows


def test_synthetic_damp_flags_per_pair(synth_env):
    # flags are checked on all S2 pairs (dev + lockbox file; test-only read)
    dev = pd.concat([data.labels_dev(), pd.read_parquet(protocol.splits_dir() / "labels_lockbox.parquet")])
    s = synth_env
    for g in s["damp_only_query"]:
        sub = dev[(dev.gene_a == g) | (dev.gene_b == g)]
        partner = np.where(sub.gene_a == g, sub.gene_b, sub.gene_a)
        is_arr = np.array([s["role"][x] == "array" for x in partner])
        # with array-only partners the pair is DAmP-only; with a deletion-query partner it is mixed
        assert is_arr.any() and sub.damp_only.to_numpy()[is_arr].all()
        hyp = np.where(sub.gene_a == g, sub.hypomorph_a, sub.hypomorph_b)
        assert (hyp == 1).all()
    g = s["damp_mixed_query"]
    sub = dev[(dev.gene_a == g) | (dev.gene_b == g)]
    assert sub.damp_mixed.all() and not sub.damp_only.any()
    hyp = np.where(sub.gene_a == g, sub.hypomorph_a, sub.hypomorph_b)
    assert (hyp == 1).all()
    assert not (dev.gene_a.isin([s["merged"]]) | dev.gene_b.isin([s["merged"]]))[dev.eval_eligible].any()


def test_synthetic_folds_partition_dev_genes(synth_env):
    dev = data.labels_dev()
    devg = set(dev.gene_a) | set(dev.gene_b)
    for seed in data.split_seeds():
        f = data.dev_folds(seed)
        assert set(f.index) == devg and f.index.is_unique
        sizes = f.value_counts()
        assert sizes.max() - sizes.min() <= 1
    a, b = (data.dev_folds(s) for s in data.split_seeds())
    assert not a.equals(b)


def test_fold_labels_refuses_eval(synth_env):
    with pytest.raises(PermissionError):
        data.fold_labels(None, 0, "C2")


# ----------------------------------------------------------------------------------- real data
def test_real_dev_has_no_lockbox(real_env):
    _check_dev_vs_lockbox()
    lb = set((protocol.ROOT / protocol.load()["lockbox"]["genes_file"]).read_text().split())
    dev = data.labels_dev()
    assert not (dev.gene_a.isin(lb) | dev.gene_b.isin(lb)).any()


def test_real_fold_splits_primary_seed(real_env):
    _check_all_folds(None)


def test_real_fold_splits_alt_seed(real_env):
    _check_all_folds(data.split_seeds()[1])


def test_real_masks_hypomorph(real_env):
    _check_masks_and_hypomorph()
