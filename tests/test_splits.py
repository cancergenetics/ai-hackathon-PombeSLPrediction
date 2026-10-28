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


# =============================================================================================
# Hardening (verify_phaseA gaps B1, B2): eval-set completeness and non-circular DAmP exclusion
# =============================================================================================
def _keyset(df) -> pd.Index:
    return S.pair_keys(df.gene_a, df.gene_b)


def _same_keys(a: pd.Index, b: pd.Index) -> bool:
    return len(a) == len(b) and len(a.symmetric_difference(b)) == 0


def check_eval_completeness(seed, k) -> list[str]:
    """B1. Independently re-derives, from labels_dev + gene folds + the C1 hold-out file only, the pairs
    that MUST be in the eval set / training set of outer fold k, and compares with data.fold_split.
    eval == eligible C2 (one gene in fold) + eligible C3 (both) + C1 hold-out; no hold-out pair in
    training; training == neither gene in fold, not hold-out, not train_drop; every part non-empty."""
    v = []
    sd = protocol.splits_dir()
    dev = data.labels_dev()
    kd = _keyset(dev)
    gfold = data.dev_folds(seed)
    n_in = ((dev.gene_a.map(gfold) == k).astype(int) + (dev.gene_b.map(gfold) == k).astype(int)).to_numpy()
    elig = dev.eval_eligible.to_numpy(bool)
    hold = pd.read_parquet(sd / f"c1_holdout_seed{data._seed(seed)}.parquet")
    hk = _keyset(hold[hold.fold == k])
    c1 = kd.isin(hk)
    if hk.duplicated().any() or c1.sum() != len(hk):
        v.append("C1 hold-out pairs are duplicated or not all in labels_dev")
    if (n_in[c1] != 0).any() or not elig[c1].all():
        v.append("C1 hold-out contains a fold gene or an ineligible pair")
    exp = {"C1": kd[c1], "C2": kd[(n_in == 1) & elig], "C3": kd[(n_in == 2) & elig]}
    exp_train = kd[(n_in == 0) & ~c1 & ~dev.train_drop.to_numpy(bool)]
    tr, ev = data.fold_split(seed, k)
    for part, e in exp.items():
        got = _keyset(ev[ev.part == part])
        if len(e) == 0 or len(got) == 0:
            v.append(f"{part}: empty (expected {len(e)}, got {len(got)})")
        if not _same_keys(got, e):
            v.append(f"{part}: eval has {len(got)} pairs, expected {len(e)} "
                     f"({len(got.difference(e))} extra, {len(e.difference(got))} missing)")
    if len(ev) != sum(len(e) for e in exp.values()):
        v.append("eval has pairs outside C1/C2/C3 of the expected sets")
    tk = _keyset(tr)
    if tk.isin(hk).any():
        v.append(f"{int(tk.isin(hk).sum())} C1 hold-out pairs are in the training set")
    if not _same_keys(tk, exp_train):
        v.append(f"training set has {len(tk)} pairs, expected {len(exp_train)}")
    return v


def _check_completeness_all_folds(seed):
    for k in range(protocol.load()["dev_cv"]["n_folds"]):
        assert check_eval_completeness(seed, k) == [], (seed, k)


def test_synthetic_eval_completeness(synth_env):
    for seed in data.split_seeds():
        _check_completeness_all_folds(seed)


def test_real_eval_completeness(real_env):
    for seed in data.split_seeds():
        _check_completeness_all_folds(seed)


def test_completeness_checker_flags_broken_splits(synth_env):
    """The B1 checker must flag: half of C3 dropped, C1 hold-out moved to training, C1 relabelled C2."""
    seed, k = None, 0
    tr0, ev0 = data.fold_split(seed, k)
    c3 = np.flatnonzero((ev0.part == "C3").to_numpy())
    c1 = np.flatnonzero((ev0.part == "C1").to_numpy())
    assert len(c3) > 1 and len(c1) > 0
    variants = {
        "evict_c3": (tr0, ev0.drop(ev0.index[c3[::2]])),
        "c1_in_train": (pd.concat([tr0, ev0.iloc[c1][["gene_a", "gene_b"]]], ignore_index=True),
                        ev0.drop(ev0.index[c1])),
        "c1_also_in_train": (pd.concat([tr0, ev0.iloc[c1][["gene_a", "gene_b"]]], ignore_index=True), ev0),
        "c1_relabelled_c2": (tr0, ev0.assign(part=ev0.part.replace({"C1": "C2"}))),
        "control_unchanged": (tr0, pd.concat([ev0, ev0.head(0)])),   # control: identical -> must NOT be flagged
    }
    orig = data.fold_split
    try:
        for name, (t, e) in variants.items():
            data.fold_split = lambda s=None, kk=0, _t=t, _e=e: (_t, _e)
            flagged = check_eval_completeness(seed, k)
            assert bool(flagged) == (name != "control_unchanged"), (name, flagged)
    finally:
        data.fold_split = orig


def test_real_eval_counts_match_splits_summary(real_env):
    """B1: recount every fold's train / C1 / C2 / C3 pair and positive counts, and the dev eligible totals,
    from labels_dev + fold_split and require equality with results/splits_summary.json (written by
    scripts/make_splits.py from its own in-memory frames)."""
    import json
    summ = json.loads((protocol.ROOT / "results" / "splits_summary.json").read_text())
    dev = data.labels_dev()
    assert summ["dev"]["pairs"] == len(dev)
    assert summ["dev"]["positives"] == int(dev.y.sum())
    assert summ["dev"]["eval_eligible_pairs"] == int(dev.eval_eligible.sum())
    assert summ["dev"]["eval_eligible_positives"] == int(dev.y[dev.eval_eligible].sum())
    assert summ["dev"]["genes"] == len(set(dev.gene_a) | set(dev.gene_b))
    ypos = pd.Series(dev.y.to_numpy(bool), index=_keyset(dev))
    n_folds = protocol.load()["dev_cv"]["n_folds"]
    assert set(summ["folds"]) == {str(s) for s in data.split_seeds()}
    for seed in data.split_seeds():
        for k in range(n_folds):
            tr, ev = data.fold_split(seed, k)
            exp = summ["folds"][str(seed)][str(k)]
            assert len(data.fold_genes(seed, k)) == exp["genes"]
            assert len(tr) == exp["train_pairs"], (seed, k)
            assert int(tr.y.sum()) == exp["train_positives"], (seed, k)
            ykeys = ypos.loc[_keyset(ev)].to_numpy()
            for p in S.PARTS:
                m = (ev.part == p).to_numpy()
                assert int(m.sum()) == exp[f"{p}_pairs"] > 0, (seed, k, p)
                assert int(ykeys[m].sum()) == exp[f"{p}_positives"], (seed, k, p)


# ----------------------------------------------------------------------------------- B2: DAmP-only
def _independent_damp_only_keys(root) -> pd.Index:
    """Pairs whose every main-allele, non-self S1 measurement used a DAmP query allele (own derivation)."""
    m = pd.read_parquet(root / "ryan_measurements.parquet",
                        columns=["query_gene", "array_gene", "allele_class", "main_allele", "self_pair"])
    m = m[m.main_allele & ~m.self_pair & m.query_gene.notna() & m.array_gene.notna()
          & (m.query_gene != m.array_gene)]
    m = m.assign(key=S.pair_keys(m.query_gene, m.array_gene), damp=(m.allele_class == "DAmP").to_numpy())
    allp = m.groupby("key").damp.all()
    return allp.index[allp.to_numpy()]


def test_real_damp_only_exclusion_noncircular(real_env):
    dev = data.labels_dev()
    P = protocol.load()["eval"]
    assert P["require_both_in_universe"] and P["exclude_damp_only_pairs"] and P["exclude_merged_id_pairs"]
    root = protocol.data_root()
    kd = _keyset(dev)
    # (1) eligibility is exactly the protocol rule, using the stored flags ...
    rule = dev.in_universe & ~dev.damp_only & ~dev.involves_merged_id
    assert (dev.eval_eligible == rule).all()
    assert (dev.in_universe & dev.damp_only).sum() > 0         # non-vacuous: DAmP-only pairs do exist in dev
    assert (dev.in_universe & dev.involves_merged_id).sum() > 0
    # (2) ... and the flags themselves match independent recomputations from the raw inputs
    u = pd.read_csv(root / "universe_genes.tsv", sep="\t")
    U = set(u.gene[u.in_universe])
    assert ((dev.gene_a.isin(U) & dev.gene_b.isin(U)) == dev.in_universe).all()
    d_only = _independent_damp_only_keys(root)
    assert (kd.isin(d_only) == dev.damp_only.to_numpy()).all()
    rp = pd.read_parquet(root / "ryan_pairs.parquet", columns=["gene_a", "gene_b", "involves_merged_id"])
    merged = rp.set_index(_keyset(rp)).involves_merged_id
    assert (merged.reindex(kd).to_numpy(bool) == dev.involves_merged_id.to_numpy()).all()
    # (3) no excluded pair appears in any evaluation set (both split seeds)
    bad = dev[dev.damp_only | dev.involves_merged_id | ~dev.in_universe]
    bk = _keyset(bad)
    assert len(bk) > 0
    for seed in data.split_seeds():
        ek = _keyset(data.all_eval_pairs(seed))
        assert not ek.isin(bk).any(), seed


def test_synthetic_planted_damp_only_pairs_never_evaluated(synth_env):
    """Planted truth (not the stored flags): query gene SPTOY070 is viable and screened with a DAmP allele
    only, so every pair with an array-only partner is DAmP-only and must be absent from all eval sets."""
    s = synth_env
    dev = data.labels_dev()
    kd = _keyset(dev)
    g = s["genes"][70]
    assert g in s["damp_only_query"] and s["role"][g] == "query"
    partner = np.where(dev.gene_a == g, dev.gene_b, dev.gene_a)
    # array-only partners that are in the universe (the two planted inviable array genes are not)
    planted = ((dev.gene_a == g) | (dev.gene_b == g)).to_numpy() & np.array(
        [s["role"].get(x) == "array" and x not in s["drop"] for x in partner])
    assert planted.sum() > 5                                   # non-vacuous
    assert dev.in_universe[planted].all()
    assert dev.damp_only[planted].all() and not dev.eval_eligible[planted].any()
    # DAmP-only pairs among viable genes only come from the planted DAmP queries
    assert (dev.damp_only & dev.in_universe).sum() == planted.sum()
    for seed in data.split_seeds():
        ev = data.all_eval_pairs(seed)
        assert not _keyset(ev).isin(kd[planted]).any()
        assert len(ev) > 0
    # the mixed query (DAmP + deletion) stays eligible when otherwise in the universe
    mixed = s["damp_mixed_query"]
    sub = dev[((dev.gene_a == mixed) | (dev.gene_b == mixed)) & dev.in_universe & ~dev.involves_merged_id]
    assert len(sub) > 0 and sub.eval_eligible.all()
    # eligibility equals the protocol rule on the synthetic labels too
    assert (dev.eval_eligible == (dev.in_universe & ~dev.damp_only & ~dev.involves_merged_id)).all()
    d_only = _independent_damp_only_keys(protocol.data_root())
    assert (kd.isin(d_only) == dev.damp_only.to_numpy()).all()
