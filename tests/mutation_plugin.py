"""pytest plugin (loaded with `-p mutation_plugin` by tests/test_mutants.py): applies ONE planted mutant
(env PSL_MUTANT) to src/ in-process, to check that the rest of the suite notices. Adapted from
reviews/verify_phaseA/mutplug.py. Never modifies files under src/, scripts/ or tests/ on disk.
For the make_splits mutants, PSL_MUTANT_DIR holds generated mutant copies of scripts/make_splits.py."""
import os
import runpy

import numpy as np
import pandas as pd

MUT = os.environ.get("PSL_MUTANT", "")
MUT_DIR = os.environ.get("PSL_MUTANT_DIR", "")


def pytest_configure(config):
    from src import splits as S, data, evaluate as E
    if MUT == "evict_c3":                      # eval set silently incomplete: every 2nd C3 row dropped
        orig = S.fold_parts

        def fp(*a, **k):
            part = orig(*a, **k)
            idx = np.flatnonzero(part == "C3")
            part[idx[::2]] = ""
            return part
        S.fold_parts = fp
    elif MUT == "c1_in_train":                 # C1 hold-out pairs also used for training
        orig = S.fold_parts

        def fp(fa, fb, k, c1, el, tok):
            part = orig(fa, fb, k, c1, el, tok)
            n_in = (np.asarray(fa) == k).astype(int) + (np.asarray(fb) == k).astype(int)
            out = part.copy()
            out[(n_in == 0) & c1 & tok] = "train"
            return out
        S.fold_parts = fp
    elif MUT == "uniform_weights":
        E.class_row_weights = lambda part, cw: np.full(len(part), 1.0 / max(len(part), 1))
    elif MUT == "boot_unshared":               # different gene counts per fold
        orig = E.bootstrap_replicates

        def br(df, counts, gidx, weightings=None, chunk=50):
            res = None
            for i, k in enumerate(sorted(df.fold.unique())):
                r = orig(df[df.fold == k], np.roll(counts, i, axis=1), gidx, weightings, chunk)
                res = r if res is None else {n: res[n] + r[n] for n in r}
            n = len(df.fold.unique())
            return {a: v / n for a, v in res.items()}
        E.bootstrap_replicates = br
    elif MUT == "inner_keep_H":
        orig = S.inner_split_generator

        def g(*a, **k):
            out = orig(*a, **k)
            return [S.InnerFold(f.j, f.genes,
                                pd.concat([f.train, f.eval[f.eval.part == "C1"].drop(columns=["part", "inner_fold"])]),
                                f.eval, f.holdout) for f in out]
        S.inner_split_generator = g
    elif MUT == "stacker_no_T":
        def ggv(inner):
            rows = pd.concat([f.eval for f in inner], ignore_index=True)
            return [(f.j, rows[rows.inner_fold != f.j], rows[rows.inner_fold == f.j]) for f in inner]
        S.gene_grouped_validation = ggv
        data.gene_grouped_validation = ggv
    elif MUT == "eval_hypomorph_true":
        orig = data.fold_split

        def fs(split_seed=None, fold=0):
            tr, ev = orig(split_seed, fold)
            dev = data.labels_dev().set_index(["gene_a", "gene_b"])
            h = dev.loc[pd.MultiIndex.from_frame(ev[["gene_a", "gene_b"]]), ["hypomorph_a", "hypomorph_b"]].to_numpy()
            ev["hypomorph_a"], ev["hypomorph_b"] = h[:, 0], h[:, 1]
            return tr, ev
        data.fold_split = fs
    elif MUT == "primary_is_allpairs":         # wrong class-weight vector used as primary
        from src import protocol
        orig = protocol.class_weights
        protocol.class_weights = lambda kind=None: orig("all_pairs" if kind in (None, "unmeasured_universe") else kind)
    elif MUT == "c2_partner_unseen":           # C1 pairs relabelled C2 (partner gene never in training)
        orig = S.fold_parts

        def fp(fa, fb, k, c1, el, tok):
            part = orig(fa, fb, k, c1, el, tok)
            part[part == "C1"] = "C2"
            return part
        S.fold_parts = fp
    elif MUT.startswith("ms_"):                # make_splits.py mutant copy
        real = runpy.run_path

        def rp(path, *a, **k):
            if str(path).endswith("scripts/make_splits.py"):
                path = os.path.join(MUT_DIR, f"make_splits_{MUT[3:]}.py")
            return real(path, *a, **k)
        runpy.run_path = rp
    elif MUT:
        raise RuntimeError(f"unknown mutant {MUT!r}")
