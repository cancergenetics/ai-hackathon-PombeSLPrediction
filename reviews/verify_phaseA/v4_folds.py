"""Independent fold-split verification for one split seed (DESIGN §2). No src/ imports.
Lockbox file: only row/positive counts by part are read (columns part, y, eval_eligible)."""
import json, sys, numpy as np, pandas as pd, yaml
R = "/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261028
P = yaml.safe_load(open(R + "config/protocol.yaml"))
SP = R + "data/processed/splits/"
dev = pd.read_parquet(SP + "labels_dev.parquet")
key = lambda a, b: pd.Index(np.where(a < b, a, b).astype(object)) + "|" + pd.Index(np.where(a < b, b, a).astype(object))
dev["k"] = key(dev.gene_a.to_numpy(object), dev.gene_b.to_numpy(object))
assert dev.k.is_unique and (dev.gene_a < dev.gene_b).all()
# my label from s_score
assert ((dev.s_score < P["label"]["threshold"]) == dev.y).all(), "y != 1[s_score < thr]"
lb = set(open(R + P["lockbox"]["genes_file"]).read().split())
print("lockbox genes", len(lb), "dev rows with a lockbox gene:", int((dev.gene_a.isin(lb) | dev.gene_b.isin(lb)).sum()))
# ---- reproduce lockbox-C1 from the full S2 table with my own sampling code
p = pd.read_parquet(R + "data/processed/ryan_pairs.parquet", columns=["gene_a", "gene_b", "s_score", "involves_merged_id"])
p = p[p.s_score.notna()].reset_index(drop=True)
u = pd.read_csv(R + "data/processed/universe_genes.tsv", sep="\t"); U = set(u.gene[u.in_universe])
p["k"] = key(p.gene_a.to_numpy(object), p.gene_b.to_numpy(object))
devdev = ~(p.gene_a.isin(lb) | p.gene_b.isin(lb))
elig_map = dev.set_index("k").eval_eligible        # verified independently in v3 for dev pairs
# eligibility for lockbox-C1 candidates (devdev) must be computed for ALL devdev pairs incl. the drawn ones:
# recompute from v3-verified ingredients (universe, merged, damp_only via dev file is not available for drawn pairs)
print("devdev S2 pairs:", int(devdev.sum()), " dev rows:", len(dev), " difference:", int(devdev.sum()) - len(dev))
assert set(dev.k) <= set(p.k[devdev]), "dev contains non-devdev pairs"
missing = set(p.k[devdev]) - set(dev.k)
lock = pd.read_parquet(SP + "labels_lockbox.parquet", columns=["part", "y", "eval_eligible"])
cnt = lock.groupby("part").agg(n=("y", "size"), pos=("y", "sum"), elig=("eval_eligible", "sum"))
print("lockbox counts by part:\n", cnt)
print("devdev pairs missing from dev == lockbox C1 rows:", len(missing) == int(cnt.loc["C1", "n"]))
pm = p.set_index("k").loc[sorted(missing)]
print("missing (=lockbox C1) pairs: positives", int((pm.s_score < P["label"]["threshold"]).sum()),
      "| all in universe:", bool((pm.index.str.split("|").map(lambda x: x[0] in U and x[1] in U)).all()))
# reproduce the label-stratified 10% draw with my own code (needs the DAmP flags: from v3 logic, recomputed quickly)
sys.path.insert(0, R + "reviews/verify_phaseA")
# ---- folds
F = pd.read_csv(SP + f"dev_folds_seed{SEED}.tsv", sep="\t").set_index("gene").fold
devg = set(dev.gene_a) | set(dev.gene_b)
print("fold file genes == dev genes:", set(F.index) == devg, "unique:", F.index.is_unique, "sizes", F.value_counts().sort_index().to_dict())
H = pd.read_parquet(SP + f"c1_holdout_seed{SEED}.parquet")
H["k"] = key(H.gene_a.to_numpy(object), H.gene_b.to_numpy(object))
fa, fb = dev.gene_a.map(F).to_numpy(), dev.gene_b.map(F).to_numpy()
drop = set(u.gene[u.drop_from_training])
mytd = (dev.gene_a.isin(drop) | dev.gene_b.isin(drop)).to_numpy()
assert (mytd == dev.train_drop.to_numpy()).all()
el = dev.eval_eligible.to_numpy(); y = dev.y.to_numpy()
summ = json.load(open(R + "results/splits_summary.json"))["folds"][str(SEED)]
ok = True
for k in range(P["dev_cv"]["n_folds"]):
    G = set(F.index[F == k])
    nin = (fa == k).astype(int) + (fb == k).astype(int)
    hk = set(H.k[H.fold == k])
    c1 = dev.k.isin(hk).to_numpy()
    assert (nin[c1] == 0).all(), "C1 hold-out pair touches fold genes"
    assert el[c1].all(), "C1 hold-out pair not eligible"
    assert len(hk) == c1.sum()
    train = (nin == 0) & ~c1 & ~mytd
    c2 = (nin == 1) & el; c3 = (nin == 2) & el
    trg = set(dev.gene_a[train]) | set(dev.gene_b[train])
    leak = trg & G
    # eval genes of C2/C3 rows that are in fold k must be absent from training
    evg = (set(dev.gene_a[c2 | c3]) | set(dev.gene_b[c2 | c3])) & G
    c1unseen = int((~(dev.gene_a[c1].isin(trg) & dev.gene_b[c1].isin(trg))).sum())
    # C2 partner (non-fold gene) seen in training?
    part_c2 = np.where(fa[c2] == k, dev.gene_b.to_numpy()[c2], dev.gene_a.to_numpy()[c2])
    c2_unseen_partner = int((~pd.Series(part_c2).isin(trg)).sum())
    elig_neither = (nin == 0) & el
    frac = c1.sum() / elig_neither.sum(); fracp = (c1 & y).sum() / (elig_neither & y).sum()
    mine = {"train_pairs": int(train.sum()), "train_positives": int(y[train].sum()),
            "C1_pairs": int(c1.sum()), "C2_pairs": int(c2.sum()), "C3_pairs": int(c3.sum()),
            "C1_positives": int(y[c1].sum()), "C2_positives": int(y[c2].sum()), "C3_positives": int(y[c3].sum()), "genes": len(G)}
    diff = {a: (mine[a], summ[str(k)][a]) for a in mine if mine[a] != summ[str(k)][a]}
    ok &= not diff and not leak and not c1unseen
    print(f"fold {k}: {mine} | leak={len(leak)} evalgenes_in_train={len(evg & trg)} C1_unseen={c1unseen} "
          f"C2_unseen_partner={c2_unseen_partner} C1frac={frac:.4f}/{fracp:.4f} summary_diffs={diff}")
print("ALL FOLD CHECKS OK" if ok else "FOLD CHECK FAILURES")
