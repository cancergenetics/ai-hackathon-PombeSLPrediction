"""Check 4: split hygiene for M2 (lockbox genes, C1/C2/C3 assignment, DAmP-only exclusion, M2 preds
coverage). Independent re-derivation from the raw split files; prints counts only (no labels/scores)."""
import common  # noqa: F401
import json

import numpy as np
import pandas as pd

from src import data
from src.models import m2

F = common.FROZEN
out = {}
lock = set(x.strip() for x in open(common.REPO / "config" / "lockbox_genes.txt") if x.strip())
dev = pd.read_parquet(F / "splits" / "labels_dev.parquet",
                      columns=["gene_a", "gene_b", "in_universe", "damp_only", "involves_merged_id", "eval_eligible",
                               "train_drop"])
folds = pd.read_csv(F / "splits" / "dev_folds_seed20261028.tsv", sep="\t").set_index("gene").fold
hold = pd.read_parquet(F / "splits" / "c1_holdout_seed20261028.parquet")
uni = pd.read_csv(F / "universe_genes.tsv", sep="\t")
U = set(uni.gene[uni.in_universe])
out["n_lockbox"] = len(lock)
out["lockbox_in_dev_folds"] = len(lock & set(folds.index))
out["dev_pairs_touching_lockbox"] = int((dev.gene_a.isin(lock) | dev.gene_b.isin(lock)).sum())

# ---- independent DAmP-only flag from the measurement file (allele columns only, dev pairs only)
ms = pd.read_parquet(F / "ryan_measurements.parquet",
                     columns=["query_gene", "array_gene", "allele_class", "suffix", "main_allele"])
ms = ms[ms.main_allele.astype(bool)]
out["main_allele_classes"] = ms.allele_class.value_counts().to_dict()
qa, ar = ms.query_gene.astype(str).to_numpy(), ms.array_gene.astype(str).to_numpy()
ms = pd.DataFrame({"gene_a": np.where(qa < ar, qa, ar), "gene_b": np.where(qa < ar, ar, qa),
                   "damp": ((ms.allele_class == "DAmP") | ms.suffix.astype(str).str.upper().str.contains("DAMP")).to_numpy()})
agg = ms.groupby(["gene_a", "gene_b"]).damp.agg(["size", "sum"]).reset_index()
agg["my_damp_only"] = agg["sum"] == agg["size"]
d2 = dev.merge(agg[["gene_a", "gene_b", "my_damp_only"]], on=["gene_a", "gene_b"], how="left")
out["dev_pairs_without_measurements"] = int(d2.my_damp_only.isna().sum())
d2["my_damp_only"] = d2.my_damp_only.fillna(False).astype(bool)
out["damp_only_flag_disagreements"] = int((d2.my_damp_only != d2.damp_only).sum())
out["my_damp_only_universe_pairs_dev"] = int((d2.my_damp_only & d2.in_universe).sum())

# ---- per fold: M2's actual training pairs (fold_pairs) and eval rows
evkeys_all = []
for k in range(5):
    tr, ev, drop = m2.fold_pairs(None, k)
    inner = data.inner_splits(tr)
    fk = {}
    trg = set(tr.gene_a) | set(tr.gene_b)
    evg = set(ev.gene_a) | set(ev.gene_b)
    inner_g = set().union(*[set(f.eval.gene_a) | set(f.eval.gene_b) | set(f.train.gene_a) | set(f.train.gene_b)
                            for f in inner])
    fk["lockbox_genes_in_train"] = len(trg & lock)
    fk["lockbox_genes_in_inner_rows"] = len(inner_g & lock)
    fk["lockbox_genes_in_eval"] = len(evg & lock)
    nin = ev.gene_a.map(folds).eq(k).astype(int) + ev.gene_b.map(folds).eq(k).astype(int)
    hk = set(zip(hold[hold.fold == k].gene_a, hold[hold.fold == k].gene_b))
    inh = np.array([(a, b) in hk for a, b in zip(ev.gene_a, ev.gene_b)])
    exp = np.where(nin == 2, "C3", np.where(nin == 1, "C2", np.where(inh, "C1", "??")))
    fk["part_mismatch"] = int((exp != ev.part.to_numpy()).sum())
    fk["n_parts"] = ev.part.value_counts().to_dict()
    c1 = ev[ev.part == "C1"]
    fk["C1_gene_unseen_in_train"] = int((~(c1.gene_a.isin(trg) & c1.gene_b.isin(trg))).sum())
    fold_genes = set(folds.index[folds == k])
    fk["fold_genes_in_train"] = len(trg & fold_genes)
    trk = set(zip(tr.gene_a, tr.gene_b))
    fk["eval_pairs_in_train"] = int(sum((a, b) in trk for a, b in zip(ev.gene_a, ev.gene_b)))
    # eval rows must be eval_eligible: both in universe, not DAmP-only (my flag), not merged id
    e2 = ev.merge(d2[["gene_a", "gene_b", "my_damp_only", "involves_merged_id", "eval_eligible"]],
                  on=["gene_a", "gene_b"], how="left")
    fk["eval_damp_only_mine"] = int(e2.my_damp_only.sum())
    fk["eval_merged_id"] = int(e2.involves_merged_id.sum())
    fk["eval_not_both_universe"] = int((~(ev.gene_a.isin(U) & ev.gene_b.isin(U))).sum())
    fk["eval_not_flagged_eligible"] = int((~e2.eval_eligible.astype(bool)).sum())
    # completeness: every eligible pair touching fold k (C2/C3) is evaluated
    m = dev.eval_eligible & ((dev.gene_a.map(folds) == k) | (dev.gene_b.map(folds) == k))
    fk["eligible_fold_pairs_not_evaluated"] = int(m.sum() - (nin > 0).sum())
    fk["train_drop_genes_in_train"] = int(tr.merge(dev[["gene_a", "gene_b", "train_drop"]], on=["gene_a", "gene_b"]).train_drop.sum())
    out[f"fold{k}"] = fk
    evkeys_all.append(ev[["gene_a", "gene_b", "fold", "part"]])
# saved M2 preds: same rows, same parts, no lockbox gene
p = pd.read_parquet(F / "preds" / "M2__20261028T183713__024416.parquet", columns=["gene_a", "gene_b", "fold", "part"])
E = pd.concat(evkeys_all)
mm = E.merge(p, on=["gene_a", "gene_b", "fold"], how="outer", indicator=True, suffixes=("", "_p"))
out["preds_vs_eval_rows"] = mm._merge.value_counts().to_dict()
out["preds_part_mismatch"] = int((mm.part != mm.part_p).sum())
out["preds_lockbox_genes"] = int((p.gene_a.isin(lock) | p.gene_b.isin(lock)).sum())
print(json.dumps(out, indent=1, default=int))
(common.OUT / "check4_hygiene.json").write_text(json.dumps(out, indent=1, default=int))
