"""Check nested inner splits with my own checks (outputs obtained via the public src.data API)."""
import sys, numpy as np, pandas as pd
R = "/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/"
sys.path.insert(0, R)
from src import data
fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tr, ev = data.fold_split(None, fold)
inner = data.inner_splits(tr)
K = lambda d: (d.gene_a + "|" + d.gene_b)
trK = K(tr); allg = set(tr.gene_a) | set(tr.gene_b)
G = {f.j: set(f.genes) for f in inner}
print("outer fold", fold, "train pairs", len(tr), "genes", len(allg), "inner gene folds partition:",
      sum(len(g) for g in G.values()) == len(allg) and set().union(*G.values()) == allg)
gf = pd.Series({g: j for j, s in G.items() for g in s})
na = tr.gene_a.map(gf).to_numpy(); nb = tr.gene_b.map(gf).to_numpy()
el = tr.eval_eligible.to_numpy(bool); yy = tr.y.to_numpy(bool)
for f in inner:
    T = G[f.j]; t = f.train; H = set(K(f.holdout)); E = set(K(f.eval)); tk = K(t)
    neither = (na != f.j) & (nb != f.j)
    exp_train = set(trK[neither]) - H
    inH = trK.isin(H).to_numpy()
    print(f" inner {f.j}: |T|={len(T)} train={len(t)} T_j-touching={int((t.gene_a.isin(T) | t.gene_b.isin(T)).sum())} "
          f"H_j-in-train={int(tk.isin(H).sum())} eval-in-train={int(tk.isin(E).sum())} train==neither-minus-H: {set(tk) == exp_train} "
          f"H frac={inH.sum() / (neither & el).sum():.4f} Hpos frac={(inH & yy).sum() / (neither & el & yy).sum():.4f} "
          f"H all eligible={bool(el[inH].all())} H touches T_j={int((f.holdout.gene_a.isin(T) | f.holdout.gene_b.isin(T)).sum())}")
c2 = pd.concat([K(f.eval[f.eval.part == "C2"]) for f in inner])
cross = set(trK[na != nb])
print("C2-like pairs kept exactly once:", c2.is_unique and set(c2) == cross, len(c2), len(cross))
rows = pd.concat([f.eval for f in inner], ignore_index=True); rk = K(rows)
for j, t, v in data.gene_grouped_validation(inner):
    T = G[j]
    exp = (rows.inner_fold != j) & ~rows.gene_a.isin(T) & ~rows.gene_b.isin(T) & ~rk.isin(set(K(v)))
    print(f" stacker fold {j}: train rows={len(t)} T_j-touching={int((t.gene_a.isin(T) | t.gene_b.isin(T)).sum())} "
          f"valid-overlap={int(K(t).isin(set(K(v))).sum())} valid==inner_fold j rows: {len(v) == int((rows.inner_fold == j).sum())} "
          f"train==expected: {set(K(t)) == set(rk[exp]) and len(t) == int(exp.sum())}")
