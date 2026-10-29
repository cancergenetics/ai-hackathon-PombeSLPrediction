"""Cheap learning-curve probe (verifier): M1 (label-free LightGBM, seed 0 only) in dev CV, trained on the
fold's full training set vs. with a random 20% of its training genes removed. Eval restricted to pairs whose
seen genes survive the subsample (C1: both; C2: the non-fold gene; C3: all), so both arms score the same pairs
and the class definitions are unchanged. Aggregates only. Usage: m1_gene_curve.py [folds...]"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import data, protocol
from src import deliverable as DL
from src.timing import set_threads
from sklearn.metrics import average_precision_score as aps
OUT = Path(__file__).resolve().parent
TH = int(__import__("os").environ.get("VTH", "2")); set_threads(TH)
THR = protocol.load()["label"]["threshold"]
folds = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4]
spec = DL.model_spec("M1"); spec["seeds"] = [0]
D = data.labels_dev()
key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a)
ylab = pd.Series(D.s_score.values < THR, index=key(D.gene_a.values, D.gene_b.values))
res = []
for k in folds:
    tr, ev = data.fold_split(None, k)
    fg = set(data.fold_genes(None, k))
    tg = np.array(sorted(set(tr.gene_a) | set(tr.gene_b)))
    rng = np.random.default_rng(1000 + k)
    drop = set(rng.choice(tg, int(round(0.2 * len(tg))), replace=False))
    sub = tr[~tr.gene_a.isin(drop) & ~tr.gene_b.isin(drop)]
    keep = ~ev.gene_a.isin(drop) & ~ev.gene_b.isin(drop)
    E = ev[keep].reset_index(drop=True)
    y = ylab.loc[key(E.gene_a.values, E.gene_b.values)].to_numpy(bool)
    row = {"fold": k, "n_train_full": len(tr), "n_train_sub": len(sub), "genes_full": len(tg), "genes_sub": len(tg) - len(drop)}
    for arm, T in (("full", tr), ("sub", sub)):
        t0 = time.time()
        m = DL.TrainedModel(spec, T, threads=TH, log=lambda *a: None).fit()
        s = m.predict(E[["gene_a", "gene_b"]])
        for c in ("C1", "C2", "C3"):
            mk = E.part.values == c
            row[f"{arm}_{c}"] = aps(y[mk], s[mk])
        row[f"{arm}_s"] = round(time.time() - t0, 1)
    print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
    res.append(row)
    pd.DataFrame(res).to_csv(OUT / "m1_gene_curve.tsv", sep="\t", index=False)
R = pd.DataFrame(res)
for c in ("C1", "C2", "C3"):
    print(c, "mean full", round(R[f"full_{c}"].mean(), 4), "sub", round(R[f"sub_{c}"].mean(), 4),
          "rel drop", round(1 - R[f"sub_{c}"].mean() / R[f"full_{c}"].mean(), 3))
