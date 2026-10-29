"""Independent re-derivation of the lockbox metrics from saved predictions (verifier code; does not use
src/evaluate.py). Labels recomputed as s_score < threshold from labels_lockbox.parquet. Own Poisson(1)
gene bootstrap (own seed), own tie-aware weighted AP, own Holm. Prints aggregates only."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P = yaml.safe_load(open(ROOT / "config/protocol.yaml"))
THR = P["label"]["threshold"]; assert P["label"]["direction"] == "less_than"
CW = P["class_weights"][P["class_weights"]["primary"]]
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 12345  # deliberately NOT the protocol seed
MODELS = os.environ.get("VMODELS", "M2r_huber,M2,M1,B1_additive_logit,B4_logistic_MF").split(",")
CONTRASTS = [("M2r_huber", "B1_additive_logit"), ("M2r_huber", "B4_logistic_MF"), ("M2r_huber", "M1"), ("M2r_huber", "M2")]

L = pd.read_parquet(ROOT / "data/processed/splits/labels_lockbox.parquet",
                    columns=["gene_a", "gene_b", "s_score", "eval_eligible", "part"])
key = lambda a, b: pd.Series(np.where(a < b, a + "|" + b, b + "|" + a))
L["k"] = key(L.gene_a.values, L.gene_b.values).values
L["y"] = L.s_score.values < THR
lab = L.set_index("k")[["y"]]

def weighted_ap(score, y, w):
    """Tie-aware weighted AP (sklearn step convention): sum over distinct thresholds of dRecall * precision."""
    o = np.argsort(-score, kind="mergesort")
    s, y, w = score[o], y[o], w[o]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    ctp = np.cumsum(w * y)[ends]; call = np.cumsum(w)[ends]
    dtp = np.diff(np.r_[0.0, ctp])
    return float(np.sum(dtp / ctp[-1] * np.divide(ctp, call, out=np.zeros_like(ctp), where=call > 0)))

class Plan:
    """Pre-sorted structure for fast bootstrap APs of one model."""
    def __init__(self, score, y, part, ia, ib):
        o = np.argsort(-score, kind="mergesort")
        s = score[o]
        self.y, self.part, self.ia, self.ib = y[o], part[o], ia[o], ib[o]
        self.ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    def ap(self, cnt, cls_w):
        base = (cnt[self.ia] * cnt[self.ib]).astype(float)
        w = np.zeros_like(base)
        for c, v in cls_w.items():
            m = self.part == c
            tot = base[m].sum()
            if tot > 0:
                w[m] = base[m] * v / tot
        ctp = np.cumsum(w * self.y)[self.ends]; call = np.cumsum(w)[self.ends]
        if ctp[-1] <= 0:
            return np.nan
        dtp = np.diff(np.r_[0.0, ctp])
        return float(np.sum(dtp / ctp[-1] * np.divide(ctp, call, out=np.zeros_like(ctp), where=call > 0)))

preds, ref = {}, None
for m in MODELS:
    d = pd.read_parquet(ROOT / f"data/processed/lockbox/preds_{m}.parquet")
    d["k"] = key(d.gene_a.values, d.gene_b.values).values
    if ref is None:
        ref = d[["gene_a", "gene_b", "k", "part"]].copy()
    else:
        assert (d.k.values == ref.k.values).all() and (d.part.values == ref.part.values).all(), m
    preds[m] = d.score.to_numpy(float)
assert ref.k.is_unique
y = lab.loc[ref.k.values, "y"].to_numpy(bool)
part = ref.part.to_numpy()
genes = pd.Index(sorted(set(ref.gene_a) | set(ref.gene_b)))
ia, ib = genes.get_indexer(ref.gene_a), genes.get_indexer(ref.gene_b)
n = {c: int((part == c).sum()) for c in CW}; pos = {c: int(y[part == c].sum()) for c in CW}
w_point = np.array([CW[c] / n[c] for c in part])

res = {"n_pairs": len(ref), "n_genes": len(genes), "n": n, "pos": pos,
       "prevalence": {c: pos[c] / n[c] for c in CW}, "n_boot": NB, "boot_seed": SEED, "models": {}}
rng = np.random.default_rng(SEED)
counts = rng.poisson(1.0, size=(NB, len(genes)))
reps = {}
onehot = {c: {x: (1.0 if x == c else 0.0) for x in CW} for c in CW}
for m in MODELS:
    s = preds[m]
    prim = weighted_ap(s, y, w_point)
    prim_sk = average_precision_score(y, s, sample_weight=w_point)
    cls = {c: weighted_ap(s[part == c], y[part == c], np.ones(n[c])) for c in CW}
    cls_sk = {c: average_precision_score(y[part == c], s[part == c]) for c in CW}
    pl = Plan(s, y, part, ia, ib)
    assert abs(pl.ap(np.ones(len(genes), int), CW) - prim) < 1e-12
    R = {"primary": np.array([pl.ap(counts[r], CW) for r in range(NB)])}
    for c in CW:
        R[c] = np.array([pl.ap(counts[r], onehot[c]) for r in range(NB)])
    reps[m] = R
    res["models"][m] = {"primary": prim, "primary_sklearn": prim_sk,
                        "primary_ci": list(np.quantile(R["primary"], [0.025, 0.975])),
                        **{f"ap_{c}": cls[c] for c in CW}, **{f"ap_{c}_sklearn": cls_sk[c] for c in CW},
                        **{f"ap_{c}_ci": list(np.quantile(R[c], [0.025, 0.975])) for c in CW}}
    print(m, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                         for k, v in res["models"][m].items()}), flush=True)
praw = {}
res["contrasts"] = {}
for a, b in CONTRASTS:
    if a not in reps or b not in reps:
        continue
    d = reps[a]["primary"] - reps[b]["primary"]
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    k = f"{a} - {b}"; praw[k] = p
    res["contrasts"][k] = {"delta": res["models"][a]["primary"] - res["models"][b]["primary"],
                           "ci": list(np.quantile(d, [0.025, 0.975])), "p": p, "p_floor": 1.0 / NB}
# Holm (own implementation)
order = sorted(praw, key=praw.get); m_ = len(order); run = 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (m_ - i) * praw[k]))
    res["contrasts"][k]["p_holm"] = run
    res["contrasts"][k]["reject"] = bool(run < 0.05)
for k, v in res["contrasts"].items():
    print(k, json.dumps({kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items()}))
# compare with claimed
C = json.load(open(ROOT / "results/lockbox/lockbox_result_deec74e31a7c.json"))
diff = {m: {k: abs(res["models"][m][k] - C["models"][m][k]) for k in ("primary", "ap_C1", "ap_C2", "ap_C3")} for m in MODELS}
diff.update({k: abs(res["contrasts"][k]["delta"] - C["contrasts"][k]["delta"]) for k in res["contrasts"]})
res["abs_diff_vs_claimed"] = diff
print("max abs diff point estimates vs claimed:", max(max(v.values()) if isinstance(v, dict) else v for v in diff.values()))
json.dump(res, open(OUT / "rederive_result.json", "w"), indent=1, default=float)
