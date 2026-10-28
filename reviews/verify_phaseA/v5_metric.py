"""Independent primary metric + bootstrap check (DESIGN §3). My own code computes everything; src.evaluate
is called ONLY through its public evaluate() API (register=False) for comparison."""
import hashlib, json, sys, time, numpy as np, pandas as pd, yaml
from sklearn.metrics import average_precision_score
R = "/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/"
P = yaml.safe_load(open(R + "config/protocol.yaml"))
CW = {"primary": P["class_weights"]["unmeasured_universe"], "all_pairs": P["class_weights"]["all_pairs"],
      "equal": {c: 1 / 3 for c in ("C1", "C2", "C3")}}
SEED = P["dev_cv"]["split_seed"]; SP = R + "data/processed/splits/"
dev = pd.read_parquet(SP + "labels_dev.parquet")
F = pd.read_csv(SP + f"dev_folds_seed{SEED}.tsv", sep="\t").set_index("gene").fold
H = pd.read_parquet(SP + f"c1_holdout_seed{SEED}.parquet")
fa, fb = dev.gene_a.map(F).to_numpy(), dev.gene_b.map(F).to_numpy()
el, td = dev.eval_eligible.to_numpy(), dev.train_drop.to_numpy()
hk = H.gene_a + "|" + H.gene_b; dk = dev.gene_a + "|" + dev.gene_b
EV, TR = [], {}
for k in range(5):
    nin = (fa == k).astype(int) + (fb == k).astype(int)
    c1 = dk.isin(set(hk[H.fold == k])).to_numpy()
    part = np.full(len(dev), "", object)
    part[c1] = "C1"; part[(nin == 1) & el] = "C2"; part[(nin == 2) & el] = "C3"
    m = part != ""
    EV.append(dev.loc[m, ["gene_a", "gene_b", "y"]].assign(fold=k, part=part[m]))
    TR[k] = dev.loc[(nin == 0) & ~c1 & ~td, ["gene_a", "gene_b", "y"]]
EV = pd.concat(EV, ignore_index=True)

def primary(df, cw):
    out = []
    for k, d in df.groupby("fold"):
        n = d.part.map(d.part.value_counts())
        w = d.part.map(cw).to_numpy() / n.to_numpy()
        out.append(average_precision_score(d.y.to_numpy(bool), d.score.to_numpy(float), sample_weight=w))
    return float(np.mean(out)), out

# (a) registered B0 run
run = sys.argv[1] if len(sys.argv) > 1 else "B0_random__20261028T160633__66505c"
pr = pd.read_parquet(R + f"data/processed/preds/{run}.parquet")
J = json.load(open(R + f"results/runs/{run}.json"))
d = pr.merge(dev[["gene_a", "gene_b", "y"]], on=["gene_a", "gene_b"], how="left")
chk = pr.merge(EV[["gene_a", "gene_b", "fold", "part"]], on=["gene_a", "gene_b", "fold"], how="outer", indicator=True, suffixes=("", "_mine"))
print("B0 pred rows vs my eval set:", chk._merge.value_counts().to_dict(), "part mismatches", int((chk.part != chk.part_mine).sum()))
for nm in CW:
    mine, _ = primary(d, CW[nm])
    reg = J["primary"] if nm == "primary" else J["metrics_mean"][f"pooled_{nm}"]
    print(f"B0 {nm}: mine={mine:.12f} registered={reg:.12f} diff={mine - reg:.2e}")
wp = np.mean([sum(CW['primary'][c] * g[g.part == c].y.mean() for c in CW['primary']) for _, g in d.groupby('fold')])
print(f"B0 weighted prevalence (mine) {wp:.6f} vs registered {J['metrics_mean']['prevalence_primary']:.6f}")

# (b) non-random scores
sys.path.insert(0, R)
from src.evaluate import evaluate   # public API only
def degree_scores(ev):
    out = np.zeros(len(ev))
    for k in range(5):
        t = TR[k]
        long = pd.concat([t[["gene_a", "y"]].rename(columns={"gene_a": "g"}), t[["gene_b", "y"]].rename(columns={"gene_b": "g"})])
        prior = long.y.mean(); s = long.groupby("g").y.agg(["sum", "size"])
        rate = (s["sum"] + 20 * prior) / (s["size"] + 20)
        m = (ev.fold == k).to_numpy()
        out[m] = ev.gene_a[m].map(rate).fillna(prior).to_numpy() + ev.gene_b[m].map(rate).fillna(prior).to_numpy()
    return out
def hash_scores(ev):
    return np.array([int(hashlib.md5((a + "|" + b).encode()).hexdigest()[:8], 16) / 2**32 for a, b in zip(ev.gene_a, ev.gene_b)])
NB = 30
for nm, fn in (("degree_train_only", degree_scores), ("md5_hash", hash_scores)):
    ev = EV.copy(); ev["score"] = fn(ev)
    mine, perfold = primary(ev, CW["primary"])
    preds = ev[["gene_a", "gene_b", "fold", "part", "score"]].assign(seed=0)
    t0 = time.time()
    res = evaluate(preds, model=f"verify_{nm}", register=False, n_boot=NB)
    print(f"{nm}: mine={mine:.12f} evaluate={res['primary']:.12f} diff={mine - res['primary']:.2e} "
          f"perfold_max_diff={max(abs(a - res['metrics_per_fold'][k]['pooled']) for k, a in enumerate(perfold)):.2e} ({time.time() - t0:.0f}s)")
    for w in ("all_pairs", "equal"):
        mm, _ = primary(ev, CW[w]); print(f"   {w}: mine={mm:.10f} evaluate={res['metrics_mean']['pooled_' + w]:.10f}")
    for c in ("C1", "C2", "C3"):
        apc = np.mean([average_precision_score(g.y[g.part == c], g.score[g.part == c]) for _, g in ev.groupby("fold")])
        print(f"   ap_{c}: mine={apc:.10f} evaluate={res['metrics_mean']['ap_' + c]:.10f}")
    # (6) bootstrap: my own brute-force Poisson(1) shared gene bootstrap of the exact primary statistic
    genes = np.array(sorted(F.index))   # all dev genes, sorted (same rng stream as protocol: seed, poisson1)
    cnt = np.random.default_rng(P["bootstrap"]["seed"]).poisson(1.0, size=(NB, len(genes)))
    gi = pd.Series(np.arange(len(genes)), index=genes)
    ia, ib = gi[ev.gene_a].to_numpy(), gi[ev.gene_b].to_numpy()
    reps = []
    for r in range(NB):
        base = cnt[r, ia] * cnt[r, ib]
        vals = []
        for k in range(5):
            m = (ev.fold == k).to_numpy(); w = np.zeros(m.sum()); pt = ev.part.to_numpy()[m]; b = base[m]
            for c, cwc in CW["primary"].items():
                mc = pt == c
                if b[mc].sum() > 0: w[mc] = b[mc] * cwc / b[mc].sum()
            keep = w > 0
            vals.append(average_precision_score(ev.y.to_numpy(bool)[m][keep], ev.score.to_numpy()[m][keep], sample_weight=w[keep]))
        reps.append(np.mean(vals))
    reps = np.array(reps); B = res["bootstrap"]["primary"]
    print(f"   bootstrap ({NB} reps): mine mean={reps.mean():.10f} se={reps.std(ddof=1):.10f} | evaluate mean={B['mean']:.10f} se={B['se']:.10f}")
