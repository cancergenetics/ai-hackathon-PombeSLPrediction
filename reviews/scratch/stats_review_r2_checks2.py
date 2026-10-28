"""(part 2, generated from part 1) Round-2 statistics review: variance of the pooled universe-weighted AUPRC and of paired deltas.
Read-only. Variance demo with synthetic score models built on real Ryan S2 labels (gene effects are leaky;
used only to get realistic gene-level correlation structure, not as a performance estimate)."""
import numpy as np, pandas as pd
R = '/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/data/processed/'
d = pd.read_parquet(R + 'ryan_pairs.parquet')
U = {'viable', 'condition-dependent'}
d = d[d.in_s2 & ~d.involves_merged_id & d.viability_a.isin(U) & d.viability_b.isin(U)].reset_index(drop=True)
y = d['neg_s2_lt_-2.3'].values.astype(np.int8)
genes, inv = np.unique(np.r_[d.gene_a.values, d.gene_b.values], return_inverse=True)
ia, ib = inv[:len(d)], inv[len(d):]
G = len(genes); print('pairs', len(d), 'genes', G, 'prev %.4f' % y.mean())
CW = np.array([0.25, 0.57, 0.18])  # C1, C2, C3 weights (unmeasured universe)

def wap(order, yy, w):
    w = w[order]; yo = yy[order]
    cw = np.cumsum(w); tp = np.cumsum(w * yo)
    P = (w * yo).sum()
    return np.sum(w * yo * tp / np.maximum(cw, 1e-12)) / P

def logit_rate(pos, n, a=1, b=50):
    r = (pos + a) / (n + b); return np.log(r / (1 - r))

rng = np.random.default_rng(1)
pos_full = np.bincount(ia, y, G) + np.bincount(ib, y, G)
n_full = np.bincount(ia, None, G) + np.bincount(ib, None, G)
beta_full = logit_rate(pos_full, n_full)
gnoise = rng.standard_normal(G)          # fixed gene-level error of "feature-predicted" propensity
gnoise2 = rng.standard_normal(G)
fold = rng.permutation(np.arange(G) % 5)

folds = []
for k in range(5):
    ha, hb = fold[ia] == k, fold[ib] == k
    c3 = ha & hb; c2 = ha ^ hb
    other = ~ha & ~hb
    c1 = other & (rng.random(len(d)) < 0.10)
    train = other & ~c1
    pos = np.bincount(ia[train], y[train], G) + np.bincount(ib[train], y[train], G)
    n = np.bincount(ia[train], None, G) + np.bincount(ib[train], None, G)
    bt = logit_rate(pos, n)
    def pred_beta(rho, noise):
        b = bt.copy(); h = fold == k
        mu = beta_full[~h].mean(); sd = beta_full[~h].std()
        b[h] = mu + rho * (beta_full[h] - mu) + np.sqrt(1 - rho ** 2) * sd * noise[h]
        return b
    idx = np.where(c1 | c2 | c3)[0]
    cls = np.where(c1[idx], 0, np.where(c2[idx], 1, 2))
    bA = pred_beta(0.50, gnoise)
    nz = 0.8*gnoise + 0.6*gnoise2   # same skill, partly different gene-level error (null difference)
    bB = pred_beta(0.50, nz)
    sA = bA[ia[idx]] + bA[ib[idx]]
    sB = bB[ia[idx]] + bB[ib[idx]]
    e = rng.standard_normal(len(idx)) + 0.6 * y[idx]
    sC = sA + 0.25 * e                                           # pair-level improvement
    jit = 1e-9 * rng.standard_normal(len(idx))
    folds.append(dict(idx=idx, cls=cls, y=y[idx], ga=ia[idx], gb=ib[idx],
                      o={m: np.argsort(-(s + jit)) for m, s in [('A', sA), ('B', sB), ('C', sC)]}))
    print('fold', k, 'n C1/C2/C3', np.bincount(cls), 'pos', np.bincount(cls, y[idx]).astype(int))

def fold_stat(f, m, gm=None, mode='shared'):
    if gm is None: w = np.ones(len(f['idx']))
    elif mode == 'shared': w = gm[f['ga']] * gm[f['gb']]
    else: w = gm[0][f['ga']] * gm[1][f['gb']]  # independent draws per side (orientation arbitrary here)
    W = np.zeros(len(w))
    for c in range(3):
        s = f['cls'] == c; W[s] = CW[c] * w[s] / max(w[s].sum(), 1e-12)
    return wap(f['o'][m], f['y'], W)


pt = {m: np.array([fold_stat(f, m) for f in folds]) for m in 'AB'}
print('null Delta B-A per fold', np.round(pt['B']-pt['A'],4), 'mean %.4f'%(pt['B']-pt['A']).mean())
Bn=200; sh=[]; ind=[]; dl=[]
for b in range(Bn):
    gm = rng.poisson(1.0, G).astype(float)
    a=np.mean([fold_stat(f,'A',gm) for f in folds]); bb=np.mean([fold_stat(f,'B',gm) for f in folds])
    sh.append(a); dl.append(bb-a)
    # per-class independent draws (design text: separate schemes per class)
    vals=[]
    for f in folds:
        W=np.zeros(len(f['idx']))
        for c in range(3):
            g2=rng.poisson(1.0,G).astype(float); s=f['cls']==c
            w=g2[f['ga'][s]]*g2[f['gb'][s]]; W[s]=CW[c]*w/w.sum()
        vals.append(wap(f['o']['A'],f['y'],W))
    ind.append(np.mean(vals))
dl=np.array(dl)
print('SE pooled A: shared-draw %.4f vs per-class-independent draws %.4f'%(np.std(sh),np.std(ind)))
print('null Delta: SE %.4f CI [%.4f, %.4f]'%(dl.std(),*np.percentile(dl,[2.5,97.5])))
