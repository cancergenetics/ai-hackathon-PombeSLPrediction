"""Read-only checks for the round-1 statistics review of DESIGN.md."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from math import comb
R = '/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/data/processed/'
d = pd.read_parquet(R+'ryan_pairs.parquet')
g = pd.read_csv(R+'ryan_genes.tsv', sep='\t').dropna(subset=['pombase_id'])
g = g.groupby('pombase_id').agg(is_query=('in_s2_rows','max'), is_array=('is_array','max'), viability=('viability','first')).reset_index()
d = d[d.in_s2 & (d.viability_a=='viable') & (d.viability_b=='viable')].copy()
d['y'] = d['neg_s2_lt_-2.3'].astype(int)
print('vv S2 pairs', len(d), 'prev', d.y.mean())
genes = pd.unique(pd.concat([d.gene_a, d.gene_b]))
gq = set(g.loc[g.is_query.astype(bool),'pombase_id'])
print('viable genes in vv S2', len(genes), 'of which query', len(set(genes)&gq))
# universe weights
U=3580; n_seen=len(genes); n_un=U-n_seen; tot=comb(U,2)
c1=comb(n_seen,2); c2=n_seen*n_un; c3=comb(n_un,2); meas=len(d)
print('weights all: C1 %.3f C2 %.3f C3 %.3f'%(c1/tot,c2/tot,c3/tot))
un=tot-meas
print('weights unmeasured only: C1 %.3f C2 %.3f C3 %.3f; C1 unmeasured=%d'%((c1-meas)/un,c2/un,c3/un,c1-meas))
nq=len(set(genes)&gq); na=n_seen-nq
print('C1 unmeasured composition: array-only x array-only pairs %d (%.2f of unmeasured C1)'%(comb(na,2), comb(na,2)/(c1-meas)))
print('C2 universe: seen partner is a query gene in %.2f of pairs'%(nq/n_seen))
# gene stats
deg = pd.concat([d[['gene_a','y']].rename(columns={'gene_a':'g'}), d[['gene_b','y']].rename(columns={'gene_b':'g'})])
gs = deg.groupby('g').y.agg(['sum','count']); gs['rate']=gs['sum']/gs['count']
gs['q']=gs.index.isin(gq)
print('hit rate by role:\n', gs.groupby('q').rate.describe())
print('prev of pairs: query x query', d[d.gene_a.isin(gq)&d.gene_b.isin(gq)].y.mean(), 'query x nonquery', d[d.gene_a.isin(gq)^d.gene_b.isin(gq)].y.mean())
# lockbox simulation
rng0 = np.random.default_rng(0)
role = gs.q.map({True:'q',False:'a'})
strat = role + '_' + pd.qcut(gs.rate.rank(method='first'),5,labels=False).astype(str)
rate = (gs['sum'] + 1)/(gs['count']+50)  # leaky propensity, variance demo only
d['score'] = np.log(rate.reindex(d.gene_a).values) + np.log(rate.reindex(d.gene_b).values)
rows=[]
for s in range(20):
    rng=np.random.default_rng(s)
    test=set()
    for k,idx in strat.groupby(strat).groups.items():
        idx=list(idx); m=int(round(0.2*len(idx))); test|=set(rng.choice(idx,m,replace=False))
    ta=d.gene_a.isin(test); tb=d.gene_b.isin(test)
    c3m=ta&tb; c2m=ta^tb
    c3=d[c3m]; c2=d[c2m]
    pos_by_gene = pd.concat([c3.gene_a[c3.y==1],c3.gene_b[c3.y==1]]).value_counts()
    rows.append(dict(seed=s,n_test=len(test),c2=len(c2),c2pos=c2.y.sum(),c3=len(c3),c3pos=c3.y.sum(),
        c3_queries=len(set(pd.concat([c3.gene_a,c3.gene_b]))&gq),
        c3_top10gene_share=pos_by_gene.head(10).sum()/max(1,2*c3.y.sum()),
        ap_c2=ap(c2.y,c2.score), ap_c3=ap(c3.y,c3.score)))
r=pd.DataFrame(rows); print(r.describe().T[['mean','std','min','max']])
# bootstrap SE for one lockbox draw
s=0; rng=np.random.default_rng(s); test=set()
for k,idx in strat.groupby(strat).groups.items():
    idx=list(idx); m=int(round(0.2*len(idx))); test|=set(rng.choice(idx,m,replace=False))
ta=d.gene_a.isin(test); tb=d.gene_b.isin(test)
for name,sub in [('C2',d[ta^tb]),('C3',d[ta&tb])]:
    sub=sub.reset_index(drop=True); y=sub.y.values; sc=sub.score.values
    B=200; pb=[]; gb=[]
    # pair bootstrap
    for b in range(B):
        i=rng.integers(0,len(sub),len(sub)); pb.append(ap(y[i],sc[i]))
    # gene bootstrap: C2 -> resample test gene; C3 -> two-way (query side x other side) product weights
    if name=='C2':
        tg=np.where(sub.gene_a.isin(test),sub.gene_a,sub.gene_b)
        ug,inv=np.unique(tg,return_inverse=True)
        for b in range(B):
            w=np.bincount(rng.integers(0,len(ug),len(ug)),minlength=len(ug))[inv]
            gb.append(ap(y,sc,sample_weight=w) if w.sum() else np.nan)
    else:
        isq_a=sub.gene_a.isin(gq).values
        qg=np.where(isq_a,sub.gene_a,sub.gene_b); og=np.where(isq_a,sub.gene_b,sub.gene_a)
        uq,iq=np.unique(qg,return_inverse=True); uo,io=np.unique(og,return_inverse=True)
        for b in range(B):
            wq=np.bincount(rng.integers(0,len(uq),len(uq)),minlength=len(uq))[iq]
            wo=np.bincount(rng.integers(0,len(uo),len(uo)),minlength=len(uo))[io]
            gb.append(ap(y,sc,sample_weight=wq*wo))
    print(name,'AP %.4f pairSE %.4f geneSE %.4f ratio %.1f'%(ap(y,sc),np.std(pb),np.nanstd(gb),np.nanstd(gb)/np.std(pb)))
# noise ceiling / replicate agreement on reciprocal pairs
p=pd.read_parquet(R+'ryan_pairs.parquet'); p=p[p.both_orientations & p.s_raw_ab.notna() & p.s_raw_ba.notna()]
yb=(p.s_raw_ba < -3.76).astype(int)  # single-orientation label on S2-equivalent threshold
print('reciprocal n',len(p),'prev single-orient label',yb.mean())
print('AP(other orientation raw -> single-orientation label) %.3f, lift %.1f'%(ap(yb,-p.s_raw_ab), ap(yb,-p.s_raw_ab)/yb.mean()))
ys=(p.s_score< -2.3).astype(int)
print('AP(one orientation -> S2 label, circular) %.3f prev %.3f'%(ap(ys,-p.s_raw_ab), ys.mean()))
