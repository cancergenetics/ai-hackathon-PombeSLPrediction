#!/usr/bin/env python
"""Sanity checks for the ESM-2 embeddings written by scripts/compute_esm_embeddings.py.

    .venv/bin/python -I scripts/verify_esm_embeddings.py [--model 650M] [--skip-model]

Checks (exit code 1 if any hard check fails; results -> results/verify_esm_embeddings.json):
  1. No NaN/inf, float32, no constant/all-zero rows, expected shapes; log JSON consistent with parquet.
  2. Coverage: every PomBase protein-coding gene (data/raw/pombase/gene_IDs_names_products.tsv) and every Ryan
     array/query gene has an embedding (missing ones are listed); same for SGD ORFs.
  3. Positive controls: mean cosine of Ensembl Compara within-pombe paralogue pairs (and one2one pombe-cerevisiae
     orthologue pairs, if cerevisiae embeddings exist) > mean cosine of random pairs (permutation test + AUC),
     plus a negative control (random pairs vs random pairs with shuffled labels ~ 0 difference).
  4. Windowing control: for long proteins (>1022 aa) the whole-protein embedding is closer to the embedding of its own
     first window than to the first window of random other long proteins (needs the model).
  5. Determinism: 5 proteins (incl. one windowed) re-embedded twice from scratch give bit-identical vectors, and agree
     with the stored vectors (max abs diff reported; stored vectors were computed in different padded batches so a tiny
     numerical difference ~1e-5 is expected, cosine must be > 0.9999).
"""
import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cee", ROOT / "scripts/compute_esm_embeddings.py")
cee = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cee)

RES = {}
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    RES[name] = {"pass": bool(ok), "detail": str(detail)}
    if not ok:
        FAIL.append(name)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pair_cos(U, ia, ib):
    return (U[ia] * U[ib]).sum(1)


def auc_pos_vs_neg(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def random_pairs(n, k, rng, exclude=None):
    a = rng.integers(0, n, size=k * 2)
    b = rng.integers(0, n, size=k * 2)
    m = a != b
    return a[m][:k], b[m][:k]


def load_ensembl_pairs():
    p = ROOT / "data/raw/ensembl_fungi_compara/Compara.116.pombe_homologies.tsv.gz"
    df = pd.read_csv(p, sep="\t", usecols=["gene_stable_id", "species", "homology_type", "homology_gene_stable_id",
                                           "homology_species"], dtype=str)
    sp = "schizosaccharomyces_pombe"
    par = df[(df.species == sp) & (df.homology_species == sp) & (df.homology_type == "within_species_paralog")]
    par = par[["gene_stable_id", "homology_gene_stable_id"]].rename(columns={"gene_stable_id": "a", "homology_gene_stable_id": "b"})
    par = par[par.a != par.b].drop_duplicates()
    orth = df[(df.homology_type == "ortholog_one2one") & (
        ((df.species == sp) & (df.homology_species == "saccharomyces_cerevisiae")) |
        ((df.species == "saccharomyces_cerevisiae") & (df.homology_species == sp)))]
    o = orth.copy()
    flip = o.species == "saccharomyces_cerevisiae"
    o["pombe"] = np.where(flip, o.homology_gene_stable_id, o.gene_stable_id)
    o["cer"] = np.where(flip, o.gene_stable_id, o.homology_gene_stable_id)
    return par, o[["pombe", "cer"]].drop_duplicates()


def control_pairs(name, U, index, a_ids, b_ids, rng, U2=None, index2=None, idx_b=None):
    """Compare mean cosine of related pairs vs random pairs (same embedding space)."""
    pos_idx = {g: i for i, g in enumerate(index)}
    if U2 is None:
        keep = [(pos_idx[a], pos_idx[b]) for a, b in zip(a_ids, b_ids) if a in pos_idx and b in pos_idx]
        ia, ib = (np.array(x) for x in zip(*keep))
        pos = pair_cos(U, ia, ib)
        ra, rb = random_pairs(len(index), 20000, rng)
        neg = pair_cos(U, ra, rb)
    else:
        pi2 = {g: i for i, g in enumerate(index2)}
        keep = [(pos_idx[a], pi2[b]) for a, b in zip(a_ids, b_ids) if a in pos_idx and b in pi2]
        ia, ib = (np.array(x) for x in zip(*keep))
        pos = (U[ia] * U2[ib]).sum(1)
        ra = rng.integers(0, len(index), 20000)
        rb = rng.integers(0, len(index2), 20000)
        neg = (U[ra] * U2[rb]).sum(1)
    auc = auc_pos_vs_neg(pos, neg)
    # permutation test on the mean difference
    allv = np.concatenate([pos, neg])
    obs = pos.mean() - neg.mean()
    perm = []
    for _ in range(1000):
        rng.shuffle(allv)
        perm.append(allv[:len(pos)].mean() - allv[len(pos):].mean())
    p = (1 + np.sum(np.array(perm) >= obs)) / 1001
    # negative control: split the random pairs in two halves -> no difference expected
    h = len(neg) // 2
    nulldiff = neg[:h].mean() - neg[h:].mean()
    ok = obs > 0 and auc > 0.5 and p < 0.01 and abs(nulldiff) < 0.01
    check(f"positive_control_{name}", ok,
          f"n_pairs={len(pos)} mean_cos_related={pos.mean():.4f} mean_cos_random={neg.mean():.4f} "
          f"AUC={auc:.3f} perm_p={p:.4f} null_half_split_diff={nulldiff:.4f}")
    RES[f"positive_control_{name}"].update(n_pairs=int(len(pos)), mean_related=float(pos.mean()),
                                           mean_random=float(neg.mean()), auc=float(auc), perm_p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--skip-model", action="store_true", help="skip checks that need the model (windowing, determinism)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    dfs = {}
    for sp in ["pombe", "cerevisiae"]:
        p = ROOT / f"data/processed/esm2_{args.model}_{sp}.parquet"
        if p.exists():
            dfs[sp] = pd.read_parquet(p)
    check("pombe_parquet_exists", "pombe" in dfs)

    # 1. integrity
    for sp, df in dfs.items():
        X = df.to_numpy()
        norms = np.linalg.norm(X, axis=1)
        log = json.loads((ROOT / f"results/runtime/esm_{sp}.json").read_text())
        check(f"{sp}_no_nan_inf", np.isfinite(X).all())
        check(f"{sp}_float32", all(t == np.float32 for t in df.dtypes))
        check(f"{sp}_index_unique", df.index.is_unique)
        check(f"{sp}_no_degenerate_rows", (norms > 1e-3).all() and (X.std(1) > 1e-6).all(),
              f"n={len(df)} dim={X.shape[1]} norm range=({norms.min():.2f},{norms.max():.2f})")
        check(f"{sp}_log_consistent", log["n_proteins"] == len(df) and log["embedding_dim"] == X.shape[1],
              f"log: model={log['model']} device={log['device']} wall={log['wall_clock_s']}s cpu_fallbacks={log['n_cpu_fallback_sequences']}")

    # 2. coverage
    if "pombe" in dfs:
        g = pd.read_csv(ROOT / "data/raw/pombase/gene_IDs_names_products.tsv", sep="\t", dtype=str)
        pc = set(g[g.gene_type == "protein coding gene"].gene_systematic_id)
        have = set(dfs["pombe"].index)
        missing = sorted(pc - have)
        extra = sorted(have - pc)
        check("pombe_all_protein_coding_covered", not missing,
              f"protein_coding={len(pc)} covered={len(pc & have)} missing={len(missing)} {missing[:20]} extra_in_embeddings={len(extra)} {extra[:10]}")
        ry = pd.read_csv(ROOT / "data/processed/ryan_genes.tsv", sep="\t", dtype=str)
        rpc = set(ry[ry.gene_type == "protein coding gene"].pombase_id.dropna())
        rmiss = sorted(rpc - have)
        check("ryan_protein_coding_genes_covered", not rmiss, f"ryan_pc={len(rpc)} missing={len(rmiss)} {rmiss[:20]}")
    if "cerevisiae" in dfs:
        n_orf = sum(1 for _ in cee.read_fasta(ROOT / "data/raw/sgd_protein/orf_trans_all.fasta.gz"))
        check("cerevisiae_all_SGD_orfs_covered", len(dfs["cerevisiae"]) == n_orf, f"fasta_records={n_orf} embedded={len(dfs['cerevisiae'])}")

    # 3. positive controls
    par, orth = load_ensembl_pairs()
    if "pombe" in dfs:
        U = unit(dfs["pombe"].to_numpy().astype(np.float64))
        control_pairs("pombe_paralogs", U, list(dfs["pombe"].index), par.a, par.b, rng)
        if "cerevisiae" in dfs:
            U2 = unit(dfs["cerevisiae"].to_numpy().astype(np.float64))
            control_pairs("pombe_cerevisiae_one2one_orthologs", U, list(dfs["pombe"].index), orth.pombe, orth.cer, rng,
                          U2=U2, index2=list(dfs["cerevisiae"].index))

    # 4/5. model-based checks
    if not args.skip_model and "pombe" in dfs:
        import torch
        seqs, _ = cee.load_proteins("pombe")
        idx = list(dfs["pombe"].index)
        lens = pd.Series({k: len(v) for k, v in seqs.items()})
        long_ids = sorted(lens[lens > cee.MAX_LEN].index)
        short_ids = sorted(lens[lens <= cee.MAX_LEN].index)
        r = np.random.default_rng(1)
        pick = list(r.choice(short_ids, 4, replace=False)) + [str(r.choice(long_ids))]
        sub = {k: seqs[k] for k in pick}
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        runs = []
        for _ in range(2):
            e = cee.Embedder(args.model, dev)
            runs.append(e.embed_proteins(sub, max_tokens=4096, quiet=True))
            del e
        ident = all(np.array_equal(runs[0][k], runs[1][k]) for k in pick)
        stored = dfs["pombe"].loc[pick].to_numpy()
        mine = np.stack([runs[0][k] for k in pick])
        maxdiff = float(np.abs(stored - mine).max())
        cos = float((unit(stored.astype(np.float64)) * unit(mine.astype(np.float64))).sum(1).min())
        check("determinism_rerun_bit_identical", ident, f"proteins={pick} device={dev}")
        check("rerun_matches_stored", cos > 0.9999, f"min_cosine={cos:.7f} max_abs_diff={maxdiff:.2e}")

        # windowing control: first-window-only embedding vs whole-protein embedding
        sel = long_ids[:20]
        emb = cee.Embedder(args.model, dev)
        firsts = emb.embed_proteins({k: seqs[k][:cee.MAX_LEN] for k in sel}, max_tokens=4096, quiet=True)
        F = unit(np.stack([firsts[k] for k in sel]).astype(np.float64))
        W = unit(dfs["pombe"].loc[sel].to_numpy().astype(np.float64))
        same = (F * W).sum(1)
        S = F @ W.T
        other = S[~np.eye(len(sel), dtype=bool)]
        # and vs random short proteins
        check("windowed_embedding_closer_to_own_first_window", same.mean() > other.mean() and (same > S.max(1) - 1e-12).mean() >= 0.9,
              f"n_long={len(sel)} cos_own={same.mean():.4f} cos_other_long={other.mean():.4f} "
              f"frac_own_is_best_match={(S.argmax(1) == np.arange(len(sel))).mean():.2f}")

    json.dump(RES, open(ROOT / "results/verify_esm_embeddings.json", "w"), indent=2)
    print("\nFAILED:" if FAIL else "\nAll checks passed.", FAIL if FAIL else "")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
