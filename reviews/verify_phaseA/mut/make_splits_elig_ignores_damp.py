#!/usr/bin/env python
"""Build the physically separated label files and the development CV splits (DESIGN.md §1–§2).

Inputs (data root = $PSL_DATA_ROOT or data/processed):
  ryan_pairs.parquet, ryan_measurements.parquet, ryan_genes.tsv, universe_genes.tsv
  config/protocol.yaml, config/lockbox_genes.txt (sha256 checked)
Outputs (<data root>/splits/):
  labels_dev.parquet        dev x dev S2 pairs minus lockbox-C1 pairs, with eval-mask columns
  labels_lockbox.parquet    S2 pairs with >=1 lockbox gene (part C2/C3) + lockbox-C1 pairs (part C1)
  dev_folds_seed<S>.tsv     gene -> fold (S = dev_cv.split_seed and dev_cv.alt_split_seed)
  c1_holdout_seed<S>.parquet  per-fold C1 pair hold-out (gene_a, gene_b, fold)
  manifest.json             sha256 of inputs/outputs + protocol hash
  results/splits_summary.json  aggregate counts only (for the real data root)

Per-pair columns (all derived here, documented):
  y            = label (protocol label column/threshold)
  y_raw        = report-only sensitivity label on s_raw_mean
  in_universe  = both genes in the universe (universe_genes.tsv)
  damp_only    = every main-allele S1 measurement of the pair used a DAmP query allele (S2 is DAmP-only)
  damp_mixed   = the pair has both DAmP and non-DAmP main-allele measurements
  hypomorph_a/b= the pair has a main-allele measurement with a DAmP allele of gene_a / gene_b
                 (training rows only; data.py sets it to 0 on eval rows)
  train_drop   = involves a PomBase-inviable gene with a Ryan deletion array strain (dropped from training)
  eval_eligible= in_universe & ~damp_only & ~involves_merged_id
  viable_viable= both genes PomBase-viable (secondary report)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/colmclaude/ClaudeExperiments/PombeSL_prediction')
sys.path.insert(0, str(ROOT))
from src import protocol  # noqa: E402
from src import splits as S  # noqa: E402

DEV_COLS = ["gene_a", "gene_b", "s_score", "y", "s_raw_mean", "y_raw", "s_raw_ab", "s_raw_ba",
            "both_orientations", "in_universe", "viable_viable", "damp_only", "damp_mixed",
            "involves_merged_id", "eval_eligible", "train_drop", "hypomorph_a", "hypomorph_b"]


def gene_roles(genes: pd.DataFrame) -> pd.Series:
    g = genes.dropna(subset=["pombase_id"]).groupby("pombase_id").agg(q=("is_query", "any"), a=("is_array", "any"))
    return pd.Series(np.where(g.q & g.a, "both", np.where(g.q, "query", "array")), index=g.index)


def damp_flags(root: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-pair DAmP flags from the S1 measurement table (main alleles = those entering S2)."""
    f = root / "ryan_measurements.parquet"
    cols = ["damp_only", "damp_mixed", "hypomorph_a", "hypomorph_b"]
    if not f.exists():
        print(f"WARNING: {f} missing -> no DAmP information (all DAmP flags False)")
        return pd.DataFrame(False, index=pairs.index, columns=cols)
    m = pd.read_parquet(f, columns=["query_gene", "array_gene", "allele_class", "main_allele", "self_pair"])
    m = m[m.main_allele & ~m.self_pair & m.query_gene.notna() & m.array_gene.notna()]
    m = m[m.query_gene != m.array_gene]
    qa = m.query_gene.to_numpy(dtype=object)
    ar = m.array_gene.to_numpy(dtype=object)
    d = pd.DataFrame({"gene_a": np.where(qa < ar, qa, ar), "gene_b": np.where(qa < ar, ar, qa)})
    damp = (m.allele_class == "DAmP").to_numpy()
    d["damp"] = damp
    d["damp_a"] = damp & (qa < ar)          # DAmP allele of the query, which is gene_a
    d["damp_b"] = damp & ~(qa < ar)
    agg = d.groupby(["gene_a", "gene_b"]).agg(n=("damp", "size"), nd=("damp", "sum"),
                                              ha=("damp_a", "any"), hb=("damp_b", "any"))
    j = pairs[["gene_a", "gene_b"]].join(agg, on=["gene_a", "gene_b"])
    n, nd = j.n.fillna(0).to_numpy(), j.nd.fillna(0).to_numpy()
    return pd.DataFrame({"damp_only": (n > 0) & (nd == n), "damp_mixed": (nd > 0) & (nd < n),
                         "hypomorph_a": j.ha.fillna(False).to_numpy(bool),
                         "hypomorph_b": j.hb.fillna(False).to_numpy(bool)}, index=pairs.index)


def load_universe(root: Path, genes: pd.DataFrame):
    f = root / "universe_genes.tsv"
    if f.exists():
        u = pd.read_csv(f, sep="\t")
        return (set(u.gene[u.in_universe]), set(u.gene[u.viability == "viable"]),
                set(u.gene[u.drop_from_training]))
    print(f"WARNING: {f} missing (toy data?) -> universe from ryan_genes.tsv viability")
    P = protocol.load()["universe"]
    g = genes.dropna(subset=["pombase_id"])
    return (set(g.pombase_id[g.viability.isin(P["viability_include"])]),
            set(g.pombase_id[g.viability == "viable"]),
            set(g.pombase_id[(g.viability == "inviable") & g.is_array]))


def load_lockbox(root: Path, s2_genes: set, roles: pd.Series, pairs: pd.DataFrame) -> list:
    L = protocol.load()["lockbox"]
    f = ROOT / L["genes_file"]
    if protocol.sha256_file(f) != L["genes_sha256"]:
        raise SystemExit(f"{f} sha256 mismatch with protocol.yaml — the lockbox must never change")
    lb = [g for g in f.read_text().split() if g]
    if len(set(lb) & s2_genes) > 0:
        return lb
    # toy data: the real lockbox genes are absent. Use/draw a toy lockbox under the toy data root only.
    if root.resolve() == (ROOT / protocol.load()["paths"]["data_root_default"]).resolve():
        raise SystemExit("no lockbox gene found among S2 genes of the real data — refusing to continue")
    tf = root / "lockbox_genes.txt"
    if not tf.exists():
        St = protocol.load()["stratification"]
        strata = S.stratify_genes(S.gene_hit_rates(pairs), roles, St["hub_quantile"], St["n_hit_rate_quantiles"])
        rng = np.random.default_rng(L["c1_seed"])
        frac = L["gene_fraction"]
        pick = []
        for s in sorted(strata.unique()):
            ids = np.sort(strata.index[strata == s].values)
            pick += list(rng.choice(ids, int(round(frac * len(ids))), replace=False))
        tf.write_text("\n".join(sorted(pick)) + "\n")
        print(f"toy data: drew toy lockbox ({len(pick)} genes) -> {tf}")
    return [g for g in tf.read_text().split() if g]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    PR = protocol.load()
    root = protocol.data_root()
    out = protocol.splits_dir()
    out.mkdir(parents=True, exist_ok=True)
    real = root.resolve() == (ROOT / PR["paths"]["data_root_default"]).resolve()

    pairs = pd.read_parquet(root / "ryan_pairs.parquet")
    lab = PR["label"]["column"]
    pairs = pairs[pairs[lab].notna()].reset_index(drop=True)
    genes = pd.read_csv(root / "ryan_genes.tsv", sep="\t")
    roles = gene_roles(genes)
    universe, viable, train_drop_genes = load_universe(root, genes)

    pairs["y"] = protocol.binarize(pairs[lab])
    pairs["y_raw"] = protocol.binarize(pairs[PR["label"]["sensitivity_column"]])
    pairs = pd.concat([pairs, damp_flags(root, pairs)], axis=1)
    pairs["in_universe"] = pairs.gene_a.isin(universe) & pairs.gene_b.isin(universe)
    pairs["viable_viable"] = pairs.gene_a.isin(viable) & pairs.gene_b.isin(viable)
    pairs["train_drop"] = pairs.gene_a.isin(train_drop_genes) | pairs.gene_b.isin(train_drop_genes)
    E = PR["eval"]
    elig = np.ones(len(pairs), bool)
    if E["require_both_in_universe"]:
        elig &= pairs.in_universe
    if E["exclude_damp_only_pairs"]:
        pass  # MUTANT: damp-only pairs not excluded
    if E["exclude_merged_id_pairs"]:
        elig &= ~pairs.involves_merged_id
    pairs["eval_eligible"] = elig
    for c in ("hypomorph_a", "hypomorph_b"):
        pairs[c] = pairs[c].astype("int8")
    pairs = pairs[DEV_COLS]

    # ---- lockbox -----------------------------------------------------------------------------
    s2_genes = set(pairs.gene_a) | set(pairs.gene_b)
    lockbox = set(load_lockbox(root, s2_genes, roles, pairs))
    n_lb = pairs.gene_a.isin(lockbox).astype(int) + pairs.gene_b.isin(lockbox).astype(int)
    devdev = (n_lb == 0).to_numpy()
    L = PR["lockbox"]
    lb_c1 = S.stratified_pair_sample(pairs.y, devdev & pairs.eval_eligible.to_numpy(), L["c1_fraction"],
                                     np.random.default_rng(L["c1_seed"]))
    lock = pairs[(n_lb > 0).to_numpy() | lb_c1].copy()
    lock["part"] = np.where(lb_c1[lock.index], "C1", np.where(n_lb[lock.index] == 2, "C3", "C2"))
    dev = pairs[devdev & ~lb_c1].reset_index(drop=True)
    viol = S.check_dev_labels(dev, lockbox, lock[lock.part == "C1"])
    assert not viol, viol
    dev.to_parquet(out / "labels_dev.parquet", index=False)
    lock.reset_index(drop=True).to_parquet(out / "labels_lockbox.parquet", index=False)

    summary = {"protocol_sha256": protocol.protocol_sha256(), "data_root": str(root.relative_to(ROOT))
               if root.is_relative_to(ROOT) else str(root),
               "s2_pairs": int(len(pairs)), "s2_genes": len(s2_genes), "lockbox_genes_in_s2": len(lockbox & s2_genes),
               "universe_genes": len(universe), "eval_eligible_pairs_all": int(pairs.eval_eligible.sum()),
               "excluded_damp_only_universe_pairs": int((pairs.in_universe & pairs.damp_only).sum()),
               "damp_mixed_universe_pairs": int((pairs.in_universe & pairs.damp_mixed).sum()),
               "merged_id_universe_pairs": int((pairs.in_universe & pairs.involves_merged_id).sum()),
               "train_drop_pairs": int(pairs.train_drop.sum())}
    le = lock[lock.eval_eligible]
    summary["lockbox"] = {p: {"pairs_eligible": int((le.part == p).sum()), "positives": int(le.y[le.part == p].sum()),
                              "pairs_all": int((lock.part == p).sum())} for p in S.PARTS}
    summary["dev"] = {"pairs": int(len(dev)), "positives": int(dev.y.sum()),
                      "genes": int(len(set(dev.gene_a) | set(dev.gene_b))),
                      "eval_eligible_pairs": int(dev.eval_eligible.sum()),
                      "eval_eligible_positives": int(dev.y[dev.eval_eligible].sum())}

    # ---- dev folds ---------------------------------------------------------------------------
    D, St = PR["dev_cv"], PR["stratification"]
    hit = S.gene_hit_rates(dev)
    strata = S.stratify_genes(hit, roles, St["hub_quantile"], St["n_hit_rate_quantiles"])
    summary["folds"] = {}
    for seed in (D["split_seed"], D["alt_split_seed"]):
        gfold = S.assign_folds(strata, D["n_folds"], seed)
        gfold.rename("fold").rename_axis("gene").reset_index().to_csv(out / f"dev_folds_seed{seed}.tsv",
                                                                       sep="\t", index=False)
        fa, fb = dev.gene_a.map(gfold).to_numpy(), dev.gene_b.map(gfold).to_numpy()
        hold, per = [], {}
        for k in range(D["n_folds"]):
            neither = (fa != k) & (fb != k)
            c1 = S.stratified_pair_sample(dev.y, neither & dev.eval_eligible.to_numpy(), D["c1_fraction"],
                                          np.random.default_rng([seed, k]))
            hold.append(dev.loc[c1, ["gene_a", "gene_b"]].assign(fold=k))
            part = S.fold_parts(fa, fb, k, c1, dev.eval_eligible.to_numpy(), ~dev.train_drop.to_numpy())
            tr = dev[part == "train"]
            ev = dev[np.isin(part, S.PARTS)].assign(part=part[np.isin(part, S.PARTS)])
            v = S.check_fold_split(tr, ev, gfold.index[gfold == k])
            assert not v, (seed, k, v)
            per[k] = {"genes": int((gfold == k).sum()), "train_pairs": int(len(tr)), "train_positives": int(tr.y.sum()),
                      **{f"{p}_pairs": int((ev.part == p).sum()) for p in S.PARTS},
                      **{f"{p}_positives": int(ev.y[ev.part == p].sum()) for p in S.PARTS}}
        pd.concat(hold).reset_index(drop=True).to_parquet(out / f"c1_holdout_seed{seed}.parquet", index=False)
        summary["folds"][str(seed)] = per

    files = sorted(p.name for p in out.iterdir() if p.suffix in (".parquet", ".tsv"))
    man = {"protocol_sha256": protocol.protocol_sha256(),
           "inputs": {f: protocol.sha256_file(root / f) for f in
                      ("ryan_pairs.parquet", "ryan_genes.tsv", "universe_genes.tsv", "ryan_measurements.parquet")
                      if (root / f).exists()},
           "lockbox_genes_sha256": L["genes_sha256"],
           "outputs": {f: protocol.sha256_file(out / f) for f in files}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    summary["manifest_outputs_sha256"] = man["outputs"]
    dest = (protocol.results_root() / "splits_summary.json") if real else (out / "splits_summary.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest_outputs_sha256"}, indent=1))
    print(f"wrote {out} and {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
