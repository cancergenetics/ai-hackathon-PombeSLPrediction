#!/usr/bin/env python
"""Tier-2 CONS feature group (DESIGN.md v3.1 §4, group CONS).  Label-free.

Source (config/data_allowlist.yaml dataset id) and the ONLY parts read:
  ensembl_fungi_compara  Compara.116.pombe_homologies.tsv.gz (gene-tree homologies, rows involving
      S. pombe): gene_stable_id, species, homology_type, homology_gene_stable_id, homology_species,
      identity, homology_identity.  Only ortholog_* rows between S. pombe and another genome are used here
      (within-pombe paralogues are used by build_seq.py).

Phylogenetic profile: for each pombe gene, a binary vector over the 367 other genomes, 1 if Compara reports
any ortholog (one2one, one2many or many2many).  Genes absent from all ortholog rows have an all-zero profile
(species-specific / orphan genes; breadth 0).

Features
  gene/CONS.parquet
    n_species             # of the 367 genomes with an ortholog (conservation breadth)
    frac_species          n_species / 367
    has_sjaponicus        ortholog in Schizosaccharomyces japonicus (Ensembl Fungi genome)
    n_schizo              # of S. japonicus / S. octosporus / S. cryophilus with an ortholog (0-3)
    has_scerevisiae       ortholog in S. cerevisiae (Compara; the CER group uses the PomBase curated map)
    mean_ortholog_identity mean over genomes of the best % identity (pombe side) of the gene's orthologs
    frac_one2one          fraction of ortho-genomes in which the relationship is one2one
    phylo_pc1..phylo_pc10 PCA of the binary profile matrix
  sim/cons_phylo.npy      Pearson r (phi coefficient) between binary profiles; NaN for constant profiles
                          (present in all 367 or in none), for which correlation is undefined.

Run: .venv/bin/python -I scripts/features/build_cons.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import sim_utils as su  # noqa: E402
from src.idmap import IDMapper  # noqa: E402

SCRIPT = "scripts/features/build_cons.py"
COMPARA = ROOT / "data/raw/ensembl_fungi_compara/Compara.116.pombe_homologies.tsv.gz"
SRC = "ensembl_fungi_compara"
POMBE = "schizosaccharomyces_pombe"
SCHIZO = ["schizosaccharomyces_japonicus", "schizosaccharomyces_octosporus", "schizosaccharomyces_cryophilus"]
N_PC = 10
CHECKS = ROOT / "results/features/cons_checks.json"


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")  # spurious Accelerate matmul flags
    checks: dict = {}
    with fs.RuntimeLog("cons") as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        cols = ["gene_stable_id", "species", "identity", "homology_type", "homology_gene_stable_id",
                "homology_species", "homology_identity"]
        c = pd.read_csv(COMPARA, sep="\t", usecols=cols, dtype=str)
        c = c[c.homology_type.str.startswith("ortholog_")]
        # orient every row as (pombe gene, other genome)
        a = c[(c.species == POMBE) & (c.homology_species != POMBE)]
        b = c[(c.homology_species == POMBE) & (c.species != POMBE)]
        o = pd.concat([pd.DataFrame({"pombe": a.gene_stable_id, "sp": a.homology_species, "pid": a.identity,
                                     "type": a.homology_type}),
                       pd.DataFrame({"pombe": b.homology_gene_stable_id, "sp": b.species, "pid": b.homology_identity,
                                     "type": b.homology_type})], ignore_index=True)
        assert len(o) == len(c), "ortholog rows not involving exactly one pombe gene"
        m = IDMapper().map(o.pombe, source="compara_pombe_orthologs")
        o["gene"] = o.pombe.map(m)
        o = o.dropna(subset=["gene"])
        o = o[o.gene.isin(gi)]
        o["pid"] = o.pid.astype(float)
        species = sorted(o.sp.unique())
        checks["input"] = {"ortholog_rows": int(len(o)), "n_genomes": len(species),
                           "types": o.type.value_counts().to_dict(),
                           "schizo_genomes_present": [s for s in SCHIZO if s in species],
                           "has_scerevisiae_genome": "saccharomyces_cerevisiae" in species}
        assert all(s in species for s in SCHIZO)
        rt.step("read")

        best = o.groupby(["gene", "sp"]).agg(pid=("pid", "max"),
                                             one2one=("type", lambda t: float((t == "ortholog_one2one").any())))
        B = (best.pid.unstack("sp").reindex(index=genes, columns=species).notna()).astype(np.float32)
        nsp = B.sum(1)
        gt = pd.DataFrame(index=genes)
        gt["n_species"] = nsp
        gt["frac_species"] = nsp / len(species)
        gt["has_sjaponicus"] = B["schizosaccharomyces_japonicus"]
        gt["n_schizo"] = B[SCHIZO].sum(1)
        gt["has_scerevisiae"] = B["saccharomyces_cerevisiae"]
        gt["mean_ortholog_identity"] = best.pid.groupby("gene").mean().reindex(genes)
        gt["frac_one2one"] = best.one2one.groupby("gene").mean().reindex(genes)
        Xc = B.values - B.values.mean(0)
        U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
        for k in range(N_PC):
            gt[f"phylo_pc{k + 1}"] = U[:, k] * sv[k]
        evr = sv ** 2 / (sv ** 2).sum()
        files = [f"{SRC}:Compara.116.pombe_homologies.tsv.gz ortholog_* rows (pombe vs {len(species)} genomes)"]
        desc = {"n_species": f"# of {len(species)} fungal genomes with a Compara ortholog",
                "frac_species": "n_species / # genomes", "has_sjaponicus": "S. japonicus ortholog (Compara)",
                "n_schizo": "# of S. japonicus/octosporus/cryophilus with an ortholog",
                "has_scerevisiae": "S. cerevisiae ortholog (Compara)",
                "mean_ortholog_identity": "mean over genomes of best ortholog % identity (pombe side)",
                "frac_one2one": "fraction of ortholog genomes with a one2one relationship",
                **{f"phylo_pc{k + 1}": f"PC{k + 1} of the binary phylogenetic profile" for k in range(N_PC)}}
        fs.save_gene_table("CONS", gt, [SRC], SCRIPT, descriptions=desc, source_files=files)
        checks["genes"] = {"n_with_any_ortholog": int((nsp > 0).sum()), "n_in_all_genomes": int((nsp == len(species)).sum()),
                           "n_species_quantiles": {q: float(nsp.quantile(q)) for q in (.1, .25, .5, .75, .9)},
                           "n_has_sjaponicus": int(gt.has_sjaponicus.sum()),
                           "n_has_scerevisiae": int(gt.has_scerevisiae.sum()),
                           "phylo_pca_explained_variance": [round(float(v), 4) for v in evr[:N_PC]]}
        checks["coverage_genes"] = {c_: int(gt[c_].notna().sum()) for c_ in gt.columns}
        rt.step("gene_table")

        S, ok = su.row_pearson(B.values)
        idx = np.arange(G)[ok]
        M = su.scatter(S, idx, G)
        fs.save_matrix("sim", "cons_phylo", M, "CONS", [SRC], SCRIPT, source_files=files,
                       description=f"Pearson (phi) correlation of binary ortholog presence over {len(species)} genomes")
        checks["sim_distribution"] = {"cons_phylo": su.dist_summary(M)}
        checks["n_genes_sim"] = int(ok.sum())
        rt.step("sim")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
