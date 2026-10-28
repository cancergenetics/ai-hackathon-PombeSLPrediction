#!/usr/bin/env python
"""Tier-2 SEQ feature group (DESIGN.md v3.1 §4, group SEQ).  Label-free.

Sources (config/data_allowlist.yaml dataset ids) and the ONLY parts read:
  pombase_protein   PeptideStats.tsv (Residues, Mass (kDa), pI, Charge, CAI);
                    transmembrane_domain_coords_and_seqs.tsv (col 4 = TM segment coordinates);
                    protein_domains_and_features.tsv (systematic_id, domain_id, database, seq_start, seq_end).
                    Domain IDs are used as OPAQUE identifiers, never mapped to GO (InterPro2GO is not used).
                    ID databases used: Pfam, PANTHER, CATH-Gene3D, SUPERFAMILY, SMART, PROSITE profiles,
                    PROSITE patterns, CDD, PRINTS (PI-confirmed sequence-computed list).  NOT used: CATH-FunFam,
                    NCBIFAM, PIRSF, HAMAP, SFLD (not on the confirmed list).  DeepTMHMM, Signal-Peptide-DeepTMHMM,
                    MobiDB-Disorder and Coiled coils rows give per-gene counts/fractions only.
  ensembl_fungi_compara  Compara.116.pombe_homologies.tsv.gz rows with both genes in S. pombe
                    (homology_type within_species_paralog / other_paralog): identity, homology_identity.
  esm2_weights      via data/processed/esm2_650M_pombe.parquet (scripts/compute_esm_embeddings.py; index = gene,
                    columns e0..e1279, mean-pooled final layer).  OPTIONAL: if the file is absent the ESM features
                    are skipped and must be built later by re-running this script.

Features
  gene/SEQ.parquet
    length, mass_kda, pI, charge, cai          PeptideStats
    n_pfam                                     # Pfam match instances;  n_pfam_ids  # distinct Pfam IDs
    n_domain_ids                               # distinct IDs over the 9 ID databases above
    has_panther                                gene has a PANTHER family
    n_tm                                       # TM segments in the PomBase TM file (0 if absent)
    n_tm_deeptmhmm, signal_peptide             DeepTMHMM helices / signal-peptide flag
    disorder_frac                              MobiDB-Disorder residues / length
    n_coiled_coils                             # coiled-coil segments
    n_paralogs, n_paralogs_recent, max_paralog_identity   Compara within-pombe paralogues
    esm_pc1..esm_pc32                          PCA of centred ESM-2 embeddings (if available)
  pair/SEQ__shared_pfam.npy      # distinct Pfam IDs shared (0 if none; defined for every gene)
  pair/SEQ__shared_panther.npy   1 if same PANTHER family, else 0 (defined for every gene)
  pair/SEQ__domain_jaccard.npy   Jaccard over the 9-database ID sets; NaN if both genes have no ID
  pair/SEQ__paralog_identity.npy max % identity over Compara paralogue rows and both directions; 0 if not
                                 paralogues (Compara covers the whole proteome: absence = no paralogy)
  pair/SEQ__paralog.npy          1 if Compara paralogues
  pair/SEQ__paralog_recent.npy   1 if paralogues with identity >= RECENT_ID (40%), else 0.
                                 "Ancient" = SEQ__paralog == 1 and SEQ__paralog_recent == 0.
  sim/esm_cos.npy                cosine similarity of centred ESM-2 embeddings (if available)

Run: .venv/bin/python -I scripts/features/build_seq.py
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

SCRIPT = "scripts/features/build_seq.py"
PP = ROOT / "data/raw/pombase_protein"
COMPARA = ROOT / "data/raw/ensembl_fungi_compara/Compara.116.pombe_homologies.tsv.gz"
ESM = ROOT / "data/processed/esm2_650M_pombe.parquet"
P_SRC, C_SRC, E_SRC = "pombase_protein", "ensembl_fungi_compara", "esm2_weights"
ID_DBS = ["Pfam", "PANTHER", "CATH-Gene3D", "SUPERFAMILY", "SMART", "PROSITE profiles", "PROSITE patterns",
          "CDD", "PRINTS"]
NOT_USED_DBS = ["CATH-FunFam", "NCBIFAM", "PIRSF", "HAMAP", "SFLD"]
RECENT_ID = 40.0
N_ESM_PC = 32
SEED = 0
CHECKS = ROOT / "results/features/seq_checks.json"
POMBE = "schizosaccharomyces_pombe"


def esm_features(E: pd.DataFrame, gi: dict, G: int, n_pc: int = N_ESM_PC):
    """E: index = systematic IDs (subset of gi), numeric columns.  Returns (PC DataFrame, G x G cosine of the
    centred embeddings, info dict)."""
    E = E[E.index.isin(gi)].astype(np.float64)
    assert E.index.is_unique and np.isfinite(E.values).all()
    X = E.values - E.values.mean(0)
    n_pc = min(n_pc, X.shape[0] - 1, X.shape[1])
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    pcs = pd.DataFrame(U[:, :n_pc] * sv[:n_pc], index=E.index, columns=[f"esm_pc{i + 1}" for i in range(n_pc)])
    evr = sv ** 2 / (sv ** 2).sum()
    S, ok = su.row_cosine(X)
    idx = np.array([gi[g] for g in E.index[ok]])
    M = su.scatter(S, idx, G)
    return pcs, M, {"n_genes": int(len(E)), "dim": int(E.shape[1]), "n_pc": n_pc,
                    "explained_variance_ratio_cum": float(evr[:n_pc].sum())}


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")  # spurious Accelerate matmul flags
    checks: dict = {"id_databases": ID_DBS, "id_databases_not_used": NOT_USED_DBS, "recent_identity": RECENT_ID}
    with fs.RuntimeLog("seq") as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        mapper = IDMapper()

        ps = pd.read_csv(PP / "PeptideStats.tsv", sep="\t")
        ps["gene"] = ps.Systematic_ID.map(mapper.map(ps.Systematic_ID, source="pombase_peptidestats"))
        ps = ps.dropna(subset=["gene"]).set_index("gene")
        gt = pd.DataFrame({"length": ps["Residues"], "mass_kda": ps["Mass (kDa)"], "pI": ps["pI"],
                           "charge": ps["Charge"], "cai": ps["CAI"]}).apply(pd.to_numeric, errors="coerce")
        base = gt.index                                    # genes with a peptide: counts below default to 0

        dom = pd.read_csv(PP / "protein_domains_and_features.tsv", sep="\t", dtype=str)
        dom["gene"] = dom.systematic_id.map(mapper.map(dom.systematic_id, source="pombase_protein_domains"))
        dom = dom.dropna(subset=["gene"])
        dom = dom[dom.gene.isin(gi)]
        checks["database_rows"] = dom.database.value_counts().to_dict()
        ids = dom[dom.database.isin(ID_DBS)]
        cnt = lambda m: m.groupby("gene").size().reindex(base, fill_value=0).astype(float)  # noqa: E731
        pf = ids[ids.database == "Pfam"]
        gt["n_pfam"] = cnt(pf)
        gt["n_pfam_ids"] = pf.groupby("gene").domain_id.nunique().reindex(base, fill_value=0).astype(float)
        gt["n_domain_ids"] = ids.groupby("gene").domain_id.nunique().reindex(base, fill_value=0).astype(float)
        gt["has_panther"] = (cnt(ids[ids.database == "PANTHER"]) > 0).astype(float)
        tm = pd.read_csv(PP / "transmembrane_domain_coords_and_seqs.tsv", sep="\t", header=None, dtype=str)
        tm["gene"] = tm[0].map(mapper.map(tm[0], source="pombase_tm"))
        tm = tm.dropna(subset=["gene"])
        ntm = tm.groupby("gene")[3].first().str.count(r"\.\.")
        gt["n_tm"] = ntm.reindex(base, fill_value=0).astype(float)
        gt["n_tm_deeptmhmm"] = cnt(dom[(dom.database == "DeepTMHMM") & (dom.domain_id == "Transmembrane alpha helix")])
        gt["signal_peptide"] = (cnt(dom[dom.database == "Signal-Peptide-DeepTMHMM"]) > 0).astype(float)
        dis = dom[dom.database == "MobiDB-Disorder"].copy()
        dis["len"] = dis.seq_end.astype(int) - dis.seq_start.astype(int) + 1
        gt["disorder_frac"] = (dis.groupby("gene").len.sum().reindex(base, fill_value=0) / gt["length"]).clip(0, 1)
        gt["n_coiled_coils"] = cnt(dom[dom.database == "Coiled coils"])
        checks["tm_agreement"] = {"genes_tm_file": int((gt.n_tm > 0).sum()),
                                  "genes_deeptmhmm": int((gt.n_tm_deeptmhmm > 0).sum()),
                                  "both": int(((gt.n_tm > 0) & (gt.n_tm_deeptmhmm > 0)).sum())}
        rt.step("protein_tables")

        # ---- domain-ID pair features
        p_files = [f"{P_SRC}:protein_domains_and_features.tsv databases {','.join(ID_DBS)} (opaque IDs)"]
        gl = np.array(sorted(base, key=gi.get))
        idx = np.array([gi[g] for g in gl])
        pos = {g: k for k, g in enumerate(gl)}

        def incidence(df):
            items = {d: k for k, d in enumerate(sorted(df.domain_id.unique()))}
            B = np.zeros((len(gl), len(items)), dtype=np.float32)
            B[[pos[g] for g in df.gene], [items[d] for d in df.domain_id]] = 1
            return B

        inter, _ = su.incidence_overlap(incidence(pf))
        fs.save_matrix("pair", "SEQ__shared_pfam", su.scatter(inter, idx, G), "SEQ", [P_SRC], SCRIPT,
                       source_files=[f"{P_SRC}:protein_domains_and_features.tsv database Pfam"],
                       description="# distinct Pfam IDs shared")
        inter, _ = su.incidence_overlap(incidence(ids[ids.database == "PANTHER"]))
        fs.save_matrix("pair", "SEQ__shared_panther", su.scatter((inter > 0).astype(np.float32), idx, G), "SEQ",
                       [P_SRC], SCRIPT, source_files=[f"{P_SRC}:protein_domains_and_features.tsv database PANTHER"],
                       description="same PANTHER family (1/0)")
        _, jac = su.incidence_overlap(incidence(ids))
        M = su.scatter(jac, idx, G)
        fs.save_matrix("pair", "SEQ__domain_jaccard", M, "SEQ", [P_SRC], SCRIPT, source_files=p_files,
                       description="Jaccard over domain/family IDs of 9 member databases")
        checks["pair_distribution"] = {"SEQ__domain_jaccard": su.dist_summary(M)}
        del M, inter, jac
        rt.step("domain_pairs")

        # ---- Compara within-pombe paralogues
        cols = ["gene_stable_id", "species", "identity", "homology_type", "homology_gene_stable_id",
                "homology_species", "homology_identity"]
        c = pd.read_csv(COMPARA, sep="\t", usecols=cols, dtype=str)
        c = c[(c.species == POMBE) & (c.homology_species == POMBE)]
        assert set(c.homology_type) <= {"within_species_paralog", "other_paralog"}, set(c.homology_type)
        m = mapper.map(pd.concat([c.gene_stable_id, c.homology_gene_stable_id]), source="compara_pombe_paralogs")
        c = c.assign(a=c.gene_stable_id.map(m), b=c.homology_gene_stable_id.map(m)).dropna(subset=["a", "b"])
        c = c[(c.a != c.b) & c.a.isin(gi) & c.b.isin(gi)]
        c["pid"] = np.fmax(c.identity.astype(float), c.homology_identity.astype(float))
        P = np.zeros((G, G), dtype=np.float32)
        ia, ib = c.a.map(gi).values, c.b.map(gi).values
        np.maximum.at(P, (ia, ib), c.pid.values.astype(np.float32))
        P = np.maximum(P, P.T)
        np.fill_diagonal(P, np.nan)
        par = (P > 0).astype(np.float32)
        rec = (P >= RECENT_ID).astype(np.float32)
        for M_, nm in ((par, "SEQ__paralog"), (rec, "SEQ__paralog_recent")):
            np.fill_diagonal(M_, np.nan)
        c_files = [f"{C_SRC}:Compara.116.pombe_homologies.tsv.gz rows pombe-pombe (within_species_paralog, "
                   "other_paralog): identity, homology_identity"]
        fs.save_matrix("pair", "SEQ__paralog_identity", P, "SEQ", [C_SRC], SCRIPT, source_files=c_files,
                       description="max % identity of Compara within-pombe paralogue pair (0 = not paralogues)")
        fs.save_matrix("pair", "SEQ__paralog", par, "SEQ", [C_SRC], SCRIPT, source_files=c_files,
                       description="Compara within-pombe paralogues (1/0)")
        fs.save_matrix("pair", "SEQ__paralog_recent", rec, "SEQ", [C_SRC], SCRIPT, source_files=c_files,
                       description=f"paralogues with max % identity >= {RECENT_ID:g} (1/0)")
        iu = np.triu_indices(G, 1)
        pv = P[iu]
        gt["n_paralogs"] = pd.Series(np.nansum(par, 1), index=genes).reindex(base)
        gt["n_paralogs_recent"] = pd.Series(np.nansum(rec, 1), index=genes).reindex(base)
        gt["max_paralog_identity"] = pd.Series(np.nanmax(np.where(np.isnan(P), 0, P), 1), index=genes).reindex(base)
        checks["paralogs"] = {"rows": int(len(c)), "types": c.homology_type.value_counts().to_dict(),
                              "n_pairs": int((pv > 0).sum()), "n_pairs_recent": int((pv >= RECENT_ID).sum()),
                              "n_genes_with_paralog": int((gt.n_paralogs > 0).sum()),
                              "identity_quantiles": {q: float(np.quantile(pv[pv > 0], q)) for q in (.1, .25, .5, .75, .9)},
                              "median_identity_by_type": c.groupby("homology_type").pid.median().to_dict()}
        del P, par, rec
        rt.step("paralogs")

        # ---- ESM-2 (optional)
        g_sources, colsrc = [P_SRC, C_SRC], {}
        if ESM.exists():
            E = pd.read_parquet(ESM)
            em = mapper.map(E.index, source="esm2_650M_pombe")
            E.index = E.index.map(em)
            E = E[E.index.notna()]
            pcs, M, info = esm_features(E, gi, G)
            fs.save_matrix("sim", "esm_cos", M, "SEQ", [E_SRC, P_SRC], SCRIPT,
                           source_files=[f"{E_SRC}:esm2_t33_650M_UR50D.pt", f"{P_SRC}:peptide.fa.gz",
                                         "data/processed/esm2_650M_pombe.parquet"],
                           description="cosine similarity of centred mean-pooled ESM-2 650M embeddings")
            checks["sim_distribution"] = {"esm_cos": su.dist_summary(M)}
            checks["esm"] = info
            del M
            gt = gt.join(pcs)
            g_sources.append(E_SRC)
            colsrc.update({c_: [E_SRC, P_SRC] for c_ in pcs.columns})
        else:
            checks["esm"] = f"SKIPPED: {ESM.relative_to(ROOT)} not found; re-run this script once it exists"
            print(checks["esm"])
        rt.step("esm")

        gt = gt[gt.index.isin(gi)]
        for c_ in ("n_paralogs", "n_paralogs_recent", "max_paralog_identity"):
            colsrc[c_] = [C_SRC]
        for c_ in gt.columns:
            colsrc.setdefault(c_, [P_SRC])
        desc = {"n_pfam": "# Pfam match instances", "n_pfam_ids": "# distinct Pfam IDs",
                "n_domain_ids": "# distinct IDs over 9 member databases", "has_panther": "has a PANTHER family",
                "n_tm": "# TM segments (PomBase TM file)", "n_tm_deeptmhmm": "# DeepTMHMM TM helices",
                "signal_peptide": "DeepTMHMM signal peptide", "disorder_frac": "MobiDB-Disorder residue fraction",
                "n_coiled_coils": "# coiled-coil segments", "n_paralogs": "# Compara within-pombe paralogues",
                "n_paralogs_recent": f"# paralogues with identity >= {RECENT_ID:g}%",
                "max_paralog_identity": "max % identity to a within-pombe paralogue (0 if none)"}
        fs.save_gene_table("SEQ", gt, g_sources, SCRIPT, column_sources=colsrc, descriptions=desc,
                           source_files=[f"{P_SRC}:PeptideStats.tsv", f"{P_SRC}:transmembrane_domain_coords_and_seqs.tsv"]
                           + p_files + c_files)
        checks["coverage_genes"] = {c_: int(gt[c_].notna().sum()) for c_ in gt.columns if not c_.startswith("esm_pc")
                                    or c_ == "esm_pc1"}
        rt.step("gene_table")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps({k: v for k, v in checks.items() if k not in ("sim_distribution", "pair_distribution")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
