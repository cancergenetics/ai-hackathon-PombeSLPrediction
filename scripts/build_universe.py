#!/usr/bin/env python
"""Derive the prediction universe and training-drop gene list (DESIGN.md §1) from PomBase + Ryan.

Rules (from config/protocol.yaml `universe`):
  * universe = protein-coding genes with FYPO deletion viability in `viability_include`
    PLUS unknown-viability genes that have a main-allele DELETION strain in Ryan: an array strain
    (Bioneer deletion library) or a main query allele that is not DAmP (unlabelled / plain deletion).
    Unknown-viability genes screened only as DAmP (sap10) are excluded.
  * drop_from_training = PomBase-inviable genes that have a Ryan "deletion" array strain.
FYPO viability is used ONLY here (universe/masks/stratification), never as a feature.

Outputs:
  <data_root>/universe_genes.tsv   one row per PomBase protein-coding gene (+ flags)
  config/protocol.yaml             derived:universe block (gene lists + counts + tsv sha256)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import protocol  # noqa: E402

RAW = ROOT / "data" / "raw"


def main() -> int:
    P = protocol.load()["universe"]
    out_dir = protocol.data_root()
    pb = pd.read_csv(RAW / "pombase/gene_IDs_names_products.tsv", sep="\t", dtype=str)
    viab = pd.read_csv(RAW / "pombase/FYPOviability.tsv", sep="\t", header=None, names=["gene", "viability"],
                       dtype=str).drop_duplicates("gene").set_index("gene").viability
    genes = pd.read_csv(out_dir / "ryan_genes.tsv", sep="\t").dropna(subset=["pombase_id"])
    meas = pd.read_parquet(out_dir / "ryan_measurements.parquet",
                           columns=["query_gene", "allele_class", "main_allele"])

    pc = pb[pb.gene_type == P["gene_type"]].copy()
    df = pd.DataFrame({"gene": pc.gene_systematic_id.values, "gene_name": pc.gene_name.values})
    df["viability"] = df.gene.map(viab).fillna("unknown")

    arr = set(genes.loc[genes.is_array, "pombase_id"])
    mainq = meas[meas.main_allele & meas.query_gene.notna()]
    del_q = set(mainq.loc[mainq.allele_class != "DAmP", "query_gene"])
    damp_q = set(mainq.loc[mainq.allele_class == "DAmP", "query_gene"])
    df["ryan_array_strain"] = df.gene.isin(arr)
    df["ryan_main_deletion_query"] = df.gene.isin(del_q)
    df["ryan_main_damp_query"] = df.gene.isin(damp_q)
    df["ryan_main_deletion_strain"] = df.ryan_array_strain | df.ryan_main_deletion_query
    in_ryan = df.ryan_array_strain | df.gene.isin(del_q | damp_q)

    via_ok = df.viability.isin(P["viability_include"])
    assert P["unknown_viability_rule"] == "unknown_with_ryan_main_deletion_strain"
    unk_del = (df.viability == "unknown") & df.ryan_main_deletion_strain
    unk_damp_only = (df.viability == "unknown") & in_ryan & ~df.ryan_main_deletion_strain
    df["in_universe"] = via_ok | unk_del
    df["universe_reason"] = ""
    df.loc[via_ok, "universe_reason"] = "viability:" + df.loc[via_ok, "viability"]
    df.loc[unk_del, "universe_reason"] = "unknown_viability_with_ryan_deletion_strain"
    df["viability_unknown_flag"] = (df.viability == "unknown") & ~df.in_universe
    assert P["drop_from_training"] == "inviable_with_ryan_array_strain"
    df["drop_from_training"] = (df.viability == "inviable") & df.ryan_array_strain
    df = df.sort_values("gene").reset_index(drop=True)
    out = out_dir / "universe_genes.tsv"
    df.to_csv(out, sep="\t", index=False)

    d = {
        "universe_genes_tsv": "universe_genes.tsv (under data root)",
        "universe_genes_tsv_sha256": protocol.sha256_file(out),
        "n_universe_genes": int(df.in_universe.sum()),
        "n_by_reason": {k: int(v) for k, v in df.loc[df.in_universe, "universe_reason"].value_counts().items()},
        "unknown_with_ryan_deletion_strain": sorted(df.loc[unk_del, "gene"]),
        "unknown_excluded_damp_only": sorted(df.loc[unk_damp_only, "gene"]),
        "n_unknown_flagged_not_in_universe": int(df.viability_unknown_flag.sum()),
        "drop_from_training_genes": sorted(df.loc[df.drop_from_training, "gene"]),
    }
    protocol.write_derived_block("universe", {"universe_derived": d}, "scripts/build_universe.py")
    print(f"universe genes: {d['n_universe_genes']} {d['n_by_reason']}")
    print(f"unknown-viability genes with Ryan deletion strain: {len(d['unknown_with_ryan_deletion_strain'])}; "
          f"excluded (DAmP-only): {d['unknown_excluded_damp_only']}")
    print(f"unknown-viability genes outside universe (flagged): {d['n_unknown_flagged_not_in_universe']}")
    print(f"drop-from-training genes (inviable with Ryan array strain): {len(d['drop_from_training_genes'])}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
