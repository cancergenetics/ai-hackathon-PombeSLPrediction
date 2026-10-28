#!/usr/bin/env python
"""Build the Ryan et al. 2012 S. pombe E-MAP gene-pair table with current PomBase IDs.

Inputs (from scripts/download_core_data.py):
  data/raw/ryan2012/extracted/mmc4/Dataset S1 - Unaveraged E-MAP with 953 Alleles.txt
      953 query alleles (rows) x 1955 array strains (cols), raw (unaveraged) S-scores.
  data/raw/ryan2012/extracted/mmc5/Dataset S2 - Averaged E-MAP one allele per gene.txt
      862 query genes x 1955 array genes, authors' averaged, symmetric dataset (CR line endings).
  data/raw/pombase/gene_IDs_names_products.tsv, FYPOviability.tsv (pinned PomBase release)

Outputs:
  data/processed/ryan_measurements.parquet  one row per individual S1 measurement (query allele x array)
  data/processed/ryan_pairs.parquet         one row per unordered measured gene pair (current PomBase IDs)
  data/processed/ryan_genes.tsv             one row per Ryan identifier with ID mapping + metadata
  data/processed/ryan_s2_transform.tsv      empirical monotone maps raw mean -> S2, separately for pairs
                                            measured in one orientation and in both (see below)

How duplicate measurements are handled (documented, not silent; flagged for PI in DATA.md):
  A gene pair {A,B} can be measured several times in S1: A-query x B-array and B-query x A-array
  (both orientations) and, for 46 query genes, with several query alleles (DAmP, point mutants, ts
  alleles, independent deletion strains ...). Columns provided:
    s_score      = the authors' Dataset S2 value. We verified (scripts/verify_core_data.py) that
                   S2 = f_k(s_raw_mean), where s_raw_mean is the mean of the per-orientation means over
                   the "main" allele(s) of each query gene (unlabelled strain plus, where present, its
                   DAmP strain, see MAIN_ALLELE_SUFFIXES; point mutants / ts / extra deletion strains are
                   excluded) and f_k is one of two global monotone maps: k=2 orientations -> f_2 ~ identity;
                   k=1 orientation -> f_1 strongly compressive. This is the E-MAP "pseudo-averaging" of
                   Collins et al. 2006 (single measurements are shrunk to be comparable with averaged
                   reciprocal measurements). s_score is the PRIMARY value (simplest defensible: the
                   authors' own one-value-per-pair dataset, on the scale where the paper's -2.3 cut-off is
                   used). NOTE: for single-orientation pairs S2 = -2.3 needs a raw score of about -3.7.
    s_raw_mean   = mean of the per-orientation raw means (main alleles) on the raw (unshrunk) scale.
    s_raw_mean_meas = plain mean over all main-allele measurements (differs only when one orientation
                   has several alleles).
    s_raw_mean_all = mean over every S1 measurement (all alleles) on the raw scale.
    s_raw_ab / s_raw_ba = orientation-specific raw means (gene_a as query / gene_b as query).
  Pairs that only exist in S1 (measured only with non-main alleles) have s_score = NaN.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import protocol  # noqa: E402  (label threshold is read from config/protocol.yaml)

THR = protocol.threshold()
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
S1_PATH = RAW / "ryan2012/extracted/mmc4/Dataset S1 - Unaveraged E-MAP with 953 Alleles.txt"
S2_PATH = RAW / "ryan2012/extracted/mmc5/Dataset S2 - Averaged E-MAP one allele per gene.txt"
PB_GENES = RAW / "pombase/gene_IDs_names_products.tsv"
PB_VIAB = RAW / "pombase/FYPOviability.tsv"

LABEL_RE = re.compile(r"^([^\s(]+)(?:\(([^)]*)\))?(?:\s*-\s*(.*))?$")
# Alleles that enter Dataset S2 (verified empirically in verify_core_data.py): the unlabelled strain,
# DAmP strains, and 'DELEITON' (sic; the only, plain-deletion allele of php5/SPBC3B8.02).
# Exception handled by the data itself: prp19 (DAMP + DELETION_NL) uses only the DAmP strain.
MAIN_ALLELE_SUFFIXES = {"", "DAMP", "DELEITON"}


def read_matrix(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    lt = "\r" if raw.count(b"\n") == 0 else None  # S2 uses classic-Mac CR line endings
    return pd.read_csv(path, sep="\t", index_col=0, lineterminator=lt)


def parse_label(label: str):
    m = LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"unparseable Ryan label: {label!r}")
    sysid, annot, suffix = m.groups()
    return sysid, (annot or ""), (suffix or "").strip()


def pombe_name(annotation: str) -> str:
    """Ryan labels concatenate the S. pombe name (lower case) with S. cerevisiae ortholog names
    (upper case), e.g. 'rps11RPS10B,RPS10A' -> 'rps11'; 'NONE'/'PUT1' -> ''."""
    m = re.match(r"^([a-z][a-z0-9]*)", annotation or "")
    return m.group(1) if m else ""


def allele_class(suffix: str) -> str:
    s = suffix.upper()
    if s == "":
        return "unlabelled"  # deletion for non-essential genes; see docs/data_inventory.md
    if s.startswith("DAMP"):
        return "DAmP"
    if s.startswith("P-MUT"):
        return "point_mutant"
    if s.startswith("DELETION") or s.startswith("DELEITON"):
        return "deletion_variant"
    if s.startswith("KLP"):
        return "KLP_strain"
    if s.startswith("DELTA"):
        return "truncation"
    if s.startswith("COMPLEME"):
        return "complementation"
    return "other_allele(ts?)"


def build_id_map(pb: pd.DataFrame):
    """Map case-insensitive identifiers to current systematic IDs: exact sysID first, then synonyms."""
    exact = {s.upper(): s for s in pb.gene_systematic_id}
    syn = {}
    for sid, syns in zip(pb.gene_systematic_id, pb.synonyms.fillna("")):
        for s in str(syns).split(","):
            s = s.strip().upper()
            if s:
                syn.setdefault(s, set()).add(sid)
    names = {}
    for sid, nm in zip(pb.gene_systematic_id, pb.gene_name.fillna("")):
        if nm:
            names.setdefault(nm.upper(), set()).add(sid)

    def resolve(ryan_id: str, ryan_name: str = ""):
        """ryan_name: S. pombe gene name parsed from the Ryan label annotation (fallback only)."""
        u = ryan_id.upper()
        if u in exact:
            return exact[u], "exact"
        if u in syn and len(syn[u]) == 1:
            return next(iter(syn[u])), "synonym"
        n = ryan_name.upper()
        if n and n in names and len(names[n]) == 1:
            return next(iter(names[n])), "gene_name_from_label"
        if u in syn:
            return None, "ambiguous_synonym:" + "|".join(sorted(syn[u]))
        return None, "unmapped"

    return resolve


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pb = pd.read_csv(PB_GENES, sep="\t", dtype=str)
    viab = pd.read_csv(PB_VIAB, sep="\t", header=None, names=["gene", "viability"], dtype=str)
    viab = viab.drop_duplicates("gene").set_index("gene").viability
    resolve = build_id_map(pb)

    s1 = read_matrix(S1_PATH)
    s2 = read_matrix(S2_PATH)

    q = pd.DataFrame([parse_label(l) for l in s1.index], columns=["ryan_id", "annotation", "suffix"])
    q["query_label"] = list(s1.index)
    q["ryan_id"] = q.ryan_id.str.upper()
    q["allele_class"] = q.suffix.map(allele_class)
    q["main_allele"] = q.suffix.str.upper().isin(MAIN_ALLELE_SUFFIXES)
    arr = pd.DataFrame([parse_label(l) for l in s1.columns], columns=["ryan_id", "annotation", "suffix"])
    arr["ryan_id"] = arr.ryan_id.str.upper()
    assert (arr.suffix == "").all(), "unexpected array allele suffix"
    assert arr.ryan_id.is_unique

    # ---- gene table / ID mapping -------------------------------------------------------------
    ids = sorted(set(q.ryan_id) | set(arr.ryan_id) | set(s2.index.str.upper()) | set(s2.columns.str.upper()))
    label_name = {}
    for rid, ann in list(zip(q.ryan_id, q.annotation)) + list(zip(arr.ryan_id, arr.annotation)):
        if pombe_name(ann):
            label_name.setdefault(rid, pombe_name(ann))
    cur_names = {}
    for sid, nm, syns in zip(pb.gene_systematic_id, pb.gene_name.fillna(""), pb.synonyms.fillna("")):
        cur_names[sid] = {x.strip().upper() for x in [nm] + str(syns).split(",") if x.strip()}
    rows = []
    qa = q.groupby("ryan_id").agg(n_query_alleles=("query_label", "size"),
                                  allele_classes=("allele_class", lambda x: ";".join(sorted(set(x)))),
                                  query_labels=("query_label", lambda x: " || ".join(x)))
    for rid in ids:
        lname = label_name.get(rid, "")
        cur, how = resolve(rid, lname)
        rec = pb.set_index("gene_systematic_id").loc[cur] if cur else None
        rows.append(dict(
            ryan_id=rid, ryan_label_name=lname, pombase_id=cur, mapping=how,
            # Ryan labels drop every '0' from S. pombe names (e.g. 'hsp9' = hsp90), so compare zero-stripped
            label_name_consistent=(None if not (cur and lname) else
                                   lname.upper() in {x.replace("0", "") for x in cur_names.get(cur, set())}),
            gene_name=(rec.gene_name if rec is not None else None),
            gene_type=(rec.gene_type if rec is not None else None),
            viability=(viab.get(cur) if cur else None),
            is_query=rid in qa.index, is_array=rid in set(arr.ryan_id),
            in_s2_rows=rid in set(s2.index.str.upper()),
            n_query_alleles=int(qa.n_query_alleles.get(rid, 0)),
            allele_classes=qa.allele_classes.get(rid, ""),
            query_labels=qa.query_labels.get(rid, ""),
        ))
    genes = pd.DataFrame(rows)
    dup = genes.pombase_id.notna() & genes.pombase_id.duplicated(keep=False)
    genes["merged_with_other_ryan_id"] = dup
    genes.to_csv(OUT / "ryan_genes.tsv", sep="\t", index=False)
    idmap = genes.set_index("ryan_id").pombase_id

    # ---- per-measurement table (S1) ----------------------------------------------------------
    m = s1.copy()
    m.index = pd.RangeIndex(len(m))
    m.columns = arr.ryan_id.values
    long = m.stack().rename("s_raw").reset_index()
    long.columns = ["qrow", "array_ryan_id", "s_raw"]
    long = long.join(q[["ryan_id", "suffix", "allele_class", "main_allele", "query_label"]], on="qrow")
    long = long.rename(columns={"ryan_id": "query_ryan_id"}).drop(columns="qrow")
    long["query_gene"] = long.query_ryan_id.map(idmap)
    long["array_gene"] = long.array_ryan_id.map(idmap)
    long["self_pair"] = long.query_ryan_id == long.array_ryan_id
    long.to_parquet(OUT / "ryan_measurements.parquet", index=False)

    use = long[~long.self_pair & long.query_gene.notna() & long.array_gene.notna()].copy()
    use = use[use.query_gene != use.array_gene]
    use["gene_a"] = np.where(use.query_gene < use.array_gene, use.query_gene, use.array_gene)
    use["gene_b"] = np.where(use.query_gene < use.array_gene, use.array_gene, use.query_gene)
    use["a_is_query"] = use.query_gene == use.gene_a

    keys = ["gene_a", "gene_b"]
    agg_all = use.groupby(keys).agg(s_raw_mean_all=("s_raw", "mean"), n_meas_all=("s_raw", "size"))
    main = use[use.main_allele]
    agg_main = main.groupby(keys).agg(s_raw_mean_meas=("s_raw", "mean"), n_meas=("s_raw", "size"),
                                      s_raw_min=("s_raw", "min"), s_raw_max=("s_raw", "max"))
    ab = main[main.a_is_query].groupby(keys).s_raw.agg(["mean", "size"]).rename(
        columns={"mean": "s_raw_ab", "size": "n_ab"})
    ba = main[~main.a_is_query].groupby(keys).s_raw.agg(["mean", "size"]).rename(
        columns={"mean": "s_raw_ba", "size": "n_ba"})

    # ---- S2 (authors' averaged, symmetric) -----------------------------------------------------
    s2c = s2.copy()
    s2c.index, s2c.columns = s2c.index.str.upper(), s2c.columns.str.upper()
    common = [g for g in s2c.index if g in set(s2c.columns)]
    sub = s2c.loc[common, common].to_numpy(dtype=float)
    both = np.isfinite(sub) & np.isfinite(sub.T)
    print(f"S2 (original IDs): {int(both.sum()) // 2} pairs present in both orientations; "
          f"max |S2[a,b]-S2[b,a]| = {np.abs(sub - sub.T)[both].max():.3g}")
    s2u = s2.copy()
    s2u.index = s2u.index.str.upper().map(idmap)
    s2u.columns = s2u.columns.str.upper().map(idmap)
    s2u = s2u.loc[s2u.index.notna(), s2u.columns.notna()]
    s2l = s2u.stack().rename("s_score").reset_index()
    s2l.columns = ["g1", "g2", "s_score"]
    s2l = s2l[s2l.g1 != s2l.g2]
    s2l["gene_a"] = np.where(s2l.g1 < s2l.g2, s2l.g1, s2l.g2)
    s2l["gene_b"] = np.where(s2l.g1 < s2l.g2, s2l.g2, s2l.g1)
    s2g = s2l.groupby(keys).s_score.agg(["mean", "min", "max", "size"])
    asym = (s2g["max"] - s2g["min"]).max()
    print(f"S2: {len(s2g)} unique pairs after ID mapping; pairs with >1 S2 value (ID merges): "
          f"{int((s2g['size'] > 2).sum())}, max spread {asym:.3g}")
    s2g = s2g[["mean"]].rename(columns={"mean": "s_score"})

    pairs = agg_all.join(agg_main, how="left").join(ab, how="left").join(ba, how="left").join(s2g, how="outer")
    pairs[["n_meas", "n_ab", "n_ba", "n_meas_all"]] = pairs[["n_meas", "n_ab", "n_ba", "n_meas_all"]].fillna(0).astype(int)
    pairs = pairs.reset_index()
    pairs["s_raw_mean"] = pairs[["s_raw_ab", "s_raw_ba"]].mean(axis=1)
    pairs["in_s2"] = pairs.s_score.notna()
    mg = set(genes.loc[genes.merged_with_other_ryan_id, "pombase_id"])
    pairs["involves_merged_id"] = pairs.gene_a.isin(mg) | pairs.gene_b.isin(mg)
    pairs["both_orientations"] = (pairs.n_ab > 0) & (pairs.n_ba > 0)
    gq = genes.dropna(subset=["pombase_id"]).groupby("pombase_id")
    vmap = gq.viability.first()
    pairs["viability_a"] = pairs.gene_a.map(vmap)
    pairs["viability_b"] = pairs.gene_b.map(vmap)
    neg_s2, neg_raw = f"neg_s2_lt_{THR}", f"neg_raw_lt_{THR}"
    pairs[neg_s2] = pairs.s_score < THR
    pairs[neg_raw] = pairs.s_raw_mean < THR
    pairs = pairs[["gene_a", "gene_b", "s_score", "s_raw_mean", "s_raw_ab", "s_raw_ba", "n_meas", "n_ab", "n_ba",
                   "s_raw_mean_meas", "s_raw_min", "s_raw_max", "s_raw_mean_all", "n_meas_all", "in_s2", "both_orientations", "involves_merged_id",
                   "viability_a", "viability_b", neg_s2, neg_raw]]
    pairs.to_parquet(OUT / "ryan_pairs.parquet", index=False)

    # empirical transform raw mean -> S2 (for documentation / threshold conversion)
    ok = pairs.dropna(subset=["s_score", "s_raw_mean"])
    ok = ok[~ok.involves_merged_id]
    qs = np.linspace(0, 1, 401)
    trs = []
    for k, sub in [(1, ok[~ok.both_orientations]), (2, ok[ok.both_orientations])]:
        trs.append(pd.DataFrame({"orientations": k, "quantile": qs, "s_raw_mean": np.quantile(sub.s_raw_mean, qs),
                                 "s_score": np.quantile(sub.s_score, qs)}))
    tr = pd.concat(trs)
    tr.to_csv(OUT / "ryan_s2_transform.tsv", sep="\t", index=False)

    print(f"genes: {len(genes)} Ryan IDs; mapping: {genes.mapping.str.split(':').str[0].value_counts().to_dict()}")
    print(f"measurements: {len(long)}; pairs: {len(pairs)} (in S2: {int(pairs.in_s2.sum())})")
    print(f"wrote {OUT}/ryan_pairs.parquet, ryan_measurements.parquet, ryan_genes.tsv, ryan_s2_transform.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
