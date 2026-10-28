#!/usr/bin/env python
"""Verify the core datasets and processed Ryan/Costanzo tables.

Checks (each prints PASS/FAIL; exit code 1 if any FAIL):
  1. sha256 of every downloaded file matches data/raw/core_manifest.tsv
  2. Raw shapes / row counts (Ryan S1, S2; Costanzo files; PomBase files)
  3. Facts stated in Ryan et al. 2012: 953 query alleles of 876 genes; ~1.6 million measurements;
     replicate/marker-swap correlation r ~ 0.4 (Fig. S1A) -> we check the reciprocal-orientation r.
  4. Toy tests of the label parser and allele classifier with known answers.
  5. Processed-table invariants: unique canonical pairs (gene_a < gene_b), S2 symmetric in the raw
     file, and S2 == f_k(raw mean over main alleles) with one monotone map per number of orientations
     k (Collins 2006 pseudo-averaging; positive control), while S2 is NOT reproduced by the mean over
     all alleles (negative control).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_ryan_pairs as brp  # noqa: E402  (our own script, not downloaded code)

RAW = ROOT / "data/raw"
P = ROOT / "data/processed"
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    # 1. checksums
    man = pd.read_csv(RAW / "core_manifest.tsv", sep="\t")
    bad = [r.path for r in man.itertuples() if sha256(RAW / r.path) != r.sha256]
    check("checksums match core_manifest.tsv", not bad, f"{len(man)} files; mismatches: {bad}")

    # 2. raw shapes
    s1 = brp.read_matrix(brp.S1_PATH)
    s2 = brp.read_matrix(brp.S2_PATH)
    check("Ryan S1 shape 953 x 1955", s1.shape == (953, 1955), str(s1.shape))
    check("Ryan S2 shape 862 x 1955", s2.shape == (862, 1955), str(s2.shape))
    cs = pd.read_csv(P / "costanzo_measurements_summary.tsv", sep="\t").set_index("file").rows
    expect = {"SGA_NxN.txt": 11295484, "SGA_ExN.txt": 3881456, "SGA_ExE.txt": 601690, "SGA_DAmP.txt": 1370104}
    check("Costanzo file row counts", all(cs[k] == v for k, v in expect.items()), cs.to_dict())
    pb = pd.read_csv(brp.PB_GENES, sep="\t")
    npc = int((pb.gene_type == "protein coding gene").sum())
    check("PomBase protein-coding gene count plausible (5000-5200)", 5000 <= npc <= 5200, str(npc))

    # 3. paper facts
    q = pd.DataFrame([brp.parse_label(l) for l in s1.index], columns=["id", "ann", "suf"])
    nq = q.id.str.upper().nunique()
    check("953 query alleles of 876 genes (paper)", len(q) == 953 and nq == 876, f"{len(q)} alleles, {nq} genes")
    nmeas = int(np.isfinite(s1.to_numpy(dtype=float)).sum())
    check("~1.6 million pairwise measurements (paper)", 1.5e6 <= nmeas <= 1.7e6, f"{nmeas:,}")
    pairs = pd.read_parquet(P / "ryan_pairs.parquet")
    bo = pairs[pairs.both_orientations]
    r = np.corrcoef(bo.s_raw_ab, bo.s_raw_ba)[0, 1]
    check("reciprocal-orientation correlation ~0.4 (paper Fig. S1A r=0.4)", 0.3 <= r <= 0.5, f"r={r:.3f}")

    # 4. toy tests
    t = brp.parse_label("SPAC458.03(tel2TEL2) - DAMP")
    check("parse_label toy 1", t == ("SPAC458.03", "tel2TEL2", "DAMP"), str(t))
    t = brp.parse_label("SPNCRNA.214 - DAMP")
    check("parse_label toy 2 (no annotation)", t == ("SPNCRNA.214", "", "DAMP"), str(t))
    t = brp.parse_label("SPBC365.16(NONE)")
    check("parse_label toy 3 (no suffix)", t == ("SPBC365.16", "NONE", ""), str(t))
    check("pombe_name toy", [brp.pombe_name(x) for x in ["rps11RPS10B,RPS10A", "NONE", "PUT1", "rev7REV7"]]
          == ["rps11", "", "", "rev7"])
    check("allele_class toy", [brp.allele_class(x) for x in ["", "DAMP", "P-MUT_F164Y", "DELETION-413", "KLP343"]]
          == ["unlabelled", "DAmP", "point_mutant", "deletion_variant", "KLP_strain"])

    # 5. processed invariants
    check("pairs canonical (gene_a < gene_b)", bool((pairs.gene_a < pairs.gene_b).all()))
    check("pairs unique", not pairs.duplicated(["gene_a", "gene_b"]).any())
    S = s2.to_numpy(dtype=float)
    up = s2.index.str.upper()
    cols = list(s2.columns.str.upper())
    common = [g for g in up if g in set(cols)]
    ri = [list(up).index(g) for g in common]
    ci = [cols.index(g) for g in common]
    sub = S[np.ix_(ri, ci)]
    both = np.isfinite(sub) & np.isfinite(sub.T)
    check("S2 symmetric for pairs present in both orientations", np.allclose(sub[both], sub.T[both]),
          f"{int(both.sum()) // 2} pairs")
    ok = pairs[~pairs.involves_merged_id & pairs.in_s2 & pairs.s_raw_mean.notna()]
    for k, sub in [(1, ok[~ok.both_orientations]), (2, ok[ok.both_orientations])]:
        srt = sub.sort_values("s_raw_mean")
        fx, fy = srt.s_raw_mean.values, srt.s_score.values
        viol = np.mean(np.diff(fy) < -1e-6)
        check(f"positive control: S2 is a monotone function of raw mean ({k} orientation(s))", viol < 0.001,
              f"{100 * viol:.3f}% order violations, n={len(srt):,}")
        if k == 1:
            f1x, f1y = fx, fy
            multi = sub[sub.n_meas >= 2]  # several alleles in one orientation
            fm = np.mean(np.abs(np.interp(multi.s_raw_mean.values, fx, fy) - multi.s_score.values) < 0.01)
            check("  ... incl. multi-allele single-orientation pairs (allele averaging)", fm >= 0.99,
                  f"{100 * fm:.2f}% within 0.01 of f_1, n={len(multi):,}")
        else:
            gap = np.interp(-2.3, fy, fx)
            check("f_2 (reciprocal pairs) ~ identity near the cut-off", abs(gap + 2.3) < 0.3,
                  f"S2=-2.3 <-> raw {gap:.2f}")
    one = ok[~ok.both_orientations]
    gap1 = np.interp(-2.3, np.sort(one.s_score.values), np.sort(one.s_raw_mean.values))
    check("f_1 (single orientation) is compressive (pseudo-averaging)", gap1 < -3.0, f"S2=-2.3 <-> raw {gap1:.2f}")
    diff_all = ok[(ok.n_meas_all != ok.n_meas) & ~ok.both_orientations]
    fa = np.mean(np.abs(np.interp(diff_all.s_raw_mean_all.values, f1x, f1y) - diff_all.s_score.values) < 0.01)
    check("negative control: mean over ALL alleles does not reproduce S2", fa < 0.5,
          f"{100 * fa:.1f}% match (n={len(diff_all):,})")
    genes = pd.read_csv(P / "ryan_genes.tsv", sep="\t")
    nunm = int(genes.pombase_id.isna().sum())
    check("ID mapping failures <= 1% of Ryan IDs", nunm <= 0.01 * len(genes), f"{nunm} of {len(genes)}")
    c = pd.read_parquet(P / "costanzo_pairs.parquet", columns=["orf_a", "orf_b"])
    check("Costanzo pairs canonical and unique", bool((c.orf_a < c.orf_b).all()) and not c.duplicated().any(),
          f"{len(c):,} pairs")

    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
