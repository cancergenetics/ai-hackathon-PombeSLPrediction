#!/usr/bin/env python
"""Tier-2 LOC feature group (DESIGN.md v3.1 §4, group LOC).  Label-free.

Source (config/data_allowlist.yaml dataset id) and the ONLY columns read:
  matsuyama2006_localization  ORFeome_Localizome_data.csv (first line is a depositor note; header on line 2),
      columns 'Gene Name' (holds SP* systematic IDs), 'Localization', 'Protein levels (driven by the nmt1
      promoter)'.  Cloning/primer/sequence columns are not read.  PomBase GO CC annotations are NOT used.

Parsing of the 'Localization' grammar
  Segments are separated by ';' (and full-width punctuation); within a segment '>' / '>>' rank compartments
  and '=' / '>=' tie them ("nucleolus>nucleus=cytosol").  Every token is mapped to one canonical compartment
  by the keyword rules in COMPARTMENTS (first match wins; '?' uncertainty marks are kept as calls).  Tokens that
  match no rule (e.g. 'ambiguous structure', 'filamentous structure') count as an observed call with no
  canonical compartment.
  primary set = canonical compartments of the first-ranked (tied) tokens of the first segment.
  Missing (NaN everywhere): NA, 'no expression clone', 'not cloned', 'not determined', 'no transformant'.
  'no apparent signal' (incl. 'almost no apparent signal') = observed, no compartment: one-hot all 0,
  no_signal = 1, pair features NaN (no compartment to compare).

Features
  gene/LOC.parquet
    <compartment> (13 multi-hot columns), no_signal, n_compartments, nmt1_protein_level_log10
    (log10(level + 0.001) of the nmt1-driven YFP fusion level; overexpression, not native abundance)
  pair/LOC__same_primary.npy  1 if the primary sets intersect, 0 otherwise; NaN if either gene has no
                              canonical compartment
  pair/LOC__jaccard.npy       Jaccard of the full compartment sets (same NaN rule)

Run: .venv/bin/python -I scripts/features/build_loc.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src import sim_utils as su  # noqa: E402
from src.idmap import IDMapper  # noqa: E402

SCRIPT = "scripts/features/build_loc.py"
CSV = ROOT / "data/raw/matsuyama2006_localization/ORFeome_Localizome_data.csv"
SRC = "matsuyama2006_localization"
COLS = {"Gene Name": "sysid", "Localization": "loc", "Protein levels (driven by the nmt1 promoter)": "level"}
CHECKS = ROOT / "results/features/loc_checks.json"
# (canonical compartment, regex) - order matters: specific before generic
COMPARTMENTS = [
    ("nucleolus", r"nucleol"),
    ("nuclear_envelope", r"nuclear envelope"),
    ("nuclear_dots", r"nuclear dot"),
    ("spb", r"\bspb\b|mtoc"),
    ("microtubules", r"microtubule|spindle"),
    ("cytoplasmic_dots", r"cytoplasmic dot|large bright dots"),
    ("nucleus", r"nucleus"),
    ("cytosol", r"cytosol"),
    ("mitochondrion", r"mitochondri"),
    ("er", r"^er\b|\ber\?|\ber with"),
    ("golgi", r"golgi"),
    ("vacuole", r"vacuol"),
    ("periphery", r"periphery|cell tip|septum|contractile ring|spore membrane"),
]
MISSING = {"na", "", "no expression clone", "no expression clone (needs check)", "not cloned", "not determined",
           "no transformant"}
NO_SIGNAL = {"no apparent signal", "almost no apparent signal"}


def canon(token: str) -> str | None:
    t = token.strip().lower()
    for name, pat in COMPARTMENTS:
        if re.search(pat, t):
            return name
    return None


def parse(loc) -> tuple[str, set, set, list]:
    """Return (status, all_compartments, primary_compartments, unmatched_tokens)."""
    s = "" if loc is None or (isinstance(loc, float) and np.isnan(loc)) else str(loc)
    s = re.sub(r"[^\x00-\x7f]+", ";", s).strip()            # full-width punctuation -> segment break
    low = s.lower().strip(" ;")
    if low in MISSING:
        return "missing", set(), set(), []
    if low in NO_SIGNAL:
        return "no_signal", set(), set(), []
    segs = [x.strip() for x in s.split(";") if x.strip()]
    allc, prim, unmatched = set(), set(), []
    for k, seg in enumerate(segs):
        ranks = re.split(r">>|>=|>", seg)                     # '>=' counts as a rank break after the first
        for r, grp in enumerate(ranks):
            for tok in grp.split("="):
                if not tok.strip():
                    continue
                c = canon(tok)
                if c is None:
                    unmatched.append(tok.strip().lower())
                    continue
                allc.add(c)
                if k == 0 and r == 0:
                    prim.add(c)
    if not prim and allc:                                   # first token unmatched: fall back to first match
        for seg in segs:
            cs = [canon(t) for t in re.split(r">>|>=|>|=", seg)]
            cs = [c for c in cs if c]
            if cs:
                prim = {cs[0]}
                break
    return "observed", allc, prim, unmatched


def main() -> int:
    np.seterr(over="ignore", divide="ignore", invalid="ignore")  # spurious Accelerate matmul flags
    checks: dict = {}
    with fs.RuntimeLog("loc") as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        txt = CSV.read_bytes().decode("utf-8", errors="replace")
        lines = txt.splitlines()
        assert lines[0].startswith("Provided by the depositor"), lines[0][:60]
        d = pd.read_csv(io.StringIO("\n".join(lines[1:])), dtype=str, usecols=list(COLS)).rename(columns=COLS)
        n_rows = len(d)
        d = d.dropna(subset=["sysid"])
        m = IDMapper().map(d.sysid, source="matsuyama2006")
        d["gene"] = d.sysid.map(m)
        d = d.dropna(subset=["gene"])
        parsed = d["loc"].map(parse)
        d["status"] = parsed.str[0]
        d["allc"] = parsed.str[1]
        d["prim"] = parsed.str[2]
        unmatched = pd.Series([t for p in parsed for t in p[3]]).value_counts()
        # several clones of one gene: keep the observed call(s); union of compartment sets
        d["order"] = d.status.map({"observed": 0, "no_signal": 1, "missing": 2})
        d = d.sort_values(["gene", "order"])
        best = d.groupby("gene").order.transform("min")
        dd = d[d.order == best]
        agg = dd.groupby("gene").agg(status=("status", "first"),
                                     allc=("allc", lambda s: set().union(*s)),
                                     prim=("prim", lambda s: set().union(*s)),
                                     level=("level", lambda s: pd.to_numeric(s, errors="coerce").mean()))
        lvl = d.assign(lv=pd.to_numeric(d.level, errors="coerce")).groupby("gene").lv.mean()
        agg["level"] = lvl.reindex(agg.index)
        agg = agg[agg.index.isin(gi)]
        checks["input"] = {"rows": n_rows, "rows_with_id": int(len(d)), "genes": int(len(agg)),
                           "genes_with_multiple_rows": int(d.gene.duplicated().sum()),
                           "status_counts": agg.status.value_counts().to_dict(),
                           "observed_without_canonical_compartment":
                               int(((agg.status == "observed") & (agg.allc.map(len) == 0)).sum()),
                           "unmatched_tokens_top": unmatched.head(25).to_dict(),
                           "n_unmatched_token_occurrences": int(unmatched.sum())}
        rt.step("parse")

        names = [c for c, _ in COMPARTMENTS]
        obs = agg[agg.status != "missing"]
        gt = pd.DataFrame({c: obs.allc.map(lambda s, c=c: float(c in s)) for c in names}, index=obs.index)
        gt["no_signal"] = (obs.status == "no_signal").astype(float)
        gt["n_compartments"] = obs.allc.map(len).astype(float)
        gt = gt.join(np.log10(agg.level + 0.001).rename("nmt1_protein_level_log10"), how="outer")
        files = [f"{SRC}:ORFeome_Localizome_data.csv cols 'Gene Name', 'Localization', "
                 "'Protein levels (driven by the nmt1 promoter)'"]
        desc = {**{c: f"Matsuyama 2006 YFP localisation call includes {c}" for c in names},
                "no_signal": "'no apparent signal'", "n_compartments": "# canonical compartments called",
                "nmt1_protein_level_log10": "log10(nmt1-driven YFP-fusion protein level + 0.001)"}
        fs.save_gene_table("LOC", gt, [SRC], SCRIPT, descriptions=desc, source_files=files)
        checks["compartment_counts"] = {c: int(gt[c].sum()) for c in names}
        checks["coverage_genes"] = {c: int(gt[c].notna().sum()) for c in ("nucleus", "nmt1_protein_level_log10")}
        rt.step("gene_table")

        comp = obs[obs.allc.map(len) > 0]
        idx = np.array([gi[g] for g in comp.index])
        Ball = np.array([[c in s for c in names] for s in comp.allc], dtype=float)
        Bpri = np.array([[c in s for c in names] for s in comp.prim], dtype=float)
        assert (Bpri.sum(1) > 0).all()
        _, jac = su.incidence_overlap(Ball)
        inter_p, _ = su.incidence_overlap(Bpri)
        same = (inter_p > 0).astype(float)
        np.fill_diagonal(same, np.nan)
        for name, sub, desc_ in (("LOC__same_primary", same, "primary compartment sets intersect"),
                                 ("LOC__jaccard", jac, "Jaccard of compartment sets")):
            M = su.scatter(sub, idx, G)
            fs.save_matrix("pair", name, M, "LOC", [SRC], SCRIPT, source_files=files, description=desc_)
            checks.setdefault("pair_distribution", {})[name] = su.dist_summary(M)
        checks["n_genes_pair_features"] = int(len(idx))
        rt.step("pairs")
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
