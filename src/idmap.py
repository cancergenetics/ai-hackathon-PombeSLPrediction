"""One shared S. pombe gene-ID mapper for every feature source (DESIGN §4 rule 5).

Resolution order (case-insensitive, first rule that gives a unique answer wins):
  1. exact current PomBase systematic ID (a trailing transcript/peptide suffix such as ".1" or ".1:pep"
     is stripped first);
  2. PomBase synonym (the `synonyms` column of gene_IDs_names_products.tsv holds old/obsolete systematic
     IDs and old gene names) -- only if the synonym points to exactly one current gene;
  3. current PomBase gene name -- only if unique (and only for sources that supply a name column, or when
     the input ID itself is a name).
Anything else is `unmapped` or `ambiguous` and is dropped (never guessed).  Mapped IDs that are not
protein-coding are reported as `not_protein_coding` and dropped from the feature universe.

Every non-exact outcome (synonym/name rescues, unmapped, ambiguous, non-coding) is written to
results/idmap/<source>.tsv, and one summary row per source to results/idmap/summary.tsv, so that no
ID is lost silently.

Reference: PomBase release pombase-2026-10-01, misc/gene_IDs_names_products.tsv (data/raw/pombase/).
"""
from __future__ import annotations

import csv
import fcntl
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PB_GENES = ROOT / "data/raw/pombase/gene_IDs_names_products.tsv"
LOG_DIR = ROOT / "results/idmap"

_SUFFIX = re.compile(r"(\.\d+)?(:pep)?$", re.IGNORECASE)


class IDMapper:
    def __init__(self, pombase_genes: Path | str = PB_GENES, log_dir: Path | str | None = LOG_DIR):
        pb = pd.read_csv(pombase_genes, sep="\t", dtype=str, keep_default_na=False)
        self.pb = pb
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.protein_coding = sorted(pb.loc[pb.gene_type == "protein coding gene", "gene_systematic_id"])
        self._pc = set(self.protein_coding)
        self.exact = {s.upper(): s for s in pb.gene_systematic_id}
        self.syn: dict[str, set[str]] = {}
        for sid, syns in zip(pb.gene_systematic_id, pb.synonyms):
            for s in str(syns).split(","):
                s = s.strip().upper()
                if s:
                    self.syn.setdefault(s, set()).add(sid)
        self.names: dict[str, set[str]] = {}
        for sid, nm in zip(pb.gene_systematic_id, pb.gene_name):
            for n in str(nm).split(","):          # a few rows hold "abp1,clo1"
                n = n.strip().upper()
                if n:
                    self.names.setdefault(n, set()).add(sid)

    # ------------------------------------------------------------------------------------------
    def resolve(self, raw_id: str, name: str = "") -> tuple[str | None, str]:
        """Return (current systematic ID or None, method/status)."""
        rid = "" if raw_id is None or (isinstance(raw_id, float)) else str(raw_id).strip()
        u = rid.upper()
        if u in self.exact:
            return self.exact[u], "exact"
        # transcript / peptide suffix ("SPAC1002.01.1", "SPAC1002.01.1:pep"): strip only if what remains
        # is an exact current ID (systematic IDs themselves contain dots, e.g. SPAC1002.01)
        stripped = _SUFFIX.sub("", u)
        if stripped != u and stripped in self.exact:
            return self.exact[stripped], "exact"
        if u in self.syn:
            hits = self.syn[u]
            if len(hits) == 1:
                return next(iter(hits)), "synonym"
        # gene-name rescue: the ID itself may be a gene name, or the source gives a name column
        for cand, how in ((u, "name_as_id"), (str(name or "").strip().upper(), "gene_name")):
            if cand and cand in self.names and len(self.names[cand]) == 1:
                return next(iter(self.names[cand])), how
        if u in self.syn:
            return None, "ambiguous_synonym:" + "|".join(sorted(self.syn[u]))
        for cand in (u, str(name or "").strip().upper()):
            if cand and cand in self.names:
                return None, "ambiguous_name:" + "|".join(sorted(self.names[cand]))
        return None, "unmapped"

    def map(self, ids, source: str, names=None, protein_coding_only: bool = True) -> pd.Series:
        """Map an iterable of IDs; returns a Series indexed by the *input* IDs (unique, input order)
        with the current systematic ID or None.  Logs non-exact outcomes to results/idmap/<source>.tsv."""
        ids = list(ids)
        names = list(names) if names is not None else [""] * len(ids)
        seen, out, log = set(), {}, []
        counts = {"n_input": 0, "exact": 0, "synonym": 0, "gene_name": 0, "name_as_id": 0,
                  "unmapped": 0, "ambiguous": 0, "not_protein_coding": 0}
        for rid, nm in zip(ids, names):
            key = rid
            if key in seen:
                continue
            seen.add(key)
            counts["n_input"] += 1
            sid, how = self.resolve(rid, nm)
            status = how
            if sid is not None and protein_coding_only and sid not in self._pc:
                status = "not_protein_coding"
                sid_logged, sid = sid, None
            else:
                sid_logged = sid
            if status.startswith("ambiguous"):
                counts["ambiguous"] += 1
            else:
                counts[status] = counts.get(status, 0) + 1
            if status != "exact":
                log.append((rid, nm or "", sid_logged or "", status))
            out[key] = sid
        res = pd.Series(out, dtype=object, name="pombase_id")
        # many-to-one collisions (two input IDs -> same current gene) are reported, not dropped
        mapped = res.dropna()
        dup = mapped[mapped.duplicated(keep=False)]
        counts["many_to_one_inputs"] = int(len(dup))
        for rid, sid in dup.items():
            log.append((rid, "", sid, "many_to_one"))
        counts["n_mapped"] = int(res.notna().sum())
        self._write_log(source, log, counts)
        return res

    # ------------------------------------------------------------------------------------------
    def _write_log(self, source: str, log, counts: dict) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / f"{source}.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["input_id", "input_name", "pombase_id", "status"])
            for row in sorted(log, key=lambda r: (r[3], str(r[0]))):
                w.writerow(row)
        summ = self.log_dir / "summary.tsv"
        cols = ["source", "n_input", "n_mapped", "exact", "synonym", "gene_name", "name_as_id",
                "unmapped", "ambiguous", "not_protein_coding", "many_to_one_inputs"]
        with open(summ, "a+") as fh:                      # read-modify-write under a lock
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            txt = fh.read()
            rows = [r.split("\t") for r in txt.strip().splitlines()[1:]] if txt.strip() else []
            rows = [r for r in rows if r and r[0] != source]
            rows.append([source] + [str(counts.get(c, 0)) for c in cols[1:]])
            fh.seek(0)
            fh.truncate()
            fh.write("\t".join(cols) + "\n")
            for r in sorted(rows):
                fh.write("\t".join(r) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)


def log_unmatched(source: str, ids, reason: str, log_dir: Path | str = LOG_DIR) -> None:
    """Log IDs of a non-pombe namespace (e.g. S. cerevisiae ORFs) that could not be matched in a source."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(map(str, ids)))
    with open(log_dir / f"{source}.tsv", "w") as fh:
        fh.write("id\treason\n")
        for i in ids:
            fh.write(f"{i}\t{reason}\n")


def write_genes_txt(out: Path | str, mapper: IDMapper | None = None) -> list[str]:
    mapper = mapper or IDMapper(log_dir=None)
    genes = sorted(mapper.protein_coding)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(genes) + "\n")
    return genes
