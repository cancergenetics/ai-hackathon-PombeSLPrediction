"""Independent re-derivation of the universe (DESIGN §1). No src/ imports."""
import pandas as pd, yaml, numpy as np
R = "/Users/colmclaude/ClaudeExperiments/PombeSL_prediction/"
P = yaml.safe_load(open(R + "config/protocol.yaml"))
pb = pd.read_csv(R + "data/raw/pombase/gene_IDs_names_products.tsv", sep="\t", dtype=str)
fy = pd.read_csv(R + "data/raw/pombase/FYPOviability.tsv", sep="\t", header=None, names=["g", "v"], dtype=str)
assert not fy.g.duplicated().any()
via = dict(zip(fy.g, fy.v))
pc = pb[pb.gene_type == "protein coding gene"].gene_systematic_id
print("protein-coding genes", len(pc), "unique", pc.nunique())
v = pc.map(lambda g: via.get(g, "unknown"))
print(v.value_counts())
base = set(pc[v.isin(["viable", "condition-dependent"])])
print("viable+cond-dep", len(base))

# Ryan strains: S1 raw matrix headers, independently parsed
s1 = R + "data/raw/ryan2012/extracted/mmc4/Dataset S1 - Unaveraged E-MAP with 953 Alleles.txt"
hdr = open(s1).readline().rstrip("\n").split("\t")
rows = pd.read_csv(s1, sep="\t", usecols=[0]).iloc[:, 0].astype(str)
import re
def sysid(l): return re.match(r"^([^\s(]+)", l.strip()).group(1).upper()
def suffix(l):
    m = re.match(r"^([^\s(]+)(?:\(([^)]*)\))?(?:\s*-\s*(.*))?$", l.strip()); return (m.group(3) or "").strip().upper()
arr_ids = {sysid(c) for c in hdr[1:] if c.strip()}
q = pd.DataFrame({"id": rows.map(sysid), "suf": rows.map(suffix)})
print("query alleles", len(q), "suffix counts", q.suf.value_counts().head(12).to_dict())
# ID map: Ryan upper-case id -> PomBase id (case-insensitive match on systematic id + synonyms)
up = {g.upper(): g for g in pb.gene_systematic_id}
syn = {}
for g, s in zip(pb.gene_systematic_id, pb.synonyms.fillna("")):
    for x in s.split(","):
        x = x.strip().upper()
        if x and x not in up: syn.setdefault(x, set()).add(g)
def mp(x):
    if x in up: return up[x]
    s = syn.get(x)
    return next(iter(s)) if s and len(s) == 1 else None
arr = {mp(x) for x in arr_ids} - {None}
qdel = {mp(i) for i, s in zip(q.id, q.suf) if s in ("", "DELEITON")} - {None}
qdamp = {mp(i) for i, s in zip(q.id, q.suf) if s == "DAMP"} - {None}
unk = set(pc[v == "unknown"])
add = unk & (arr | qdel)
dampo = unk & (qdamp - arr - qdel)
print("unknown with Ryan main deletion strain:", len(add))
print("unknown DAmP-only:", sorted(dampo))
mine = base | add
u = pd.read_csv(R + "data/processed/universe_genes.tsv", sep="\t")
built = set(u.gene[u.in_universe])
print("mine", len(mine), "built", len(built), "mine-built", sorted(mine - built), "built-mine", sorted(built - mine))
print("derived list equal:", sorted(add) == sorted(P["universe_derived"]["unknown_with_ryan_deletion_strain"]))
# which of the 22 is 'extra' relative to DESIGN's 21?  report how each qualifies
for g in sorted(add):
    print(" ", g, "array" if g in arr else "", "qdel" if g in qdel else "", "qdamp" if g in qdamp else "")
drop = set(pc[v == "inviable"]) & arr
print("drop_from_training", len(drop), sorted(drop) == sorted(P["universe_derived"]["drop_from_training_genes"]))
# unmapped Ryan ids
print("unmapped array ids", sorted(x for x in arr_ids if mp(x) is None)[:20])
print("unmapped query ids", sorted({i for i in q.id if mp(i) is None})[:20])
