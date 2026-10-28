#!/usr/bin/env python
"""Tier-1 PPI feature group (DESIGN.md v3 §4, group PPI).

Inputs (DATA.md dataset ids): biogrid (BioGRID 5.0.262 physical, S. pombe + S. cerevisiae), pombase
(curated pombe->cerevisiae orthologs; gene IDs).

Step 1  PMID blocklist.  Re-download BIOGRID-ORGANISM-5.0.262.tab3.zip into a fresh directory, verify its
        sha256 against data/raw/biogrid/SOURCE_ZIP_SHA256.txt, stream ONLY the S. pombe member, keep ONLY
        the publication IDs of rows with Experimental System Type == 'genetic' (+ row counts), write
        <features>/biogrid_genetic_pmids.txt and delete the zip.  No genetic interaction pair is kept.
        Skipped if the PMID file already exists.
Step 2  Physical edges.  S. pombe physical rows -> ID mapper -> drop non protein-protein systems
        (Affinity Capture-RNA, Protein-RNA, Co-localization) -> drop rows whose publication is in the
        blocklist (by PMID, never by pair) -> drop self edges.  Rows/edges removed by each filter are
        reported.  HTP variant: rows with Throughput == 'High Throughput' exactly (rows flagged
        'High Throughput|Low Throughput' are excluded from the HTP network).
        S. cerevisiae physical rows get the same system + blocklist filters and are mapped to pombe
        genes through PomBase curated orthologs.
Step 3  Features (default network and HTP network, suffix _htp):
        gene/PPI.parquet: degree, pagerank (x G, mean 1), degree_htp, pagerank_htp, cer_ortholog_degree
        pair/PPI__edge[_htp], PPI__shared_nbrs[_htp], PPI__jaccard[_htp], PPI__shared_noedge[_htp],
             PPI__sp[_htp] (shortest path 1..4, 5 = longer or disconnected), PPI__edge_cer
        sim/ppi_rwr[_htp].npy, sim/ppi_cer_rwr.npy: random walk with restart (restart prob 0.5),
             symmetric form S_ij = sqrt(R_ij R_ji) = a sqrt(d_i d_j) [(D - (1-a)A)^-1]_ij, in [0,1].
Missingness: degree/pagerank/edge/shared counts are defined for every gene (0 = no recorded
interaction); jaccard, sp and rwr are NaN when either gene has no edge in that network; PPI__edge_cer and
cer_ortholog_degree are NaN when a gene has no curated cerevisiae ortholog.

Run: .venv/bin/python -I scripts/features/build_ppi.py
"""
from __future__ import annotations

import datetime as dt
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src.idmap import IDMapper, log_unmatched  # noqa: E402
from src.orthologs import SOURCE_FILE as ORTH_FILE, incidence, load_orthologs  # noqa: E402

SCRIPT = "scripts/features/build_ppi.py"
VER = "5.0.262"
URL = (f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{VER}/"
       f"BIOGRID-ORGANISM-{VER}.tab3.zip")
RAW = ROOT / "data/raw/biogrid"
P_FILE = RAW / f"biogrid_{VER}_spombe_physical.tab3.tsv.gz"
C_FILE = RAW / f"biogrid_{VER}_scerevisiae_physical.tab3.tsv.gz"
DROP_SYSTEMS = {"Affinity Capture-RNA", "Protein-RNA", "Co-localization"}
RESTART = 0.5
SP_CAP = 4
CHECKS = ROOT / "results/features/ppi_checks.json"
SOURCES = ["biogrid_physical", "pombase_orthologs"]


# ------------------------------------------------------------------------------------------- step 1
def genetic_pmids() -> pd.DataFrame:
    out = fs.feat_dir() / "biogrid_genetic_pmids.txt"
    if out.exists():
        print(f"[skip] {out.name} exists")
        return pd.read_csv(out, sep="\t", comment="#", dtype={"publication": str, "n_genetic_rows": int})
    want = (RAW / "SOURCE_ZIP_SHA256.txt").read_text().split()[0]
    zdir = ROOT / "data/cache/biogrid_refetch"          # dedicated directory, holds only the zip
    zpath = zdir / f"BIOGRID-ORGANISM-{VER}.tab3.zip"
    if not zpath.exists():
        zdir.mkdir(parents=True, exist_ok=True)
        assert not any(p for p in zdir.iterdir() if p.name != zpath.name + ".part"), f"{zdir} not empty"
        tmp = zpath.with_suffix(".zip.part")
        subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-delay", "5", "-A",
                        "Mozilla/5.0 (PombeSL feature builder)", "-o", str(tmp), URL], check=True)
        tmp.rename(zpath)
    got = fs.sha256(zpath)
    if got != want:
        raise RuntimeError(f"BioGRID zip sha256 {got} != recorded {want}; refusing to use it")
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if "Schizosaccharomyces_pombe_972h" in n]
        assert len(members) == 1, members
        member = members[0]
        cols = ["Experimental System Type", "Publication Source"]
        counts: dict[str, int] = {}
        n_rows = n_gen = 0
        with z.open(member) as fh:
            for chunk in pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), sep="\t", usecols=cols,
                                     dtype=str, chunksize=200_000):
                n_rows += len(chunk)
                g = chunk[chunk["Experimental System Type"] == "genetic"]
                n_gen += len(g)
                for pub, n in g["Publication Source"].value_counts().items():
                    counts[pub] = counts.get(pub, 0) + int(n)
    df = pd.DataFrame(sorted(counts.items()), columns=["publication", "n_genetic_rows"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"# BioGRID {VER} ORGANISM tab3 zip sha256 {got} (matches data/raw/biogrid/SOURCE_ZIP_SHA256.txt)\n"
                 f"# member {member}: {n_rows} rows, {n_gen} with Experimental System Type == genetic\n"
                 f"# publications contributing >= 1 S. pombe-member genetic row: {len(df)}; retrieved "
                 f"{dt.date.today().isoformat()} by {SCRIPT}; zip deleted after extraction\n")
        df.to_csv(fh, sep="\t", index=False)
    zpath.unlink()
    shutil.rmtree(zdir)
    print(f"genetic PMIDs: {len(df)} publications, {n_gen} genetic rows (of {n_rows}); zip deleted")
    return df


# ------------------------------------------------------------------------------------------- step 2
def edges_of(df: pd.DataFrame, a="a", b="b") -> set:
    x = df[df[a] != df[b]]
    return set(map(tuple, np.sort(x[[a, b]].values, axis=1)))


def filter_report(df, block: set, label: str, mapped: bool) -> tuple[pd.DataFrame, list]:
    rep = []

    def rec(step, d):
        rep.append({"network": label, "step": step, "rows": int(len(d)),
                    "edges": int(len(edges_of(d)))})
    rec("input (both IDs mapped)" if mapped else "input", df)
    d1 = df[~df["Experimental System"].isin(DROP_SYSTEMS)]
    rec("drop non protein-protein systems", d1)
    d2 = d1[~d1["Publication Source"].isin(block)]
    rec("drop PMID blocklist", d2)
    d3 = d2[d2.a != d2.b]
    rec("drop self edges", d3)
    # marginal effect of the blocklist alone (on the system-filtered, self-free rows)
    base = d1[d1.a != d1.b]
    blk = base[base["Publication Source"].isin(block)]
    rep.append({"network": label, "step": "blocklist marginal: rows removed", "rows": int(len(blk)),
                "edges": int(len(edges_of(base)) - len(edges_of(base[~base["Publication Source"].isin(block)]))),
                "n_publications_hit": int(blk["Publication Source"].nunique()),
                "top_publications": blk["Publication Source"].value_counts().head(10).to_dict(),
                "htp_rows_removed": int((blk.Throughput == "High Throughput").sum())})
    return d3, rep


def adjacency(edges: set, gi: dict, G: int) -> np.ndarray:
    A = np.zeros((G, G), dtype=np.float32)
    if edges:
        e = np.array([(gi[a], gi[b]) for a, b in edges])
        A[e[:, 0], e[:, 1]] = 1
        A[e[:, 1], e[:, 0]] = 1
    np.fill_diagonal(A, 0)
    return A


# ------------------------------------------------------------------------------------------- step 3
def rwr_sim(A: np.ndarray) -> np.ndarray:
    G = A.shape[0]
    deg = A.sum(1)
    nz = np.where(deg > 0)[0]
    S = np.full((G, G), np.nan, dtype=np.float32)
    As = A[np.ix_(nz, nz)].astype(np.float64)
    d = deg[nz].astype(np.float64)
    K = np.linalg.inv(np.diag(d) - (1 - RESTART) * As)
    K = (K + K.T) / 2
    sq = np.sqrt(d)
    Ssub = RESTART * sq[:, None] * K * sq[None, :]
    S[np.ix_(nz, nz)] = np.clip(Ssub, 0, 1).astype(np.float32)
    np.fill_diagonal(S, np.nan)
    return S


def topology(A: np.ndarray, genes: list[str]) -> dict:
    G = A.shape[0]
    deg = A.sum(1)
    iso = deg == 0
    g = nx.Graph()
    g.add_nodes_from(range(G))
    iu = np.argwhere(np.triu(A, 1) > 0)
    g.add_edges_from(map(tuple, iu))
    pr = nx.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500)
    pagerank = np.array([pr[i] for i in range(G)]) * G
    A2 = A @ A
    shared = A2.copy()
    union = deg[:, None] + deg[None, :] - shared
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, shared / union, 0).astype(np.float32)
    jac[iso, :] = np.nan
    jac[:, iso] = np.nan
    noedge = ((shared > 0) & (A == 0)).astype(np.float32)
    # capped shortest path on reachability matrices
    sp = np.full((G, G), SP_CAP + 1, dtype=np.float32)
    reach = A > 0
    sp[reach] = 1
    Rk = reach.astype(np.float32)
    for k in range(2, SP_CAP + 1):
        Rk = ((Rk @ A) > 0).astype(np.float32)
        new = (Rk > 0) & (sp > k)
        sp[new] = k
    sp[iso, :] = np.nan
    sp[:, iso] = np.nan
    return {"degree": deg, "pagerank": pagerank, "edge": A.copy(), "shared_nbrs": shared,
            "jaccard": jac, "shared_noedge": noedge, "sp": sp, "rwr": rwr_sim(A)}


def main() -> int:
    # macOS Accelerate BLAS sets spurious FP-exception flags in float32 matmul (numpy then warns
    # "overflow/divide by zero in matmul" although results are exact); verified against scipy.sparse
    # and networkx shortest paths (tests/features/test_tier1_features.py re-checks on the stored output).
    np.seterr(over="ignore", divide="ignore", invalid="ignore")
    with fs.RuntimeLog("ppi", restart=RESTART, sp_cap=SP_CAP) as rt:
        genes = fs.genes()
        G = len(genes)
        gi = {g: i for i, g in enumerate(genes)}
        mapper = IDMapper()
        pm = genetic_pmids()
        block = set(pm.publication)
        rt.step("pmid_blocklist")

        # ---- S. pombe physical
        p = pd.read_csv(P_FILE, sep="\t", dtype=str)
        assert set(p["Experimental System Type"]) == {"physical"}
        ids = pd.concat([p["Systematic Name Interactor A"], p["Systematic Name Interactor B"]])
        nms = pd.concat([p["Official Symbol Interactor A"], p["Official Symbol Interactor B"]])
        m = mapper.map(ids, source="biogrid_spombe_physical", names=nms)
        p["a"] = p["Systematic Name Interactor A"].map(m)
        p["b"] = p["Systematic Name Interactor B"].map(m)
        n_unmapped_rows = int((p.a.isna() | p.b.isna()).sum())
        p = p.dropna(subset=["a", "b"])
        report = [{"network": "pombe", "step": "raw physical rows (both interactors S. pombe)",
                   "rows": int(n_unmapped_rows + len(p)), "edges": None},
                  {"network": "pombe", "step": "rows dropped: an interactor not mapped to a protein-coding gene",
                   "rows": n_unmapped_rows, "edges": None}]
        pf, rep = filter_report(p, block, "pombe", mapped=True)
        report += rep
        ph = pf[pf.Throughput == "High Throughput"]
        report.append({"network": "pombe_htp", "step": "HTP-only (Throughput == 'High Throughput')",
                       "rows": int(len(ph)), "edges": int(len(edges_of(ph)))})
        nets = {"": adjacency(edges_of(pf), gi, G), "_htp": adjacency(edges_of(ph), gi, G)}
        rt.step("pombe_edges")

        # ---- S. cerevisiae physical, mapped through orthologs
        c = pd.read_csv(C_FILE, sep="\t", dtype=str, usecols=[
            "Systematic Name Interactor A", "Systematic Name Interactor B", "Experimental System",
            "Experimental System Type", "Publication Source", "Throughput"])
        assert set(c["Experimental System Type"]) == {"physical"}
        c = c.rename(columns={"Systematic Name Interactor A": "a", "Systematic Name Interactor B": "b"})
        cf, rep = filter_report(c, block, "cerevisiae", mapped=False)
        report += rep
        orth = load_orthologs(mapper)
        cer_edges = edges_of(cf)
        orfs = sorted({x for e in cer_edges for x in e})
        ci = {o: j for j, o in enumerate(orfs)}
        miss = sorted(set(orth.sc) - set(ci))
        log_unmatched("biogrid_scerevisiae_physical_ortholog_orfs_absent", miss,
                      "curated ortholog ORF has no physical edge in filtered cerevisiae BioGRID")
        C = len(orfs)
        P = np.zeros((C, C), dtype=np.float32)
        e = np.array([(ci[a], ci[b]) for a, b in cer_edges])
        P[e[:, 0], e[:, 1]] = 1
        P[e[:, 1], e[:, 0]] = 1
        np.fill_diagonal(P, 0)
        O = incidence(orth, genes, orfs)
        has_orth = np.zeros(G, bool)
        has_orth[[gi[x] for x in orth.pombe.unique() if x in gi]] = True
        Ecer = ((O @ P @ O.T) > 0).astype(np.float32)
        cer_deg_orf = P.sum(1)
        n_o = O.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cer_ortholog_degree = np.where(n_o > 0, (O @ cer_deg_orf) / n_o, 0.0)
        cer_ortholog_degree[~has_orth] = np.nan       # no curated ortholog at all
        np.fill_diagonal(Ecer, 0)
        rt.step("cerevisiae_edges")

        # ---- features
        sf_p = [f"biogrid:{P_FILE.name} (Experimental System not in {sorted(DROP_SYSTEMS)}; "
                f"publications in biogrid_genetic_pmids.txt removed)",
                "biogrid:BIOGRID-ORGANISM-5.0.262.tab3.zip S. pombe member, 'Publication Source' of genetic rows only (blocklist)"]
        sf_c = [f"biogrid:{C_FILE.name} (same system + publication filters)", ORTH_FILE]
        gene_cols, desc = {}, {}
        checks = {"filters": report, "n_blocklist_publications": len(block)}
        for suf, A in nets.items():
            t = topology(A, genes)
            gene_cols[f"degree{suf}"] = t["degree"]
            gene_cols[f"pagerank{suf}"] = t["pagerank"]
            for name in ("edge", "shared_nbrs", "jaccard", "shared_noedge", "sp"):
                fs.save_matrix("pair", f"PPI__{name}{suf}", t[name], "PPI", SOURCES[:1], SCRIPT,
                               source_files=sf_p, network="pombe" + (suf or "_default"))
            fs.save_matrix("sim", f"ppi_rwr{suf}", t["rwr"], "PPI", SOURCES[:1], SCRIPT,
                           description=f"RWR similarity (restart {RESTART}) on pombe physical network{suf}",
                           source_files=sf_p)
            deg = t["degree"]
            checks[f"network{suf or '_default'}"] = {
                "n_edges": int(A.sum() / 2), "n_genes_with_edge": int((deg > 0).sum()),
                "degree_quantiles": {q: float(np.quantile(deg[deg > 0], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
                "degree_max": float(deg.max()),
                "top_hubs": [genes[i] for i in np.argsort(-deg)[:10]],
                "sp_counts": {str(k): int(v) for k, v in zip(*np.unique(t["sp"][np.triu_indices(G, 1)][
                    np.isfinite(t["sp"][np.triu_indices(G, 1)])], return_counts=True))}}
            del t
            rt.step(f"features{suf or '_default'}")
        gene_cols["cer_ortholog_degree"] = cer_ortholog_degree
        Ecer_out = Ecer.copy()
        Ecer_out[~has_orth, :] = np.nan
        Ecer_out[:, ~has_orth] = np.nan
        fs.save_matrix("pair", "PPI__edge_cer", Ecer_out, "PPI", SOURCES, SCRIPT, source_files=sf_c,
                       description="1 if any curated cerevisiae ortholog pair (x != y) has a filtered physical edge")
        fs.save_matrix("sim", "ppi_cer_rwr", rwr_sim(Ecer), "PPI", SOURCES, SCRIPT, source_files=sf_c,
                       description=f"RWR similarity (restart {RESTART}) on the ortholog-mapped cerevisiae network")
        desc = {"degree": "# distinct physical partners (filtered BioGRID pombe network)",
                "pagerank": "PageRank (alpha 0.85) x G",
                "degree_htp": "degree in HTP-only network", "pagerank_htp": "PageRank in HTP-only network",
                "cer_ortholog_degree": "mean filtered cerevisiae physical degree over curated orthologs"}
        colsrc = {k: (SOURCES if k.startswith("cer_") else ["biogrid_physical"]) for k in gene_cols}
        fs.save_gene_table("PPI", pd.DataFrame(gene_cols, index=genes), SOURCES, SCRIPT,
                           column_sources=colsrc, descriptions=desc, source_files=sf_p + sf_c)
        rt.step("cerevisiae_features")

        # ---- positive control: cerevisiae-ortholog edges enriched among pombe edges
        A = nets[""]
        both = has_orth & (A.sum(1) > 0) & (O.sum(1) > 0)
        idx = np.where(both)[0]
        sub_p = A[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        sub_c = Ecer[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        tab = {"p1c1": int(((sub_p == 1) & (sub_c == 1)).sum()), "p1c0": int(((sub_p == 1) & (sub_c == 0)).sum()),
               "p0c1": int(((sub_p == 0) & (sub_c == 1)).sum()), "p0c0": int(((sub_p == 0) & (sub_c == 0)).sum())}
        orr = (tab["p1c1"] * tab["p0c0"]) / max(1, tab["p1c0"] * tab["p0c1"])
        checks["positive_control_cer_vs_pombe_edges"] = {
            "n_genes": int(len(idx)), **tab, "odds_ratio": orr,
            "P(pombe edge | cer edge)": tab["p1c1"] / max(1, tab["p1c1"] + tab["p0c1"]),
            "P(pombe edge | no cer edge)": tab["p1c0"] / max(1, tab["p1c0"] + tab["p0c0"])}
        checks["n_cer_mapped_edges"] = int(np.triu(Ecer, 1).sum())
        checks["n_genes_with_cer_ortholog"] = int(has_orth.sum())
        CHECKS.parent.mkdir(parents=True, exist_ok=True)
        CHECKS.write_text(json.dumps(checks, indent=1, default=str) + "\n")
        pmid_path = fs.feat_dir() / "biogrid_genetic_pmids.txt"
        fs.update_manifest("biogrid_genetic_pmids.txt", {
            "group": "PPI", "sources": ["biogrid_physical"], "script": SCRIPT, "sha256": fs.sha256(pmid_path),
            "kind": "blocklist", "n_publications": len(block),
            "source_files": ["biogrid:BIOGRID-ORGANISM-5.0.262.tab3.zip (sha256 096a9e4d...), S. pombe member, "
                             "'Publication Source' of rows with Experimental System Type == genetic"]})
        rt.step("checks")
    print(json.dumps(checks["positive_control_cer_vs_pombe_edges"], indent=1))
    for r in report:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
