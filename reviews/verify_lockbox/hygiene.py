"""Lockbox eval-set and training-set hygiene + prereg provenance (verifier code). Aggregates only."""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
P = yaml.safe_load(open(ROOT / "config/protocol.yaml"))
THR = P["label"]["threshold"]
R = json.load(open(ROOT / "results/lockbox/lockbox_result_deec74e31a7c.json"))
out = {}
key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a)

LB = set((ROOT / "config/lockbox_genes.txt").read_text().split())
out["n_lockbox_genes"] = len(LB)
out["lockbox_genes_sha256"] = hashlib.sha256((ROOT / "config/lockbox_genes.txt").read_bytes()).hexdigest()
U = pd.read_csv(ROOT / "data/processed/universe_genes.tsv", sep="\t")
univ = set(U.gene[U.in_universe])

# ---- source of truth: raw Ryan S2 pairs
RP = pd.read_parquet(ROOT / "data/processed/ryan_pairs.parquet",
                     columns=["gene_a", "gene_b", "s_score", "in_s2", "involves_merged_id"])
RP = RP[RP.in_s2].reset_index(drop=True)
RP["k"] = key(RP.gene_a.values, RP.gene_b.values)
RP["nlb"] = RP.gene_a.isin(LB).astype(int) + RP.gene_b.isin(LB).astype(int)

L = pd.read_parquet(ROOT / "data/processed/splits/labels_lockbox.parquet")
L["k"] = key(L.gene_a.values, L.gene_b.values)
L["nlb"] = L.gene_a.isin(LB).astype(int) + L.gene_b.isin(LB).astype(int)
D = pd.read_parquet(ROOT / "data/processed/splits/labels_dev.parquet")
D["k"] = key(D.gene_a.values, D.gene_b.values)

# labels file vs raw
m = RP.set_index("k").s_score
out["lockbox_s_score_matches_raw"] = bool(np.allclose(L.s_score.values, m.loc[L.k].values, equal_nan=True))
out["lockbox_y_eq_recomputed"] = bool((L.y.values == (L.s_score.values < THR)).all())
# part recomputed from lockbox gene count
exp_part = np.where(L.nlb == 2, "C3", np.where(L.nlb == 1, "C2", "C1"))
out["part_mismatch"] = int((exp_part != L.part.values).sum())
out["raw_pairs_with_lockbox_gene"] = int((RP.nlb > 0).sum())
out["lockbox_file_C2C3"] = int(L.part.isin(["C2", "C3"]).sum())
out["raw_lockbox_gene_pairs_missing_from_lockbox_file"] = int((~RP.k[RP.nlb > 0].isin(set(L.k))).sum())
# eligibility recomputed: both in universe, not DAmP-only, not merged-ID
inu = L.gene_a.isin(univ) & L.gene_b.isin(univ)
out["in_universe_mismatch"] = int((inu.values != L.in_universe.values).sum())
elig = inu & ~L.damp_only & ~L.involves_merged_id
out["eligible_mismatch"] = int((elig.values != L.eval_eligible.values).sum())
exp_eval = set(L.k[elig & L.part.isin(["C1", "C2", "C3"])])
pr = pd.read_parquet(ROOT / "data/processed/lockbox/preds_M2r_huber.parquet")
pr["k"] = key(pr.gene_a.values, pr.gene_b.values)
out["preds_n"] = len(pr)
out["expected_eval_n"] = len(exp_eval)
out["preds_set_equal_expected"] = set(pr.k) == exp_eval
partmap = dict(zip(L.k, exp_part))
out["preds_part_mismatch"] = int((pr.k.map(partmap).values != pr.part.values).sum())
ev = L.set_index("k").loc[pr.k]
out["eval_damp_only"] = int(ev.damp_only.sum()); out["eval_merged"] = int(ev.involves_merged_id.sum())
out["eval_not_in_universe"] = int((~ev.in_universe).sum())
out["eval_by_class"] = pr.part.value_counts().to_dict()
out["excluded_ineligible_by_class"] = L[~elig].part.value_counts().to_dict()
out["excluded_damp_only_by_class"] = L[L.damp_only].part.value_counts().to_dict()
c1 = L[L.part == "C1"]
out["C1_with_lockbox_gene"] = int((c1.nlb > 0).sum())
out["C1_fraction_of_eligible_devdev"] = len(c1) / (len(c1) + int(D.eval_eligible.sum()))
# raw S2 dev x dev = dev file + lockbox C1, disjoint
devdev = set(RP.k[RP.nlb == 0])
out["devdev_raw"] = len(devdev); out["dev_file"] = len(D); out["lockbox_C1"] = len(c1)
out["devdev_eq_dev_union_C1"] = devdev == (set(D.k) | set(c1.k))
out["dev_file_intersect_C1"] = len(set(D.k) & set(c1.k))

# ---- training set: src.data.dev_pairs (what lockbox_eval.run passes to TrainedModel)
from src import data
T = data.dev_pairs()
Tk = set(key(T.gene_a.values, T.gene_b.values))
out["train_n"] = len(T); out["train_pos"] = int(T.y.sum())
out["train_genes"] = int(len(set(T.gene_a) | set(T.gene_b)))
out["train_lockbox_genes"] = len((set(T.gene_a) | set(T.gene_b)) & LB)
out["train_intersect_lockboxC1"] = len(Tk & set(c1.k))
out["train_intersect_any_lockbox_pair"] = len(Tk & set(L.k))
out["train_intersect_eval_pairs"] = len(Tk & exp_eval)
out["train_y_eq_recomputed"] = bool((T.y.values == (T.s_score.values < THR)).all())

def thash(T):   # same formula as src/deliverable.train_hash (re-implemented)
    cols = [c for c in ("gene_a", "gene_b", "y", "s_score", "hypomorph_a", "hypomorph_b", "eval_eligible") if c in T]
    return hashlib.sha256(pd.util.hash_pandas_object(T[cols], index=False).values.tobytes()).hexdigest()
pc = set((ROOT / "data/processed/features/genes.txt").read_text().split())
keep = T.gene_a.isin(pc).values & T.gene_b.isin(pc).values
Tn = T[keep].reset_index(drop=True)
out["hash_all_dev"] = thash(T); out["hash_all_dev_coding"] = thash(Tn)
out["n_coding"] = len(Tn); out["pos_coding"] = int(Tn.y.sum()); out["genes_coding"] = len(set(Tn.gene_a) | set(Tn.gene_b))
F = R["fits"]
out["fits_check"] = {mid: {"hash_match": f["train_hash"] == (out["hash_all_dev_coding"] if f["noncoding_dropped"] else out["hash_all_dev"]),
                           "n_match": f["n_train_pairs"] == (len(Tn) if f["noncoding_dropped"] else len(T)),
                           "seeds": [x["seed"] for x in f["fits"]]} for mid, f in F.items()}
out["n_dev_training_pairs_match"] = R["n_dev_training_pairs"] == len(T)

# ---- provenance
git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout
pre = yaml.safe_load(open(ROOT / "config/lockbox_prereg.yaml"))
out["prereg_sha256"] = hashlib.sha256((ROOT / "config/lockbox_prereg.yaml").read_bytes()).hexdigest()
run_commit = R["git"]["git_commit"]
out["prereg_sha_at_run_commit"] = hashlib.sha256(git("show", f"{run_commit}:config/lockbox_prereg.yaml").encode()).hexdigest()
out["code_diff_codecommit_vs_run"] = git("diff", "--name-only", pre["code_commit"], run_commit, "--", "src", "scripts",
                                         "config/models.yaml", "config/protocol.yaml", "config/deliverable.yaml").split()
out["code_diff_run_vs_HEAD"] = git("diff", "--name-only", run_commit, "HEAD", "--", "src", "scripts", "config").split()
out["models_sha_at_code_commit"] = hashlib.sha256(git("show", f"{pre['code_commit']}:config/models.yaml").encode()).hexdigest()
out["models_sha_prereg"] = pre["models_sha256"]; out["models_sha_result"] = R["models_sha256"]
out["protocol_sha_ok"] = R["protocol_sha256"] == pre["protocol_sha256"]
M = yaml.safe_load(git("show", f"{pre['code_commit']}:config/models.yaml"))
out["models_yaml_seeds"] = {k: (v.get("seeds") if isinstance(v, dict) else None) for k, v in (M.get("models", M)).items()}
out["threads"] = R["threads"]
log = pd.read_csv(ROOT / "results/lockbox/access_log.tsv", sep="\t", dtype=str, keep_default_na=False)
out["access_log_events"] = log[["event", "git_commit", "prereg_sha256"]].assign(
    git_commit=lambda d: d.git_commit.str[:8], prereg_sha256=lambda d: d.prereg_sha256.str[:12]).values.tolist()
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(OUT / "hygiene_result.json", "w"), indent=1, default=str)
