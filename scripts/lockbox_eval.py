#!/usr/bin/env python
"""ONE-SHOT lockbox evaluation (DESIGN.md §2 physical separation, §3 "Lockbox"). The ONLY reader of the lockbox
label file (docs/INTERFACES.md; tests/test_lint.py).

    lockbox_eval.py [--prereg config/lockbox_prereg.yaml] [--i-understand-second-access --reason "..."]

Refuses (exit 2, logged) unless ALL hold:
  * the prereg file is tracked by git, committed and unmodified (git status of that path is empty);
  * it has no TODO left and its required fields are valid (shortlist, headline in shortlist, contrasts);
  * protocol / lockbox-gene / models-file sha256s equal the prereg's; HEAD does not differ from the prereg's
    code_commit in src/, scripts/, config/models.yaml, config/protocol.yaml, config/deliverable.yaml;
    on the real data the code tree is clean (src.evaluate.git_info);
  * results/lockbox/access_log.tsv has no `completed` run for this prereg sha256 (one-shot) — unless
    --i-understand-second-access and --reason are given; the override is logged.
Every invocation appends rows to results/lockbox/access_log.tsv (timestamp, event, git commit, prereg sha256,
user, data root, override, reason, detail): refused | started | completed | failed.

Run: every shortlisted model and baseline (config/models.yaml, prereg seeds) is retrained on ALL dev training
pairs (src.data.dev_pairs) through src/deliverable.py — the code path of the deliverable — and scored once on
the eligible lockbox C1/C2/C3 pairs. Metrics via src/evaluate.pooled_metrics (single "fold"), shared Poisson
gene bootstrap (src/evaluate.bootstrap_replicates) for primary and per-class CIs, paired bootstrap p-values for
the prereg contrasts with Holm adjustment, raw-mean-label sensitivity, dev-minus-lockbox gap.
Outputs: results/lockbox/lockbox_result_<sha12>.json (aggregates only); <data root>/lockbox/preds_<model>.parquet
(scores only; used by predict_all_pairs.py --train-set all as out-of-sample scores of lockbox pairs).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, protocol  # noqa: E402
from src import universe as U  # noqa: E402

LOG_COLUMNS = ["timestamp", "event", "git_commit", "git_dirty", "prereg_path", "prereg_sha256", "user", "data_root",
               "override", "reason", "detail"]
CODE_PATHS = ["src", "scripts", "config/models.yaml", "config/protocol.yaml", "config/deliverable.yaml"]
LOCKBOX_FILE = "labels_lockbox.parquet"
TOY_SKIP = "TOY-ONLY-SKIP"


class Refused(Exception):
    pass


# ============================================================================================ access log
def log_path() -> Path:
    return protocol.results_root() / "lockbox" / "access_log.tsv"


def read_log() -> pd.DataFrame:
    p = log_path()
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def append_log(event: str, prereg: Path | None, sha: str, override=False, reason="", detail="") -> None:
    from src.evaluate import git_info
    gi = git_info()
    row = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"), "event": event,
           "git_commit": gi["git_commit"], "git_dirty": gi["git_dirty"], "prereg_path": str(prereg or ""),
           "prereg_sha256": sha, "user": getpass.getuser(), "data_root": str(protocol.data_root()),
           "override": bool(override), "reason": reason, "detail": detail}
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a") as fh:
        if new:
            fh.write("\t".join(LOG_COLUMNS) + "\n")
        fh.write("\t".join(str(row[c]).replace("\t", " ").replace("\n", " ") for c in LOG_COLUMNS) + "\n")


def completed_runs(sha: str | None = None) -> pd.DataFrame:
    L = read_log()
    m = L.event == "completed"
    if sha is not None:
        m &= L.prereg_sha256 == sha
    return L[m]


# ============================================================================================ lockbox data
def _lockbox_frame() -> pd.DataFrame:
    return pd.read_parquet(protocol.splits_dir() / LOCKBOX_FILE)


def lockbox_eval_pairs() -> pd.DataFrame:
    """Eligible lockbox eval pairs WITH labels (only used inside this script's run)."""
    L = _lockbox_frame()
    L = L[L.eval_eligible.astype(bool) & L.part.isin(["C1", "C2", "C3"])].reset_index(drop=True)
    return L


def unblinded_lockbox_pairs(include_train_drop: bool = False) -> pd.DataFrame:
    """Lockbox pairs with training columns, for the FINAL deliverable retrain (predict_all_pairs.py
    --train-set all). Refuses unless the access log records a completed lockbox evaluation (the lockbox is
    opened only after the one-shot evaluation)."""
    if completed_runs().empty:
        raise PermissionError("the lockbox has not been evaluated yet (no completed run in "
                              f"{log_path()}); --train-set all is only allowed after scripts/lockbox_eval.py")
    L = _lockbox_frame()
    if not include_train_drop:
        L = L[~L.train_drop.astype(bool)]
    return L[data.TRAIN_COLUMNS].reset_index(drop=True)


def lockbox_preds_path(model: str) -> Path:
    return protocol.data_root() / "lockbox" / f"preds_{model}.parquet"


# ============================================================================================ checks
def _git(repo: Path, *a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)


def check_prereg_committed(path: Path) -> None:
    repo = path.resolve().parent
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise Refused(f"prereg {path} is not inside a git repository")
    rel = str(path.resolve().relative_to(Path(top.stdout.strip()).resolve()))
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise Refused(f"prereg {rel} is not committed (untracked)")
    st = _git(repo, "status", "--porcelain", "--", rel).stdout.strip()
    if st:
        raise Refused(f"prereg {rel} has uncommitted modifications: {st}")


def _has_todo(x) -> bool:
    if isinstance(x, dict):
        return any(_has_todo(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_todo(v) for v in x)
    return isinstance(x, str) and "TODO" in x


def validate_prereg(P: dict, real: bool) -> None:
    if _has_todo(P):
        raise Refused("prereg still contains TODO fields (the PI must fill the shortlist etc.)")
    for k in ("protocol_sha256", "lockbox_genes_sha256", "models_file", "models_sha256", "code_commit", "shortlist",
              "headline_model", "baselines", "seeds", "boosting_rounds", "metric", "holm_family"):
        if k not in P:
            raise Refused(f"prereg lacks field {k!r}")
    if not (P["seeds"] == "model" or isinstance(P["seeds"], list) or P["seeds"] in protocol.load()["seeds"]):
        raise Refused(f"prereg seeds {P['seeds']!r}: 'model', a protocol seeds key or a list")
    from src import deliverable as DL
    for mid in list(P["shortlist"]) + list(P["baselines"]):
        try:
            DL.model_spec(mid, ROOT / P["models_file"])
        except KeyError as e:
            raise Refused(f"model {mid!r} is not in {P['models_file']}: {e}")
    if P["headline_model"] not in P["shortlist"]:
        raise Refused("headline_model must be in the shortlist")
    if len(P["shortlist"]) > 5:
        raise Refused("shortlist: one headline + <= 4 others (DESIGN §3)")
    models = set(P["shortlist"]) | set(P["baselines"])
    for a, b in P["holm_family"]["contrasts"]:
        if a not in models or b not in models:
            raise Refused(f"contrast {a} - {b} references a model outside shortlist + baselines")
    if P["protocol_sha256"] != protocol.protocol_sha256():
        raise Refused("protocol.yaml sha256 differs from the prereg")
    if real and P["lockbox_genes_sha256"] != protocol.load()["lockbox"]["genes_sha256"]:
        raise Refused("lockbox genes sha256 differs from the prereg")
    mf = ROOT / P["models_file"]
    if protocol.sha256_file(mf) != P["models_sha256"]:
        raise Refused(f"{P['models_file']} sha256 differs from the prereg models_sha256")
    if P["code_commit"] == TOY_SKIP:
        if real:
            raise Refused(f"code_commit {TOY_SKIP} is only allowed on toy data")
    else:
        if _git(ROOT, "merge-base", "--is-ancestor", P["code_commit"], "HEAD").returncode != 0:
            raise Refused(f"code_commit {P['code_commit']} is not an ancestor of HEAD")
        d = _git(ROOT, "diff", "--name-only", P["code_commit"], "HEAD", "--", *CODE_PATHS).stdout.strip()
        if d:
            raise Refused(f"code differs from code_commit in: {d.splitlines()[:5]}")


def preflight(prereg: Path, override: bool, reason: str) -> tuple[dict, str]:
    """All refusal checks; returns (prereg dict, sha256). Raises Refused."""
    from src.evaluate import git_info
    real = U.is_real_data()
    if not prereg.exists():
        raise Refused(f"prereg {prereg} not found")
    sha = protocol.sha256_file(prereg)
    check_prereg_committed(prereg)
    P = yaml.safe_load(prereg.read_text())
    validate_prereg(P, real)
    if real and git_info()["git_dirty"]:
        raise Refused("code tree is dirty (src/evaluate.git_info); commit before opening the lockbox")
    done = completed_runs(sha)
    if not done.empty:
        if not override:
            raise Refused(f"the lockbox was already evaluated for this prereg (sha256 {sha[:12]}, "
                          f"{len(done)} completed run(s)); one-shot. Use --i-understand-second-access --reason ...")
        if not reason.strip():
            raise Refused("--i-understand-second-access requires a non-empty --reason")
    return P, sha


# ============================================================================================ statistics
def holm(p: dict) -> dict:
    """Holm step-down adjusted p-values {name: p_adj} (monotone)."""
    names = sorted(p, key=lambda k: p[k])
    m = len(names)
    out, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def boot_p(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def class_weightings() -> dict:
    W = {k: v for k, v in protocol.weightings().items() if k != "primary_name"}
    for c in ("C1", "C2", "C3"):
        W[f"class_{c}"] = {x: (1.0 if x == c else 0.0) for x in ("C1", "C2", "C3")}
    return W


# ============================================================================================ run
def resolve_seeds(model_seeds, prereg_seeds) -> list:
    """Seeds of one model in the lockbox run: prereg `seeds` = 'model' keeps the model's own config/models.yaml
    seeds (e.g. M2r_huber = ladder 0-2, M2r_bag5_huber = all 0-4); a protocol seeds key or an explicit list
    overrides every seeded model."""
    if prereg_seeds == "model":
        return [int(s) for s in model_seeds]
    return [int(s) for s in (protocol.load()["seeds"][prereg_seeds] if isinstance(prereg_seeds, str) else prereg_seeds)]


def model_spec_for(mid: str, P: dict) -> dict:
    """Resolved config/models.yaml spec of a shortlisted model / baseline with the prereg seed rule applied."""
    from src import deliverable as DL
    spec = DL.model_spec(mid, ROOT / P["models_file"])
    if "seeds" in spec:
        spec["seeds"] = resolve_seeds(spec["seeds"], P["seeds"])
    return spec


def run(P: dict, sha: str, threads: int, n_boot: int | None, log=print) -> dict:
    from src import deliverable as DL
    from src.evaluate import _ci, bootstrap_counts, bootstrap_replicates, pooled_metrics
    from src import pairs as PA
    t_start = time.perf_counter()
    lfspec = PA.load_spec()
    dev = data.dev_pairs()
    ev = lockbox_eval_pairs()
    genes, _ = U.universe_genes(lfspec.genes)
    seeds = P["seeds"]
    models = list(dict.fromkeys(list(P["shortlist"]) + list(P["baselines"])))
    scores, fits = {}, {}
    q = ev[["gene_a", "gene_b"]]
    for mid in models:
        spec = model_spec_for(mid, P)
        t0 = time.perf_counter()
        m = DL.TrainedModel(spec, dev, threads=threads, lfspec=lfspec, cover_genes=genes, log=log).fit()
        s = m.predict(q)
        assert np.isfinite(s).all(), f"{mid}: non-finite lockbox scores"
        scores[mid] = s
        fits[mid] = {**m.info, "timings": m.timings, "wall_s": round(time.perf_counter() - t0, 1)}
        out = lockbox_preds_path(mid)
        out.parent.mkdir(parents=True, exist_ok=True)
        q.assign(part=ev.part.values, score=s).to_parquet(out, index=False)
        log(f"[{mid}] fitted + scored {len(q)} lockbox pairs in {fits[mid]['wall_s']} s")
    base = ev[["gene_a", "gene_b", "part", "y"]].assign(fold=0)
    for c in ("both_orientations", "viable_viable", "damp_mixed"):
        if c in ev:
            base[c] = ev[c].values
    counts, gidx = bootstrap_counts(np.r_[ev.gene_a.to_numpy(), ev.gene_b.to_numpy()], n_boot)
    W = class_weightings()
    lvl = protocol.load()["bootstrap"]["ci_level"]
    res, reps = {}, {}
    for mid in models:
        df = base.assign(score=scores[mid])
        met = pooled_metrics(df)["mean"]
        raw = pooled_metrics(df.assign(y=ev.y_raw.astype(bool).values))["mean"]["pooled"] if "y_raw" in ev else None
        r = bootstrap_replicates(df, counts, gidx, weightings=W)
        reps[mid] = r
        res[mid] = {"primary": met["pooled"], "primary_ci": list(_ci(r["primary"], lvl)),
                    "primary_se": float(np.std(r["primary"], ddof=1)),
                    **{f"ap_{c}": met[f"ap_{c}"] for c in ("C1", "C2", "C3")},
                    **{f"ap_{c}_ci": list(_ci(r[f"class_{c}"], lvl)) for c in ("C1", "C2", "C3")},
                    "pooled_all_pairs": met["pooled_all_pairs"], "pooled_equal": met["pooled_equal"],
                    "auroc": met["auroc"], "precision_at_k": met["precision_at_k"],
                    "prevalence_primary": met["prevalence_primary"], "pooled_raw_mean_label": raw,
                    **{f"n_{c}": met[f"n_{c}"] for c in ("C1", "C2", "C3")},
                    **{f"pos_{c}": met[f"pos_{c}"] for c in ("C1", "C2", "C3")}}
        spec = DL.model_spec(mid, ROOT / P["models_file"])
        dev_run = DL.dev_run_id(spec)
        if dev_run:
            from src.evaluate import load_run
            dp = load_run(dev_run)["primary"]
            res[mid].update({"dev_run": dev_run, "dev_primary": dp, "dev_minus_lockbox": dp - met["pooled"]})
    p_raw, contrasts = {}, {}
    for a, b in P["holm_family"]["contrasts"]:
        k = f"{a} - {b}"
        d = reps[a]["primary"] - reps[b]["primary"]
        p_raw[k] = boot_p(d)
        contrasts[k] = {"delta": res[a]["primary"] - res[b]["primary"], "ci": list(_ci(d, lvl)), "p": p_raw[k]}
    padj = holm(p_raw) if p_raw else {}
    alpha = float(P["holm_family"].get("alpha", 0.05))
    for k in contrasts:
        contrasts[k]["p_holm"] = padj[k]
        contrasts[k]["reject_at_alpha"] = bool(padj[k] < alpha)
    ranking = sorted(models, key=lambda m: -res[m]["primary"])
    return {"prereg_sha256": sha, "headline_model": P["headline_model"], "shipped_model": P["headline_model"],
            "note": "the headline model is shipped regardless of the lockbox ranking (prereg)",
            "ranking_by_primary": ranking, "models": res, "contrasts": contrasts, "holm_alpha": alpha,
            "n_eval_pairs": int(len(ev)), "n_eval_genes": int(len(gidx)), "n_boot": int(counts.shape[0]),
            "n_dev_training_pairs": int(len(dev)), "seeds": seeds, "fits": fits,
            "wall_s": round(time.perf_counter() - t_start, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one-shot lockbox evaluation")
    ap.add_argument("--prereg", default=str(ROOT / "config" / "lockbox_prereg.yaml"))
    ap.add_argument("--i-understand-second-access", dest="override", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None, help="toy/testing only (real data: protocol value)")
    a = ap.parse_args(argv)
    from src.timing import peak_rss_mb, set_threads
    prereg = Path(a.prereg)
    threads = a.threads or U.deliverable_cfg()["scoring"]["threads"]
    set_threads(threads)
    sha = protocol.sha256_file(prereg) if prereg.exists() else ""
    try:
        if a.n_boot is not None and U.is_real_data():
            raise Refused("--n-boot is for toy/testing only")
        P, sha = preflight(prereg, a.override, a.reason)
    except Refused as e:
        append_log("refused", prereg, sha, a.override, a.reason, str(e))
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    append_log("started", prereg, sha, a.override, a.reason,
               "second access (override)" if a.override and not completed_runs(sha).empty else "")
    try:
        out = run(P, sha, threads, a.n_boot)
    except Exception as e:
        append_log("failed", prereg, sha, a.override, a.reason, f"{type(e).__name__}: {e}"[:500])
        raise
    out["peak_rss_mb"] = round(peak_rss_mb(), 1)
    from src.evaluate import git_info
    out.update({"git": git_info(), "models_sha256": protocol.sha256_file(ROOT / P["models_file"]),
                "protocol_sha256": protocol.protocol_sha256(), "data_root": str(protocol.data_root()),
                "override": bool(a.override), "reason": a.reason})
    d = protocol.results_root() / "lockbox"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"lockbox_result_{sha[:12]}{'_access' + str(len(completed_runs(sha)) + 1) if a.override else ''}.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    append_log("completed", prereg, sha, a.override, a.reason, f"result {f.name}")
    print(json.dumps({m: {k: v for k, v in r.items() if k in ("primary", "primary_ci", "ap_C1", "ap_C2", "ap_C3")}
                      for m, r in out["models"].items()}, indent=1, default=float))
    print(json.dumps(out["contrasts"], indent=1, default=float))
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
