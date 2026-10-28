"""(d) Protocol lint (DESIGN §0, §2):
1. The label threshold and the frozen class weights must not appear as numeric literals in src/ or
   scripts/ (config-reading code obtains them from config/protocol.yaml).
2. Full-label / EDA files may only be referenced by the explicitly whitelisted files.
Both checkers are shown to flag deliberately broken source snippets."""
import io
import tokenize
from pathlib import Path

import pytest

from src import protocol

ROOT = protocol.ROOT
SCAN_DIRS = ("src", "scripts")

# Files allowed to contain the threshold literal. Keep this list explicit and short.
THRESHOLD_WHITELIST = {
    "scripts/draw_lockbox.py",      # executed once to draw the lockbox (2026-10-28); frozen historical record
}

# Restricted file names -> files allowed to mention them (data-build / split / eval-owner scripts only).
DATA_BUILD = {"scripts/build_ryan_pairs.py", "scripts/verify_core_data.py", "scripts/eda_core_data.py"}
READ_WHITELIST = {
    "ryan_pairs": DATA_BUILD | {"scripts/draw_lockbox.py", "scripts/make_splits.py",
                                "scripts/compute_class_weights.py", "scripts/toy/make_toy.py"},
    "ryan_measurements": DATA_BUILD | {"scripts/build_universe.py", "scripts/make_splits.py",
                                       "scripts/toy/make_toy.py"},
    "results/eda": {"scripts/eda_core_data.py"},
    "data_inventory": DATA_BUILD,
    "labels_lockbox": {"scripts/make_splits.py", "scripts/lockbox_eval.py"},
    "labels_dev": {"src/data.py", "src/evaluate.py", "scripts/make_splits.py", "scripts/noise_reference.py"},
}


def py_files():
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" not in p.parts:
                yield p.relative_to(ROOT).as_posix(), p.read_text()


def numeric_literals(src: str):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    out = []
    for t in toks:
        if t.type == tokenize.NUMBER:
            try:
                out.append((t.string, float(t.string.replace("_", "")), t.start[0]))
            except ValueError:
                pass
    return out


def literal_violations(src: str) -> list[str]:
    thr = abs(protocol.threshold())
    weights = [w for k in ("unmeasured_universe", "all_pairs") for w in protocol.class_weights(k).values()]
    lits = numeric_literals(src)
    v = [f"line {ln}: threshold literal {s}" for s, x, ln in lits if x == thr]
    for s, x, ln in lits:
        dec = len(s.split(".")[1]) if "." in s else 0
        if dec >= 3 and any(abs(x - w) < 0.5 * 10 ** (-dec) + 1e-12 for w in weights):
            v.append(f"line {ln}: class-weight literal {s}")
    # 2-decimal roundings (e.g. 0.25/0.57/0.18) are common numbers: flag when >= 2 of a weight set co-occur
    two = {round(x, 2) for s, x, _ in lits if "." in s and len(s.split(".")[1]) == 2}
    for k in ("unmeasured_universe", "all_pairs"):
        hits = [w for w in protocol.class_weights(k).values() if round(w, 2) in two]
        if len(hits) >= 2:
            v.append(f"{k} class weights typed as 2-decimal literals")
    return v


def read_violations(path: str, src: str) -> list[str]:
    return [f"{path} mentions {name}" for name, allowed in READ_WHITELIST.items()
            if name in src and path not in allowed]


# ------------------------------------------------------------------------------------------- tests
def test_no_threshold_or_weight_literals():
    bad = {p: v for p, s in py_files() if p not in THRESHOLD_WHITELIST and (v := literal_violations(s))}
    assert not bad, bad


def test_restricted_files_only_in_whitelist():
    bad = [v for p, s in py_files() for v in read_violations(p, s)]
    assert not bad, bad


def test_whitelisted_files_exist_or_are_planned():
    planned = {"scripts/lockbox_eval.py"}
    for f in THRESHOLD_WHITELIST | set().union(*READ_WHITELIST.values()):
        assert (ROOT / f).exists() or f in planned, f


@pytest.mark.parametrize("snippet", [
    "THR = -2.3\n",
    "y = s < -2.30\n",
    "w = {'C1': %.6f}\n" % protocol.class_weights()["C1"],
    "w = [%.2f, %.2f, 0.1]\n" % (protocol.class_weights()["C1"], protocol.class_weights()["C2"]),
])
def test_literal_lint_catches_broken_snippets(snippet):
    assert literal_violations(snippet)


def test_literal_lint_ignores_strings_and_unrelated_numbers():
    assert not literal_violations('label = "S2 < -2.3"\nq = 0.25\nx = 2.31\n')


def test_read_lint_catches_broken_snippets():
    assert read_violations("src/model_x.py", 'pd.read_parquet(ROOT / "data/processed/ryan_pairs.parquet")\n')
    assert read_violations("src/data.py", 'open("splits/labels_lockbox.parquet")\n')
    assert read_violations("scripts/features/f.py", 'open("results/eda/ryan_degree_s2_neg.tsv")\n')
    assert not read_violations("scripts/make_splits.py", '"ryan_pairs.parquet"\n')
