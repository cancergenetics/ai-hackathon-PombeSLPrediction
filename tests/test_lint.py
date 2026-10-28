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


# =============================================================================================
# B3: import-based label-access lint (AST, not substring)
# =============================================================================================
import ast  # noqa: E402
import fnmatch  # noqa: E402

# Only these files may touch src.evaluate's label-returning helpers. Keep explicit and short.
EVAL_HELPER_ALLOW = ("scripts/run_*.py", "scripts/lockbox_eval.py", "src/evaluate.py",
                     "scripts/rarely_seen.py")  # report-only evaluation of saved OOF preds (PI, 2026-10-28)
# Helpers that return (or join) eval labels. The AST test below checks this set covers every function of
# src/evaluate.py that touches the label file, so a newly added label helper cannot slip through.
LABEL_HELPERS = {"prepare", "_labels_for", "_run_frame"}
EVAL_MODULE = "src.evaluate"


def _label_helpers_in_evaluate() -> set:
    """Top-level functions of src/evaluate.py that directly call _labels_for / labels_dev / read labels."""
    tree = ast.parse((ROOT / "src" / "evaluate.py").read_text())
    out = set()
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        if names & {"_labels_for", "labels_dev"}:
            out.add(fn.name)
    return out


def _is_eval_module(mod: str | None, level: int) -> bool:
    mod = mod or ""
    return mod in (EVAL_MODULE, "evaluate") or (level > 0 and mod == "evaluate")


def eval_helper_violations(path: str, src: str) -> list[str]:
    """Imports / attribute accesses of src.evaluate's label helpers outside the allowlist."""
    if any(fnmatch.fnmatch(path, pat) for pat in EVAL_HELPER_ALLOW):
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [f"{path}: unparseable"]
    v, mod_names = [], set()          # names bound to the src.evaluate module
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if _is_eval_module(n.module, n.level):
                for a in n.names:
                    if a.name == "*":
                        v.append(f"{path}:{n.lineno}: star import from src.evaluate")
                    elif a.name in LABEL_HELPERS:
                        v.append(f"{path}:{n.lineno}: imports {a.name}")
            elif n.module in ("src", None):                       # from src import evaluate [as E]
                mod_names |= {a.asname or a.name for a in n.names if a.name == "evaluate"}
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == EVAL_MODULE:
                    if a.asname:
                        mod_names.add(a.asname)
                    else:
                        mod_names.add("src.evaluate")
        elif isinstance(n, ast.Call):
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if fname in ("import_module", "__import__") and any(
                    isinstance(a, ast.Constant) and a.value in (EVAL_MODULE, "evaluate") for a in n.args):
                v.append(f"{path}:{n.lineno}: dynamic import of src.evaluate")
    for _ in range(3):                                            # simple alias propagation: X = E
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in mod_names:
                mod_names |= {t.id for t in n.targets if isinstance(t, ast.Name)}

    def dotted(node) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in LABEL_HELPERS and dotted(n.value) in mod_names:
            v.append(f"{path}:{n.lineno}: accesses evaluate.{n.attr}")
        elif isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr" and len(n.args) >= 2:
            if dotted(n.args[0]) in mod_names and isinstance(n.args[1], ast.Constant) \
                    and n.args[1].value in LABEL_HELPERS:
                v.append(f"{path}:{n.lineno}: getattr(evaluate, {n.args[1].value!r})")
    return v


def test_label_helper_set_covers_evaluate_label_functions():
    found = _label_helpers_in_evaluate()
    assert {"prepare", "_labels_for", "_run_frame"} <= found, found
    assert found <= LABEL_HELPERS | {"evaluate"}, f"new label-touching function(s) in evaluate.py: " \
                                                  f"{found - LABEL_HELPERS - {'evaluate'}}"


def test_only_allowlisted_files_import_label_helpers():
    bad = [x for p, s in py_files() for x in eval_helper_violations(p, s)]
    assert not bad, bad


MODEL_CODE = ("src/ryanfeat.py", "src/stacking.py", "src/pairs.py", "src/models/*.py", "src/features/*.py")


def test_model_code_does_not_access_label_helpers():
    """Explicit check of the model code (a subset of the scan above, kept separate for a clear message).
    A future model file that needs these helpers must be added to EVAL_HELPER_ALLOW deliberately."""
    files = {p: s for p, s in py_files() if any(fnmatch.fnmatch(p, pat) for pat in MODEL_CODE)}
    assert "src/models/lgbm.py" in files
    bad = [x for p, s in files.items() for x in eval_helper_violations(p, s)]
    assert not bad, bad


@pytest.mark.parametrize("snippet", [
    "from src.evaluate import prepare\n",
    "from src.evaluate import evaluate, _labels_for\n",
    "from src.evaluate import *\n",
    "from src import evaluate as E\ndf, _ = E.prepare(p)\n",
    "from src import evaluate\nevaluate.prepare(p)\n",
    "import src.evaluate as ev\nev._labels_for(1)\n",
    "import src.evaluate\nsrc.evaluate.prepare(p)\n",
    "import src.evaluate as ev\nf = ev\nf._run_frame(r)\n",
    "from src import evaluate as E\ng = getattr(E, '_labels_for')\n",
    "import importlib\nm = importlib.import_module('src.evaluate')\n",
    "def f():\n    from src.evaluate import prepare as p\n    return p\n",
    "from .evaluate import prepare\n",
])
def test_eval_helper_lint_catches_broken_snippets(snippet):
    assert eval_helper_violations("src/ryanfeat.py", snippet)
    assert not eval_helper_violations("scripts/run_b0.py", snippet)          # allowlisted
    assert not eval_helper_violations("scripts/lockbox_eval.py", snippet)


@pytest.mark.parametrize("snippet", [
    "from src.evaluate import evaluate, compare, pooled_metrics, class_row_weights\n",
    "from src import evaluate as E\nE.evaluate(p, model='x')\nE.bootstrap_replicates\n",
    "# from src.evaluate import prepare\nx = 'prepare'\n",
    "class A:\n    def prepare(self): ...\nA().prepare()\n",
    "from src import data\ndata.fold_split()\n",
])
def test_eval_helper_lint_ignores_legit_code(snippet):
    assert not eval_helper_violations("src/ryanfeat.py", snippet)
