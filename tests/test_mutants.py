"""Mutation check of the test suite itself (verify_phaseA gap 7: 7/11 planted mutants were caught).

Each planted mutant (tests/mutation_plugin.py, in-process monkeypatches of src/, or generated mutant copies
of scripts/make_splits.py) is applied in a subprocess that runs the rest of the suite; the suite must FAIL
(pytest exit code 1 = tests failed; any other non-zero code means the plugin itself broke and does not count).
Set PSL_SKIP_MUTANTS=1 to skip (slow: runs the suite once per mutant, 4 in parallel)."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MUTANTS = ["evict_c3", "c1_in_train", "uniform_weights", "boot_unshared", "inner_keep_H", "stacker_no_T",
           "eval_hypomorph_true", "primary_is_allpairs", "c2_partner_unseen", "ms_no_damp", "ms_elig_ignores_damp"]

# Test files exercising the protocol-critical code (this file and the plugin are excluded: no recursion)
SUITE = ["test_bootstrap.py", "test_evaluate.py", "test_inner.py", "test_lint.py", "test_metrics.py",
         "test_splits.py"]

# (marker in scripts/make_splits.py, replacement) for the make_splits mutants
MS_EDITS = {
    "ms_no_damp": ('damp = (m.allele_class == "DAmP").to_numpy()',
                   'damp = (m.allele_class == "DAmP").to_numpy() & False'),
    "ms_elig_ignores_damp": ("elig &= ~pairs.damp_only", "pass"),
}
ROOT_LINE = "ROOT = Path(__file__).resolve().parents[1]"

pytestmark = pytest.mark.skipif(bool(os.environ.get("PSL_SKIP_MUTANTS") or os.environ.get("PSL_MUTANT")),
                                reason="mutation run skipped (PSL_SKIP_MUTANTS / already inside a mutant run)")


def _make_ms_mutants(d: Path) -> None:
    src = (ROOT / "scripts" / "make_splits.py").read_text()
    assert ROOT_LINE in src
    src = src.replace(ROOT_LINE, f"ROOT = Path({str(ROOT)!r})")
    for name, (old, new) in MS_EDITS.items():
        assert src.count(old) == 1, f"make_splits.py changed shape: cannot build mutant {name}"
        (d / f"make_splits_{name[3:]}.py").write_text(src.replace(old, new))


def _run(name: str, d: Path):
    env = dict(os.environ, PSL_MUTANT=name, PSL_MUTANT_DIR=str(d),
               PYTHONPATH=os.pathsep.join([str(HERE), str(ROOT), os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", "pytest", "-p", "mutation_plugin", "-x", "-q", "-p", "no:cacheprovider",
           *[str(HERE / f) for f in SUITE]]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    d = tmp_path_factory.mktemp("mutants")
    _make_ms_mutants(d)
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(zip(MUTANTS, ex.map(lambda n: _run(n, d), MUTANTS)))


def test_mutant_list_matches_verifier():
    assert len(MUTANTS) == 11 and len(set(MUTANTS)) == 11
    src = (HERE / "mutation_plugin.py").read_text()
    for m in MUTANTS:
        assert f'"{m}"' in src or (m.startswith("ms_") and m in MS_EDITS), m


@pytest.mark.parametrize("name", MUTANTS)
def test_mutant_is_caught(name, outcomes):
    code, out = outcomes[name]
    assert code == 1, f"mutant {name}: pytest exit code {code} (0 = NOT CAUGHT, other = harness error)\n{out[-1500:]}"
    failed = [ln for ln in out.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    assert failed, out[-1500:]
    print(f"{name}: caught by {failed[0]}")
