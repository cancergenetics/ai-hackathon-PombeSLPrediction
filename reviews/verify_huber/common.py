"""Shared setup for the independent Huber-stacker verification (reviewer scripts; never imported by src/).
PRIVATE data root = symlinks to every entry of data/processed (the store the Huber runs used, via the m2plus
worktree) EXCEPT cache/ and preds/, which are private and empty, so nothing here can read or poison the real
feature cache; PRIVATE results root (no registry writes). Threads capped at 2 (shared machine)."""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/private/tmp/claude-503/-Users-colmclaude-ClaudeExperiments-PombeSL-prediction/"
               "b2f61750-ea96-4fbb-9f26-9936c21c7a69/scratchpad/verify_huber")
ROOT = SCRATCH / "root"
LIVE = REPO / "data" / "processed"
if not ROOT.exists():
    ROOT.mkdir(parents=True)
    for p in LIVE.iterdir():
        if p.name in ("cache", "preds"):
            (ROOT / p.name).mkdir()
        else:
            (ROOT / p.name).symlink_to(p)
(SCRATCH / "results").mkdir(parents=True, exist_ok=True)
os.environ["PSL_DATA_ROOT"] = str(ROOT)
os.environ["PSL_RESULTS_ROOT"] = str(SCRATCH / "results")
OUT = Path(__file__).resolve().parent
REG_PREDS = LIVE / "preds"          # registered OOF predictions (read-only)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
THREADS = 2
