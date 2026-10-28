"""Shared setup for the independent M2 verification (reviewer scripts; never imported by src/).
Sets a PRIVATE data root (symlinks to data/frozen_m2, but its own empty cache/ and preds/) so nothing
here can read or poison the real M2 feature cache, and a private results root (no registry writes)."""
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "3"
SCRATCH = Path("/private/tmp/claude-503/-Users-colmclaude-ClaudeExperiments-PombeSL-prediction/"
               "b2f61750-ea96-4fbb-9f26-9936c21c7a69/scratchpad")
os.environ["PSL_DATA_ROOT"] = str(SCRATCH / "root")
os.environ["PSL_RESULTS_ROOT"] = str(SCRATCH / "results")
REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen_m2"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
THREADS = 3
