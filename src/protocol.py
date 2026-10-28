"""Single reader of config/protocol.yaml (DESIGN.md §0 protocol freeze).

All evaluation-defining values (label threshold, class weights, seeds, fractions ...) must be obtained
from here; tests/test_lint.py forbids them as literals elsewhere in src/ and scripts/.
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config" / "protocol.yaml"
CLASSES = ("C1", "C2", "C3")


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load(path: Path | None = None) -> dict:
    p = Path(path or PROTOCOL_PATH)
    return _load(str(p), p.stat().st_mtime)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_sha256() -> str:
    return sha256_file(PROTOCOL_PATH)


# ---------------------------------------------------------------------------------------- paths
def data_root() -> Path:
    P = load()["paths"]
    p = Path(os.environ.get(P["data_root_env"], P["data_root_default"]))
    return p if p.is_absolute() else ROOT / p


def results_root() -> Path:
    P = load()["paths"]
    p = Path(os.environ.get(P["results_root_env"], P["results_root_default"]))
    return p if p.is_absolute() else ROOT / p


def splits_dir() -> Path:
    return data_root() / load()["paths"]["splits_subdir"]


def preds_dir() -> Path:
    return data_root() / load()["paths"]["preds_subdir"]


def cache_root() -> Path:
    return data_root() / load()["paths"]["cache_subdir"]


# ---------------------------------------------------------------------------------------- label
def threshold() -> float:
    return float(load()["label"]["threshold"])


def label_column() -> str:
    return load()["label"]["column"]


def binarize(values) -> np.ndarray:
    """y = 1[value <direction> threshold]; NaN -> False."""
    lab = load()["label"]
    v = np.asarray(values, dtype=float)
    if lab["direction"] == "less_than":
        return v < float(lab["threshold"])
    raise ValueError(f"unsupported label direction {lab['direction']!r}")


# ---------------------------------------------------------------------------------------- weights
def class_weights(kind: str | None = None) -> dict:
    """Frozen class weights {C1, C2, C3}; kind in {None (= primary), 'unmeasured_universe',
    'all_pairs', 'equal'}. 'equal' is 1/3 each."""
    cw = load().get("class_weights")
    if kind == "equal":
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    if not cw:
        raise RuntimeError("class weights not yet computed: run scripts/compute_class_weights.py")
    kind = kind or cw["primary"]
    w = cw[kind]
    return {c: float(w[c]) for c in CLASSES}


def weightings() -> dict:
    """All weightings used for reporting: primary first."""
    cw = load()["class_weights"]
    out = {"primary": class_weights()}
    for k in ("all_pairs", "equal"):
        out[k] = class_weights(k)
    out["primary_name"] = cw["primary"]
    return out


# ---------------------------------------------------------------------------------------- writing
def write_derived_block(name: str, payload: dict, script: str, path: Path | None = None) -> None:
    """Replace the text between the '>>> BEGIN derived:<name>' and '<<< END derived:<name>' markers."""
    p = Path(path or PROTOCOL_PATH)
    text = p.read_text()
    pat = re.compile(rf"(# >>> BEGIN derived:{re.escape(name)}[^\n]*\n)(.*?)(# <<< END derived:{re.escape(name)})",
                     re.S)
    if not pat.search(text):
        raise RuntimeError(f"marker block derived:{name} not found in {p}")
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=None, width=110)
    head = f"# >>> BEGIN derived:{name} (written by {script}; do not edit by hand)\n"
    text = pat.sub(lambda m: head + body + m.group(3), text)
    p.write_text(text)
    _load.cache_clear()
