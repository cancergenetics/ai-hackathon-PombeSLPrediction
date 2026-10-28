#!/usr/bin/env python
"""Merge a staged feature group into the live feature store, with checksum verification
(docs/INTERFACES.md; docs/features_xsp.md "To adopt").

    merge_staged.py [--staging <data root>/features_staging] [--manifest manifest_xsp.json] [--fix-aliases]
                    [--dry-run]

Steps (all-or-nothing per file; idempotent):
  1. Every entry of the staged manifest must name an existing staged file whose sha256 equals the recorded
     sha256, and every source must be a use=feature dataset id of config/data_allowlist.yaml (exact id).
  2. Each staged file is copied to <features>/<key> through a temporary file + atomic rename; the copy is
     re-hashed and must match. A destination that already exists with the same sha256 is left alone; one
     with a DIFFERENT sha256 aborts the merge (no silent overwrite).
  3. The staged entries are merged into <features>/manifest.json under the store's manifest lock
     (src.feature_store.update_manifest).
  4. --fix-aliases rewrites short source aliases in the live manifest to exact allowlist ids
     (ALIAS_FIX: `biogrid` -> `biogrid_physical`; `pombase` -> `pombase_orthologs` for build_cer/build_ppi
     outputs, `pombase_ids_viability` for genes.txt), including column-level sources. Only the `sources`
     fields change, never a file or a sha256; a timestamped backup of the manifest is written first.
A JSON record of what was done goes to results/features/merge_staged_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from src import feature_store as fs  # noqa: E402

ALLOWLIST = ROOT / "config" / "data_allowlist.yaml"
# alias -> exact dataset id, per producing script (manifest "script" field); "*" = any script
ALIAS_FIX = {
    "biogrid": {"*": "biogrid_physical"},
    "pombase": {"scripts/features/build_cer.py": "pombase_orthologs",
                "scripts/features/build_ppi.py": "pombase_orthologs",
                "scripts/features/build_genes.py": "pombase_ids_viability"},
}


def feature_ids(path: Path = ALLOWLIST) -> dict:
    """dataset id -> use (from config/data_allowlist.yaml)."""
    return {k: v.get("use") for k, v in yaml.safe_load(Path(path).read_text())["datasets"].items()}


def verify_staged(staging: Path, man: dict, allow: dict) -> list:
    """Problems with the staged manifest/files (empty list = OK)."""
    bad = []
    for key, e in man.items():
        p = staging / key
        if not key.startswith(("gene/", "pair/", "sim/")):
            bad.append(f"{key}: unexpected key (expect gene/, pair/ or sim/)")
        if not p.exists():
            bad.append(f"{key}: staged file missing")
            continue
        if fs.sha256(p) != e.get("sha256"):
            bad.append(f"{key}: sha256 mismatch (staged file != manifest)")
        for s in e.get("sources") or []:
            if allow.get(s) != "feature":
                bad.append(f"{key}: source '{s}' is not an exact use=feature dataset id")
        if not e.get("sources"):
            bad.append(f"{key}: no sources")
    return bad


def copy_verified(src: Path, dst: Path, sha: str, dry: bool) -> str:
    """'copied' | 'identical' (already present) ; raises on a conflicting destination or a bad copy."""
    if dst.exists():
        if fs.sha256(dst) == sha:
            return "identical"
        raise SystemExit(f"{dst} exists with a different sha256; refusing to overwrite")
    if dry:
        return "would_copy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".merge_tmp")
    shutil.copy2(src, tmp)
    if fs.sha256(tmp) != sha:
        tmp.unlink()
        raise SystemExit(f"copy of {src} -> {tmp} has a wrong sha256")
    os.replace(tmp, dst)
    return "copied"


def has_alias(srcs: list, allow: dict) -> bool:
    return any(s in ALIAS_FIX and allow.get(s) is None for s in srcs)


def fixed_sources(srcs: list, script: str, allow: dict) -> list:
    out = []
    for s in srcs:
        if s in ALIAS_FIX and allow.get(s) is None:
            m = ALIAS_FIX[s]
            t = m.get(script, m.get("*"))
            if t is None:
                raise SystemExit(f"no exact id for alias '{s}' from {script}")
            s = t
        out.append(s)
    return sorted(set(out))


def fix_aliases(man: dict, allow: dict) -> tuple[dict, list]:
    """(new manifest, list of changes). Rewrites entry-level and column-level `sources`."""
    new, changes = json.loads(json.dumps(man)), []
    for key, e in new.items():
        script = e.get("script", "")
        if "sources" in e and has_alias(e["sources"], allow):
            f = fixed_sources(e["sources"], script, allow)
            if f != e["sources"]:
                changes.append({"key": key, "from": e["sources"], "to": f})
                e["sources"] = f
        for c, ce in (e.get("columns") or {}).items():
            if isinstance(ce, dict) and "sources" in ce and has_alias(ce["sources"], allow):
                f = fixed_sources(ce["sources"], script, allow)
                if f != ce["sources"]:
                    changes.append({"key": f"{key}:{c}", "from": ce["sources"], "to": f})
                    ce["sources"] = f
    return new, changes


def rewrite_manifest(fdir: Path, transform, dry: bool) -> list:
    """Apply transform(manifest) -> (manifest, changes) under the store lock; backup first."""
    mpath = fdir / "manifest.json"
    with open(fdir / ".manifest.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        man = json.loads(mpath.read_text())
        new, changes = transform(man)
        if changes and not dry:
            ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            shutil.copy2(mpath, fdir / f"manifest.json.bak-{ts}")
            tmp = mpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(dict(sorted(new.items())), indent=1) + "\n")
            tmp.replace(mpath)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return changes


def merge(staging: Path, manifest_name: str, fdir: Path, dry: bool = False, do_fix: bool = False) -> dict:
    allow = feature_ids()
    sman = json.loads((staging / manifest_name).read_text())
    bad = verify_staged(staging, sman, allow)
    if bad:
        raise SystemExit("staged files failed verification:\n" + "\n".join(bad))
    live = json.loads((fdir / "manifest.json").read_text())
    rec = {"staging": str(staging), "manifest": manifest_name, "features_dir": str(fdir), "dry_run": dry,
           "files": {}, "alias_changes": []}
    for key, e in sman.items():
        if key in live and live[key].get("sha256") not in (None, e["sha256"]):
            raise SystemExit(f"live manifest already has {key} with a different sha256")
        rec["files"][key] = copy_verified(staging / key, fdir / key, e["sha256"], dry)
    if not dry:
        for key, e in sman.items():
            fs.update_manifest(key, e, fdir)
            assert fs.sha256(fdir / key) == json.loads((fdir / "manifest.json").read_text())[key]["sha256"]
    if do_fix:
        rec["alias_changes"] = rewrite_manifest(fdir, lambda m: fix_aliases(m, allow), dry)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=None, help="default: <data root>/features_staging")
    ap.add_argument("--manifest", default="manifest_xsp.json")
    ap.add_argument("--features", default=None, help="default: <data root>/features")
    ap.add_argument("--fix-aliases", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fdir = Path(a.features) if a.features else fs.feat_dir()
    staging = Path(a.staging) if a.staging else fs.data_root() / "features_staging"
    rec = merge(staging, a.manifest, fdir, a.dry_run, a.fix_aliases)
    rec["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
    print(json.dumps({k: rec[k] for k in ("files",)}, indent=1), f"\n{len(rec['alias_changes'])} alias changes")
    if not a.dry_run:
        out = ROOT / "results" / "features" / f"merge_staged_{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1))
        print(f"record -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
