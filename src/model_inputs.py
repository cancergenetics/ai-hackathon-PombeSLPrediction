"""Default label-free model inputs, pinned in ONE config entry: `default_inputs` of config/models.yaml
(LESSONS: never let "every group in the store" decide the model inputs).

    from src import model_inputs as MI
    spec = MI.default_spec()                  # FeatureSpec restricted to default_inputs.lf_groups
    excl = MI.default_gene_table_exclude()    # store gene-table groups outside the default (GP inputs, m2.gene_table)
    sims = MI.default_sim_store()             # SimStore of default_inputs.sims (NBR / GP inputs)

`src.pairs.load_spec()`, `src.models.m2.gene_table()` and `src.ryanfeat.SimStore()` stay raw accessors of
everything in the store (needed for ablations such as XSP); model scripts (run_m1, run_m2, deliverable specs)
take their defaults from here, so a group merged into data/processed/features later (e.g. XSP) does not
change them. On the real data the pinned lists apply (every listed group / source must exist). On toy or
synthetic stores (not the real splits) the default is every group / source of that store, since those
stores have their own fixed contents (`real=` overrides the detection; tests).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src import pairs as PA
from src import protocol

MODELS_PATH = protocol.ROOT / "config" / "models.yaml"


def default_inputs(path: Path | None = None) -> dict:
    """{'lf_groups': [...], 'sims': [...]} from config/models.yaml `default_inputs`."""
    d = yaml.safe_load(Path(path or MODELS_PATH).read_text())["default_inputs"]
    assert isinstance(d["lf_groups"], list) and isinstance(d["sims"], list), d
    return {"lf_groups": list(d["lf_groups"]), "sims": list(d["sims"])}


def _real(real: bool | None) -> bool:
    if real is not None:
        return bool(real)
    from src import universe as U
    return U.is_real_data()


def default_lf_groups(spec: PA.FeatureSpec | None = None, real: bool | None = None) -> list:
    """Label-free feature groups of the default models (sorted)."""
    spec = spec or PA.load_spec()
    if not _real(real):
        return spec.groups()
    g = sorted(default_inputs()["lf_groups"])
    missing = set(g) - set(spec.groups())
    if missing:
        raise KeyError(f"default_inputs.lf_groups {sorted(missing)} not in the feature store {spec.groups()}")
    return g


def default_sims(real: bool | None = None) -> list | None:
    """Similarity sources of the default models; None = the SimStore default of a toy / synthetic store."""
    return sorted(default_inputs()["sims"]) if _real(real) else None


def restrict(spec: PA.FeatureSpec, groups) -> PA.FeatureSpec:
    """A FeatureSpec containing only `groups` (same manifest snapshot, genes and table cache)."""
    groups = set(groups)
    unknown = groups - set(spec.groups())
    if unknown:
        raise KeyError(f"unknown feature groups {sorted(unknown)}")
    if PA.HYPO_GROUP not in groups:
        raise ValueError("HYPO must be among the model groups (FeatureSpec always provides it)")
    out = PA.FeatureSpec(spec.fdir, spec.genes, spec.manifest,
                         [t for t in spec.gene_tables if t[0] in groups],
                         [m for m in spec.matrices if m[0] in groups])
    out._cache = spec._cache
    return out


def default_spec(fdir: Path | None = None, real: bool | None = None) -> PA.FeatureSpec:
    spec = PA.load_spec(fdir)
    return restrict(spec, default_lf_groups(spec, real))


def gene_table_exclude(spec: PA.FeatureSpec, groups) -> tuple:
    """Gene-table groups of the store that are not in `groups` (the `exclude_groups` argument of
    src.models.m2.gene_table / fold_tables_m2 / m2_cache_name)."""
    return tuple(sorted({g for g, _, _ in spec.gene_tables} - set(groups)))


def default_gene_table_exclude(real: bool | None = None) -> tuple:
    spec = PA.load_spec()
    return gene_table_exclude(spec, default_lf_groups(spec, real))


def default_sim_store(real: bool | None = None):
    from src.ryanfeat import SimStore
    names = default_sims(real)
    return SimStore(names=names if names is not None else SimStore().names)
