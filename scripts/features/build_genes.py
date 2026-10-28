#!/usr/bin/env python
"""Write <data root>/features/genes.txt: all PomBase protein-coding genes (systematic IDs, sorted).

This is the global gene order of the feature store (docs/INTERFACES.md): index i of every G x G
matrix refers to line i.  Source: PomBase pombase-2026-10-01 misc/gene_IDs_names_products.tsv.
Run: .venv/bin/python -I scripts/features/build_genes.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import feature_store as fs  # noqa: E402
from src.idmap import IDMapper, write_genes_txt  # noqa: E402


def main() -> int:
    out = fs.feat_dir() / "genes.txt"
    gl = write_genes_txt(out, IDMapper(log_dir=None))
    assert gl == sorted(set(gl))
    print(f"genes.txt: {len(gl)} protein-coding genes -> {out}")
    fs.update_manifest("genes.txt", {"group": "META", "sources": ["pombase"],
                                     "script": "scripts/features/build_genes.py", "sha256": fs.sha256(out),
                                     "kind": "gene_order", "n_genes": len(gl),
                                     "source_files": ["pombase:misc/gene_IDs_names_products.tsv (gene_type == 'protein coding gene')"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
