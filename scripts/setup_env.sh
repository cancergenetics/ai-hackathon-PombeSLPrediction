#!/usr/bin/env bash
# Reproducible environment setup. Creates .venv and installs pinned requirements.
# macOS without Homebrew: LightGBM/XGBoost wheels need libomp; we reuse the copy bundled with scikit-learn.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
SP=$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])")
if [[ "$(uname)" == "Darwin" ]]; then
  for lib in "$SP/lightgbm/lib/lib_lightgbm.dylib" "$SP/xgboost/lib/libxgboost.dylib"; do
    d=$(dirname "$lib")
    if [[ ! -f "$d/libomp.dylib" ]]; then
      cp "$SP/sklearn/.dylibs/libomp.dylib" "$d/"
      install_name_tool -add_rpath @loader_path "$lib" 2>/dev/null || true
      codesign --force -s - "$lib"
    fi
  done
fi
.venv/bin/python -c "import lightgbm, xgboost, sklearn; print('env ok', lightgbm.__version__, xgboost.__version__, sklearn.__version__)"
