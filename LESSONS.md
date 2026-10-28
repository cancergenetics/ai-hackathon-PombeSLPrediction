# LESSONS — process rules for running this project efficiently

(One line per rule, with evidence. Pruned at each milestone.)

1. On this Mac, LightGBM/XGBoost wheels fail to import without libomp; fix by copying sklearn's bundled libomp + `install_name_tool -add_rpath @loader_path` (scripts/setup_env.sh). Evidence: M0 import errors, ~5 min lost.
