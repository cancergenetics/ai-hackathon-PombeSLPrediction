# Research tree — S. pombe synthetic-lethality prediction

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (latest first)

### 2026-10-28 — M0: project kickoff
Project initialised: git repo, Python 3.10 venv (`scripts/setup_env.sh`, `requirements.txt`; LightGBM/XGBoost need a libomp rpath fix on this Mac), PROMPT.md saved verbatim. Literature review and data discovery launched in parallel as independent agents.

Next: integrate LITERATURE.md + data inventory, draft DESIGN.md, run up to three rounds of multi-reviewer critique, then build the data pipeline and leakage-safe splits.

## Tree
- [~] 1. Literature review → LITERATURE.md
- [~] 2. Data discovery & acquisition → DATA.md, scripts/download_*.py
- [ ] 3. Design → DESIGN.md (≤3 review rounds)
- [ ] 4. Data pipeline, gene ID mapping, labels, splits (+ verification)
- [ ] 5. Baselines
- [ ] 6. Feature-based models
- [ ] 7. Advanced models / cross-species transfer
- [ ] 8. Final evaluation on held-out test → REPORT.md

## Decision log
- 2026-10-28: Use a project-local venv; pin with `pip freeze`. Rationale: bare system Python, no conda/brew.
