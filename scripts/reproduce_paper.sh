#!/usr/bin/env bash
# Reproduce every number and figure in the RobustSec paper.
# Runtime: about 2 hours on 2 CPU cores; much less with a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -e ".[dev]"
pytest -q
robustsec fetch
robustsec run configs/nsl_kdd_paper.json -o results/nsl_kdd
python scripts/make_figures.py results/nsl_kdd results/nsl_kdd/figures
