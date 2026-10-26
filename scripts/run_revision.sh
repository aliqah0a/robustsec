#!/usr/bin/env bash
# Run every experiment of the revised paper (SOFTX-D-26-01402, revision 1),
# draw the figures and write the numbers into the manuscript.
#
#   bash scripts/run_revision.sh            # all studies
#   SKIP_UNSW=1 bash scripts/run_revision.sh  # without UNSW-NB15
#
# UNSW-NB15: place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in
# $ROBUSTSEC_DATA/unsw_nb15 (default ~/.robustsec/data/unsw_nb15).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs

pip install -e ".[dev]"
pytest -q
robustsec fetch

run() {  # name config
  echo "=== $1 ($(date))"
  robustsec run "configs/$2.json" -o "results/$1" 2>&1 | tee "logs/$1.log"
}
run nsl_kdd           nsl_kdd_paper
run nsl_kdd_ablation  nsl_kdd_ablation
run nsl_kdd_defenses  nsl_kdd_defenses
UNSW_ARG=""
if [ -z "${SKIP_UNSW:-}" ]; then
  run unsw_nb15 unsw_nb15_paper
  UNSW_ARG="--run unsw=results/unsw_nb15"
fi

python scripts/time_repair.py --out results/repair_timing.json
python scripts/check_saturation.py configs/unsw_nb15_paper.json --out results/unsw_nb15/saturation.json
python scripts/make_figures.py results/nsl_kdd paper/figures results/nsl_kdd_ablation
python scripts/paper_numbers.py --run nsl=results/nsl_kdd --run abl=results/nsl_kdd_ablation \
  --run def=results/nsl_kdd_defenses $UNSW_ARG \
  --timing results/repair_timing.json --json sat=results/unsw_nb15/saturation.json \
  -o paper/results_macros.tex
echo "=== done ($(date)). Now compile the paper (see paper/README_REVISION.md)."
