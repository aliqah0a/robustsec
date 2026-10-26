# RobustSec v1.1.0

This release answers the reviews of the SoftwareX submission SOFTX-D-26-01402.
It fixes three problems in how attacks select and repair candidates, adds a
budget-aware repair mode, and adds the experiments of the revised paper.

## Fixes

- **PGD candidate selection.** PGD kept only the last iterate of each restart,
  so a success found at an intermediate step could be lost. It now repairs and
  scores the iterate after every step and keeps the best repaired candidate
  over all steps and restarts.
- **Carlini–Wagner success after repair.** Success and the L2 norm were
  measured before repair. They are now measured on the repaired candidate, and
  the attack returns either a misclassified repaired record or the clean one.
- **Budget after rounding.** `TabularEncoder.repair` could move an integer
  feature past the L-infinity budget. The new default `budget="strict"` keeps
  every candidate inside the eps-ball; `budget="nominal"` keeps the v1.0
  behaviour.
- **Exported intervals.** `results.csv` now keeps the per-seed Wilson bounds
  (`asr_ci_lo/hi`, `evasion_ci_lo/hi`).
- **Unseen categories** decode as `"<unknown>"` instead of the first category.
- **UNSW-NB15 loader** reads the byte-order mark of the official CSV files and
  detects copies whose training and testing file names are swapped.

## New

- `GreedySearch`: a score-based coordinate search over repaired records.
- `adversarial_training(repair=True)`: training on repaired examples.
- `Relation` objects in a schema, with `NSL_KDD_DILUTION` for error rates.
- `paired_difference_ci` (Newcombe 1998, method 10) and `paired.csv`.
- Budget checks (`budget_violation_rate`, `budget_excess_mean/max`),
  `benign_flip_rate`, balanced accuracy and confusion counts.
- `run_info.json` records CPU, cores, RAM, GPU, peak memory, stage times,
  dataset sizes and unseen categories.
- Configurations for the NSL-KDD ablation, the defense comparison and
  UNSW-NB15; `scripts/run_revision.sh` runs them all.
- `scripts/paper_numbers.py` and `scripts/fill_placeholders.py` write the
  results into the manuscript; `scripts/time_repair.py` measures repair cost.
- `scripts/check_saturation.py`: detects cross-entropy saturation of PGD.
- `examples/sklearn_model.py`: a gradient-boosting detector with the
  gradient-free attacks.
- 58 tests, including a regression test for each fix.

## Behaviour changes

- Strict repair is the default. Binary and categorical changes cost 1 in the
  encoded space, so they need `eps >= 1` in strict mode.
- `ThreatModel` repair functions now receive `eps`:
  `repair(X_adv, X_orig, eps)`.
- Results of v1.0 are not directly comparable with v1.1, because attacks now
  select candidates after repair and the budget is strict.

## Reference results

`results/` holds the outputs used in the revised paper: `nsl_kdd`,
`nsl_kdd_ablation`, `nsl_kdd_defenses`, `unsw_nb15` and `repair_timing.json`.
