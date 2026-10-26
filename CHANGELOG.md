# Changelog

## 1.1.0 (2026-11)

Changes made in response to the SoftwareX reviews (SOFTX-D-26-01402).

### Fixed
- `PGD` kept only the last iterate of each restart. It now scores the repaired
  candidate after every step and keeps the best one over all steps and restarts.
- `CarliniWagnerL2` judged success before repair. It now measures success and
  the L2 norm on the repaired candidate.
- `TabularEncoder.repair` could exceed the budget after rounding. The new
  default `budget="strict"` keeps every candidate inside the eps-ball.
- `run_benchmark` dropped the Wilson intervals from `results.csv`.
- Unseen categorical values decoded as the first category; they now decode
  as `"<unknown>"` and are counted.

### Added
- `budget="nominal"` reproduces the v1.0 rounding; reports give the share and
  size of budget violations (`budget_violation_rate`, `budget_excess_*`).
- `Relation` objects in a schema, and `NSL_KDD_DILUTION` for error rates.
- `GreedySearch`, a score-based coordinate search over repaired records.
- `adversarial_training(repair=True)` trains on repaired examples.
- `paired_difference_ci` (Newcombe 1998, method 10) and `paired.csv`.
- `benign_flip_rate`, balanced accuracy and confusion counts.
- `TabularEncoder(handle_unknown=...)` and `unknown_counts_`.
- Hardware, peak memory, stage times and dataset sizes in `run_info.json`.
- UNSW-NB15 study, NSL-KDD ablation and defense configurations.
- `scripts/time_repair.py`, `scripts/paper_numbers.py`,
  `scripts/fill_placeholders.py`, `scripts/run_revision.sh`,
  `examples/sklearn_model.py`.
- Regression tests for every fix (58 tests in total).
- UNSW-NB15 loader reads the byte-order mark and detects swapped file names.

### Changed
- Strict repair is the default. Binary and categorical changes cost 1 in
  encoded space, so they need `eps >= 1` in strict mode.
- Figures use distinct line styles and markers for grayscale printing.

## 1.0.0 (2026-10-07)

First public release.

- Feature schema with feature types, mutability and allowed direction of change.
- Reversible encoder (log scaling, min-max scaling, one-hot encoding) with a
  repair step that maps adversarial inputs back to valid raw records.
- Attacks: FGSM, PGD with restarts, Carlini-Wagner L2 with an L-infinity bound,
  label-only random search, and transfer from a surrogate model.
- PGD adversarial training.
- Metrics: clean and robust accuracy, attack success rate, evasion rate,
  adversarial recall and F1, perturbation norms, Wilson confidence intervals.
- JSON-driven benchmark runner and command-line interface.
- NSL-KDD loader with checksum verification; UNSW-NB15 and generic CSV loaders.
- Unit tests and continuous integration.
