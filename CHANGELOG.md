# Changelog

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
