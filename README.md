# RobustSec

[![tests](https://github.com/aliqah0a/robustsec/actions/workflows/ci.yml/badge.svg)](https://github.com/aliqah0a/robustsec/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23216057.svg)](https://doi.org/10.5281/zenodo.23216057)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

RobustSec is a Python package for evaluating the adversarial robustness of
tabular network intrusion detection systems (NIDS). It separates two questions
that are often mixed in the literature:

1. How robust is a detector when an attacker may change every encoded feature?
2. How robust is it when the attacker may change only the traffic properties
   it actually controls, in the direction it can change them, and the result
   must still be a valid record?

The first question uses the *unconstrained* threat model. The second uses a
*feasibility-aware* threat model, which you declare once in a feature schema.
The same attacks, metrics and reports then run under both.

## Features

- **Feature schema.** Each feature has a type (continuous, integer, binary,
  categorical), a mutability flag and an allowed direction of change.
- **Reversible encoder.** Optional `log1p`, min-max scaling fitted on training
  data, and one-hot encoding. `TabularEncoder.repair` maps a perturbed input
  back to a valid raw record: it restores immutable features, rounds integer
  and binary features, keeps one category active, and enforces the direction.
- **Attacks.** FGSM, PGD (random start, restarts, best iterate), Carlini-Wagner
  L2 bounded in L-infinity, label-only random search, and transfer from a
  surrogate model.
- **Defense.** PGD adversarial training with a mixed clean/adversarial loss.
- **Metrics.** Clean and robust accuracy, attack success rate (ASR) on
  correctly classified inputs, evasion rate on detected attacks, adversarial
  recall and F1, L-infinity/L2/L0 norms of successful perturbations, and Wilson
  95% confidence intervals.
- **Reproducible runs.** A JSON configuration fixes data, threat models,
  attacks, budgets, defenses and seeds. Each run writes per-seed results, a
  mean ± s.d. summary, and the software versions used.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/aliqah0a/robustsec.git
cd robustsec
pip install -e ".[dev]"
pytest -q
```

On machines without a GPU, install the CPU build of PyTorch first to save
disk space:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data

NSL-KDD is downloaded on first use and checked against SHA-256 checksums:

```bash
robustsec fetch                 # stores files in ~/.robustsec/data/nsl_kdd
```

Set `ROBUSTSEC_DATA` or pass `--data-dir` to use another location. If your
machine has no internet access, copy `KDDTrain+.txt` and `KDDTest+.txt` into
that folder.

UNSW-NB15 is not redistributed. Download the official
`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the UNSW
Canberra Cyber Range Lab and load them with `robustsec.load_unsw_nb15`. Any
other binary-labelled CSV dataset can be loaded with `robustsec.load_csv`.

## Quick start

```python
import robustsec as rs

ds = rs.load_nsl_kdd(download=True)
enc = rs.TabularEncoder(ds.schema).fit(ds.X_train)
X_tr, X_te = enc.transform(ds.X_train), enc.transform(ds.X_test)

clf = rs.fit_classifier(X_tr, ds.y_train, seed=0)

# Attacker controls five traffic volume and timing features and can only increase them.
feasible = rs.ThreatModel.from_encoder(enc, rs.nsl_kdd_feasible_schema())
unconstrained = rs.ThreatModel.unconstrained(enc.n_features_)

for name, tm in [("unconstrained", unconstrained), ("feasible", feasible)]:
    X_adv = rs.PGD(eps=0.1, steps=50, restarts=2, threat_model=tm).generate(clf, X_te, ds.y_test)
    r = rs.robustness_report(clf, X_te, X_adv, ds.y_test)
    print(f"{name:14s} robust acc={r['robust_accuracy']:.3f} "
          f"ASR={r['asr']:.3f} evasion={r['evasion_rate']:.3f} L0={r['l0']:.1f}")

# Adversarial records in the original feature space
raw = enc.to_raw(X_adv)
```

`eps` is a fraction of each feature's scaled range. For log-scaled features
this is a fraction of the log range.

### Declaring your own threat model

```python
schema = ds.schema.with_threat_model(
    mutable=["duration", "src_bytes", "dst_bytes"],
    directions={"duration": 1, "src_bytes": 1, "dst_bytes": 1},  # +1: increase only
)
tm = rs.ThreatModel.from_encoder(enc, schema)
```

## Command line

```bash
robustsec run configs/nsl_kdd_quick.json -o results/quick      # about 1 minute on 2 CPU cores
robustsec run configs/nsl_kdd_paper.json -o results/nsl_kdd    # full study, 5 seeds
robustsec plot results/nsl_kdd/summary.csv --metric evasion_rate
```

Each run writes:

| File | Content |
| --- | --- |
| `results.csv` | one row per seed, model, threat model, attack and budget |
| `summary.csv` | mean and standard deviation over seeds |
| `clean.csv` | clean test metrics and validation accuracy per seed |
| `run_info.json` | configuration, package versions, platform and wall time |

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh
```

The script runs the tests, the full NSL-KDD study and the figure script. The
reference outputs used in the paper are in `results/nsl_kdd/`.

## Metric definitions

Let *C* be the test inputs the model classifies correctly and *D* the attack
inputs it detects.

- **Robust accuracy**: accuracy on adversarial inputs over all test inputs.
- **ASR**: share of *C* whose prediction changes after the attack.
- **Evasion rate**: share of *D* predicted benign after the attack.
- **L-infinity, L2, L0**: perturbation size in the encoded space, averaged over
  successful adversarial inputs only.

## Repository layout

```
src/robustsec/   package source
tests/           unit and integration tests (pytest)
configs/         benchmark configurations
examples/        quick-start script and Colab notebook
scripts/         reproduction and figure scripts
results/         reference results for the paper
docs/            user guide
```

## Citation

See `CITATION.cff`. The software is archived on Zenodo: version 1.0.0 has the DOI
[10.5281/zenodo.23216058](https://doi.org/10.5281/zenodo.23216058), and
[10.5281/zenodo.23216057](https://doi.org/10.5281/zenodo.23216057) always points to the
latest version. A BibTeX entry for the paper will be added when it is published.

## License

MIT. See `LICENSE.txt`.
