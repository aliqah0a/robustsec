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
*schema-constrained* threat model, which you declare once in a feature schema.
The same attacks, metrics and reports then run under both.

**Scope.** A repaired record has valid types and domains, unchanged immutable
features, the allowed directions, the nominal budget (strict mode) and the
declared relations. RobustSec does not show that a packet sequence exists that
produces the record. Results are comparisons between feature-space threat
models.

## Features

- **Feature schema.** Each feature has a type (continuous, integer, binary,
  categorical), a mutability flag and an allowed direction of change.
  Optional `Relation` objects recompute derived features (for example,
  `rs.NSL_KDD_DILUTION` rescales the NSL-KDD error rates when `count` grows).
- **Reversible encoder.** Optional `log1p`, min-max scaling fitted on training
  data, and one-hot encoding. `TabularEncoder.repair` maps a perturbed input
  back to a valid raw record: it restores immutable features, rounds integer
  and binary features, keeps one category active, enforces the direction and
  applies the relations. With `budget="strict"` (default) the result stays
  inside the eps-ball; `budget="nominal"` reproduces the v1.0 rounding.
  Unseen categories become an all-zero group (`handle_unknown="ignore"`) or
  raise an error (`handle_unknown="error"`).
- **Attacks.** FGSM, PGD (random start, restarts, best repaired candidate over
  all steps), Carlini-Wagner L2 bounded in L-infinity with success checked
  after repair, label-only random search, score-based greedy search over
  repaired records, and transfer from a surrogate model.
- **Defense.** PGD adversarial training with a mixed clean/adversarial loss,
  on repaired examples (`repair=True`, default) or on the continuous
  relaxation (`repair=False`).
- **Metrics.** Clean and robust accuracy, attack success rate (ASR) on
  correctly classified inputs, evasion rate on detected attacks, adversarial
  recall, F1 and balanced accuracy, benign flip rate, L-infinity/L2/L0 norms
  of successful perturbations, budget checks after repair, Wilson 95%
  intervals per seed, and paired intervals between threat models.
- **Reproducible runs.** A JSON configuration fixes data, threat models,
  attacks, budgets, defenses and seeds. Each run writes per-seed results, a
  mean ± s.d. summary, paired comparisons, and the software and hardware used.

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
Canberra Cyber Range Lab, place them in `~/.robustsec/data/unsw_nb15/`, and
load them with `robustsec.load_unsw_nb15()`. The official training partition
has 175,341 records and the testing partition 82,332. Some copies of the files
carry swapped names; the loader detects this from the record counts, warns,
and uses the larger file for training. It also strips the byte-order mark at
the start of the files. Any
other binary-labelled CSV dataset can be loaded with `robustsec.load_csv`.

## Quick start

```python
import robustsec as rs

ds = rs.load_nsl_kdd(download=True)
enc = rs.TabularEncoder(ds.schema).fit(ds.X_train)
X_tr, X_te = enc.transform(ds.X_train), enc.transform(ds.X_test)

clf = rs.fit_classifier(X_tr, ds.y_train, seed=0)

# Attacker controls five traffic volume and timing features and can only increase them.
constrained = rs.ThreatModel.from_encoder(enc, rs.nsl_kdd_feasible_schema())
unconstrained = rs.ThreatModel.unconstrained(enc.n_features_)

for name, tm in [("unconstrained", unconstrained), ("constrained", constrained)]:
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
| `results.csv` | one row per seed, model, threat model, attack and budget, with Wilson bounds, budget checks and attack time |
| `summary.csv` | mean and standard deviation over seeds |
| `paired.csv` | paired differences between threat models with Newcombe intervals |
| `clean.csv` | clean metrics, confusion counts and validation accuracy per seed |
| `run_info.json` | configuration, versions, CPU, cores, RAM, GPU, peak memory, stage times, dataset sizes |

## Reproducing the paper

```bash
bash scripts/run_revision.sh          # all four studies, figures and paper numbers
```

The script runs the tests, the main NSL-KDD study, the ablation, the defense
comparison and (if the files are present) the UNSW-NB15 study. It then draws
the figures and writes `paper/results_macros.tex`, which fills every number in
the manuscript. The reference outputs used in the paper are in `results/`.

## Metric definitions

Let *C* be the test inputs the model classifies correctly and *D* the attack
inputs it detects.

- **Robust accuracy**: accuracy on adversarial inputs over all test inputs.
- **ASR**: share of *C* whose prediction changes after the attack.
- **Evasion rate**: share of *D* predicted benign after the attack.
- **Benign flip rate**: share of correctly classified benign inputs predicted
  as attacks after the attack.
- **Budget violation rate**: share of perturbed inputs whose realized
  L-infinity distance on the attacker-controlled columns exceeds eps.
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
