# RobustSec user guide

## 1. Concepts

**Feature schema.** A `FeatureSchema` is a list of `FeatureSpec` objects.
Each spec stores a name, a type (`continuous`, `integer`, `binary`,
`categorical`), a flag `mutable`, a `direction` (`0` both ways, `+1` increase
only, `-1` decrease only) and a flag `log` for heavy-tailed counts.

**Encoder.** `TabularEncoder(schema).fit(train_df)` learns scaling statistics
and categories from training data only. `transform` returns a float32 matrix in
[0, 1] for numeric features plus one-hot columns. `to_raw` inverts it.

**Threat model.** A `ThreatModel` holds a mask of mutable encoded columns, a
direction per column, box bounds, and an optional repair function. Attacks call
`project` after each step and score the repaired iterate (`finalize`), so
success always refers to a valid record. `finalize` applies
`TabularEncoder.repair`, so every returned input decodes to a valid record.

**Budget.** `eps` bounds the L-infinity change in the encoded space. Because
numeric features are scaled to [0, 1], `eps=0.1` means one tenth of the
(log-)range observed in training data. For a log-scaled feature it bounds the
ratio `(1 + x') / (1 + x)` by `exp(eps * span)`.

**Budget after repair.** `ThreatModel.from_encoder(enc, schema, budget="strict")`
(default) rounds integer features to the nearest value inside the eps-ball,
so every candidate keeps the nominal budget. Binary and categorical changes
cost 1 in encoded space and therefore need `eps >= 1`. `budget="nominal"`
rounds to the nearest integer and may exceed `eps`; the reports then give
`budget_violation_rate`, `budget_excess_mean` and `budget_excess_max`.

**Relations.** A `Relation(name, inputs, outputs, fn)` recomputes derived
features after repair. `fn(adv, orig)` gets dicts of raw arrays and returns
new raw values for `outputs`. Pass relations to `with_threat_model(...,
relations=[...])` or list their names under `"relations"` in a configuration.

**Unseen categories.** `TabularEncoder(schema, handle_unknown="ignore")`
encodes a value not seen in training as an all-zero group and counts it in
`unknown_counts_`; `"error"` raises instead.

## 2. Attacks

| Class | Access | Notes |
| --- | --- | --- |
| `FGSM(eps)` | gradients | one signed-gradient step |
| `PGD(eps, steps, alpha, restarts, random_start)` | gradients | keeps the best repaired candidate over all steps and restarts |
| `CarliniWagnerL2(eps, steps, lr, c, kappa, search_steps)` | gradients | binary search on `c`; returns the smallest successful L2 perturbation within the L-infinity bound |
| `RandomSearch(eps, queries)` | labels only | candidates are repaired before each query |
| `GreedySearch(eps, rounds, grid)` | class probabilities | one feature per round, grid of valid values, every query repaired |
| `TransferAttack(base, surrogate)` | none on target | crafts on the surrogate, evaluates on the target |

Gradient attacks need a `TorchClassifier`. Wrap a scikit-learn model with
`SklearnClassifier` and use `RandomSearch`, or attack a PyTorch surrogate and
use `TransferAttack`.

## 3. Metrics

`robustness_report(clf, X_clean, X_adv, y)` returns a dictionary:

- `clean_accuracy`, `robust_accuracy`
- `asr`, `asr_ci`: label flips among correctly classified inputs
- `evasion_rate`, `evasion_ci`: detected attacks that become "benign"
- `adv_recall`, `adv_f1`: detection quality on adversarial inputs
- `linf`, `l2`, `l0`: mean norms over successful adversarial inputs
- `n`, `n_correct`, `n_detected`

Confidence intervals use the Wilson score interval at 95%. It covers
test-set sampling for one trained model. The standard deviation over seeds in
`summary.csv` covers variation between trained models. `paired_difference_ci`
(Newcombe 1998, method 10) compares two threat models on the same samples;
list pairs under `"paired"` in a configuration to get `paired.csv`.

## 4. Configuration file

```json
{
  "dataset": {"name": "nsl_kdd", "download": true},
  "seeds": [0, 1, 2, 3, 4],
  "eval_samples": null,
  "eps": [0.01, 0.02, 0.05, 0.1, 0.2],
  "threat_models": {
    "unconstrained": {"mutable": null, "repair": false},
    "constrained": {"mutable": ["duration", "src_bytes"],
                 "directions": {"duration": 1, "src_bytes": 1},
                 "repair": true}
  },
  "attacks": {"fgsm": {}, "pgd": {"steps": 50, "restarts": 2}},
  "train": {"hidden": [128, 64], "epochs": 15},
  "adversarial_training": {"threat_model": "unconstrained", "eps": 0.05, "steps": 7},
  "transfer": true
}
```

`eval_samples: null` evaluates the full test set. An integer draws a
class-stratified sample per seed.

## 5. Using another dataset

```python
ds = rs.load_csv("train.csv", "test.csv", label="label", positive="attack",
                 categorical=["proto"], exclude=["id"])
schema = ds.schema.with_threat_model(mutable=["pkt_len", "iat"], directions={"pkt_len": 1})
```

`run_benchmark(config, out_dir, dataset=ds)` accepts any `TabularDataset`.
