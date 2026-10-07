"""Configuration-driven benchmark runner.

A JSON configuration lists the dataset, the threat models, the attacks,
the budgets, the defenses and the seeds. :func:`run_benchmark` executes
every combination and writes one row per (seed, model, threat model,
attack, eps) to ``results.csv`` together with ``summary.csv`` (mean and
standard deviation over seeds) and ``run_info.json`` (configuration,
package versions, wall time).
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import __version__
from .attacks import ThreatModel, TransferAttack, make_attack
from .data import TabularDataset, load_nsl_kdd
from .defenses import adversarial_training
from .metrics import clean_report, robustness_report
from .models import fit_classifier
from .preprocessing import TabularEncoder

DATASETS = {"nsl_kdd": load_nsl_kdd}
METRIC_COLS = ["clean_accuracy", "robust_accuracy", "asr", "evasion_rate",
               "adv_recall", "adv_f1", "linf", "l2", "l0"]


def load_config(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def build_threat_model(encoder: TabularEncoder, spec: dict) -> ThreatModel:
    """Turn a threat-model entry of the configuration into a ThreatModel.

    ``{"mutable": null, "repair": false}`` gives the unconstrained setting
    common in the literature. A list of ``mutable`` features, optional
    ``directions`` and ``repair: true`` give a feasibility-aware setting.
    """
    if spec.get("mutable") is None and not spec.get("repair", False):
        return ThreatModel.unconstrained(encoder.n_features_)
    schema = encoder.schema.with_threat_model(spec.get("mutable"), spec.get("directions"))
    return ThreatModel.from_encoder(encoder, schema, repair=spec.get("repair", True))


def stratified_sample(y: np.ndarray, n: Optional[int], seed: int) -> np.ndarray:
    """Indices of a class-stratified random sample (all indices if ``n`` is None)."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        k = int(round(n * len(pool) / len(y)))
        idx.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def run_benchmark(config: dict, out_dir, dataset: Optional[TabularDataset] = None,
                  verbose: bool = True) -> pd.DataFrame:
    """Run the benchmark described by ``config`` and write results to ``out_dir``."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    if dataset is None:
        dcfg = dict(config["dataset"])
        dataset = DATASETS[dcfg.pop("name")](**dcfg)
    enc = TabularEncoder(dataset.schema).fit(dataset.X_train)
    X_tr, X_te = enc.transform(dataset.X_train), enc.transform(dataset.X_test)
    y_tr, y_te = dataset.y_train, dataset.y_test
    log(f"[data] {dataset.name}: train {X_tr.shape}, test {X_te.shape}")

    tms = {name: build_threat_model(enc, spec)
           for name, spec in config["threat_models"].items()}
    rows: List[Dict] = []
    clean_rows: List[Dict] = []
    train_cfg = config.get("train", {})
    for seed in config.get("seeds", [0]):
        idx = stratified_sample(y_te, config.get("eval_samples"), seed)
        Xe, ye = X_te[idx], y_te[idx]
        models = {"baseline": fit_classifier(X_tr, y_tr, seed=seed, **train_cfg)}
        if "adversarial_training" in config:
            at = dict(config["adversarial_training"])
            at_tm = tms[at.pop("threat_model", next(iter(tms)))]
            models["adv_trained"] = adversarial_training(
                X_tr, y_tr, seed=seed, threat_model=at_tm, **{**train_cfg, **at})
        for mname, clf in models.items():
            rep = clean_report(clf, X_te, y_te)
            clean_rows.append(dict(seed=seed, model=mname, val_accuracy=clf.val_accuracy, **rep))
            log(f"[seed {seed}] {mname}: clean acc={rep['accuracy']:.4f} f1={rep['f1']:.4f}")

        for tname, tm in tms.items():
            for eps in config["eps"]:
                for aname, params in config["attacks"].items():
                    for mname, clf in models.items():
                        atk = make_attack(aname, eps, tm, seed=seed, **params)
                        X_adv = atk.generate(clf, Xe, ye)
                        rows.append(dict(seed=seed, model=mname, threat_model=tname,
                                         attack=atk.name, eps=eps,
                                         **robustness_report(clf, Xe, X_adv, ye)))
                if config.get("transfer", True) and "adv_trained" in models:
                    src = make_attack("pgd", eps, tm, seed=seed, **config["attacks"].get("pgd", {}))
                    atk = TransferAttack(src, models["baseline"])
                    X_adv = atk.generate(models["adv_trained"], Xe, ye)
                    rows.append(dict(seed=seed, model="adv_trained", threat_model=tname,
                                     attack=atk.name, eps=eps,
                                     **robustness_report(models["adv_trained"], Xe, X_adv, ye)))
                log(f"[seed {seed}] {tname} eps={eps} done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.drop(columns=["asr_ci", "evasion_ci"]).to_csv(out / "results.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(out / "clean.csv", index=False)
    summarize(df).to_csv(out / "summary.csv", index=False)
    with open(out / "run_info.json", "w") as fh:
        json.dump(dict(config=config, robustsec=__version__, python=platform.python_version(),
                       versions=_versions(), platform=platform.platform(),
                       wall_time_s=round(time.time() - t0, 1)), fh, indent=2)
    log(f"[done] results in {out} ({time.time() - t0:.0f}s)")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every configuration."""
    keys = ["model", "threat_model", "attack", "eps"]
    g = df.groupby(keys, sort=False)[METRIC_COLS]
    s = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    return s.reset_index()


def _versions() -> dict:
    import sklearn
    import torch
    return dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
                scikit_learn=sklearn.__version__)
