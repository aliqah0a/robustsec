"""Configuration-driven benchmark runner.

A JSON configuration lists the dataset, the threat models, the attacks,
the budgets, the defenses and the seeds. :func:`run_benchmark` executes
every combination and writes:

``results.csv``  one row per (seed, model, threat model, attack, eps), with
                 the per-seed Wilson intervals of ASR and evasion rate, the
                 budget checks and the wall time of the attack;
``summary.csv``  mean and standard deviation over seeds;
``paired.csv``   paired differences between threat models (same seed,
                 model, attack and eps) with Newcombe score intervals;
``clean.csv``    clean metrics and confusion counts per seed and model;
``run_info.json`` configuration, package versions, hardware, peak memory
                 and wall time.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import __version__
from .attacks import ThreatModel, TransferAttack, make_attack
from .data import RELATIONS, TabularDataset, load_nsl_kdd, load_unsw_nb15
from .defenses import adversarial_training
from .metrics import clean_report, paired_difference_ci, robustness_report
from .models import fit_classifier
from .preprocessing import TabularEncoder

DATASETS = {"nsl_kdd": load_nsl_kdd, "unsw_nb15": load_unsw_nb15}
METRIC_COLS = ["clean_accuracy", "robust_accuracy", "asr", "evasion_rate",
               "benign_flip_rate", "adv_recall", "adv_f1", "adv_balanced_accuracy",
               "linf", "l2", "l0", "linf_max", "budget_violation_rate",
               "budget_excess_mean", "budget_excess_max",
               "asr_ci_lo", "asr_ci_hi", "evasion_ci_lo", "evasion_ci_hi", "time_s"]


def load_config(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def build_threat_model(encoder: TabularEncoder, spec: dict) -> ThreatModel:
    """Turn a threat-model entry of the configuration into a ThreatModel.

    ``{"mutable": null, "repair": false}`` gives the unconstrained setting
    common in the literature. A list of ``mutable`` features, optional
    ``directions``, ``repair`` (default true), ``budget`` (``"strict"`` or
    ``"nominal"``) and ``relations`` (names in :data:`robustsec.data.RELATIONS`)
    give a constrained setting.
    """
    if spec.get("mutable") is None and not spec.get("repair", False) \
            and not spec.get("directions"):
        return ThreatModel.unconstrained(encoder.n_features_)
    rel = [RELATIONS[r] for r in spec.get("relations", [])]
    schema = encoder.schema.with_threat_model(spec.get("mutable"), spec.get("directions"),
                                              relations=rel)
    return ThreatModel.from_encoder(encoder, schema, repair=spec.get("repair", True),
                                    budget=spec.get("budget", "strict"))


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


def _defenses(config: dict) -> Dict[str, dict]:
    """Named adversarial-training variants (a single dict means ``adv_trained``)."""
    at = config.get("adversarial_training")
    if not at:
        return {}
    if "eps" in at:
        return {"adv_trained": dict(at)}
    return {k: dict(v) for k, v in at.items()}


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
    unknown = {k: v for k, v in enc.unknown_counts_.items() if v}
    y_tr, y_te = dataset.y_train, dataset.y_test
    log(f"[data] {dataset.name}: train {X_tr.shape}, test {X_te.shape}, "
        f"unseen categories in test: {unknown or 'none'}")

    tms = {name: build_threat_model(enc, spec)
           for name, spec in config["threat_models"].items()}
    rows: List[Dict] = []
    clean_rows: List[Dict] = []
    paired_rows: List[Dict] = []
    train_cfg = config.get("train", {})
    defenses = _defenses(config)
    stage_time: Dict[str, float] = {}
    for seed in config.get("seeds", [0]):
        idx = stratified_sample(y_te, config.get("eval_samples"), seed)
        Xe, ye = X_te[idx], y_te[idx]
        t = time.time()
        models = {"baseline": fit_classifier(X_tr, y_tr, seed=seed, **train_cfg)}
        stage_time["train_baseline"] = stage_time.get("train_baseline", 0) + time.time() - t
        for dname, at in defenses.items():
            t = time.time()
            at_tm = tms[at.pop("threat_model", next(iter(tms)))]
            models[dname] = adversarial_training(
                X_tr, y_tr, seed=seed, threat_model=at_tm, **{**train_cfg, **at})
            stage_time[f"train_{dname}"] = stage_time.get(f"train_{dname}", 0) + time.time() - t
            at["threat_model"] = next(k for k, v in tms.items() if v is at_tm)
        for mname, clf in models.items():
            rep = clean_report(clf, X_te, y_te)
            clean_rows.append(dict(seed=seed, model=mname, val_accuracy=clf.val_accuracy, **rep))
            log(f"[seed {seed}] {mname}: clean acc={rep['accuracy']:.4f} f1={rep['f1']:.4f}")

        outcomes = {}

        def record(mname, tname, aname, eps, clf, X_adv, elapsed):
            rep, oc = robustness_report(clf, Xe, X_adv, ye, eps=eps, mask=tms[tname].mask,
                                        return_outcomes=True)
            (alo, ahi), (elo, ehi) = rep.pop("asr_ci"), rep.pop("evasion_ci")
            rows.append(dict(seed=seed, model=mname, threat_model=tname, attack=aname,
                             eps=eps, asr_ci_lo=alo, asr_ci_hi=ahi, evasion_ci_lo=elo,
                             evasion_ci_hi=ehi, time_s=round(elapsed, 2), **rep))
            outcomes[(mname, tname, aname, eps)] = oc

        for tname, tm in tms.items():
            for eps in config["eps"]:
                for aname, params in config["attacks"].items():
                    for mname, clf in models.items():
                        atk = make_attack(aname, eps, tm, seed=seed, **params)
                        t = time.time()
                        X_adv = atk.generate(clf, Xe, ye)
                        record(mname, tname, atk.name, eps, clf, X_adv, time.time() - t)
                if config.get("transfer", True) and defenses:
                    src = make_attack("pgd", eps, tm, seed=seed, **config["attacks"].get("pgd", {}))
                    t = time.time()
                    X_src = src.generate(models["baseline"], Xe, ye)
                    elapsed = time.time() - t
                    for dname in defenses:
                        record(dname, tname, f"Transfer-{src.name}", eps, models[dname],
                               X_src, elapsed)
                log(f"[seed {seed}] {tname} eps={eps} done ({time.time() - t0:.0f}s)")

        for a, b in config.get("paired", []):
            for (mname, tname, aname, eps), oc in outcomes.items():
                if tname != a or (mname, b, aname, eps) not in outcomes:
                    continue
                ob = outcomes[(mname, b, aname, eps)]
                for metric, key in (("evasion_rate", "evaded"), ("asr", "flipped")):
                    d, lo, hi = paired_difference_ci(oc[key], ob[key])
                    paired_rows.append(dict(seed=seed, model=mname, attack=aname, eps=eps,
                                            metric=metric, threat_model_a=a, threat_model_b=b,
                                            diff=d, ci_lo=lo, ci_hi=hi, n=len(oc[key])))

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(out / "clean.csv", index=False)
    summarize(df).to_csv(out / "summary.csv", index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(out / "paired.csv", index=False)
    with open(out / "run_info.json", "w") as fh:
        json.dump(dict(config=config, robustsec=__version__, python=platform.python_version(),
                       versions=_versions(), hardware=_hardware(),
                       unseen_categories=unknown,
                       data=dict(name=dataset.name, n_train=int(len(y_tr)), n_test=int(len(y_te)),
                                 n_raw_features=len(dataset.schema),
                                 n_encoded_columns=int(enc.n_features_),
                                 attack_share_train=round(float(np.mean(y_tr)), 4),
                                 attack_share_test=round(float(np.mean(y_te)), 4),
                                 n_eval=int(len(stratified_sample(
                                     y_te, config.get("eval_samples"), 0)))),
                       stage_time_s={k: round(v, 1) for k, v in stage_time.items()},
                       peak_rss_mb=_peak_rss_mb(),
                       wall_time_s=round(time.time() - t0, 1)), fh, indent=2)
    log(f"[done] results in {out} ({time.time() - t0:.0f}s, peak RSS {_peak_rss_mb():.0f} MB)")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every configuration."""
    keys = ["model", "threat_model", "attack", "eps"]
    cols = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby(keys, sort=False)[cols]
    s = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    return s.reset_index()


def _versions() -> dict:
    import sklearn
    import torch
    return dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
                scikit_learn=sklearn.__version__)


def _hardware() -> dict:
    """CPU model, core count, total RAM, GPU and PyTorch thread count."""
    import torch
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    ram_gb = None
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return dict(platform=platform.platform(), machine=platform.machine(), cpu=cpu,
                logical_cores=os.cpu_count(), ram_gb=ram_gb, gpu=gpu,
                torch_threads=torch.get_num_threads())


def _peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1024 if platform.system() != "Darwin" else r / 2**20, 1)
    except (ImportError, OSError):
        return float("nan")
