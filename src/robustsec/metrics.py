"""Robustness metrics with explicit denominators.

Absolute error on adversarial inputs mixes clean mistakes with attack
success. RobustSec therefore reports, next to robust accuracy, the
*attack success rate* (flip rate among samples that were classified
correctly before the attack) and the *evasion rate* (fraction of detected
malicious samples that the attack turns into benign predictions).
Perturbation sizes are averaged over successful samples only.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion ``k / n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def clean_report(clf, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall and F1 on clean data (class 1 = positive)."""
    pred = clf.predict(X)
    return dict(accuracy=float((pred == y).mean()),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                n=int(len(y)))


def robustness_report(clf, X_clean: np.ndarray, X_adv: np.ndarray, y: np.ndarray,
                      positive: int = 1, tol: float = 1e-6) -> Dict[str, float]:
    """Compare clean and adversarial predictions of one classifier.

    Returns
    -------
    dict with
      ``clean_accuracy``, ``robust_accuracy`` : accuracy before/after attack.
      ``asr`` : attack success rate, ``P(adv wrong | clean correct)``.
      ``evasion_rate`` : ``P(pred = benign | y = positive, clean detected)``.
      ``adv_recall``, ``adv_f1`` : detection quality on adversarial inputs.
      ``linf``, ``l2``, ``l0`` : mean distances over successful samples
      (L0 counts encoded columns changed by more than ``tol``).
      ``asr_ci``, ``evasion_ci`` : 95 % Wilson intervals.
    """
    y = np.asarray(y)
    p_clean = clf.predict(X_clean)
    p_adv = clf.predict(X_adv)
    correct = p_clean == y
    flipped = correct & (p_adv != y)
    detected = correct & (y == positive)
    evaded = detected & (p_adv != positive)

    diff = np.asarray(X_adv, np.float64) - np.asarray(X_clean, np.float64)
    succ = flipped
    if succ.any():
        d = diff[succ]
        linf = float(np.abs(d).max(1).mean())
        l2 = float(np.linalg.norm(d, axis=1).mean())
        l0 = float((np.abs(d) > tol).sum(1).mean())
    else:
        linf = l2 = l0 = float("nan")

    n_c, n_d = int(correct.sum()), int(detected.sum())
    return dict(
        clean_accuracy=float(correct.mean()),
        robust_accuracy=float((p_adv == y).mean()),
        asr=float(flipped.sum() / n_c) if n_c else float("nan"),
        asr_ci=wilson_interval(int(flipped.sum()), n_c),
        evasion_rate=float(evaded.sum() / n_d) if n_d else float("nan"),
        evasion_ci=wilson_interval(int(evaded.sum()), n_d),
        adv_recall=float(recall_score(y, p_adv, zero_division=0)),
        adv_f1=float(f1_score(y, p_adv, zero_division=0)),
        linf=linf, l2=l2, l0=l0,
        n=int(len(y)), n_correct=n_c, n_detected=n_d,
    )
