"""Robustness metrics with explicit denominators.

Absolute error on adversarial inputs mixes clean mistakes with attack
success. RobustSec therefore reports, next to robust accuracy, the
*attack success rate* (flip rate among samples that were classified
correctly before the attack) and the *evasion rate* (fraction of detected
malicious samples that the attack turns into benign predictions).
Perturbation sizes are averaged over successful samples only.

Two kinds of uncertainty appear in a study. A Wilson interval
(:func:`wilson_interval`) covers sampling error of the test set for one
trained model. The standard deviation over seeds covers variation between
trained models. :func:`paired_difference_ci` compares two threat models on
the same samples and the same model, so it accounts for their correlation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion ``k / n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def paired_difference_ci(a: np.ndarray, b: np.ndarray, z: float = 1.96):
    """Difference of two paired proportions with a 95 % score interval.

    ``a`` and ``b`` are boolean outcomes on the same samples (for example,
    evasion under two threat models). Returns ``(mean(a) - mean(b), lo,
    hi)`` using Newcombe's hybrid score method for paired data (method 10,
    Newcombe 1998), which uses the Wilson interval of each proportion and
    the correlation between the paired outcomes.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0 or len(b) != n:
        return (float("nan"),) * 3
    n11 = int((a & b).sum()); n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum()); n00 = n - n11 - n10 - n01
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson_interval(n11 + n10, n, z)
    l2, u2 = wilson_interval(n11 + n01, n, z)
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = (n11 * n00 - n10 * n01) / den if den > 0 else 0.0
    d = p1 - p2
    lo = d - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    hi = d + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return (d, max(lo, -1.0), min(hi, 1.0))


def clean_report(clf, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Clean-data metrics (class 1 = positive) with the confusion counts.

    ``balanced_accuracy`` is the mean of the per-class recalls, which does
    not depend on the class ratio of the test set.
    """
    y = np.asarray(y)
    pred = clf.predict(X)
    return dict(accuracy=float((pred == y).mean()),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                tp=int(((pred == 1) & (y == 1)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
                tn=int(((pred == 0) & (y == 0)).sum()), fn=int(((pred == 0) & (y == 1)).sum()),
                n=int(len(y)))


def robustness_report(clf, X_clean: np.ndarray, X_adv: np.ndarray, y: np.ndarray,
                      positive: int = 1, tol: float = 1e-6, eps: Optional[float] = None,
                      mask: Optional[np.ndarray] = None,
                      return_outcomes: bool = False) -> Dict[str, float]:
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
      ``benign_flip_rate`` : ``P(pred = attack | y = benign, clean correct)``,
      the false alarms the attack creates.
      ``adv_balanced_accuracy`` : mean per-class recall on adversarial inputs.
      ``linf_max`` : largest L-infinity distance over all perturbed samples.
      ``budget_violation_rate``, ``budget_excess_mean``, ``budget_excess_max`` :
      with ``eps`` given, the share of perturbed samples whose realized
      L-infinity distance on the ``mask`` columns exceeds ``eps + tol``, and
      the mean and largest excess over ``eps`` among them.
    With ``return_outcomes=True`` a second value holds per-sample boolean
    arrays ``flipped``, ``evaded`` and the index sets ``correct``, ``detected``.
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

    perturbed = (np.abs(diff) > tol).any(1)
    linf_all = np.abs(diff).max(1) if diff.size else np.zeros(len(y))
    linf_max = float(linf_all[perturbed].max()) if perturbed.any() else 0.0
    viol_rate = excess_mean = excess_max = float("nan")
    if eps is not None:
        dm = diff if mask is None else diff[:, np.asarray(mask) > 0]
        lm = np.abs(dm).max(1) if dm.shape[1] else np.zeros(len(y))
        excess = lm - eps
        viol = perturbed & (excess > tol)
        viol_rate = float(viol.sum() / perturbed.sum()) if perturbed.any() else 0.0
        excess_mean = float(excess[viol].mean()) if viol.any() else 0.0
        excess_max = float(excess[viol].max()) if viol.any() else 0.0
    benign = correct & (y != positive)
    n_b = int(benign.sum())
    benign_flip = benign & (p_adv == positive)

    n_c, n_d = int(correct.sum()), int(detected.sum())
    out = dict(
        clean_accuracy=float(correct.mean()),
        robust_accuracy=float((p_adv == y).mean()),
        asr=float(flipped.sum() / n_c) if n_c else float("nan"),
        asr_ci=wilson_interval(int(flipped.sum()), n_c),
        evasion_rate=float(evaded.sum() / n_d) if n_d else float("nan"),
        evasion_ci=wilson_interval(int(evaded.sum()), n_d),
        adv_recall=float(recall_score(y, p_adv, zero_division=0)),
        adv_f1=float(f1_score(y, p_adv, zero_division=0)),
        benign_flip_rate=float(benign_flip.sum() / n_b) if n_b else float("nan"),
        adv_balanced_accuracy=float(balanced_accuracy_score(y, p_adv)),
        linf=linf, l2=l2, l0=l0, linf_max=linf_max,
        budget_violation_rate=viol_rate, budget_excess_mean=excess_mean,
        budget_excess_max=excess_max,
        n=int(len(y)), n_correct=n_c, n_detected=n_d,
    )
    if return_outcomes:
        return out, dict(flipped=flipped[correct], evaded=evaded[detected],
                         correct=np.flatnonzero(correct), detected=np.flatnonzero(detected))
    return out
