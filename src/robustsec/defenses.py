"""Training-time defenses."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .attacks import PGD, ThreatModel
from .models import TorchClassifier, fit_classifier


def adversarial_training(X: np.ndarray, y: np.ndarray, *, eps: float = 0.05,
                         steps: int = 7, alpha: Optional[float] = None,
                         threat_model: Optional[ThreatModel] = None,
                         adv_weight: float = 0.5, **fit_kw) -> TorchClassifier:
    """PGD adversarial training (Madry et al., 2018) with a mixed loss.

    Each mini-batch is perturbed by a PGD attack with ``steps`` iterations
    inside ``threat_model``. The training loss weights clean and
    adversarial cross-entropy by ``1 - adv_weight`` and ``adv_weight``.
    Remaining keyword arguments go to :func:`fit_classifier`.
    """
    tm = threat_model or ThreatModel.unconstrained(X.shape[1])
    pgd = PGD(eps, steps=steps, alpha=alpha if alpha is not None else 2.5 * eps / steps)

    def adversary(clf, xb, yb):
        return pgd.run(clf, xb, yb, tm)

    return fit_classifier(X, y, adversary=adversary, adv_weight=adv_weight, **fit_kw)
