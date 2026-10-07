"""Classifier wrappers and a reference MLP.

Attacks interact with models only through the :class:`Classifier`
interface. :class:`TorchClassifier` exposes logits and input gradients and
supports every attack. :class:`SklearnClassifier` (or any object with a
``predict`` method) exposes labels only and supports decision-based
attacks such as :class:`robustsec.attacks.RandomSearch`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class Classifier:
    """Minimal interface used by attacks and metrics."""

    has_gradients: bool = False

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SklearnClassifier(Classifier):
    """Label-only wrapper around any object with ``predict``."""

    has_gradients = False

    def __init__(self, model):
        if not hasattr(model, "predict"):
            raise TypeError("model must implement predict()")
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X)))


class TorchClassifier(Classifier):
    """White-box wrapper around a ``torch.nn.Module`` that returns logits."""

    has_gradients = True

    def __init__(self, module: nn.Module, device: Optional[str] = None,
                 batch_size: int = 4096):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.module = module.to(self.device)
        self.batch_size = batch_size

    def to_tensor(self, X) -> torch.Tensor:
        return torch.as_tensor(np.array(X, dtype=np.float32, copy=True), device=self.device)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass in evaluation mode."""
        self.module.eval()
        return self.module(x)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.module.eval()
        out = []
        for i in range(0, len(X), self.batch_size):
            z = self.module(self.to_tensor(X[i:i + self.batch_size]))
            out.append(torch.softmax(z, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class MLP(nn.Module):
    """Feed-forward network with ReLU and dropout."""

    def __init__(self, n_features: int, hidden: Sequence[int] = (128, 64),
                 n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_validation(X: np.ndarray, y: np.ndarray, fraction: float, seed: int):
    """Stratified train/validation split drawn from the training set only."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val_idx.append(idx[: int(round(fraction * len(idx)))])
    val = np.zeros(len(y), dtype=bool)
    val[np.concatenate(val_idx)] = True
    return X[~val], y[~val], X[val], y[val]


def fit_classifier(X: np.ndarray, y: np.ndarray, *, hidden: Sequence[int] = (128, 64),
                   epochs: int = 15, lr: float = 1e-3, batch_size: int = 256,
                   val_fraction: float = 0.1, seed: int = 0,
                   device: Optional[str] = None,
                   adversary: Optional[Callable] = None, adv_weight: float = 0.5,
                   verbose: bool = False) -> TorchClassifier:
    """Train an :class:`MLP` and keep the weights with the best validation accuracy.

    ``adversary`` turns this into adversarial training: it is called as
    ``adversary(clf, xb, yb)`` on each mini-batch and must return perturbed
    inputs as a tensor. The loss is ``(1 - adv_weight) * CE(clean) +
    adv_weight * CE(adversarial)``. The validation split comes from the
    training data; the test set is never used here.
    """
    set_seed(seed)
    X_tr, y_tr, X_va, y_va = split_validation(X, y, val_fraction, seed)
    clf = TorchClassifier(MLP(X.shape[1], hidden), device=device)
    opt = torch.optim.Adam(clf.module.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    Xt, yt = torch.as_tensor(X_tr), torch.as_tensor(y_tr)
    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(clf.device), yt[idx].to(clf.device)
            if adversary is not None:
                x_adv = adversary(clf, xb, yb).detach()
            clf.module.train()
            opt.zero_grad()
            loss = loss_fn(clf.module(xb), yb)
            if adversary is not None:
                loss = (1 - adv_weight) * loss + adv_weight * loss_fn(clf.module(x_adv), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        acc = float((clf.predict(X_va) == y_va).mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in clf.module.state_dict().items()}
        if verbose:
            print(f"epoch {epoch + 1:2d}  loss={total / len(Xt):.4f}  val_acc={acc:.4f}")
    clf.module.load_state_dict(best_state)
    clf.val_accuracy = best_acc
    return clf
