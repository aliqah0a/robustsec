"""Evasion attacks under an L-infinity budget and a feature-level threat model.

Every attack works in the encoded space produced by
:class:`robustsec.preprocessing.TabularEncoder`. A :class:`ThreatModel`
decides which encoded columns may move, in which direction, and inside
which box. When the threat model carries a repair function, every
candidate is mapped back to a valid raw input before it is scored, and
attacks keep only candidates whose success holds after repair. Gradient
steps are taken on the continuous relaxation; selection uses repaired points.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import Classifier, TorchClassifier


class ThreatModel:
    """Attacker capabilities expressed on encoded columns.

    Parameters
    ----------
    n_features : number of encoded columns.
    mask : 1 for columns the attacker may change, 0 otherwise.
    direction : +1 increase only, -1 decrease only, 0 both.
    lower, upper : box bounds of the encoded space (default [0, 1]).
    repair : optional ``f(X_adv, X_orig, eps) -> X_feasible`` applied to
        candidates, for example :meth:`TabularEncoder.repair`.
    encoder : the encoder the threat model was built from, if any. The
        greedy search uses it to enumerate valid raw values.
    """

    def __init__(self, n_features: int, mask=None, direction=None, lower=None,
                 upper=None, repair: Optional[Callable] = None, encoder=None,
                 schema=None, budget: str = "strict"):
        ones = np.ones(n_features, dtype=np.float32)
        self.mask = np.asarray(ones if mask is None else mask, dtype=np.float32)
        self.direction = np.asarray(0 * ones if direction is None else direction,
                                    dtype=np.float32)
        self.lower = np.asarray(0 * ones if lower is None else lower, dtype=np.float32)
        self.upper = np.asarray(ones if upper is None else upper, dtype=np.float32)
        self.repair = repair
        self.encoder, self.schema, self.budget = encoder, schema, budget
        self._cache = {}

    @classmethod
    def unconstrained(cls, n_features: int) -> "ThreatModel":
        """All columns mutable in both directions, box [0, 1], no repair."""
        return cls(n_features)

    @classmethod
    def from_encoder(cls, encoder, schema=None, repair: bool = True,
                     budget: str = "strict") -> "ThreatModel":
        """Build a threat model from an encoder and a (threat-model) schema.

        ``budget="strict"`` makes repair keep every candidate inside the
        eps-ball; ``"nominal"`` allows rounding to exceed it (v1.0).
        """
        v = encoder.constraint_vectors(schema)
        fn = ((lambda Xa, Xo, eps=None: encoder.repair(Xa, Xo, schema, eps=eps, budget=budget))
              if repair else None)
        return cls(len(v["mask"]), v["mask"], v["direction"], v["lower"], v["upper"], fn,
                   encoder=encoder, schema=schema or encoder.schema, budget=budget)

    def _t(self, device):
        if device not in self._cache:
            self._cache[device] = tuple(torch.as_tensor(a, device=device) for a in
                                        (self.mask, self.direction, self.lower, self.upper))
        return self._cache[device]

    def project(self, x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
        """Project ``x`` onto {eps-ball around x0} ∩ {allowed moves} ∩ box."""
        mask, direction, lower, upper = self._t(x.device)
        delta = (x - x0).clamp(-eps, eps) * mask
        delta = torch.where(direction > 0, delta.clamp(min=0), delta)
        delta = torch.where(direction < 0, delta.clamp(max=0), delta)
        lo = torch.minimum(lower, x0)   # keep the clean point feasible
        hi = torch.maximum(upper, x0)
        return torch.max(torch.min(x0 + delta, hi), lo)

    def finalize(self, X_adv: np.ndarray, X_orig: np.ndarray,
                 eps: Optional[float] = None) -> np.ndarray:
        """Repair candidates (no-op without a repair function)."""
        if self.repair is None:
            return np.asarray(X_adv, dtype=np.float32)
        return self.repair(X_adv, X_orig, eps)

    def finalize_tensor(self, x: torch.Tensor, x0: torch.Tensor,
                        eps: Optional[float] = None) -> torch.Tensor:
        """Tensor version of :meth:`finalize` (returns ``x`` without repair)."""
        if self.repair is None:
            return x
        out = self.repair(x.detach().cpu().numpy(), x0.detach().cpu().numpy(), eps)
        return torch.as_tensor(out, device=x.device)


class Attack:
    """Base class. Subclasses implement :meth:`_attack_batch`."""

    name = "attack"
    requires_gradients = True

    def __init__(self, eps: float, threat_model: Optional[ThreatModel] = None,
                 batch_size: int = 4096, seed: int = 0):
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.eps = float(eps)
        self.threat_model = threat_model
        self.batch_size = batch_size
        self.seed = seed

    def generate(self, clf: Classifier, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return adversarial versions of ``X`` (same shape, float32)."""
        if self.requires_gradients and not getattr(clf, "has_gradients", False):
            raise TypeError(f"{self.name} needs a white-box TorchClassifier")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        tm = self.threat_model or ThreatModel.unconstrained(X.shape[1])
        gen = torch.Generator().manual_seed(self.seed)
        out = []
        for i in range(0, len(X), self.batch_size):
            xb, yb = X[i:i + self.batch_size], y[i:i + self.batch_size]
            adv = self._attack_batch(clf, xb, yb, tm, gen)
            out.append(tm.finalize(adv, xb, self.eps))
        return np.concatenate(out) if out else X.copy()

    def _attack_batch(self, clf, X, y, tm, gen):  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eps={self.eps})"


def _grad(clf: TorchClassifier, x: torch.Tensor, y: torch.Tensor):
    x = x.detach().requires_grad_(True)
    logits = clf.logits(x)
    loss = F.cross_entropy(logits, y, reduction="none")
    (g,) = torch.autograd.grad(loss.sum(), x)
    return g, loss.detach(), logits.detach()


class FGSM(Attack):
    """Fast Gradient Sign Method (Goodfellow et al., 2015): one signed step."""

    name = "FGSM"

    def _attack_batch(self, clf, X, y, tm, gen):
        x0, yt = clf.to_tensor(X), torch.as_tensor(np.array(y, dtype=np.int64), device=clf.device)
        g, _, _ = _grad(clf, x0, yt)
        return tm.project(x0 + self.eps * g.sign(), x0, self.eps).cpu().numpy()


class PGD(Attack):
    """Projected gradient descent on the cross-entropy loss (Madry et al., 2018).

    Starts from a uniform random point in the feasible eps-ball (or from the
    clean point with ``random_start=False``) and runs ``restarts``
    independent runs. After every step, including the start, the iterate is
    repaired and scored. Per sample, the attack keeps the repaired candidate
    that is misclassified (if any) or that reached the highest loss, across
    all steps and restarts. Steps are taken on the unrepaired iterate.
    """

    name = "PGD"

    def __init__(self, eps, steps: int = 50, alpha: Optional[float] = None,
                 restarts: int = 1, random_start: bool = True, **kw):
        super().__init__(eps, **kw)
        self.steps = steps
        self.alpha = alpha if alpha is not None else 2.5 * self.eps / max(steps, 1)
        self.restarts = restarts
        self.random_start = random_start

    def run(self, clf, x0: torch.Tensor, yt: torch.Tensor, tm: ThreatModel,
            gen: Optional[torch.Generator] = None, repair: bool = True) -> torch.Tensor:
        """Tensor-level PGD, also used inside adversarial training.

        ``repair=False`` scores the raw iterates of the continuous
        relaxation instead of their repaired versions.
        """
        best = x0.clone()
        best_score = torch.full((len(x0),), -float("inf"), device=x0.device)
        use_repair = repair and tm.repair is not None

        def consider(x):
            nonlocal best, best_score
            cand = tm.finalize_tensor(x, x0, self.eps) if use_repair else x
            with torch.no_grad():
                logits = clf.logits(cand)
                loss = F.cross_entropy(logits, yt, reduction="none")
                score = loss + 1e6 * (logits.argmax(1) != yt).float()
                better = score > best_score
                best[better] = cand[better]
                best_score = torch.where(better, score, best_score)

        for _ in range(self.restarts):
            if self.random_start:
                noise = torch.rand(x0.shape, generator=gen).to(x0.device) * 2 - 1
                x = tm.project(x0 + self.eps * noise, x0, self.eps)
            else:
                x = x0.clone()
            consider(x)
            for _ in range(self.steps):
                g, _, _ = _grad(clf, x, yt)
                x = tm.project(x + self.alpha * g.sign(), x0, self.eps)
                consider(x)
        return best.detach()

    def _attack_batch(self, clf, X, y, tm, gen):
        x0, yt = clf.to_tensor(X), torch.as_tensor(np.array(y, dtype=np.int64), device=clf.device)
        return self.run(clf, x0, yt, tm, gen).cpu().numpy()


class CarliniWagnerL2(Attack):
    """Untargeted Carlini-Wagner L2 attack with an extra L-infinity bound.

    Minimises ``||delta||_2^2 + c * max(z_y - max_{j != y} z_j, -kappa)``
    with Adam, projecting onto the feasible eps-ball after every step. A
    binary search adapts ``c`` per sample. After every step the iterate is
    repaired; success and the L2 norm are measured on the repaired point.
    The attack returns, for each sample, the successful repaired
    perturbation with the smallest L2 norm, or the clean input when no
    repaired candidate succeeds.
    """

    name = "CW-L2"

    def __init__(self, eps, steps: int = 100, lr: float = 0.01, c: float = 1.0,
                 kappa: float = 0.0, search_steps: int = 4, **kw):
        super().__init__(eps, **kw)
        self.steps, self.lr, self.c0 = steps, lr, c
        self.kappa, self.search_steps = kappa, search_steps

    def _attack_batch(self, clf, X, y, tm, gen):
        x0, yt = clf.to_tensor(X), torch.as_tensor(np.array(y, dtype=np.int64), device=clf.device)
        n = len(x0)
        c = torch.full((n,), self.c0, device=x0.device)
        lo_c = torch.zeros(n, device=x0.device)
        hi_c = torch.full((n,), float("inf"), device=x0.device)
        best = x0.clone()
        best_l2 = torch.full((n,), float("inf"), device=x0.device)
        onehot = F.one_hot(yt, num_classes=clf.logits(x0[:1]).shape[1]).bool()
        for _ in range(self.search_steps):
            delta = torch.zeros_like(x0, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=self.lr)
            found = torch.zeros(n, dtype=torch.bool, device=x0.device)
            for _ in range(self.steps):
                # delta is kept feasible by the projection after each step, so
                # the forward pass uses it directly; projecting inside the graph
                # would zero the gradient at the boundary of one-sided moves.
                x = x0 + delta
                z = clf.logits(x)
                z_y = z[onehot]
                z_other = z.masked_fill(onehot, -float("inf")).max(1).values
                f = torch.clamp(z_y - z_other, min=-self.kappa)
                l2 = ((x - x0) ** 2).sum(1)
                loss = (l2 + c * f).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    delta.data = tm.project(x0 + delta.data, x0, self.eps) - x0
                    cand = tm.finalize_tensor(x.detach(), x0, self.eps)
                    zc = z.detach() if tm.repair is None else clf.logits(cand)
                    succ = zc.argmax(1) != yt
                    l2c = ((cand - x0) ** 2).sum(1)
                    improve = succ & (l2c < best_l2)
                    best[improve] = cand[improve]
                    best_l2 = torch.where(improve, l2c, best_l2)
                    found |= succ
            with torch.no_grad():   # binary search on c
                hi_c = torch.where(found, torch.minimum(hi_c, c), hi_c)
                lo_c = torch.where(found, lo_c, torch.maximum(lo_c, c))
                c = torch.where(torch.isinf(hi_c), c * 10, (lo_c + hi_c) / 2)
        return best.detach().cpu().numpy()


class RandomSearch(Attack):
    """Decision-based baseline: uniform random points in the feasible eps-ball.

    Uses only predicted labels, so it works with any :class:`Classifier`.
    Each candidate is repaired before it is queried. The first candidate
    that changes the label is kept. This is a weak attack and serves as a
    lower bound, not as a measure of black-box robustness.
    """

    name = "RandomSearch"
    requires_gradients = False

    def __init__(self, eps, queries: int = 200, **kw):
        super().__init__(eps, **kw)
        self.queries = queries

    def _attack_batch(self, clf, X, y, tm, gen):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        x0 = torch.as_tensor(X)
        adv = X.copy()
        open_ = clf.predict(X) == y          # still to be attacked
        for _ in range(self.queries):
            if not open_.any():
                break
            idx = np.flatnonzero(open_)
            noise = torch.as_tensor(rng.uniform(-1, 1, (len(idx), X.shape[1])).astype(np.float32))
            cand = tm.project(x0[idx] + self.eps * noise, x0[idx], self.eps).numpy()
            cand = tm.finalize(cand, X[idx])
            flip = clf.predict(cand) != y[idx]
            adv[idx[flip]] = cand[flip]
            open_[idx[flip]] = False
        return adv


class GreedySearch(Attack):
    """Score-based coordinate search on the valid, repaired input grid.

    The attack needs only class probabilities. In each round it tries, for
    every attacker-controlled feature, ``grid`` values spread over the
    interval that the threat model allows for that feature (budget,
    direction and domain), repairs each candidate, and applies the single
    change that raises the loss most. It stops for a sample once that
    sample is misclassified, after ``rounds`` rounds, or when no change
    raises the loss. Because every query is a valid repaired record, the
    search does not depend on gradients or on the continuous relaxation,
    which makes it a check on gradient masking caused by rounding.
    """

    name = "Greedy"
    requires_gradients = False

    def __init__(self, eps, rounds: int = 10, grid: int = 4, chunk: int = 256, **kw):
        super().__init__(eps, **kw)
        self.rounds, self.grid, self.chunk = rounds, grid, chunk

    def _candidate_values(self, tm, X0):
        """Per encoded column: (n, grid) encoded target values, or None."""
        mask, direction = tm.mask, tm.direction
        lo = np.maximum(np.minimum(tm.lower, X0), X0 - self.eps)
        hi = np.minimum(np.maximum(tm.upper, X0), X0 + self.eps)
        lo = np.where(direction > 0, X0, lo)
        hi = np.where(direction < 0, X0, hi)
        t = np.linspace(0.0, 1.0, self.grid)
        cols = [j for j in range(X0.shape[1]) if mask[j] > 0]
        return cols, {j: lo[:, j, None] + (hi[:, j] - lo[:, j])[:, None] * t for j in cols}

    def _attack_batch(self, clf, X, y, tm, gen):
        out = X.copy()
        for s0 in range(0, len(X), self.chunk):
            out[s0:s0 + self.chunk] = self._chunk(clf, X[s0:s0 + self.chunk],
                                                  y[s0:s0 + self.chunk], tm)
        return out

    def _loss(self, clf, Xc, yc):
        p = np.clip(clf.predict_proba(Xc), 1e-12, 1.0)
        return -np.log(p[np.arange(len(yc)), yc]), p.argmax(1)

    def _chunk(self, clf, X0, y, tm):
        n, d = X0.shape
        cur = X0.copy()
        cur_loss, pred = self._loss(clf, cur, y)
        active = pred == y
        cols, values = self._candidate_values(tm, X0)
        if not cols:
            return cur
        for _ in range(self.rounds):
            idx = np.flatnonzero(active)
            if not len(idx):
                break
            k = len(cols) * self.grid
            cand = np.repeat(cur[idx], k, axis=0).reshape(len(idx), k, d)
            for a, j in enumerate(cols):
                cand[:, a * self.grid:(a + 1) * self.grid, j] = values[j][idx]
            flat = cand.reshape(-1, d)
            base = np.repeat(X0[idx], k, axis=0)
            flat = tm.finalize(flat, base, self.eps)
            loss, p = self._loss(clf, flat, np.repeat(y[idx], k))
            loss = loss.reshape(len(idx), k)
            mis = (p != np.repeat(y[idx], k)).reshape(len(idx), k)
            score = loss + 1e6 * mis
            b = score.argmax(1)
            gain = score[np.arange(len(idx)), b] > cur_loss[idx] + 1e-12
            take = idx[gain]
            cur[take] = flat.reshape(len(idx), k, d)[gain, b[gain]]
            cur_loss[take] = loss[gain, b[gain]]
            done = mis[np.arange(len(idx)), b] & gain
            active[idx[done]] = False
            active[idx[~gain]] = False
        return cur


class TransferAttack(Attack):
    """Craft examples on a surrogate model and evaluate them on the target.

    A defense that resists white-box attacks but fails against examples
    transferred from an undefended surrogate shows signs of gradient
    masking (Athalye et al., 2018).
    """

    requires_gradients = False

    def __init__(self, base: Attack, surrogate: Classifier):
        super().__init__(base.eps, base.threat_model, base.batch_size, base.seed)
        self.base, self.surrogate = base, surrogate
        self.name = f"Transfer-{base.name}"

    def generate(self, clf, X, y):
        return self.base.generate(self.surrogate, X, y)


ATTACKS = {"fgsm": FGSM, "pgd": PGD, "cw": CarliniWagnerL2, "random": RandomSearch,
           "greedy": GreedySearch}


def make_attack(name: str, eps: float, threat_model: Optional[ThreatModel] = None,
                seed: int = 0, **params) -> Attack:
    """Instantiate an attack by its short name."""
    try:
        cls = ATTACKS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown attack '{name}'; choose from {sorted(ATTACKS)}") from None
    return cls(eps, threat_model=threat_model, seed=seed, **params)
