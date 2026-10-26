"""Schema-aware encoding of tabular data into a bounded attack space.

Numeric features are optionally ``log1p``-transformed and then min-max
scaled to [0, 1] with training statistics. Categorical features are one-hot
encoded. A perturbation budget ``eps`` is therefore a fraction of each
feature's (log-)range on the training set, which is comparable across
features with very different units.

The encoder also maps an encoded matrix back to valid raw values
(:meth:`TabularEncoder.repair`). This step rounds integer and binary
features, restores immutable features, enforces the allowed direction of
change, and re-projects one-hot groups to a single category. With
``budget="strict"`` (the default since v1.1) the result is also guaranteed
to stay inside the L-infinity ball of radius ``eps`` around the clean
input. With ``budget="nominal"`` (the v1.0 behaviour) rounding may move a
value past ``eps``; metrics then report the realized distance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import FeatureSchema


@dataclass
class _Column:
    name: str          # raw feature name
    kind: str
    log: bool
    lo: float          # training minimum (after log1p if used)
    span: float        # training range (1.0 for constant columns)
    raw_min: float
    raw_max: float


_TOL = 1e-9
BUDGET_MODES = ("strict", "nominal")


class TabularEncoder:
    """Fit on training data, then encode, decode and repair samples.

    Parameters
    ----------
    schema : the feature schema.
    handle_unknown : what :meth:`transform` does with a categorical value
        that was not seen during :meth:`fit`. ``"ignore"`` (default) encodes
        it as an all-zero one-hot group and counts it in
        ``unknown_counts_``; ``"error"`` raises a ``ValueError``.
    """

    def __init__(self, schema: FeatureSchema, handle_unknown: str = "ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.schema = schema
        self.handle_unknown = handle_unknown
        self.unknown_counts_: Dict[str, int] = {}
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self.numeric_: List[_Column] = []
        self.categories_: Dict[str, List[str]] = {}
        self.constant_features_: List[str] = []
        for f in self.schema.features:
            if f.name not in df.columns:
                raise KeyError(f"column '{f.name}' missing from data")
            if f.kind == "categorical":
                self.categories_[f.name] = sorted(df[f.name].astype(str).unique())
                continue
            raw = df[f.name].to_numpy(dtype=np.float64)
            if f.log and raw.min() < 0:
                raise ValueError(f"{f.name}: log1p needs non-negative values")
            v = np.log1p(raw) if f.log else raw
            lo, hi = float(v.min()), float(v.max())
            span = hi - lo
            if span == 0.0:
                self.constant_features_.append(f.name)
                span = 1.0
            self.numeric_.append(_Column(f.name, f.kind, f.log, lo, span,
                                         float(raw.min()), float(raw.max())))
        self._build_layout()
        self._fitted = True
        return self

    def _build_layout(self) -> None:
        """Record which encoded columns belong to which raw feature."""
        self.columns_: List[str] = []
        self.groups_: Dict[str, slice] = {}
        num = {c.name: c for c in self.numeric_}
        idx = 0
        for f in self.schema.features:
            if f.kind == "categorical":
                cats = self.categories_[f.name]
                self.columns_ += [f"{f.name}={c}" for c in cats]
                self.groups_[f.name] = slice(idx, idx + len(cats))
                idx += len(cats)
            else:
                self.columns_.append(f.name)
                self.groups_[f.name] = slice(idx, idx + 1)
                idx += 1
        self._num_by_name = num
        self.n_features_ = idx
        self.numeric_mask_ = np.zeros(idx, dtype=bool)
        for c in self.numeric_:
            self.numeric_mask_[self.groups_[c.name].start] = True

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check()
        out = np.zeros((len(df), self.n_features_), dtype=np.float64)
        for f in self.schema.features:
            sl = self.groups_[f.name]
            if f.kind == "categorical":
                vals = df[f.name].astype(str).to_numpy()
                cats = self.categories_[f.name]
                for k, c in enumerate(cats):
                    out[:, sl.start + k] = (vals == c)
                n_unknown = int((~np.isin(vals, cats)).sum())
                if n_unknown and self.handle_unknown == "error":
                    raise ValueError(f"{f.name}: {n_unknown} values not seen during fit")
                self.unknown_counts_[f.name] = n_unknown
            else:
                c = self._num_by_name[f.name]
                v = df[f.name].to_numpy(dtype=np.float64)
                v = np.log1p(np.maximum(v, 0.0)) if c.log else v
                out[:, sl.start] = (v - c.lo) / c.span
        return out.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def to_raw(self, X: np.ndarray) -> pd.DataFrame:
        """Decode an encoded matrix to raw feature values.

        Integer and binary features are rounded to the nearest integer, which
        removes float32 round-off introduced by the encoding.
        """
        self._check()
        X = np.asarray(X, dtype=np.float64)
        data = {}
        for f in self.schema.features:
            sl = self.groups_[f.name]
            if f.kind == "categorical":
                cats = np.array(self.categories_[f.name], dtype=object)
                block = X[:, sl]
                vals = cats[block.argmax(axis=1)]
                vals[block.max(axis=1) < 0.5] = "<unknown>"   # all-zero group
                data[f.name] = vals
            else:
                v = self._decode_num(f.name, X[:, sl.start])
                data[f.name] = np.rint(v).astype(np.int64) if f.kind in ("integer", "binary") else v
        return pd.DataFrame(data)

    def _decode_num(self, name: str, s: np.ndarray) -> np.ndarray:
        c = self._num_by_name[name]
        v = s * c.span + c.lo
        return np.expm1(v) if c.log else v

    def _encode_num(self, name: str, r: np.ndarray) -> np.ndarray:
        c = self._num_by_name[name]
        v = np.log1p(np.maximum(r, 0.0)) if c.log else r
        return (v - c.lo) / c.span

    # ---------------------------------------------------- attack constraints
    def constraint_vectors(self, schema: Optional[FeatureSchema] = None) -> dict:
        """Per-encoded-column mutability, direction and box bounds.

        ``schema`` may carry a different threat model than the one used to
        fit the encoder (same features, different ``mutable``/``direction``).
        """
        self._check()
        schema = schema or self.schema
        mask = np.zeros(self.n_features_, dtype=np.float32)
        direction = np.zeros(self.n_features_, dtype=np.float32)
        for f in schema.features:
            sl = self.groups_[f.name]
            if f.mutable and f.name not in self.constant_features_:
                mask[sl] = 1.0
                direction[sl] = f.direction
        return dict(mask=mask, direction=direction,
                    lower=np.zeros(self.n_features_, dtype=np.float32),
                    upper=np.ones(self.n_features_, dtype=np.float32))

    def raw_bounds(self, name: str, x0: np.ndarray, eps: Optional[float],
                   direction: int = 0):
        """Raw-value interval an attacker may reach for one numeric feature.

        The interval is the training range (widened to include the clean
        value), intersected with the eps-ball in encoded space when ``eps``
        is given, and cut at the clean value for one-sided directions.
        Integer and binary features get integer end points.
        """
        c = self._num_by_name[name]
        f = self.schema[name]
        r_org = self._decode_num(name, x0)
        if f.kind in ("integer", "binary"):
            r_org = np.round(r_org)
        lo = np.minimum(c.raw_min, r_org)
        hi = np.maximum(c.raw_max, r_org)
        if eps is not None:
            e_lo = np.maximum(np.minimum(0.0, x0), x0 - eps)
            e_hi = np.minimum(np.maximum(1.0, x0), x0 + eps)
            lo = np.maximum(lo, self._decode_num(name, e_lo))
            hi = np.minimum(hi, self._decode_num(name, e_hi))
            if f.kind in ("integer", "binary"):
                lo = np.ceil(lo - _TOL)
                hi = np.floor(hi + _TOL)
            lo = np.minimum(lo, r_org)       # the clean value is always allowed
            hi = np.maximum(hi, r_org)
        if direction > 0:
            lo = r_org
        elif direction < 0:
            hi = r_org
        return lo, hi, r_org

    def repair(self, X_adv: np.ndarray, X_orig: np.ndarray,
               schema: Optional[FeatureSchema] = None, eps: Optional[float] = None,
               budget: str = "strict") -> np.ndarray:
        """Map encoded adversarial samples back onto the valid input domain.

        For each raw feature the method (i) restores immutable values,
        (ii) clips to the training range (widened to include the clean
        value), (iii) rounds integer and binary features, (iv) enforces the
        allowed direction, and (v) forces each one-hot group to a single
        category. A category changes only if its column exceeds 0.5.
        Relations of the schema are applied last.

        ``budget="strict"`` with a given ``eps`` also keeps every
        attacker-controlled encoded column within ``eps`` of the clean value:
        integer features are rounded to the nearest integer inside the ball,
        and binary or categorical changes, which cost 1 in encoded space,
        are allowed only when ``eps >= 1``. ``budget="nominal"`` reproduces
        v1.0, where rounding may exceed ``eps``. The method is idempotent.
        """
        self._check()
        if budget not in BUDGET_MODES:
            raise ValueError(f"budget must be one of {BUDGET_MODES}")
        strict = budget == "strict" and eps is not None
        schema = schema or self.schema
        X_adv = np.array(X_adv, dtype=np.float64, copy=True)
        X_orig = np.asarray(X_orig, dtype=np.float64)
        for f in schema.features:
            sl = self.groups_[f.name]
            if not f.mutable or f.name in self.constant_features_:
                X_adv[:, sl] = X_orig[:, sl]
                continue
            if f.kind == "categorical":
                if strict and eps < 1.0:
                    X_adv[:, sl] = X_orig[:, sl]
                    continue
                block = X_adv[:, sl]
                hot = block.argmax(axis=1)
                onehot = np.zeros_like(block)
                onehot[np.arange(len(hot)), hot] = 1.0
                keep = block.max(axis=1) < 0.5
                onehot[keep] = X_orig[keep][:, sl]
                X_adv[:, sl] = onehot
                continue
            j = sl.start
            lo, hi, r_org = self.raw_bounds(f.name, X_orig[:, j], eps if strict else None)
            r_adv = np.clip(self._decode_num(f.name, X_adv[:, j]), lo, hi)
            if f.kind in ("integer", "binary"):
                r_adv = np.clip(np.round(r_adv), np.ceil(lo - _TOL), np.floor(hi + _TOL))
            if f.direction > 0:
                r_adv = np.maximum(r_adv, r_org)
            elif f.direction < 0:
                r_adv = np.minimum(r_adv, r_org)
            X_adv[:, j] = self._encode_num(f.name, r_adv)
            # keep unchanged entries bit-identical to the original encoding
            same = r_adv == r_org
            X_adv[same, j] = X_orig[same, j]
        if schema.relations:
            X_adv = self._apply_relations(X_adv, X_orig, schema)
        return X_adv.astype(np.float32)

    def _apply_relations(self, X_adv, X_orig, schema):
        num = {c.name for c in self.numeric_}
        for rel in schema.relations:
            bad = (set(rel.inputs) | set(rel.outputs)) - num
            if bad:
                raise ValueError(f"relation {rel.name}: only numeric features, got {sorted(bad)}")
            names = set(rel.inputs) | set(rel.outputs)
            adv = {n: self._decode_num(n, X_adv[:, self.groups_[n].start]) for n in names}
            org = {n: self._decode_num(n, X_orig[:, self.groups_[n].start]) for n in names}
            for n in names:
                if self.schema[n].kind in ("integer", "binary"):
                    adv[n], org[n] = np.round(adv[n]), np.round(org[n])
            for n, v in rel.fn(adv, org).items():
                # relation outputs are trusted values; they are not clipped to
                # the training range, which may not cover diluted rates
                j = self.groups_[n].start
                v = np.asarray(v, dtype=np.float64)
                if self.schema[n].kind in ("integer", "binary"):
                    v = np.round(v)
                X_adv[:, j] = self._encode_num(n, v)
                same = v == org[n]
                X_adv[same, j] = X_orig[same, j]
        return X_adv

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("TabularEncoder is not fitted")
