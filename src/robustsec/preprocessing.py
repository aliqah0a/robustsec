"""Schema-aware encoding of tabular data into a bounded attack space.

Numeric features are optionally ``log1p``-transformed and then min-max
scaled to [0, 1] with training statistics. Categorical features are one-hot
encoded. A perturbation budget ``eps`` is therefore a fraction of each
feature's (log-)range on the training set, which is comparable across
features with very different units.

The encoder also maps an encoded matrix back to valid raw values
(:meth:`TabularEncoder.repair`). This step rounds integer and binary
features, restores immutable features, enforces the allowed direction of
change, and re-projects one-hot groups to a single category.
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


class TabularEncoder:
    """Fit on training data, then encode, decode and repair samples."""

    def __init__(self, schema: FeatureSchema):
        self.schema = schema
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

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check()
        out = np.zeros((len(df), self.n_features_), dtype=np.float64)
        for f in self.schema.features:
            sl = self.groups_[f.name]
            if f.kind == "categorical":
                vals = df[f.name].astype(str).to_numpy()
                for k, c in enumerate(self.categories_[f.name]):
                    out[:, sl.start + k] = (vals == c)
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
                data[f.name] = cats[X[:, sl].argmax(axis=1)]
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

    def repair(self, X_adv: np.ndarray, X_orig: np.ndarray,
               schema: Optional[FeatureSchema] = None) -> np.ndarray:
        """Map encoded adversarial samples back onto the valid input domain.

        For each raw feature the method (i) restores immutable values,
        (ii) enforces the allowed direction, (iii) clips to the training
        range (extended to include the original value), (iv) rounds
        integer and binary features, and (v) forces each one-hot group to a
        single category. The result can lie slightly outside the original
        eps-ball because of rounding; metrics report the realized distance.
        """
        self._check()
        schema = schema or self.schema
        X_adv = np.array(X_adv, dtype=np.float64, copy=True)
        X_orig = np.asarray(X_orig, dtype=np.float64)
        for f in schema.features:
            sl = self.groups_[f.name]
            if not f.mutable or f.name in self.constant_features_:
                X_adv[:, sl] = X_orig[:, sl]
                continue
            if f.kind == "categorical":
                hot = X_adv[:, sl].argmax(axis=1)
                onehot = np.zeros_like(X_adv[:, sl])
                onehot[np.arange(len(hot)), hot] = 1.0
                X_adv[:, sl] = onehot
                continue
            j = sl.start
            c = self._num_by_name[f.name]
            r_adv = self._decode_num(f.name, X_adv[:, j])
            r_org = self._decode_num(f.name, X_orig[:, j])
            if f.kind in ("integer", "binary"):
                r_org = np.round(r_org)
            lo = np.minimum(c.raw_min, r_org)
            hi = np.maximum(c.raw_max, r_org)
            r_adv = np.clip(r_adv, lo, hi)
            if f.kind == "binary":
                r_adv = np.clip(np.round(r_adv), 0, 1)
            elif f.kind == "integer":
                r_adv = np.round(r_adv)
            if f.direction > 0:
                r_adv = np.maximum(r_adv, r_org)
            elif f.direction < 0:
                r_adv = np.minimum(r_adv, r_org)
            X_adv[:, j] = self._encode_num(f.name, r_adv)
            # keep unchanged entries bit-identical to the original encoding
            same = r_adv == r_org
            X_adv[same, j] = X_orig[same, j]
        return X_adv.astype(np.float32)

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("TabularEncoder is not fitted")
