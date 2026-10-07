"""Feature schema: types, value domains, and attacker capabilities.

A :class:`FeatureSchema` records, for every raw input column, what kind of
value it holds and whether an attacker can change it. Attacks use this
information to keep adversarial examples inside the feasible input space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

VALID_TYPES = ("continuous", "integer", "binary", "categorical")


@dataclass
class FeatureSpec:
    """Description of one raw feature.

    Attributes
    ----------
    name : column name in the raw table.
    kind : one of ``continuous``, ``integer``, ``binary``, ``categorical``.
    mutable : whether the attacker may change the value.
    direction : ``0`` (both ways), ``+1`` (increase only) or ``-1``
        (decrease only). Ignored when ``mutable`` is False.
    log : apply ``log1p`` before scaling (for heavy-tailed, non-negative
        counts such as byte volumes).
    """

    name: str
    kind: str = "continuous"
    mutable: bool = True
    direction: int = 0
    log: bool = False

    def __post_init__(self) -> None:
        if self.kind not in VALID_TYPES:
            raise ValueError(f"{self.name}: unknown kind '{self.kind}'")
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"{self.name}: direction must be -1, 0 or +1")
        if self.kind in ("binary", "categorical") and self.log:
            raise ValueError(f"{self.name}: log1p is only valid for numeric features")


@dataclass
class FeatureSchema:
    """Ordered collection of :class:`FeatureSpec` objects."""

    features: List[FeatureSpec] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.features]

    def __getitem__(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.features)

    def of_kind(self, *kinds: str) -> List[str]:
        return [f.name for f in self.features if f.kind in kinds]

    def with_threat_model(self, mutable: Optional[Iterable[str]] = None,
                          directions: Optional[Dict[str, int]] = None
                          ) -> "FeatureSchema":
        """Return a copy with a new attacker capability profile.

        ``mutable=None`` keeps the current flags. Otherwise only the listed
        features are mutable. ``directions`` overrides per-feature direction.
        """
        mutable_set = None if mutable is None else set(mutable)
        unknown = (mutable_set or set()) - set(self.names)
        if unknown:
            raise KeyError(f"unknown features in threat model: {sorted(unknown)}")
        directions = directions or {}
        out = []
        for f in self.features:
            out.append(FeatureSpec(
                name=f.name, kind=f.kind, log=f.log,
                mutable=f.mutable if mutable_set is None else f.name in mutable_set,
                direction=directions.get(f.name, f.direction),
            ))
        return FeatureSchema(out)

    @classmethod
    def infer(cls, df, categorical: Iterable[str] = (),
              exclude: Iterable[str] = ()) -> "FeatureSchema":
        """Infer a schema from a pandas DataFrame.

        Non-numeric columns and columns listed in ``categorical`` become
        categorical. Columns holding only {0, 1} become binary. Integer
        columns become integer. All features are mutable in both directions.
        """
        import numpy as np
        import pandas as pd

        categorical = set(categorical)
        exclude = set(exclude)
        specs = []
        for col in df.columns:
            if col in exclude:
                continue
            s = df[col]
            if col in categorical or not pd.api.types.is_numeric_dtype(s.dtype):
                kind = "categorical"
            else:
                vals = s.dropna().unique()
                if len(vals) <= 2 and set(np.asarray(vals, dtype=float)) <= {0.0, 1.0}:
                    kind = "binary"
                elif pd.api.types.is_integer_dtype(s.dtype):
                    kind = "integer"
                else:
                    kind = "continuous"
            specs.append(FeatureSpec(col, kind))
        return cls(specs)
