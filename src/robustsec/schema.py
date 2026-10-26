"""Feature schema: types, value domains, and attacker capabilities.

A :class:`FeatureSchema` records, for every raw input column, what kind of
value it holds and whether an attacker can change it. Attacks use this
information to keep adversarial examples inside the feasible input space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
class Relation:
    """A dependency between raw features, applied after repair.

    ``fn(adv, orig)`` receives two dicts that map feature names to raw
    value arrays (the repaired adversarial records and the clean records)
    and returns a dict with new raw values for the features in ``outputs``.
    The function must be deterministic and must depend on the clean values
    and the attacker-controlled inputs only, so that repair stays idempotent.
    """

    name: str
    inputs: Sequence[str]
    outputs: Sequence[str]
    fn: Callable[[Dict, Dict], Dict]


@dataclass
class FeatureSchema:
    """Ordered collection of :class:`FeatureSpec` objects.

    ``relations`` lists optional :class:`Relation` objects. They let a
    threat model update derived features (for example rates that depend on
    a count) when the attacker changes the features they depend on.
    """

    features: List[FeatureSpec] = field(default_factory=list)
    relations: Tuple[Relation, ...] = ()

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
                          directions: Optional[Dict[str, int]] = None,
                          relations: Optional[Sequence[Relation]] = None
                          ) -> "FeatureSchema":
        """Return a copy with a new attacker capability profile.

        ``mutable=None`` keeps the current flags. Otherwise only the listed
        features are mutable. ``directions`` overrides per-feature direction.
        ``relations=None`` keeps the current relations.
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
        rel = self.relations if relations is None else tuple(relations)
        for r in rel:
            missing = (set(r.inputs) | set(r.outputs)) - set(self.names)
            if missing:
                raise KeyError(f"relation {r.name}: unknown features {sorted(missing)}")
        return FeatureSchema(out, rel)

    @classmethod
    def infer(cls, df, categorical: Iterable[str] = (),
              exclude: Iterable[str] = (), log_heavy_tailed: bool = False,
              log_threshold: float = 100.0) -> "FeatureSchema":
        """Infer a schema from a pandas DataFrame.

        Non-numeric columns and columns listed in ``categorical`` become
        categorical. Columns holding only {0, 1} become binary. Integer
        columns become integer. All features are mutable in both directions.
        With ``log_heavy_tailed=True``, non-negative numeric columns whose
        maximum exceeds ``log_threshold`` get a ``log1p`` transform.
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
            log = False
            if log_heavy_tailed and kind in ("integer", "continuous"):
                v = s.dropna()
                log = bool(len(v) and v.min() >= 0 and v.max() > log_threshold)
            specs.append(FeatureSpec(col, kind, log=log))
        return cls(specs)
