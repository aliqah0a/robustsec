"""Dataset loaders with local caching and checksum verification.

RobustSec never redistributes datasets. :func:`load_nsl_kdd` reads the
official files from a local directory. If they are absent, it can fetch a
public mirror and checks the SHA-256 digest before use. UNSW-NB15 and any
other tabular dataset are read from local CSV files.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import FeatureSchema, FeatureSpec, Relation


@dataclass
class TabularDataset:
    """Raw train/test tables plus the feature schema and binary labels."""

    name: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    schema: FeatureSchema
    positive_label: str = "attack"


# --------------------------------------------------------------------- NSL-KDD
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NSL_KDD_FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}
NSL_KDD_MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

_NSL_BINARY = ["land", "logged_in", "root_shell", "is_host_login", "is_guest_login"]
_NSL_CATEGORICAL = ["protocol_type", "service", "flag"]
_NSL_INTEGER = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "count", "srv_count", "dst_host_count",
    "dst_host_srv_count",
]
_NSL_LOG = ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised",
            "num_root", "num_file_creations", "num_access_files", "count",
            "srv_count"]

#: Features an external attacker can plausibly raise without breaking the
#: attack: connection length (delays), payload volume (padding) and the
#: number of connections in the 2-second window (dummy connections).
#: ``dst_bytes`` is the server's response volume; the attacker controls it
#: only indirectly (for example by requesting larger resources), so the
#: ablation configuration also runs a variant without it.
NSL_KDD_FEASIBLE_MUTABLE = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
NSL_KDD_FEASIBLE_DIRECTIONS = {n: +1 for n in NSL_KDD_FEASIBLE_MUTABLE}


def _nsl_kdd_dilution(adv, org):
    """Error rates after the attacker adds error-free connections.

    ``serror_rate`` and ``rerror_rate`` are fractions of the ``count``
    connections to the same host in the last two seconds;
    ``srv_serror_rate`` and ``srv_rerror_rate`` are fractions of the
    ``srv_count`` connections to the same service. Extra connections that
    raise ``count`` or ``srv_count`` without SYN or REJ errors keep the
    number of errored connections fixed, so each rate scales by
    ``old_count / new_count``. Values are rounded to two decimals, as in
    the dataset.
    """
    out = {}
    for rate, cnt in (("serror_rate", "count"), ("rerror_rate", "count"),
                      ("srv_serror_rate", "srv_count"), ("srv_rerror_rate", "srv_count")):
        new = np.maximum(adv[cnt], 1.0)
        out[rate] = np.round(org[rate] * np.maximum(org[cnt], 1.0) / new, 2)
    return out


NSL_KDD_DILUTION = Relation(
    "nsl_kdd_dilution",
    inputs=("count", "srv_count"),
    outputs=("serror_rate", "rerror_rate", "srv_serror_rate", "srv_rerror_rate"),
    fn=_nsl_kdd_dilution)

#: Named relations that a configuration file can refer to.
RELATIONS = {"nsl_kdd_dilution": NSL_KDD_DILUTION}


def nsl_kdd_schema() -> FeatureSchema:
    """Schema of the 41 NSL-KDD features (all mutable, both directions)."""
    specs = []
    for name in NSL_KDD_COLUMNS:
        if name in _NSL_CATEGORICAL:
            kind = "categorical"
        elif name in _NSL_BINARY:
            kind = "binary"
        elif name in _NSL_INTEGER:
            kind = "integer"
        else:
            kind = "continuous"   # rates in [0, 1]
        specs.append(FeatureSpec(name, kind, log=name in _NSL_LOG))
    return FeatureSchema(specs)


def nsl_kdd_feasible_schema(dilution: bool = False) -> FeatureSchema:
    """Constrained threat model: volume/timing features, increase only.

    ``dilution=True`` adds :data:`NSL_KDD_DILUTION`, which updates the four
    error-rate features when ``count`` or ``srv_count`` grow.
    """
    return nsl_kdd_schema().with_threat_model(
        mutable=NSL_KDD_FEASIBLE_MUTABLE, directions=NSL_KDD_FEASIBLE_DIRECTIONS,
        relations=(NSL_KDD_DILUTION,) if dilution else ())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(directory: Path, fname: str, digest: str, download: bool,
                 mirror: str) -> Path:
    path = directory / fname
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} not found. Place the official NSL-KDD files in "
                f"{directory} or call load_nsl_kdd(download=True).")
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(mirror + fname, tmp)
        os.replace(tmp, path)
    actual = _sha256(path)
    if actual != digest:
        raise ValueError(f"checksum mismatch for {path}: {actual} != {digest}")
    return path


def default_data_dir() -> Path:
    return Path(os.environ.get("ROBUSTSEC_DATA", Path.home() / ".robustsec" / "data"))


def load_nsl_kdd(data_dir: Optional[os.PathLike] = None, download: bool = False,
                 mirror: str = NSL_KDD_MIRROR) -> TabularDataset:
    """Load KDDTrain+ and KDDTest+ with binary labels (1 = attack)."""
    d = Path(data_dir) if data_dir is not None else default_data_dir() / "nsl_kdd"
    tables = []
    for fname, digest in NSL_KDD_FILES.items():
        path = _ensure_file(d, fname, digest, download, mirror)
        df = pd.read_csv(path, header=None,
                         names=NSL_KDD_COLUMNS + ["label", "difficulty"])
        tables.append(df)
    train, test = tables
    y_tr = (train["label"] != "normal").to_numpy(dtype=np.int64)
    y_te = (test["label"] != "normal").to_numpy(dtype=np.int64)
    return TabularDataset("nsl_kdd", train[NSL_KDD_COLUMNS], y_tr,
                          test[NSL_KDD_COLUMNS], y_te, nsl_kdd_schema())


# ------------------------------------------------------------------- UNSW-NB15
UNSW_NB15_FILES = ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
#: official partition sizes are 175,341 (training) and 82,332 (testing)
UNSW_NB15_SWAPPED_SIZES = (82_332, 175_341)

#: Source-side volume and timing features of UNSW-NB15 that a sender can
#: raise: flow duration, packets sent and bytes sent (increase only).
UNSW_NB15_FEASIBLE_MUTABLE = ["dur", "spkts", "sbytes"]


def load_unsw_nb15(train_csv: Optional[os.PathLike] = None,
                   test_csv: Optional[os.PathLike] = None) -> TabularDataset:
    """Load the official UNSW-NB15 training and testing partitions.

    Download ``UNSW_NB15_training-set.csv`` and ``UNSW_NB15_testing-set.csv``
    from the UNSW Canberra dataset page. Pass their paths, or place them in
    ``$ROBUSTSEC_DATA/unsw_nb15``. The schema is inferred: ``proto``,
    ``service`` and ``state`` are categorical, and non-negative numeric
    features with a maximum above 10 (durations, counts, bytes, loads) get a
    ``log1p`` transform.
    """
    d = default_data_dir() / "unsw_nb15"
    train_csv = Path(train_csv) if train_csv else d / UNSW_NB15_FILES[0]
    test_csv = Path(test_csv) if test_csv else d / UNSW_NB15_FILES[1]
    for p in (train_csv, test_csv):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} not found. Download the official UNSW-NB15 "
                                    "partitions and pass their paths.")
    # the distributed CSV files start with a UTF-8 byte-order mark
    train = pd.read_csv(train_csv, encoding="utf-8-sig")
    test = pd.read_csv(test_csv, encoding="utf-8-sig")
    if (len(train), len(test)) == UNSW_NB15_SWAPPED_SIZES:
        # Some copies of the official files carry swapped names: the file
        # called "training-set" holds the 82,332-record testing partition.
        warnings.warn("UNSW-NB15 file names are swapped (82,332-record file named "
                      "'training'); using the 175,341-record file for training.")
        train, test = test, train
    drop = [c for c in ("id", "label", "attack_cat") if c in train.columns]
    return load_csv_frames("unsw_nb15", train, test,
                           y_train=train["label"].to_numpy(dtype=np.int64),
                           y_test=test["label"].to_numpy(dtype=np.int64),
                           exclude=drop, categorical=("proto", "service", "state"),
                           log_heavy_tailed=True, log_threshold=10.0)


# --------------------------------------------------------------------- generic
def load_csv_frames(name: str, train: pd.DataFrame, test: pd.DataFrame,
                    y_train: np.ndarray, y_test: np.ndarray,
                    exclude: Iterable[str] = (),
                    categorical: Sequence[str] = (),
                    log_heavy_tailed: bool = False,
                    log_threshold: float = 100.0) -> TabularDataset:
    """Wrap any pair of DataFrames as a :class:`TabularDataset`."""
    schema = FeatureSchema.infer(train, categorical=categorical, exclude=exclude,
                                 log_heavy_tailed=log_heavy_tailed, log_threshold=log_threshold)
    cols = schema.names
    return TabularDataset(name, train[cols].reset_index(drop=True), np.asarray(y_train),
                          test[cols].reset_index(drop=True), np.asarray(y_test), schema)


def load_csv(train_csv: os.PathLike, test_csv: os.PathLike, label: str,
             positive: Optional[str] = None, categorical: Sequence[str] = (),
             exclude: Iterable[str] = ()) -> TabularDataset:
    """Load any binary-labelled tabular dataset from two CSV files.

    ``positive`` selects the label value treated as class 1. When omitted,
    the label column must already hold 0/1 values.
    """
    train, test = pd.read_csv(train_csv), pd.read_csv(test_csv)

    def lab(df):
        return ((df[label].astype(str) == str(positive)) if positive is not None
                else df[label]).to_numpy(dtype=np.int64)

    return load_csv_frames(Path(train_csv).stem, train, test, lab(train), lab(test),
                           exclude=list(exclude) + [label], categorical=categorical)
