"""RobustSec: robustness evaluation of tabular intrusion detectors under
feasibility-aware threat models."""

__version__ = "1.0.0"

from .schema import FeatureSchema, FeatureSpec  # noqa: E402
from .preprocessing import TabularEncoder  # noqa: E402
from .data import (TabularDataset, load_csv, load_csv_frames, load_nsl_kdd,  # noqa: E402
                   load_unsw_nb15, nsl_kdd_feasible_schema, nsl_kdd_schema)
from .models import (MLP, Classifier, SklearnClassifier, TorchClassifier,  # noqa: E402
                     fit_classifier, set_seed)
from .attacks import (FGSM, PGD, Attack, CarliniWagnerL2, RandomSearch,  # noqa: E402
                      ThreatModel, TransferAttack, make_attack)
from .metrics import clean_report, robustness_report, wilson_interval  # noqa: E402
from .defenses import adversarial_training  # noqa: E402

__all__ = [
    "FeatureSchema", "FeatureSpec", "TabularEncoder", "TabularDataset", "load_csv",
    "load_csv_frames", "load_nsl_kdd", "load_unsw_nb15", "nsl_kdd_schema",
    "nsl_kdd_feasible_schema", "MLP", "Classifier", "SklearnClassifier",
    "TorchClassifier", "fit_classifier", "set_seed", "Attack", "FGSM", "PGD",
    "CarliniWagnerL2", "RandomSearch", "TransferAttack", "ThreatModel", "make_attack",
    "clean_report", "robustness_report", "wilson_interval", "adversarial_training",
]
