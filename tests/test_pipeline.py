import json

import numpy as np
import pandas as pd
import pytest

import robustsec as rs
from robustsec.benchmark import run_benchmark, stratified_sample
from robustsec.data import _ensure_file


def test_training_is_deterministic(toy):
    _, X, y, _, _ = toy
    a = rs.fit_classifier(X, y, hidden=(16,), epochs=3, seed=7).predict_proba(X)
    b = rs.fit_classifier(X, y, hidden=(16,), epochs=3, seed=7).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_adversarial_training_runs(toy):
    _, X, y, _, _ = toy
    clf = rs.adversarial_training(X, y, eps=0.05, steps=2, hidden=(16,), epochs=2, seed=0)
    assert 0 <= clf.val_accuracy <= 1


def test_stratified_sample_keeps_ratio():
    y = np.array([0] * 900 + [1] * 100)
    idx = stratified_sample(y, 100, seed=0)
    assert len(idx) == 100 and y[idx].sum() == 10


def test_benchmark_end_to_end(toy, tmp_path):
    df, _, y, schema, _ = toy
    ds = rs.TabularDataset("toy", df.iloc[:400], y[:400], df.iloc[400:], y[400:], schema)
    cfg = {
        "seeds": [0, 1], "eval_samples": None, "eps": [0.1],
        "threat_models": {"unconstrained": {"mutable": None, "repair": False},
                          "feasible": {"mutable": ["bytes", "count"],
                                       "directions": {"bytes": 1}, "repair": True}},
        "attacks": {"fgsm": {}, "pgd": {"steps": 3}, "cw": {"steps": 5, "search_steps": 1},
                    "random": {"queries": 5}},
        "train": {"hidden": [8], "epochs": 2},
        "adversarial_training": {"threat_model": "unconstrained", "eps": 0.1, "steps": 2},
    }
    res = run_benchmark(cfg, tmp_path, dataset=ds, verbose=False)
    # 2 seeds x 2 threat models x (4 attacks x 2 models + transfer)
    assert len(res) == 2 * 2 * 9
    for f in ["results.csv", "summary.csv", "clean.csv", "run_info.json"]:
        assert (tmp_path / f).exists()
    s = pd.read_csv(tmp_path / "summary.csv")
    assert {"asr_mean", "asr_std", "evasion_rate_mean"} <= set(s.columns)
    assert json.loads((tmp_path / "run_info.json").read_text())["robustsec"] == rs.__version__


def test_missing_data_and_checksum(tmp_path):
    with pytest.raises(FileNotFoundError):
        rs.load_nsl_kdd(tmp_path, download=False)
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(ValueError):
        _ensure_file(tmp_path, "f.txt", "0" * 64, download=False, mirror="")


def test_load_csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=50), "c": rng.choice(["x", "y"], 50),
                       "label": rng.choice(["benign", "malicious"], 50)})
    df.to_csv(tmp_path / "tr.csv", index=False)
    df.to_csv(tmp_path / "te.csv", index=False)
    ds = rs.load_csv(tmp_path / "tr.csv", tmp_path / "te.csv", label="label", positive="malicious")
    assert ds.schema.names == ["a", "c"]
    assert ds.y_train.sum() == (df.label == "malicious").sum()
