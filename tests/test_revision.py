"""Regression tests for the v1.1 fixes (reviewer comments R1.1, R1.3, R1.4,
R1.6, R2.3 and the constraint-aware greedy search)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import robustsec as rs
from robustsec.benchmark import run_benchmark


class Band(torch.nn.Module):
    """Two-class model: class 1 iff a quadratic bump in column 0 is positive."""

    def __init__(self, centre, width2, height=1.0, slope_b=0.0):
        super().__init__()
        self.c, self.w2, self.h, self.sb = centre, width2, height, slope_b

    def forward(self, x):
        d = self.h - (x[:, 0] - self.c) ** 2 / self.w2
        if x.shape[1] > 1:
            d = d + self.sb * x[:, 1]
        return torch.stack([torch.zeros_like(d), d], 1)


# ---------------------------------------------------------------- R1.3 PGD
def test_pgd_keeps_intermediate_success():
    # From x=0 with alpha=0.25: step 1 lands at 0.25 (inside the band,
    # misclassified), step 2 overshoots to 0.5 (outside, correct again).
    clf = rs.TorchClassifier(Band(0.3, 0.01), device="cpu")
    X, y = np.zeros((1, 1), np.float32), np.zeros(1, np.int64)
    pgd = rs.PGD(1.0, steps=2, alpha=0.25, random_start=False)
    X_adv = pgd.generate(clf, X, y)
    assert clf.predict(X_adv)[0] == 1
    np.testing.assert_allclose(X_adv, [[0.25]], atol=1e-6)


def _int_encoder():
    df = pd.DataFrame({"a": np.arange(11).repeat(3), "b": np.linspace(0, 1, 33)})
    schema = rs.FeatureSchema([rs.FeatureSpec("a", "integer"), rs.FeatureSpec("b")])
    enc = rs.TabularEncoder(schema).fit(df)
    X = enc.transform(pd.DataFrame({"a": [5] * 4, "b": [0.5] * 4}))
    return enc, X


# --------------------------------------------------------- R1.3 CW after repair
def test_cw_success_is_checked_after_repair():
    # Success region is a = 5.3 .. 5.7 (encoded 0.53 .. 0.57), between two
    # integers. Every repaired candidate rounds back to a = 5 or 6, which are
    # both classified correctly. CW must then return the clean input, never a
    # repaired record that changed b but no longer evades.
    enc, X = _int_encoder()
    clf = rs.TorchClassifier(Band(0.55, 0.0004, slope_b=0.05), device="cpu")
    y = np.zeros(len(X), np.int64)
    assert (clf.predict(X) == 0).all()
    tm = rs.ThreatModel.from_encoder(enc)
    X_adv = rs.CarliniWagnerL2(0.2, steps=60, lr=0.01, search_steps=3,
                               threat_model=tm).generate(clf, X, y)
    changed = (X_adv != X).any(1)
    assert (clf.predict(X_adv)[changed] == 1).all()


@pytest.mark.parametrize("name", ["pgd", "cw", "greedy", "random", "fgsm"])
def test_returned_records_are_repaired(toy, toy_clf, name):
    _, X, y, schema, enc = toy
    tm = rs.ThreatModel.from_encoder(enc, schema.with_threat_model(["bytes", "count"],
                                                                   {"bytes": 1}))
    kw = {"cw": {"steps": 5, "search_steps": 1}, "pgd": {"steps": 5}}.get(name, {})
    X_adv = rs.make_attack(name, 0.1, tm, **kw).generate(toy_clf, X, y)
    np.testing.assert_array_equal(tm.finalize(X_adv, X, 0.1), X_adv)


# ------------------------------------------------------------ R1.1 budget
def test_strict_repair_never_exceeds_budget():
    enc, X = _int_encoder()
    eps = 0.07                      # one unit of `a` costs 0.1 > eps
    X_try = X.copy(); X_try[:, 0] += eps
    nominal = enc.repair(X_try, X, eps=eps, budget="nominal")
    strict = enc.repair(X_try, X, eps=eps, budget="strict")
    assert np.abs(nominal - X).max() > eps + 1e-6      # rounding overshoots
    assert np.abs(strict - X).max() <= eps + 1e-6
    np.testing.assert_array_equal(strict[:, 0], X[:, 0])  # stays on the grid


@pytest.mark.parametrize("name", ["fgsm", "pgd", "cw", "random", "greedy"])
def test_attacks_preserve_budget_after_repair(toy, toy_clf, name):
    _, X, y, schema, enc = toy
    tm = rs.ThreatModel.from_encoder(enc, schema.with_threat_model(
        ["bytes", "count", "rate"], {"bytes": 1, "count": 1}))
    for eps in (0.01, 0.05):
        kw = {"steps": 5, "search_steps": 1} if name == "cw" else (
            {"steps": 5} if name == "pgd" else {})
        X_adv = rs.make_attack(name, eps, tm, **kw).generate(toy_clf, X, y)
        assert np.abs(X_adv - X).max() <= eps + 1e-6
        r = rs.robustness_report(toy_clf, X, X_adv, y, eps=eps, mask=tm.mask)
        assert r["budget_violation_rate"] == 0.0


def test_report_counts_budget_violations(toy, toy_clf):
    _, X, y, _, _ = toy
    X_adv = X.copy(); X_adv[:10, 0] = np.clip(X_adv[:10, 0] + 0.2, 0, 1)
    r = rs.robustness_report(toy_clf, X, X_adv, y, eps=0.05)
    assert 0 < r["budget_violation_rate"] <= 1 and r["budget_excess_max"] > 0


def test_repair_is_idempotent(toy):
    _, X, _, schema, enc = toy
    rng = np.random.default_rng(0)
    X_try = np.clip(X + rng.uniform(-0.3, 0.3, X.shape), 0, 1).astype(np.float32)
    for budget in ("strict", "nominal"):
        once = enc.repair(X_try, X, eps=0.1, budget=budget)
        np.testing.assert_array_equal(enc.repair(once, X, eps=0.1, budget=budget), once)


# -------------------------------------------------- R1.4 adversarial training
def test_pgd_run_returns_repaired_points_when_asked(toy, toy_clf):
    _, X, y, schema, enc = toy
    tm = rs.ThreatModel.from_encoder(enc, schema.with_threat_model(["bytes", "count"]))
    x0, yt = torch.as_tensor(X[:64].copy()), torch.as_tensor(y[:64].copy())
    pgd = rs.PGD(0.1, steps=3)
    rep = pgd.run(toy_clf, x0, yt, tm, repair=True).numpy()
    np.testing.assert_array_equal(tm.finalize(rep, X[:64], 0.1), rep)
    relaxed = pgd.run(toy_clf, x0, yt, tm, repair=False).numpy()
    assert not np.array_equal(tm.finalize(relaxed, X[:64], 0.1), relaxed)


def test_adversarial_training_with_repair_runs(toy):
    _, X, y, schema, enc = toy
    tm = rs.ThreatModel.from_encoder(enc, schema.with_threat_model(["bytes", "count"]))
    for repair in (True, False):
        clf = rs.adversarial_training(X, y, eps=0.05, steps=2, threat_model=tm, repair=repair,
                                      hidden=(8,), epochs=1, seed=0)
        assert 0 <= clf.val_accuracy <= 1


# ------------------------------------------------------ R2.3 unseen categories
def test_unseen_categories(toy):
    df, _, _, schema, _ = toy
    enc = rs.TabularEncoder(schema).fit(df)
    new = df.iloc[:3].copy(); new.loc[new.index[0], "proto"] = "sctp"
    X = enc.transform(new)
    sl = enc.groups_["proto"]
    assert X[0, sl].sum() == 0 and enc.unknown_counts_["proto"] == 1
    assert enc.to_raw(X)["proto"].iloc[0] == "<unknown>"
    np.testing.assert_array_equal(enc.repair(X, X, eps=0.1)[0, sl], X[0, sl])
    with pytest.raises(ValueError):
        rs.TabularEncoder(schema, handle_unknown="error").fit(df).transform(new)


# ----------------------------------------------------------------- relations
def test_dilution_relation_scales_error_rates():
    ds_schema = rs.nsl_kdd_feasible_schema(dilution=True)
    cols = ds_schema.names
    base = {c: 0 for c in cols}
    base.update(protocol_type="tcp", service="http", flag="SF", count=10, srv_count=4,
                serror_rate=0.5, srv_serror_rate=0.25, rerror_rate=0.1, srv_rerror_rate=0.0)
    train = pd.DataFrame([base, {**base, "count": 511, "srv_count": 511, "src_bytes": 10**6}])
    enc = rs.TabularEncoder(ds_schema).fit(train)
    X = enc.transform(train.iloc[:1])
    X_try = X.copy()
    X_try[0, enc.groups_["count"].start] = enc._encode_num("count", np.array([20.0]))[0]
    out = enc.repair(X_try, X)
    raw = enc.to_raw(out).iloc[0]
    assert raw["count"] == 20 and raw["serror_rate"] == pytest.approx(0.25)
    assert raw["srv_serror_rate"] == pytest.approx(0.25)      # srv_count unchanged
    np.testing.assert_array_equal(enc.repair(out, X), out)


# ------------------------------------------------------------ greedy search
def test_greedy_flips_linear_model_and_needs_scores(linear_clf):
    X = np.full((2, 4), 0.2, np.float32)        # sum 0.8 < 1 -> class 0
    y = np.zeros(2, np.int64)
    X_adv = rs.GreedySearch(0.1, rounds=4, grid=3).generate(linear_clf, X, y)
    assert (linear_clf.predict(X_adv) == 1).all()
    assert np.abs(X_adv - X).max() <= 0.1 + 1e-6

    class LabelOnly:
        def predict(self, X):
            return np.zeros(len(X), int)
    with pytest.raises(TypeError):
        rs.GreedySearch(0.1).generate(rs.SklearnClassifier(LabelOnly()), X, y)


# ------------------------------------------------------- R1.6 paired + export
def test_paired_difference_ci():
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1], bool)
    d, lo, hi = rs.paired_difference_ci(a, a)
    assert d == 0 and lo <= 0 <= hi
    d, lo, hi = rs.paired_difference_ci(np.ones(50, bool), np.zeros(50, bool))
    assert d == 1 and lo > 0.85


def test_benchmark_exports_intervals_and_hardware(toy, tmp_path):
    df, _, y, schema, _ = toy
    ds = rs.TabularDataset("toy", df.iloc[:400], y[:400], df.iloc[400:], y[400:], schema)
    cfg = {
        "seeds": [0], "eps": [0.1],
        "threat_models": {"unconstrained": {"mutable": None, "repair": False},
                          "constrained": {"mutable": ["bytes", "count"],
                                          "directions": {"bytes": 1}},
                          "nominal": {"mutable": ["bytes", "count"],
                                      "directions": {"bytes": 1}, "budget": "nominal"}},
        "attacks": {"pgd": {"steps": 3}, "greedy": {"rounds": 2}},
        "train": {"hidden": [8], "epochs": 1},
        "adversarial_training": {
            "adv_relaxed": {"threat_model": "constrained", "eps": 0.1, "steps": 2, "repair": False},
            "adv_repaired": {"threat_model": "constrained", "eps": 0.1, "steps": 2}},
        "paired": [["unconstrained", "constrained"]],
    }
    res = run_benchmark(cfg, tmp_path, dataset=ds, verbose=False)
    assert {"asr_ci_lo", "asr_ci_hi", "evasion_ci_lo", "evasion_ci_hi",
            "budget_violation_rate", "time_s"} <= set(res.columns)
    assert len(res) == 3 * (2 * 3 + 2)       # 3 TMs x (2 attacks x 3 models + 2 transfers)
    p = pd.read_csv(tmp_path / "paired.csv")
    assert set(p.metric) == {"evasion_rate", "asr"} and (p.ci_lo <= p.ci_hi).all()
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["hardware"]["logical_cores"] >= 1 and info["peak_rss_mb"] > 0
    assert {"adv_relaxed", "adv_repaired"} <= set(pd.read_csv(tmp_path / "clean.csv").model)


def test_unsw_loader_handles_bom_and_swapped_names(tmp_path, monkeypatch):
    import robustsec.data as d
    cols = ["id", "dur", "proto", "service", "state", "spkts", "sbytes", "attack_cat", "label"]
    def make(n):
        rng = np.random.default_rng(n)
        return pd.DataFrame({"id": range(n), "dur": rng.exponential(1, n),
                             "proto": rng.choice(["tcp", "udp"], n), "service": "-",
                             "state": rng.choice(["FIN", "INT"], n),
                             "spkts": rng.integers(1, 300, n), "sbytes": rng.integers(1, 9000, n),
                             "attack_cat": "Normal", "label": rng.integers(0, 2, n)})[cols]
    make(30).to_csv(tmp_path / "tr.csv", index=False, encoding="utf-8-sig")
    make(60).to_csv(tmp_path / "te.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(d, "UNSW_NB15_SWAPPED_SIZES", (30, 60))
    with pytest.warns(UserWarning):
        ds = d.load_unsw_nb15(tmp_path / "tr.csv", tmp_path / "te.csv")
    assert len(ds.X_train) == 60 and len(ds.X_test) == 30
    assert "id" not in ds.schema.names and "dur" in ds.schema.names
    assert ds.schema["sbytes"].log and ds.schema["proto"].kind == "categorical"
