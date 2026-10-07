import numpy as np
import pandas as pd
import pytest

import robustsec as rs


def test_encoded_range_and_layout(toy):
    df, X, _, _, enc = toy
    assert X.shape == (len(df), 7)              # 4 numeric + 3 one-hot
    assert X.min() >= 0 and X.max() <= 1
    assert enc.columns_[3:6] == ["proto=icmp", "proto=tcp", "proto=udp"]


def test_round_trip(toy):
    df, X, _, _, enc = toy
    raw = enc.to_raw(X)
    np.testing.assert_allclose(raw["bytes"], df["bytes"], rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(raw["rate"], df["rate"], atol=1e-5)
    assert (raw["proto"] == df["proto"]).all()


def test_constant_column_is_flagged():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": [5, 5, 5]})
    enc = rs.TabularEncoder(rs.FeatureSchema.infer(df)).fit(df)
    assert enc.constant_features_ == ["c"]
    assert enc.constraint_vectors()["mask"].tolist() == [1.0, 0.0]


def test_repair_enforces_domain(toy):
    df, X, _, schema, enc = toy
    tm_schema = schema.with_threat_model(mutable=["bytes", "rate", "flag", "count"],
                                         directions={"bytes": 1})
    rng = np.random.default_rng(1)
    X_adv = np.clip(X + rng.uniform(-0.3, 0.3, X.shape), 0, 1).astype(np.float32)
    R = enc.repair(X_adv, X, tm_schema)
    raw, raw0 = enc.to_raw(R), enc.to_raw(X)
    np.testing.assert_array_equal(R[:, 3:6], X[:, 3:6])          # immutable proto
    assert np.allclose(raw["count"], np.round(raw["count"]), atol=1e-3)
    assert set(np.round(raw["flag"]).astype(int)) <= {0, 1}
    assert (raw["bytes"].to_numpy() >= raw0["bytes"].to_numpy() - 1e-2).all()


def test_repair_onehot_single_category(toy):
    _, X, _, schema, enc = toy
    X_adv = X.copy()
    X_adv[:, 3:6] = np.array([0.4, 0.5, 0.45], dtype=np.float32)
    R = enc.repair(X_adv, X, schema)          # proto mutable in default schema
    assert np.allclose(R[:, 3:6].sum(1), 1.0)
    assert (R[:, 4] == 1.0).all()


def test_schema_validation():
    with pytest.raises(ValueError):
        rs.FeatureSpec("x", "ordinal")
    with pytest.raises(ValueError):
        rs.FeatureSpec("x", "binary", log=True)
    with pytest.raises(KeyError):
        rs.nsl_kdd_schema().with_threat_model(mutable=["not_a_feature"])


def test_infer_types():
    df = pd.DataFrame({"i": [1, 2, 3], "b": [0, 1, 0], "f": [0.1, 0.2, 0.3], "s": ["a", "b", "a"]})
    kinds = {f.name: f.kind for f in rs.FeatureSchema.infer(df).features}
    assert kinds == {"i": "integer", "b": "binary", "f": "continuous", "s": "categorical"}


def test_nsl_kdd_schema():
    s = rs.nsl_kdd_schema()
    assert len(s) == 41
    assert s.of_kind("categorical") == ["protocol_type", "service", "flag"]
    feas = rs.nsl_kdd_feasible_schema()
    assert sorted(f.name for f in feas.features if f.mutable) == \
        sorted(["duration", "src_bytes", "dst_bytes", "count", "srv_count"])


def test_to_raw_returns_integers(toy):
    df, X, _, _, enc = toy
    raw = enc.to_raw(X)
    assert raw["bytes"].dtype.kind == "i" and raw["flag"].dtype.kind == "i"
    assert (raw["bytes"].to_numpy() == df["bytes"].to_numpy()).all()
