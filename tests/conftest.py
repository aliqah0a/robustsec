import numpy as np
import pandas as pd
import pytest
import torch

import robustsec as rs


@pytest.fixture(scope="session")
def toy():
    """Small mixed-type dataset with a learnable rule."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "bytes": rng.integers(0, 10_000, n),
        "rate": rng.uniform(0, 1, n),
        "flag": rng.integers(0, 2, n),
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "count": rng.integers(0, 50, n),
    })
    y = ((df["rate"] > 0.5) ^ (df["proto"] == "udp")).astype(int).to_numpy()
    schema = rs.FeatureSchema([
        rs.FeatureSpec("bytes", "integer", log=True),
        rs.FeatureSpec("rate", "continuous"),
        rs.FeatureSpec("flag", "binary"),
        rs.FeatureSpec("proto", "categorical"),
        rs.FeatureSpec("count", "integer"),
    ])
    enc = rs.TabularEncoder(schema).fit(df)
    X = enc.transform(df)
    return df, X, y, schema, enc


@pytest.fixture(scope="session")
def toy_clf(toy):
    _, X, y, _, _ = toy
    return rs.fit_classifier(X, y, hidden=(32,), epochs=30, seed=0, batch_size=64)


@pytest.fixture
def linear_clf():
    """Two-class linear model with known gradient: z1 - z0 = sum(x) - 1."""
    m = torch.nn.Linear(4, 2)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[0.0] * 4, [1.0] * 4]))
        m.bias.copy_(torch.tensor([0.0, -1.0]))
    return rs.TorchClassifier(m, device="cpu")
