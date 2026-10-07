import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import robustsec as rs

EPS = 0.1
GRADIENT_ATTACKS = [
    lambda tm: rs.FGSM(EPS, threat_model=tm),
    lambda tm: rs.PGD(EPS, steps=10, restarts=2, threat_model=tm),
    lambda tm: rs.CarliniWagnerL2(EPS, steps=30, search_steps=2, threat_model=tm),
]


@pytest.mark.parametrize("make", GRADIENT_ATTACKS + [lambda tm: rs.RandomSearch(EPS, queries=20, threat_model=tm)])
def test_budget_and_box(toy, toy_clf, make):
    _, X, y, _, _ = toy
    X_adv = make(None).generate(toy_clf, X, y)
    assert X_adv.shape == X.shape and X_adv.dtype == np.float32
    assert np.abs(X_adv - X).max() <= EPS + 1e-6
    assert X_adv.min() >= -1e-7 and X_adv.max() <= 1 + 1e-7


@pytest.mark.parametrize("make", GRADIENT_ATTACKS)
def test_mask_and_direction(toy, toy_clf, make):
    _, X, y, _, _ = toy
    n = X.shape[1]
    mask = np.zeros(n, np.float32); mask[[0, 1]] = 1
    direction = np.zeros(n, np.float32); direction[0] = 1
    tm = rs.ThreatModel(n, mask=mask, direction=direction)
    D = make(tm).generate(toy_clf, X, y) - X
    assert np.all(D[:, 2:] == 0)
    assert np.all(D[:, 0] >= -1e-7)


def test_fgsm_matches_gradient_sign(linear_clf):
    X = np.full((3, 4), 0.1, np.float32)        # sum = 0.4 < 1 -> class 0
    y = np.zeros(3, np.int64)
    X_adv = rs.FGSM(0.05).generate(linear_clf, X, y)
    np.testing.assert_allclose(X_adv, X + 0.05, atol=1e-6)


def test_one_step_pgd_equals_fgsm(linear_clf):
    X = np.full((5, 4), 0.2, np.float32)
    y = np.zeros(5, np.int64)
    pgd = rs.PGD(0.05, steps=1, alpha=0.1, restarts=1).generate(linear_clf, X, y)
    fgsm = rs.FGSM(0.05).generate(linear_clf, X, y)
    np.testing.assert_allclose(pgd, fgsm, atol=1e-6)


def test_pgd_flips_linear_model(linear_clf):
    X = np.full((4, 4), 0.2, np.float32)        # margin 0.2; needs eps > 0.05
    y = np.zeros(4, np.int64)
    assert (linear_clf.predict(rs.PGD(0.1, steps=20).generate(linear_clf, X, y)) == 1).all()
    assert (linear_clf.predict(rs.PGD(0.04, steps=20).generate(linear_clf, X, y)) == 0).all()


def test_cw_returns_successful_or_clean(toy, toy_clf):
    _, X, y, _, _ = toy
    X_adv = rs.CarliniWagnerL2(0.2, steps=40, search_steps=3).generate(toy_clf, X, y)
    changed = np.abs(X_adv - X).max(1) > 0
    assert changed.any()
    assert (toy_clf.predict(X_adv[changed]) != y[changed]).all()


def test_cw_minimal_perturbation(linear_clf):
    X = np.full((2, 4), 0.2, np.float32)        # boundary at sum = 1 -> delta 0.05 each
    y = np.zeros(2, np.int64)
    X_adv = rs.CarliniWagnerL2(0.2, steps=300, lr=0.01, search_steps=4).generate(linear_clf, X, y)
    assert (linear_clf.predict(X_adv) == 1).all()
    assert np.abs(X_adv - X).max() < 0.08


def test_random_search_label_only_and_reproducible(toy):
    _, X, y, _, _ = toy
    clf = rs.SklearnClassifier(LogisticRegression(max_iter=500).fit(X, y))
    a = rs.RandomSearch(0.2, queries=30, seed=3).generate(clf, X, y)
    b = rs.RandomSearch(0.2, queries=30, seed=3).generate(clf, X, y)
    np.testing.assert_array_equal(a, b)
    changed = np.abs(a - X).max(1) > 0
    assert (clf.predict(a[changed]) != y[changed]).all()


def test_gradient_attack_rejects_label_only_model(toy):
    _, X, y, _, _ = toy
    clf = rs.SklearnClassifier(LogisticRegression(max_iter=200).fit(X, y))
    with pytest.raises(TypeError):
        rs.PGD(0.1).generate(clf, X, y)


def test_repair_is_applied(toy, toy_clf):
    _, X, y, schema, enc = toy
    tm = rs.ThreatModel.from_encoder(enc, schema.with_threat_model(mutable=["count"]))
    X_adv = rs.PGD(0.2, steps=10, threat_model=tm).generate(toy_clf, X, y)
    raw = enc.to_raw(X_adv)
    assert np.allclose(raw["count"], np.round(raw["count"]), atol=1e-3)
    np.testing.assert_array_equal(np.delete(X_adv, 6, axis=1), np.delete(X, 6, axis=1))


def test_transfer_uses_surrogate(toy, toy_clf):
    _, X, y, _, _ = toy
    target = rs.fit_classifier(X, y, hidden=(16,), epochs=5, seed=1)
    base = rs.PGD(0.1, steps=5, seed=0)
    np.testing.assert_array_equal(rs.TransferAttack(base, toy_clf).generate(target, X, y),
                                  rs.PGD(0.1, steps=5, seed=0).generate(toy_clf, X, y))


def test_make_attack():
    assert isinstance(rs.make_attack("PGD", 0.1, steps=3), rs.PGD)
    with pytest.raises(KeyError):
        rs.make_attack("deepfool", 0.1)
