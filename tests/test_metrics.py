import numpy as np

import robustsec as rs


class Threshold(rs.Classifier):
    def predict(self, X):
        return (np.asarray(X)[:, 0] > 0.5).astype(int)


def test_report_denominators():
    clf = Threshold()
    X = np.array([[0.9], [0.8], [0.2], [0.1], [0.7]], np.float32)
    y = np.array([1, 1, 0, 0, 0])               # last sample is a clean error
    X_adv = np.array([[0.4], [0.8], [0.6], [0.1], [0.7]], np.float32)
    r = rs.robustness_report(clf, X, X_adv, y)
    assert r["clean_accuracy"] == 0.8
    assert r["robust_accuracy"] == 0.4
    assert r["n_correct"] == 4 and r["asr"] == 0.5        # 2 of 4 correct samples flip
    assert r["n_detected"] == 2 and r["evasion_rate"] == 0.5
    np.testing.assert_allclose(r["linf"], 0.45)            # mean over the 2 successes
    assert r["l0"] == 1.0
    lo, hi = r["asr_ci"]
    assert lo < 0.5 < hi


def test_no_success_gives_nan_cost():
    clf = Threshold()
    X = np.array([[0.9], [0.1]], np.float32)
    r = rs.robustness_report(clf, X, X, np.array([1, 0]))
    assert r["asr"] == 0 and np.isnan(r["linf"])


def test_wilson_interval():
    lo, hi = rs.wilson_interval(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert np.isnan(rs.wilson_interval(0, 0)[0])
