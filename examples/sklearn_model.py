"""Evaluate a non-PyTorch detector (gradient boosting) with RobustSec.

Gradient attacks need a differentiable PyTorch model. Any other model can be
wrapped as a SklearnClassifier and tested with the label-only random search,
the score-based greedy search, and transfer from a PyTorch surrogate.

Run with:  python examples/sklearn_model.py
"""
from sklearn.ensemble import HistGradientBoostingClassifier

import robustsec as rs

ds = rs.load_nsl_kdd(download=True)
enc = rs.TabularEncoder(ds.schema).fit(ds.X_train)
X_tr, X_te, y_te = enc.transform(ds.X_train), enc.transform(ds.X_test), ds.y_test
X_te, y_te = X_te[:5000], y_te[:5000]          # a subset keeps the example quick

gbdt = rs.SklearnClassifier(HistGradientBoostingClassifier(random_state=0).fit(X_tr, ds.y_train))
surrogate = rs.fit_classifier(X_tr, ds.y_train, epochs=5, seed=0)   # PyTorch MLP
tm = rs.ThreatModel.from_encoder(enc, rs.nsl_kdd_feasible_schema())

for attack in [rs.RandomSearch(0.1, queries=200, threat_model=tm),
               rs.GreedySearch(0.1, rounds=10, grid=4, threat_model=tm),
               rs.TransferAttack(rs.PGD(0.1, steps=50, threat_model=tm), surrogate)]:
    X_adv = attack.generate(gbdt, X_te, y_te)
    r = rs.robustness_report(gbdt, X_te, X_adv, y_te, eps=0.1, mask=tm.mask)
    print(f"{attack.name:13s} evasion={r['evasion_rate']:.3f} "
          f"benign flips={r['benign_flip_rate']:.3f} budget violations={r['budget_violation_rate']:.3f}")
