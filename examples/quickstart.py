"""Minimal RobustSec example: train a detector on NSL-KDD and attack it
under the unconstrained and the feasibility-aware threat models.

Run with:  python examples/quickstart.py
"""
import robustsec as rs

ds = rs.load_nsl_kdd(download=True)
enc = rs.TabularEncoder(ds.schema).fit(ds.X_train)
X_tr, X_te = enc.transform(ds.X_train), enc.transform(ds.X_test)
y_te = ds.y_test

clf = rs.fit_classifier(X_tr, ds.y_train, epochs=5, seed=0)
print("clean:", {k: round(v, 4) for k, v in rs.clean_report(clf, X_te, y_te).items()})

threat_models = {
    "unconstrained": rs.ThreatModel.unconstrained(enc.n_features_),
    "feasible": rs.ThreatModel.from_encoder(enc, rs.nsl_kdd_feasible_schema()),
}
for name, tm in threat_models.items():
    for attack in [rs.PGD(0.1, steps=20, threat_model=tm),
                   rs.RandomSearch(0.1, queries=100, threat_model=tm)]:
        X_adv = attack.generate(clf, X_te, y_te)
        r = rs.robustness_report(clf, X_te, X_adv, y_te)
        lo, hi = r["evasion_ci"]
        print(f"{name:14s} {attack.name:13s} robust acc={r['robust_accuracy']:.3f} "
              f"ASR={r['asr']:.3f} evasion={r['evasion_rate']:.3f} [{lo:.3f}, {hi:.3f}] "
              f"L0={r['l0']:.1f}")

# The last adversarial batch as valid raw records.
print(enc.to_raw(X_adv[:3]))
