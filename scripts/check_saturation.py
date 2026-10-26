"""Check cross-entropy saturation of PGD on a trained detector.

PGD follows the sign of the cross-entropy gradient. When a model is very
confident, float32 softmax gives p = 1 exactly and the gradient vanishes, so
sign steps stop moving. This script measures, for the seed-0 baseline of a
configuration, the share of detected attack records whose gradient at the
PGD start point is exactly zero, and compares PGD with CW-L2 (margin loss).

Usage: python scripts/check_saturation.py configs/unsw_nb15_paper.json \
           --out results/unsw_nb15/saturation.json [--n 3000]
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import robustsec as rs
from robustsec.benchmark import DATASETS, stratified_sample

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
cfg = json.load(open(a.config))
dcfg = dict(cfg["dataset"]); ds = DATASETS[dcfg.pop("name")](**dcfg)
enc = rs.TabularEncoder(ds.schema).fit(ds.X_train)
X_tr, X_te = enc.transform(ds.X_train), enc.transform(ds.X_test)
clf = rs.fit_classifier(X_tr, ds.y_train, seed=0, **cfg.get("train", {}))
idx = stratified_sample(ds.y_test, cfg.get("eval_samples"), 0)
Xe, ye = X_te[idx], ds.y_test[idx]
det = (clf.predict(Xe) == ye) & (ye == 1)
Xd, yd = Xe[det][: a.n], ye[det][: a.n]
tm = rs.ThreatModel.unconstrained(Xd.shape[1])
pgd_params = cfg["attacks"].get("pgd", {}); cw_params = cfg["attacks"].get("cw", {})
out = {"n": int(len(Xd))}
for eps in cfg["eps"]:
    x0, yt = clf.to_tensor(Xd), torch.as_tensor(yd)
    g = torch.Generator().manual_seed(0)
    x = tm.project(x0 + eps * (torch.rand(x0.shape, generator=g) * 2 - 1), x0, eps)
    x.requires_grad_(True)
    loss = F.cross_entropy(clf.logits(x), yt, reduction="none")
    (grad,) = torch.autograd.grad(loss.sum(), x)
    zero = float((grad.abs().sum(1) == 0).float().mean())
    ev = {}
    for name, atk in [("PGD", rs.PGD(eps, threat_model=tm, **pgd_params)),
                      ("CW-L2", rs.CarliniWagnerL2(eps, threat_model=tm, **cw_params))]:
        ev[name] = float((clf.predict(atk.generate(clf, Xd, yd)) == 0).mean())
    out[f"{eps:g}"] = {"zero_grad_share": round(zero, 4), "evasion": {k: round(v, 4) for k, v in ev.items()}}
    print(eps, out[f"{eps:g}"], flush=True)
json.dump(out, open(a.out, "w"), indent=2)
