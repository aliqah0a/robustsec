"""Measure the cost of TabularEncoder.repair (reviewer comment R2.4).

Usage: python scripts/time_repair.py [--rows 100000] [--repeats 5]
Prints rows per second for the constrained and the all-mutable NSL-KDD
threat models, and writes repair_timing.json next to the results.
"""
import argparse
import json
import time

import numpy as np

import robustsec as rs
from robustsec.benchmark import _hardware

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=100_000)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--out", default="results/repair_timing.json")
a = ap.parse_args()

ds = rs.load_nsl_kdd()
enc = rs.TabularEncoder(ds.schema).fit(ds.X_train)
X = enc.transform(ds.X_test)
X = np.resize(X, (a.rows, X.shape[1]))           # tile the test set to the target size
rng = np.random.default_rng(0)
X_try = np.clip(X + rng.uniform(-0.1, 0.1, X.shape), 0, 1).astype(np.float32)
out = {}
for name, schema in [("constrained", rs.nsl_kdd_feasible_schema()),
                     ("constrained_dilution", rs.nsl_kdd_feasible_schema(dilution=True)),
                     ("all_mutable", ds.schema)]:
    times = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        enc.repair(X_try, X, schema, eps=0.1)
        times.append(time.perf_counter() - t)
    med = float(np.median(times))
    out[name] = dict(rows=a.rows, median_s=round(med, 4), rows_per_s=round(a.rows / med))
    print(f"{name:22s} {a.rows} rows: median {med:.3f} s, {a.rows / med:,.0f} rows/s")
out["hardware"] = _hardware()
with open(a.out, "w") as fh:
    json.dump(out, fh, indent=2)
