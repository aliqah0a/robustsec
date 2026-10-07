"""Command-line interface: ``robustsec run``, ``robustsec plot``, ``robustsec fetch``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="robustsec", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark from a JSON configuration")
    r.add_argument("config")
    r.add_argument("-o", "--out", default="results")
    r.add_argument("--quiet", action="store_true")

    g = sub.add_parser("plot", help="plot summary.csv")
    g.add_argument("summary")
    g.add_argument("--metric", default="evasion_rate")
    g.add_argument("-o", "--out", default=None)

    f = sub.add_parser("fetch", help="download and verify NSL-KDD")
    f.add_argument("--data-dir", default=None)

    a = p.parse_args(argv)
    if a.cmd == "run":
        from .benchmark import load_config, run_benchmark
        run_benchmark(load_config(a.config), a.out, verbose=not a.quiet)
    elif a.cmd == "plot":
        import pandas as pd
        from .plotting import plot_metric
        out = a.out or str(Path(a.summary).with_name(f"{a.metric}.png"))
        plot_metric(pd.read_csv(a.summary), a.metric, out, ylabel=a.metric.replace("_", " "))
        print(out)
    elif a.cmd == "fetch":
        from .data import load_nsl_kdd
        ds = load_nsl_kdd(a.data_dir, download=True)
        print(f"NSL-KDD ready: {len(ds.X_train)} train / {len(ds.X_test)} test rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
