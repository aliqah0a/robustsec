"""Turn finished benchmark runs into LaTeX macros for the paper.

Every number in the revised manuscript is written as ``\\res{key}``. Until a
key is defined, it prints a red ``[X.XX]``. This script reads the result
folders and writes ``results_macros.tex`` with one ``\\defres{key}{value}``
per number, so the paper fills itself on the next compile.

Usage:
  python scripts/paper_numbers.py \\
      --run nsl=results/nsl_kdd --run abl=results/nsl_kdd_ablation \\
      --run def=results/nsl_kdd_defenses --run unsw=results/unsw_nb15 \\
      --timing results/repair_timing.json -o paper/results_macros.tex

Key formats (``run`` is the short name given with --run):
  run/<model>/<threat model>/<attack>/<eps>/<metric>     mean over seeds
      ... /sd   standard deviation      ... /pct  mean in percent
  run/clean/<model>/<metric>[/sd]                        clean metrics
  run/paired/<model>/<attack>/<eps>/<metric>/<a>/<b>/{mean,lo_min,hi_max,excl0}
  run/ratio/<model>/<attack>/<eps>/<a>/<b>               evasion ratio a / b
  run/best/<model>/<threat model>/<eps>/{evasion_rate,attack}
  run/info/{wall_min,peak_gb,cpu,cores,ram_gb,gpu}
  run/data/{n_train,n_test,n_eval,n_raw_features,n_encoded_columns,
            attack_share_train,attack_share_test}
  timing/<schema>/rows_per_s
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, str):
        return v.replace("_", r"\_").replace("%", r"\%")
    if float(v).is_integer() and abs(v) >= 1000:
        return f"{int(v):,}".replace(",", "{,}")
    return f"{v:.{digits}f}"


def eps_key(e):
    return f"{float(e):g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="name=folder")
    ap.add_argument("--timing", default=None)
    ap.add_argument("--json", action="append", default=[],
                    help="name=file.json; numbers become name/<path>/<to>/<value>")
    ap.add_argument("-o", "--out", default="paper/results_macros.tex")
    a = ap.parse_args()
    macros = {}

    for spec in a.run:
        name, folder = spec.split("=", 1)
        d = Path(folder)
        s = pd.read_csv(d / "summary.csv")
        for _, r in s.iterrows():
            base = f"{name}/{r.model}/{r.threat_model}/{r.attack}/{eps_key(r.eps)}"
            for c in s.columns:
                if c.endswith("_mean"):
                    m = c[:-5]
                    macros[f"{base}/{m}"] = fmt(r[c], 1 if m == "l0" else 3)
                    macros[f"{base}/{m}/sd"] = fmt(r.get(f"{m}_std"))
                    if m in ("asr", "evasion_rate", "benign_flip_rate", "budget_violation_rate"):
                        macros[f"{base}/{m}/pct"] = fmt(100 * r[c], 1)
        # largest value over seeds for "max" metrics (results.csv)
        r = pd.read_csv(d / "results.csv")
        for keys, g in r.groupby(["model", "threat_model", "attack", "eps"]):
            mo, tm, at, e = keys
            base = f"{name}/{mo}/{tm}/{at}/{eps_key(e)}"
            for m in ("budget_excess_max", "linf_max"):
                if m in g:
                    macros[f"{base}/{m}/max"] = fmt(g[m].max())
        # strongest attack per (model, threat model, eps)
        for (mo, tm, e), g in s.groupby(["model", "threat_model", "eps"]):
            g = g[~g.attack.str.startswith("Transfer")]
            if len(g):
                top = g.loc[g.evasion_rate_mean.idxmax()]
                macros[f"{name}/best/{mo}/{tm}/{eps_key(e)}/evasion_rate"] = fmt(top.evasion_rate_mean)
                macros[f"{name}/best/{mo}/{tm}/{eps_key(e)}/attack"] = fmt(top.attack)
        c = pd.read_csv(d / "clean.csv")
        num = [k for k in c.columns if k not in ("seed", "model")]
        for mo, g in c.groupby("model"):
            for k in num:
                digits = 0 if k in ("tp", "fp", "tn", "fn", "n") else 3
                macros[f"{name}/clean/{mo}/{k}"] = fmt(round(g[k].mean(), digits), digits)
                macros[f"{name}/clean/{mo}/{k}/sd"] = fmt(g[k].std(ddof=1), digits)
        if (d / "paired.csv").exists():
            p = pd.read_csv(d / "paired.csv")
            for key, g in p.groupby(["model", "attack", "eps", "metric", "threat_model_a",
                                     "threat_model_b"]):
                mo, at, e, me, ta, tb = key
                b = f"{name}/paired/{mo}/{at}/{eps_key(e)}/{me}/{ta}/{tb}"
                macros[f"{b}/mean"] = fmt(g["diff"].mean())
                macros[f"{b}/lo_min"] = fmt(g.ci_lo.min())
                macros[f"{b}/hi_max"] = fmt(g.ci_hi.max())
                macros[f"{b}/excl0"] = str(int(((g.ci_lo > 0) | (g.ci_hi < 0)).sum()))
                macros[f"{b}/nseeds"] = str(len(g))
                ra = s[(s.model == mo) & (s.attack == at) & (s.eps == e) & (s.threat_model == ta)]
                rb = s[(s.model == mo) & (s.attack == at) & (s.eps == e) & (s.threat_model == tb)]
                if me == "evasion_rate" and len(ra) and len(rb) and rb.evasion_rate_mean.iloc[0] > 0:
                    macros[f"{name}/ratio/{mo}/{at}/{eps_key(e)}/{ta}/{tb}"] = fmt(
                        ra.evasion_rate_mean.iloc[0] / rb.evasion_rate_mean.iloc[0], 1)
        info = json.loads((d / "run_info.json").read_text())
        hw = info.get("hardware", {})
        macros[f"{name}/info/wall_min"] = fmt(info["wall_time_s"] / 60, 0)
        macros[f"{name}/info/peak_gb"] = fmt(info.get("peak_rss_mb", float("nan")) / 1024, 1)
        macros[f"{name}/info/cpu"] = fmt(str(hw.get("cpu", "n/a")))
        macros[f"{name}/info/cores"] = str(hw.get("logical_cores", "n/a"))
        macros[f"{name}/info/ram_gb"] = fmt(hw.get("ram_gb", float("nan")), 1)
        macros[f"{name}/info/gpu"] = fmt(str(hw.get("gpu") or "none"))
        unseen = info.get("unseen_categories", {}) or {}
        macros[f"{name}/data/unseen_total"] = str(sum(unseen.values()))
        for k, v in unseen.items():
            macros[f"{name}/data/unseen/{k}"] = str(v)
        macros[f"{name}/data/unseen_features"] = fmt(", ".join(sorted(unseen)) or "none")
        for k, v in info.get("data", {}).items():
            digits = 1 if k.startswith("attack_share") else 0
            macros[f"{name}/data/{k}"] = (fmt(100 * v, 1) if k.startswith("attack_share")
                                          else fmt(v, digits) if isinstance(v, (int, float))
                                          else fmt(str(v)))

    if a.timing:
        t = json.loads(Path(a.timing).read_text())
        for k, v in t.items():
            if isinstance(v, dict) and "rows_per_s" in v:
                macros[f"timing/{k}/rows_per_s"] = fmt(round(v["rows_per_s"], -3), 0)

    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}/{k}", v)
        elif isinstance(obj, (int, float)):
            macros[prefix] = fmt(obj)
            if isinstance(obj, float) and 0 <= obj <= 1:
                macros[prefix + "/pct"] = fmt(100 * obj, 1)
    for spec in a.json:
        name, path = spec.split("=", 1)
        flatten(name, json.loads(Path(path).read_text()))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("% generated by scripts/paper_numbers.py -- do not edit by hand\n")
        for k in sorted(macros):
            fh.write(f"\\defres{{{k}}}{{{macros[k]}}}\n")
    print(f"{len(macros)} macros written to {out}")


if __name__ == "__main__":
    main()
