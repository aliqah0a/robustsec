"""Create the paper figures and table from a finished benchmark run.

Usage: python scripts/make_figures.py results/nsl_kdd paper/figures
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from robustsec.plotting import plot_metric  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/nsl_kdd")
out = Path(sys.argv[2] if len(sys.argv) > 2 else run / "figures")
out.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(run / "summary.csv")
names = {"baseline": "Baseline", "adv_trained": "Adversarially trained",
         "unconstrained": "unconstrained", "feasible": "feasible"}
plt.rcParams.update({"font.size": 9})

# Figure: evasion rate per threat model and model.
fig = plot_metric(s.replace({"model": names}), "evasion_rate", out / "evasion_rate.pdf", ylabel="Evasion rate")
plt.close(fig)

# Figure: PGD robust accuracy, baseline vs adversarially trained, both threat models.
fig, ax = plt.subplots(figsize=(4.6, 3.4))
style = {"unconstrained": "-", "feasible": "--"}
color = {"baseline": "#d62728", "adv_trained": "#1f77b4"}
for (m, tm), d in s[s.attack == "PGD"].groupby(["model", "threat_model"], sort=False):
    d = d.sort_values("eps")
    mu, sd = d["robust_accuracy_mean"], d["robust_accuracy_std"].fillna(0)
    ax.plot(d["eps"], mu, style[tm], color=color[m], marker="o", ms=3, label=f"{names[m]}, {tm}")
    ax.fill_between(d["eps"], mu - sd, mu + sd, color=color[m], alpha=0.15)
ax.set_xlabel(r"budget $\varepsilon$")
ax.set_ylabel("Robust accuracy (PGD)")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False, ncol=2, handlelength=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig(out / "robust_accuracy_pgd.pdf", bbox_inches="tight")
plt.close(fig)

# LaTeX table: PGD and CW at eps = 0.05 and 0.1.
clean = pd.read_csv(run / "clean.csv").groupby("model")[["accuracy", "f1"]].agg(["mean", "std"])
print(clean.round(4))
rows = s[s.attack.isin(["PGD", "CW-L2", "RandomSearch", "Transfer-PGD"]) & s.eps.isin([0.05, 0.1])]
cols = ["model", "threat_model", "attack", "eps", "robust_accuracy_mean", "robust_accuracy_std",
        "asr_mean", "asr_std", "evasion_rate_mean", "evasion_rate_std", "l0_mean"]
rows[cols].round(3).to_csv(out / "table_main.csv", index=False)
print(rows[cols].round(3).to_string(index=False))
